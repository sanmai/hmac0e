// hmac0e - search for an HMAC-SHA256 digest of the form 0e[0-9]{N}.
//
// candidate = prefix || 20 zero-padded decimal digits, under one fixed key.
#include <cuda_runtime.h>

#include <cinttypes>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <string>
#include <vector>

#include "hmac0e.cuh"
#include "launch_range.cuh"
#include "hmac_kernel.cuh"

namespace {

struct Options {
  std::vector<uint8_t> key;
  bool have_key = false;
  std::vector<uint8_t> prefix;
  unsigned digits = 62;
  uint64_t start = 0;
  uint64_t count = 0;       // 0 = unbounded
  uint64_t stop_after = 1;  // stop once this many hits are found
  double time_limit = 0;    // seconds, 0 = unbounded
  int device = 0;
  uint32_t blocks = 0;      // 0 = derive from occupancy
  uint32_t threads = 256;
  uint32_t per_thread = 512;
  double progress = 5.0;
  std::string checkpoint;
  std::string hits_file;
  bool resume = false;
  bool verify = true;
};

void usage(const char* argv0) {
  std::printf(
      "usage: %s --key K [options]\n"
      "\n"
      "  --key STR | --key-hex HEX     HMAC key (hex for binary keys); may be empty\n"
      "  --prefix STR | --prefix-hex H fixed message prefix before the counter\n"
      "  --digits N                    decimal digits required after \"0e\" (0..62, default 62)\n"
      "\n"
      "  --start N                     first counter without an existing checkpoint (default 0)\n"
      "  --count N                     stop after N candidates\n"
      "  --stop-after N                stop after N hits (default 1)\n"
      "  --time SECONDS                wall-clock limit\n"
      "\n"
      "  --device N                    CUDA device index (default 0)\n"
      "  --threads N                   threads per block, 32..256 (default 256)\n"
      "  --batch N                     counters per thread per launch (default 512)\n"
      "  --blocks N                    blocks per launch (default: fill the GPU)\n"
      "\n"
      "  --checkpoint FILE             auto-resume if present; otherwise start a new search\n"
      "  --resume                      require an existing --checkpoint FILE\n"
      "  --hits-file FILE              append hits to FILE\n"
      "  --progress SECONDS            rate line to stderr (0 disables, default 5)\n"
      "  --no-verify                   skip the CPU re-check of each hit\n"
      "\n"
      "The message is --prefix followed by the counter as 20 zero-padded decimal\n"
      "digits (the full 2^64 range, searched in 10^8 windows). The prefix length\n"
      "must satisfy ((len + 12) %% 64) <= 47 so the 8-digit counter slot and the\n"
      "SHA-256 padding share one final block.\n"
      "\n"
      "Exit status: 0 done, 1 CUDA or I/O failure (rerun to resume), 2 invalid\n"
      "arguments, 3 a rerun cannot help. A launch records at most %u hits; status 3\n"
      "means a launch overflowed that buffer or the CPU re-check (off with\n"
      "--no-verify) rejected a reported digest. The checkpoint stays before that\n"
      "batch.\n",
      argv0, (unsigned)HMAC0E_MAX_HITS);
}

bool hex_decode(const std::string& in, std::vector<uint8_t>* out) {
  if (in.size() % 2) return false;
  out->clear();
  out->reserve(in.size() / 2);
  for (size_t i = 0; i < in.size(); i += 2) {
    int hi, lo;
    auto nib = [](char c) -> int {
      if (c >= '0' && c <= '9') return c - '0';
      if (c >= 'a' && c <= 'f') return c - 'a' + 10;
      if (c >= 'A' && c <= 'F') return c - 'A' + 10;
      return -1;
    };
    if ((hi = nib(in[i])) < 0 || (lo = nib(in[i + 1])) < 0) return false;
    out->push_back(uint8_t((hi << 4) | lo));
  }
  return true;
}

std::string hex_encode(const uint8_t* p, size_t n) {
  static const char* kHex = "0123456789abcdef";
  std::string s;
  s.resize(n * 2);
  for (size_t i = 0; i < n; ++i) {
    s[2 * i] = kHex[p[i] >> 4];
    s[2 * i + 1] = kHex[p[i] & 0xf];
  }
  return s;
}

// Decimal only, leading zeros allowed: matches the zero-padded counter digits.
bool parse_u64(const char* s, uint64_t* out) {
  if (*s < '0' || *s > '9') return false;
  char* end = nullptr;
  errno = 0;
  const unsigned long long v = std::strtoull(s, &end, 10);
  if (end == s || *end != '\0' || errno == ERANGE || v > UINT64_MAX) return false;
  *out = v;
  return true;
}

bool parse_u32(const char* s, uint32_t* out) {
  uint64_t v = 0;
  if (!parse_u64(s, &v) || v > UINT32_MAX) return false;
  *out = uint32_t(v);
  return true;
}

bool parse_seconds(const char* s, double* out) {
  if (*s < '0' || *s > '9') return false;
  char* end = nullptr;
  errno = 0;
  const double v = std::strtod(s, &end);
  if (*end != '\0' || errno == ERANGE || !std::isfinite(v)) return false;
  *out = v;
  return true;
}

bool need_value(int argc, char** argv, int* i, const char* flag, const char** out) {
  if (*i + 1 >= argc) {
    std::fprintf(stderr, "%s requires a value\n", flag);
    return false;
  }
  *out = argv[++(*i)];
  return true;
}

bool parse_args(int argc, char** argv, Options* o) {
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const char* v = nullptr;
    if (a == "--key") {
      if (!need_value(argc, argv, &i, "--key", &v)) return false;
      o->key.assign(reinterpret_cast<const uint8_t*>(v), reinterpret_cast<const uint8_t*>(v) +
                                                            std::strlen(v));
      o->have_key = true;
    } else if (a == "--key-hex") {
      if (!need_value(argc, argv, &i, "--key-hex", &v)) return false;
      if (!hex_decode(v, &o->key)) {
        std::fprintf(stderr, "bad --key-hex\n");
        return false;
      }
      o->have_key = true;
    } else if (a == "--prefix") {
      if (!need_value(argc, argv, &i, "--prefix", &v)) return false;
      o->prefix.assign(reinterpret_cast<const uint8_t*>(v),
                       reinterpret_cast<const uint8_t*>(v) + std::strlen(v));
    } else if (a == "--prefix-hex") {
      if (!need_value(argc, argv, &i, "--prefix-hex", &v)) return false;
      if (!hex_decode(v, &o->prefix)) {
        std::fprintf(stderr, "bad --prefix-hex\n");
        return false;
      }
    } else if (a == "--digits") {
      if (!need_value(argc, argv, &i, "--digits", &v)) return false;
      uint64_t d = 0;
      if (!parse_u64(v, &d) || d > 62) {
        std::fprintf(stderr, "--digits must be 0..62\n");
        return false;
      }
      o->digits = unsigned(d);
    } else if (a == "--start") {
      if (!need_value(argc, argv, &i, "--start", &v) || !parse_u64(v, &o->start)) {
        std::fprintf(stderr, "bad --start\n");
        return false;
      }
    } else if (a == "--count") {
      if (!need_value(argc, argv, &i, "--count", &v) || !parse_u64(v, &o->count)) {
        std::fprintf(stderr, "bad --count\n");
        return false;
      }
    } else if (a == "--stop-after") {
      if (!need_value(argc, argv, &i, "--stop-after", &v) || !parse_u64(v, &o->stop_after)) {
        std::fprintf(stderr, "bad --stop-after\n");
        return false;
      }
    } else if (a == "--time") {
      if (!need_value(argc, argv, &i, "--time", &v) || !parse_seconds(v, &o->time_limit)) {
        std::fprintf(stderr, "bad --time\n");
        return false;
      }
    } else if (a == "--device") {
      uint32_t d = 0;
      if (!need_value(argc, argv, &i, "--device", &v) || !parse_u32(v, &d) || d > INT_MAX) {
        std::fprintf(stderr, "bad --device\n");
        return false;
      }
      o->device = int(d);
    } else if (a == "--threads") {
      if (!need_value(argc, argv, &i, "--threads", &v) || !parse_u32(v, &o->threads)) {
        std::fprintf(stderr, "bad --threads\n");
        return false;
      }
    } else if (a == "--batch") {
      if (!need_value(argc, argv, &i, "--batch", &v) || !parse_u32(v, &o->per_thread)) {
        std::fprintf(stderr, "bad --batch\n");
        return false;
      }
    } else if (a == "--blocks") {
      if (!need_value(argc, argv, &i, "--blocks", &v) || !parse_u32(v, &o->blocks)) {
        std::fprintf(stderr, "bad --blocks\n");
        return false;
      }
    } else if (a == "--progress") {
      if (!need_value(argc, argv, &i, "--progress", &v) || !parse_seconds(v, &o->progress)) {
        std::fprintf(stderr, "bad --progress\n");
        return false;
      }
    } else if (a == "--checkpoint") {
      if (!need_value(argc, argv, &i, "--checkpoint", &v)) return false;
      o->checkpoint = v;
    } else if (a == "--hits-file") {
      if (!need_value(argc, argv, &i, "--hits-file", &v)) return false;
      o->hits_file = v;
    } else if (a == "--resume") {
      o->resume = true;
    } else if (a == "--no-verify") {
      o->verify = false;
    } else if (a == "--help" || a == "-h") {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
      return false;
    }
  }
  if (!o->have_key) {
    std::fprintf(stderr, "--key or --key-hex is required\n");
    return false;
  }
  if (o->threads == 0 || o->threads > 256 || o->threads % 32) {
    std::fprintf(stderr, "--threads must be a multiple of 32, at most 256\n");
    return false;
  }
  if (o->per_thread == 0) {
    std::fprintf(stderr, "--batch must be >= 1\n");
    return false;
  }
  if (o->resume && o->checkpoint.empty()) {
    std::fprintf(stderr, "--resume needs --checkpoint FILE\n");
    return false;
  }
  return true;
}

bool close_written_file(FILE* f, const char* path) {
  const bool failed = std::ferror(f) != 0;
  const int closed = std::fclose(f);
  if (failed || closed != 0) {
    std::fprintf(stderr, "cannot finish writing %s\n", path);
    return false;
  }
  return true;
}

bool checkpoint_write(const Options& o, uint64_t next, uint64_t scanned) {
  if (o.checkpoint.empty()) return true;
  const std::string tmp = o.checkpoint + ".tmp";
  FILE* f = std::fopen(tmp.c_str(), "wb");
  if (!f) {
    std::fprintf(stderr, "cannot write %s\n", tmp.c_str());
    return false;
  }
  std::fprintf(f, "key_hex=%s\n", hex_encode(o.key.data(), o.key.size()).c_str());
  std::fprintf(f, "prefix_hex=%s\n", hex_encode(o.prefix.data(), o.prefix.size()).c_str());
  std::fprintf(f, "digits=%u\n", o.digits);
  std::fprintf(f, "next=%" PRIu64 "\n", next);
  std::fprintf(f, "scanned=%" PRIu64 "\n", scanned);
  if (!close_written_file(f, tmp.c_str())) return false;
  if (std::rename(tmp.c_str(), o.checkpoint.c_str()) != 0) {
    std::fprintf(stderr, "cannot rename checkpoint into place\n");
    return false;
  }
  return true;
}

bool checkpoint_read(const Options& o, uint64_t* next, bool allow_missing = false) {
  FILE* f = std::fopen(o.checkpoint.c_str(), "rb");
  if (!f) {
    if (allow_missing && errno == ENOENT) return true;
    std::fprintf(stderr, "cannot open checkpoint %s: %s\n", o.checkpoint.c_str(),
                 std::strerror(errno));
    return false;
  }
  std::string key_hex, prefix_hex;
  uint64_t ck_next = 0, ck_digits = 0;
  bool have_key = false, have_prefix = false, have_next = false, have_digits = false;
  bool valid = true;

  // key_hex and prefix_hex are as long as the key and prefix, so a record has
  // no fixed bound; getline grows its buffer to fit the full record.
  char* buf = nullptr;
  size_t cap = 0;
  ssize_t len;
  while ((len = getline(&buf, &cap, f)) >= 0) {
    const std::string line(buf, size_t(len));
    const size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    const std::string k = line.substr(0, eq);
    std::string val = line.substr(eq + 1);
    while (!val.empty() && (val.back() == '\n' || val.back() == '\r')) val.pop_back();
    if (k == "key_hex") {
      key_hex = val;
      have_key = true;
    } else if (k == "prefix_hex") {
      prefix_hex = val;
      have_prefix = true;
    } else if (k == "digits") {
      have_digits = true;
      if (!parse_u64(val.c_str(), &ck_digits) || ck_digits > 62) valid = false;
    } else if (k == "next") {
      have_next = true;
      if (!parse_u64(val.c_str(), &ck_next)) valid = false;
    }
  }
  std::free(buf);
  const bool read_failed = std::ferror(f) != 0;
  const int closed = std::fclose(f);
  if (read_failed || closed != 0 || !valid || !have_key || !have_prefix || !have_next ||
      !have_digits) {
    std::fprintf(stderr, "incomplete or invalid checkpoint at %s\n", o.checkpoint.c_str());
    return false;
  }

  if (key_hex != hex_encode(o.key.data(), o.key.size())) {
    std::fprintf(stderr, "checkpoint key does not match --key\n");
    return false;
  }
  if (prefix_hex != hex_encode(o.prefix.data(), o.prefix.size())) {
    std::fprintf(stderr, "checkpoint prefix does not match --prefix\n");
    return false;
  }
  if (ck_digits != o.digits) {
    std::fprintf(stderr, "checkpoint digits=%" PRIu64 " does not match --digits %u\n", ck_digits,
                 o.digits);
    return false;
  }
  *next = ck_next;
  std::fprintf(stderr, "resuming at counter %" PRIu64 "\n", *next);
  return true;
}

bool hit_write(const Options& o, uint64_t counter, const std::string& body,
               const std::string& body_hex, const char* digest) {
  if (o.hits_file.empty()) return true;
  FILE* f = std::fopen(o.hits_file.c_str(), "ab");
  if (!f) {
    std::fprintf(stderr, "cannot append hits to %s\n", o.hits_file.c_str());
    return false;
  }
  std::fprintf(f, "counter=%" PRIu64 "\nbody=%s\nbody_hex=%s\ndigest=%s\nkey_hex=%s\n\n",
               counter, body.c_str(), body_hex.c_str(), digest, hex_encode(o.key.data(), o.key.size()).c_str());
  return close_written_file(f, o.hits_file.c_str());
}

double seconds_since(std::chrono::steady_clock::time_point t0) {
  using namespace std::chrono;
  return duration<double>(steady_clock::now() - t0).count();
}

// The counter becomes the 20 zero-padded decimal digits appended to the
// prefix, and a window of 10^8 counters fixes the leading 12 of them, so one
// context build per window keeps the two-compression floor. The format itself
// lives in hmac0e.cuh, shared with the test oracles.
const uint64_t kWindow = HMAC0E_WINDOW;

// Exit status when a rerun with the same arguments cannot help: a hit buffer
// overflow repeats at the same geometry, and a rejected device digest means
// the device results are wrong. Status 1 (CUDA or I/O failure) is retryable.
const int kExitNoRetry = 3;

void report_line(uint64_t scanned, double secs, uint64_t hits) {
  const double rate = secs > 0 ? double(scanned) / secs : 0.0;
  std::fprintf(stderr, "\r%8.1fs  %6.2f GH/s  %3.4e candidates  %" PRIu64 " %s        ", secs,
               rate / 1e9, double(scanned), hits, hits == 1 ? "hit" : "hits");
  std::fflush(stderr);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage(argv[0]);
    return 2;
  }
  Options o;
  if (!parse_args(argc, argv, &o)) return 2;

  uint64_t next = o.start;
  if (!o.checkpoint.empty() && !checkpoint_read(o, &next, !o.resume)) return 1;

  // The context depends on the window (the leading 12 digits), so build one
  // per iteration below. Validate the shape once here, at window 0: legality
  // depends only on the length, which is the same in every window.
  HmacCtx ctx;
  const char* err = nullptr;
  auto build_window_ctx = [&](uint64_t window) -> bool {
    const std::vector<uint8_t> eff = window_prefix(o.prefix, window);
    return hmac_ctx_build(o.key.data(), o.key.size(), eff.data(), eff.size(), &ctx, &err);
  };
  if (!build_window_ctx(0)) {
    std::fprintf(stderr, "cannot build HMAC context: %s\n", err);
    return 2;
  }

  if (cudaSetDevice(o.device) != cudaSuccess) {
    std::fprintf(stderr, "cannot select CUDA device %d\n", o.device);
    return 1;
  }
  cudaDeviceProp props{};
  if (cudaGetDeviceProperties(&props, o.device) != cudaSuccess) {
    std::fprintf(stderr, "cannot query CUDA device %d\n", o.device);
    return 1;
  }

  // Ask the runtime how many blocks actually fit per SM rather than guessing;
  // the answer follows from the kernel's register usage.
  uint32_t blocks_per_sm = 0;
  if (hmac0e_blocks_per_sm(ctx.counter_word, o.threads, &blocks_per_sm) != 0) return 1;
  if (blocks_per_sm == 0) {
    std::fprintf(stderr, "kernel cannot be resident with %u threads\n", o.threads);
    return 1;
  }
  const uint32_t blocks = (o.blocks != 0) ? o.blocks : props.multiProcessorCount * blocks_per_sm;

  // Checked once here: launch_range needs a coverage of at least 1, or an
  // empty launch would advance nothing and spin forever.
  const uint64_t coverage = grid_coverage(blocks, o.threads, o.per_thread);
  if (coverage == 0) {
    std::fprintf(stderr, "grid %u x %u x %u exceeds 2^64 counters per launch\n", blocks,
                 o.threads, o.per_thread);
    return 2;
  }

  HitBuffer buf;
  if (hmac0e_alloc(&buf) != 0) return 1;

  std::fprintf(stderr,
               "device %d: %s (%d SMs)\n"
               "key %zu bytes, prefix %zu bytes, digits %u, body = prefix || 20 digits\n"
               "grid %u x %u, %u counters/thread -> %.3e per launch\n",
               o.device, props.name, props.multiProcessorCount, o.key.size(), o.prefix.size(),
               o.digits, blocks, o.threads, o.per_thread, double(coverage));

  std::vector<Hit> hits(HMAC0E_MAX_HITS);
  const auto t0 = std::chrono::steady_clock::now();
  auto last_ckpt = t0 - std::chrono::seconds(1);  // first batch checkpoints immediately
  double next_report = o.progress;
  uint64_t scanned = 0;
  uint64_t total_hits = 0;
  bool exhausted = false;
  int rc = 0;

  while (true) {
    if (total_hits >= o.stop_after) break;
    if (o.count != 0 && scanned >= o.count) break;
    if (o.time_limit > 0 && seconds_since(t0) >= o.time_limit) break;

    const uint64_t count_left = (o.count == 0) ? 0 : o.count - scanned;
    const LaunchRange r = launch_range(next, coverage, count_left);
    if (!build_window_ctx(r.window)) {
      std::fprintf(stderr, "cannot build HMAC context for window %" PRIu64 ": %s\n", r.window, err);
      rc = 1;
      break;
    }

    if (hmac0e_launch(ctx, r.c0, r.want, o.digits, blocks, o.threads, o.per_thread, &buf, 0) !=
        0) {
      rc = 1;
      break;
    }

    uint32_t found = 0;
    if (hmac0e_drain(&buf, hits.data(), HMAC0E_MAX_HITS, &found) != 0) {
      rc = 1;
      break;
    }
    if (found > hits.size()) {
      // A rejected slot is a gap in the result set. Record nothing from this
      // batch and leave the checkpoint before it, so a resumed run re-scans it
      // with a smaller range instead of reporting a partial hit set.
      std::fprintf(stderr,
                   "hit buffer overflow: %" PRIu32 " hits in one launch, %zu slots;"
                   " lower --blocks/--batch or raise --digits;"
                   " checkpoint left before this batch\n",
                   found, hits.size());
      rc = kExitNoRetry;
      break;
    }

    for (uint32_t i = 0; i < found; ++i) {
      const uint64_t g = r.window * kWindow + hits[i].counter;
      const std::string body = body_of(o.prefix, g);
      char digest[65];
      digest_to_hex(hits[i].digest, digest);

      bool ok = true;
      if (o.verify) {
        uint32_t want_digest[8] = {0};
        hmac_sha256(o.key.data(), o.key.size(),
                    reinterpret_cast<const uint8_t*>(body.data()), body.size(), want_digest);
        char check[65];
        digest_to_hex(want_digest, check);
        ok = std::strcmp(check, digest) == 0 && magic_ok(want_digest, o.digits);
      }

      if (o.progress > 0) {
        // Terminate the \r progress line so the hit starts on a new line; the
        // hit's own \n leaves a fresh line for the next progress update.
        std::fputc('\n', stderr);
      }
      // body_hex is authoritative when the body is not printable (NUL bytes
      // truncate the text form, exactly as in the hits file).
      const std::string body_hex =
          hex_encode(reinterpret_cast<const uint8_t*>(body.data()), body.size());
      const int printed =
          std::fprintf(stdout, "%s counter=%" PRIu64 " body=%s body_hex=%s digest=%s\n",
                       ok ? "HIT" : "REJECT", g, body.c_str(), body_hex.c_str(), digest);
      const int flushed = std::fflush(stdout);
      if (printed < 0 || flushed != 0 || (ok && !hit_write(o, g, body, body_hex, digest))) {
        std::fprintf(stderr, "cannot save result; checkpoint left before this batch\n");
        rc = 1;
        break;
      }
      if (ok) {
        ++total_hits;
      } else {
        // The CPU re-check is the only integrity gate on a long unattended run.
        // A digest the device reported that the reference HMAC rejects means no
        // device result can be trusted, so stop with the checkpoint before this
        // batch rather than scanning for hours and exiting 0.
        std::fprintf(stderr,
                     "verification failed for counter %" PRIu64
                     "; checkpoint left before this batch\n",
                     g);
        rc = kExitNoRetry;
        break;
      }
    }
    if (rc != 0) break;

    exhausted = r.exhausted;
    next += r.want;
    scanned += r.want;
    if (exhausted) {
      std::fprintf(stderr, "counter space exhausted\n");
      break;
    }
    // Checkpoint about once per second rather than per launch: a launch
    // finishes in a few milliseconds, and each checkpoint is an
    // open/write/close/rename. A crash or Ctrl+C re-scans at most the last
    // second of candidates.
    const auto now = std::chrono::steady_clock::now();
    if (now - last_ckpt >= std::chrono::seconds(1)) {
      if (!checkpoint_write(o, next, scanned)) {
        rc = 1;
        break;
      }
      last_ckpt = now;
    }

    if (o.progress > 0) {
      const double secs = seconds_since(t0);
      if (secs >= next_report) {
        report_line(scanned, secs, total_hits);
        next_report = secs + o.progress;
      }
    }
  }

  // Persist the final state even when the run ends less than a second after
  // the last timed write. A wrapped counter must never be written: resume
  // would read next == 0 as "start from the beginning" and rescan all of 2^64.
  if (!exhausted && !checkpoint_write(o, next, scanned)) rc = 1;

  if (o.progress > 0) std::fprintf(stderr, "\n");
  const double secs = seconds_since(t0);
  std::fprintf(stderr,
               "done: %" PRIu64 " candidates in %.2fs (%.2f GH/s), %" PRIu64 " %s\n", scanned,
               secs, secs > 0 ? double(scanned) / secs / 1e9 : 0.0, total_hits,
               total_hits == 1 ? "hit" : "hits");

  hmac0e_free(&buf);
  return rc;
}
