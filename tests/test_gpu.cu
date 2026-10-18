// GPU test suite. Requires a usable CUDA context.
//
// Strategy: the kernel only reports hits, so make hits dense enough to
// enumerate. With --digits 0 roughly 1 candidate in 256 matches, which lets the
// CPU oracle enumerate the same counter range and compare the *exact hit set*.
// That checks the kernel's digests, its predicate, its counter placement and its
// thread/block index arithmetic all at once.
#include <cuda_runtime.h>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "hmac0e.cuh"
#include "hmac_kernel.cuh"

static int g_checks = 0;
static int g_failures = 0;

static void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("  FAIL  %s\n", what.c_str());
  }
}

static std::vector<uint8_t> str(const char* s) {
  const size_t n = std::strlen(s);
  return std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(s),
                              reinterpret_cast<const uint8_t*>(s) + n);
}

// The counter becomes the 20 zero-padded decimal digits appended to the
// prefix, searched in 10^8 windows (the leading 12 digits are fixed per
// window). See hmac0e.cuh.
static const uint64_t kWindow = HMAC0E_WINDOW;

static std::string padded_dec(uint64_t v, unsigned width) {
  std::string s(width, '0');
  for (int i = int(width) - 1; i >= 0; --i) {
    s[i] = char('0' + v % 10u);
    v /= 10u;
  }
  return s;
}

static std::vector<uint8_t> window_prefix(const std::vector<uint8_t>& p, uint64_t window) {
  std::vector<uint8_t> eff = p;
  const std::string w = padded_dec(window, 12);
  eff.insert(eff.end(), w.begin(), w.end());
  return eff;
}

// Independent oracle: full HMAC-SHA256 per counter over the true body, no
// hoisting involved.
static std::vector<uint64_t> oracle_hits(const std::vector<uint8_t>& key,
                                         const std::vector<uint8_t>& prefix, uint64_t base,
                                         uint64_t n, unsigned digits) {
  std::vector<uint64_t> out;
  std::vector<uint8_t> msg = prefix;
  msg.resize(prefix.size() + 20);
  for (uint64_t i = 0; i < n; ++i) {
    const std::string d20 = padded_dec(base + i, 20);
    std::memcpy(msg.data() + prefix.size(), d20.data(), 20);
    uint32_t d[8] = {0};
    hmac_sha256(key.data(), key.size(), msg.data(), msg.size(), d);
    if (magic_ok(d, digits)) out.push_back(base + i);
  }
  return out;
}

// Launches per window, building the window context on the way; hit counters
// come back global.
static bool gpu_hits(const std::vector<uint8_t>& key, const std::vector<uint8_t>& prefix,
                     uint64_t base, uint64_t n, unsigned digits, uint32_t blocks,
                     uint32_t threads, uint32_t per_thread, std::vector<Hit>* out) {
  HitBuffer buf;
  if (hmac0e_alloc(&buf) != 0) return false;
  bool ok = true;
  uint64_t g = base, left = n;
  while (left > 0 && ok) {
    const uint64_t window = g / kWindow;
    const uint64_t c0 = g % kWindow;
    const std::vector<uint8_t> eff = window_prefix(prefix, window);
    HmacCtx ctx;
    const char* err = nullptr;
    if (!hmac_ctx_build(key.data(), key.size(), eff.data(), eff.size(), &ctx, &err)) {
      check(false, "window " + std::to_string(window) + " build failed: " + (err ? err : "?"));
      ok = false;
      break;
    }
    const uint64_t space = (g == 0) ? UINT64_MAX : (~g + 1); // left in 2^64
    const uint64_t avail = std::min(kWindow - c0, space);    // left in this window
    const uint64_t grid_cov = uint64_t(blocks) * threads * per_thread;
    const uint64_t take = std::min({left, avail, grid_cov});
    if (hmac0e_launch(ctx, c0, take, digits, blocks, threads, per_thread, &buf, 0) != 0) {
      ok = false;
      break;
    }
    uint32_t found = 0;
    if (hmac0e_drain(&buf, nullptr, 0, &found) != 0) {
      ok = false;
      break;
    }
    check(found <= buf.max_hits, "hits exceeded the device buffer in a test range");
    const uint32_t at = (uint32_t)out->size();
    out->resize(at + found);
    if (found && hmac0e_drain(&buf, out->data() + at, found, &found) != 0) {
      ok = false;
      break;
    }
    for (uint32_t i = 0; i < found; ++i)
      out->data()[at + i].counter = window * kWindow + out->data()[at + i].counter;
    g += take;
    left -= take;
  }
  hmac0e_free(&buf);
  return ok;
}

// Every hit the GPU reports must be real, and the set must equal the oracle's.
static void compare_hit_sets(const std::string& label, const std::vector<uint8_t>& key,
                             const std::vector<uint8_t>& prefix, uint64_t base, uint64_t n,
                             unsigned digits, uint32_t blocks, uint32_t threads,
                             uint32_t per_thread) {
  std::vector<Hit> got;
  if (!gpu_hits(key, prefix, base, n, digits, blocks, threads, per_thread, &got)) {
    check(false, label + ": kernel launch failed");
    return;
  }
  const std::vector<uint64_t> want = oracle_hits(key, prefix, base, n, digits);

  std::vector<uint64_t> got_counters;
  int bad_digest = 0;
  for (const Hit& h : got) {
    // Each reported digest must reproduce under the independent reference.
    std::vector<uint8_t> msg = prefix;
    const std::string d20 = padded_dec(h.counter, 20);
    msg.insert(msg.end(), d20.begin(), d20.end());
    uint32_t d[8] = {0};
    hmac_sha256(key.data(), key.size(), msg.data(), msg.size(), d);
    if (std::memcmp(d, h.digest, sizeof(d)) != 0) ++bad_digest;
    got_counters.push_back(h.counter);
  }
  check(bad_digest == 0, label + ": " + std::to_string(bad_digest) + " reported digests wrong");

  std::sort(got_counters.begin(), got_counters.end());
  const bool dup = std::adjacent_find(got_counters.begin(), got_counters.end()) !=
                   got_counters.end();
  check(!dup, label + ": duplicate counters reported");
  check(got_counters == want, label + ": hit set gpu=" + std::to_string(got_counters.size()) +
                                  " cpu=" + std::to_string(want.size()));
}

static void test_prefix_lengths() {
  std::printf("kernel hit set vs CPU oracle, across prefix lengths\n");
  const auto key = str("session-secret");
  // Lengths of the user prefix. The context sees prefix || 12 digits, so the
  // counter-slot lane is (plen + 12) % 64; this list covers 12, 13, 0, 1, 2,
  // 3, 4, 7, 15, 16, 31, 47 (max) and 0 again with two hoisted blocks.
  const size_t lens[] = {0, 1, 52, 53, 54, 55, 56, 59, 67, 68, 83, 99, 116};
  for (size_t plen : lens) {
    std::vector<uint8_t> prefix(plen);
    uint32_t s = 0x9e3779b9u + uint32_t(plen);
    for (size_t i = 0; i < plen; ++i) {
      s = s * 1664525u + 1013904223u;
      prefix[i] = uint8_t(s >> 24);
    }
    compare_hit_sets("plen=" + std::to_string(plen) + " digits=0", key, prefix, 0, 8192, 0, 16,
                     256, 8);
  }
}

// Vary the launch geometry and the base offset so the tid*per_thread index
// math is exercised, not just the hash.
static void test_geometry() {
  std::printf("launch geometry and base offsets\n");
  const auto key = str("k3y");
  const auto prefix = str("user=42&action=export&nonce=");
  // The under-covering guard fires before the launch, so the context content
  // is irrelevant; build the real window-0 shape anyway.
  const std::vector<uint8_t> eff = window_prefix(prefix, 0);
  HmacCtx ctx;
  const char* err = nullptr;
  if (!hmac_ctx_build(key.data(), key.size(), eff.data(), eff.size(), &ctx, &err)) {
    check(false, std::string("build failed: ") + err);
    return;
  }
  // An under-covering grid must be rejected, not silently truncated: a skipped
  // candidate range would otherwise be stamped as scanned by the checkpoint.
  {
    HitBuffer buf;
    if (hmac0e_alloc(&buf) == 0) {
      check(hmac0e_launch(ctx, 0, 4096, 0, 1, 32, 1, &buf, 0) != 0,
            "under-covering grid must be rejected");
      check(hmac0e_launch(ctx, kWindow - 1, 2, 0, 1, 32, 1, &buf, 0) != 0,
            "range past the window end must be rejected");
      check(hmac0e_launch(ctx, kWindow, 1, 0, 1, 32, 1, &buf, 0) != 0,
            "base outside the window must be rejected");
      hmac0e_free(&buf);
    }
  }

  struct Geo { uint64_t base; uint64_t n; uint32_t blocks; uint32_t threads; uint32_t per; };
  const Geo geos[] = {
      {0, 4096, 16, 256, 1},
      {0, 4096, 1, 256, 16},
      {0, 4096, 4, 128, 8},
      {1000000, 4096, 4, 256, 4},
      {99999500, 1000, 4, 256, 4},  // spans the 10^8 window boundary
      {0xffffffffffffff00ull, 256, 2, 128, 8},  // last partial window, 2^64 - 256
      {12345, 5000, 7, 96, 11},                 // non-power-of-two everything
      {0, 1, 1, 32, 1},                         // single candidate
  };
  for (const Geo& g : geos) {
    const std::string label = "base=" + std::to_string(g.base) + " n=" + std::to_string(g.n) +
                              " grid=" + std::to_string(g.blocks) + "x" +
                              std::to_string(g.threads) + "x" + std::to_string(g.per);
    compare_hit_sets(label, key, prefix, g.base, g.n, 0, g.blocks, g.threads, g.per);
  }
}

// A denser-than-production predicate, verified end to end.
static void test_digits8() {
  std::printf("digits=8 search\n");
  const auto key = str("php-type-juggling");
  const auto prefix = str("token=");
  compare_hit_sets("digits=8", key, prefix, 0, 200000, 8, 32, 256, 64);
}

int main() {
  if (cudaSetDevice(0) != cudaSuccess || cudaFree(0) != cudaSuccess) {
    std::printf("SKIP: no usable CUDA context (device busy or absent)\n");
    return 77;
  }
  cudaDeviceProp props{};
  cudaGetDeviceProperties(&props, 0);
  std::printf("device 0: %s (%d SMs)\n\n", props.name, props.multiProcessorCount);

  test_prefix_lengths();
  test_geometry();
  test_digits8();

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
