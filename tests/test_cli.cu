// Exercise the actual host helpers without starting a search or using a GPU.
#define main hmac0e_main
#include "../src/main.cu"
#undef main

#include <csignal>
#include <fstream>
#include <iterator>
#include <sys/resource.h>
#include <sys/stat.h>

static int checks = 0;
static int failures = 0;

static void check(bool ok, const char* what) {
  ++checks;
  if (!ok) {
    ++failures;
    std::printf("FAIL: %s\n", what);
  }
}

static std::string read_file(const std::string& path) {
  std::ifstream f(path);
  return std::string(std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>());
}

static void write_file(const std::string& path, const std::string& content) {
  std::ofstream f(path);
  f << content;
  f.close();
  check(!f.fail(), "write test fixture");
}

static void test_numbers() {
  struct Case { const char* text; uint64_t value; };
  const Case good[] = {{"0", 0}, {"42", 42}, {"010", 10}, {"08", 8},
                       {"00000434871031216900", 434871031216900},
                       {"18446744073709551615", UINT64_MAX}};
  for (const Case& c : good) {
    uint64_t value = 123;
    errno = ERANGE;
    check(parse_u64(c.text, &value) && value == c.value, c.text);
  }
  const char* bad[] = {"", "-1", "-0", "+1", " 1", "1 ", "1junk", "0x", "0xff",
                       "18446744073709551616", "999999999999999999999999"};
  for (const char* text : bad) {
    uint64_t value = 123;
    check(!parse_u64(text, &value) && value == 123, text);
  }
  uint64_t value = 0;
  check(parse_u64("000010", &value) && value == 10, "checkpoint values are decimal");

  const char* flags[] = {"--start", "--count", "--stop-after"};
  for (const char* flag : flags) {
    for (const char* text : {"", "-1", "18446744073709551616"}) {
      const char* args[] = {"hmac0e", flag, text};
      check(hmac0e_main(3, const_cast<char**>(args)) == 2, "reject invalid CLI counter");
    }
  }

  const char* seconds[] = {"--time", "--progress"};
  for (const char* flag : seconds) {
    for (const char* text : {"", "abc", "-1", "2h", "1e999", "nan"}) {
      const char* args[] = {"hmac0e", "--key", "k", flag, text};
      check(hmac0e_main(5, const_cast<char**>(args)) == 2, "reject invalid CLI seconds");
    }
  }
  const char* geometry[] = {"--device", "--threads", "--batch", "--blocks"};
  for (const char* flag : geometry) {
    for (const char* text : {"", "abc", "-1", "2h", "1.5", "4294967296"}) {
      const char* args[] = {"hmac0e", "--key", "k", flag, text};
      check(hmac0e_main(5, const_cast<char**>(args)) == 2, "reject invalid CLI geometry");
    }
  }
  const char* big_device[] = {"hmac0e", "--key", "k", "--device", "2147483648"};
  check(hmac0e_main(5, const_cast<char**>(big_device)) == 2, "reject device beyond int");

  Options o;
  const char* args[] = {"hmac0e", "--key", "k", "--start", "00000434871031216900",
                        "--time", "1.5", "--progress", "0", "--device", "1",
                        "--threads", "128", "--batch", "64", "--blocks", "7"};
  check(parse_args(17, const_cast<char**>(args), &o), "accept valid CLI values");
  check(o.start == 434871031216900, "--start parses zero-padded decimal");
  check(o.time_limit == 1.5 && o.progress == 0, "parse CLI seconds");
  check(o.device == 1 && o.threads == 128 && o.per_thread == 64 && o.blocks == 7,
        "parse CLI geometry");
}

static void test_key_required() {
  const char* no_key[] = {"hmac0e", "--prefix", "magic-hash-"};
  check(hmac0e_main(3, const_cast<char**>(no_key)) == 2, "reject missing key");
  // An explicit empty key is a valid HMAC key. Parse directly: accepting the
  // args would let hmac0e_main proceed to the CUDA context.
  const char* empty_key[] = {"hmac0e", "--key", ""};
  Options o;
  check(parse_args(3, const_cast<char**>(empty_key), &o) && o.key.empty() && o.have_key,
        "accept explicit empty --key");
  const char* empty_hex[] = {"hmac0e", "--key-hex", ""};
  Options o2;
  check(parse_args(3, const_cast<char**>(empty_hex), &o2) && o2.key.empty() && o2.have_key,
        "accept explicit empty --key-hex");
}

static void test_checkpoints(Options o) {
  check(checkpoint_write(o, UINT64_MAX, 123), "write checkpoint");
  uint64_t next = 0;
  check(checkpoint_read(o, &next) && next == UINT64_MAX, "round-trip max counter");

  Options mismatch = o;
  mismatch.digits = 61;
  check(!checkpoint_read(mismatch, &next), "reject predicate mismatch");
  mismatch = o;
  mismatch.key.push_back('x');
  check(!checkpoint_read(mismatch, &next), "reject key mismatch");
  mismatch = o;
  mismatch.prefix.push_back('x');
  check(!checkpoint_read(mismatch, &next), "reject prefix mismatch");

  const std::string fields[] = {"key_hex=\n", "prefix_hex=\n", "digits=62\n", "next=10\n"};
  for (unsigned omitted = 0; omitted < 4; ++omitted) {
    std::string content;
    for (unsigned i = 0; i < 4; ++i) if (i != omitted) content += fields[i];
    write_file(o.checkpoint, content);
    next = 123;
    check(!checkpoint_read(o, &next) && next == 123, "reject missing checkpoint field");
  }
  const char* bad[] = {"", "-1", "10junk", "18446744073709551616"};
  for (const char* value : bad) {
    write_file(o.checkpoint, "key_hex=\nprefix_hex=\ndigits=62\nnext=" +
                                 std::string(value) + "\n");
    check(!checkpoint_read(o, &next), "reject invalid checkpoint counter");
  }
  write_file(o.checkpoint, "key_hex=\nprefix_hex=\ndigits=garbage\nnext=10\n");
  check(!checkpoint_read(o, &next), "reject invalid checkpoint digits");
  write_file(o.checkpoint, "key_hex=\nprefix_hex=\ndigits=62\nnext=000010\n");
  check(checkpoint_read(o, &next) && next == 10, "read decimal checkpoint counter");
}

static void test_auto_resume(Options o) {
  o.checkpoint += ".auto-resume";
  o.start = 123;
  uint64_t next = o.start;
  check(checkpoint_read(o, &next, true) && next == o.start,
        "missing optional checkpoint preserves start counter");
  check(!std::ifstream(o.checkpoint).good(), "reading missing checkpoint creates no file");
  check(!checkpoint_read(o, &next) && next == o.start, "explicit resume requires checkpoint");

  check(checkpoint_write(o, 456, 333), "write auto-resume fixture");
  const std::string original = read_file(o.checkpoint);
  check(checkpoint_read(o, &next, true) && next == 456, "auto-resume uses saved counter");
  check(read_file(o.checkpoint) == original, "auto-resume preserves checkpoint content");
  Options mismatch = o;
  mismatch.key.push_back('x');
  check(!checkpoint_read(mismatch, &next, true), "auto-resume rejects mismatched search");

  write_file(o.checkpoint, "invalid checkpoint\n");
  next = o.start;
  check(!checkpoint_read(o, &next, true) && next == o.start,
        "invalid existing checkpoint never falls back to a new search");
  const char* args[] = {"hmac0e", "--key", "k", "--checkpoint", o.checkpoint.c_str()};
  check(hmac0e_main(5, const_cast<char**>(args)) == 1,
        "CLI checks existing checkpoint without --resume before accessing GPU");
  check(read_file(o.checkpoint) == "invalid checkpoint\n", "failed startup preserves checkpoint");

  Options inaccessible = o;
  inaccessible.checkpoint += "/checkpoint";
  check(!checkpoint_read(inaccessible, &next, true),
        "filesystem errors never fall back to a new search");

  const char* no_path[] = {"hmac0e", "--key", "k", "--resume"};
  check(hmac0e_main(4, const_cast<char**>(no_path)) == 2, "--resume requires a path");
  o.checkpoint += ".missing";
  const char* required[] = {"hmac0e", "--key", "k", "--checkpoint", o.checkpoint.c_str(),
                            "--resume"};
  check(hmac0e_main(6, const_cast<char**>(required)) == 1,
        "CLI explicit resume rejects missing checkpoint before accessing GPU");
}

static void test_write_failures(const Options& o) {
  check(checkpoint_write(o, 100, 100), "write original checkpoint");
  const std::string original = read_file(o.checkpoint);
  FILE* unbuffered = std::fopen((o.hits_file + ".unbuffered").c_str(), "wb");
  if (!unbuffered) {
    check(false, "open unbuffered test file");
    return;
  }
  std::setvbuf(unbuffered, nullptr, _IONBF, 0);

  // Limit regular files to zero bytes to force a buffered flush/close failure.
  // No real search paths or devices are opened. Restore before further writes.
  struct rlimit old_limit{};
  if (getrlimit(RLIMIT_FSIZE, &old_limit) != 0) {
    std::fclose(unbuffered);
    check(false, "get file size limit");
    return;
  }
  auto old_signal = std::signal(SIGXFSZ, SIG_IGN);
  struct rlimit limit = old_limit;
  limit.rlim_cur = 0;
  if (setrlimit(RLIMIT_FSIZE, &limit) != 0) {
    std::fclose(unbuffered);
    std::signal(SIGXFSZ, old_signal);
    check(false, "set file size limit");
    return;
  }
  const bool checkpoint_ok = checkpoint_write(o, 200, 200);
  const bool hit_ok = hit_write(o, 200, "00", "3030", "digest");
  const int printed = std::fprintf(unbuffered, "hit");
  const bool unbuffered_ok = close_written_file(unbuffered, "unbuffered test file");
  const int restored = setrlimit(RLIMIT_FSIZE, &old_limit);
  std::signal(SIGXFSZ, old_signal);
  check(restored == 0, "restore file size limit");
  check(!checkpoint_ok, "reject buffered checkpoint failure");
  check(read_file(o.checkpoint) == original, "preserve previous checkpoint on write failure");
  check(!hit_ok, "reject buffered hit-file failure");
  check(printed < 0 && !unbuffered_ok, "reject immediate write failure even if close succeeds");

  Options missing = o;
  missing.hits_file += "/missing/hits";
  check(!hit_write(missing, 0, "body", "626f6479", "digest"), "reject hit-file open failure");
  missing.checkpoint += "/missing/checkpoint";
  check(!checkpoint_write(missing, 0, 0), "reject checkpoint open failure");
}

static void test_hits(const Options& o) {
  check(hit_write(o, 7, "body", "626f6479", "digest"), "save hit");
  check(hit_write(o, 8, "body2", "626f647932", "digest2"), "append hit");
  check(read_file(o.hits_file) ==
            "counter=7\nbody=body\nbody_hex=626f6479\ndigest=digest\nkey_hex=\n\n"
            "counter=8\nbody=body2\nbody_hex=626f647932\ndigest=digest2\nkey_hex=\n\n",
        "preserve both saved hits");
  Options binary = o;
  binary.hits_file += ".binary";
  const std::string body("\0ab", 3);
  const std::string body_hex =
      hex_encode(reinterpret_cast<const uint8_t*>(body.data()), body.size());
  check(body_hex == "006162", "hex-encode bytes after NUL");
  check(hit_write(binary, 9, body, body_hex, "digest"), "save binary-prefix hit");
  check(read_file(binary.hits_file) ==
            std::string("counter=9\nbody=\nbody_hex=006162\ndigest=digest\nkey_hex=\n\n"),
        "body_hex preserves bytes after NUL");
  Options disabled = o;
  disabled.hits_file.clear();
  disabled.checkpoint.clear();
  check(hit_write(disabled, 0, "body", "626f6479", "digest"), "optional hit file");
  check(checkpoint_write(disabled, 0, 0), "optional checkpoint");
}

int main() {
  // mkdtemp gives each invocation its own files, left available for inspection.
  if (mkdir("tmp", 0777) != 0 && errno != EEXIST) return 1;
  char path[] = "tmp/hmac0e-cli-tests-XXXXXX";
  if (!mkdtemp(path)) return 1;
  Options o;
  o.checkpoint = std::string(path) + "/checkpoint";
  o.hits_file = std::string(path) + "/hits";
  test_numbers();
  test_key_required();
  test_checkpoints(o);
  test_auto_resume(o);
  test_write_failures(o);
  test_hits(o);
  std::printf("%d checks, %d failures (fixtures: %s)\n", checks, failures, path);
  return failures == 0 ? 0 : 1;
}
