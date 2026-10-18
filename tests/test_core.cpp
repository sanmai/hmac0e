// CPU-only test suite. No CUDA required, so it runs even when the GPU is fully
// occupied by another process.
//
// Covers: SHA-256 against FIPS vectors, HMAC-SHA256 against RFC 4231 plus
// Python-generated boundary vectors, an exhaustive sweep of the magic
// predicate, and - most importantly - the hoisted two-compression candidate
// path against the independent reference HMAC across every legal prefix length.
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "hmac0e.cuh"
#include "launch_range.cuh"

static int g_checks = 0;
static int g_failures = 0;

static void check(bool ok, const std::string& what) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    std::printf("  FAIL  %s\n", what.c_str());
  }
}

static std::string hex(const uint32_t d[8]) {
  char buf[65];
  digest_to_hex(d, buf);
  return std::string(buf);
}

static std::vector<uint8_t> str(const char* s) {
  const size_t n = std::strlen(s);
  return std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(s),
                              reinterpret_cast<const uint8_t*>(s) + n);
}

static std::vector<uint8_t> rep(uint8_t v, size_t n) { return std::vector<uint8_t>(n, v); }

// Deterministic filler so failures reproduce.
static void fill_pseudo(std::vector<uint8_t>& v) {
  uint32_t s = 0x12345678u + uint32_t(v.size()) * 2654435761u;
  for (size_t i = 0; i < v.size(); ++i) {
    s = s * 1664525u + 1013904223u;
    v[i] = uint8_t(s >> 24);
  }
}

// The counter becomes the 8 zero-padded ASCII digits the kernel varies, most
// significant first. The oracles here share window_prefix and body_of with the
// search, so they cannot catch a bug in the body format; the published-fixture
// test anchors that format to a value produced outside this code base.
static void put_ascii8(uint8_t* p, uint64_t c) {
  for (int i = 7; i >= 0; --i) {
    p[i] = uint8_t('0' + c % 10u);
    c /= 10u;
  }
}

// The window baked into the contexts below: a large real window, so the leading
// 12 digits are nonzero (window 0, i.e. all-zeros, is covered by the end-to-end
// shape test and by the published fixture).
const uint64_t kWindowNumber = 18446744073ull;

// Nibble `pos` of the digest as written in hex, 0 = most significant.
static void set_nibble(uint32_t d[8], unsigned pos, unsigned v) {
  const unsigned shift = 28 - 4 * (pos % 8);
  d[pos / 8] = (d[pos / 8] & ~(0xfu << shift)) | ((v & 0xfu) << shift);
}

// ---------------------------------------------------------------- SHA-256

static void test_sha256() {
  std::printf("sha256 vectors\n");
  struct { const char* msg; const char* want; } cases[] = {
      {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
      {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
      {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
       "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
  };
  for (const auto& c : cases) {
    const auto m = str(c.msg);
    uint32_t d[8] = {0};
    sha256::hash_bytes(m.data(), m.size(), d);
    check(hex(d) == c.want, std::string("sha256(\"") + c.msg + "\") -> " + hex(d));
  }

  // One million 'a' exercises the multi-block loop and the 64-bit length field.
  const std::vector<uint8_t> many(1000000, 'a');
  uint32_t d[8] = {0};
  sha256::hash_bytes(many.data(), many.size(), d);
  check(hex(d) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0",
        "sha256(1e6 x 'a') -> " + hex(d));
}

// -------------------------------------------------------------------- HMAC
//
// RFC 4231 cases are marked. The rest were generated with Python's hmac module
// as an independent oracle, and cover the key-length and padding boundaries
// that matter here: the inner hash sees 64 + len(data) bytes, so data lengths
// 55/56 and 119/120 straddle SHA-256's padding switch.

static void test_hmac() {
  std::printf("hmac-sha256 vectors\n");
  struct Case {
    const char* name;
    std::vector<uint8_t> key;
    std::vector<uint8_t> data;
    const char* want;
  } cases[] = {
      {"empty key, empty data", rep('x', 0), rep('y', 0),
       "b613679a0814d9ec772f95d778c35fc5ff1697c493715653c6c712144292c5ad"},
      {"empty key, data", rep('x', 0), str("abc"),
       "fd7adb152c05ef80dccf50a1fa4c05d5a3ec6da95575fc312ae7c5d091836351"},
      {"short key", str("key"), str("The quick brown fox jumps over the lazy dog"),
       "f7bc83f430538424b13298e6aa6fb143ef4d59a14946175997479dbc2d1a3cd8"},
      {"key 63", rep('k', 63), str("msg"),
       "ac14c28089a16fbb10c53caf1ec03ddf477041407b5eacc3f0478749c34e3dff"},
      {"key 64", rep('k', 64), str("msg"),
       "d62ad25bb128e96ab6ef43464aaf2bb91b5b85f013381e9ba6be747e8d0911b2"},
      {"key 65", rep('k', 65), str("msg"),
       "c65585780e821bd312cfaa440c33cb28d6af6b323d4977ae83934834ce589e38"},
      {"key 200", rep('k', 200), str("msg"),
       "e2adadca233bc31c6e6126c865132c3e945f9dedd44797a1e5acc3c037bc21fc"},
      {"data 0", str("K"), rep('d', 0),
       "17e59b72dda7ed7c533fcb5392e94ad08ae60f453c73890d681a7db9e4c91aad"},
      {"data 55", str("K"), rep('d', 55),
       "54b9e35d921d5a7ea0ed4ccc2644a0f1b8e2fecf9ccb6d0d8ab3007745d8aeb9"},
      {"data 56", str("K"), rep('d', 56),
       "0b985d2a9b35ae50e3105d6e9f21b5b03c5a3f39c1914f6f0f4a921965261297"},
      {"data 63", str("K"), rep('d', 63),
       "d3d036f3937f24254de6f137ec3d6e969ceee90e830bcf458bccabd9e7f4d53c"},
      {"data 64", str("K"), rep('d', 64),
       "7ee58299fe12024b7064fa3f0a620ebd90ea9fc2c8cfea1059f05a07cd133b06"},
      {"data 65", str("K"), rep('d', 65),
       "c0c319b0319065c3b705bf5850a1818a947c155840673f72c4c121d6468432c2"},
      {"data 119", str("K"), rep('d', 119),
       "377c7201a9b4a90d5d4eb56ab7eb23745e0a6640616dec32d6d78de412b5d736"},
      {"data 120", str("K"), rep('d', 120),
       "7c4021a5b449e5bc03b2c982d1e862e42f2860bf66e1757e38886ffcfc58af71"},
      {"data 128", str("K"), rep('d', 128),
       "3b59ca275c13d249517c162257844f466d7146b9f9bd995b5343d1f9372adb1b"},
      {"rfc4231 tc1", rep(0x0b, 20), str("Hi There"),
       "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"},
      {"rfc4231 tc2", str("Jefe"), str("what do ya want for nothing?"),
       "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"},
      {"rfc4231 tc3", rep(0xaa, 20), rep(0xdd, 50),
       "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe"},
      {"rfc4231 tc4", rep(0x01, 20), rep(0xcd, 50),
       "884b6750df482000343346a4718fa9e56ac72ed76f3ff71241e5d7f964062d32"},
  };
  for (const auto& c : cases) {
    uint32_t d[8] = {0};
    hmac_sha256(c.key.data(), c.key.size(), c.data.data(), c.data.size(), d);
    check(hex(d) == c.want, std::string(c.name) + " -> " + hex(d));
  }
}

// --------------------------------------------------------------- predicate
//
// Built programmatically rather than from typed-out hex: construct a digest
// whose hex is "0e" followed by 62 decimal digits, then corrupt exactly one
// nibble and assert the exact accept/reject boundary.

static void fill_all_digit(uint32_t d[8]) {
  for (int i = 0; i < 8; ++i) d[i] = 0;
  set_nibble(d, 0, 0);
  set_nibble(d, 1, 0xe);
  for (unsigned p = 2; p < 64; ++p) set_nibble(d, p, (p - 2) % 10);
}

static void test_magic_ok() {
  std::printf("magic predicate\n");
  uint32_t base[8];
  fill_all_digit(base);
  check(hex(base) == "0e01234567890123456789012345678901234567890123456789012345678901",
        "constructed all-digit digest -> " + hex(base));

  for (unsigned digits = 0; digits <= 62; ++digits) {
    check(magic_ok(base, digits), "all-digit digest accepted at digits=" + std::to_string(digits));
  }

  // A non-digit nibble at hex position p must reject exactly when the required
  // run of digits reaches p, i.e. when digits >= p - 1.
  for (unsigned p = 2; p < 64; ++p) {
    uint32_t d[8];
    fill_all_digit(d);
    set_nibble(d, p, 0xa);  // 'a'
    for (unsigned digits = 0; digits <= 62; ++digits) {
      const bool want = digits <= p - 2;
      if (magic_ok(d, digits) != want) {
        check(false, "bad nibble at " + std::to_string(p) + " digits=" + std::to_string(digits) +
                         " got=" + std::to_string(magic_ok(d, digits) ? 1 : 0) +
                         " want=" + std::to_string(want ? 1 : 0));
      }
    }
  }
  check(true, "exhaustive single-bad-nibble sweep");

  // The "0e" prefix itself is required at every digit count, including 0.
  for (unsigned p = 0; p < 2; ++p) {
    for (unsigned v = 0; v < 16; ++v) {
      const unsigned expect_e = 0xe;
      if (p == 1 && v == expect_e) continue;
      if (p == 0 && v == 0) continue;
      uint32_t d[8];
      fill_all_digit(d);
      set_nibble(d, p, v);
      for (unsigned digits = 0; digits <= 62; ++digits) {
        check(!magic_ok(d, digits), "prefix nibble " + std::to_string(p) + " = " +
                                        std::to_string(v) + " rejected at digits=" +
                                        std::to_string(digits));
      }
    }
  }

  uint32_t zero[8] = {0};
  check(!magic_ok(zero, 0), "all-zero digest is not 0e-prefixed");
}

// ------------------------------------------------- hoisted candidate path

static void test_hoisting() {
  std::printf("hoisted candidate path vs reference HMAC\n");
  const size_t key_lens[] = {0, 1, 19, 20, 31, 32, 63, 64, 65, 131};
  const uint64_t counters[] = {0ull, 1ull, 2ull, 0xffull, 5ull, 10000000ull, 42424242ull,
                               99999999ull, 12345678ull, 98765432ull};
  int mismatches = 0;
  int compared = 0;
  int rejected = 0;
  for (size_t kl : key_lens) {
    const std::vector<uint8_t> key = rep(0x5a, kl);

    for (size_t plen = 0; plen <= 140; ++plen) {
      std::vector<uint8_t> prefix(plen);
      fill_pseudo(prefix);
      // Windowed search: the context is built over prefix || leading-12-digits.
      const std::vector<uint8_t> eff = window_prefix(prefix, kWindowNumber);

      HmacCtx ctx;
      const char* err = nullptr;
      const bool ok = hmac_ctx_build(key.data(), key.size(), eff.data(), eff.size(), &ctx, &err);
      const bool legal = (eff.size() % 64) <= 47;
      if (ok != legal) {
        check(false, "prefix_len=" + std::to_string(plen) + " key_len=" + std::to_string(kl) +
                         " accepted=" + std::to_string(ok ? 1 : 0) +
                         " legal=" + std::to_string(legal ? 1 : 0));
        continue;
      }
      if (!ok) {
        ++rejected;
        continue;
      }

      check(ctx.counter_word == (eff.size() % 64) / 4,
            "counter_word for eff_len=" + std::to_string(eff.size()));
      check(ctx.counter_lane == (eff.size() % 64) % 4,
            "counter_lane for eff_len=" + std::to_string(eff.size()));

      for (uint64_t c : counters) {
        // Reference: the full body, prefix || leading-12-digits || ascii8(c).
        std::vector<uint8_t> msg = eff;
        msg.resize(eff.size() + 8);
        put_ascii8(msg.data() + eff.size(), c);

        uint32_t want[8] = {0};
        hmac_sha256(key.data(), key.size(), msg.data(), msg.size(), want);

        uint32_t got[8] = {0};
        hmac_candidate_any(ctx, c, got);
        ++compared;
        if (hex(want) != hex(got) && mismatches++ < 5) {
          check(false, "mismatch key_len=" + std::to_string(kl) +
                           " prefix_len=" + std::to_string(plen) + " counter=" +
                           std::to_string(c) + " want=" + hex(want) + " got=" + hex(got));
        }
      }
    }
  }
  check(mismatches == 0, std::to_string(mismatches) + " mismatches");
  check(compared > 2000, "compared " + std::to_string(compared) + " candidates");
  check(rejected == 320, "rejected " + std::to_string(rejected) + " illegal prefix lengths");
  std::printf("  (%d candidates cross-checked, %d illegal prefix lengths rejected)\n", compared,
              rejected);
}

// A realistic end-to-end shape: fixed key, ASCII prefix, counter suffix.
static void test_end_to_end_shape() {
  std::printf("end-to-end search shape\n");
  const auto key = str("session-secret");
  const auto prefix = str("user=42&action=export&nonce=");
  const std::vector<uint8_t> eff = window_prefix(prefix, 0);  // window 0
  HmacCtx ctx;
  const char* err = nullptr;
  if (!hmac_ctx_build(key.data(), key.size(), eff.data(), eff.size(), &ctx, &err)) {
    check(false, std::string("ctx build failed: ") + err);
    return;
  }
  int bad = 0;
  for (uint64_t c = 0; c < 500; ++c) {
    std::vector<uint8_t> msg = eff;
    msg.resize(msg.size() + 8);
    put_ascii8(msg.data() + msg.size() - 8, c);
    uint32_t want[8] = {0};
    hmac_sha256(key.data(), key.size(), msg.data(), msg.size(), want);
    uint32_t got[8] = {0};
    hmac_candidate_any(ctx, c, got);
    if (hex(want) != hex(got)) ++bad;
  }
  check(bad == 0, "500 sequential counters, " + std::to_string(bad) + " mismatches");
}

// The published fixture: a real key, prefix, counter, and digest, taken from a
// completed search and verified with PHP's hash_hmac. It is the only externally
// fixed statement of the body format, so it pins the whole CPU pipeline: the
// 20-digit layout, the window split, the hoisted candidate path, and the
// predicate all have to agree with a value produced outside this code base.
static void test_published_fixture() {
  std::printf("published fixture\n");
  const auto key = str("hmac-sha256-magic-hash-key");
  const auto prefix = str("magic-hash-");
  const uint64_t counter = 434871031216900ull;
  const char* const want_body = "magic-hash-00000434871031216900";
  const char* const want_digest =
      "0e54716623902408569054496933889872294553517404006854343223678539";

  const std::string body = body_of(prefix, counter);
  check(body == want_body, "body_of -> " + body);

  uint32_t d[8] = {0};
  hmac_sha256(key.data(), key.size(), reinterpret_cast<const uint8_t*>(body.data()), body.size(),
              d);
  check(hex(d) == want_digest, "fixture digest -> " + hex(d));
  check(magic_ok(d, 62), "fixture satisfies the full predicate");

  // The search reaches the same body through a window context plus the local
  // counter: window 4348710 fixes "000004348710" and the kernel writes 31216900.
  const uint64_t window = counter / HMAC0E_WINDOW;
  const uint64_t local = counter % HMAC0E_WINDOW;
  check(window == 4348710ull && local == 31216900ull,
        "window split " + std::to_string(window) + "/" + std::to_string(local));
  const std::vector<uint8_t> eff = window_prefix(prefix, window);
  const std::string eff_str(eff.begin(), eff.end());
  check(eff_str == "magic-hash-000004348710", "window prefix -> " + eff_str);
  check(body == eff_str + padded_dec(local, HMAC0E_COUNTER_DIGITS),
        "window prefix plus local digits rebuild the body");

  HmacCtx ctx;
  const char* err = nullptr;
  if (!hmac_ctx_build(key.data(), key.size(), eff.data(), eff.size(), &ctx, &err)) {
    check(false, std::string("fixture ctx build failed: ") + err);
    return;
  }
  uint32_t got[8] = {0};
  hmac_candidate_any(ctx, local, got);
  check(hex(got) == want_digest, "hoisted candidate reproduces the fixture -> " + hex(got));
}

// An empty key is a legal HMAC key, and an empty effective prefix is the
// degenerate body of only the counter digits. Both hand a null pointer and a
// zero length to memcpy in the naive implementation, which is undefined even
// when nothing is copied, so both paths run here and get compared.
static void test_empty_buffers() {
  std::printf("empty key and empty prefix\n");
  const std::vector<uint8_t> none;  // data() is null
  const std::string digits = body_of(none, 7);
  check(digits == "00000000000000000007", "body with no prefix is just 20 digits");

  HmacCtx ctx;
  const char* err = nullptr;
  if (!hmac_ctx_build(none.data(), 0, none.data(), 0, &ctx, &err)) {
    check(false, std::string("empty ctx build failed: ") + (err ? err : "?"));
    return;
  }
  check(ctx.counter_word == 0 && ctx.counter_lane == 0, "empty prefix puts the slot at word 0");

  // With no prefix the message is only the varying slot: a window's 12 fixed
  // digits belong to the prefix that a real search supplies.
  int bad = 0;
  for (uint64_t c = 0; c < 16; ++c) {
    const std::string msg = padded_dec(c, HMAC0E_COUNTER_DIGITS);
    uint32_t want[8] = {0};
    hmac_sha256(none.data(), 0, reinterpret_cast<const uint8_t*>(msg.data()), msg.size(), want);
    uint32_t got[8] = {0};
    hmac_candidate_any(ctx, c, got);
    if (hex(want) != hex(got)) ++bad;
  }
  check(bad == 0, "16 empty-key candidates, " + std::to_string(bad) + " mismatches");
}

// The candidate-slice arithmetic is the only unsigned wrap-around logic in the
// tool. It is pure, so it is tested here with no CUDA toolkit.
static void test_launch_range() {
  std::printf("launch ranges\n");
  const uint64_t kWindow = HMAC0E_WINDOW;
  const uint64_t cov = 1020ull * 256 * 512;  // the default grid, 133693440
  check(cov > kWindow, "the default grid reaches past one window");

  LaunchRange r = launch_range(0, cov, 0);
  check(r.window == 0 && r.c0 == 0 && r.want == kWindow && !r.exhausted,
        "first launch stops at the window end");
  check(launch_range(0, 1000, 0).want == 1000, "a smaller grid limits the launch");
  check(launch_range(0, cov, 4096).want == 4096, "--count limits the launch");

  r = launch_range(kWindow - 1000, cov, 0);
  check(r.window == 0 && r.c0 == kWindow - 1000 && r.want == 1000 && !r.exhausted,
        "the tail of a window is not padded");
  r = launch_range(kWindow - 1000, cov, 500);
  check(r.want == 500, "--count wins over the rest of the window");
  r = launch_range(kWindow, cov, 0);
  check(r.window == 1 && r.c0 == 0 && r.want == kWindow && !r.exhausted,
        "a window boundary starts a fresh window");

  // The last window is partial: 2^64 is not a multiple of the window size.
  const uint64_t last_window = UINT64_MAX / kWindow;   // 184467440737
  const uint64_t last_start = last_window * kWindow;
  const uint64_t tail = ~last_start + 1;               // 9551616
  check(tail == 9551616, "the last window has 9551616 counters");
  r = launch_range(last_start, cov, 0);
  check(r.window == last_window && r.c0 == 0 && r.want == tail && r.exhausted,
        "the last launch is capped by the counter space, not the window");
  check(launch_range(last_start + 1000, cov, 0).want == tail - 1000,
        "the partial window still limits the range");

  r = launch_range(UINT64_MAX, cov, 0);
  check(r.want == 1 && r.exhausted, "the final counter is scanned once");
  check(r.window == last_window && r.c0 == tail - 1, "the final counter sits at the tail end");
  r = launch_range(UINT64_MAX - 999, cov, 0);
  check(r.want == 1000 && r.exhausted, "the launch that reaches 2^64 reports exhaustion");
  r = launch_range(UINT64_MAX - 999, cov, 500);
  check(r.want == 500 && !r.exhausted, "--count before the end is not exhaustion");

  // 12 digits must represent every window index, or the leading digits would
  // truncate and the kernel's slot would land in the wrong place.
  check(last_window < 1000000000000ull, "every window index fits 12 digits");
}

// The whole search is compute-only: nothing per candidate lives in memory, so
// the device footprint is a fixed hit buffer plus a counter. Pin it down so a
// future change that adds a per-candidate allocation cannot go unnoticed.
static void test_footprint() {
  std::printf("device footprint\n");
  check(sizeof(HmacCtx) <= 4096,
        "HmacCtx = " + std::to_string(sizeof(HmacCtx)) + " bytes must fit kernel params");
  check(sizeof(Hit) == 40, "Hit = " + std::to_string(sizeof(Hit)) + " bytes");
  const size_t device_bytes = size_t(1024) * sizeof(Hit) + sizeof(uint32_t);
  check(device_bytes < 64 * 1024,
        "device buffers total " + std::to_string(device_bytes) + " bytes");
}

int main() {
  test_sha256();
  test_hmac();
  test_magic_ok();
  test_hoisting();
  test_end_to_end_shape();
  test_published_fixture();
  test_empty_buffers();
  test_launch_range();
  test_footprint();
  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
}
