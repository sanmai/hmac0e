// SHA-256 core, shared verbatim by host reference code and CUDA device code.
//
// All state and message blocks are held as big-endian 32-bit words. Because
// SHA-256 is defined over big-endian words, keeping everything in that domain
// means the 32-byte output of the inner HMAC hash can be fed straight into the
// outer block with no byte shuffling at all.
#pragma once

#include <cstddef>
#include <cstdint>

#ifdef __CUDACC__
#define HMAC0E_HD __host__ __device__
#else
#define HMAC0E_HD
#endif

// Unroll hint for device code only; the host compiler warns on this pragma.
#ifdef __CUDA_ARCH__
#define HMAC0E_UNROLL _Pragma("unroll")
#else
#define HMAC0E_UNROLL
#endif

// Round constants, instantiated once per compilation pass. Device code reads them
// from constant memory at compile-time indices, so they fold to immediates.
#define HMAC0E_ROUND_K                    \
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, \
    0x923f82a4u, 0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, \
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, \
    0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, \
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, \
    0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, \
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu, \
    0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, \
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, \
    0x5b9cca4fu, 0x682e6ff3u, 0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, \
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u, 

namespace sha256 {

#ifdef __CUDA_ARCH__
__device__ __constant__ const uint32_t kRoundK[64] = {HMAC0E_ROUND_K};
#else
inline constexpr uint32_t kRoundK[64] = {HMAC0E_ROUND_K};
#endif

HMAC0E_HD inline uint32_t ror32(uint32_t x, unsigned n) {
#ifdef __CUDA_ARCH__
  return __funnelshift_r(x, x, n);
#else
  return (x >> n) | (x << ((32u - n) & 31u));
#endif
}

HMAC0E_HD inline uint32_t be32(const uint8_t* p) {
  return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) | (uint32_t(p[2]) << 8) |
         uint32_t(p[3]);
}

HMAC0E_HD inline void put_be32(uint8_t* p, uint32_t v) {
  p[0] = uint8_t(v >> 24);
  p[1] = uint8_t(v >> 16);
  p[2] = uint8_t(v >> 8);
  p[3] = uint8_t(v);
}

HMAC0E_HD inline void init(uint32_t h[8]) {
  h[0] = 0x6a09e667u;
  h[1] = 0xbb67ae85u;
  h[2] = 0x3c6ef372u;
  h[3] = 0xa54ff53au;
  h[4] = 0x510e527fu;
  h[5] = 0x9b05688cu;
  h[6] = 0x1f83d9abu;
  h[7] = 0x5be0cd19u;
}

// One SHA-256 compression: h becomes h + F(h, blk). Message schedule uses the
// 16-register rolling form so the whole compression stays in registers.
HMAC0E_HD inline void compress(uint32_t h[8], const uint32_t blk[16]) {
  uint32_t w[16];
HMAC0E_UNROLL
  for (int i = 0; i < 16; ++i) w[i] = blk[i];

  uint32_t a = h[0], b = h[1], c = h[2], d = h[3];
  uint32_t e = h[4], f = h[5], g = h[6], hh = h[7];

HMAC0E_UNROLL
  for (int i = 0; i < 64; ++i) {
    const int j = i & 15;
    const uint32_t s1 = ror32(e, 6) ^ ror32(e, 11) ^ ror32(e, 25);
    const uint32_t ch = (e & f) ^ (~e & g);
    const uint32_t t1 = hh + s1 + ch + kRoundK[i] + w[j];
    const uint32_t s0 = ror32(a, 2) ^ ror32(a, 13) ^ ror32(a, 22);
    const uint32_t mj = (a & b) ^ (a & c) ^ (b & c);
    const uint32_t t2 = s0 + mj;
    hh = g;
    g = f;
    f = e;
    e = d + t1;
    d = c;
    c = b;
    b = a;
    a = t1 + t2;

    // Slot (j+d)&15 still holds W[i+d] for d in 1..15, so W[i+16] can be
    // folded into the slot we just consumed. Rounds 48..63 need no new words.
    if (i < 48) {
      const uint32_t w1 = w[(j + 1) & 15];
      const uint32_t w14 = w[(j + 14) & 15];
      const uint32_t w9 = w[(j + 9) & 15];
      const uint32_t s0w = ror32(w1, 7) ^ ror32(w1, 18) ^ (w1 >> 3);
      const uint32_t s1w = ror32(w14, 17) ^ ror32(w14, 19) ^ (w14 >> 10);
      w[j] += s1w + w9 + s0w;
    }
  }

  h[0] += a;
  h[1] += b;
  h[2] += c;
  h[3] += d;
  h[4] += e;
  h[5] += f;
  h[6] += g;
  h[7] += hh;
}

// Plain SHA-256 over a byte buffer. Host reference path only.
HMAC0E_HD inline void hash_bytes(const uint8_t* msg, size_t len, uint32_t out[8]) {
  init(out);
  const size_t nfull = len / 64;
  for (size_t b = 0; b < nfull; ++b) {
    uint32_t blk[16];
HMAC0E_UNROLL
    for (int i = 0; i < 16; ++i) blk[i] = be32(msg + b * 64 + size_t(i) * 4);
    compress(out, blk);
  }

  uint8_t tail[128];
  for (int i = 0; i < 128; ++i) tail[i] = 0;
  const size_t rem = len % 64;
  const uint8_t* rest = msg + nfull * 64;
  for (size_t i = 0; i < rem; ++i) tail[i] = rest[i];
  const size_t total = (rem < 56) ? 64 : 128;
  tail[rem] = 0x80;
  const uint64_t bits = uint64_t(len) * 8;
  put_be32(tail + total - 8, uint32_t(bits >> 32));
  put_be32(tail + total - 4, uint32_t(bits));

  for (size_t b = 0; b < total / 64; ++b) {
    uint32_t blk[16];
HMAC0E_UNROLL
    for (int i = 0; i < 16; ++i) blk[i] = be32(tail + b * 64 + size_t(i) * 4);
    compress(out, blk);
  }
}

}  // namespace sha256
