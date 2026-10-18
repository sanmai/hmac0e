// HMAC-SHA256 with the search-specific key/prefix work hoisted out of the
// candidate loop, plus the magic-hash predicate.
//
// Search space, under one fixed key:
//
//   body = prefix || D,  D = the counter as 20 zero-padded decimal digits
//
// (the full 2^64 counter range). Within a window of 10^8 counters the leading
// 12 digits of D are constant, so only the last 8 digits (the counter slot,
// written as ASCII) vary. That keeps the body printable - essential, because
// the hit is what gets pasted into the test.
//
//   inner = SHA256(ipad || body)
//   outer = SHA256(opad || inner)
//
// ipad and opad are each exactly one 64-byte block, and every byte before the
// 8-digit slot is fixed within a window, so all complete blocks preceding the
// slot can be compressed once on the host. That leaves exactly two compression
// calls per candidate: the inner tail block, and the outer tail block. That is
// the floor for this construction.
//
// Requirement: the counter slot (8 bytes) plus SHA-256 padding (1 + 8 bytes)
// must fit in the final inner block, i.e. ((prefix_len + 12) % 64) <= 47.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "sha256.cuh"

// Counter word index inside the final inner block, 0..11. The kernel is
// instantiated once per value so the block array keeps constant indices and
// never spills to local memory.
#define HMAC0E_MAX_COUNTER_WORD 11

struct HmacCtx {
  uint32_t inner[8];        // chained state after all hoisted inner blocks
  uint32_t outer[8];        // chained state after the opad block
  uint32_t tail[16];        // final inner block template, counter slot zeroed
  uint32_t outer_tail[8];   // words 8..15 of the final outer block
  uint32_t counter_word;    // 0..11
  uint32_t counter_lane;    // 0..3, byte offset of the counter inside that word
  uint32_t inner_blocks;    // hoisted inner block count, informational
};

struct Hit {
  uint64_t counter;
  uint32_t digest[8];
};

// One HMAC-SHA256 candidate. CW is the compile-time counter word index.
// `counter` is window-local, 0 <= counter < 10^8: it becomes the last 8
// digits of the body's 20-digit decimal tail; the leading 12 digits are fixed
// per window and already baked into the hoisted prefix.
// Used by both the CUDA kernel and the CPU tests, so the tests exercise the
// exact code the GPU runs.
template <unsigned CW>
HMAC0E_HD inline void hmac_candidate(const HmacCtx& x, uint64_t counter, uint32_t out[8]) {
  uint32_t blk[16];
HMAC0E_UNROLL
  for (int i = 0; i < 16; ++i) blk[i] = x.tail[i];

  // The 8 ASCII digits of the window-local counter, most significant first,
  // as a big-endian byte value. All divisions are by constants, so this
  // compiles to multiply/shift, not divide.
  uint32_t t = uint32_t(counter);
  uint64_t A = uint64_t('0' + t / 10000000u);
  t %= 10000000u;
  A = (A << 8) | uint64_t('0' + t / 1000000u);
  t %= 1000000u;
  A = (A << 8) | uint64_t('0' + t / 100000u);
  t %= 100000u;
  A = (A << 8) | uint64_t('0' + t / 10000u);
  t %= 10000u;
  A = (A << 8) | uint64_t('0' + t / 1000u);
  t %= 1000u;
  A = (A << 8) | uint64_t('0' + t / 100u);
  t %= 100u;
  A = (A << 8) | uint64_t('0' + t / 10u);
  t %= 10u;
  A = (A << 8) | uint64_t('0' + t);

  // Place the 8 big-endian bytes starting at byte (4*CW + lane) of the block:
  // the value shifted left by `lane` bytes across three words. With lane == 0
  // the third word contributes zero, so it is a no-op.
  uint32_t v0 = uint32_t(A >> 32);
  uint32_t v1 = uint32_t(A);
  uint32_t v2 = 0u;
  if (x.counter_lane != 0) {
    const unsigned s = x.counter_lane * 8u;
    v2 = (v2 >> s) | (v1 << (32u - s));
    v1 = (v1 >> s) | (v0 << (32u - s));
    v0 >>= s;
  }
  blk[CW + 0] |= v0;
  blk[CW + 1] |= v1;
  blk[CW + 2] |= v2;

  // compress() folds the Merkle-Damgard feed-forward into its input state, so
  // feeding it a copy of the hoisted chained state yields the finished digest.
HMAC0E_UNROLL
  for (int i = 0; i < 8; ++i) out[i] = x.inner[i];
  sha256::compress(out, blk);

  uint32_t blk2[16];
HMAC0E_UNROLL
  for (int i = 0; i < 8; ++i) blk2[i] = out[i];
HMAC0E_UNROLL
  for (int i = 0; i < 8; ++i) blk2[8 + i] = x.outer_tail[i];

HMAC0E_UNROLL
  for (int i = 0; i < 8; ++i) out[i] = x.outer[i];
  sha256::compress(out, blk2);
}

// Runtime dispatch over the counter word index. Host side only; the kernel
// dispatches at launch instead.
HMAC0E_HD inline void hmac_candidate_any(const HmacCtx& x, uint64_t counter, uint32_t out[8]) {
  switch (x.counter_word) {
    case 0: hmac_candidate<0>(x, counter, out); break;
    case 1: hmac_candidate<1>(x, counter, out); break;
    case 2: hmac_candidate<2>(x, counter, out); break;
    case 3: hmac_candidate<3>(x, counter, out); break;
    case 4: hmac_candidate<4>(x, counter, out); break;
    case 5: hmac_candidate<5>(x, counter, out); break;
    case 6: hmac_candidate<6>(x, counter, out); break;
    case 7: hmac_candidate<7>(x, counter, out); break;
    case 8: hmac_candidate<8>(x, counter, out); break;
    case 9: hmac_candidate<9>(x, counter, out); break;
    case 10: hmac_candidate<10>(x, counter, out); break;
    default: hmac_candidate<11>(x, counter, out); break;
  }
}

// True when the digest hex starts with "0e" followed by `digits` decimal
// digits. digits == 62 is the full PHP magic-hash shape ^0e[0-9]{62}$.
HMAC0E_HD inline bool magic_ok(const uint32_t d[8], unsigned digits) {
  if ((d[0] >> 24) != 0x0eu) return false;  // kills 255/256 in one shift
  unsigned byte = 1;
  unsigned left = digits;
  while (left >= 2) {
    const uint32_t b = (d[byte >> 2] >> (24 - 8 * (byte & 3))) & 0xffu;
    if ((b >> 4) > 9u || (b & 15u) > 9u) return false;
    ++byte;
    left -= 2;
  }
  if (left == 1) {
    const uint32_t b = (d[byte >> 2] >> (24 - 8 * (byte & 3))) & 0xf0u;
    if (b > 0x90u) return false;
  }
  return true;
}

inline void digest_to_hex(const uint32_t d[8], char out[65]) {
  static const char* kHex = "0123456789abcdef";
  for (int i = 0; i < 8; ++i) {
    for (int n = 0; n < 8; ++n) {
      out[i * 8 + n] = kHex[(d[i] >> (28 - 4 * n)) & 0xf];
    }
  }
  out[64] = 0;
}

// Straightforward HMAC-SHA256, written for clarity rather than speed. It shares
// only the SHA-256 core with the hoisted path, so it serves as the oracle in
// tests and for verifying hits.
inline void hmac_sha256(const uint8_t* key, size_t keylen, const uint8_t* msg, size_t msglen,
                        uint32_t out[8]) {
  uint8_t kb[64];
  std::memset(kb, 0, sizeof(kb));
  if (keylen > 64) {
    sha256::hash_bytes(key, keylen, out);
    for (int i = 0; i < 8; ++i) sha256::put_be32(kb + i * 4, out[i]);
  } else {
    std::memcpy(kb, key, keylen);
  }

  uint8_t* inner_in = new uint8_t[64 + msglen];
  for (int i = 0; i < 64; ++i) inner_in[i] = uint8_t(kb[i] ^ 0x36);
  std::memcpy(inner_in + 64, msg, msglen);
  uint32_t inner[8];
  sha256::hash_bytes(inner_in, 64 + msglen, inner);
  delete[] inner_in;

  uint8_t outer_in[96];
  for (int i = 0; i < 64; ++i) outer_in[i] = uint8_t(kb[i] ^ 0x5c);
  for (int i = 0; i < 8; ++i) sha256::put_be32(outer_in + 64 + i * 4, inner[i]);
  sha256::hash_bytes(outer_in, 96, out);
}

// Precompute everything that does not depend on the counter.
inline bool hmac_ctx_build(const uint8_t* key, size_t keylen, const uint8_t* prefix,
                           size_t plen, HmacCtx* ctx, const char** err) {
  *err = nullptr;
  const size_t lane = plen % 64;
  if (lane > 47) {
    *err = "prefix length mod 64 must be <= 47 so the 8-byte counter slot "
           "and SHA-256 padding share one final block (callers of the windowed"
           " search pass prefix || leading-12-digits)";
    return false;
  }

  uint8_t kb[64];
  std::memset(kb, 0, sizeof(kb));
  if (keylen > 64) {
    uint32_t h[8];
    sha256::hash_bytes(key, keylen, h);
    for (int i = 0; i < 8; ++i) sha256::put_be32(kb + i * 4, h[i]);
  } else {
    std::memcpy(kb, key, keylen);
  }

  uint8_t block[64];
  for (int i = 0; i < 64; ++i) block[i] = uint8_t(kb[i] ^ 0x36);

  uint32_t st[8];
  sha256::init(st);
  uint32_t words[16];
  for (int i = 0; i < 16; ++i) words[i] = sha256::be32(block + i * 4);
  sha256::compress(st, words);
  ctx->inner_blocks = 1;

  // Any further complete 64-byte blocks come straight from the prefix.
  for (size_t b = 0; b + 64 <= plen; b += 64) {
    for (int i = 0; i < 16; ++i) words[i] = sha256::be32(prefix + b + i * 4);
    sha256::compress(st, words);
    ctx->inner_blocks += 1;
  }
  std::memcpy(ctx->inner, st, sizeof(st));

  // Final inner block: prefix tail, zeroed counter slot, 0x80, bit length.
  const size_t msglen = 64 + plen + 8;  // ipad || prefix || counter
  std::memset(block, 0, sizeof(block));
  std::memcpy(block, prefix + (plen - lane), lane);
  block[lane + 8] = 0x80;
  const uint64_t bits = uint64_t(msglen) * 8;
  sha256::put_be32(block + 56, uint32_t(bits >> 32));
  sha256::put_be32(block + 60, uint32_t(bits));
  for (int i = 0; i < 16; ++i) ctx->tail[i] = sha256::be32(block + i * 4);
  ctx->counter_word = uint32_t(lane / 4);
  ctx->counter_lane = uint32_t(lane % 4);

  // Outer: opad is one hoisted block, then the 32-byte inner digest plus
  // padding fills the rest of the final block.
  for (int i = 0; i < 64; ++i) block[i] = uint8_t(kb[i] ^ 0x5c);
  sha256::init(st);
  for (int i = 0; i < 16; ++i) words[i] = sha256::be32(block + i * 4);
  sha256::compress(st, words);
  std::memcpy(ctx->outer, st, sizeof(st));

  std::memset(block, 0, sizeof(block));
  block[32] = 0x80;
  sha256::put_be32(block + 56, 0);
  sha256::put_be32(block + 60, uint32_t(96 * 8));
  for (int i = 0; i < 8; ++i) ctx->outer_tail[i] = sha256::be32(block + 32 + i * 4);

  return true;
}
