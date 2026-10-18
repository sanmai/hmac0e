#include "hmac_kernel.cuh"

#include <cinttypes>
#include <cstdio>

#define HMAC0E_CHECK(call)                                                              \
  do {                                                                                  \
    cudaError_t err_ = (call);                                                          \
    if (err_ != cudaSuccess) {                                                          \
      std::fprintf(stderr, "cuda error at %s:%d: %s\n", __FILE__, __LINE__,             \
                   cudaGetErrorString(err_));                                           \
      return 1;                                                                         \
    }                                                                                   \
  } while (0)

// The context travels by value in the kernel parameter space.
static_assert(sizeof(HmacCtx) <= 4096, "HmacCtx must stay small enough for kernel params");

__device__ __forceinline__ void record_hit(uint64_t counter, const uint32_t d[8], Hit* hits,
                                           uint32_t max_hits, uint32_t* nhits) {
  const uint32_t slot = atomicAdd(nhits, 1u);
  if (slot >= max_hits) return;
  hits[slot].counter = counter;
HMAC0E_UNROLL
  for (int i = 0; i < 8; ++i) hits[slot].digest[i] = d[i];
}

// CW is the counter word index, baked in so blk[] keeps constant indices. A
// runtime index here would push the 16-word block into local memory and cost
// several times the throughput.
template <unsigned CW>
__global__ __launch_bounds__(256) void search_kernel(const HmacCtx ctx, uint64_t base,
                                                     uint64_t n, uint32_t per_thread,
                                                     uint32_t digits, Hit* hits,
                                                     uint32_t max_hits, uint32_t* nhits) {
  const uint64_t tid = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const uint64_t off = tid * per_thread;
  if (off >= n) return;
  // The counter space is mod 2^64, so [base, base+n) wraps naturally past
  // 0xffffffffffffffff. Count, not limit, bounds the loop: a precomputed
  // base+n would wrap to 0 and kill the tail of the search.
  const uint64_t count = min(uint64_t(per_thread), n - off);
  uint64_t c = base + off;

  uint32_t d[8];
  for (uint64_t i = 0; i < count; ++i, ++c) {
    hmac_candidate<CW>(ctx, c, d);
    if ((d[0] >> 24) != 0x0eu) continue;
    if (!magic_ok(d, digits)) continue;
    record_hit(c, d, hits, max_hits, nhits);
  }
}

#define HMAC0E_OCCUPANCY(CW)                                                        \
  do {                                                                              \
    int n = 0;                                                                      \
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, search_kernel<CW>,        \
                                                      int(threads), 0) != cudaSuccess) \
      return 1;                                                                     \
    *blocks_per_sm = uint32_t(n);                                                   \
    return 0;                                                                       \
  } while (0)

int hmac0e_blocks_per_sm(unsigned counter_word, uint32_t threads, uint32_t* blocks_per_sm) {
  switch (counter_word) {
    case 0: HMAC0E_OCCUPANCY(0);
    case 1: HMAC0E_OCCUPANCY(1);
    case 2: HMAC0E_OCCUPANCY(2);
    case 3: HMAC0E_OCCUPANCY(3);
    case 4: HMAC0E_OCCUPANCY(4);
    case 5: HMAC0E_OCCUPANCY(5);
    case 6: HMAC0E_OCCUPANCY(6);
    case 7: HMAC0E_OCCUPANCY(7);
    case 8: HMAC0E_OCCUPANCY(8);
    case 9: HMAC0E_OCCUPANCY(9);
    case 10: HMAC0E_OCCUPANCY(10);
    default: HMAC0E_OCCUPANCY(11);
  }
  return 1;
}

int hmac0e_alloc(HitBuffer* buf) {
  HMAC0E_CHECK(cudaMalloc(reinterpret_cast<void**>(&buf->hits), buf->max_hits * sizeof(Hit)));
  HMAC0E_CHECK(cudaMalloc(reinterpret_cast<void**>(&buf->nhits), sizeof(uint32_t)));
  HMAC0E_CHECK(cudaMemset(buf->nhits, 0, sizeof(uint32_t)));
  return 0;
}

void hmac0e_free(HitBuffer* buf) {
  if (buf->hits) cudaFree(buf->hits);
  if (buf->nhits) cudaFree(buf->nhits);
  buf->hits = nullptr;
  buf->nhits = nullptr;
}

#define HMAC0E_LAUNCH(CW)                                                             \
  search_kernel<CW><<<blocks, threads, 0, stream>>>(ctx, base, n, per_thread,          \
                                                    digits, buf->hits, buf->max_hits, \
                                                    buf->nhits)

int hmac0e_launch(const HmacCtx& ctx, uint64_t base, uint64_t n, unsigned digits,
                  uint32_t blocks, uint32_t threads, uint32_t per_thread, HitBuffer* buf,
                  cudaStream_t stream) {
  if (n == 0) return 0;
  if (base >= HMAC0E_WINDOW || n > HMAC0E_WINDOW - base) {
    std::fprintf(stderr,
                 "hmac0e_launch: range [%" PRIu64 ", +%" PRIu64 ") exceeds the %llu-counter window\n",
                 base, n, HMAC0E_WINDOW);
    return 1;
  }
  if (n > uint64_t(blocks) * threads * per_thread) {
    std::fprintf(stderr,
                 "hmac0e_launch: range of %" PRIu64
                 " exceeds grid coverage of %llu; raise blocks/threads/per_thread\n",
                 n, (unsigned long long)(uint64_t(blocks) * threads * per_thread));
    return 1;
  }
  HMAC0E_CHECK(cudaMemsetAsync(buf->nhits, 0, sizeof(uint32_t), stream));
  switch (ctx.counter_word) {
    case 0: HMAC0E_LAUNCH(0); break;
    case 1: HMAC0E_LAUNCH(1); break;
    case 2: HMAC0E_LAUNCH(2); break;
    case 3: HMAC0E_LAUNCH(3); break;
    case 4: HMAC0E_LAUNCH(4); break;
    case 5: HMAC0E_LAUNCH(5); break;
    case 6: HMAC0E_LAUNCH(6); break;
    case 7: HMAC0E_LAUNCH(7); break;
    case 8: HMAC0E_LAUNCH(8); break;
    case 9: HMAC0E_LAUNCH(9); break;
    case 10: HMAC0E_LAUNCH(10); break;
    default: HMAC0E_LAUNCH(11); break;
  }
  HMAC0E_CHECK(cudaGetLastError());
  HMAC0E_CHECK(cudaStreamSynchronize(stream));
  return 0;
}

int hmac0e_drain(HitBuffer* buf, Hit* out, uint32_t cap, uint32_t* found) {
  uint32_t n = 0;
  HMAC0E_CHECK(cudaMemcpy(&n, buf->nhits, sizeof(uint32_t), cudaMemcpyDeviceToHost));
  const uint32_t take = (n < cap) ? n : cap;
  if (take > 0) {
    HMAC0E_CHECK(cudaMemcpy(out, buf->hits, take * sizeof(Hit), cudaMemcpyDeviceToHost));
  }
  *found = n;  // unclamped total; the caller caps the copy
  return 0;
}
