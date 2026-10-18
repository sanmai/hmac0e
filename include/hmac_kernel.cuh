#pragma once

#include <cuda_runtime.h>

#include "hmac0e.cuh"

#define HMAC0E_MAX_HITS 1024u
#define HMAC0E_WINDOW 100000000ull

struct HitBuffer {
  Hit* hits = nullptr;
  uint32_t* nhits = nullptr;
  uint32_t max_hits = HMAC0E_MAX_HITS;
};

int hmac0e_alloc(HitBuffer* buf);
void hmac0e_free(HitBuffer* buf);

// Scan the window-local counters [base, base+n). The range must not exceed
// HMAC0E_WINDOW: each counter becomes the last 8 digits of the body. Each
// thread takes a contiguous run of per_thread counters, so a launch
// covers blocks*threads*per_thread; fails if that is less than n.
int hmac0e_launch(const HmacCtx& ctx, uint64_t base, uint64_t n, unsigned digits,
                  uint32_t blocks, uint32_t threads, uint32_t per_thread, HitBuffer* buf,
                  cudaStream_t stream);

// How many blocks of `threads` fit resident per SM for the kernel serving this
// counter word index. Threads must not exceed the kernel's launch bounds.
int hmac0e_blocks_per_sm(unsigned counter_word, uint32_t threads, uint32_t* blocks_per_sm);

// Copies at most `cap` hits into `out` and reports in *found the total number
// of hits recorded on the device (which may exceed cap, in which case the
// excess was not copied and is lost). Call with cap 0 to learn the count.
// The device counter is reset by the next launch, not here.
int hmac0e_drain(HitBuffer* buf, Hit* out, uint32_t cap, uint32_t* found);
