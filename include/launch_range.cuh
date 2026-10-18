// Host-only slicing of the 2^64 counter space into launches, free of CUDA so
// the CPU suite can test it.
#pragma once

#include <algorithm>
#include <cstdint>

#include "hmac0e.cuh"

// The slice of the candidate space given to one launch.
struct LaunchRange {
  uint64_t window;   // window containing the first counter
  uint64_t c0;       // window-local first counter, the kernel's `base`
  uint64_t want;     // counters to scan, at least 1
  bool exhausted;    // this launch ends the 2^64 counter space
};

// The next launch for a search at global counter `next`: at most the rest of
// the window, at most `coverage` (what the grid reaches, at least 1), and at
// most `count_left` remaining candidates (0 = no limit).
inline LaunchRange launch_range(uint64_t next, uint64_t coverage, uint64_t count_left) {
  LaunchRange r{};
  r.window = next / HMAC0E_WINDOW;
  r.c0 = next % HMAC0E_WINDOW;
  // Counters left in the 2^64 space. 2^64 itself does not fit in a uint64, so
  // next == 0 (the whole space) is represented as UINT64_MAX; the one missing
  // counter can never be skipped, since a launch covers at most one window.
  const uint64_t space = (next == 0) ? UINT64_MAX : (~next + 1);
  r.want = std::min({uint64_t(HMAC0E_WINDOW) - r.c0, space, coverage});
  if (count_left != 0) r.want = std::min(r.want, count_left);
  // Wrapped past the end of the space. Such a `next` must never reach the
  // checkpoint: resume would read 0 as "start from the beginning".
  r.exhausted = next + r.want == 0;
  return r;
}
