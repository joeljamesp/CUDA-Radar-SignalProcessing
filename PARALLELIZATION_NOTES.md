# Parallelization design notes

This is the "why", not the "what" — the source files already say what each
kernel does. This is the reasoning behind the choices, and what I'd change
if this went further than a portfolio project.

## Stage-by-stage strategy

**Chirp generation.** Each fast-time sample's phase, `pi * chirp_rate *
(n/Fs)^2`, depends only on its own index `n`. No inter-sample dependency, no
communication, no synchronization — one thread per sample, one block. This
is the baseline case GPUs exist for, and it's not interesting on its own;
it's here because the received-signal model in `scenario.cpp` needs the same
waveform and the pipeline needs to generate it on-device too, not just host
a copy.

**Matched filtering (pulse compression).** FFT-based matched filtering is
`IFFT(FFT(row) * conj(FFT(ref)))` per chirp row. The FFTs are handed to
cuFFT's batched C2C mode (`NUM_CHIRPS` independent `SAMPLES_PER_CHIRP`-point
transforms in one call) — cuFFT's batching exists exactly for "many
same-size, independent transforms," and there is no reason to write a
custom radix kernel that will be slower and buggier than NVIDIA's. What *is*
hand-written is the frequency-domain multiply: every `(chirp, range-bin)`
cell is an independent complex multiply against a broadcast, read-only
reference spectrum. That's the part that actually demonstrates
kernel-writing judgment (memory access pattern, folding the 1/N
normalization into the multiply so the following inverse FFT can stay
unnormalized, avoiding an extra pass over the data) — so it's a real kernel,
not a library call.

**Range-Doppler map (slow-time FFT).** This is the one worth explaining in
detail, because the naive approach (transpose the matrix, then batch-FFT the
now-contiguous rows) is unnecessary here. The compressed data is row-major
`[chirp][range]`: for a fixed range bin `k`, the `NUM_CHIRPS` samples across
chirps live at stride `SAMPLES_PER_CHIRP`. cuFFT's advanced data layout
(`cufftPlanMany` with `istride`/`idist`) expresses that stride directly —
`istride = SAMPLES_PER_CHIRP`, `idist = 1` — so cuFFT batches all
`SAMPLES_PER_CHIRP` of these strided per-range-bin FFTs in a single plan,
no transpose kernel, no extra global memory traffic. Each range bin's
Doppler FFT is completely independent of every other range bin, which is
precisely the "many independent, moderate-size transforms" shape cuFFT
batching targets. The one custom piece is `fftshift`: cuFFT places
zero-Doppler at index 0 with negative frequencies wrapped to the back half;
a small elementwise kernel swaps the two halves so the map reads naturally
(zero-Doppler centered) for CFAR and for a human looking at it.

**Coherent/non-coherent integration.** Two purpose-built kernels, since
there's no library primitive that fits either job:
- `accumulate_power_kernel` — `integrated_power[i] += |rd[i]|^2`,
  elementwise across frames. Embarrassingly parallel; each cell is touched
  by exactly one thread per launch, so no atomics or synchronization needed
  even though the accumulation spans multiple kernel launches (one per
  frame) into the same buffer.
- `mean_power_reduction_kernel` — a classic shared-memory tree reduction
  computing the map's mean power (reported as a noise-floor sanity check
  alongside CFAR output). Each thread first folds several elements via a
  grid-stride loop, then a per-block shared-memory tree reduction collapses
  that block's partial sums to one value, and only one `atomicAdd` per
  block (not per thread) folds it into the global total — this is the
  textbook pattern for keeping atomic contention negligible while still
  producing a single scalar from a few hundred thousand elements.

**Peak detection (CA-CFAR).** One thread per range-Doppler cell. This is
the kernel with genuine data-dependent branching: cells too close to the
range-axis edges to have a full training window return early (a divergence
band only ~`training + guard` cells wide at each edge — small relative to
1024 range bins, so the warp-divergence cost is minor), and every surviving
thread does an independent averaging-window sum plus a threshold compare.
Detections are appended to a fixed-capacity output array via `atomicAdd` on
a single counter — the standard "unknown output size" GPU pattern, and far
cheaper here than a stream-compaction pass since true detections are a
tiny fraction of the cells (order of tens out of `NUM_CHIRPS *
SAMPLES_PER_CHIRP`, by construction).

## Memory layout

Range-Doppler data is stored row-major, `[chirp][range]`, range contiguous.
This makes:
- The matched-filter frequency-domain multiply fully coalesced (consecutive
  threads read/write consecutive range bins within a chirp row).
- The slow-time FFT's strided cuFFT access pattern well-defined and
  matched to the layout without a transpose.
- The CFAR kernel's training-window reads (`row[k - guard - d]` /
  `row[k + guard + d]`) coalesced across threads processing adjacent range
  bins in the same Doppler row, since a warp's threads land on nearby `k`
  values reading nearby offsets.

## What I'd do next if this went further

- **CUDA streams** to overlap the `NUM_FRAMES` per-frame pipelines —
  currently each frame's matched-filter → range-Doppler → accumulate chain
  runs to completion before the next frame's H2D copy starts. With four
  frames and independent-per-frame processing (accumulation into the shared
  power buffer is the only cross-frame dependency, and it's associative),
  this is a natural multi-stream overlap: copy frame `f+1` while frame `f`
  is still computing.
- **Shared-memory tiling** in the frequency-domain-multiply and CFAR
  kernels — the reference spectrum in matched filtering and the local
  training window in CFAR are both re-read by nearby threads; a per-block
  shared-memory cache of the reference spectrum tile (or the CFAR row
  window) would cut redundant global loads, though on modern GPUs the L2
  cache already absorbs a lot of this for these problem sizes.
- **Half-precision (FP16/TF32) matched filtering** for a further throughput
  push, since the matched filter's dynamic range is tolerant of reduced
  precision compared to the CFAR threshold comparison, which should stay
  FP32.
- **Persistent kernels or CUDA graphs** to cut per-frame launch overhead
  once the pipeline is bottlenecked by kernel launch latency rather than
  compute, which is plausible at these problem sizes (1024 x 128 per frame)
  on current-generation GPUs.
