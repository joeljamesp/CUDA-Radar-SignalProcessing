# GPU-Accelerated Radar Signal Processing (CUDA)

CUDA implementation of a monostatic pulse-Doppler radar range-Doppler
processing chain: LFM chirp generation, FFT-based matched filtering (pulse
compression), 2D range-Doppler map formation, non-coherent integration
across bursts, and CA-CFAR peak detection.

## Origin

This started as MATLAB. I'm a research intern at IIT Jodhpur building
simulations of a monostatic pulse-Doppler radar — LFM/OFDM waveforms,
2D-FFT range-Doppler processing, RMSE analysis across SNR/RCS/bandwidth. My
resume is strong on hardware (VLSI/RTL, FPGA acceleration, DDR4
verification) and on that DSP work, but had zero GPU/CUDA experience
anywhere on it. Range-Doppler processing — batched 2D FFTs across many
chirps, matched filtering, per-cell CFAR — is naturally data-parallel and a
legitimate real-world GPU workload (this is genuinely how modern
radar/SAR systems hit real-time throughput), so this project reimplements
the compute-heavy stages of that same pipeline as actual CUDA kernels,
validated numerically against a CPU reference, rather than bolting on an
unrelated toy CUDA example.

The MATLAB source itself lives in the IIT Jodhpur research codebase, not on
this machine, so this isn't a line-for-line port — it's the same pipeline
structure (LFM waveform → matched filter → 2D-FFT range-Doppler → detection)
rebuilt from the same domain knowledge, with a self-consistent parameter set
(see `include/radar_params.h`) chosen to be physically realistic rather than
copied from a file I didn't have on hand.

## Pipeline

```
LFM chirp gen  ->  Matched filter (pulse compression)  ->  Range-Doppler map (slow-time FFT)
                                                                    |
                                                                    v
                        CA-CFAR peak detection   <-   Non-coherent integration across frames
```

Each stage has both a CPU reference implementation (`src/cpu_reference/`)
and a CUDA implementation (`src/cuda/`); `src/main.cu` runs both against
identical synthetic input, times both, and reports the actual numerical
difference between them.

| Stage | File | GPU strategy |
|---|---|---|
| Chirp generation | `chirp_generation.cu` | One thread per fast-time sample, fully independent |
| Matched filter | `matched_filter.cu` | Batched cuFFT (per-chirp FFT/IFFT) + custom kernel for the frequency-domain multiply |
| Range-Doppler map | `range_doppler.cu` | Batched cuFFT across chirps via strided `cufftPlanMany` (no transpose needed) + custom fftshift kernel |
| Integration | `coherent_integration.cu` | Custom elementwise accumulate kernel + custom shared-memory reduction kernel |
| Peak detection | `peak_detection.cu` | Custom CA-CFAR kernel, one thread per cell, data-dependent branching |

See [`PARALLELIZATION_NOTES.md`](PARALLELIZATION_NOTES.md) for the reasoning
behind each of those choices, not just the choices themselves.

## Scenario

Two synthetic targets, X-band (10 GHz carrier), 50 MHz LFM sweep, 100 MHz
sample rate, 1024 samples/chirp, 128 chirps/CPI, 5 kHz PRF, 4 independent
frames non-coherently integrated. See `include/radar_params.h` for the full
parameter set and the derivation of range resolution / max unambiguous
velocity from those numbers.

## Build

### CMake (primary)

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
./radar_cuda        # or radar_cuda.exe on Windows
```

Requires the CUDA Toolkit (nvcc + cuFFT) and CMake 3.18+. Compute
capability target defaults to `60;70;75;80;86;89;90;120` (Pascal through
Blackwell) — override with `-DCMAKE_CUDA_ARCHITECTURES=<arch>` for a
narrower/faster build.

### Makefile (alternative)

```bash
make               # override ARCHES=... if needed
./radar_cuda
```

## Hardware / results

Built and run on an actual GPU — output below is copy-pasted from a real
run on this machine (`NVIDIA GeForce RTX 5060 Laptop GPU`, SM 12.0 /
Blackwell, 26 SMs, CUDA Toolkit 13.4), not fabricated:

```
=== GPU-Accelerated Radar Signal Processing ===
Carrier: 10.0 GHz | Bandwidth: 50.0 MHz | Sample rate: 100.0 MHz
Chirps/CPI: 128 | Samples/chirp: 1024 | Frames: 4 | PRF: 5.0 kHz
Range resolution: 3.00 m | Range bin spacing: 1.499 m | Max unambig. velocity: 37.5 m/s

GPU: NVIDIA GeForce RTX 5060 Laptop GPU (SM 12.0, 26 SMs)

=== Correctness: GPU vs CPU integrated range-Doppler map ===
Max abs error: 4.915200e+05 | RMSE: 1.573922e+03
CPU detections: 6 | GPU detections: 6

=== Detections (GPU/CFAR) ===
  range=897.88 m  velocity=-19.91 m/s  power=3405774080.000  (matches known target)
  range=899.38 m  velocity=-19.91 m/s  power=8392272896.000  (matches known target)
  range=900.88 m  velocity=-19.91 m/s  power=3391989248.000  (matches known target)
  range=448.19 m  velocity=+15.22 m/s  power=27821074432.000  (matches known target)
  range=449.69 m  velocity=+15.22 m/s  power=68716847104.000  (matches known target)
  range=451.19 m  velocity=+15.22 m/s  power=27890450432.000  (matches known target)

=== Ground truth targets ===
  range=450.00 m  velocity=+15.22 m/s  amplitude=1.00
  range=900.00 m  velocity=-19.91 m/s  amplitude=0.35

=== Timing (measured, this run, this machine) ===
CPU pipeline (wall clock):            20.216 ms
GPU pipeline (CUDA events, compute):   5.482 ms
GPU pipeline (CUDA events, incl setup): 9.838 ms
Speedup (compute-only):  3.69x
Speedup (incl. setup):   2.05x
```

Both pipelines find both targets, at the correct range and velocity, with
the expected 3-bin-wide matched-filter mainlobe on the range axis. GPU vs
CPU max absolute error is ~5e5 against peak power values of ~7e10 —
relative error ~7e-6, i.e. float32 rounding, not a real discrepancy.

Compute-only GPU speedup at these problem sizes (1024 samples x 128 chirps
x 4 frames — deliberately modest so the whole thing runs in milliseconds on
a laptop GPU) is ~3.7x; that ratio should grow with problem size (more
chirps, more range bins, more frames) since kernel-launch and cuFFT-plan
overhead is roughly fixed while the actual FFT/kernel work scales up. The
"incl. setup" number folds in cuFFT plan creation and all host↔device
transfers, which dominate at this small a problem size — a longer-running,
streams-overlapped version would recover most of that gap (see
`PARALLELIZATION_NOTES.md`).

## Known simplifications

- Range delay is applied as an integer-sample circular shift within a
  single chirp's fast-time record (no dechirp/stretch processing, no
  fractional-delay interpolation), so range is bounded by the chirp's
  record length (`MAX_RANGE_WINDOW` in `radar_params.h`, ~1536 m for the
  default parameters) and matched filtering is a circular (not linear)
  correlation — acceptable because target delays are constructed to stay
  well inside the window, but a boundary effect worth knowing about if you
  extend the target list.
- Range walk within a single 128-chirp CPI is ignored (sub-bin at the
  chosen velocities over a 25.6 ms burst) — targets are treated as
  stationary-range within a frame.
- Two hardcoded targets and four frames; no CLI for scenario parameters.
  Everything needed to add more is in `include/radar_params.h` and
  `src/scenario.cpp`.
