#ifndef CUDA_PIPELINE_CUH
#define CUDA_PIPELINE_CUH

#include <cufft.h>
#include <vector>
#include "radar_params.h"

// Host-side wrappers around each CUDA stage. main.cu owns all device memory
// and cuFFT plan lifetimes; these functions just launch kernels / cuFFT
// calls against caller-provided device pointers, so the same buffers can be
// reused frame-to-frame without reallocating.
namespace radar {
namespace gpu {

// chirp_generation.cu
// Generates the LFM reference chirp directly on the GPU, one thread per
// fast-time sample -- SAMPLES_PER_CHIRP independent, branch-free FMA-style
// computations, the canonical "embarrassingly parallel" GPU workload.
void generate_reference_chirp(cufftComplex* d_chirp, cudaStream_t stream = 0);

// matched_filter.cu
// Pulse compression: forward batched FFT (one FFT per chirp row) -> custom
// frequency-domain multiply kernel (elementwise, hand-written) -> inverse
// batched FFT. `plan_fwd_rows`/`plan_inv_rows` are NUM_CHIRPS-batch,
// SAMPLES_PER_CHIRP-point C2C plans owned by the caller.
void compute_ref_fft_conj(cufftHandle plan_single, cufftComplex* d_ref_chirp,
                           cufftComplex* d_ref_fft_conj, cudaStream_t stream = 0);
void matched_filter(cufftHandle plan_fwd_rows, cufftHandle plan_inv_rows,
                     cufftComplex* d_rows /* in-place, NUM_CHIRPS x SAMPLES_PER_CHIRP */,
                     const cufftComplex* d_ref_fft_conj, cudaStream_t stream = 0);

// range_doppler.cu
// Slow-time FFT across chirps, batched over range bins via cuFFT's strided
// advanced data layout (no transpose kernel needed -- see
// PARALLELIZATION_NOTES.md), followed by a custom fftshift kernel.
void range_doppler(cufftHandle plan_slowtime, cufftComplex* d_rd /* in-place */,
                    cudaStream_t stream = 0);

// coherent_integration.cu
// Elementwise magnitude-squared accumulation across frames (custom kernel),
// plus a shared-memory block-reduction kernel that computes the mean power
// of the integrated map (used as a sanity-check noise-floor estimate).
void accumulate_power(const cufftComplex* d_rd, float* d_integrated_power, cudaStream_t stream = 0);
float mean_power(const float* d_integrated_power, cudaStream_t stream = 0);

// peak_detection.cu
// CA-CFAR detector, one thread per range-Doppler cell.
int cfar_detect(const float* d_integrated_power, Detection* d_detections, cudaStream_t stream = 0);

} // namespace gpu
} // namespace radar

#endif // CUDA_PIPELINE_CUH
