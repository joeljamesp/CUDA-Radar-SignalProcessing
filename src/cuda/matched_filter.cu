#include "cuda_pipeline.cuh"
#include "cuda_check.cuh"

// Pulse compression via FFT-based matched filtering.
//
// The FFTs themselves (forward per-row, inverse per-row) are delegated to
// cuFFT's batched C2C mode -- NUM_CHIRPS independent SAMPLES_PER_CHIRP-point
// transforms, which is exactly the workload cuFFT's batching is designed
// for and not worth hand-rolling.
//
// The frequency-domain multiply -- out[m][k] = fft_row[m][k] * conj(refFFT[k])
// -- is written as a custom kernel because it's the part of this stage that
// actually demonstrates kernel design: every (chirp, range-bin) cell is
// independent, the reference spectrum is broadcast (read-only, reused
// NUM_CHIRPS times so it benefits from the L2/texture cache), and the 1/N
// inverse-FFT normalization is folded in here so the following cuFFT
// inverse call can stay unnormalized.

namespace radar {
namespace gpu {

__global__ void freq_domain_multiply_kernel(cufftComplex* rows,
                                             const cufftComplex* ref_fft_conj,
                                             int num_chirps, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_chirps * n;
    if (idx >= total) return;

    int k = idx % n; // range-bin (frequency) index within the row
    cufftComplex a = rows[idx];
    cufftComplex b = ref_fft_conj[k];

    const float inv_n = 1.0f / static_cast<float>(n);
    cufftComplex result;
    result.x = (a.x * b.x - a.y * b.y) * inv_n;
    result.y = (a.x * b.y + a.y * b.x) * inv_n;
    rows[idx] = result;
}

__global__ void conjugate_kernel(const cufftComplex* in, cufftComplex* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    out[idx].x = in[idx].x;
    out[idx].y = -in[idx].y;
}

void compute_ref_fft_conj(cufftHandle plan_single, cufftComplex* d_ref_chirp,
                           cufftComplex* d_ref_fft_conj, cudaStream_t stream) {
    const int n = SAMPLES_PER_CHIRP;
    CUFFT_CHECK(cufftSetStream(plan_single, stream));
    // FFT the reference chirp in place into a scratch copy, then conjugate.
    CUFFT_CHECK(cufftExecC2C(plan_single, d_ref_chirp, d_ref_fft_conj, CUFFT_FORWARD));
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    conjugate_kernel<<<blocks, threads, 0, stream>>>(d_ref_fft_conj, d_ref_fft_conj, n);
    CUDA_CHECK(cudaGetLastError());
}

void matched_filter(cufftHandle plan_fwd_rows, cufftHandle plan_inv_rows,
                     cufftComplex* d_rows, const cufftComplex* d_ref_fft_conj,
                     cudaStream_t stream) {
    const int n = SAMPLES_PER_CHIRP;
    const int total = NUM_CHIRPS * n;

    CUFFT_CHECK(cufftSetStream(plan_fwd_rows, stream));
    CUFFT_CHECK(cufftSetStream(plan_inv_rows, stream));

    CUFFT_CHECK(cufftExecC2C(plan_fwd_rows, d_rows, d_rows, CUFFT_FORWARD));

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    freq_domain_multiply_kernel<<<blocks, threads, 0, stream>>>(
        d_rows, d_ref_fft_conj, NUM_CHIRPS, n);
    CUDA_CHECK(cudaGetLastError());

    CUFFT_CHECK(cufftExecC2C(plan_inv_rows, d_rows, d_rows, CUFFT_INVERSE));
}

} // namespace gpu
} // namespace radar
