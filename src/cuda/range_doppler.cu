#include "cuda_pipeline.cuh"
#include "cuda_check.cuh"

// 2D range-Doppler map, second axis: slow-time FFT across the NUM_CHIRPS
// pulses of the burst, one independent FFT per range bin.
//
// Why batched cuFFT is the right call here (see PARALLELIZATION_NOTES.md
// for the full writeup): the data is stored row-major as [chirp][range], so
// for a fixed range bin k, the NUM_CHIRPS samples across chirps sit at
// stride SAMPLES_PER_CHIRP. cuFFT's advanced layout (cufftPlanMany with
// istride/idist) expresses that stride directly -- no transpose kernel is
// needed, and cuFFT batches all SAMPLES_PER_CHIRP of these strided FFTs as
// one launch. Each range bin's Doppler FFT is fully independent of every
// other range bin, which is exactly the "many independent, moderate-size
// transforms" shape cuFFT batching is built for.
//
// The one genuinely custom piece is fftshift: cuFFT (like every standard
// FFT library) returns bin 0 = DC/zero-Doppler at index 0 with negative
// frequencies wrapped to the back half. For a human- and CFAR-readable map
// we want zero-Doppler centered, so a small elementwise kernel swaps the
// two halves along the chirp axis.

namespace radar {
namespace gpu {

__global__ void fftshift_doppler_kernel(const cufftComplex* in, cufftComplex* out,
                                         int num_chirps, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_chirps * n;
    if (idx >= total) return;

    int m = idx / n;
    int k = idx % n;
    int shifted = (m + num_chirps / 2) % num_chirps;
    out[shifted * n + k] = in[idx];
}

void range_doppler(cufftHandle plan_slowtime, cufftComplex* d_rd, cudaStream_t stream) {
    const int n = SAMPLES_PER_CHIRP;
    const int m = NUM_CHIRPS;
    const int total = m * n;

    CUFFT_CHECK(cufftSetStream(plan_slowtime, stream));
    CUFFT_CHECK(cufftExecC2C(plan_slowtime, d_rd, d_rd, CUFFT_FORWARD));

    cufftComplex* d_shifted;
    CUDA_CHECK(cudaMallocAsync(&d_shifted, sizeof(cufftComplex) * total, stream));

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    fftshift_doppler_kernel<<<blocks, threads, 0, stream>>>(d_rd, d_shifted, m, n);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(d_rd, d_shifted, sizeof(cufftComplex) * total,
                               cudaMemcpyDeviceToDevice, stream));
    CUDA_CHECK(cudaFreeAsync(d_shifted, stream));
}

} // namespace gpu
} // namespace radar
