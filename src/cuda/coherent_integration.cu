#include "cuda_pipeline.cuh"
#include "cuda_check.cuh"
#include <algorithm>

// Two genuinely hand-written kernels -- no cuFFT/cuBLAS equivalent to reach
// for here, which is the point: this is what "coherent_integration.cu
// should be a custom kernel" in the spec is asking for.
//
// 1) accumulate_power_kernel: elementwise |range_doppler|^2, accumulated
//    into a running sum across NUM_FRAMES independent bursts (non-coherent
//    integration for detection gain). Purely data-parallel, no reduction
//    needed -- each cell only ever touched by one thread per launch.
//
// 2) mean_power_reduction_kernel: a classic shared-memory tree reduction.
//    Used to estimate the map's mean noise power (reported alongside CFAR
//    results as a sanity check). Each block reduces its chunk into a single
//    partial sum in shared memory, then thread 0 folds that partial sum
//    into a global accumulator with one atomicAdd per block (not per
//    thread), which keeps atomic contention negligible.

namespace radar {
namespace gpu {

__global__ void accumulate_power_kernel(const cufftComplex* rd, float* integrated_power, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    float re = rd[idx].x;
    float im = rd[idx].y;
    integrated_power[idx] += re * re + im * im;
}

template <int BLOCK_SIZE>
__global__ void mean_power_reduction_kernel(const float* data, int total, float* g_sum) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Grid-stride first-level reduction: each thread folds two elements
    // (or more, via the stride loop) before ever touching shared memory.
    float sum = 0.0f;
    int stride = blockDim.x * 2 * gridDim.x;
    while (idx < total) {
        sum += data[idx];
        if (idx + blockDim.x < total) sum += data[idx + blockDim.x];
        idx += stride;
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        volatile float* vsmem = sdata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid + 8];
        vsmem[tid] += vsmem[tid + 4];
        vsmem[tid] += vsmem[tid + 2];
        vsmem[tid] += vsmem[tid + 1];
    }
    if (tid == 0) atomicAdd(g_sum, sdata[0]);
}

void accumulate_power(const cufftComplex* d_rd, float* d_integrated_power, cudaStream_t stream) {
    const int total = NUM_CHIRPS * SAMPLES_PER_CHIRP;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    accumulate_power_kernel<<<blocks, threads, 0, stream>>>(d_rd, d_integrated_power, total);
    CUDA_CHECK(cudaGetLastError());
}

float mean_power(const float* d_integrated_power, cudaStream_t stream) {
    const int total = NUM_CHIRPS * SAMPLES_PER_CHIRP;
    constexpr int BLOCK_SIZE = 256;
    const int blocks = std::min(64, (total + BLOCK_SIZE * 2 - 1) / (BLOCK_SIZE * 2));

    float* d_sum;
    CUDA_CHECK(cudaMallocAsync(&d_sum, sizeof(float), stream));
    CUDA_CHECK(cudaMemsetAsync(d_sum, 0, sizeof(float), stream));

    mean_power_reduction_kernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE, 0, stream>>>(
        d_integrated_power, total, d_sum);
    CUDA_CHECK(cudaGetLastError());

    float h_sum = 0.0f;
    CUDA_CHECK(cudaMemcpyAsync(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(d_sum, stream));

    return h_sum / static_cast<float>(total);
}

} // namespace gpu
} // namespace radar
