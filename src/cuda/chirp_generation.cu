#include "cuda_pipeline.cuh"
#include "cuda_check.cuh"
#include <cmath>

// LFM reference chirp generation. Each fast-time sample n's phase
// pi * chirp_rate * (n/Fs)^2 depends only on n -- there is zero
// inter-thread dependency, so this maps to one thread per sample with no
// synchronization, no shared memory, and fully coalesced writes to
// d_chirp[n]. SAMPLES_PER_CHIRP = 1024 launches as a single block.

namespace radar {
namespace gpu {

__global__ void generate_reference_chirp_kernel(cufftComplex* chirp, int n,
                                                  double sample_rate, double chirp_rate) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    double t = static_cast<double>(idx) / sample_rate;
    double phase = PI * chirp_rate * t * t;
    chirp[idx].x = static_cast<float>(cos(phase));
    chirp[idx].y = static_cast<float>(sin(phase));
}

void generate_reference_chirp(cufftComplex* d_chirp, cudaStream_t stream) {
    const int n = SAMPLES_PER_CHIRP;
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    generate_reference_chirp_kernel<<<blocks, threads, 0, stream>>>(
        d_chirp, n, SAMPLE_RATE, CHIRP_RATE);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace gpu
} // namespace radar
