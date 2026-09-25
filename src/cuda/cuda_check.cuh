#ifndef CUDA_CHECK_CUH
#define CUDA_CHECK_CUH

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cufft.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call,         \
                         __FILE__, __LINE__, cudaGetErrorString(err__));        \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                       \
    } while (0)

#define CUFFT_CHECK(call)                                                        \
    do {                                                                         \
        cufftResult err__ = (call);                                              \
        if (err__ != CUFFT_SUCCESS) {                                            \
            std::fprintf(stderr, "cuFFT error %s at %s:%d: code %d\n", #call,    \
                         __FILE__, __LINE__, static_cast<int>(err__));           \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                        \
    } while (0)

#endif // CUDA_CHECK_CUH
