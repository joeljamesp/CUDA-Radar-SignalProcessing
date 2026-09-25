#ifndef CPU_PIPELINE_H
#define CPU_PIPELINE_H

#include <complex>
#include <vector>
#include "radar_params.h"
#include "scenario.h"

// CPU-only reference implementation of the full pipeline: LFM chirp
// generation, FFT-based matched filtering, 2D range-Doppler map (slow-time
// FFT across chirps), non-coherent integration across frames, and CA-CFAR
// peak detection. This is the correctness baseline the CUDA pipeline is
// checked against in main.cu.
namespace radar {

// In-place radix-2 Cooley-Tukey FFT, n must be a power of two.
// Forward transform is unnormalized (matches cuFFT convention); inverse
// transform is also left unnormalized here -- callers apply 1/n scaling
// explicitly where needed, mirroring the GPU path exactly.
void fft_radix2(Complex* data, int n, bool inverse);

void generate_reference_chirp_cpu(Complex* chirp /* size SAMPLES_PER_CHIRP */);

// Pulse compression: for each of NUM_CHIRPS rows, FFT -> multiply by
// conj(refFFT)/N -> IFFT (unnormalized). `raw` and `out` are
// NUM_CHIRPS x SAMPLES_PER_CHIRP, row-major.
void matched_filter_cpu(const Complex* raw, const Complex* ref_chirp, Complex* out);

// Slow-time FFT across chirps for each range bin, followed by an fftshift
// along the Doppler axis so zero-Doppler sits at NUM_CHIRPS/2.
// `compressed` and `out` are NUM_CHIRPS x SAMPLES_PER_CHIRP, row-major.
void range_doppler_cpu(const Complex* compressed, Complex* out);

// integrated_power += |range_doppler|^2, elementwise, size NUM_CHIRPS*SAMPLES_PER_CHIRP.
void accumulate_power_cpu(const Complex* range_doppler, float* integrated_power);

float mean_power_cpu(const float* integrated_power);

// CA-CFAR along the range axis, one row (Doppler bin) at a time.
std::vector<Detection> cfar_cpu(const float* integrated_power);

struct CpuResult {
    std::vector<float> integrated_power; // NUM_CHIRPS x SAMPLES_PER_CHIRP
    std::vector<Detection> detections;
    double elapsed_ms;
};

CpuResult run_cpu_pipeline(const std::vector<FrameData>& frames);

} // namespace radar

#endif // CPU_PIPELINE_H
