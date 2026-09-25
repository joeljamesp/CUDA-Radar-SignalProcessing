#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include "radar_params.h"
#include "scenario.h"
#include "cpu_reference/cpu_pipeline.h"
#include "cuda/cuda_pipeline.cuh"
#include "cuda/cuda_check.cuh"

using namespace radar;

namespace {

struct GpuResult {
    std::vector<float> integrated_power;
    std::vector<Detection> detections;
    double compute_ms;   // pipeline + peak detection only (excludes setup/H2D of raw frames' first alloc)
    double total_ms;     // includes cuFFT plan creation and all transfers
};

GpuResult run_gpu_pipeline(const std::vector<FrameData>& frames) {
    const int N = SAMPLES_PER_CHIRP;
    const int M = NUM_CHIRPS;
    const int total = M * N;
    const size_t bytes_c = sizeof(cufftComplex) * total;

    cudaEvent_t ev_total_start, ev_total_end, ev_compute_start, ev_compute_end;
    CUDA_CHECK(cudaEventCreate(&ev_total_start));
    CUDA_CHECK(cudaEventCreate(&ev_total_end));
    CUDA_CHECK(cudaEventCreate(&ev_compute_start));
    CUDA_CHECK(cudaEventCreate(&ev_compute_end));
    CUDA_CHECK(cudaEventRecord(ev_total_start));

    // --- cuFFT plans (created once, reused across all frames) ---
    cufftHandle plan_single;   // 1x SAMPLES_PER_CHIRP, for the reference chirp
    cufftHandle plan_fwd_rows; // NUM_CHIRPS batches of SAMPLES_PER_CHIRP (fast-time)
    cufftHandle plan_inv_rows;
    cufftHandle plan_slowtime; // SAMPLES_PER_CHIRP batches of NUM_CHIRPS (slow-time, strided)

    CUFFT_CHECK(cufftPlan1d(&plan_single, N, CUFFT_C2C, 1));
    CUFFT_CHECK(cufftPlan1d(&plan_fwd_rows, N, CUFFT_C2C, M));
    CUFFT_CHECK(cufftPlan1d(&plan_inv_rows, N, CUFFT_C2C, M));

    // NOTE: inembed/onembed must be non-NULL for cuFFT to honor the
    // istride/idist parameters at all -- passing NULL silently discards
    // them and falls back to contiguous (wrong-axis) batching. embed={M}
    // just describes the logical extent of the strided dimension.
    int n_slow[1] = {M};
    int embed_slow[1] = {M};
    CUFFT_CHECK(cufftPlanMany(&plan_slowtime, 1, n_slow,
                               embed_slow, N, 1,   // input: stride N (chirp axis), dist 1 (range axis)
                               embed_slow, N, 1,   // output: same strided layout, in-place
                               CUFFT_C2C, N));  // batch = N range bins

    // --- device buffers ---
    cufftComplex* d_ref_chirp;
    cufftComplex* d_ref_fft_conj;
    cufftComplex* d_rows;       // reused per frame: raw -> compressed -> range-doppler
    float* d_integrated_power;
    Detection* d_detections;

    CUDA_CHECK(cudaMalloc(&d_ref_chirp, bytes_c / M)); // N complex samples
    CUDA_CHECK(cudaMalloc(&d_ref_fft_conj, bytes_c / M));
    CUDA_CHECK(cudaMalloc(&d_rows, bytes_c));
    CUDA_CHECK(cudaMalloc(&d_integrated_power, sizeof(float) * total));
    CUDA_CHECK(cudaMalloc(&d_detections, sizeof(Detection) * MAX_DETECTIONS));
    CUDA_CHECK(cudaMemset(d_integrated_power, 0, sizeof(float) * total));

    static_assert(sizeof(cufftComplex) == sizeof(Complex), "layout mismatch");

    CUDA_CHECK(cudaEventRecord(ev_compute_start));

    gpu::generate_reference_chirp(d_ref_chirp);
    gpu::compute_ref_fft_conj(plan_single, d_ref_chirp, d_ref_fft_conj);

    for (const auto& frame : frames) {
        CUDA_CHECK(cudaMemcpy(d_rows, frame.data(), bytes_c, cudaMemcpyHostToDevice));
        gpu::matched_filter(plan_fwd_rows, plan_inv_rows, d_rows, d_ref_fft_conj);
        gpu::range_doppler(plan_slowtime, d_rows);
        gpu::accumulate_power(d_rows, d_integrated_power);
    }

    const float noise_floor = gpu::mean_power(d_integrated_power);
    (void)noise_floor;
    const int num_detections = gpu::cfar_detect(d_integrated_power, d_detections);

    CUDA_CHECK(cudaEventRecord(ev_compute_end));

    GpuResult result;
    result.integrated_power.resize(total);
    CUDA_CHECK(cudaMemcpy(result.integrated_power.data(), d_integrated_power,
                          sizeof(float) * total, cudaMemcpyDeviceToHost));

    result.detections.resize(num_detections);
    if (num_detections > 0) {
        CUDA_CHECK(cudaMemcpy(result.detections.data(), d_detections,
                              sizeof(Detection) * num_detections, cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaEventRecord(ev_total_end));
    CUDA_CHECK(cudaEventSynchronize(ev_total_end));

    float compute_ms = 0.0f, total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&compute_ms, ev_compute_start, ev_compute_end));
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, ev_total_start, ev_total_end));
    result.compute_ms = compute_ms;
    result.total_ms = total_ms;

    cudaFree(d_ref_chirp);
    cudaFree(d_ref_fft_conj);
    cudaFree(d_rows);
    cudaFree(d_integrated_power);
    cudaFree(d_detections);
    cufftDestroy(plan_single);
    cufftDestroy(plan_fwd_rows);
    cufftDestroy(plan_inv_rows);
    cufftDestroy(plan_slowtime);
    cudaEventDestroy(ev_total_start);
    cudaEventDestroy(ev_total_end);
    cudaEventDestroy(ev_compute_start);
    cudaEventDestroy(ev_compute_end);

    return result;
}

bool detection_matches_target(const Detection& d, const Target& t) {
    // Within +/- 2 range bins and +/- 1 Doppler bin of the true target.
    const double range_tol = 2.0 * RANGE_BIN_SPACING;
    const double vel_tol = 1.0 * (PRF / NUM_CHIRPS) * WAVELENGTH / 2.0;
    return std::fabs(d.range_m - t.range_m) <= range_tol &&
           std::fabs(d.velocity_mps - t.velocity_mps) <= vel_tol;
}

} // namespace

int main() {
    int device_count = 0;
    cudaError_t dev_err = cudaGetDeviceCount(&device_count);

    std::printf("=== GPU-Accelerated Radar Signal Processing ===\n");
    std::printf("Carrier: %.1f GHz | Bandwidth: %.1f MHz | Sample rate: %.1f MHz\n",
                CARRIER_FREQ / 1e9, BANDWIDTH / 1e6, SAMPLE_RATE / 1e6);
    std::printf("Chirps/CPI: %d | Samples/chirp: %d | Frames: %d | PRF: %.1f kHz\n",
                NUM_CHIRPS, SAMPLES_PER_CHIRP, NUM_FRAMES, PRF / 1e3);
    std::printf("Range resolution: %.2f m | Range bin spacing: %.3f m | Max unambig. velocity: %.1f m/s\n\n",
                RANGE_RESOLUTION, RANGE_BIN_SPACING, MAX_UNAMBIG_VELOCITY);

    if (dev_err != cudaSuccess || device_count == 0) {
        std::printf("No CUDA-capable GPU detected on this machine (cudaGetDeviceCount: %s).\n",
                    cudaGetErrorString(dev_err));
        std::printf("Compiled successfully; GPU pipeline cannot run here. See README.md.\n");
        return 0;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::printf("GPU: %s (SM %d.%d, %d SMs)\n\n", prop.name, prop.major, prop.minor,
                prop.multiProcessorCount);

    std::printf("Generating scenario: %d targets over %d frames...\n", NUM_TARGETS, NUM_FRAMES);
    std::vector<FrameData> frames = generate_scenario();

    std::printf("Running CPU reference pipeline...\n");
    CpuResult cpu = run_cpu_pipeline(frames);

    std::printf("Running CUDA pipeline...\n");
    GpuResult gpu = run_gpu_pipeline(frames);

    // --- Validation: GPU integrated power map vs CPU reference ---
    double max_abs_err = 0.0, sum_sq_err = 0.0;
    const size_t total = cpu.integrated_power.size();
    for (size_t i = 0; i < total; ++i) {
        double err = std::fabs(static_cast<double>(cpu.integrated_power[i]) -
                                static_cast<double>(gpu.integrated_power[i]));
        max_abs_err = std::max(max_abs_err, err);
        sum_sq_err += err * err;
    }
    const double rmse = std::sqrt(sum_sq_err / static_cast<double>(total));

    std::printf("\n=== Correctness: GPU vs CPU integrated range-Doppler map ===\n");
    std::printf("Max abs error: %.6e | RMSE: %.6e\n", max_abs_err, rmse);
    std::printf("CPU detections: %zu | GPU detections: %zu\n",
                cpu.detections.size(), gpu.detections.size());

    std::printf("\n=== Detections (GPU/CFAR) ===\n");
    const Target* targets = default_scenario();
    for (const auto& d : gpu.detections) {
        bool matched = false;
        for (int t = 0; t < NUM_TARGETS; ++t) {
            if (detection_matches_target(d, targets[t])) { matched = true; break; }
        }
        std::printf("  range=%.2f m  velocity=%+.2f m/s  power=%.3f  %s\n",
                    d.range_m, d.velocity_mps, d.power,
                    matched ? "(matches known target)" : "");
    }

    std::printf("\n=== Ground truth targets ===\n");
    for (int t = 0; t < NUM_TARGETS; ++t) {
        std::printf("  range=%.2f m  velocity=%+.2f m/s  amplitude=%.2f\n",
                    targets[t].range_m, targets[t].velocity_mps, targets[t].amplitude);
    }

    std::printf("\n=== Timing (measured, this run, this machine) ===\n");
    std::printf("CPU pipeline (wall clock):            %.3f ms\n", cpu.elapsed_ms);
    std::printf("GPU pipeline (CUDA events, compute):   %.3f ms\n", gpu.compute_ms);
    std::printf("GPU pipeline (CUDA events, incl setup): %.3f ms\n", gpu.total_ms);
    std::printf("Speedup (compute-only):  %.2fx\n", cpu.elapsed_ms / gpu.compute_ms);
    std::printf("Speedup (incl. setup):   %.2fx\n", cpu.elapsed_ms / gpu.total_ms);

    return 0;
}
