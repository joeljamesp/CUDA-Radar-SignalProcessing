#include "cuda_pipeline.cuh"
#include "cuda_check.cuh"

// CA-CFAR (cell-averaging constant false alarm rate) detector.
//
// One thread per range-Doppler cell -- a good showcase of data-dependent
// branching on GPU: cells too close to the range-axis edges to have a full
// training window simply return early (divergent within a warp only near
// the two small edge bands, negligible cost), and every surviving thread
// makes an independent threshold-compare branch. Detections are appended
// to a shared output array via atomicAdd on a single counter, which is the
// standard "unknown output size" GPU pattern -- far cheaper here than a
// stream-compaction pass since true detections are a tiny fraction of the
// NUM_CHIRPS x SAMPLES_PER_CHIRP cells.

namespace radar {
namespace gpu {

__global__ void cfar_kernel(const float* integrated_power, Detection* detections,
                             int* detection_count, int num_chirps, int num_range,
                             int guard, int training, float alpha, int max_detections) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_chirps * num_range;
    if (idx >= total) return;

    int m = idx / num_range; // Doppler bin
    int k = idx % num_range; // range bin

    const int lo = training + guard;
    const int hi = num_range - training - guard;
    if (k < lo || k >= hi) return; // not enough training cells on both sides

    const float* row = &integrated_power[static_cast<size_t>(m) * num_range];

    float training_sum = 0.0f;
    for (int d = 1; d <= training; ++d) {
        training_sum += row[k - guard - d] + row[k + guard + d];
    }
    const float noise_est = training_sum / static_cast<float>(2 * training);
    const float threshold = alpha * noise_est;
    const float cell_power = row[k];

    if (cell_power > threshold) {
        int out_idx = atomicAdd(detection_count, 1);
        if (out_idx < max_detections) {
            Detection det;
            det.doppler_bin = m;
            det.range_bin = k;
            det.power = cell_power;
            det.range_m = k * RANGE_BIN_SPACING;
            det.velocity_mps = (m - num_chirps / 2) * (PRF / num_chirps) * WAVELENGTH / 2.0;
            detections[out_idx] = det;
        }
    }
}

int cfar_detect(const float* d_integrated_power, Detection* d_detections, cudaStream_t stream) {
    const int total = NUM_CHIRPS * SAMPLES_PER_CHIRP;
    const int total_training = 2 * CFAR_TRAINING_CELLS;
    const float alpha = static_cast<float>(
        total_training * (pow(CFAR_PFA, -1.0 / total_training) - 1.0));

    int* d_count;
    CUDA_CHECK(cudaMallocAsync(&d_count, sizeof(int), stream));
    CUDA_CHECK(cudaMemsetAsync(d_count, 0, sizeof(int), stream));

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    cfar_kernel<<<blocks, threads, 0, stream>>>(
        d_integrated_power, d_detections, d_count, NUM_CHIRPS, SAMPLES_PER_CHIRP,
        CFAR_GUARD_CELLS, CFAR_TRAINING_CELLS, alpha, MAX_DETECTIONS);
    CUDA_CHECK(cudaGetLastError());

    int h_count = 0;
    CUDA_CHECK(cudaMemcpyAsync(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    CUDA_CHECK(cudaFreeAsync(d_count, stream));

    return h_count < MAX_DETECTIONS ? h_count : MAX_DETECTIONS;
}

} // namespace gpu
} // namespace radar
