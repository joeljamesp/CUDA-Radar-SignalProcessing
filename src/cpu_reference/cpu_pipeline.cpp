#include "cpu_pipeline.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>
#include <vector>

namespace radar {

void fft_radix2(Complex* data, int n, bool inverse) {
    // Bit-reversal permutation.
    for (int i = 1, j = 0; i < n; ++i) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(data[i], data[j]);
    }
    // Iterative Cooley-Tukey butterflies.
    for (int len = 2; len <= n; len <<= 1) {
        const double ang = (inverse ? 2.0 : -2.0) * PI / len;
        const Complex wlen(static_cast<float>(std::cos(ang)), static_cast<float>(std::sin(ang)));
        for (int i = 0; i < n; i += len) {
            Complex w(1.0f, 0.0f);
            for (int j = 0; j < len / 2; ++j) {
                Complex u = data[i + j];
                Complex v = data[i + j + len / 2] * w;
                data[i + j] = u + v;
                data[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

void generate_reference_chirp_cpu(Complex* chirp) {
    for (int n = 0; n < SAMPLES_PER_CHIRP; ++n) {
        chirp[n] = lfm_chirp_sample(n);
    }
}

void matched_filter_cpu(const Complex* raw, const Complex* ref_chirp, Complex* out) {
    const int N = SAMPLES_PER_CHIRP;
    std::vector<Complex> ref_fft(ref_chirp, ref_chirp + N);
    fft_radix2(ref_fft.data(), N, false);

    std::vector<Complex> row(N);
    for (int m = 0; m < NUM_CHIRPS; ++m) {
        const Complex* src = &raw[static_cast<size_t>(m) * N];
        std::copy(src, src + N, row.begin());
        fft_radix2(row.data(), N, false);
        for (int k = 0; k < N; ++k) {
            // Fold the 1/N inverse-FFT normalization into this multiply so
            // the following inverse transform can stay unnormalized, same
            // convention used by the GPU frequency-domain-multiply kernel.
            row[k] = row[k] * std::conj(ref_fft[k]) * (1.0f / static_cast<float>(N));
        }
        fft_radix2(row.data(), N, true);
        std::copy(row.begin(), row.end(), &out[static_cast<size_t>(m) * N]);
    }
}

void range_doppler_cpu(const Complex* compressed, Complex* out) {
    const int N = SAMPLES_PER_CHIRP;
    const int M = NUM_CHIRPS;
    std::vector<Complex> col(M);

    for (int k = 0; k < N; ++k) {
        for (int m = 0; m < M; ++m) {
            col[m] = compressed[static_cast<size_t>(m) * N + k];
        }
        fft_radix2(col.data(), M, false);
        // fftshift along Doppler axis: swap the two halves.
        for (int m = 0; m < M; ++m) {
            const int shifted = (m + M / 2) % M;
            out[static_cast<size_t>(shifted) * N + k] = col[m];
        }
    }
}

void accumulate_power_cpu(const Complex* range_doppler, float* integrated_power) {
    const size_t total = static_cast<size_t>(NUM_CHIRPS) * SAMPLES_PER_CHIRP;
    for (size_t i = 0; i < total; ++i) {
        const float re = range_doppler[i].real();
        const float im = range_doppler[i].imag();
        integrated_power[i] += re * re + im * im;
    }
}

float mean_power_cpu(const float* integrated_power) {
    const size_t total = static_cast<size_t>(NUM_CHIRPS) * SAMPLES_PER_CHIRP;
    double sum = 0.0;
    for (size_t i = 0; i < total; ++i) sum += integrated_power[i];
    return static_cast<float>(sum / static_cast<double>(total));
}

std::vector<Detection> cfar_cpu(const float* integrated_power) {
    const int N = SAMPLES_PER_CHIRP;
    const int M = NUM_CHIRPS;
    const int G = CFAR_GUARD_CELLS;
    const int T = CFAR_TRAINING_CELLS;
    const int total_training = 2 * T;
    const float alpha = static_cast<float>(
        total_training * (std::pow(CFAR_PFA, -1.0 / total_training) - 1.0));

    std::vector<Detection> detections;
    for (int m = 0; m < M; ++m) {
        const float* row = &integrated_power[static_cast<size_t>(m) * N];
        for (int k = T + G; k < N - T - G; ++k) {
            float training_sum = 0.0f;
            for (int d = 1; d <= T; ++d) {
                training_sum += row[k - G - d] + row[k + G + d];
            }
            const float noise_est = training_sum / total_training;
            const float threshold = alpha * noise_est;
            if (row[k] > threshold) {
                Detection det;
                det.doppler_bin = m;
                det.range_bin = k;
                det.power = row[k];
                det.range_m = k * RANGE_BIN_SPACING;
                det.velocity_mps = (m - M / 2) * (PRF / M) * WAVELENGTH / 2.0;
                detections.push_back(det);
            }
        }
    }
    return detections;
}

CpuResult run_cpu_pipeline(const std::vector<FrameData>& frames) {
    const auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<Complex> ref_chirp(SAMPLES_PER_CHIRP);
    generate_reference_chirp_cpu(ref_chirp.data());

    const size_t total = static_cast<size_t>(NUM_CHIRPS) * SAMPLES_PER_CHIRP;
    std::vector<float> integrated_power(total, 0.0f);
    std::vector<Complex> compressed(total);
    std::vector<Complex> range_doppler(total);

    for (const auto& frame : frames) {
        matched_filter_cpu(frame.data(), ref_chirp.data(), compressed.data());
        range_doppler_cpu(compressed.data(), range_doppler.data());
        accumulate_power_cpu(range_doppler.data(), integrated_power.data());
    }

    std::vector<Detection> detections = cfar_cpu(integrated_power.data());

    const auto t1 = std::chrono::high_resolution_clock::now();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    return CpuResult{std::move(integrated_power), std::move(detections), elapsed_ms};
}

} // namespace radar
