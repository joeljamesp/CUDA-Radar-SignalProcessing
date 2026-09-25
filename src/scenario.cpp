#include "scenario.h"
#include <random>
#include <cmath>

namespace radar {

std::vector<FrameData> generate_scenario(unsigned base_seed) {
    const Target* targets = default_scenario();
    std::vector<FrameData> frames(NUM_FRAMES);

    for (int f = 0; f < NUM_FRAMES; ++f) {
        FrameData& frame = frames[f];
        frame.assign(static_cast<size_t>(NUM_CHIRPS) * SAMPLES_PER_CHIRP, Complex(0.0f, 0.0f));

        // Each target contributes a range/Doppler-shifted, amplitude-scaled
        // copy of the transmit chirp to every row of the frame. Range walk
        // across a single CPI (128 chirps * 200us = 25.6ms) is sub-bin at
        // these velocities, so range is held fixed within a frame.
        for (int ti = 0; ti < NUM_TARGETS; ++ti) {
            const Target& tgt = targets[ti];
            const int delay_samples = static_cast<int>(
                std::lround(2.0 * tgt.range_m / C * SAMPLE_RATE));
            const double f_doppler = 2.0 * tgt.velocity_mps / WAVELENGTH;

            for (int m = 0; m < NUM_CHIRPS; ++m) {
                const double doppler_phase = 2.0 * PI * f_doppler * m * PRI;
                const Complex doppler_term(static_cast<float>(std::cos(doppler_phase)),
                                            static_cast<float>(std::sin(doppler_phase)));
                Complex* row = &frame[static_cast<size_t>(m) * SAMPLES_PER_CHIRP];

                for (int k = 0; k < SAMPLES_PER_CHIRP; ++k) {
                    int src = k - delay_samples;
                    src = ((src % SAMPLES_PER_CHIRP) + SAMPLES_PER_CHIRP) % SAMPLES_PER_CHIRP;
                    row[k] += tgt.amplitude * lfm_chirp_sample(src) * doppler_term;
                }
            }
        }

        // Complex AWGN, independent realization per frame.
        std::mt19937 rng(base_seed + static_cast<unsigned>(f) * 7919u);
        std::normal_distribution<float> noise(0.0f, NOISE_STDDEV);
        for (auto& sample : frame) {
            sample += Complex(noise(rng), noise(rng));
        }
    }

    return frames;
}

} // namespace radar
