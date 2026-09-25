#ifndef SCENARIO_H
#define SCENARIO_H

#include <complex>
#include <vector>
#include "radar_params.h"

// Scenario synthesis lives on the CPU (host orchestration, item 3 of the
// build spec) and is run exactly once. Both the CPU reference pipeline and
// the CUDA pipeline consume the *same* generated raw IQ data, so the
// GPU-vs-CPU validation downstream measures divergence in the signal
// processing stages only, not from two different random data sources.
namespace radar {

using Complex = std::complex<float>;

// One frame == one coherent processing interval (CPI): NUM_CHIRPS rows of
// SAMPLES_PER_CHIRP complex baseband samples, row-major (chirp-major).
using FrameData = std::vector<Complex>;

// Generates NUM_FRAMES independent CPI bursts for the default target
// scenario. Each frame re-draws its own AWGN realization (frame_seed offset)
// so non-coherent integration across frames yields real detection gain.
std::vector<FrameData> generate_scenario(unsigned base_seed = 42);

} // namespace radar

#endif // SCENARIO_H
