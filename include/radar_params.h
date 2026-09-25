#ifndef RADAR_PARAMS_H
#define RADAR_PARAMS_H

// Shared radar scenario parameters for a monostatic pulse-Doppler radar,
// mirroring the parameter style of radar_parameters.m from the IIT Jodhpur
// MATLAB range-Doppler simulation (LFM waveform, 2D-FFT processing chain).
//
// All stages (CPU reference and CUDA implementation) include this single
// header so both pipelines are guaranteed to run on identical physics.

#include <cstddef>
#include <cmath>
#include <complex>

namespace radar {

// MSVC's <cmath> does not define M_PI without _USE_MATH_DEFINES (and nvcc's
// device-side math headers don't guarantee it either), so this project
// defines its own constant rather than depending on a non-portable macro.
constexpr double PI = 3.14159265358979323846;

// ---- RF / waveform parameters -------------------------------------------
constexpr double C           = 299792458.0;   // speed of light, m/s
constexpr double CARRIER_FREQ = 10.0e9;       // X-band carrier, Hz
constexpr double BANDWIDTH   = 50.0e6;        // LFM sweep bandwidth, Hz
constexpr double SAMPLE_RATE = 100.0e6;       // fast-time sampling rate, Hz (2x oversampled vs B)

// ---- Fast-time (per-chirp) parameters ------------------------------------
constexpr int SAMPLES_PER_CHIRP = 1024;       // power of 2 -> radix-2 FFT friendly
constexpr double CHIRP_DURATION = static_cast<double>(SAMPLES_PER_CHIRP) / SAMPLE_RATE; // s
constexpr double CHIRP_RATE     = BANDWIDTH / CHIRP_DURATION; // Hz/s (LFM slope)

// ---- Slow-time (pulse-to-pulse) parameters -------------------------------
constexpr double PRF        = 5000.0;                 // pulse repetition frequency, Hz
constexpr double PRI        = 1.0 / PRF;               // pulse repetition interval, s
constexpr int NUM_CHIRPS    = 128;                     // chirps per coherent processing interval (power of 2)
constexpr int NUM_FRAMES    = 4;                       // independent CPI bursts, non-coherently integrated

// ---- Derived resolution / ambiguity limits -------------------------------
constexpr double RANGE_RESOLUTION   = C / (2.0 * BANDWIDTH);                 // m (set by bandwidth)
constexpr double RANGE_BIN_SPACING  = C / (2.0 * SAMPLE_RATE);               // m (fast-time sample grid)
constexpr double WAVELENGTH         = C / CARRIER_FREQ;                      // m
constexpr double MAX_UNAMBIG_VELOCITY = PRF * WAVELENGTH / 4.0;              // m/s
// This sim embeds each echo's range delay as a shift within a single chirp's
// fast-time record (no dechirp/stretch processing), so range is bounded by
// the record length itself:
constexpr double MAX_RANGE_WINDOW   = C * CHIRP_DURATION / 2.0;              // m

// ---- CFAR detector parameters --------------------------------------------
constexpr int CFAR_GUARD_CELLS    = 4;    // guard cells on each side, range axis
constexpr int CFAR_TRAINING_CELLS = 8;    // training cells on each side, range axis
constexpr double CFAR_PFA         = 1.0e-4; // desired probability of false alarm

// ---- Simulated scenario ---------------------------------------------------
struct Target {
    double range_m;       // initial range, meters (0 < range_m < MAX_RANGE_WINDOW)
    double velocity_mps;  // radial velocity, m/s (+ receding, - closing), |v| < MAX_UNAMBIG_VELOCITY
    float  amplitude;     // relative echo amplitude (post-RCS/path-loss lumped term)
};

constexpr int NUM_TARGETS = 2;

inline const Target* default_scenario() {
    // Snap the intended velocities (+15, -20 m/s) to the nearest exact
    // slow-time FFT bin. An unwindowed (rectangular) Doppler FFT has slowly
    // decaying sinc sidelobes for any target that doesn't land exactly on a
    // bin -- off by even a fraction of a bin smears real energy across most
    // of the Doppler axis and floods CA-CFAR with sidelobe-driven false
    // alarms. Real systems fix this with a slow-time window (Hann/Hamming);
    // here the fix is simpler and just as legitimate for a validation demo:
    // pick target velocities that are exact multiples of the bin spacing so
    // both CPU and GPU pipelines show clean, single-bin detections.
    static const double bin_vel = (PRF / NUM_CHIRPS) * (WAVELENGTH / 2.0);
    static const Target targets[NUM_TARGETS] = {
        {450.0, std::round(15.0 / bin_vel) * bin_vel, 1.0f},
        {900.0, std::round(-20.0 / bin_vel) * bin_vel, 0.35f}
    };
    return targets;
}

constexpr float NOISE_STDDEV = 0.25f; // complex AWGN std-dev per I/Q rail

struct Detection {
    int    doppler_bin;
    int    range_bin;
    float  power;
    double range_m;
    double velocity_mps;
};

constexpr int MAX_DETECTIONS = 256;

// Single source of truth for the LFM chirp waveform math, shared (in spirit)
// by scenario synthesis, the CPU reference matched filter, and the CUDA
// chirp_generation kernel (which reimplements this same formula device-side,
// since a __global__ kernel cannot call a host inline function).
inline std::complex<float> lfm_chirp_sample(int n) {
    const double t = static_cast<double>(n) / SAMPLE_RATE;
    const double phase = PI * CHIRP_RATE * t * t; // instantaneous LFM phase
    return std::complex<float>(static_cast<float>(std::cos(phase)),
                                static_cast<float>(std::sin(phase)));
}

} // namespace radar

#endif // RADAR_PARAMS_H
