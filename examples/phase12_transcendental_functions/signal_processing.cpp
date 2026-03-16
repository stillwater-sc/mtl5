// signal_processing.cpp -- Transcendental Functions in Signal Processing
//
// This example demonstrates how element-wise transcendental functions enable
// clean, expressive signal processing workflows:
//
//   1. Discrete sine wave generation using sin()
//   2. Exponential decay envelopes using exp()
//   3. Amplitude-modulated signal (wave * envelope)
//   4. Phase angle extraction using atan()
//   5. Decibel conversion using log10()
//   6. Frequency-domain magnitude using sqrt() and pow()
//
// All operations work on entire vectors at once -- no manual loops needed.
// The functions map directly to mathematical notation, making DSP code
// readable and maintainable.

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <numbers>

using namespace mtl;

int main() {
    std::cout << std::fixed << std::setprecision(6);

    // -- 1. Generate a discrete time vector ------------------------------
    // 64 samples at a "sample rate" of 64 Hz -> 1 second of signal
    const std::size_t N = 64;
    const double fs = 64.0;  // sample rate
    const double f0 = 4.0;   // signal frequency (4 Hz)

    dense_vector<double> t(N);
    for (std::size_t i = 0; i < N; ++i) {
        t(i) = static_cast<double>(i) / fs;
    }

    std::cout << "=== Signal Processing with MTL5 Transcendental Functions ===\n\n";
    std::cout << "Sample rate: " << fs << " Hz, Signal freq: " << f0 << " Hz\n";
    std::cout << "Duration: " << static_cast<double>(N) / fs << " s (" << N << " samples)\n\n";

    // -- 2. Generate sine wave: x(t) = sin(2*pi*f0*t) -------------------
    // Scale time vector by angular frequency, then apply sin()
    auto omega_t = (2.0 * std::numbers::pi * f0) * t;
    auto sine_wave = mtl::sin(omega_t);

    std::cout << "-- Sine Wave (first 8 samples) --\n";
    for (std::size_t i = 0; i < 8; ++i) {
        std::cout << "  t=" << t(i) << "s  sin=" << sine_wave(i) << "\n";
    }

    // -- 3. Exponential decay envelope: e(t) = exp(-3*t) -----------------
    auto decay = mtl::exp(-3.0 * t);

    std::cout << "\n-- Exponential Decay Envelope (first 8 samples) --\n";
    for (std::size_t i = 0; i < 8; ++i) {
        std::cout << "  t=" << t(i) << "s  decay=" << decay(i) << "\n";
    }

    // -- 4. Amplitude-modulated signal: AM(t) = sin(w*t) * exp(-3*t) ----
    // Element-wise multiply: modulated signal
    dense_vector<double> am_signal(N);
    for (std::size_t i = 0; i < N; ++i) {
        am_signal(i) = sine_wave(i) * decay(i);
    }

    std::cout << "\n-- AM Signal (first 8 samples) --\n";
    for (std::size_t i = 0; i < 8; ++i) {
        std::cout << "  t=" << t(i) << "s  AM=" << am_signal(i) << "\n";
    }

    // -- 5. Power spectrum (squared magnitudes) --------------------------
    auto power = mtl::pow(am_signal, 2.0);

    // -- 6. Convert to decibels: dB = 10*log10(power) --------------------
    // Add small epsilon to avoid log(0)
    dense_vector<double> power_safe(N);
    for (std::size_t i = 0; i < N; ++i) {
        power_safe(i) = power(i) + 1e-20;
    }
    auto power_db = 10.0 * mtl::log10(power_safe);

    std::cout << "\n-- Power in dB (first 8 samples) --\n";
    for (std::size_t i = 0; i < 8; ++i) {
        std::cout << "  t=" << t(i) << "s  power=" << power(i)
                  << "  dB=" << power_db(i) << "\n";
    }

    // -- 7. Phase angles: atan(imag/real) using in-phase and quadrature --
    auto cosine_wave = mtl::cos(omega_t);
    // Phase = atan(sin/cos) -- recover the original angle
    dense_vector<double> ratio(N);
    for (std::size_t i = 0; i < N; ++i) {
        // Avoid division by very small cos values
        if (std::abs(cosine_wave(i)) > 1e-10) {
            ratio(i) = sine_wave(i) / cosine_wave(i);
        } else {
            ratio(i) = 0.0;
        }
    }
    auto phase = mtl::atan(ratio);

    std::cout << "\n-- Phase Angles (first 8 samples) --\n";
    for (std::size_t i = 0; i < 8; ++i) {
        std::cout << "  t=" << t(i) << "s  phase=" << phase(i) << " rad\n";
    }

    // -- 8. Verify Pythagorean identity on the signal --------------------
    auto sin2 = mtl::pow(sine_wave, 2.0);
    auto cos2 = mtl::pow(cosine_wave, 2.0);
    double max_identity_error = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        double err = std::abs(sin2(i) + cos2(i) - 1.0);
        if (err > max_identity_error) max_identity_error = err;
    }
    std::cout << "\n-- Identity Check: max |sin^2 + cos^2 - 1| = "
              << max_identity_error << " --\n";

    // -- 9. RMS amplitude using sqrt -------------------------------------
    double sum_sq = 0.0;
    for (std::size_t i = 0; i < N; ++i) {
        sum_sq += am_signal(i) * am_signal(i);
    }
    double rms = std::sqrt(sum_sq / static_cast<double>(N));
    std::cout << "\n-- RMS amplitude of AM signal: " << rms << " --\n";

    std::cout << "\nAll signal processing operations completed successfully.\n";
    return 0;
}
