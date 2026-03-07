// phase12c_radioactive_decay.cpp — Exponential Decay & Chemical Kinetics
//
// This example demonstrates how exponential and logarithmic functions
// model fundamental processes in nuclear physics and chemistry:
//
//   1. Single-species radioactive decay:  N(t) = N₀ · exp(-λt)
//   2. Half-life extraction via log:      t½ = ln(2) / λ
//   3. Two-species decay chain:           Parent → Daughter (Bateman equations)
//   4. Arrhenius reaction rates:          k(T) = A · exp(-Eₐ/(R·T))
//   5. Logarithmic scales in chemistry:   pH = -log₁₀[H⁺]
//   6. Carbon-14 dating:                  age = -t½/ln(2) · ln(N/N₀)
//
// Mathematics: The exponential function is the unique function satisfying
// dN/dt = -λN, making it the natural language of first-order kinetics.
// The logarithm inverts this relationship, enabling measurement of rates
// from observed quantities. These are arguably the most important
// transcendental functions in all of science.
//
// MTL5 functions used: exp, log, log2, log10, pow, abs, sqrt

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <numbers>
#include <cmath>

using namespace mtl;

int main() {
    std::cout << "=============================================================\n";
    std::cout << " Phase 12C: Radioactive Decay & Chemical Kinetics\n";
    std::cout << "=============================================================\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // Part 1: Single-Species Radioactive Decay
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== Part 1: Single-Species Radioactive Decay ===\n\n";

    // Iodine-131: half-life = 8.02 days, used in thyroid cancer treatment.
    // The decay constant λ = ln(2)/t½ connects the half-life to the
    // continuous exponential model: N(t) = N₀ · exp(-λt).

    const double half_life_I131 = 8.02;  // days
    const double lambda = std::numbers::ln2 / half_life_I131;
    const double N0 = 1.0e6;  // initial atoms (normalized)

    std::cout << "Iodine-131 (thyroid cancer treatment):\n";
    std::cout << "  Half-life:      " << half_life_I131 << " days\n";
    std::cout << std::scientific << std::setprecision(6);
    std::cout << "  Decay constant: " << lambda << " per day\n";
    std::cout << "  Initial atoms:  " << N0 << "\n\n";

    // Sample at 10 time points spanning 5 half-lives
    const std::size_t nt = 40;
    dense_vector<double> t(nt);
    for (std::size_t i = 0; i < nt; ++i) {
        t(i) = static_cast<double>(i) * half_life_I131 * 5.0 / (nt - 1);
    }

    // N(t) = N₀ · exp(-λt) — one function call on the entire time vector
    auto N_t = N0 * mtl::exp(-lambda * t);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << std::setw(10) << "Day"
              << std::setw(16) << "Atoms"
              << std::setw(14) << "Half-lives"
              << std::setw(14) << "% remaining\n";
    std::cout << std::string(54, '-') << "\n";

    for (std::size_t i = 0; i < nt; i += 4) {
        double half_lives = t(i) / half_life_I131;
        double pct = 100.0 * N_t(i) / N0;
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(10) << t(i)
                  << std::scientific << std::setprecision(4)
                  << std::setw(16) << N_t(i)
                  << std::fixed << std::setprecision(2)
                  << std::setw(14) << half_lives
                  << std::setw(13) << pct << "%\n";
    }

    // Verify: after exactly 1 half-life, N should be N₀/2
    double N_at_halflife = N0 * std::exp(-lambda * half_life_I131);
    std::cout << std::scientific << std::setprecision(10);
    std::cout << "\nVerification: N(t½) = " << N_at_halflife
              << " (should be " << N0 / 2.0 << ")\n";
    std::cout << "  Error: " << std::abs(N_at_halflife - N0 / 2.0) << "\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // Part 2: Extracting Half-Life from Data Using log
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== Part 2: Extracting Half-Life from Data Using log ===\n\n";

    // Key insight: if N(t) = N₀·exp(-λt), then ln(N/N₀) = -λt.
    // Plotting ln(N/N₀) vs t gives a straight line with slope = -λ.
    // This is how experimentalists determine decay constants.

    // Compute ln(N(t)/N₀) — should be a perfect straight line
    auto ratio = (1.0 / N0) * N_t;
    auto log_ratio = mtl::log(ratio);

    std::cout << "ln(N/N₀) vs t — should be linear with slope -λ:\n\n";
    std::cout << std::setw(10) << "Day"
              << std::setw(16) << "ln(N/N₀)"
              << std::setw(16) << "-λ·t"
              << std::setw(14) << "Error\n";
    std::cout << std::string(56, '-') << "\n";

    double max_log_error = 0.0;
    for (std::size_t i = 0; i < nt; i += 5) {
        double expected = -lambda * t(i);
        double err = std::abs(log_ratio(i) - expected);
        max_log_error = std::max(max_log_error, err);
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(10) << t(i)
                  << std::setprecision(8)
                  << std::setw(16) << log_ratio(i)
                  << std::setw(16) << expected
                  << std::scientific << std::setw(14) << err << "\n";
    }
    std::cout << "\nMax |ln(N/N₀) - (-λt)| = " << max_log_error
              << " (exp-log round trip)\n";

    // Recover half-life from the data: t½ = ln(2)/λ
    // Estimate λ from two data points: λ_est = -[ln(N₂) - ln(N₁)] / (t₂ - t₁)
    double lambda_est = -(log_ratio(nt - 1) - log_ratio(0)) / (t(nt - 1) - t(0));
    double half_life_est = std::numbers::ln2 / lambda_est;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\nRecovered from data:\n";
    std::cout << "  λ_estimated = " << lambda_est << " (true: " << lambda << ")\n";
    std::cout << "  t½_estimated = " << half_life_est << " days (true: " << half_life_I131 << ")\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // Part 3: Two-Species Decay Chain (Bateman Equations)
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== Part 3: Two-Species Decay Chain ===\n\n";

    // Model: Parent (P) → Daughter (D) → stable
    //   dP/dt = -λ₁P             →  P(t) = P₀·exp(-λ₁t)
    //   dD/dt = λ₁P - λ₂D        →  D(t) = P₀·λ₁/(λ₂-λ₁)·[exp(-λ₁t) - exp(-λ₂t)]
    //
    // Example: Sr-90 (t½=28.8y) → Y-90 (t½=2.67d) → Zr-90 (stable)
    // This is important in nuclear waste management.

    const double half_life_Sr90 = 28.8 * 365.25;    // convert years to days
    const double half_life_Y90 = 2.67;               // days
    const double lambda1 = std::numbers::ln2 / half_life_Sr90;
    const double lambda2 = std::numbers::ln2 / half_life_Y90;
    const double P0 = 1.0e6;

    std::cout << "Sr-90 → Y-90 → Zr-90 (stable):\n";
    std::cout << "  Sr-90 half-life: 28.8 years (" << half_life_Sr90 << " days)\n";
    std::cout << "  Y-90 half-life:  2.67 days\n\n";

    // Sample over 30 days (shows Y-90 buildup and equilibrium)
    const std::size_t nc = 60;
    dense_vector<double> tc(nc);
    for (std::size_t i = 0; i < nc; ++i) {
        tc(i) = static_cast<double>(i) * 30.0 / (nc - 1);
    }

    // Parent: P(t) = P₀·exp(-λ₁t)
    auto P_t = P0 * mtl::exp(-lambda1 * tc);

    // Daughter (Bateman): D(t) = P₀·λ₁/(λ₂-λ₁)·[exp(-λ₁t) - exp(-λ₂t)]
    double coeff = P0 * lambda1 / (lambda2 - lambda1);
    auto exp_parent = mtl::exp(-lambda1 * tc);
    auto exp_daughter = mtl::exp(-lambda2 * tc);
    dense_vector<double> D_t(nc);
    for (std::size_t i = 0; i < nc; ++i) {
        D_t(i) = coeff * (exp_parent(i) - exp_daughter(i));
    }

    // Conservation: P(t) + D(t) + stable(t) = P₀
    std::cout << std::setw(8) << "Day"
              << std::setw(14) << "Parent"
              << std::setw(14) << "Daughter"
              << std::setw(14) << "Stable"
              << std::setw(14) << "Total\n";
    std::cout << std::string(64, '-') << "\n";

    double max_conservation_err = 0.0;
    for (std::size_t i = 0; i < nc; i += 6) {
        double stable = P0 - P_t(i) - D_t(i);
        double total = P_t(i) + D_t(i) + stable;
        max_conservation_err = std::max(max_conservation_err, std::abs(total - P0));
        std::cout << std::fixed << std::setprecision(1)
                  << std::setw(8) << tc(i)
                  << std::scientific << std::setprecision(4)
                  << std::setw(14) << P_t(i)
                  << std::setw(14) << D_t(i)
                  << std::setw(14) << stable
                  << std::setw(14) << total << "\n";
    }
    std::cout << "\nConservation check: max |P + D + stable - P₀| = "
              << max_conservation_err << "\n";

    // Find the time of peak daughter activity
    // dD/dt = 0 → t_peak = ln(λ₂/λ₁) / (λ₂ - λ₁)
    double t_peak = std::log(lambda2 / lambda1) / (lambda2 - lambda1);
    double D_peak = coeff * (std::exp(-lambda1 * t_peak) - std::exp(-lambda2 * t_peak));
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nPeak daughter activity at t = " << t_peak << " days\n";
    std::cout << "  D(t_peak) = " << std::scientific << D_peak << "\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // Part 4: Arrhenius Equation — Temperature-Dependent Reaction Rates
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== Part 4: Arrhenius Reaction Rates ===\n\n";

    // k(T) = A · exp(-Eₐ/(R·T))
    // The Arrhenius equation relates reaction rate to temperature.
    // A plot of ln(k) vs 1/T gives a straight line (Arrhenius plot).

    const double A = 1.0e13;    // pre-exponential factor (s⁻¹)
    const double Ea = 75000.0;  // activation energy (J/mol)
    const double R = 8.314;     // gas constant (J/(mol·K))

    std::cout << "Arrhenius equation: k(T) = A·exp(-Ea/(R·T))\n";
    std::cout << "  A  = " << std::scientific << A << " s^-1\n";
    std::cout << "  Ea = " << std::fixed << std::setprecision(0)
              << Ea << " J/mol\n\n";

    // Temperature range: 300K to 600K (room temp to ~325°C)
    const std::size_t nT = 7;
    dense_vector<double> T_kelvin(nT);
    for (std::size_t i = 0; i < nT; ++i) {
        T_kelvin(i) = 300.0 + static_cast<double>(i) * 50.0;
    }

    // k(T) = A · exp(-Ea/(R·T)) — element-wise on the temperature vector
    // Compute -Ea/(R·T) for each temperature
    dense_vector<double> exponent(nT);
    for (std::size_t i = 0; i < nT; ++i) {
        exponent(i) = -Ea / (R * T_kelvin(i));
    }
    auto rate_constants = A * mtl::exp(exponent);

    // Arrhenius plot: ln(k) vs 1000/T should be linear
    dense_vector<double> inv_T(nT);
    for (std::size_t i = 0; i < nT; ++i) {
        inv_T(i) = 1000.0 / T_kelvin(i);
    }
    auto ln_k = mtl::log(rate_constants);

    std::cout << std::setw(8) << "T (K)"
              << std::setw(10) << "1000/T"
              << std::setw(14) << "k (s^-1)"
              << std::setw(14) << "ln(k)"
              << std::setw(16) << "Speedup vs 300K\n";
    std::cout << std::string(62, '-') << "\n";

    for (std::size_t i = 0; i < nT; ++i) {
        double speedup = rate_constants(i) / rate_constants(0);
        std::cout << std::fixed << std::setprecision(0)
                  << std::setw(8) << T_kelvin(i)
                  << std::setprecision(3)
                  << std::setw(10) << inv_T(i)
                  << std::scientific << std::setprecision(4)
                  << std::setw(14) << rate_constants(i)
                  << std::fixed << std::setprecision(4)
                  << std::setw(14) << ln_k(i)
                  << std::scientific << std::setprecision(2)
                  << std::setw(16) << speedup << "\n";
    }

    // Verify Arrhenius linearity: slope of ln(k) vs 1/T = -Ea/R
    double slope = (ln_k(nT - 1) - ln_k(0)) / (1.0 / T_kelvin(nT - 1) - 1.0 / T_kelvin(0));
    double Ea_recovered = -slope * R;
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "\nArrhenius plot slope: " << slope << "\n";
    std::cout << "Recovered Ea: " << Ea_recovered << " J/mol (true: " << Ea << ")\n";
    std::cout << "Error: " << std::scientific << std::abs(Ea_recovered - Ea) << "\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // Part 5: Logarithmic Scales — pH and Decibels
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== Part 5: Logarithmic Scales ===\n\n";

    // pH = -log₁₀([H⁺])
    // Demonstrates log₁₀ compressing a huge dynamic range into manageable numbers.

    dense_vector<double> H_conc = {1.0, 1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-14};
    auto pH = -1.0 * mtl::log10(H_conc);

    std::cout << "pH scale — compressing 14 orders of magnitude:\n\n";
    std::cout << std::setw(14) << "[H+] (mol/L)"
              << std::setw(8) << "pH"
              << std::setw(20) << "Character\n";
    std::cout << std::string(42, '-') << "\n";

    const char* labels[] = {"Strong acid", "Acid", "Weak acid", "Mild acid",
                            "Neutral", "Mild base", "Base", "Strong base"};
    for (std::size_t i = 0; i < H_conc.size(); ++i) {
        std::cout << std::scientific << std::setprecision(1)
                  << std::setw(14) << H_conc(i)
                  << std::fixed << std::setprecision(1)
                  << std::setw(8) << pH(i)
                  << std::setw(20) << labels[i] << "\n";
    }

    // Verify round-trip: 10^(-pH) = [H⁺]
    dense_vector<double> H_recovered(H_conc.size());
    for (std::size_t i = 0; i < H_conc.size(); ++i) {
        H_recovered(i) = std::pow(10.0, -pH(i));
    }
    double max_pH_error = 0.0;
    for (std::size_t i = 0; i < H_conc.size(); ++i) {
        double rel_err = std::abs(H_recovered(i) - H_conc(i)) / H_conc(i);
        max_pH_error = std::max(max_pH_error, rel_err);
    }
    std::cout << "\nRound-trip: max |10^(-pH) - [H+]| / [H+] = "
              << std::scientific << max_pH_error << "\n";

    // ══════════════════════════════════════════════════════════════════════
    // Part 6: Carbon-14 Dating
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "\n=== Part 6: Carbon-14 Dating ===\n\n";

    // age = -(t½/ln2) · ln(N/N₀)
    // Given a measured fraction N/N₀, we invert the decay formula to find age.

    const double half_life_C14 = 5730.0;  // years
    const double lambda_C14 = std::numbers::ln2 / half_life_C14;

    // Simulate artifacts with known fractions of C-14 remaining
    dense_vector<double> fractions = {1.0, 0.75, 0.50, 0.25, 0.10, 0.01};
    auto ages = (-1.0 / lambda_C14) * mtl::log(fractions);

    std::cout << "Dating artifacts from C-14 remaining fraction:\n\n";
    std::cout << std::setw(14) << "N/N₀"
              << std::setw(14) << "Age (years)"
              << std::setw(20) << "Context\n";
    std::cout << std::string(48, '-') << "\n";

    const char* contexts[] = {"Modern", "~2400 ya", "1 half-life",
                              "~11500 ya", "~19000 ya", "~38000 ya"};
    for (std::size_t i = 0; i < fractions.size(); ++i) {
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(14) << fractions(i)
                  << std::setprecision(0)
                  << std::setw(14) << ages(i)
                  << std::setw(20) << contexts[i] << "\n";
    }

    // Verify: age of 50% remaining should be exactly one half-life
    std::cout << std::setprecision(2);
    std::cout << "\nVerification: age at N/N₀ = 0.50 is " << ages(2)
              << " years (should be " << half_life_C14 << ")\n";
    std::cout << "  Error: " << std::scientific << std::abs(ages(2) - half_life_C14) << " years\n";

    // ── Key Takeaways ───────────────────────────────────────────────────
    std::cout << "\n=== Key Takeaways ===\n";
    std::cout << "1. exp(-λt) is the universal solution to first-order decay dN/dt = -λN.\n";
    std::cout << "   Element-wise exp() on a time vector computes entire decay curves.\n";
    std::cout << "2. log() inverts exponential processes, enabling rate extraction from\n";
    std::cout << "   data. The Arrhenius plot (ln k vs 1/T) linearizes exponential\n";
    std::cout << "   temperature dependence to extract activation energies.\n";
    std::cout << "3. log10() compresses enormous dynamic ranges into human-readable\n";
    std::cout << "   scales (pH spans 14 orders of magnitude in just 0-14).\n";
    std::cout << "4. The exp-log round trip preserves values to machine precision,\n";
    std::cout << "   which is essential for scientific computing reliability.\n";
    std::cout << "5. Bateman equations for decay chains combine multiple exp() terms.\n";
    std::cout << "   Conservation laws (total atoms constant) provide built-in checks.\n";

    return EXIT_SUCCESS;
}
