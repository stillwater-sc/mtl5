// phase12d_coordinate_geometry.cpp — Trigonometric & Hyperbolic Geometry
//
// This example demonstrates how trigonometric, hyperbolic, and special
// functions arise naturally in coordinate geometry and physics:
//
//   1. Polar ↔ Cartesian conversion using sin, cos, atan, sqrt
//   2. 2D rotation matrices and angle recovery
//   3. The catenary curve: cosh(x) — shape of a hanging chain
//   4. Normal distribution CDF via erf — probability and statistics
//   5. Complex phasor arithmetic with real, imag, abs, atan
//   6. Rounding and discretization for signal quantization
//
// Mathematics: Trigonometric functions parameterize circles and rotations.
// Hyperbolic functions parameterize hyperbolas and arise in catenary curves,
// special relativity, and thermal physics. The error function connects to
// the Gaussian distribution — the most important distribution in statistics.
//
// MTL5 functions used: sin, cos, tan, asin, acos, atan, sinh, cosh, tanh,
//                      sqrt, pow, abs, exp, erf, erfc, real, imag,
//                      ceil, floor, round, signum

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <numbers>
#include <complex>
#include <cmath>

using namespace mtl;

int main() {
    std::cout << "=============================================================\n";
    std::cout << " Phase 12D: Coordinate Transforms & Classical Curves\n";
    std::cout << "=============================================================\n\n";

    const double pi = std::numbers::pi;

    // ══════════════════════════════════════════════════════════════════════
    // Part 1: Polar ↔ Cartesian Conversion
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== Part 1: Polar to Cartesian Conversion ===\n\n";

    // A set of points on a cardioid: r(θ) = 1 + cos(θ)
    // Converts to Cartesian: x = r·cos(θ), y = r·sin(θ)
    // This demonstrates sin/cos on vectors and the power of element-wise ops.

    const std::size_t np = 12;
    dense_vector<double> theta(np);
    for (std::size_t i = 0; i < np; ++i) {
        theta(i) = static_cast<double>(i) * 2.0 * pi / np;
    }

    // r(θ) = 1 + cos(θ) — cardioid in polar coordinates
    auto cos_theta = mtl::cos(theta);
    dense_vector<double> r(np);
    for (std::size_t i = 0; i < np; ++i) {
        r(i) = 1.0 + cos_theta(i);
    }

    // Convert to Cartesian: x = r·cos(θ), y = r·sin(θ)
    auto sin_theta = mtl::sin(theta);
    dense_vector<double> x(np), y(np);
    for (std::size_t i = 0; i < np; ++i) {
        x(i) = r(i) * cos_theta(i);
        y(i) = r(i) * sin_theta(i);
    }

    std::cout << "Cardioid: r(θ) = 1 + cos(θ)\n\n";
    std::cout << std::setw(8) << "θ/π"
              << std::setw(10) << "r"
              << std::setw(12) << "x"
              << std::setw(12) << "y\n";
    std::cout << std::string(42, '-') << "\n";

    for (std::size_t i = 0; i < np; ++i) {
        std::cout << std::fixed << std::setprecision(4)
                  << std::setw(8) << theta(i) / pi
                  << std::setw(10) << r(i)
                  << std::setw(12) << x(i)
                  << std::setw(12) << y(i) << "\n";
    }

    // Verify round-trip: r_recovered = sqrt(x² + y²)
    auto r_recovered = mtl::sqrt(mtl::pow(x, 2.0) + mtl::pow(y, 2.0));
    double max_r_error = 0.0;
    for (std::size_t i = 0; i < np; ++i) {
        max_r_error = std::max(max_r_error, std::abs(r_recovered(i) - r(i)));
    }
    std::cout << "\nRound-trip: max |sqrt(x²+y²) - r| = "
              << std::scientific << max_r_error << "\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // Part 2: Rotation Matrices and Angle Recovery
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== Part 2: Rotation Matrix ===\n\n";

    // R(θ) = [[cos θ, -sin θ], [sin θ, cos θ]]
    // Applying R to a set of points rotates them.
    // We can recover θ from R using acos and asin.

    const double angle = pi / 6.0;  // 30 degrees
    std::cout << "Rotation angle: π/6 = " << std::fixed << std::setprecision(4)
              << angle << " rad (" << angle * 180.0 / pi << "°)\n\n";

    mat::dense2D<double> Rot(2, 2);
    Rot(0, 0) = std::cos(angle);  Rot(0, 1) = -std::sin(angle);
    Rot(1, 0) = std::sin(angle);  Rot(1, 1) = std::cos(angle);

    std::cout << "R(π/6) = [" << Rot(0,0) << ", " << Rot(0,1) << "]\n";
    std::cout << "         [" << Rot(1,0) << ", " << Rot(1,1) << "]\n\n";

    // Rotate a unit vector along x-axis
    dense_vector<double> px = {1.0, 0.0};
    dense_vector<double> rotated(2);
    rotated(0) = Rot(0, 0) * px(0) + Rot(0, 1) * px(1);
    rotated(1) = Rot(1, 0) * px(0) + Rot(1, 1) * px(1);

    std::cout << "Rotating (1, 0):\n";
    std::cout << "  Result: (" << rotated(0) << ", " << rotated(1) << ")\n";
    std::cout << "  Expected: (cos π/6, sin π/6) = ("
              << std::cos(angle) << ", " << std::sin(angle) << ")\n\n";

    // Recover angle from rotation matrix
    double recovered_acos = std::acos(Rot(0, 0));
    double recovered_asin = std::asin(Rot(1, 0));
    std::cout << "Angle recovery:\n";
    std::cout << "  acos(R₀₀) = " << recovered_acos << " rad (error: "
              << std::scientific << std::abs(recovered_acos - angle) << ")\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "  asin(R₁₀) = " << recovered_asin << " rad (error: "
              << std::scientific << std::abs(recovered_asin - angle) << ")\n";

    // Verify R·R^T = I (orthogonality)
    auto Rt = trans(Rot);
    auto RRt = Rot * Rt;
    double ortho_err = std::abs(RRt(0,0) - 1.0) + std::abs(RRt(0,1))
                     + std::abs(RRt(1,0)) + std::abs(RRt(1,1) - 1.0);
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\nR·Rᵀ = [" << RRt(0,0) << ", " << RRt(0,1) << "]\n";
    std::cout << "        [" << RRt(1,0) << ", " << RRt(1,1) << "]\n";
    std::cout << "Orthogonality error: " << std::scientific << ortho_err << "\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // Part 3: The Catenary — cosh(x) as a Physical Curve
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== Part 3: The Catenary Curve ===\n\n";

    // The catenary y = a·cosh(x/a) is the shape formed by a chain hanging
    // under its own weight. It is NOT a parabola, though it looks like one.
    // The distinction matters in architecture (the Gateway Arch is an
    // inverted catenary, not a parabola, because it distributes load evenly).

    const double a = 2.0;  // catenary parameter (chain tension/weight ratio)
    const std::size_t nc_pts = 11;
    dense_vector<double> xs(nc_pts);
    for (std::size_t i = 0; i < nc_pts; ++i) {
        xs(i) = -3.0 + 6.0 * static_cast<double>(i) / (nc_pts - 1);
    }

    // Catenary: y = a·cosh(x/a)
    auto xs_scaled = (1.0 / a) * xs;
    auto catenary = a * mtl::cosh(xs_scaled);

    // Parabola approximation: y = a + x²/(2a) (Taylor expansion of cosh)
    auto parabola = mtl::pow(xs, 2.0);
    dense_vector<double> para_approx(nc_pts);
    for (std::size_t i = 0; i < nc_pts; ++i) {
        para_approx(i) = a + parabola(i) / (2.0 * a);
    }

    std::cout << "Catenary y = " << a << "·cosh(x/" << a << ") vs parabola y = "
              << a << " + x²/" << 2.0 * a << ":\n\n";
    std::cout << std::setw(8) << "x"
              << std::setw(12) << "Catenary"
              << std::setw(12) << "Parabola"
              << std::setw(12) << "Difference\n";
    std::cout << std::string(44, '-') << "\n";

    for (std::size_t i = 0; i < nc_pts; ++i) {
        double diff = catenary(i) - para_approx(i);
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(8) << xs(i)
                  << std::setprecision(6)
                  << std::setw(12) << catenary(i)
                  << std::setw(12) << para_approx(i)
                  << std::scientific << std::setw(12) << diff << "\n";
    }

    // Verify hyperbolic identity: cosh²(x) - sinh²(x) = 1
    auto cat_cosh = mtl::cosh(xs_scaled);
    auto cat_sinh = mtl::sinh(xs_scaled);
    double max_hyp_err = 0.0;
    for (std::size_t i = 0; i < nc_pts; ++i) {
        double err = std::abs(cat_cosh(i) * cat_cosh(i)
                            - cat_sinh(i) * cat_sinh(i) - 1.0);
        max_hyp_err = std::max(max_hyp_err, err);
    }
    std::cout << "\nIdentity check: max |cosh²-sinh²-1| = "
              << std::scientific << max_hyp_err << "\n";

    // Arc length of catenary: s = a·sinh(x/a)
    // This is one of the rare curves with a closed-form arc length!
    auto arc_len = a * mtl::sinh(xs_scaled);
    std::cout << "\nArc length s(x) = " << a << "·sinh(x/" << a << "):\n";
    std::cout << std::fixed;
    for (std::size_t i = 0; i < nc_pts; i += 2) {
        std::cout << std::setprecision(2)
                  << "  s(" << xs(i) << ") = "
                  << std::setprecision(6) << arc_len(i) << "\n";
    }
    std::cout << "\n";

    // ══════════════════════════════════════════════════════════════════════
    // Part 4: Normal Distribution CDF via erf
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== Part 4: Normal Distribution CDF via erf ===\n\n";

    // Φ(x) = ½[1 + erf(x/√2)]
    // The CDF of the standard normal distribution is built entirely from erf.
    // This connection is why erf appears in statistics, heat diffusion,
    // and quantum mechanics.

    dense_vector<double> z_scores = {-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0};
    const double inv_sqrt2 = 1.0 / std::sqrt(2.0);
    auto erf_vals = mtl::erf(inv_sqrt2 * z_scores);

    dense_vector<double> cdf(z_scores.size());
    for (std::size_t i = 0; i < z_scores.size(); ++i) {
        cdf(i) = 0.5 * (1.0 + erf_vals(i));
    }

    std::cout << "Standard normal CDF: Φ(z) = ½[1 + erf(z/√2)]\n\n";
    std::cout << std::setw(8) << "z"
              << std::setw(14) << "Φ(z)"
              << std::setw(14) << "1 - Φ(z)"
              << std::setw(18) << "Interpretation\n";
    std::cout << std::string(54, '-') << "\n";

    const char* interp[] = {
        "0.13% below", "2.28% below", "15.87% below",
        "30.85% below", "50% (median)", "69.15% below",
        "84.13% below", "97.72% below", "99.87% below"
    };
    for (std::size_t i = 0; i < z_scores.size(); ++i) {
        std::cout << std::fixed << std::setprecision(1)
                  << std::setw(8) << z_scores(i)
                  << std::setprecision(6)
                  << std::setw(14) << cdf(i)
                  << std::setw(14) << 1.0 - cdf(i)
                  << std::setw(18) << interp[i] << "\n";
    }

    // Verify symmetry: Φ(z) + Φ(-z) = 1  ⟺  erf(x) + erfc(x) = 1
    auto erfc_vals = mtl::erfc(inv_sqrt2 * z_scores);
    double max_sym_err = 0.0;
    for (std::size_t i = 0; i < z_scores.size(); ++i) {
        double err = std::abs(erf_vals(i) + erfc_vals(i) - 1.0);
        max_sym_err = std::max(max_sym_err, err);
    }
    std::cout << "\nIdentity check: max |erf(x) + erfc(x) - 1| = "
              << std::scientific << max_sym_err << "\n";

    // The 68-95-99.7 rule
    std::cout << std::fixed << std::setprecision(4);
    double within_1sigma = cdf(6) - cdf(2);  // Φ(1) - Φ(-1)
    double within_2sigma = cdf(7) - cdf(1);  // Φ(2) - Φ(-2)
    double within_3sigma = cdf(8) - cdf(0);  // Φ(3) - Φ(-3)
    std::cout << "\nThe 68-95-99.7 rule:\n";
    std::cout << "  P(|z| < 1) = " << 100.0 * within_1sigma << "% (expect ~68.27%)\n";
    std::cout << "  P(|z| < 2) = " << 100.0 * within_2sigma << "% (expect ~95.45%)\n";
    std::cout << "  P(|z| < 3) = " << 100.0 * within_3sigma << "% (expect ~99.73%)\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // Part 5: Complex Phasors — real, imag, abs, atan
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== Part 5: Complex Phasor Decomposition ===\n\n";

    // In electrical engineering, AC circuits use phasors: V = V₀ · e^(jφ)
    // real() and imag() extract in-phase and quadrature components.

    using cd = std::complex<double>;
    dense_vector<cd> phasors = {
        cd(5.0, 0.0),    // 5∠0°  — purely resistive
        cd(0.0, 3.0),    // 3∠90° — purely inductive
        cd(4.0, 3.0),    // 5∠36.87° — RL circuit
        cd(-2.0, 2.0),   // 2√2∠135° — leading
    };

    auto re = real(phasors);
    auto im = imag(phasors);
    auto mag = abs(phasors);

    std::cout << "Phasor decomposition (voltage phasors in AC circuits):\n\n";
    std::cout << std::setw(16) << "Phasor"
              << std::setw(10) << "Real"
              << std::setw(10) << "Imag"
              << std::setw(12) << "|V|"
              << std::setw(16) << "Circuit type\n";
    std::cout << std::string(64, '-') << "\n";

    const char* circuit[] = {"Resistive", "Inductive", "RL (lagging)", "Leading"};
    for (std::size_t i = 0; i < phasors.size(); ++i) {
        std::cout << std::fixed << std::setprecision(2)
                  << "  " << std::setw(6) << phasors(i).real()
                  << " + " << std::setw(5) << phasors(i).imag() << "j"
                  << std::setw(10) << re(i)
                  << std::setw(10) << im(i)
                  << std::setw(12) << mag(i)
                  << std::setw(16) << circuit[i] << "\n";
    }

    // Verify: |V|² = real² + imag²
    double max_phasor_err = 0.0;
    for (std::size_t i = 0; i < phasors.size(); ++i) {
        double err = std::abs(mag(i) * mag(i) - re(i) * re(i) - im(i) * im(i));
        max_phasor_err = std::max(max_phasor_err, err);
    }
    std::cout << "\nIdentity: max ||V|² - Re² - Im²| = "
              << std::scientific << max_phasor_err << "\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // Part 6: Signal Quantization — ceil, floor, round, signum
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== Part 6: Signal Quantization ===\n\n";

    // In digital signal processing, continuous signals must be quantized
    // to discrete levels. Different rounding strategies produce different
    // quantization errors and biases.

    dense_vector<double> signal = {-2.7, -1.5, -0.3, 0.0, 0.3, 1.5, 2.7};
    auto q_ceil  = mtl::ceil(signal);
    auto q_floor = mtl::floor(signal);
    auto q_round = mtl::round(signal);
    auto q_sign  = signum(signal);

    std::cout << "Comparing quantization strategies:\n\n";
    std::cout << std::setw(10) << "Signal"
              << std::setw(10) << "ceil"
              << std::setw(10) << "floor"
              << std::setw(10) << "round"
              << std::setw(10) << "signum\n";
    std::cout << std::string(50, '-') << "\n";

    for (std::size_t i = 0; i < signal.size(); ++i) {
        std::cout << std::fixed << std::setprecision(1)
                  << std::setw(10) << signal(i)
                  << std::setw(10) << q_ceil(i)
                  << std::setw(10) << q_floor(i)
                  << std::setw(10) << q_round(i)
                  << std::setw(10) << q_sign(i) << "\n";
    }

    // Compute quantization error (bias) for each strategy
    double bias_ceil = 0.0, bias_floor = 0.0, bias_round = 0.0;
    for (std::size_t i = 0; i < signal.size(); ++i) {
        bias_ceil  += q_ceil(i) - signal(i);
        bias_floor += q_floor(i) - signal(i);
        bias_round += q_round(i) - signal(i);
    }
    auto n_sig = static_cast<double>(signal.size());

    std::cout << std::setprecision(4);
    std::cout << "\nMean quantization error (bias):\n";
    std::cout << "  ceil:  " << std::setw(8) << bias_ceil / n_sig
              << " (always rounds up → positive bias)\n";
    std::cout << "  floor: " << std::setw(8) << bias_floor / n_sig
              << " (always rounds down → negative bias)\n";
    std::cout << "  round: " << std::setw(8) << bias_round / n_sig
              << " (rounds to nearest → minimal bias)\n";

    // ── Key Takeaways ───────────────────────────────────────────────────
    std::cout << "\n=== Key Takeaways ===\n";
    std::cout << "1. sin/cos parameterize circles. Converting polar to Cartesian\n";
    std::cout << "   coordinates is just element-wise sin() and cos() on angle vectors.\n";
    std::cout << "   The inverse functions (asin, acos, atan) recover angles from\n";
    std::cout << "   coordinates — essential for rotation matrix decomposition.\n";
    std::cout << "2. cosh(x) is the catenary — a fundamental curve in architecture\n";
    std::cout << "   and physics. It looks like a parabola near x=0 but diverges\n";
    std::cout << "   exponentially. sinh(x) gives its arc length in closed form.\n";
    std::cout << "3. The error function erf() builds the normal CDF: Φ(z) = ½[1+erf(z/√2)].\n";
    std::cout << "   This single function encodes the 68-95-99.7 rule and all\n";
    std::cout << "   Gaussian probability calculations.\n";
    std::cout << "4. real()/imag() decompose complex phasors into in-phase and\n";
    std::cout << "   quadrature components — the foundation of AC circuit analysis.\n";
    std::cout << "5. ceil/floor/round have different bias characteristics. round()\n";
    std::cout << "   minimizes mean error; ceil/floor introduce systematic bias.\n";
    std::cout << "   Understanding this matters for ADC design and financial rounding.\n";

    return EXIT_SUCCESS;
}
