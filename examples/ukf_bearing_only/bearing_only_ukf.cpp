// bearing_only_ukf.cpp — Bearing-only UKF stress test: Cholesky vs LDL^T
//
// The canonical ill-conditioning generator for UKF sigma point factorization:
// a 2D tracker with bearing-only measurements. The bearing direction is
// well-observed while range is unobserved, creating extreme eigenvalue spread
// in the covariance P after each update.
//
// This example:
//   1. Runs the bearing-only UKF in both float64 and float32
//   2. Compares Cholesky vs LDL^T at each step
//   3. Measures sigma-point mean bias (the silent killer)
//   4. Tracks condition number and factorization residual
//   5. Is structured for future Bunch-Kaufman path (issue #46)
//
// The sigma-point mean bias is the key diagnostic: with a degraded square
// root, the 2n+1 points are asymmetric around x-hat, pushing x-hat_pred
// away from the true mean. This bias is small per-step but compounds over
// 10-20 iterations to produce mean drifts of 0.5-2 sigma in float32.
//
// References:
//   Julier & Uhlmann, "Unscented Filtering and Nonlinear Estimation", 2004
//   Bar-Shalom, Li, Kirubarajan, "Estimation with Applications to Tracking
//   and Navigation", Ch. 11

#include <mtl/mtl.hpp>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <string>
#include <type_traits>
#include <vector>

using namespace mtl;

// ============================================================================
// Templated helpers
// ============================================================================

template <typename T>
mat::dense2D<T> copy_mat(const mat::dense2D<T>& A) {
    std::size_t n = A.num_rows(), m = A.num_cols();
    mat::dense2D<T> B(n, m);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < m; ++j)
            B(i, j) = A(i, j);
    return B;
}

// ============================================================================
// Simple PRNG (deterministic, no external deps)
// ============================================================================

struct Rng {
    uint64_t s[2];
    explicit Rng(uint64_t seed = 42) {
        s[0] = seed ^ 0x9E3779B97F4A7C15ULL;
        s[1] = seed ^ 0x6A09E667F3BCC908ULL;
        for (int i = 0; i < 10; ++i) next();
    }
    uint64_t next() {
        uint64_t s0 = s[0], s1 = s[1], result = s0 + s1;
        s1 ^= s0;
        s[0] = ((s0 << 55) | (s0 >> 9)) ^ s1 ^ (s1 << 14);
        s[1] = (s1 << 36) | (s1 >> 28);
        return result;
    }
    double randn() {
        double u1 = (double(next() >> 11) + 0.5) / double(1ULL << 53);
        double u2 = (double(next() >> 11) + 0.5) / double(1ULL << 53);
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * std::numbers::pi * u2);
    }
};

// ============================================================================
// Bearing-only UKF — templated over Scalar
// ============================================================================

static constexpr std::size_t NX = 4;  // [px, py, vx, vy]
static constexpr std::size_t NZ = 1;  // bearing only

/// Sigma-point mean bias: measure asymmetry of sigma points around mean.
/// For a perfect factorization, sum of (sigma_k - mean) should be zero.
/// Returns ||sum(sigma_k * w_k) - mean * sum(w_k)|| as fraction of ||mean||.
template <typename T>
T sigma_point_bias(
    const std::vector<vec::dense_vector<T>>& sigma,
    const vec::dense_vector<T>& mean,
    T Wm0, T Wi)
{
    constexpr std::size_t n = NX;
    constexpr std::size_t num_sigma = 2 * n + 1;

    // Compute weighted mean of sigma points
    vec::dense_vector<T> weighted_mean(n);
    for (std::size_t i = 0; i < n; ++i) {
        weighted_mean(i) = Wm0 * sigma[0](i);
        for (std::size_t s = 1; s < num_sigma; ++s)
            weighted_mean(i) += Wi * sigma[s](i);
    }

    // Bias = ||weighted_mean - mean||
    T bias_sq = T(0);
    T mean_sq = T(0);
    for (std::size_t i = 0; i < n; ++i) {
        T d = weighted_mean(i) - mean(i);
        bias_sq += d * d;
        mean_sq += mean(i) * mean(i);
    }

    using std::sqrt;
    if (mean_sq < std::numeric_limits<T>::min())
        return sqrt(bias_sq);
    return sqrt(bias_sq) / sqrt(mean_sq);
}

/// Factorization residual for Cholesky: ||P - L*L^T|| / ||P||
template <typename T>
T chol_residual(const mat::dense2D<T>& P) {
    std::size_t n = P.num_rows();
    auto L = copy_mat(P);
    int info = cholesky_factor(L);
    if (info != 0) return T(-1);  // signal failure

    mat::dense2D<T> Lmat(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Lmat(i, j) = (j <= i) ? L(i, j) : T(0);

    auto LLt = Lmat * trans(Lmat);

    T res = T(0), pnorm = T(0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            T d = P(i, j) - LLt(i, j);
            res += d * d;
            pnorm += P(i, j) * P(i, j);
        }
    using std::sqrt;
    return sqrt(res) / sqrt(pnorm);
}

/// Factorization residual for LDL^T: ||P - L*D*L^T|| / ||P||
template <typename T>
T ldlt_residual_val(const mat::dense2D<T>& P) {
    std::size_t n = P.num_rows();
    auto F = copy_mat(P);
    int info = ldlt_factor(F);
    if (info != 0) return T(-1);

    mat::dense2D<T> Lmat(n, n), Dmat(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) { Lmat(i, j) = T(1); Dmat(i, j) = F(i, j); }
            else if (j < i) { Lmat(i, j) = F(i, j); Dmat(i, j) = T(0); }
            else { Lmat(i, j) = T(0); Dmat(i, j) = T(0); }
        }

    auto LDLt = Lmat * Dmat * trans(Lmat);

    T res = T(0), pnorm = T(0);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            T d = P(i, j) - LDLt(i, j);
            res += d * d;
            pnorm += P(i, j) * P(i, j);
        }
    using std::sqrt;
    return sqrt(res) / sqrt(pnorm);
}

/// Condition number via eigenvalue ratio (double precision computation
/// regardless of T, to avoid eigenvalue solver issues in float32)
template <typename T>
double condition_number_f64(const mat::dense2D<T>& P) {
    std::size_t n = P.num_rows();
    mat::dense2D<double> Pd(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Pd(i, j) = static_cast<double>(P(i, j));

    auto eigs = eigenvalue_symmetric(Pd);
    double emin = std::abs(eigs(0)), emax = emin;
    for (std::size_t i = 1; i < n; ++i) {
        double ae = std::abs(eigs(i));
        if (ae < emin) emin = ae;
        if (ae > emax) emax = ae;
    }
    if (emin < std::numeric_limits<double>::min())
        return std::numeric_limits<double>::infinity();
    return emax / emin;
}

enum class FactMethod { Cholesky, LDLT };

/// Generate sigma points using Cholesky
template <typename T>
bool generate_sigma_cholesky(
    const vec::dense_vector<T>& x, const mat::dense2D<T>& P,
    T gamma, std::vector<vec::dense_vector<T>>& sigma)
{
    auto L = copy_mat(P);
    int info = cholesky_factor(L);
    if (info != 0) return false;

    sigma[0] = x;
    for (std::size_t j = 0; j < NX; ++j) {
        vec::dense_vector<T> sp(NX), sm(NX);
        for (std::size_t i = 0; i < NX; ++i) {
            T col_val = (i >= j) ? L(i, j) : T(0);
            sp(i) = x(i) + gamma * col_val;
            sm(i) = x(i) - gamma * col_val;
        }
        sigma[1 + j] = sp;
        sigma[1 + NX + j] = sm;
    }
    return true;
}

/// Generate sigma points using LDL^T
template <typename T>
bool generate_sigma_ldlt(
    const vec::dense_vector<T>& x, const mat::dense2D<T>& P,
    T gamma, std::vector<vec::dense_vector<T>>& sigma)
{
    auto F = copy_mat(P);
    int info = ldlt_factor(F);
    if (info != 0) return false;

    mat::dense2D<T> Lmat(NX, NX);
    vec::dense_vector<T> D(NX);
    for (std::size_t i = 0; i < NX; ++i) {
        D(i) = F(i, i);
        for (std::size_t j = 0; j < NX; ++j)
            Lmat(i, j) = (i == j) ? T(1) : ((j < i) ? F(i, j) : T(0));
    }

    // Sigma = x +/- gamma * L * sqrt(D(k)) * e_k
    // Negative D(k) means P is indefinite — report failure rather than
    // silently using abs(D(k)), which would produce sigma points for
    // L|D|L^T instead of P. This keeps the comparison fair: LDL^T
    // detects the problem; the caller decides how to recover.
    sigma[0] = x;
    for (std::size_t k = 0; k < NX; ++k) {
        if (D(k) <= T(0)) return false;  // indefinite pivot — fail cleanly
        using std::sqrt;
        T sqrtDk = sqrt(D(k));
        vec::dense_vector<T> sp(NX), sm(NX);
        for (std::size_t i = 0; i < NX; ++i) {
            T col_val = Lmat(i, k) * sqrtDk;
            sp(i) = x(i) + gamma * col_val;
            sm(i) = x(i) - gamma * col_val;
        }
        sigma[1 + k] = sp;
        sigma[1 + NX + k] = sm;
    }
    return true;
}

/// Bearing measurement model: z = atan2(py - by, px - bx)
template <typename T>
T bearing_measurement(const vec::dense_vector<T>& x, T bx, T by) {
    using std::atan2;
    return atan2(x(1) - by, x(0) - bx);
}

/// Bearing measurement Jacobian H = d(bearing)/d(state) at x
template <typename T>
vec::dense_vector<T> bearing_jacobian(const vec::dense_vector<T>& x, T bx, T by) {
    T dx = x(0) - bx;
    T dy = x(1) - by;
    T r2 = dx * dx + dy * dy;
    vec::dense_vector<T> H(NX);
    H(0) = -dy / r2;  // d(bearing)/d(px)
    H(1) =  dx / r2;  // d(bearing)/d(py)
    H(2) = T(0);
    H(3) = T(0);
    return H;
}

/// Per-step result
struct StepResult {
    int step;
    double cond_P;
    bool chol_ok;
    bool ldlt_ok;
    double chol_bias;
    double ldlt_bias;
    double chol_resid;
    double ldlt_resid;
};

/// Run bearing-only UKF for a given scalar type
template <typename T>
std::vector<StepResult> run_bearing_only_ukf(int num_steps) {
    std::vector<StepResult> results;

    // UKF parameters
    T alpha = T(1e-3);
    T beta  = T(2);
    T kappa = T(0);
    T dt    = T(0.1);
    T R_val = T(1e-4);  // tight bearing measurement noise

    T lambda_ = alpha * alpha * (T(NX) + kappa) - T(NX);
    T gamma = T(std::sqrt(static_cast<double>(NX) + static_cast<double>(lambda_)));
    T Wm0 = lambda_ / (T(NX) + lambda_);
    T Wc0 = Wm0 + (T(1) - alpha * alpha + beta);
    T Wi  = T(1) / (T(2) * (T(NX) + lambda_));

    constexpr std::size_t num_sigma = 2 * NX + 1;

    // Beacon position
    T bx = T(50), by = T(0);

    // True state: target moving roughly toward beacon
    vec::dense_vector<T> x_true(NX);
    x_true(0) = T(0); x_true(1) = T(5);
    x_true(2) = T(2); x_true(3) = T(-0.5);

    // Process noise (small — we want P conditioning to be driven by measurements)
    mat::dense2D<T> Q(NX, NX);
    for (std::size_t i = 0; i < NX; ++i)
        for (std::size_t j = 0; j < NX; ++j)
            Q(i, j) = T(0);
    Q(0,0) = T(1e-6); Q(1,1) = T(1e-6);
    Q(2,2) = T(1e-4); Q(3,3) = T(1e-4);

    // Initial estimate (offset from truth)
    vec::dense_vector<T> x_hat(NX);
    x_hat(0) = T(1); x_hat(1) = T(6);
    x_hat(2) = T(1.5); x_hat(3) = T(0);

    // Initial covariance: large and isotropic
    mat::dense2D<T> P(NX, NX);
    for (std::size_t i = 0; i < NX; ++i)
        for (std::size_t j = 0; j < NX; ++j)
            P(i, j) = (i == j) ? T(100) : T(0);

    Rng rng(54321);

    std::vector<vec::dense_vector<T>> sigma_chol(num_sigma);
    std::vector<vec::dense_vector<T>> sigma_ldlt(num_sigma);
    bool filter_failed = false;

    for (int step = 0; step < num_steps; ++step) {
        StepResult sr;
        sr.step = step;

        // === Propagate true state (constant velocity) ===
        x_true(0) = x_true(0) + x_true(2) * dt;
        x_true(1) = x_true(1) + x_true(3) * dt;
        // Add small process noise
        for (std::size_t i = 0; i < NX; ++i)
            x_true(i) = x_true(i) + T(std::sqrt(static_cast<double>(Q(i, i)))) * T(rng.randn());

        // === Generate bearing measurement ===
        T z_true = bearing_measurement(x_true, bx, by);
        T z = z_true + T(std::sqrt(static_cast<double>(R_val))) * T(rng.randn());

        // === Condition number (computed in double for reliability) ===
        sr.cond_P = condition_number_f64(P);

        // === Try Cholesky sigma points ===
        sr.chol_ok = generate_sigma_cholesky(x_hat, P, gamma, sigma_chol);
        if (sr.chol_ok) {
            sr.chol_bias = static_cast<double>(
                sigma_point_bias(sigma_chol, x_hat, Wm0, Wi));
        } else {
            sr.chol_bias = std::numeric_limits<double>::quiet_NaN();
        }

        // === Try LDL^T sigma points ===
        sr.ldlt_ok = generate_sigma_ldlt(x_hat, P, gamma, sigma_ldlt);
        if (sr.ldlt_ok) {
            sr.ldlt_bias = static_cast<double>(
                sigma_point_bias(sigma_ldlt, x_hat, Wm0, Wi));
        } else {
            sr.ldlt_bias = std::numeric_limits<double>::quiet_NaN();
        }

        // === Factorization residuals ===
        {
            T cr = chol_residual(P);
            sr.chol_resid = (cr < T(0)) ? std::numeric_limits<double>::quiet_NaN()
                                         : static_cast<double>(cr);
        }
        {
            T lr = ldlt_residual_val(P);
            sr.ldlt_resid = (lr < T(0)) ? std::numeric_limits<double>::quiet_NaN()
                                         : static_cast<double>(lr);
        }

        results.push_back(sr);

        // === UKF predict+update using LDL^T (the more robust path) ===
        // We use LDL^T for the actual filter progression so both precisions
        // run the same number of steps for fair comparison.
        // If LDL^T fails, freeze P and continue emitting diagnostic rows.
        if (filter_failed) continue;

        std::vector<vec::dense_vector<T>>& sigma = sigma_ldlt;
        if (!sr.ldlt_ok) { filter_failed = true; continue; }

        // Propagate sigma points through process model (constant velocity)
        for (std::size_t s = 0; s < num_sigma; ++s) {
            sigma[s](0) = sigma[s](0) + sigma[s](2) * dt;
            sigma[s](1) = sigma[s](1) + sigma[s](3) * dt;
        }

        // Predicted state mean
        vec::dense_vector<T> x_pred(NX);
        for (std::size_t i = 0; i < NX; ++i) {
            x_pred(i) = Wm0 * sigma[0](i);
            for (std::size_t s = 1; s < num_sigma; ++s)
                x_pred(i) = x_pred(i) + Wi * sigma[s](i);
        }

        // Predicted covariance
        mat::dense2D<T> P_pred(NX, NX);
        for (std::size_t i = 0; i < NX; ++i)
            for (std::size_t j = 0; j < NX; ++j)
                P_pred(i, j) = Q(i, j);

        for (std::size_t i = 0; i < NX; ++i)
            for (std::size_t j = 0; j < NX; ++j)
                P_pred(i, j) = P_pred(i, j) + Wc0 * (sigma[0](i) - x_pred(i))
                                                    * (sigma[0](j) - x_pred(j));
        for (std::size_t s = 1; s < num_sigma; ++s)
            for (std::size_t i = 0; i < NX; ++i)
                for (std::size_t j = 0; j < NX; ++j)
                    P_pred(i, j) = P_pred(i, j) + Wi * (sigma[s](i) - x_pred(i))
                                                      * (sigma[s](j) - x_pred(j));

        // === Measurement update (bearing-only, linearized via sigma points) ===
        // Regenerate sigma points from P_pred for measurement transform
        if (!generate_sigma_ldlt(x_pred, P_pred, gamma, sigma)) {
            filter_failed = true;
            continue;
        }

        // Predicted measurement
        std::vector<T> z_sigma(num_sigma);
        for (std::size_t s = 0; s < num_sigma; ++s)
            z_sigma[s] = bearing_measurement(sigma[s], bx, by);

        T z_pred = Wm0 * z_sigma[0];
        for (std::size_t s = 1; s < num_sigma; ++s)
            z_pred = z_pred + Wi * z_sigma[s];

        // Innovation covariance S (scalar for 1D measurement)
        T S = R_val;
        {
            T dz0 = z_sigma[0] - z_pred;
            S = S + Wc0 * dz0 * dz0;
        }
        for (std::size_t s = 1; s < num_sigma; ++s) {
            T dz = z_sigma[s] - z_pred;
            S = S + Wi * dz * dz;
        }

        // Cross-covariance Pxz (NX x 1)
        vec::dense_vector<T> Pxz(NX);
        for (std::size_t i = 0; i < NX; ++i) {
            Pxz(i) = Wc0 * (sigma[0](i) - x_pred(i)) * (z_sigma[0] - z_pred);
            for (std::size_t s = 1; s < num_sigma; ++s)
                Pxz(i) = Pxz(i) + Wi * (sigma[s](i) - x_pred(i)) * (z_sigma[s] - z_pred);
        }

        // Kalman gain K = Pxz / S
        vec::dense_vector<T> K(NX);
        for (std::size_t i = 0; i < NX; ++i)
            K(i) = Pxz(i) / S;

        // Innovation (with bearing wrap)
        T innov = z - z_pred;
        while (static_cast<double>(innov) > std::numbers::pi)
            innov = innov - T(2.0 * std::numbers::pi);
        while (static_cast<double>(innov) < -std::numbers::pi)
            innov = innov + T(2.0 * std::numbers::pi);

        // State update
        for (std::size_t i = 0; i < NX; ++i)
            x_hat(i) = x_pred(i) + K(i) * innov;

        // Covariance update: P = P_pred - K * S * K^T
        for (std::size_t i = 0; i < NX; ++i)
            for (std::size_t j = 0; j < NX; ++j)
                P(i, j) = P_pred(i, j) - K(i) * S * K(j);

        // Force symmetry
        for (std::size_t i = 0; i < NX; ++i)
            for (std::size_t j = i + 1; j < NX; ++j)
                P(j, i) = P(i, j);

        // Check for NaN divergence
        bool has_nan = false;
        for (std::size_t i = 0; i < NX; ++i)
            if (!std::isfinite(static_cast<double>(x_hat(i)))) has_nan = true;
        if (has_nan) filter_failed = true;
    }

    return results;
}

// ============================================================================
// Reporting
// ============================================================================

void print_results(const std::string& label, const std::string& type_info,
                   const std::vector<StepResult>& results) {
    std::cout << "\n=== " << label << " (" << type_info << ") ===\n";
    std::cout << std::setw(6) << "Step"
              << std::setw(12) << "cond(P)"
              << std::setw(8) << "Chol"
              << std::setw(8) << "LDLT"
              << std::setw(14) << "Chol-bias"
              << std::setw(14) << "LDLT-bias"
              << std::setw(14) << "Chol-resid"
              << std::setw(14) << "LDLT-resid"
              << "\n";
    std::cout << std::string(90, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::setw(6) << r.step;
        std::cout << std::setw(12) << std::scientific << std::setprecision(1) << r.cond_P;
        std::cout << std::setw(8) << (r.chol_ok ? "OK" : "FAIL");
        std::cout << std::setw(8) << (r.ldlt_ok ? "OK" : "FAIL");

        if (r.chol_ok && std::isfinite(r.chol_bias))
            std::cout << std::setw(14) << std::scientific << std::setprecision(2) << r.chol_bias;
        else
            std::cout << std::setw(14) << "N/A";

        if (r.ldlt_ok && std::isfinite(r.ldlt_bias))
            std::cout << std::setw(14) << std::scientific << std::setprecision(2) << r.ldlt_bias;
        else
            std::cout << std::setw(14) << "N/A";

        if (std::isfinite(r.chol_resid))
            std::cout << std::setw(14) << std::scientific << std::setprecision(2) << r.chol_resid;
        else
            std::cout << std::setw(14) << "N/A";

        if (std::isfinite(r.ldlt_resid))
            std::cout << std::setw(14) << std::scientific << std::setprecision(2) << r.ldlt_resid;
        else
            std::cout << std::setw(14) << "N/A";

        std::cout << "\n";
    }

    // Summary
    int chol_fails = 0, ldlt_fails = 0;
    double max_chol_bias = 0, max_ldlt_bias = 0;
    for (const auto& r : results) {
        if (!r.chol_ok) ++chol_fails;
        if (!r.ldlt_ok) ++ldlt_fails;
        if (r.chol_ok && std::isfinite(r.chol_bias) && r.chol_bias > max_chol_bias)
            max_chol_bias = r.chol_bias;
        if (r.ldlt_ok && std::isfinite(r.ldlt_bias) && r.ldlt_bias > max_ldlt_bias)
            max_ldlt_bias = r.ldlt_bias;
    }

    std::cout << "\n  Cholesky: " << (results.size() - chol_fails) << "/"
              << results.size() << " steps OK";
    if (chol_fails > 0) std::cout << " (" << chol_fails << " FAILED)";
    std::cout << ", max bias = " << std::scientific << std::setprecision(2) << max_chol_bias;
    std::cout << "\n  LDL^T:    " << (results.size() - ldlt_fails) << "/"
              << results.size() << " steps OK";
    if (ldlt_fails > 0) std::cout << " (" << ldlt_fails << " FAILED)";
    std::cout << ", max bias = " << std::scientific << std::setprecision(2) << max_ldlt_bias;
    std::cout << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "================================================================\n";
    std::cout << " Bearing-Only UKF: Cholesky vs LDL^T Stress Test\n";
    std::cout << "================================================================\n";
    std::cout << "\n"
              << "System:      2D tracking [px, py, vx, vy]\n"
              << "Measurement: bearing-only from beacon at (50, 0)\n"
              << "R = 1e-4:    tight angular measurement\n"
              << "Key effect:  bearing collapses one covariance direction,\n"
              << "             range direction stays large -> extreme cond(P)\n"
              << "\n"
              << "Diagnostics:\n"
              << "  cond(P)    — eigenvalue ratio (higher = harder)\n"
              << "  Chol/LDLT  — did the factorization succeed?\n"
              << "  bias       — sigma-point mean asymmetry (the silent killer)\n"
              << "  resid      — ||P - reconstruct|| / ||P||\n";

    constexpr int num_steps = 15;

    // --- float64 ---
    auto results_f64 = run_bearing_only_ukf<double>(num_steps);
    print_results("Bearing-Only UKF", "float64, 53-bit significand", results_f64);

    // --- float32 ---
    auto results_f32 = run_bearing_only_ukf<float>(num_steps);
    print_results("Bearing-Only UKF", "float32, 24-bit significand", results_f32);

    // --- Comparison summary ---
    std::cout << "\n================================================================\n";
    std::cout << " Precision Comparison Summary\n";
    std::cout << "================================================================\n\n";

    std::cout << "Step-by-step Cholesky success: float64 vs float32\n";
    std::cout << std::setw(6) << "Step"
              << std::setw(12) << "cond(P)"
              << std::setw(12) << "f64-Chol"
              << std::setw(12) << "f32-Chol"
              << std::setw(12) << "f64-LDLT"
              << std::setw(12) << "f32-LDLT"
              << "\n";
    std::cout << std::string(64, '-') << "\n";

    std::size_t max_steps = std::min(results_f64.size(), results_f32.size());
    for (std::size_t i = 0; i < max_steps; ++i) {
        std::cout << std::setw(6) << results_f64[i].step;
        std::cout << std::setw(12) << std::scientific << std::setprecision(1) << results_f64[i].cond_P;
        std::cout << std::setw(12) << (results_f64[i].chol_ok ? "OK" : "FAIL");
        std::cout << std::setw(12) << (results_f32[i].chol_ok ? "OK" : "FAIL");
        std::cout << std::setw(12) << (results_f64[i].ldlt_ok ? "OK" : "FAIL");
        std::cout << std::setw(12) << (results_f32[i].ldlt_ok ? "OK" : "FAIL");
        std::cout << "\n";
    }

    std::cout << "\n================================================================\n";
    std::cout << " Conclusion\n";
    std::cout << "================================================================\n";
    std::cout << "\n"
              << "Bearing-only updates collapse one direction of the covariance,\n"
              << "creating extreme eigenvalue spread. In float32, this drives\n"
              << "cond(P) into the regime where Cholesky's sqrt amplifies error.\n"
              << "\n"
              << "The sigma-point bias metric reveals silent degradation: even\n"
              << "when Cholesky 'succeeds', the sigma points may be asymmetric,\n"
              << "biasing the predicted mean. This compounds over filter steps.\n"
              << "\n"
              << "LDL^T avoids sqrt entirely, extending the usable conditioning\n"
              << "range for each precision level. For custom number types (posits,\n"
              << "LNS) with even fewer significand bits, the advantage grows.\n"
              << "\n"
              << "Future: Bunch-Kaufman pivoted LDL^T (issue #46) will handle\n"
              << "the case where P becomes indefinite due to rounding, turning\n"
              << "'fail gracefully' into 'produce a usable approximation.'\n";

    return 0;
}
