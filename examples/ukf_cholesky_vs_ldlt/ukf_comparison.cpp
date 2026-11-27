// ukf_comparison.cpp -- UKF sigma points: Cholesky against LDL^T
//
// The Unscented Kalman Filter takes its sigma points from a square root of the
// state covariance P. This example runs one UKF with Cholesky (P = L L^T) and
// one with LDL^T (P = L D L^T, the columns L(:, k) sqrt(D_k)) on the same
// scenarios, and factors test matrices directly, to compare the two.
//
// System model:
//   State: [px, py, vx, vy]^T -- 2D position + velocity
//   Process: constant velocity with quadratic drag
//   Measurement: range + bearing from a known beacon
//
// What it shows (issue 953; the conclusion at the end is computed from the
// runs):
//   - On an SPD covariance the two are equally accurate, at any condition
//     number: the filters give the same final error in every scenario, and
//     both factorizations' residuals stay at rounding level up to cond 1e15.
//     When Cholesky completes, its backward error is small at any condition
//     number (Higham, "Accuracy and Stability of Numerical Algorithms", Thm
//     10.3). Completion is guaranteed for c n^(3/2) cond(P) eps < 1; beyond
//     it a rounded pivot can come out non-positive and the factorization
//     stops, as LDL^T's D gets an entry <= 0. Neither fails silently. A
//     scalar square root is perfectly conditioned: a relative error e in its
//     argument becomes e/2. LDL^T does not avoid the square root either: its sigma points use
//     sqrt(D_k).
//   - A tiny measurement noise does not make P ill-conditioned here: the
//     process noise Q, added at every predict step, keeps its eigenvalues up.
//   - The difference is in kind, on an indefinite P: Cholesky stops, while
//     LDL^T's D reports the inertia. Unpivoted LDL^T is not stable on an
//     indefinite matrix, though; Bunch-Kaufman (ldlt_bk) is.
// The float32 side, and the UKF weights that decide it, are in
// ../ukf_bearing_only/bearing_only_ukf.cpp.
//
// Reference:
//   Julier & Uhlmann, "Unscented Filtering and Nonlinear Estimation",
//   Proceedings of the IEEE, 92(3), 2004.

#include <mtl/mtl.hpp>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

using namespace mtl;

// ============================================================================
// Helpers
// ============================================================================

static mat::dense2D<double> copy_matrix(const mat::dense2D<double>& A) {
    std::size_t n = A.num_rows(), m = A.num_cols();
    mat::dense2D<double> B(n, m);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < m; ++j)
            B(i, j) = A(i, j);
    return B;
}

/// Condition number via eigenvalues of a symmetric matrix
static double condition_number(const mat::dense2D<double>& P) {
    std::size_t n = P.num_rows();
    auto Pcopy = copy_matrix(P);
    auto eigs = eigenvalue_symmetric(Pcopy);
    double emin = std::abs(eigs(0)), emax = std::abs(eigs(0));
    for (std::size_t i = 1; i < n; ++i) {
        double ae = std::abs(eigs(i));
        if (ae < emin) emin = ae;
        if (ae > emax) emax = ae;
    }
    if (emin < std::numeric_limits<double>::min())
        return std::numeric_limits<double>::infinity();
    return emax / emin;
}

/// Frobenius-norm-based factorization residual ||P - reconstruct|| / ||P||
static double cholesky_residual(const mat::dense2D<double>& P) {
    std::size_t n = P.num_rows();
    auto L = copy_matrix(P);
    int info = cholesky_factor(L);
    if (info != 0) return std::numeric_limits<double>::quiet_NaN();

    // Extract lower triangle
    mat::dense2D<double> Lmat(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            Lmat(i, j) = (j <= i) ? L(i, j) : 0.0;

    auto LLt = Lmat * trans(Lmat);

    double res = 0.0, pnorm = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double d = P(i, j) - LLt(i, j);
            res += d * d;
            pnorm += P(i, j) * P(i, j);
        }
    return std::sqrt(res) / std::sqrt(pnorm);
}

static double ldlt_residual(const mat::dense2D<double>& P) {
    std::size_t n = P.num_rows();
    auto F = copy_matrix(P);
    int info = ldlt_factor(F);
    if (info != 0) return std::numeric_limits<double>::quiet_NaN();

    // Extract L (unit lower) and D
    mat::dense2D<double> Lmat(n, n), Dmat(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            if (i == j) { Lmat(i, j) = 1.0; Dmat(i, j) = F(i, j); }
            else if (j < i) { Lmat(i, j) = F(i, j); Dmat(i, j) = 0.0; }
            else { Lmat(i, j) = 0.0; Dmat(i, j) = 0.0; }
        }

    auto LDLt = Lmat * Dmat * trans(Lmat);

    double res = 0.0, pnorm = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double d = P(i, j) - LDLt(i, j);
            res += d * d;
            pnorm += P(i, j) * P(i, j);
        }
    return std::sqrt(res) / std::sqrt(pnorm);
}

// ============================================================================
// Simple pseudo-random number generator (Xoshiro128+ -- no external deps)
// ============================================================================

struct Rng {
    uint64_t s[2];

    explicit Rng(uint64_t seed = 42) {
        s[0] = seed ^ 0x9E3779B97F4A7C15ULL;
        s[1] = seed ^ 0x6A09E667F3BCC908ULL;
        for (int i = 0; i < 10; ++i) next();
    }

    uint64_t next() {
        uint64_t s0 = s[0], s1 = s[1];
        uint64_t result = s0 + s1;
        s1 ^= s0;
        s[0] = ((s0 << 55) | (s0 >> 9)) ^ s1 ^ (s1 << 14);
        s[1] = (s1 << 36) | (s1 >> 28);
        return result;
    }

    /// Standard normal via Box-Muller
    double randn() {
        double u1 = (double(next() >> 11) + 0.5) / double(1ULL << 53);
        double u2 = (double(next() >> 11) + 0.5) / double(1ULL << 53);
        return std::sqrt(-2.0 * std::log(u1)) * std::cos(2.0 * std::numbers::pi * u2);
    }
};

// ============================================================================
// UKF Implementation
// ============================================================================

static constexpr std::size_t NX = 4;  // state dimension
static constexpr std::size_t NZ = 2;  // measurement dimension

/// UKF tuning parameters
struct UkfParams {
    double alpha = 1e-3;
    double beta  = 2.0;
    double kappa = 0.0;
    double dt    = 0.1;    // time step
    double drag  = 0.01;   // quadratic drag coefficient
    double beacon_x = 10.0;
    double beacon_y = 0.0;
};

/// Process model: constant velocity with quadratic drag
static vec::dense_vector<double> process_model(
    const vec::dense_vector<double>& x, const UkfParams& p)
{
    vec::dense_vector<double> xn(NX);
    double vx = x(2), vy = x(3);
    double speed = std::sqrt(vx * vx + vy * vy);
    double drag_factor = 1.0 - p.drag * speed * p.dt;
    if (drag_factor < 0.1) drag_factor = 0.1;

    xn(0) = x(0) + vx * p.dt;
    xn(1) = x(1) + vy * p.dt;
    xn(2) = vx * drag_factor;
    xn(3) = vy * drag_factor;
    return xn;
}

/// Measurement model: range and bearing from beacon
static vec::dense_vector<double> measurement_model(
    const vec::dense_vector<double>& x, const UkfParams& p)
{
    vec::dense_vector<double> z(NZ);
    double dx = x(0) - p.beacon_x;
    double dy = x(1) - p.beacon_y;
    z(0) = std::sqrt(dx * dx + dy * dy);  // range
    z(1) = std::atan2(dy, dx);             // bearing
    return z;
}

/// Wrap each sigma point's bearing relative to the central point's, before
/// the means and covariances are formed. A set that straddles atan2's cut at
/// +-pi (the target near the beacon's -x axis) would otherwise hold bearings
/// almost 2 pi apart.
static void wrap_bearings(std::vector<vec::dense_vector<double>>& sigma_z) {
    const double z0 = sigma_z[0](1);
    for (std::size_t s = 1; s < sigma_z.size(); ++s) {
        double d = sigma_z[s](1) - z0;
        while (d > std::numbers::pi)  d -= 2.0 * std::numbers::pi;
        while (d < -std::numbers::pi) d += 2.0 * std::numbers::pi;
        sigma_z[s](1) = z0 + d;
    }
}

/// Result of one UKF run
struct UkfResult {
    std::vector<double> estimation_error;
    std::vector<double> cond_P;
    std::vector<double> factorization_residual;
    bool diverged = false;
    int diverge_step = -1;
};

enum class FactMethod { Cholesky, LDLT };

/// Generate sigma points using Cholesky factorization
static bool generate_sigma_cholesky(
    const vec::dense_vector<double>& x_hat,
    const mat::dense2D<double>& P,
    double gamma,
    std::vector<vec::dense_vector<double>>& sigma)
{
    auto L = copy_matrix(P);
    int info = cholesky_factor(L);
    if (info != 0) return false;

    sigma[0] = x_hat;
    for (std::size_t j = 0; j < NX; ++j) {
        vec::dense_vector<double> col(NX);
        for (std::size_t i = 0; i < NX; ++i)
            col(i) = (i >= j) ? L(i, j) : 0.0;

        vec::dense_vector<double> sp(NX), sm(NX);
        for (std::size_t i = 0; i < NX; ++i) {
            sp(i) = x_hat(i) + gamma * col(i);
            sm(i) = x_hat(i) - gamma * col(i);
        }
        sigma[1 + j] = sp;
        sigma[1 + NX + j] = sm;
    }
    return true;
}

/// Generate sigma points using LDL^T factorization
static bool generate_sigma_ldlt(
    const vec::dense_vector<double>& x_hat,
    const mat::dense2D<double>& P,
    double gamma,
    std::vector<vec::dense_vector<double>>& sigma)
{
    auto F = copy_matrix(P);
    int info = ldlt_factor(F);
    if (info != 0) return false;

    // Extract L (unit lower) and D
    mat::dense2D<double> Lmat(NX, NX);
    vec::dense_vector<double> D(NX);
    for (std::size_t i = 0; i < NX; ++i) {
        D(i) = F(i, i);
        for (std::size_t j = 0; j < NX; ++j)
            Lmat(i, j) = (i == j) ? 1.0 : ((j < i) ? F(i, j) : 0.0);
    }

    // Sigma points: x_hat +/- gamma * L * sqrt(D_k) * e_k, column k of
    // L sqrt(D). A D_k <= 0 means P is not SPD: it has no real square root,
    // and sqrt(|D_k|) would give the sigma points of another covariance,
    // L |D| L^T. So it fails, as Cholesky does; D's signs say why.
    sigma[0] = x_hat;
    for (std::size_t k = 0; k < NX; ++k) {
        if (!(D(k) > 0.0)) return false;
        double sqrtDk = std::sqrt(D(k));
        vec::dense_vector<double> col(NX);
        for (std::size_t i = 0; i < NX; ++i)
            col(i) = Lmat(i, k) * sqrtDk;

        vec::dense_vector<double> sp(NX), sm(NX);
        for (std::size_t i = 0; i < NX; ++i) {
            sp(i) = x_hat(i) + gamma * col(i);
            sm(i) = x_hat(i) - gamma * col(i);
        }
        sigma[1 + k] = sp;
        sigma[1 + NX + k] = sm;
    }
    return true;
}

/// Run UKF for a given scenario
static UkfResult run_ukf(
    FactMethod method,
    const mat::dense2D<double>& Q,  // process noise
    const mat::dense2D<double>& R,  // measurement noise
    int num_steps,
    const UkfParams& params)
{
    UkfResult result;

    // UKF weights
    double lambda_ = params.alpha * params.alpha * (NX + params.kappa) - NX;
    double gamma = std::sqrt(NX + lambda_);
    double Wm0 = lambda_ / (NX + lambda_);
    double Wc0 = Wm0 + (1.0 - params.alpha * params.alpha + params.beta);
    double Wi  = 1.0 / (2.0 * (NX + lambda_));

    constexpr std::size_t num_sigma = 2 * NX + 1;
    std::vector<vec::dense_vector<double>> sigma_x(num_sigma);
    std::vector<vec::dense_vector<double>> sigma_z(num_sigma);

    // True state and estimate
    vec::dense_vector<double> x_true(NX);
    x_true(0) = 0.0; x_true(1) = 0.0; x_true(2) = 1.0; x_true(3) = 0.5;

    vec::dense_vector<double> x_hat(NX);
    x_hat(0) = 0.5; x_hat(1) = -0.5; x_hat(2) = 0.8; x_hat(3) = 0.3;

    // Initial covariance
    mat::dense2D<double> P(NX, NX);
    for (std::size_t i = 0; i < NX; ++i)
        for (std::size_t j = 0; j < NX; ++j)
            P(i, j) = (i == j) ? 1.0 : 0.0;

    Rng rng(12345);

    for (int step = 0; step < num_steps; ++step) {
        // === True state propagation with process noise ===
        x_true = process_model(x_true, params);
        for (std::size_t i = 0; i < NX; ++i)
            x_true(i) += std::sqrt(Q(i, i)) * rng.randn();

        // === Generate measurement with noise ===
        auto z_true = measurement_model(x_true, params);
        vec::dense_vector<double> z(NZ);
        for (std::size_t i = 0; i < NZ; ++i)
            z(i) = z_true(i) + std::sqrt(R(i, i)) * rng.randn();

        // === PREDICT: Generate sigma points from current P ===
        bool ok = false;
        if (method == FactMethod::Cholesky)
            ok = generate_sigma_cholesky(x_hat, P, gamma, sigma_x);
        else
            ok = generate_sigma_ldlt(x_hat, P, gamma, sigma_x);

        if (!ok) {
            result.diverged = true;
            result.diverge_step = step;
            break;
        }

        // Propagate sigma points through process model
        for (std::size_t s = 0; s < num_sigma; ++s)
            sigma_x[s] = process_model(sigma_x[s], params);

        // Predicted state mean
        vec::dense_vector<double> x_pred(NX);
        for (std::size_t i = 0; i < NX; ++i) {
            x_pred(i) = Wm0 * sigma_x[0](i);
            for (std::size_t s = 1; s < num_sigma; ++s)
                x_pred(i) += Wi * sigma_x[s](i);
        }

        // Predicted covariance
        mat::dense2D<double> P_pred(NX, NX);
        for (std::size_t i = 0; i < NX; ++i)
            for (std::size_t j = 0; j < NX; ++j)
                P_pred(i, j) = Q(i, j);

        // P_pred += Wc0 * (sigma_x[0] - x_pred) * (sigma_x[0] - x_pred)^T
        for (std::size_t i = 0; i < NX; ++i)
            for (std::size_t j = 0; j < NX; ++j)
                P_pred(i, j) += Wc0 * (sigma_x[0](i) - x_pred(i))
                                     * (sigma_x[0](j) - x_pred(j));

        for (std::size_t s = 1; s < num_sigma; ++s)
            for (std::size_t i = 0; i < NX; ++i)
                for (std::size_t j = 0; j < NX; ++j)
                    P_pred(i, j) += Wi * (sigma_x[s](i) - x_pred(i))
                                       * (sigma_x[s](j) - x_pred(j));

        // === UPDATE: Regenerate sigma points for measurement ===
        if (method == FactMethod::Cholesky)
            ok = generate_sigma_cholesky(x_pred, P_pred, gamma, sigma_x);
        else
            ok = generate_sigma_ldlt(x_pred, P_pred, gamma, sigma_x);

        if (!ok) {
            result.diverged = true;
            result.diverge_step = step;
            break;
        }

        // Propagate through measurement model
        for (std::size_t s = 0; s < num_sigma; ++s)
            sigma_z[s] = measurement_model(sigma_x[s], params);
        wrap_bearings(sigma_z);

        // Predicted measurement mean
        vec::dense_vector<double> z_pred(NZ);
        for (std::size_t i = 0; i < NZ; ++i) {
            z_pred(i) = Wm0 * sigma_z[0](i);
            for (std::size_t s = 1; s < num_sigma; ++s)
                z_pred(i) += Wi * sigma_z[s](i);
        }

        // Innovation covariance S = R + sum Wi * (z_sigma - z_pred)(z_sigma - z_pred)^T
        mat::dense2D<double> S(NZ, NZ);
        for (std::size_t i = 0; i < NZ; ++i)
            for (std::size_t j = 0; j < NZ; ++j)
                S(i, j) = R(i, j);

        for (std::size_t i = 0; i < NZ; ++i)
            for (std::size_t j = 0; j < NZ; ++j)
                S(i, j) += Wc0 * (sigma_z[0](i) - z_pred(i))
                                * (sigma_z[0](j) - z_pred(j));

        for (std::size_t s = 1; s < num_sigma; ++s)
            for (std::size_t i = 0; i < NZ; ++i)
                for (std::size_t j = 0; j < NZ; ++j)
                    S(i, j) += Wi * (sigma_z[s](i) - z_pred(i))
                                  * (sigma_z[s](j) - z_pred(j));

        // Cross-covariance Pxz
        mat::dense2D<double> Pxz(NX, NZ);
        for (std::size_t i = 0; i < NX; ++i)
            for (std::size_t j = 0; j < NZ; ++j)
                Pxz(i, j) = 0.0;

        for (std::size_t i = 0; i < NX; ++i)
            for (std::size_t j = 0; j < NZ; ++j)
                Pxz(i, j) += Wc0 * (sigma_x[0](i) - x_pred(i))
                                  * (sigma_z[0](j) - z_pred(j));

        for (std::size_t s = 1; s < num_sigma; ++s)
            for (std::size_t i = 0; i < NX; ++i)
                for (std::size_t j = 0; j < NZ; ++j)
                    Pxz(i, j) += Wi * (sigma_x[s](i) - x_pred(i))
                                    * (sigma_z[s](j) - z_pred(j));

        // Kalman gain K = Pxz * S^{-1}  (solve via LU for 2x2 S)
        // For a 2x2, just invert directly
        double det = S(0,0) * S(1,1) - S(0,1) * S(1,0);
        if (std::abs(det) < 1e-30) {
            result.diverged = true;
            result.diverge_step = step;
            break;
        }
        mat::dense2D<double> Sinv(NZ, NZ);
        Sinv(0,0) =  S(1,1) / det;
        Sinv(0,1) = -S(0,1) / det;
        Sinv(1,0) = -S(1,0) / det;
        Sinv(1,1) =  S(0,0) / det;

        auto K = Pxz * Sinv;

        // Innovation
        vec::dense_vector<double> innov(NZ);
        innov(0) = z(0) - z_pred(0);
        innov(1) = z(1) - z_pred(1);
        // Wrap bearing innovation to [-pi, pi]
        while (innov(1) > std::numbers::pi)  innov(1) -= 2.0 * std::numbers::pi;
        while (innov(1) < -std::numbers::pi) innov(1) += 2.0 * std::numbers::pi;

        // State update: x_hat = x_pred + K * innov
        auto Kinnov = K * innov;
        for (std::size_t i = 0; i < NX; ++i)
            x_hat(i) = x_pred(i) + Kinnov(i);

        // Covariance update: P = P_pred - K * S * K^T
        auto KS = K * S;
        auto KSKt = KS * trans(K);
        for (std::size_t i = 0; i < NX; ++i)
            for (std::size_t j = 0; j < NX; ++j)
                P(i, j) = P_pred(i, j) - KSKt(i, j);

        // Force symmetry
        for (std::size_t i = 0; i < NX; ++i)
            for (std::size_t j = i + 1; j < NX; ++j)
                P(j, i) = P(i, j);

        // === Record metrics ===
        double err = 0.0;
        for (std::size_t i = 0; i < NX; ++i) {
            double d = x_true(i) - x_hat(i);
            err += d * d;
        }
        err = std::sqrt(err);

        // Check for NaN/Inf divergence
        if (!std::isfinite(err) || err > 1e6) {
            result.diverged = true;
            result.diverge_step = step;
            result.estimation_error.push_back(err);
            result.cond_P.push_back(std::numeric_limits<double>::quiet_NaN());
            result.factorization_residual.push_back(std::numeric_limits<double>::quiet_NaN());
            break;
        }

        result.estimation_error.push_back(err);
        result.cond_P.push_back(condition_number(P));

        if (method == FactMethod::Cholesky)
            result.factorization_residual.push_back(cholesky_residual(P));
        else
            result.factorization_residual.push_back(ldlt_residual(P));
    }

    return result;
}

// ============================================================================
// Reporting
// ============================================================================

static void print_scenario_header(const std::string& name,
                                  const mat::dense2D<double>& R) {
    std::cout << "\n=== Scenario: " << name << " (R = diag("
              << std::scientific << std::setprecision(1)
              << R(0,0) << ", " << R(1,1) << ")) ===\n";
    std::cout << std::setw(6) << "Step"
              << std::setw(14) << "cond(P)"
              << std::setw(14) << "Chol-err"
              << std::setw(14) << "LDLT-err"
              << std::setw(14) << "Chol-resid"
              << std::setw(14) << "LDLT-resid"
              << "\n";
    std::cout << std::string(76, '-') << "\n";
}

static void print_row(int step,
                      const UkfResult& chol, const UkfResult& ldlt,
                      std::size_t idx)
{
    std::cout << std::setw(6) << step;

    // Condition number (from LDLT -- it runs longer when Cholesky diverges)
    if (idx < ldlt.cond_P.size())
        std::cout << std::setw(14) << std::scientific << std::setprecision(2) << ldlt.cond_P[idx];
    else if (idx < chol.cond_P.size())
        std::cout << std::setw(14) << std::scientific << std::setprecision(2) << chol.cond_P[idx];
    else
        std::cout << std::setw(14) << "N/A";

    // Cholesky error
    if (idx < chol.estimation_error.size() && std::isfinite(chol.estimation_error[idx]))
        std::cout << std::setw(14) << std::scientific << std::setprecision(2) << chol.estimation_error[idx];
    else
        std::cout << std::setw(14) << "diverged";

    // LDLT error
    if (idx < ldlt.estimation_error.size() && std::isfinite(ldlt.estimation_error[idx]))
        std::cout << std::setw(14) << std::scientific << std::setprecision(2) << ldlt.estimation_error[idx];
    else
        std::cout << std::setw(14) << "diverged";

    // Cholesky residual
    if (idx < chol.factorization_residual.size() && std::isfinite(chol.factorization_residual[idx]))
        std::cout << std::setw(14) << std::scientific << std::setprecision(2) << chol.factorization_residual[idx];
    else
        std::cout << std::setw(14) << "NaN";

    // LDLT residual
    if (idx < ldlt.factorization_residual.size() && std::isfinite(ldlt.factorization_residual[idx]))
        std::cout << std::setw(14) << std::scientific << std::setprecision(2) << ldlt.factorization_residual[idx];
    else
        std::cout << std::setw(14) << "NaN";

    std::cout << "\n";
}

/// What the conclusion reads from each scenario.
struct ScenarioSummary {
    std::string name;
    bool chol_done = false, ldlt_done = false;
    double chol_err = 0.0, ldlt_err = 0.0;   // final errors (when done)
    double max_cond = 0.0;                    // largest cond(P) of the LDL^T run
};

static ScenarioSummary run_scenario(const std::string& name,
                                    const mat::dense2D<double>& R,
                                    int num_steps)
{
    // Process noise covariance
    mat::dense2D<double> Q(NX, NX);
    for (std::size_t i = 0; i < NX; ++i)
        for (std::size_t j = 0; j < NX; ++j)
            Q(i, j) = 0.0;
    Q(0,0) = 1e-4; Q(1,1) = 1e-4;
    Q(2,2) = 1e-2; Q(3,3) = 1e-2;

    UkfParams params;
    auto chol = run_ukf(FactMethod::Cholesky, Q, R, num_steps, params);
    auto ldlt = run_ukf(FactMethod::LDLT,     Q, R, num_steps, params);

    print_scenario_header(name, R);

    // Print at selected intervals
    std::vector<int> report_steps;
    for (int s = 0; s < num_steps; s += std::max(1, num_steps / 10))
        report_steps.push_back(s);
    if (report_steps.back() != num_steps - 1)
        report_steps.push_back(num_steps - 1);

    for (int s : report_steps) {
        std::size_t idx = static_cast<std::size_t>(s);
        std::size_t max_idx = std::max(chol.estimation_error.size(),
                                        ldlt.estimation_error.size());
        if (idx >= max_idx) break;
        print_row(s, chol, ldlt, idx);
    }

    // Summary
    std::cout << "\n";
    if (chol.diverged)
        std::cout << "  Cholesky: DIVERGED at step " << chol.diverge_step << "\n";
    else
        std::cout << "  Cholesky: completed " << num_steps << " steps, final error = "
                  << std::scientific << std::setprecision(3)
                  << chol.estimation_error.back() << "\n";

    if (ldlt.diverged)
        std::cout << "  LDL^T:    DIVERGED at step " << ldlt.diverge_step << "\n";
    else
        std::cout << "  LDL^T:    completed " << num_steps << " steps, final error = "
                  << std::scientific << std::setprecision(3)
                  << ldlt.estimation_error.back() << "\n";

    ScenarioSummary sum;
    sum.name = name;
    sum.chol_done = !chol.diverged;
    sum.ldlt_done = !ldlt.diverged;
    if (sum.chol_done) sum.chol_err = chol.estimation_error.back();
    if (sum.ldlt_done) sum.ldlt_err = ldlt.estimation_error.back();
    for (double c : ldlt.cond_P)
        if (std::isfinite(c) && c > sum.max_cond) sum.max_cond = c;
    std::cout << "  largest cond(P): " << std::scientific << std::setprecision(2) << sum.max_cond << "\n";
    return sum;
}

// ============================================================================
// Main
// ============================================================================

int main() {
    std::cout << "================================================================\n";
    std::cout << " UKF Numerical Stability: Cholesky vs LDL^T Decomposition\n";
    std::cout << "================================================================\n";
    std::cout << "\n"
              << "State:       [px, py, vx, vy] (4-state 2D tracking)\n"
              << "Process:     constant velocity + quadratic drag\n"
              << "Measurement: range + bearing from beacon at (10, 0)\n"
              << "Key knob:    measurement noise R -- smaller R drives P ill-conditioned\n"
              << "\n";

    constexpr int num_steps = 200;
    std::vector<ScenarioSummary> summaries;

    // Scenario 1: Benign
    {
        mat::dense2D<double> R(NZ, NZ);
        for (std::size_t i = 0; i < NZ; ++i)
            for (std::size_t j = 0; j < NZ; ++j)
                R(i, j) = (i == j) ? 1.0 : 0.0;
        summaries.push_back(run_scenario("Benign", R, num_steps));
    }

    // Scenario 2: Moderate stress
    {
        mat::dense2D<double> R(NZ, NZ);
        for (std::size_t i = 0; i < NZ; ++i)
            for (std::size_t j = 0; j < NZ; ++j)
                R(i, j) = 0.0;
        R(0,0) = 0.01; R(1,1) = 0.1;
        summaries.push_back(run_scenario("Moderate stress", R, num_steps));
    }

    // Scenario 3: Severe stress
    {
        mat::dense2D<double> R(NZ, NZ);
        for (std::size_t i = 0; i < NZ; ++i)
            for (std::size_t j = 0; j < NZ; ++j)
                R(i, j) = 0.0;
        R(0,0) = 1e-14; R(1,1) = 1e-2;
        summaries.push_back(run_scenario("Severe stress", R, num_steps));
    }

    // === Direct factorization stress test ===
    // Demonstrate the factorization robustness difference by testing:
    // 1. SPD matrices with increasing condition number
    // 2. Matrices that are slightly indefinite (which happen in practice
    //    when UKF covariance updates introduce rounding errors)
    std::cout << "\n================================================================\n";
    std::cout << " Direct Factorization Stress Test\n";
    std::cout << "================================================================\n\n";

    // Test 1: SPD matrices with controlled condition numbers
    std::cout << "--- SPD matrices, n = 8, eigenvalues spread over cond(P) ---\n";
    double spd_max_cr = 0.0, spd_max_lr = 0.0;
    bool spd_all_ok = true;
    std::cout << std::setw(14) << "cond(P)"
              << std::setw(16) << "Chol residual"
              << std::setw(16) << "LDLT residual"
              << "\n";
    std::cout << std::string(46, '-') << "\n";

    for (double log_cond : {2.0, 4.0, 8.0, 12.0, 15.0}) {
        double target_cond = std::pow(10.0, log_cond);
        std::size_t n = 8;
        mat::dense2D<double> M(n, n);
        Rng rng2(static_cast<uint64_t>(log_cond * 1000));
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                M(i, j) = rng2.randn();

        vec::dense_vector<double> tau(n);
        qr_factor(M, tau);
        auto Qmat = qr_extract_Q(M, tau);

        mat::dense2D<double> P(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                P(i, j) = 0.0;

        for (std::size_t k = 0; k < n; ++k) {
            double eig = std::pow(target_cond, double(k) / double(n - 1));
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j < n; ++j)
                    P(i, j) += eig * Qmat(i, k) * Qmat(j, k);
        }

        double cr = cholesky_residual(P);
        double lr = ldlt_residual(P);
        if (std::isfinite(cr) && std::isfinite(lr)) {
            spd_max_cr = std::max(spd_max_cr, cr);
            spd_max_lr = std::max(spd_max_lr, lr);
        } else {
            spd_all_ok = false;
        }

        std::cout << std::setw(14) << std::scientific << std::setprecision(1) << target_cond
                  << std::setw(16) << std::scientific << std::setprecision(3);
        if (std::isfinite(cr)) std::cout << cr;
        else std::cout << "FAIL";
        std::cout << std::setw(16) << std::scientific << std::setprecision(3) << lr << "\n";
    }

    // Test 2: indefinite matrices. A covariance update P = P_pred - K S K^T
    // can lose positive definiteness to rounding (the float32 filter in
    // ../ukf_bearing_only shows it). Two families, neither diagonal:
    //   - diag(1, 0.5, 0.1, min_eig) rotated by a random orthogonal Q;
    //   - the Hilbert matrix H_6 (cond 1.5e7) shifted by twice its smallest
    //     eigenvalue, so that lambda_min = -lambda_min(H_6) = -1.1e-7;
    //   - ones off the diagonal and a tiny e on it, whose first pivot
    //     unpivoted LDL^T must divide by.
    // Cholesky stops on every one. Unpivoted LDL^T returns, with D's signs,
    // but on an indefinite matrix it is not backward stable: a small pivot
    // makes L large. Bunch-Kaufman (ldlt_bk) pivots with 1x1 and 2x2 blocks,
    // and its D gives the inertia. Both are measured by the backward error of
    // a solve with their factors, ||P x - b|| / (||P|| ||x||).
    std::cout << "\n--- Indefinite matrices ---\n";
    std::cout << std::setw(26) << "matrix" << std::setw(11) << "lambda_min" << std::setw(5) << "neg"
              << std::setw(9) << "Cholesky" << std::setw(15) << "LDL^T D signs" << std::setw(12) << "LDL^T berr"
              << std::setw(8) << "BK neg" << std::setw(11) << "BK berr" << "\n";
    std::cout << std::string(97, '-') << "\n";

    auto eig_min_count = [](const mat::dense2D<double>& P, int& negatives) {
        auto e = eigenvalue_symmetric(copy_matrix(P));
        double lo = e(0);
        negatives = 0;
        for (std::size_t i = 0; i < e.size(); ++i) {
            lo = std::min(lo, e(i));
            if (e(i) < 0.0) ++negatives;
        }
        return lo;
    };
    // Bunch-Kaufman: the number of negative eigenvalues of D (1x1 pivots by
    // sign, 2x2 blocks by their determinant), and ||P - P^T L D L^T P|| / ||P||
    // through the solve: P x = P e_j column by column, compared with e_j.
    // The backward error of x for P x = b.
    auto backward_error = [](const mat::dense2D<double>& P, const vec::dense_vector<double>& x,
                             const vec::dense_vector<double>& bv) {
        const std::size_t n = P.num_rows();
        double r2 = 0.0, a2 = 0.0, x2 = 0.0;
        for (std::size_t i = 0; i < n; ++i) {
            double ax = 0.0;
            for (std::size_t j = 0; j < n; ++j) { ax += P(i, j) * x(j); a2 += P(i, j) * P(i, j); }
            r2 += (ax - bv(i)) * (ax - bv(i));
            x2 += x(i) * x(i);
        }
        return std::sqrt(r2) / (std::sqrt(a2) * std::sqrt(x2));
    };
    // b = P (1, 2, ..., n)^T.
    auto rhs = [](const mat::dense2D<double>& P) {
        vec::dense_vector<double> bv(P.num_rows(), 0.0);
        for (std::size_t i = 0; i < P.num_rows(); ++i)
            for (std::size_t j = 0; j < P.num_cols(); ++j) bv(i) += P(i, j) * double(j + 1);
        return bv;
    };
    auto bk = [&](const mat::dense2D<double>& P, int& negatives, double& res) {
        const std::size_t n = P.num_rows();
        auto F = copy_matrix(P);
        bk_pivot_info piv;
        if (ldlt_bk_factor(F, piv) != 0) return false;
        negatives = 0;
        for (std::size_t k = 0; k < n; ++k) {
            if (piv.ipiv[k] > 0) {
                if (F(k, k) < 0.0) ++negatives;
            } else {
                const double a = F(k, k), b = F(k + 1, k), c = F(k + 1, k + 1);
                const double det = a * c - b * b;
                negatives += det < 0.0 ? 1 : (a < 0.0 ? 2 : 0);
                ++k;
            }
        }
        const auto bv = rhs(P);
        vec::dense_vector<double> x(n, 0.0);
        ldlt_bk_solve(F, piv, x, bv);
        res = backward_error(P, x, bv);
        return true;
    };

    struct Indef { std::string name; mat::dense2D<double> P; };
    std::vector<Indef> indef;
    for (double min_eig : {-1e-14, -1e-10, -1e-6, -1e-2}) {
        const std::size_t n = 4;
        mat::dense2D<double> P(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                P(i, j) = 0.0;
        P(0,0) = 1.0; P(1,1) = 0.5; P(2,2) = 0.1; P(3,3) = min_eig;
        mat::dense2D<double> M2(n, n);
        Rng rng3(99);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j)
                M2(i, j) = rng3.randn();
        vec::dense_vector<double> tau2(n);
        qr_factor(M2, tau2);
        auto Qmat2 = qr_extract_Q(M2, tau2);
        auto QP = Qmat2 * P;
        mat::dense2D<double> P_rot = QP * trans(Qmat2);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j)
                P_rot(j, i) = P_rot(i, j);
        std::ostringstream nm;
        nm << "Q diag(.., " << std::scientific << std::setprecision(0) << min_eig << ") Q^T";
        indef.push_back({nm.str(), std::move(P_rot)});
    }
    {
        const std::size_t n = 6;
        mat::dense2D<double> H(n, n);
        for (std::size_t i = 0; i < n; ++i)
            for (std::size_t j = 0; j < n; ++j) H(i, j) = 1.0 / double(i + j + 1);
        int neg = 0;
        const double lmin = eig_min_count(H, neg);
        for (std::size_t i = 0; i < n; ++i) H(i, i) -= 2.0 * lmin;
        indef.push_back({"H_6 - 2 lambda_min I", std::move(H)});
    }
    for (double e : {1e-8, 1e-12}) {
        mat::dense2D<double> T(4, 4);
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = 0; j < 4; ++j) T(i, j) = (i == j) ? e : 1.0;
        std::ostringstream nm;
        nm << "ones, diagonal " << std::scientific << std::setprecision(0) << e;
        indef.push_back({nm.str(), std::move(T)});
    }

    bool indef_chol_all_fail = true, indef_bk_inertia_ok = true;
    double indef_max_ldlt_res = 0.0, indef_max_bk_res = 0.0;
    for (const auto& c : indef) {
        int neg_eig = 0;
        const double lmin = eig_min_count(c.P, neg_eig);
        auto Pc = copy_matrix(c.P);
        const bool chol_ok = cholesky_factor(Pc) == 0;
        if (chol_ok) indef_chol_all_fail = false;
        auto Pl = copy_matrix(c.P);
        const bool ldlt_ok = ldlt_factor(Pl) == 0;
        std::string signs = "[";
        for (std::size_t i = 0; i < c.P.num_rows(); ++i) signs += (i ? "," : "") + std::string(Pl(i, i) >= 0 ? "+" : "-");
        signs += "]";
        double lres = std::numeric_limits<double>::quiet_NaN();
        if (ldlt_ok) {
            const auto bv = rhs(c.P);
            vec::dense_vector<double> x(c.P.num_rows(), 0.0);
            ldlt_solve(Pl, x, bv);
            lres = backward_error(c.P, x, bv);
        }
        int bk_neg = -1;
        double bres = std::numeric_limits<double>::quiet_NaN();
        const bool bk_ok = bk(c.P, bk_neg, bres);
        if (!bk_ok || bk_neg != neg_eig) indef_bk_inertia_ok = false;
        if (std::isfinite(lres)) indef_max_ldlt_res = std::max(indef_max_ldlt_res, lres);
        if (std::isfinite(bres)) indef_max_bk_res = std::max(indef_max_bk_res, bres);
        std::cout << std::setw(26) << c.name << std::setw(11) << std::scientific << std::setprecision(1) << lmin
                  << std::setw(5) << neg_eig << std::setw(9) << (chol_ok ? "OK" : "stops") << std::setw(15)
                  << (ldlt_ok ? signs : "FAIL") << std::setw(12) << lres << std::setw(8)
                  << (bk_ok ? std::to_string(bk_neg) : "FAIL") << std::setw(11) << bres << "\n";
    }
    std::cout << "\n  neg is the number of negative eigenvalues; BK neg the number ldlt_bk's D has.\n"
              << "  berr is the backward error of a solve with that factorization.\n";

    // Scenario 4: Intermittent
    std::cout << "\n=== Scenario: Intermittent (alternating R) ===\n";
    std::cout << "  (alternates between R=diag(1e-8,1e-2) and R=diag(1,1) every 25 steps)\n";
    {
        mat::dense2D<double> Q(NX, NX);
        for (std::size_t i = 0; i < NX; ++i)
            for (std::size_t j = 0; j < NX; ++j)
                Q(i, j) = 0.0;
        Q(0,0) = 1e-4; Q(1,1) = 1e-4;
        Q(2,2) = 1e-2; Q(3,3) = 1e-2;

        // Run intermittent scenario manually with switching R
        UkfParams params;
        mat::dense2D<double> R_stress(NZ, NZ), R_benign(NZ, NZ);
        for (std::size_t i = 0; i < NZ; ++i)
            for (std::size_t j = 0; j < NZ; ++j) {
                R_stress(i, j) = 0.0;
                R_benign(i, j) = 0.0;
            }
        R_stress(0,0) = 1e-8; R_stress(1,1) = 1e-2;
        R_benign(0,0) = 1.0;  R_benign(1,1) = 1.0;

        // For intermittent, we run single-step UKFs sequentially.
        // This is simpler than modifying run_ukf to accept per-step R.
        // We'll just report whether each method completes without divergence.
        std::cout << "  (Running " << num_steps << " steps with alternating precision...)\n";

        bool chol_ok = true, ldlt_ok = true;
        int chol_div = -1, ldlt_div = -1;
        double chol_err = 0.0, ldlt_err = 0.0;
        double ldlt_max_cond = 0.0;   // the LDL^T run's largest posterior cond(P), as run_scenario's

        for (int method_idx = 0; method_idx < 2; ++method_idx) {
            FactMethod method = (method_idx == 0) ? FactMethod::Cholesky : FactMethod::LDLT;
            auto& ok = (method_idx == 0) ? chol_ok : ldlt_ok;
            auto& div_step = (method_idx == 0) ? chol_div : ldlt_div;
            auto& final_err = (method_idx == 0) ? chol_err : ldlt_err;

            // Reset state
            double lambda_ = params.alpha * params.alpha * (NX + params.kappa) - NX;
            double gamma = std::sqrt(NX + lambda_);
            double Wm0 = lambda_ / (NX + lambda_);
            double Wc0 = Wm0 + (1.0 - params.alpha * params.alpha + params.beta);
            double Wi  = 1.0 / (2.0 * (NX + lambda_));
            constexpr std::size_t num_sigma = 2 * NX + 1;

            vec::dense_vector<double> x_true(NX);
            x_true(0) = 0.0; x_true(1) = 0.0; x_true(2) = 1.0; x_true(3) = 0.5;
            vec::dense_vector<double> x_hat(NX);
            x_hat(0) = 0.5; x_hat(1) = -0.5; x_hat(2) = 0.8; x_hat(3) = 0.3;
            mat::dense2D<double> P(NX, NX);
            for (std::size_t i = 0; i < NX; ++i)
                for (std::size_t j = 0; j < NX; ++j)
                    P(i, j) = (i == j) ? 1.0 : 0.0;

            Rng rng(12345);
            std::vector<vec::dense_vector<double>> sigma_x(num_sigma), sigma_z(num_sigma);

            for (int step = 0; step < num_steps; ++step) {
                const auto& R = ((step / 25) % 2 == 0) ? R_stress : R_benign;

                x_true = process_model(x_true, params);
                for (std::size_t i = 0; i < NX; ++i)
                    x_true(i) += std::sqrt(Q(i, i)) * rng.randn();

                auto z_true = measurement_model(x_true, params);
                vec::dense_vector<double> z(NZ);
                for (std::size_t i = 0; i < NZ; ++i)
                    z(i) = z_true(i) + std::sqrt(R(i, i)) * rng.randn();

                bool sigma_ok = false;
                if (method == FactMethod::Cholesky)
                    sigma_ok = generate_sigma_cholesky(x_hat, P, gamma, sigma_x);
                else
                    sigma_ok = generate_sigma_ldlt(x_hat, P, gamma, sigma_x);

                if (!sigma_ok) { ok = false; div_step = step; break; }

                for (std::size_t s = 0; s < num_sigma; ++s)
                    sigma_x[s] = process_model(sigma_x[s], params);

                vec::dense_vector<double> x_pred(NX);
                for (std::size_t i = 0; i < NX; ++i) {
                    x_pred(i) = Wm0 * sigma_x[0](i);
                    for (std::size_t s = 1; s < num_sigma; ++s)
                        x_pred(i) += Wi * sigma_x[s](i);
                }

                mat::dense2D<double> P_pred(NX, NX);
                for (std::size_t i = 0; i < NX; ++i)
                    for (std::size_t j = 0; j < NX; ++j)
                        P_pred(i, j) = Q(i, j) + Wc0 * (sigma_x[0](i) - x_pred(i)) * (sigma_x[0](j) - x_pred(j));
                for (std::size_t s = 1; s < num_sigma; ++s)
                    for (std::size_t i = 0; i < NX; ++i)
                        for (std::size_t j = 0; j < NX; ++j)
                            P_pred(i, j) += Wi * (sigma_x[s](i) - x_pred(i)) * (sigma_x[s](j) - x_pred(j));

                if (method == FactMethod::Cholesky)
                    sigma_ok = generate_sigma_cholesky(x_pred, P_pred, gamma, sigma_x);
                else
                    sigma_ok = generate_sigma_ldlt(x_pred, P_pred, gamma, sigma_x);
                if (!sigma_ok) { ok = false; div_step = step; break; }

                for (std::size_t s = 0; s < num_sigma; ++s)
                    sigma_z[s] = measurement_model(sigma_x[s], params);
                wrap_bearings(sigma_z);

                vec::dense_vector<double> z_pred(NZ);
                for (std::size_t i = 0; i < NZ; ++i) {
                    z_pred(i) = Wm0 * sigma_z[0](i);
                    for (std::size_t s = 1; s < num_sigma; ++s)
                        z_pred(i) += Wi * sigma_z[s](i);
                }

                mat::dense2D<double> S(NZ, NZ);
                for (std::size_t i = 0; i < NZ; ++i)
                    for (std::size_t j = 0; j < NZ; ++j)
                        S(i, j) = R(i, j) + Wc0 * (sigma_z[0](i) - z_pred(i)) * (sigma_z[0](j) - z_pred(j));
                for (std::size_t s = 1; s < num_sigma; ++s)
                    for (std::size_t i = 0; i < NZ; ++i)
                        for (std::size_t j = 0; j < NZ; ++j)
                            S(i, j) += Wi * (sigma_z[s](i) - z_pred(i)) * (sigma_z[s](j) - z_pred(j));

                mat::dense2D<double> Pxz(NX, NZ);
                for (std::size_t i = 0; i < NX; ++i)
                    for (std::size_t j = 0; j < NZ; ++j) {
                        Pxz(i, j) = Wc0 * (sigma_x[0](i) - x_pred(i)) * (sigma_z[0](j) - z_pred(j));
                        for (std::size_t s = 1; s < num_sigma; ++s)
                            Pxz(i, j) += Wi * (sigma_x[s](i) - x_pred(i)) * (sigma_z[s](j) - z_pred(j));
                    }

                double det = S(0,0)*S(1,1) - S(0,1)*S(1,0);
                if (std::abs(det) < 1e-30) { ok = false; div_step = step; break; }
                mat::dense2D<double> Sinv(NZ, NZ);
                Sinv(0,0) =  S(1,1)/det; Sinv(0,1) = -S(0,1)/det;
                Sinv(1,0) = -S(1,0)/det; Sinv(1,1) =  S(0,0)/det;
                auto K = Pxz * Sinv;

                vec::dense_vector<double> innov(NZ);
                innov(0) = z(0) - z_pred(0);
                innov(1) = z(1) - z_pred(1);
                while (innov(1) > std::numbers::pi)  innov(1) -= 2.0 * std::numbers::pi;
                while (innov(1) < -std::numbers::pi) innov(1) += 2.0 * std::numbers::pi;

                auto Kinnov = K * innov;
                for (std::size_t i = 0; i < NX; ++i)
                    x_hat(i) = x_pred(i) + Kinnov(i);

                auto KS = K * S;
                auto KSKt = KS * trans(K);
                for (std::size_t i = 0; i < NX; ++i)
                    for (std::size_t j = 0; j < NX; ++j)
                        P(i, j) = P_pred(i, j) - KSKt(i, j);
                for (std::size_t i = 0; i < NX; ++i)
                    for (std::size_t j = i + 1; j < NX; ++j)
                        P(j, i) = P(i, j);
                if (method == FactMethod::LDLT) {
                    const double c = condition_number(P);
                    if (std::isfinite(c) && c > ldlt_max_cond) ldlt_max_cond = c;
                }

                double err = 0.0;
                for (std::size_t i = 0; i < NX; ++i) {
                    double d = x_true(i) - x_hat(i);
                    err += d * d;
                }
                if (!std::isfinite(std::sqrt(err)) || std::sqrt(err) > 1e6) {
                    ok = false; div_step = step; break;
                }
                final_err = std::sqrt(err);
            }
        }

        if (chol_ok)
            std::cout << "  Cholesky: completed " << num_steps << " steps, final error = "
                      << std::scientific << std::setprecision(3) << chol_err << "\n";
        else
            std::cout << "  Cholesky: DIVERGED at step " << chol_div << "\n";

        if (ldlt_ok)
            std::cout << "  LDL^T:    completed " << num_steps << " steps, final error = "
                      << std::scientific << std::setprecision(3) << ldlt_err << "\n";
        else
            std::cout << "  LDL^T:    DIVERGED at step " << ldlt_div << "\n";

        ScenarioSummary sum;
        sum.name = "Intermittent";
        sum.chol_done = chol_ok;
        sum.ldlt_done = ldlt_ok;
        sum.chol_err = chol_err;
        sum.ldlt_err = ldlt_err;
        sum.max_cond = ldlt_max_cond;
        std::cout << "  largest cond(P): " << std::scientific << std::setprecision(2) << sum.max_cond << "\n";
        summaries.push_back(sum);
    }

    std::cout << "\n================================================================\n";
    std::cout << " Conclusion (computed from the runs above)\n";
    std::cout << "================================================================\n\n";
    bool both_done = true;
    double worst_cond = 0.0, worst_rel = 0.0;
    for (const auto& sm : summaries) {
        if (!sm.chol_done || !sm.ldlt_done) both_done = false;
        else worst_rel = std::max(worst_rel, std::abs(sm.chol_err - sm.ldlt_err) / sm.ldlt_err);
        worst_cond = std::max(worst_cond, sm.max_cond);
    }
    std::cout << std::scientific << std::setprecision(1);
    if (both_done)
        std::cout << "- In all " << summaries.size() << " scenarios both filters complete, and their final errors\n"
                  << "  agree to a relative " << worst_rel << ": rounding, amplified by the UKF weights at\n"
                  << "  alpha = 1e-3 (W_0 = -1e6; see ../ukf_bearing_only). ";
    else
        std::cout << "- In at least one scenario one filter did not complete (see above). ";
    std::cout << "cond(P) never exceeded " << worst_cond << ": the process noise Q,\n"
              << "  added at every predict step, keeps P's eigenvalues up, even with R = 1e-14.\n";
    if (spd_all_ok)
        std::cout << "- On SPD matrices up to cond 1e15 both factorizations' residuals stay at rounding\n"
                  << "  level: at most " << spd_max_cr << " (Cholesky) and " << spd_max_lr << " (LDL^T).\n"
                  << "  When Cholesky completes, its backward error is small at any condition number; it\n"
                  << "  is guaranteed to complete for c n^(3/2) cond(P) eps < 1, and beyond that it stops\n"
                  << "  rather than return a poor factor (as LDL^T's D gets an entry <= 0). A scalar\n"
                  << "  square root halves a relative error. Neither method is the safer choice above\n"
                  << "  some condition number, and LDL^T's sigma points take sqrt(D_k) anyway.\n";
    std::cout << "- On an indefinite matrix the two differ in kind: Cholesky "
              << (indef_chol_all_fail ? "stops on every one" : "does not always stop") << ", while\n"
              << "  LDL^T returns, with a negative D entry. Unpivoted LDL^T is not stable there: its\n"
              << "  solve's backward error reached " << indef_max_ldlt_res << ". Bunch-Kaufman (ldlt_bk) "
              << (indef_bk_inertia_ok ? "found the\n  inertia of every matrix" : "missed an\n  inertia")
              << ", with a backward error of at most " << indef_max_bk_res << ".\n"
              << "- An indefinite covariance has no sigma points: either factorization can only report\n"
              << "  it. Where a UKF does meet one, look at the update's arithmetic (and, in low\n"
              << "  precision, at the weights: ../ukf_bearing_only/bearing_only_ukf.cpp).\n";

    return 0;
}
