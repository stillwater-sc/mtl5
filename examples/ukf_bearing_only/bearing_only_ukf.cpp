// bearing_only_ukf.cpp -- Bearing-only UKF: Cholesky against LDL^T, float64
// against float32, and what really breaks the float32 filter
//
// A 2-D tracker [px, py, vx, vy] with one measurement, the bearing from a
// beacon. The bearing pins down one direction of the position and leaves the
// range unobserved, the classic source of a covariance P with a large
// eigenvalue spread.
//
// This example runs TWO filters on one scenario (the same truth and the same
// measurements), one taking its sigma points from Cholesky, P = L L^T, the
// other from LDL^T, P = L D L^T. It runs them in float64 and in float32, and
// prints:
//   1. each step's cond(P), the sigma-point mean bias of each filter, and how
//      far the two filters' estimates are apart;
//   2. where a filter fails, and on what;
//   3. the UKF weights in float32 for alpha = 1e-3 ... 1, and the bias and
//      the outcome of both float32 filters at each alpha.
//
// What it shows (issue 953; the conclusion at the end is computed from the
// runs):
//   - The two factorizations give the same filter. Both are backward stable
//     square roots of an SPD P, at any condition number; any square root of P
//     gives sigma points symmetric about x-hat. (LDL^T does not avoid the
//     square root: its sigma points are x-hat +- gamma L sqrt(D_k) e_k.)
//   - float32 fails because of the weights, not the factorization. At
//     alpha = 1e-3, W_0 = lambda / (n + lambda) is about -1e6 and the other
//     weights about 1.25e5, so the weighted sums cancel six of float32's seven
//     digits. The covariance P^- - K S K^T then loses its positive
//     definiteness, and BOTH filters stop, at the same step. With alpha = 1
//     (W_0 = 0) both float32 filters run to the end.
//   - An indefinite P is where the methods differ in kind: Cholesky stops,
//     and LDL^T's D shows the inertia. For a matrix that needs pivoting to be
//     factored at all, see ldlt_bk (Bunch-Kaufman).
//
// The sigma-point mean bias is ||sum_k W_k sigma_k - x-hat|| / ||x-hat||. For
// a symmetric set of points the exact weighted mean is x-hat, since the
// weights sum to 1, so the bias measures only the rounding of that sum.
//
// References:
//   Julier & Uhlmann, "Unscented Filtering and Nonlinear Estimation", 2004
//   Bar-Shalom, Li, Kirubarajan, "Estimation with Applications to Tracking
//   and Navigation", Ch. 11
//   Higham, "Accuracy and Stability of Numerical Algorithms", 2nd ed., Thm 10.3

#include <mtl/mtl.hpp>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numbers>
#include <sstream>
#include <string>
#include <vector>

using namespace mtl;

static constexpr std::size_t NX = 4;  // [px, py, vx, vy]
static constexpr std::size_t num_sigma = 2 * NX + 1;

// ============================================================================
// Helpers
// ============================================================================

template <typename T>
mat::dense2D<T> copy_mat(const mat::dense2D<T>& A) {
    mat::dense2D<T> B(A.num_rows(), A.num_cols());
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t j = 0; j < A.num_cols(); ++j)
            B(i, j) = A(i, j);
    return B;
}

/// Deterministic PRNG (xoroshiro128+), so the run is the same everywhere.
struct Rng {
    std::uint64_t s[2];
    explicit Rng(std::uint64_t seed = 42) {
        s[0] = seed ^ 0x9E3779B97F4A7C15ULL;
        s[1] = seed ^ 0x6A09E667F3BCC908ULL;
        for (int i = 0; i < 10; ++i) next();
    }
    std::uint64_t next() {
        std::uint64_t s0 = s[0], s1 = s[1], result = s0 + s1;
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

/// Eigenvalues of a symmetric P, computed in double whatever T is.
template <typename T>
vec::dense_vector<double> eigenvalues_f64(const mat::dense2D<T>& P) {
    mat::dense2D<double> Pd(P.num_rows(), P.num_cols());
    for (std::size_t i = 0; i < P.num_rows(); ++i)
        for (std::size_t j = 0; j < P.num_cols(); ++j)
            Pd(i, j) = static_cast<double>(P(i, j));
    return eigenvalue_symmetric(Pd);
}

/// max |lambda| / min |lambda|, in double.
template <typename T>
double condition_number_f64(const mat::dense2D<T>& P) {
    const auto e = eigenvalues_f64(P);
    double lo = std::abs(e(0)), hi = lo;
    for (std::size_t i = 1; i < e.size(); ++i) {
        lo = std::min(lo, std::abs(e(i)));
        hi = std::max(hi, std::abs(e(i)));
    }
    return lo < std::numeric_limits<double>::min() ? std::numeric_limits<double>::infinity() : hi / lo;
}

template <typename T>
double min_eigenvalue_f64(const mat::dense2D<T>& P) {
    const auto e = eigenvalues_f64(P);
    double lo = e(0);
    for (std::size_t i = 1; i < e.size(); ++i) lo = std::min(lo, e(i));
    return lo;
}

/// a mapped into [-pi, pi].
template <typename T>
T wrap_pi(T a) {
    const double pi = std::numbers::pi;
    while (static_cast<double>(a) > pi) a = a - T(2.0 * pi);
    while (static_cast<double>(a) < -pi) a = a + T(2.0 * pi);
    return a;
}

// ============================================================================
// Sigma points
// ============================================================================

enum class FactMethod { Cholesky, LDLT };

inline const char* name(FactMethod m) { return m == FactMethod::Cholesky ? "Cholesky" : "LDL^T"; }

/// x +- gamma * (column k of a square root of P), from P = L L^T or from
/// P = L D L^T with the column L(:, k) sqrt(D_k). False if P is not SPD to the
/// factorization: Cholesky meets a non-positive pivot, or D has an entry <= 0
/// (an indefinite P has no real square root; L |D| L^T would be another
/// covariance).
template <typename T>
bool make_sigma(FactMethod m, const vec::dense_vector<T>& x, const mat::dense2D<T>& P, T gamma,
                std::vector<vec::dense_vector<T>>& sigma) {
    using std::sqrt;
    auto F = copy_mat(P);
    mat::dense2D<T> S(NX, NX);   // the square root's columns
    if (m == FactMethod::Cholesky) {
        if (cholesky_factor(F) != 0) return false;
        for (std::size_t i = 0; i < NX; ++i)
            for (std::size_t j = 0; j < NX; ++j) S(i, j) = (i >= j) ? F(i, j) : T(0);
    } else {
        if (ldlt_factor(F) != 0) return false;
        for (std::size_t k = 0; k < NX; ++k) {
            if (!(F(k, k) > T(0))) return false;
            const T d = sqrt(F(k, k));
            for (std::size_t i = 0; i < NX; ++i) S(i, k) = (i == k) ? d : (i > k ? F(i, k) * d : T(0));
        }
    }
    sigma[0] = x;
    for (std::size_t k = 0; k < NX; ++k) {
        vec::dense_vector<T> sp(NX), sm(NX);
        for (std::size_t i = 0; i < NX; ++i) {
            sp(i) = x(i) + gamma * S(i, k);
            sm(i) = x(i) - gamma * S(i, k);
        }
        sigma[1 + k] = sp;
        sigma[1 + NX + k] = sm;
    }
    return true;
}

/// ||sum_k W_k sigma_k - x|| / ||x||: the rounding of the weighted mean of a
/// symmetric set of points, whose exact mean is x.
template <typename T>
double sigma_point_bias(const std::vector<vec::dense_vector<T>>& sigma, const vec::dense_vector<T>& x,
                        T Wm0, T Wi) {
    double b2 = 0.0, x2 = 0.0;
    for (std::size_t i = 0; i < NX; ++i) {
        T m = Wm0 * sigma[0](i);
        for (std::size_t s = 1; s < num_sigma; ++s) m = m + Wi * sigma[s](i);
        const double d = static_cast<double>(m - x(i));
        b2 += d * d;
        x2 += static_cast<double>(x(i)) * static_cast<double>(x(i));
    }
    return x2 > 0.0 ? std::sqrt(b2 / x2) : std::sqrt(b2);
}

// ============================================================================
// One scenario, two filters
// ============================================================================

static constexpr double beacon_x = 50.0, beacon_y = 0.0;
static constexpr double dt = 0.1;
static constexpr double R_bearing = 1e-4;   // a tight angular measurement
static constexpr std::array<double, NX> q_diag{1e-6, 1e-6, 1e-4, 1e-4};

/// The truth and the measurements, made once in double so that every filter,
/// in every precision, sees the same ones.
struct Scenario {
    std::vector<std::array<double, NX>> truth;
    std::vector<double> z;
};

Scenario make_scenario(int steps) {
    Scenario sc;
    std::array<double, NX> x{0.0, 5.0, 2.0, -0.5};   // moving toward the beacon
    Rng rng(54321);
    for (int k = 0; k < steps; ++k) {
        x[0] += x[2] * dt;
        x[1] += x[3] * dt;
        for (std::size_t i = 0; i < NX; ++i) x[i] += std::sqrt(q_diag[i]) * rng.randn();
        sc.truth.push_back(x);
        sc.z.push_back(std::atan2(x[1] - beacon_y, x[0] - beacon_x) + std::sqrt(R_bearing) * rng.randn());
    }
    return sc;
}

template <typename T>
struct Weights {
    T gamma, Wm0, Wc0, Wi;
};

/// The scaled unscented transform's weights, formed in T: beta = 2, kappa = 0.
template <typename T>
Weights<T> weights(double alpha_d) {
    const T alpha = T(alpha_d), beta = T(2), kappa = T(0), n = T(NX);
    const T lambda = alpha * alpha * (n + kappa) - n;
    using std::sqrt;
    return {sqrt(n + lambda), lambda / (n + lambda), lambda / (n + lambda) + (T(1) - alpha * alpha + beta),
            T(1) / (T(2) * (n + lambda))};
}

struct Row {
    double cond = 0.0;   // cond(P) at the start of the step
    double bias = 0.0;   // that step's sigma-point mean bias
    double err = 0.0;    // ||x-hat - x_true|| after the update
    std::array<double, NX> x{};
};

struct Run {
    std::vector<Row> rows;      // the steps that completed
    int failed_at = -1;         // the step a factorization refused, or -1
    std::string failed_on;      // which covariance it refused
    double failed_min_eig = 0;  // that covariance's smallest eigenvalue (in double)
};

template <typename T>
Run run_filter(FactMethod method, double alpha, const Scenario& sc) {
    const Weights<T> w = weights<T>(alpha);
    const T R = T(R_bearing), bx = T(beacon_x), by = T(beacon_y), tdt = T(dt);

    vec::dense_vector<T> x_hat(NX);
    x_hat(0) = T(1); x_hat(1) = T(6); x_hat(2) = T(1.5); x_hat(3) = T(0);
    mat::dense2D<T> P(NX, NX);
    for (std::size_t i = 0; i < NX; ++i)
        for (std::size_t j = 0; j < NX; ++j) P(i, j) = (i == j) ? T(100) : T(0);

    Run run;
    std::vector<vec::dense_vector<T>> sigma(num_sigma);
    for (std::size_t k = 0; k < sc.z.size(); ++k) {
        Row row;
        row.cond = condition_number_f64(P);
        if (!make_sigma(method, x_hat, P, w.gamma, sigma)) {
            run.failed_at = int(k);
            run.failed_on = "the incoming P, the last update's P- - K S K^T";
            run.failed_min_eig = min_eigenvalue_f64(P);
            break;
        }
        row.bias = sigma_point_bias(sigma, x_hat, w.Wm0, w.Wi);

        // Predict: constant velocity.
        for (auto& s : sigma) {
            s(0) = s(0) + s(2) * tdt;
            s(1) = s(1) + s(3) * tdt;
        }
        vec::dense_vector<T> x_pred(NX);
        for (std::size_t i = 0; i < NX; ++i) {
            x_pred(i) = w.Wm0 * sigma[0](i);
            for (std::size_t s = 1; s < num_sigma; ++s) x_pred(i) = x_pred(i) + w.Wi * sigma[s](i);
        }
        mat::dense2D<T> P_pred(NX, NX);
        for (std::size_t i = 0; i < NX; ++i)
            for (std::size_t j = 0; j < NX; ++j) {
                T acc = (i == j) ? T(q_diag[i]) : T(0);
                for (std::size_t s = 0; s < num_sigma; ++s)
                    acc = acc + (s == 0 ? w.Wc0 : w.Wi) * (sigma[s](i) - x_pred(i)) * (sigma[s](j) - x_pred(j));
                P_pred(i, j) = acc;
            }

        // Update: sigma points of the prediction, through the bearing.
        if (!make_sigma(method, x_pred, P_pred, w.gamma, sigma)) {
            run.failed_at = int(k);
            run.failed_on = "the predicted P";
            run.failed_min_eig = min_eigenvalue_f64(P_pred);
            break;
        }
        std::vector<T> zs(num_sigma);
        for (std::size_t s = 0; s < num_sigma; ++s) {
            using std::atan2;
            zs[s] = atan2(sigma[s](1) - by, sigma[s](0) - bx);
        }
        // Wrap each bearing relative to the central point's before any sum: a
        // set that straddles atan2's cut at +-pi would otherwise hold bearings
        // almost 2 pi apart, and its mean and spread would be meaningless.
        for (std::size_t s = 1; s < num_sigma; ++s) zs[s] = zs[0] + wrap_pi(T(zs[s] - zs[0]));
        T z_pred = w.Wm0 * zs[0];
        for (std::size_t s = 1; s < num_sigma; ++s) z_pred = z_pred + w.Wi * zs[s];
        T S = R;
        vec::dense_vector<T> Pxz(NX, T(0));
        for (std::size_t s = 0; s < num_sigma; ++s) {
            const T ws = (s == 0) ? w.Wc0 : w.Wi;
            const T dz = zs[s] - z_pred;
            S = S + ws * dz * dz;
            for (std::size_t i = 0; i < NX; ++i) Pxz(i) = Pxz(i) + ws * (sigma[s](i) - x_pred(i)) * dz;
        }
        const T innov = wrap_pi(T(T(sc.z[k]) - z_pred));
        for (std::size_t i = 0; i < NX; ++i) x_hat(i) = x_pred(i) + Pxz(i) / S * innov;
        for (std::size_t i = 0; i < NX; ++i)
            for (std::size_t j = 0; j < NX; ++j) P(i, j) = P_pred(i, j) - Pxz(i) * Pxz(j) / S;
        for (std::size_t i = 0; i < NX; ++i)
            for (std::size_t j = i + 1; j < NX; ++j) P(j, i) = P(i, j);

        double e2 = 0.0;
        for (std::size_t i = 0; i < NX; ++i) {
            row.x[i] = static_cast<double>(x_hat(i));
            const double d = row.x[i] - sc.truth[k][i];
            e2 += d * d;
        }
        row.err = std::sqrt(e2);
        run.rows.push_back(row);
    }
    return run;
}

// ============================================================================
// Reporting
// ============================================================================

struct Pair {
    Run chol, ldlt;
};

template <typename T>
Pair run_both(double alpha, const Scenario& sc) {
    return {run_filter<T>(FactMethod::Cholesky, alpha, sc), run_filter<T>(FactMethod::LDLT, alpha, sc)};
}

/// The largest ||x-hat_Cholesky - x-hat_LDLT|| over the steps both completed.
double max_gap(const Pair& p) {
    double g = 0.0;
    const std::size_t m = std::min(p.chol.rows.size(), p.ldlt.rows.size());
    for (std::size_t k = 0; k < m; ++k) {
        double d2 = 0.0;
        for (std::size_t i = 0; i < NX; ++i) {
            const double d = p.chol.rows[k].x[i] - p.ldlt.rows[k].x[i];
            d2 += d * d;
        }
        g = std::max(g, std::sqrt(d2));
    }
    return g;
}

std::string outcome(const Run& r, int steps) {
    std::ostringstream o;
    if (r.failed_at < 0)
        o << "all " << steps << " steps, final error " << std::fixed << std::setprecision(3) << r.rows.back().err;
    else
        o << "stopped at step " << r.failed_at << ", on " << r.failed_on << " (lambda_min = " << std::scientific
          << std::setprecision(2) << r.failed_min_eig << ")";
    return o.str();
}

void print_pair(const std::string& title, const Pair& p, int steps) {
    std::cout << "\n=== " << title << " ===\n";
    std::cout << std::setw(5) << "step" << std::setw(11) << "cond(P)" << std::setw(12) << "bias Chol"
              << std::setw(12) << "bias LDLT" << std::setw(13) << "|xC - xL|" << std::setw(11) << "err Chol"
              << std::setw(11) << "err LDLT" << "\n"
              << std::string(75, '-') << "\n";
    const std::size_t m = std::max(p.chol.rows.size(), p.ldlt.rows.size());
    for (std::size_t k = 0; k < m; ++k) {
        const bool c = k < p.chol.rows.size(), l = k < p.ldlt.rows.size();
        const Row& r = c ? p.chol.rows[k] : p.ldlt.rows[k];
        std::cout << std::setw(5) << k << std::setw(11) << std::scientific << std::setprecision(1) << r.cond
                  << std::setprecision(2);
        if (c) std::cout << std::setw(12) << p.chol.rows[k].bias; else std::cout << std::setw(12) << "-";
        if (l) std::cout << std::setw(12) << p.ldlt.rows[k].bias; else std::cout << std::setw(12) << "-";
        if (c && l) {
            double d2 = 0.0;
            for (std::size_t i = 0; i < NX; ++i) {
                const double d = p.chol.rows[k].x[i] - p.ldlt.rows[k].x[i];
                d2 += d * d;
            }
            std::cout << std::setw(13) << std::sqrt(d2);
        } else std::cout << std::setw(13) << "-";
        std::cout << std::fixed << std::setprecision(3);
        if (c) std::cout << std::setw(11) << p.chol.rows[k].err; else std::cout << std::setw(11) << "-";
        if (l) std::cout << std::setw(11) << p.ldlt.rows[k].err; else std::cout << std::setw(11) << "-";
        std::cout << "\n";
    }
    std::cout << "\n  Cholesky: " << outcome(p.chol, steps) << "\n  LDL^T:    " << outcome(p.ldlt, steps) << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main() {
    constexpr int steps = 15;
    const Scenario sc = make_scenario(steps);

    std::cout << "================================================================\n"
              << " Bearing-only UKF: Cholesky against LDL^T, float64 and float32\n"
              << "================================================================\n\n"
              << "State [px, py, vx, vy]; one measurement, the bearing from a beacon at (50, 0),\n"
              << "R = 1e-4. Two filters per precision, one per factorization, on the same truth\n"
              << "and the same measurements. bias is the sigma-point mean's rounding\n"
              << "(||sum W_k sigma_k - x|| / ||x||); |xC - xL| is how far the two filters'\n"
              << "estimates are apart; err is ||x-hat - x_true||.\n";

    const double alpha = 1e-3;
    const Pair f64 = run_both<double>(alpha, sc);
    const Pair f32 = run_both<float>(alpha, sc);   // the sweep below repeats them
    print_pair("float64, alpha = 1e-3", f64, steps);
    print_pair("float32, alpha = 1e-3", f32, steps);

    // -- the weights ---------------------------------------------------------
    // The same two filters at four alphas, in both precisions. The gap between
    // them is rounding: the two factorizations give the same P^{1/2} up to
    // rounding, and the weights amplify it by about |W_0|.
    struct Sweep {
        double alpha;
        Weights<float> w32;
        Pair f64, f32;
    };
    std::vector<Sweep> sweep;
    for (const double a : {1e-3, 1e-2, 1e-1, 1.0})
        sweep.push_back({a, weights<float>(a), run_both<double>(a, sc), run_both<float>(a, sc)});
    auto steps_of = [&](const Run& r) {
        return r.failed_at < 0 ? std::string("all ") + std::to_string(steps)
                               : std::string("stops at ") + std::to_string(r.failed_at);
    };
    std::cout << "\n=== The weights, and both filters, by alpha ===\n"
              << "lambda = alpha^2 n - n, W_0 = lambda / (n + lambda), W_i = 1 / (2 (n + lambda)), n = 4.\n"
              << "Each weighted sum adds terms about |W_0| times its result, which cancel. gap is the\n"
              << "largest |xC - xL| over the run.\n\n"
              << std::setw(7) << "alpha" << std::setw(13) << "W_0 (f32)" << std::setw(11) << "W_i (f32)"
              << std::setw(13) << "f32 bias_0" << std::setw(11) << "f64 gap" << std::setw(11) << "f32 gap"
              << "   f32 Cholesky / LDL^T\n"
              << std::string(94, '-') << "\n";
    for (const auto& sw : sweep)
        std::cout << std::setw(7) << std::defaultfloat << sw.alpha << std::setw(13) << std::setprecision(6)
                  << static_cast<double>(sw.w32.Wm0) << std::setw(11) << static_cast<double>(sw.w32.Wi)
                  << std::scientific << std::setprecision(1) << std::setw(13)
                  << (sw.f32.chol.rows.empty() ? 0.0 : sw.f32.chol.rows[0].bias) << std::setw(11)
                  << max_gap(sw.f64) << std::setw(11) << max_gap(sw.f32) << "   " << steps_of(sw.f32.chol)
                  << " / " << steps_of(sw.f32.ldlt) << "\n";
    const Sweep& lo = sweep.front();
    const Sweep& hi = sweep.back();
    print_pair("float32, alpha = 1", hi.f32, steps);

    // -- conclusion, from the runs ------------------------------------------
    auto max_cond = [](const Run& r) {
        double c = 0.0;
        for (const auto& row : r.rows) c = std::max(c, row.cond);
        return c;
    };
    std::cout << "\n================================================================\n"
              << " Conclusion (computed from the runs above)\n"
              << "================================================================\n\n"
              << std::scientific << std::setprecision(1);
    std::cout << "- float64: the Cholesky and LDL^T filters agree to " << max_gap(f64) << " at alpha = 1e-3, and to "
              << max_gap(hi.f64) << "\n  at alpha = 1, with cond(P) up to " << max_cond(f64.chol)
              << ". The two factorizations give the same filter; the gap is\n  rounding, amplified by the weights.\n";
    std::cout << "- float32, alpha = 1e-3: ";
    if (f32.chol.failed_at >= 0 || f32.ldlt.failed_at >= 0) {
        std::cout << "Cholesky " << outcome(f32.chol, steps) << ";\n  LDL^T " << outcome(f32.ldlt, steps) << ".\n";
        if (f32.chol.failed_at == f32.ldlt.failed_at)
            std::cout << "  Both stop at the same step: P is no longer positive definite, and no real square\n"
                         "  root of it exists, by either method.\n";
    } else {
        std::cout << "the step-0 bias is already " << lo.f32.chol.rows[0].bias << " (P = 100 I, cond 1), and\n"
                  << "  the two filters drift up to " << max_gap(f32) << " apart, while cond(P) peaks at "
                  << max_cond(f32.chol) << ".\n";
    }
    const bool a1_ok = hi.f32.chol.failed_at < 0 && hi.f32.ldlt.failed_at < 0;
    std::cout << "- It follows alpha, not the factorization: as alpha goes from 1e-3 to 1, the float32\n"
              << "  step-0 bias falls from " << lo.f32.chol.rows[0].bias << " to " << hi.f32.chol.rows[0].bias
              << ", and the gap between the filters\n  from " << max_gap(lo.f32) << " to " << max_gap(hi.f32) << ".";
    if (a1_ok)
        std::cout << " With alpha = 1 both float32 filters end at error " << std::fixed << std::setprecision(3)
                  << hi.f32.chol.rows.back().err << " / " << hi.f32.ldlt.rows.back().err << ",\n  against "
                  << hi.f64.chol.rows.back().err << " in float64.";
    std::cout << "\n- In low precision, keep W_0 small: alpha near 1, or kappa chosen so that n + lambda is\n"
              << "  not tiny. The factorization is not the lever.\n"
              << "- LDL^T does not avoid the square root here: its sigma points are x-hat +- gamma L\n"
              << "  sqrt(D_k) e_k. Where it differs is an indefinite P: Cholesky stops, while D shows\n"
              << "  the inertia (and ldlt_bk, Bunch-Kaufman, factors a matrix that needs pivoting).\n";
    return 0;
}
