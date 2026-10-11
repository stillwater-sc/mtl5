// The textbook Householder reflector (#559).
//
//   householder(x, tag::textbook_reflector{})                 -> {v, vtv, alpha}
//   apply_householder_left(A, v, vtv, row, col, tag::textbook_reflector{})
//   apply_householder_right(A, v, vtv, row, col, tag::textbook_reflector{})
//   qr_factor(A, tau, tag::textbook_reflector{})
//   lq_factor(A, tau, tag::textbook_reflector{})
//
// The tagged paths exist so a caller whose results are pinned to a hand-written
// textbook QR can move to MTL5 without changing a bit, so the tests compare BITS
// (==) against in-test serial reference loops, not values within a tolerance.
// Three properties make that comparison meaningful:
//   - The pool is sized to 4 threads before first use, so the column update runs
//     threaded while the references run serial.
//   - The tagged qr_factor never dispatches to LAPACK, so the double and float
//     cases hold in MTL5_HAS_LAPACK builds as well.
//   - The target is built with -ffp-contract=off (tests/unit/CMakeLists.txt).
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>

#include <mtl/concepts/scalar.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/householder.hpp>
#include <mtl/operation/qr.hpp>
#include <mtl/operation/lq.hpp>
#include <mtl/detail/thread_pool.hpp>
#include <mtl/tag/reflector_form.hpp>

using namespace mtl;

namespace {

// Size the process-wide pool multithreaded before its first (lazy) use.
const int g_set_threads = [] {
#if defined(_WIN32)
    _putenv_s("MTL5_NUM_THREADS", "4");
#else
    setenv("MTL5_NUM_THREADS", "4", /*overwrite=*/1);
#endif
    return 0;
}();

/// A class-typed scalar wrapping a double: no BLAS, no LAPACK, no SIMD lane.
struct emul {
    double v = 0.0;
    emul() = default;
    constexpr emul(double d) : v(d) {}
    friend emul operator+(emul a, emul b) { return a.v + b.v; }
    friend emul operator-(emul a, emul b) { return a.v - b.v; }
    friend emul operator*(emul a, emul b) { return a.v * b.v; }
    friend emul operator/(emul a, emul b) { return a.v / b.v; }
    friend emul operator-(emul a) { return -a.v; }
    emul& operator+=(emul o) { v += o.v; return *this; }
    emul& operator-=(emul o) { v -= o.v; return *this; }
    emul& operator*=(emul o) { v *= o.v; return *this; }
    emul& operator/=(emul o) { v /= o.v; return *this; }
    friend bool operator==(emul a, emul b) { return a.v == b.v; }
    friend auto operator<=>(emul a, emul b) { return a.v <=> b.v; }
    friend emul sqrt(emul a) { return std::sqrt(a.v); }
    friend emul abs(emul a) { return std::abs(a.v); }
};
static_assert(!std::is_arithmetic_v<emul> && Field<emul>);

double bits_of(double x) { return x; }
double bits_of(emul x) { return x.v; }
float  bits_of(float x) { return x; }

template <typename T>
bool same_bits(const T& a, const T& b) {
    const auto x = bits_of(a), y = bits_of(b);
    return std::memcmp(&x, &y, sizeof(x)) == 0;
}

// ---- Serial references, written as a hand-written port has them ----------

template <typename T>
struct ref_reflector { std::vector<T> v; T vtv; T alpha; bool skip; };

template <typename T>
ref_reflector<T> ref_householder(const std::vector<T>& x) {
    using std::sqrt;
    ref_reflector<T> r{x, T(0), T(0), true};
    if (x.empty()) return r;
    T sum = T(0);
    for (const T& xi : x) sum += xi * xi;
    const T norm = sqrt(sum);
    if (!(norm > T(0))) return r;
    r.alpha = x[0] >= T(0) ? T(-norm) : norm;
    r.v[0] = x[0] - r.alpha;
    T vtv = T(0);
    for (const T& vi : r.v) vtv += vi * vi;
    r.vtv = vtv;
    r.skip = !(vtv > T(0));
    return r;
}

template <typename T>
void ref_apply_left(mat::dense2D<T>& A, const std::vector<T>& v, T vtv,
                    std::size_t row, std::size_t col) {
    if (!(vtv > T(0))) return;
    for (std::size_t j = col; j < A.num_cols(); ++j) {
        T s = T(0);
        for (std::size_t i = 0; i < v.size(); ++i) s += v[i] * A(row + i, j);
        const T f = (T(2) * s) / vtv;
        for (std::size_t i = 0; i < v.size(); ++i) A(row + i, j) -= f * v[i];
    }
}

template <typename T>
void ref_apply_right(mat::dense2D<T>& A, const std::vector<T>& v, T vtv,
                     std::size_t row, std::size_t col) {
    if (!(vtv > T(0))) return;
    for (std::size_t i = row; i < A.num_rows(); ++i) {
        T s = T(0);
        for (std::size_t j = 0; j < v.size(); ++j) s += A(i, col + j) * v[j];
        const T f = (T(2) * s) / vtv;
        for (std::size_t j = 0; j < v.size(); ++j) A(i, col + j) -= f * v[j];
    }
}

/// Textbook LQ, L only: L in the lower triangle, exact zeros above.
template <typename T>
void ref_lq_l(mat::dense2D<T>& A) {
    const std::size_t m = A.num_rows(), n = A.num_cols();
    for (std::size_t k = 0; k < std::min(m, n); ++k) {
        std::vector<T> x(n - k);
        for (std::size_t j = k; j < n; ++j) x[j - k] = A(k, j);
        auto r = ref_householder(x);
        if (r.skip) continue;
        ref_apply_right(A, r.v, r.vtv, k + 1, k);
        A(k, k) = r.alpha;
        for (std::size_t j = k + 1; j < n; ++j) A(k, j) = T(0);
    }
}

/// Textbook QR, R only: R in the upper triangle, exact zeros below.
template <typename T>
void ref_qr_r(mat::dense2D<T>& A) {
    const std::size_t m = A.num_rows(), n = A.num_cols();
    for (std::size_t k = 0; k < std::min(m, n); ++k) {
        std::vector<T> x(m - k);
        for (std::size_t i = k; i < m; ++i) x[i - k] = A(i, k);
        auto r = ref_householder(x);
        if (r.skip) continue;
        ref_apply_left(A, r.v, r.vtv, k, k + 1);
        A(k, k) = r.alpha;
        for (std::size_t i = k + 1; i < m; ++i) A(i, k) = T(0);
    }
}

template <typename T>
mat::dense2D<T> random_matrix(std::size_t m, std::size_t n, std::uint64_t seed) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    mat::dense2D<T> A(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = T(u(g));
    return A;
}

template <typename T>
std::size_t upper_mismatches(const mat::dense2D<T>& A, const mat::dense2D<T>& B) {
    std::size_t bad = 0;
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t j = i; j < A.num_cols(); ++j)
            bad += !same_bits(A(i, j), B(i, j));
    return bad;
}

template <typename T>
std::size_t all_mismatches(const mat::dense2D<T>& A, const mat::dense2D<T>& B) {
    std::size_t bad = 0;
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t j = 0; j < A.num_cols(); ++j)
            bad += !same_bits(A(i, j), B(i, j));
    return bad;
}

template <typename T>
std::size_t lower_mismatches(const mat::dense2D<T>& A, const mat::dense2D<T>& B) {
    std::size_t bad = 0;
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t j = 0; j <= i && j < A.num_cols(); ++j)
            bad += !same_bits(A(i, j), B(i, j));
    return bad;
}

template <typename T>
void check_lq(std::size_t m, std::size_t n, std::uint64_t seed) {
    auto A = random_matrix<T>(m, n, seed), L = A;
    vec::dense_vector<T> tau;
    REQUIRE(lq_factor(A, tau, tag::textbook_reflector{}) == 0);
    ref_lq_l(L);
    REQUIRE(lower_mismatches(A, L) == 0);
}

template <typename T>
void check_reflector(const std::vector<T>& x) {
    vec::dense_vector<T> xv(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) xv(i) = x[i];
    auto [v, vtv, alpha] = householder(xv, tag::textbook_reflector{});
    auto r = ref_householder(x);
    REQUIRE(same_bits(vtv, r.skip ? T(0) : r.vtv));
    if (r.skip) return;
    REQUIRE(same_bits(alpha, r.alpha));
    REQUIRE(v.size() == r.v.size());
    for (std::size_t i = 0; i < x.size(); ++i)
        REQUIRE(same_bits(v(i), r.v[i]));
}

template <typename T>
void check_qr(std::size_t m, std::size_t n, std::uint64_t seed) {
    auto A = random_matrix<T>(m, n, seed), R = A;
    vec::dense_vector<T> tau;
    REQUIRE(qr_factor(A, tau, tag::textbook_reflector{}) == 0);
    ref_qr_r(R);
    REQUIRE(upper_mismatches(A, R) == 0);
}

} // namespace

TEST_CASE("textbook householder equals the reference reflector bit for bit",
          "[operation][householder][textbook][559]") {
    // x0 positive, negative and zero; a tail of zeros (not skipped: the
    // reflector flips the sign of x0); and the empty vector.
    check_reflector<double>({3.0, 1.0, -2.0, 0.5});
    check_reflector<double>({-3.0, 1.0, -2.0, 0.5});
    check_reflector<double>({0.0, 1.0, -2.0, 0.5});
    check_reflector<double>({2.5, 0.0, 0.0});
    check_reflector<double>({-2.5, 0.0, 0.0});
    check_reflector<double>({});
    for (std::uint64_t s = 0; s < 8; ++s) {
        std::mt19937_64 g(5590 + s);
        std::uniform_real_distribution<double> u(-1.0, 1.0);
        std::vector<double> xd(1 + s * 7);
        for (auto& e : xd) e = u(g);
        std::vector<float> xf(xd.begin(), xd.end());
        std::vector<emul> xe(xd.begin(), xd.end());
        check_reflector(xd);
        check_reflector(xf);
        check_reflector(xe);
    }

    // The textbook alpha is -sign(x0)*||x||: R's diagonal is NOT forced positive.
    vec::dense_vector<double> x{3.0, 4.0};
    auto [v, vtv, alpha] = householder(x, tag::textbook_reflector{});
    REQUIRE(alpha == -5.0);
    REQUIRE(v(0) == 8.0);
    REQUIRE(vtv == 80.0);
}

TEST_CASE("textbook householder marks zero and NaN columns as skipped",
          "[operation][householder][textbook][nan][559]") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (const auto& x : {std::vector<double>{0.0, 0.0, 0.0},
                          std::vector<double>{1.0, nan, 2.0},
                          std::vector<double>{nan}}) {
        vec::dense_vector<double> xv(x.size());
        for (std::size_t i = 0; i < x.size(); ++i) xv(i) = x[i];
        auto h = householder(xv, tag::textbook_reflector{});
        REQUIRE(h.vtv == 0.0);
        REQUIRE(ref_householder(x).skip);

        // ... and applying a skipped reflector leaves A untouched.
        auto A = random_matrix<double>(x.size(), 4, 7), B = A;
        apply_householder_left(A, h.v, h.vtv, 0, 0, tag::textbook_reflector{});
        REQUIRE(all_mismatches(A, B) == 0);
    }
}

TEST_CASE("textbook apply_householder_left equals the reference update bit for bit",
          "[operation][householder][textbook][559]") {
    if (detail::thread_pool::instance().size() < 2)
        WARN("single-core runner: threading not exercised");

    auto run = [](auto tag_value, std::size_t m, std::size_t n, std::size_t row,
                  std::size_t col, std::uint64_t seed) {
        using T = decltype(tag_value);
        auto A = random_matrix<T>(m, n, seed), B = A;
        std::vector<T> x(m - row);
        for (std::size_t i = row; i < m; ++i) x[i - row] = A(i, col);
        auto r = ref_householder(x);
        vec::dense_vector<T> v(r.v.size());
        for (std::size_t i = 0; i < r.v.size(); ++i) v(i) = r.v[i];
        apply_householder_left(A, v, r.vtv, row, col, tag::textbook_reflector{});
        ref_apply_left(B, r.v, r.vtv, row, col);
        REQUIRE(all_mismatches(A, B) == 0);
    };
    run(double{}, 9, 6, 0, 0, 1);
    run(double{}, 9, 6, 3, 2, 2);
    run(float{}, 17, 11, 4, 1, 3);
    run(emul{}, 17, 11, 2, 5, 4);
    // The cortex shape: three reflectors applied across an augmented [H_f | H_x | r].
    run(double{}, 40, 3 + 81 + 1, 1, 1, 5);
    // Enough columns that the update splits across the pool.
    run(double{}, 300, 2000, 0, 0, 6);
}

TEST_CASE("qr_factor with the textbook reflector produces the reference R bit for bit",
          "[operation][qr][textbook][559]") {
    for (auto [m, n] : {std::pair<std::size_t, std::size_t>{1, 1}, {2, 2}, {5, 3},
                        {15, 15}, {33, 15}, {81, 81}}) {
        INFO("m = " << m << ", n = " << n);
        check_qr<double>(m, n, 559 + m * 100 + n);
        check_qr<float>(m, n, 560 + m * 100 + n);
        check_qr<emul>(m, n, 561 + m * 100 + n);
    }
    check_qr<double>(256, 200, 5591);
}

TEST_CASE("qr_factor with the textbook reflector skips zero columns and keeps sign",
          "[operation][qr][textbook][559]") {
    //   column 0 is entirely zero         -> skipped: tau 0, column untouched
    //   column 1 is zero below the pivot  -> NOT skipped: H = I - 2*e1*e1^T
    //                                        reflects R(1,1) = 2 onto -2, and
    //                                        negates the rest of row 1
    //   column 2 then has the 1x1 x = [4] -> reflected onto -4
    // All exact, so the expected values are known, not just equal to ref_qr_r.
    mat::dense2D<double> A{{0.0, 1.0, 1.0},
                           {0.0, 2.0, 3.0},
                           {0.0, 0.0, 4.0}};
    auto R = A;
    vec::dense_vector<double> tau;
    REQUIRE(qr_factor(A, tau, tag::textbook_reflector{}) == 0);
    ref_qr_r(R);
    REQUIRE(upper_mismatches(A, R) == 0);

    REQUIRE(tau(0) == 0.0);
    REQUIRE(A(0, 0) == 0.0);
    REQUIRE(A(0, 1) == 1.0);
    REQUIRE(A(0, 2) == 1.0);
    REQUIRE(tau(1) == 2.0);
    REQUIRE(A(1, 1) == -2.0);
    REQUIRE(A(1, 2) == -3.0);
    REQUIRE(tau(2) == 2.0);
    REQUIRE(A(2, 2) == -4.0);
}

TEST_CASE("qr_factor's textbook result works with qr_extract_Q and qr_solve",
          "[operation][qr][textbook][559]") {
    const std::size_t m = 12, n = 7;
    auto A0 = random_matrix<double>(m, n, 5592), A = A0;
    vec::dense_vector<double> tau;
    REQUIRE(qr_factor(A, tau, tag::textbook_reflector{}) == 0);
    auto Q = qr_extract_Q(A, tau);
    auto R = qr_extract_R(A);

    double err = 0.0, orth = 0.0;
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < m; ++k) s += Q(i, k) * R(k, j);
            err = std::max(err, std::abs(s - A0(i, j)));
        }
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < m; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < m; ++k) s += Q(k, i) * Q(k, j);
            orth = std::max(orth, std::abs(s - (i == j ? 1.0 : 0.0)));
        }
    REQUIRE(err < 1e-13);
    REQUIRE(orth < 1e-13);

    // A square system, solved exactly to rounding.
    auto S0 = random_matrix<double>(n, n, 5593), S = S0;
    vec::dense_vector<double> x_true(n), b(n, 0.0), x(n);
    for (std::size_t i = 0; i < n; ++i) x_true(i) = 1.0 + double(i);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) b(i) += S0(i, j) * x_true(j);
    REQUIRE(qr_factor(S, tau, tag::textbook_reflector{}) == 0);
    qr_solve(S, tau, x, b);
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(std::abs(x(i) - x_true(i)) < 1e-11);
}

TEST_CASE("qr_solve skips a column the textbook qr_factor skipped",
          "[operation][qr][textbook][nan][559]") {
    // Column 0 has a NaN below the diagonal, so it is skipped: tau(0) == 0 and
    // the NaN stays in the packed storage. The identity reflection must not
    // read it, or y -= 0*v*dot turns the whole right-hand side into NaN.
    const double nan = std::numeric_limits<double>::quiet_NaN();
    mat::dense2D<double> A{{2.0, 1.0, 0.0},
                           {0.0, 3.0, 1.0},
                           {nan, 1.0, 4.0}};
    vec::dense_vector<double> tau, b{1.0, 2.0, 3.0}, x(3);
    REQUIRE(qr_factor(A, tau, tag::textbook_reflector{}) == 0);
    REQUIRE(tau(0) == 0.0);
    REQUIRE(std::isnan(A(2, 0)));
    qr_solve(A, tau, x, b);
    for (std::size_t i = 0; i < 3; ++i) {
        INFO("i = " << i);
        REQUIRE(std::isfinite(x(i)));
    }
    // ... and x solves the system that was factored: column 0 as stored
    // above the NaN, i.e. R x = Q^T b with Q built from the other reflectors.
    auto Q = qr_extract_Q(A, tau);
    auto R = qr_extract_R(A);
    for (std::size_t i = 0; i < 3; ++i) {
        double qx = 0.0;
        for (std::size_t k = 0; k < 3; ++k) {
            double rx = 0.0;
            for (std::size_t j = 0; j < 3; ++j) rx += R(k, j) * x(j);
            qx += Q(i, k) * rx;
        }
        REQUIRE(std::abs(qx - b(i)) < 1e-13);
    }
}

TEST_CASE("the textbook and default reflectors differ",
          "[operation][qr][textbook][559]") {
    // Otherwise the tag would be selecting the same arithmetic. emul keeps the
    // default on the generic path in LAPACK builds too.
    auto A = random_matrix<emul>(33, 15, 5594), B = A;
    vec::dense_vector<emul> ta, tb;
    REQUIRE(qr_factor(A, ta) == 0);
    REQUIRE(qr_factor(B, tb, tag::textbook_reflector{}) == 0);
    REQUIRE(upper_mismatches(A, B) > 0);
}

TEST_CASE("textbook apply_householder_right equals the reference update bit for bit",
          "[operation][householder][textbook][559]") {
    auto run = [](auto tag_value, std::size_t m, std::size_t n, std::size_t row,
                  std::size_t col, std::uint64_t seed) {
        using T = decltype(tag_value);
        auto A = random_matrix<T>(m, n, seed), B = A;
        std::vector<T> x(n - col);
        for (std::size_t j = col; j < n; ++j) x[j - col] = A(row, j);
        auto r = ref_householder(x);
        vec::dense_vector<T> v(r.v.size());
        for (std::size_t j = 0; j < r.v.size(); ++j) v(j) = r.v[j];
        apply_householder_right(A, v, r.vtv, row, col, tag::textbook_reflector{});
        ref_apply_right(B, r.v, r.vtv, row, col);
        REQUIRE(all_mismatches(A, B) == 0);
    };
    run(double{}, 6, 9, 0, 0, 11);
    run(double{}, 6, 9, 2, 3, 12);
    run(float{}, 11, 17, 1, 4, 13);
    run(emul{}, 11, 17, 5, 2, 14);
    // Enough rows that the update splits across the pool.
    run(double{}, 2000, 300, 0, 0, 15);

    // A skipped reflector leaves A untouched.
    auto A = random_matrix<double>(4, 3, 16), B = A;
    vec::dense_vector<double> z{0.0, 0.0, 0.0};
    auto h = householder(z, tag::textbook_reflector{});
    apply_householder_right(A, h.v, h.vtv, 0, 0, tag::textbook_reflector{});
    REQUIRE(all_mismatches(A, B) == 0);
}

TEST_CASE("lq_factor with the textbook reflector produces the reference L bit for bit",
          "[operation][lq][textbook][559]") {
    for (auto [m, n] : {std::pair<std::size_t, std::size_t>{1, 1}, {2, 2}, {3, 5},
                        {15, 15}, {15, 33}, {81, 81}}) {
        INFO("m = " << m << ", n = " << n);
        check_lq<double>(m, n, 659 + m * 100 + n);
        check_lq<float>(m, n, 660 + m * 100 + n);
        check_lq<emul>(m, n, 661 + m * 100 + n);
    }
    check_lq<double>(200, 256, 6591);

    // The transpose of the QR case: exact, so the expected values are known.
    mat::dense2D<double> A{{0.0, 0.0, 0.0},
                           {1.0, 2.0, 0.0},
                           {1.0, 3.0, 4.0}};
    auto L = A;
    vec::dense_vector<double> tau;
    REQUIRE(lq_factor(A, tau, tag::textbook_reflector{}) == 0);
    ref_lq_l(L);
    REQUIRE(lower_mismatches(A, L) == 0);
    REQUIRE(tau(0) == 0.0);
    REQUIRE(A(1, 0) == 1.0);
    REQUIRE(tau(1) == 2.0);
    REQUIRE(A(1, 1) == -2.0);
    REQUIRE(A(2, 1) == -3.0);
    REQUIRE(tau(2) == 2.0);
    REQUIRE(A(2, 2) == -4.0);
}

TEST_CASE("lq_factor's textbook result works with lq_extract_Q",
          "[operation][lq][textbook][559]") {
    const std::size_t m = 7, n = 12;
    auto A0 = random_matrix<double>(m, n, 6592), A = A0;
    vec::dense_vector<double> tau;
    REQUIRE(lq_factor(A, tau, tag::textbook_reflector{}) == 0);
    auto Q = lq_extract_Q(A, tau);
    auto L = lq_extract_L(A);

    double err = 0.0, orth = 0.0;
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < n; ++k) s += L(i, k) * Q(k, j);
            err = std::max(err, std::abs(s - A0(i, j)));
        }
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0.0;
            for (std::size_t k = 0; k < n; ++k) s += Q(i, k) * Q(j, k);
            orth = std::max(orth, std::abs(s - (i == j ? 1.0 : 0.0)));
        }
    REQUIRE(err < 1e-13);
    REQUIRE(orth < 1e-13);

    // Otherwise the tag would be selecting the same arithmetic.
    auto B = random_matrix<emul>(15, 33, 6594), C = B;
    vec::dense_vector<emul> tb, tc;
    REQUIRE(lq_factor(B, tb) == 0);
    REQUIRE(lq_factor(C, tc, tag::textbook_reflector{}) == 0);
    REQUIRE(lower_mismatches(B, C) > 0);
}
