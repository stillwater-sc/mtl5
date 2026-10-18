// MTL5 -- Accumulator policy for svd (#585).
//
//   reach     -- a counting accumulator sees the inner products of every
//                overload (Jacobi pair sums, column norms, the U completion,
//                the QR preconditioner); the void default sees none;
//   identity  -- a plain element-type accumulator gives the void path's bits:
//                both start each sum at zero and add the same products in the
//                same order. Built with -ffp-contract=off for that reason;
//   use       -- wider and exact accumulators still give a valid SVD.
//
// [586:svd]: NO ORDERING, and why. The singular values of a float A, measured
// against an exact-accumulated double SVD of the same (float) entries, max
// relative error over all singular values (plain / fma / exact / double):
//
//   Hilbert(4):  1.4e-6 / 1.4e-6 / 9.2e-5 / 9.2e-5
//   Hilbert(6):  5.6e-2 / 5.7e-2 / 3.1e-2 / 3.1e-2
//   Hilbert(8):  3.8e+0 / 3.5e-1 / 3.0e+0 / 3.0e+0
//   sin, n = 8:  1.0e-6 / 4.1e-6 / 9.1e-6 / 9.1e-6
//
// One-sided Jacobi's error lives in APPLYING the plane rotations,
// b_p' = c b_p - s b_q, rounded in float at every sweep; the inner products
// alpha, betac, gamma only choose the rotation angle, and an exact angle
// applied in float is no better than a rounded one. (Exact and double agree
// to the bit: both round sums that float could not distinguish anyway.) The
// same shape as the Krylov and symmetric-QR rows of #586: the accumulator
// covers the reductions, and here the error is not in them.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/svd.hpp>

#include "exact_accumulator.hpp"

using namespace mtl;

namespace counting {

struct count_acc { double s; };
inline std::size_t products = 0;

inline void accumulator_clear(count_acc& a) { a.s = 0.0; }
template <typename V>
void accumulator_assign(count_acc& a, const V& v) { a.s = static_cast<double>(v); }
template <typename V>
void accumulator_add_product(count_acc& a, const V& m, const V& v) {
    ++products;
    a.s += static_cast<double>(m) * static_cast<double>(v);
}
template <typename R>
R accumulator_value(const count_acc& a, std::type_identity<R>) { return static_cast<R>(a.s); }

} // namespace counting

namespace {

template <typename T>
mat::dense2D<T> fixture(std::size_t m, std::size_t n) {
    mat::dense2D<T> A(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = T(std::sin(0.7 * double(i) + 1.3 * double(j) + 0.1) + (i == j ? 0.5 : 0.0));
    return A;
}

template <typename T>
bool same_bits(const mat::dense2D<T>& X, const mat::dense2D<T>& Y) {
    if (X.num_rows() != Y.num_rows() || X.num_cols() != Y.num_cols()) return false;
    for (std::size_t i = 0; i < X.num_rows(); ++i)
        for (std::size_t j = 0; j < X.num_cols(); ++j) {
            const T x = X(i, j), y = Y(i, j);
            if (std::memcmp(&x, &y, sizeof(T)) != 0) return false;
        }
    return true;
}

template <typename T>
bool same_bits(const vec::dense_vector<T>& x, const vec::dense_vector<T>& y) {
    if (x.size() != y.size()) return false;
    for (std::size_t i = 0; i < x.size(); ++i)
        if (std::memcmp(&x(i), &y(i), sizeof(T)) != 0) return false;
    return true;
}

/// max |A V - U diag(s)| and max |U^T U - I| for a thin SVD, in double.
template <typename T>
std::pair<double, double> thin_errors(const mat::dense2D<T>& A, const mat::dense2D<T>& U,
                                      const vec::dense_vector<T>& s, const mat::dense2D<T>& V) {
    const std::size_t m = A.num_rows(), n = A.num_cols(), k = s.size();
    double rec = 0.0, orth = 0.0;
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < k; ++j) {
            double av = 0.0;
            for (std::size_t l = 0; l < n; ++l) av += double(A(i, l)) * double(V(l, j));
            rec = std::max(rec, std::fabs(av - double(U(i, j)) * double(s(j))));
        }
    for (std::size_t a = 0; a < k; ++a)
        for (std::size_t b = 0; b < k; ++b) {
            double d = 0.0;
            for (std::size_t i = 0; i < m; ++i) d += double(U(i, a)) * double(U(i, b));
            orth = std::max(orth, std::fabs(d - (a == b ? 1.0 : 0.0)));
        }
    return {rec, orth};
}

} // namespace

TEST_CASE("svd: the accumulator reaches every overload (#585)", "[operation][svd][accumulator]") {
    const auto A = fixture<double>(9, 5);    // tall: the QR preconditioner runs
    const auto W = fixture<double>(4, 7);    // wide: Jacobi on A itself

    SECTION("full svd(A, U, S, V)") {
        mat::dense2D<double> U, S, V;
        counting::products = 0;
        svd<counting::count_acc>(A, U, S, V);
        CHECK(counting::products > 0);
    }
    SECTION("V-only svd(A, s, V), tall and wide") {
        vec::dense_vector<double> s;
        mat::dense2D<double> V;
        counting::products = 0;
        svd<counting::count_acc>(A, s, V);
        CHECK(counting::products > 0);
        counting::products = 0;
        svd<counting::count_acc>(W, s, V);
        CHECK(counting::products > 0);
    }
    SECTION("thin svd(A, U, s, V)") {
        mat::dense2D<double> U, V;
        vec::dense_vector<double> s;
        counting::products = 0;
        svd<counting::count_acc>(A, U, s, V);
        CHECK(counting::products > 0);
    }
    SECTION("tuple svd(A) and null_vector") {
        counting::products = 0;
        (void)svd<counting::count_acc>(A);
        CHECK(counting::products > 0);
        counting::products = 0;
        (void)null_vector<counting::count_acc>(W);
        CHECK(counting::products > 0);
    }
    SECTION("the void default sees none") {
        mat::dense2D<double> U, S, V;
        vec::dense_vector<double> s;
        counting::products = 0;
        svd(A, U, S, V);
        svd(A, s, V);
        svd(A, U, s, V);
        (void)null_vector(W);
        CHECK(counting::products == 0);
    }
}

TEST_CASE("svd: a plain element-type accumulator is the void path bit for bit (#585)",
          "[operation][svd][accumulator]") {
    // Not under LAPACK, where the void full and thin svd go to gesdd and the
    // accumulator path, by design, does not.
#ifndef MTL5_HAS_LAPACK
    for (auto [m, n] : {std::pair<std::size_t, std::size_t>{8, 5}, {5, 8}, {6, 6}}) {
        const auto A = fixture<float>(m, n);
        INFO(m << " x " << n);
        mat::dense2D<float> Uv, Sv, Vv, Up, Sp, Vp;
        svd(A, Uv, Sv, Vv);
        svd<float>(A, Up, Sp, Vp);
        CHECK(same_bits(Uv, Up));
        CHECK(same_bits(Sv, Sp));
        CHECK(same_bits(Vv, Vp));

        mat::dense2D<float> tUv, tVv, tUp, tVp;
        vec::dense_vector<float> tsv, tsp;
        svd(A, tUv, tsv, tVv);
        svd<float>(A, tUp, tsp, tVp);
        CHECK(same_bits(tUv, tUp));
        CHECK(same_bits(tsv, tsp));
        CHECK(same_bits(tVv, tVp));
    }
#endif
    // The V-only overload never calls gesdd, but a TALL A goes through the
    // Householder QR preconditioner, and the void qr_factor dispatches to
    // LAPACK's geqrf where the accumulator path, by design, does not. So a
    // wide A (no QR) is checked in every build, a tall one only without LAPACK.
    std::vector<std::pair<std::size_t, std::size_t>> shapes = {{6, 9}};
#ifndef MTL5_HAS_LAPACK
    shapes.push_back({9, 6});
#endif
    for (auto [m, n] : shapes) {
        const auto A = fixture<float>(m, n);
        INFO("V-only, " << m << " x " << n);
        vec::dense_vector<float> sv, sp;
        mat::dense2D<float> Vv, Vp;
        svd(A, sv, Vv);
        svd<float>(A, sp, Vp);
        CHECK(same_bits(sv, sp));
        CHECK(same_bits(Vv, Vp));
    }
}

TEST_CASE("[586:svd] svd: every configuration gives a valid SVD; no ordering (see the header)",
          "[586:svd][operation][svd][accumulator]") {
    using exact = exact_test::exact_acc<float>;
    using fma   = math::fma_accumulator<float>;
    const auto A = fixture<float>(10, 6);
    auto check = [&](auto tag, const char* name) {
        using Acc = typename decltype(tag)::type;
        mat::dense2D<float> U, V;
        vec::dense_vector<float> s;
        svd<Acc>(A, U, s, V);
        const auto [rec, orth] = thin_errors(A, U, s, V);
        INFO(name << ": |AV - US| = " << rec << ", |U^T U - I| = " << orth);
        CHECK(rec < 1e-5);
        CHECK(orth < 1e-5);
    };
    check(std::type_identity<float>{}, "plain");
    check(std::type_identity<fma>{}, "fma");
    check(std::type_identity<double>{}, "double");
    check(std::type_identity<exact>{}, "exact");
}
