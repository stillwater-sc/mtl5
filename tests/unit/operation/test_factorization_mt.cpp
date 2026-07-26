// Multithreaded dense factorizations (#297, batch 1): the LU trailing-submatrix
// update and the Cholesky column update are partitioned across the persistent
// thread pool over independent rows. Each output row is written by exactly one
// chunk and reads only shared (read-only) pivot/column data, so the threaded
// result is BIT-IDENTICAL to the serial path -- we assert exact equality (==)
// against an in-test serial reference, plus a residual check for correctness.
//
// The library kernels use the env-sized singleton pool, so this file sets
// MTL5_NUM_THREADS before the pool's first use (the only in-process way to
// exercise the factorization threading -- CI otherwise runs serial). Run under
// TSan (-DMTL5_SANITIZE=thread) to confirm race-freedom.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <random>
#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/lu.hpp>
#include <mtl/operation/cholesky.hpp>
#include <mtl/detail/thread_pool.hpp>

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

// Faithful serial copy of mtl::lu_factor's generic path -- the golden reference
// the threaded factorization must match bit-for-bit.
template <typename T>
void ref_lu(mat::dense2D<T>& A, std::vector<std::size_t>& piv) {
    const std::size_t n = A.num_rows();
    piv.resize(n);
    for (std::size_t k = 0; k < n; ++k) {
        std::size_t max_row = k;
        T max_val = std::abs(A(k, k));
        for (std::size_t i = k + 1; i < n; ++i) {
            T v = std::abs(A(i, k));
            if (v > max_val) { max_val = v; max_row = i; }
        }
        piv[k] = max_row;
        if (max_row != k)
            for (std::size_t j = 0; j < n; ++j) { T t = A(k, j); A(k, j) = A(max_row, j); A(max_row, j) = t; }
        for (std::size_t i = k + 1; i < n; ++i) {
            A(i, k) /= A(k, k);
            for (std::size_t j = k + 1; j < n; ++j)
                A(i, j) -= A(i, k) * A(k, j);
        }
    }
}

// Faithful serial copy of mtl::cholesky_factor's generic path.
template <typename T>
void ref_chol(mat::dense2D<T>& A) {
    const std::size_t n = A.num_rows();
    for (std::size_t j = 0; j < n; ++j) {
        T sum = T(0);
        for (std::size_t k = 0; k < j; ++k) sum += A(j, k) * A(j, k);
        A(j, j) = std::sqrt(A(j, j) - sum);
        for (std::size_t i = j + 1; i < n; ++i) {
            T s = T(0);
            for (std::size_t k = 0; k < j; ++k) s += A(i, k) * A(j, k);
            A(i, j) = (A(i, j) - s) / A(j, j);
        }
    }
}

// Row-major dense2D exercises the generic (parallelized) path even when LAPACK
// is linked (the LAPACK LU/Cholesky dispatch fires only for column-major).
template <typename T>
mat::dense2D<T> random_diagdom(std::size_t n, std::uint64_t seed) {
    mat::dense2D<T> A(n, n);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    for (std::size_t i = 0; i < n; ++i) {
        double rowsum = 0.0;
        for (std::size_t j = 0; j < n; ++j) {
            if (i != j) { T v = static_cast<T>(d(rng)); A(i, j) = v; rowsum += std::abs(double(v)); }
        }
        A(i, i) = static_cast<T>(rowsum + 1.0);   // strictly diagonally dominant
    }
    return A;
}

// Symmetric positive definite: B = A + A^T + n*I on a random A.
template <typename T>
mat::dense2D<T> random_spd(std::size_t n, std::uint64_t seed) {
    mat::dense2D<T> R(n, n);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    mat::dense2D<T> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) A(i, j) = static_cast<T>(d(rng));
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            R(i, j) = A(i, j) + A(j, i) + (i == j ? static_cast<T>(2 * n) : T(0));
    return R;
}

} // namespace

TEST_CASE("Threaded LU factorization is bit-identical to serial (#297)",
          "[operation][lu][threading][mt]") {
    // The pool must actually be multithreaded for this to mean anything.
    if (detail::thread_pool::instance().size() < 2) {
        WARN("single-core runner: threading not exercised");
    }
    const std::size_t n = 512;   // large enough that the trailing update splits
    auto A = random_diagdom<double>(n, 12345);
    auto Aref = A;               // copy for the serial reference

    std::vector<mat::dense2D<double>::size_type> piv;
    int info = lu_factor(A, piv);              // library (threaded) path
    REQUIRE(info == 0);

    std::vector<std::size_t> rpiv;
    ref_lu(Aref, rpiv);                        // serial golden reference

    // Bit-for-bit equality: exposes any race or ordering divergence.
    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE(static_cast<std::size_t>(piv[i]) == rpiv[i]);
        for (std::size_t j = 0; j < n; ++j)
            REQUIRE(A(i, j) == Aref(i, j));
    }

    // Correctness: the factorization solves a system to a tiny residual.
    vec::dense_vector<double> xexact(n), b(n, 0.0);
    for (std::size_t i = 0; i < n; ++i) xexact[i] = 1.0 + 0.001 * double(i % 13);
    auto A0 = random_diagdom<double>(n, 12345);
    for (std::size_t i = 0; i < n; ++i) { double s = 0.0; for (std::size_t j = 0; j < n; ++j) s += A0(i, j) * xexact[j]; b[i] = s; }
    vec::dense_vector<double> x(n, 0.0);
    lu_solve(A, piv, x, b);
    double err = 0.0;
    for (std::size_t i = 0; i < n; ++i) err = std::max(err, std::abs(x[i] - xexact[i]));
    REQUIRE(err < 1e-9);
}

TEST_CASE("Threaded Cholesky factorization is bit-identical to serial (#297)",
          "[operation][cholesky][threading][mt]") {
    if (detail::thread_pool::instance().size() < 2) {
        WARN("single-core runner: threading not exercised");
    }
    const std::size_t n = 768;   // large enough that the column update splits
    auto A = random_spd<double>(n, 6789);
    auto Aref = A;

    int info = cholesky_factor(A);             // library (threaded) path
    REQUIRE(info == 0);

    ref_chol(Aref);                            // serial golden reference

    // Bit-for-bit equality over the lower triangle (where L is stored).
    for (std::size_t j = 0; j < n; ++j)
        for (std::size_t i = j; i < n; ++i)
            REQUIRE(A(i, j) == Aref(i, j));

    // Correctness: L*L^T reconstructs the original SPD matrix.
    auto A0 = random_spd<double>(n, 6789);
    double resid = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j <= i; ++j) {
            double llt = 0.0;
            for (std::size_t k = 0; k <= j; ++k) llt += A(i, k) * A(j, k);
            resid = std::max(resid, std::abs(llt - double(A0(i, j))));
        }
    REQUIRE(resid < 1e-8);
}

TEST_CASE("Threaded factorizations degrade correctly at the single-chunk boundary (#297)",
          "[operation][lu][cholesky][threading][mt][edge]") {
    // n = 1 and n = 2: trailing/rows collapse to 0 or 1, so parallel_for takes
    // its serial-fallback branch. The result must still match the serial
    // reference bit-for-bit. Explicit (deterministic) tiny matrices so the SPD /
    // diagonally-dominant property never depends on the RNG.

    auto check_lu = [](mat::dense2D<double> A) {
        auto Aref = A;
        std::vector<mat::dense2D<double>::size_type> piv;
        REQUIRE(lu_factor(A, piv) == 0);
        std::vector<std::size_t> rpiv;
        ref_lu(Aref, rpiv);
        const std::size_t n = A.num_rows();
        for (std::size_t i = 0; i < n; ++i) {
            REQUIRE(static_cast<std::size_t>(piv[i]) == rpiv[i]);
            for (std::size_t j = 0; j < n; ++j) REQUIRE(A(i, j) == Aref(i, j));
        }
    };
    auto check_chol = [](mat::dense2D<double> A) {
        auto Aref = A;
        REQUIRE(cholesky_factor(A) == 0);
        ref_chol(Aref);
        const std::size_t n = A.num_rows();
        for (std::size_t j = 0; j < n; ++j)
            for (std::size_t i = j; i < n; ++i) REQUIRE(A(i, j) == Aref(i, j));
    };

    SECTION("1x1") {
        mat::dense2D<double> A(1, 1); A(0, 0) = 4.0;
        check_lu(A);
        check_chol(A);
    }
    SECTION("2x2") {
        mat::dense2D<double> A(2, 2);
        A(0, 0) = 4.0; A(0, 1) = 1.0;
        A(1, 0) = 1.0; A(1, 1) = 3.0;   // symmetric, SPD, diagonally dominant
        check_lu(A);
        check_chol(A);
    }
}
