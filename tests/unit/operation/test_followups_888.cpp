// Follow-ups of issue 888 (QA sweep #872):
//
//   - range::extent() and slices with a huge step (it wrapped to 0);
//   - ndarray::shares_memory_with needs signed coefficients: two views of one
//     buffer whose overlap is 5 - 1 = 4 elements apart read as disjoint, and
//     assigning one to the other skipped its temporary;
//   - mtl::min/max over an ndarray with NaN;
//   - randsvd(n, kappa, mode, seed), the square seeded form (it bound the
//     rectangular randsvd(m, n, kappa, mode) and threw on mode = seed);
//   - eigen() with clustered eigenvalues: each vector pairs with its value
//     (the pairing of eigenvalue()'s values with the Schur run's);
//   - ldlt_bk_factor with a 2x2 pivot whose determinant is NaN, and the
//     Accumulator path;
//   - lu_iterative_refine with a Working-precision copy that is singular
//     while A is not;
//   - dense assignment from expressions over coordinate2D, ell_matrix and
//     block_diagonal2D operands (expr_alias's known-storage branches).
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

#include <mtl/array/ndarray.hpp>
#include <mtl/array/slice.hpp>
#include <mtl/generators/randsvd.hpp>
#include <mtl/mat/block_diagonal2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/coordinate2D.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/ell_matrix.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/operation/eigenvalue.hpp>
#include <mtl/operation/ldlt_bk.hpp>
#include <mtl/operation/lu_iterative_refine.hpp>
#include <mtl/operation/max.hpp>
#include <mtl/operation/min.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/spectral_properties.hpp>
#include <mtl/vec/dense_vector.hpp>

using namespace mtl;

TEST_CASE("range::extent and a slice with a huge step (issue 888)", "[array][slice][888]") {
    // (stop - start + step - 1) / step wrapped: range(0, 3, SIZE_MAX) was 0.
    constexpr std::size_t huge = std::numeric_limits<std::size_t>::max();
    CHECK(array::range(0, 3, huge).extent() == 1);
    CHECK(array::range(0, 3, huge - 1).extent() == 1);
    CHECK(array::range(2, 9, 3).extent() == 3);   // 2, 5, 8
    CHECK(array::range(2, 8, 3).extent() == 2);   // 2, 5
    CHECK(array::range(5, 5, 1).extent() == 0);
    array::ndarray<double, 1> a(array::shape<1>{6});
    for (std::size_t i = 0; i < 6; ++i) a(i) = double(i) + 10.0;
    auto s = array::slice(a, array::range(1, 6, huge));
    REQUIRE(s.extent(0) == 1);
    CHECK(s(0) == 11.0);
}

TEST_CASE("shares_memory_with sees an overlap that needs signed coefficients (issue 888)", "[array][ndarray][888]") {
    // A: offsets {0, 1, 5, 6}; B = A displaced by 4: {4, 5, 9, 10}. They
    // share offset 5 (A(1,0), B(0,1)): 4 = 1*5 - 1*1. The greedy test took
    // only nonnegative coefficients and said no overlap, so B = A wrote
    // B(1,0) (offset 9) from A(1,0) (offset 5) after B(0,1) had overwritten it.
    std::vector<double> buf(11);
    for (std::size_t i = 0; i < buf.size(); ++i) buf[i] = double(i);
    array::ndarray<double, 2> A(buf.data(), array::shape<2>{2, 2}, {5, 1});
    array::ndarray<double, 2> B(buf.data() + 4, array::shape<2>{2, 2}, {5, 1});
    B = A;   // through a temporary: B gets A's values as they were
    CHECK(buf[4] == 0.0);
    CHECK(buf[5] == 1.0);
    CHECK(buf[9] == 5.0);    // A(1,0) before B(0,1) overwrote offset 5
    CHECK(buf[10] == 6.0);
}

TEST_CASE("mtl::min and max over an ndarray with NaN (issue 888)", "[array][ndarray][888]") {
    // The result depended on the NaN's position; it is NaN wherever it is.
    for (std::size_t at = 0; at < 4; ++at) {
        array::ndarray<double, 1> a(array::shape<1>{4});
        for (std::size_t i = 0; i < 4; ++i) a(i) = double(i);
        a(at) = std::numeric_limits<double>::quiet_NaN();
        CHECK(std::isnan(mtl::min(a)));
        CHECK(std::isnan(mtl::max(a)));
    }
}

TEST_CASE("randsvd(n, kappa, mode, seed): the square seeded form (issue 888)", "[generators][randsvd][888]") {
    // It bound randsvd(m = 5, n = 1000, kappa = 3, mode = 42) and threw.
    const auto A = generators::randsvd<double>(5, 1e3, 3, std::uint64_t(42));
    REQUIRE(A.num_rows() == 5);
    REQUIRE(A.num_cols() == 5);
    CHECK(std::abs(condition_number(A) / 1e3 - 1.0) < 1e-8);
    // Reproducible, and the same matrix as randsvd(n, n, kappa, mode, seed).
    const auto B = generators::randsvd<double>(5, 5, 1e3, 3, std::uint64_t(42));
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j) CHECK(A(i, j) == B(i, j));
    // The deduced form takes it too, and is the same matrix.
    const auto C = generators::randsvd(5, 1e3, 3, 42);
    REQUIRE(C.num_rows() == 5);
    REQUIRE(C.num_cols() == 5);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 5; ++j) CHECK(C(i, j) == A(i, j));
    // A floating kappa in third place is the rectangular form, explicit T or
    // not. (An integral kappa doesn't compile: compile_fail/randsvd_integral_kappa.)
    const auto R = generators::randsvd(5, 4, 1e3, 3);
    CHECK(R.num_rows() == 5);
    CHECK(R.num_cols() == 4);
    const auto R2 = generators::randsvd<double>(5, 4, 1e3, 3);
    CHECK(R2.num_cols() == 4);
}

TEST_CASE("eigen() pairs clustered eigenvalues with their own vectors (issue 888)", "[operation][eigen][888]") {
    // A near-Jordan 3x3, eigenvalues 1 +- 1e-4 and 2. eigen() pairs
    // eigenvalue()'s values with the Schur run's by nearest distance; a swap
    // in the cluster would give each of the pair the other's vector. (Closer, 1 +- 1e-8 from a
    // 1e-16 subdiagonal, deflates at tol = 1e-10 to a double eigenvalue.)
    mat::dense2D<double> A = {{1.0, 1.0, 0.5}, {1e-8, 1.0, 0.25}, {0.0, 0.0, 2.0}};
    const auto [eigs, V] = eigen(A);
    REQUIRE(eigs.size() == 3);
    std::size_t near_one = 0;
    for (std::size_t k = 0; k < 3; ++k)
        if (std::abs(eigs(k) - 1.0) < 2e-4) ++near_one;
    CHECK(near_one == 2);
    for (std::size_t k = 0; k < 3; ++k) {
        INFO("k = " << k << ", lambda = " << eigs(k));
        double r = 0, nv = 0;
        for (std::size_t i = 0; i < 3; ++i) {
            std::complex<double> av = 0.0;
            for (std::size_t j = 0; j < 3; ++j) av += A(i, j) * V(j, k);
            r += std::norm(av - eigs(k) * V(i, k));
            nv += std::norm(V(i, k));
        }
        CHECK(std::sqrt(r / nv) <= 1e-12);
    }
}

TEST_CASE("ldlt_bk_factor: a 2x2 pivot with a NaN determinant, and the Accumulator path (issue 888)", "[operation][ldlt_bk][888]") {
    {
        // A 2x2 pivot block with a NaN d11: |a00| = 1 fails tests 1 and 2, the
        // NaN fails test 3, so the 2x2 pivot is taken and det is NaN. (Finite
        // entries can't make det = inf - inf: an a00 large enough for that
        // passes test 2, |a00| sigma >= alpha lambda^2, and takes a 1x1.)
        mat::dense2D<double> A = {{1.0, 10.0}, {10.0, std::numeric_limits<double>::quiet_NaN()}};
        bk_pivot_info piv;
        CHECK(ldlt_bk_factor(A, piv) != 0);
    }
    {
        // An indefinite 4x4 that needs a 2x2 pivot: the Accumulator path gives
        // the same factors as the void one.
        const mat::dense2D<double> S = {{0, 2, 1, 0}, {2, 0, 0, 1}, {1, 0, 3, 1}, {0, 1, 1, 4}};
        mat::dense2D<double> A = S, B = S;
        bk_pivot_info pa, pb;
        REQUIRE(ldlt_bk_factor(A, pa) == 0);
        REQUIRE(ldlt_bk_factor<long double>(B, pb) == 0);
        CHECK(pa.ipiv == pb.ipiv);
        for (std::size_t i = 0; i < 4; ++i)
            for (std::size_t j = 0; j <= i; ++j) CHECK(std::abs(A(i, j) - B(i, j)) <= 1e-13);
    }
}

TEST_CASE("lu_iterative_refine with a singular Working copy of a nonsingular A (issue 888)", "[operation][lu][888]") {
    // diag(1, 1e-50) is nonsingular in double, but 1e-50 is 0 in float: the
    // float factor is singular, which throws std::runtime_error (#861).
    const mat::dense2D<double> A = {{1.0, 0.0}, {0.0, 1e-50}};
    const vec::dense_vector<double> b = {1.0, 1.0};
    vec::dense_vector<double> x(2);
    CHECK_THROWS_AS(lu_iterative_refine<float>(A, b, x), std::runtime_error);
}

TEST_CASE("dense assignment from coordinate2D, ell_matrix and block_diagonal2D operands (issue 888)", "[mat][alias][888]") {
    // expr_alias's known-storage branches: the operands' storage does not
    // overlap the target's, so the assignment stays in place and is right.
    using csr = mat::compressed2D<double>;
    csr S(3, 3);
    {
        mat::inserter<csr> ins(S);
        ins[0][0] << 1.0; ins[1][2] << 2.0; ins[2][1] << 3.0;
    }
    mat::coordinate2D<double> Co(3, 3);
    Co.insert(0, 0, 1.0); Co.insert(1, 2, 2.0); Co.insert(2, 1, 3.0);
    const mat::ell_matrix<double> E(S);
    mat::block_diagonal2D<double> D;
    D.add_block(mat::dense2D<double>{{1.0, 2.0}, {3.0, 4.0}});
    D.add_block(mat::dense2D<double>{{5.0}});

    mat::dense2D<double> T = {{1, 1, 1}, {1, 1, 1}, {1, 1, 1}};
    T = T + Co;
    CHECK(T(0, 0) == 2.0); CHECK(T(1, 2) == 3.0); CHECK(T(2, 1) == 4.0); CHECK(T(0, 1) == 1.0);
    T = E + E;
    CHECK(T(1, 2) == 4.0); CHECK(T(0, 1) == 0.0);
    T = D + D;
    CHECK(T(0, 1) == 4.0); CHECK(T(2, 2) == 10.0); CHECK(T(0, 2) == 0.0);
}
