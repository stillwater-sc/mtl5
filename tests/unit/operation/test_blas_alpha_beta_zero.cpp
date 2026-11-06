// The BLAS alpha/beta convention for syrk, syr2k, symv, symm and trmm (#692).
//
// BLAS defines beta == 0 to mean C (or y) is not read, so it may hold NaN or
// garbage, and alpha == 0 to mean A (and B, x) are not read. The generic paths
// used to compute alpha*sum + beta*C regardless, so a NaN in C survived
// beta == 0 (0 * NaN = NaN), and a NaN in A survived alpha == 0. symv and symm
// also read A's full matrix on the generic paths, while BLAS reads only the
// lower triangle; they now read the lower triangle everywhere.
//
// Each routine runs three ways: row-major (the generic path), column-major (the
// BLAS path in a BLAS build, the generic path otherwise), and with an explicit
// Accumulator (the accumulator path). The values are small integers, so every
// path is exact and the results compare with ==.
#include <catch2/catch_test_macros.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/tag/orientation.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/symm.hpp>
#include <mtl/operation/symv.hpp>
#include <mtl/operation/syr2k.hpp>
#include <mtl/operation/syrk.hpp>
#include <mtl/operation/trmm.hpp>

#include <cmath>
#include <cstddef>
#include <limits>

using namespace mtl;

namespace {

constexpr double nan_v = std::numeric_limits<double>::quiet_NaN();

using rowmat = mat::dense2D<double>;
using colmat = mat::dense2D<double, mat::parameters<tag::col_major>>;
using vec_t  = vec::dense_vector<double>;

// Which code path a check runs: Accumulator = void takes the generic or BLAS
// path, depending on the matrix orientation; double takes the accumulator path.
struct via_default { using acc = void; };
struct via_accumulator { using acc = double; };

template <typename M>
M filled(std::size_t r, std::size_t c, double v) {
    M m(r, c);
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j) m(i, j) = v;
    return m;
}

// A 3x2 and B 3x2 with small integer entries.
template <typename M>
M make_a() {
    M a(3, 2);
    a(0, 0) = 1; a(0, 1) = 2;
    a(1, 0) = -1; a(1, 1) = 3;
    a(2, 0) = 2; a(2, 1) = 0;
    return a;
}
template <typename M>
M make_b() {
    M b(3, 2);
    b(0, 0) = 0; b(0, 1) = 1;
    b(1, 0) = 2; b(1, 1) = -2;
    b(2, 0) = 1; b(2, 1) = 1;
    return b;
}

// A symmetric matrix's lower triangle, with NaN in the strict upper triangle.
template <typename M>
M lower_only() {
    M a = filled<M>(3, 3, nan_v);
    a(0, 0) = 2;
    a(1, 0) = 1; a(1, 1) = 3;
    a(2, 0) = -1; a(2, 1) = 4; a(2, 2) = 1;
    return a;
}
double sym_lower(std::size_t i, std::size_t j) {
    static const double s[3][3] = {{2, 1, -1}, {1, 3, 4}, {-1, 4, 1}};
    return s[i][j];
}

template <typename M>
bool all_equal(const M& c, std::size_t r, std::size_t n, double v) {
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < n; ++j)
            if (!(c(i, j) == v)) return false;
    return true;
}

template <typename M, typename Via>
void check_syrk() {
    using Acc = typename Via::acc;
    const M A = make_a<M>();
    // beta == 0: C is not read.
    M C = filled<M>(3, 3, nan_v);
    syrk<Acc>(2.0, A, 0.0, C);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) {
            const double ref = 2.0 * (A(i, 0) * A(j, 0) + A(i, 1) * A(j, 1));
            CAPTURE(i, j);
            REQUIRE(C(i, j) == ref);
        }
    // alpha == 0: A is not read, C = beta*C.
    const M Anan = filled<M>(3, 2, nan_v);
    M C2 = filled<M>(3, 3, 5.0);
    syrk<Acc>(0.0, Anan, 3.0, C2);
    REQUIRE(all_equal(C2, 3, 3, 15.0));
    // alpha == beta == 0: neither is read, C = 0.
    M C3 = filled<M>(3, 3, nan_v);
    syrk<Acc>(0.0, Anan, 0.0, C3);
    REQUIRE(all_equal(C3, 3, 3, 0.0));
}

template <typename M, typename Via>
void check_syr2k() {
    using Acc = typename Via::acc;
    const M A = make_a<M>(), B = make_b<M>();
    M C = filled<M>(3, 3, nan_v);
    syr2k<Acc>(1.0, A, B, 0.0, C);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j) {
            double ref = 0;
            for (std::size_t l = 0; l < 2; ++l) ref += A(i, l) * B(j, l) + B(i, l) * A(j, l);
            CAPTURE(i, j);
            REQUIRE(C(i, j) == ref);
        }
    const M Anan = filled<M>(3, 2, nan_v);
    M C2 = filled<M>(3, 3, 5.0);
    syr2k<Acc>(0.0, Anan, Anan, -1.0, C2);
    REQUIRE(all_equal(C2, 3, 3, -5.0));
    M C3 = filled<M>(3, 3, nan_v);
    syr2k<Acc>(0.0, Anan, Anan, 0.0, C3);
    REQUIRE(all_equal(C3, 3, 3, 0.0));
}

template <typename M, typename Via>
void check_symv() {
    using Acc = typename Via::acc;
    const M A = lower_only<M>();          // NaN in the strict upper triangle
    vec_t x = {1.0, -2.0, 3.0};
    vec_t y = {nan_v, nan_v, nan_v};
    symv<Acc>(2.0, A, x, 0.0, y);
    for (std::size_t i = 0; i < 3; ++i) {
        double ref = 0;
        for (std::size_t j = 0; j < 3; ++j) ref += sym_lower(i, j) * x(j);
        CAPTURE(i);
        REQUIRE(y(i) == 2.0 * ref);
    }
    const M Anan = filled<M>(3, 3, nan_v);
    vec_t xnan = {nan_v, nan_v, nan_v};
    vec_t y2 = {1.0, 2.0, 3.0};
    symv<Acc>(0.0, Anan, xnan, 2.0, y2);
    REQUIRE((y2(0) == 2.0 && y2(1) == 4.0 && y2(2) == 6.0));
    vec_t y3 = {nan_v, nan_v, nan_v};
    symv<Acc>(0.0, Anan, xnan, 0.0, y3);
    REQUIRE((y3(0) == 0.0 && y3(1) == 0.0 && y3(2) == 0.0));
}

template <typename M, typename Via>
void check_symm() {
    using Acc = typename Via::acc;
    const M A = lower_only<M>();
    const M B = make_a<M>();              // 3x2
    M C = filled<M>(3, 2, nan_v);
    symm<Acc>(1.0, A, B, 0.0, C);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 2; ++j) {
            double ref = 0;
            for (std::size_t k = 0; k < 3; ++k) ref += sym_lower(i, k) * B(k, j);
            CAPTURE(i, j);
            REQUIRE(C(i, j) == ref);
        }
    const M Anan = filled<M>(3, 3, nan_v);
    const M Bnan = filled<M>(3, 2, nan_v);
    M C2 = filled<M>(3, 2, 4.0);
    symm<Acc>(0.0, Anan, Bnan, 0.5, C2);
    REQUIRE(all_equal(C2, 3, 2, 2.0));
    M C3 = filled<M>(3, 2, nan_v);
    symm<Acc>(0.0, Anan, Bnan, 0.0, C3);
    REQUIRE(all_equal(C3, 3, 2, 0.0));
}

template <typename M, typename Via>
void check_trmm() {
    using Acc = typename Via::acc;
    // alpha == 0: A and B are not read, B = 0.
    const M Anan = filled<M>(3, 3, nan_v);
    for (bool upper : {true, false})
        for (bool unit : {true, false}) {
            M B = filled<M>(3, 2, nan_v);
            trmm<Acc>(0.0, Anan, B, upper, unit);
            CAPTURE(upper, unit);
            REQUIRE(all_equal(B, 3, 2, 0.0));
        }
    // alpha != 0 is unchanged: upper triangular A times B.
    M A = filled<M>(3, 3, nan_v);          // the other triangle is not read
    A(0, 0) = 1; A(0, 1) = 2; A(0, 2) = 3;
    A(1, 1) = 4; A(1, 2) = 5;
    A(2, 2) = 6;
    M B = make_a<M>();
    const M B0 = B;
    trmm<Acc>(1.0, A, B, true, false);
    for (std::size_t c = 0; c < 2; ++c)
        for (std::size_t i = 0; i < 3; ++i) {
            double ref = 0;
            for (std::size_t j = i; j < 3; ++j) ref += A(i, j) * B0(j, c);
            CAPTURE(i, c);
            REQUIRE(B(i, c) == ref);
        }
}

} // namespace

#define MTL5_ALL_PATHS(fn)                                         \
    SECTION("row-major, generic") { fn<rowmat, via_default>(); }   \
    SECTION("column-major, BLAS when available") { fn<colmat, via_default>(); } \
    SECTION("explicit Accumulator") { fn<rowmat, via_accumulator>(); }

TEST_CASE("syrk: beta == 0 does not read C, alpha == 0 does not read A", "[operation][blas][692]") {
    MTL5_ALL_PATHS(check_syrk)
}

TEST_CASE("syr2k: beta == 0 does not read C, alpha == 0 does not read A or B", "[operation][blas][692]") {
    MTL5_ALL_PATHS(check_syr2k)
}

TEST_CASE("symv: lower triangle only; beta == 0 does not read y, alpha == 0 does not read A or x",
          "[operation][blas][692]") {
    MTL5_ALL_PATHS(check_symv)
}

TEST_CASE("symm: lower triangle only; beta == 0 does not read C, alpha == 0 does not read A or B",
          "[operation][blas][692]") {
    MTL5_ALL_PATHS(check_symm)
}

TEST_CASE("trmm: alpha == 0 sets B to zero without reading A or B", "[operation][blas][692]") {
    MTL5_ALL_PATHS(check_trmm)
}
