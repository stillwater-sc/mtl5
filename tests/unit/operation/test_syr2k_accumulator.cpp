// MTL5 -- accumulator policy for syr2k (#261, Part C, BLAS L3).
// syr2k sums two product terms (A(i,l)*B(j,l) and B(i,l)*A(j,l)) per l into
// the same accumulator, seeded with clear, then combines with alpha/beta
// once, outside the reduction -- same lower-triangle/mirror shape as syrk.
// The mirror to the upper triangle afterward is a plain unrounded copy, not
// a re-sum.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>

#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/syr2k.hpp>
#include <mtl/math/accumulator_traits.hpp>

using namespace mtl;
using Catch::Matchers::WithinRel;

namespace {

struct counting_acc {
    double v = 0.0;
    static inline int clears = 0, assigns = 0, products = 0, values = 0;
    static void reset() { clears = assigns = products = values = 0; }
};

} // namespace

namespace mtl::math {
template <typename Value>
struct accumulator_traits<::counting_acc, Value> {
    static void clear(::counting_acc& a) { a.v = 0.0; ++::counting_acc::clears; }
    static void assign(::counting_acc& a, const Value& x) {
        a.v = static_cast<double>(x); ++::counting_acc::assigns;
    }
    template <typename Result = Value>
    static Result value(const ::counting_acc& a) {
        ++::counting_acc::values; return static_cast<Result>(a.v);
    }
    static void add_product(::counting_acc& a, const Value& m, const Value& x) {
        a.v += static_cast<double>(m) * static_cast<double>(x);
        ++::counting_acc::products;
    }
};
} // namespace mtl::math

TEST_CASE("syr2k default behavior is unchanged -- produces a full symmetric result",
          "[operation][syr2k][accumulator]") {
    mat::dense2D<double> A(2, 2);
    A(0,0)=1; A(0,1)=2;
    A(1,0)=3; A(1,1)=4;
    mat::dense2D<double> B(2, 2);
    B(0,0)=5; B(0,1)=6;
    B(1,0)=7; B(1,1)=8;
    mat::dense2D<double> C(2, 2);
    C(0,0)=0; C(0,1)=0;
    C(1,0)=0; C(1,1)=0;
    syr2k(1.0, A, B, 0.0, C);
    // C = A*B^T + B*A^T
    // C(0,0) = 2*(1*5+2*6) = 34
    // C(1,1) = 2*(3*7+4*8) = 106
    // C(0,1) = A(0,.)*B(1,.) + B(0,.)*A(1,.) = (1*7+2*8) + (5*3+6*4) = 23+39=62
    REQUIRE_THAT(C(0,0), WithinRel(34.0, 1e-12));
    REQUIRE_THAT(C(1,1), WithinRel(106.0, 1e-12));
    REQUIRE_THAT(C(0,1), WithinRel(62.0, 1e-12));
    REQUIRE_THAT(C(1,0), WithinRel(62.0, 1e-12));
}

TEST_CASE("syr2k mirrors the lower triangle exactly, not by re-summing",
          "[operation][syr2k][accumulator]") {
    mat::dense2D<double> A(3, 2);
    A(0,0)=1; A(0,1)=1;
    A(1,0)=2; A(1,1)=1;
    A(2,0)=3; A(2,1)=1;
    mat::dense2D<double> B(3, 2);
    B(0,0)=1; B(0,1)=0;
    B(1,0)=1; B(1,1)=1;
    B(2,0)=1; B(2,1)=2;
    mat::dense2D<double> C(3, 3);
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) C(i,j) = 0;

    counting_acc::reset();
    syr2k<counting_acc>(1.0, A, B, 0.0, C);

    const int m = 3, k = 2;
    const int lower_count = m * (m + 1) / 2;
    REQUIRE(counting_acc::clears   == lower_count);
    REQUIRE(counting_acc::products == lower_count * k * 2);  // two terms per l
    REQUIRE(counting_acc::values   == lower_count);
    REQUIRE(counting_acc::assigns  == 0);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            REQUIRE(C(i,j) == C(j,i));
}

TEST_CASE("syr2k fp64 accumulator beats fp32 on a near-cancelling row",
          "[operation][syr2k][accumulator]") {
    const std::size_t n = 2000;
    mat::dense2D<float> A(n, n), B(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            float v = (j % 2 == 0) ? 1.0f : -1.0f + 1.0e-6f;
            A(i, j) = v;
            B(i, j) = v;
        }

    double ref = 0.0;
    for (std::size_t l = 0; l < n; ++l)
        ref += static_cast<double>(A(0, l)) * static_cast<double>(B(0, l)) * 2.0;

    mat::dense2D<float> C_naive(n, n), C_wide(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) { C_naive(i,j) = 0.0f; C_wide(i,j) = 0.0f; }

    syr2k(1.0f, A, B, 0.0f, C_naive);
    syr2k<double>(1.0, A, B, 0.0, C_wide);

    double e_naive = std::abs(static_cast<double>(C_naive(0,0)) - ref);
    double e_wide  = std::abs(static_cast<double>(C_wide(0,0))  - ref);
    INFO("ref=" << ref << " naive=" << e_naive << " wide=" << e_wide);
    REQUIRE(e_wide <= e_naive);
}

TEST_CASE("syr2k accumulator/result types are honored", "[operation][syr2k][accumulator]") {
    mat::dense2D<float> A(2, 2);
    A(0,0)=1; A(0,1)=2;
    A(1,0)=3; A(1,1)=4;
    mat::dense2D<float> B(2, 2);
    B(0,0)=1; B(0,1)=0;
    B(1,0)=0; B(1,1)=1;
    mat::dense2D<float> C_wide(2, 2);
    C_wide(0,0)=0.0f; C_wide(0,1)=0.0f; C_wide(1,0)=0.0f; C_wide(1,1)=0.0f;
    syr2k<double>(1.0f, A, B, 0.0f, C_wide);
    // C(0,0) = A(0,.)*B(0,.)*2 = (1*1+2*0)*2 = 2
    // C(1,1) = A(1,.)*B(1,.)*2 = (3*0+4*1)*2 = 8
    // C(0,1) = A(0,.)*B(1,.) + B(0,.)*A(1,.) = (1*0+2*1) + (1*3+0*4) = 2+3=5
    REQUIRE_THAT(C_wide(0,0), WithinRel(2.0f, 1e-6f));
    REQUIRE_THAT(C_wide(1,1), WithinRel(8.0f, 1e-6f));
    REQUIRE_THAT(C_wide(0,1), WithinRel(5.0f, 1e-6f));
}
