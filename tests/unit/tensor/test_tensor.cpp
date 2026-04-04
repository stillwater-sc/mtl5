#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <mtl/tensor/tensor.hpp>
#include <mtl/tensor/index.hpp>
#include <mtl/tensor/symmetric.hpp>
#include <mtl/tensor/metric.hpp>

using namespace mtl::tensor;

// ── Core tensor construction ───────────────────────────────────────

TEST_CASE("tensor default construction is zero", "[tensor][ctor]") {
    tensor<double, 2, 3> t;
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE(t(i, j) == 0.0);
}

TEST_CASE("tensor compile-time properties", "[tensor][static]") {
    using T = tensor<double, 2, 3>;
    STATIC_REQUIRE(T::rank == 2);
    STATIC_REQUIRE(T::dimension == 3);
    STATIC_REQUIRE(T::num_components == 9);

    using T4 = tensor<double, 4, 3>;
    STATIC_REQUIRE(T4::num_components == 81);
}

TEST_CASE("tensor fill construction", "[tensor][ctor]") {
    tensor<int, 2, 2> t(42);
    REQUIRE(t(0, 0) == 42);
    REQUIRE(t(1, 1) == 42);
}

TEST_CASE("tensor initializer list", "[tensor][ctor]") {
    tensor<int, 2, 2> t{1, 2, 3, 4};
    REQUIRE(t(0, 0) == 1);
    REQUIRE(t(0, 1) == 2);
    REQUIRE(t(1, 0) == 3);
    REQUIRE(t(1, 1) == 4);
}

// ── Element access ─────────────────────────────────────────────────

TEST_CASE("tensor element access rank-1", "[tensor][access]") {
    tensor<double, 1, 3> v;
    v(0) = 1.0; v(1) = 2.0; v(2) = 3.0;
    REQUIRE(v(0) == 1.0);
    REQUIRE(v(1) == 2.0);
    REQUIRE(v(2) == 3.0);
}

TEST_CASE("tensor element access rank-2", "[tensor][access]") {
    tensor<int, 2, 3> t;
    t(1, 2) = 42;
    REQUIRE(t(1, 2) == 42);
    REQUIRE(t[{1, 2}] == 42);
}

TEST_CASE("tensor element access rank-4", "[tensor][access]") {
    tensor<int, 4, 2> t;
    t(1, 0, 1, 0) = 7;
    REQUIRE(t(1, 0, 1, 0) == 7);
}

// ── Arithmetic ─────────────────────────────────────────────────────

TEST_CASE("tensor addition", "[tensor][arithmetic]") {
    tensor<int, 2, 2> a{1, 2, 3, 4};
    tensor<int, 2, 2> b{10, 20, 30, 40};
    auto c = a + b;
    REQUIRE(c(0, 0) == 11);
    REQUIRE(c(1, 1) == 44);
}

TEST_CASE("tensor scalar multiplication", "[tensor][arithmetic]") {
    tensor<int, 2, 2> t{1, 2, 3, 4};
    auto s = t * 3;
    REQUIRE(s(0, 0) == 3);
    REQUIRE(s(1, 1) == 12);

    auto s2 = 2 * t;
    REQUIRE(s2(0, 1) == 4);
}

TEST_CASE("tensor negation", "[tensor][arithmetic]") {
    tensor<int, 1, 3> v;
    v(0) = 1; v(1) = -2; v(2) = 3;
    auto neg = -v;
    REQUIRE(neg(0) == -1);
    REQUIRE(neg(1) == 2);
    REQUIRE(neg(2) == -3);
}

// ── Trace ──────────────────────────────────────────────────────────

TEST_CASE("trace of rank-2 tensor", "[tensor][trace]") {
    tensor<double, 2, 3> t;
    t(0, 0) = 1.0; t(1, 1) = 2.0; t(2, 2) = 3.0;
    REQUIRE(trace(t) == Catch::Approx(6.0));
}

TEST_CASE("trace of identity", "[tensor][trace]") {
    auto I = identity<double, 3>();
    REQUIRE(trace(I) == Catch::Approx(3.0));
}

// ── Determinant ────────────────────────────────────────────────────

TEST_CASE("determinant 2x2", "[tensor][det]") {
    tensor<double, 2, 2> t{1, 2, 3, 4};
    REQUIRE(determinant(t) == Catch::Approx(-2.0));
}

TEST_CASE("determinant 3x3", "[tensor][det]") {
    tensor<double, 2, 3> t{
        1, 2, 3,
        0, 1, 4,
        5, 6, 0
    };
    REQUIRE(determinant(t) == Catch::Approx(1.0));
}

TEST_CASE("determinant of identity", "[tensor][det]") {
    REQUIRE(determinant(identity<double, 2>()) == Catch::Approx(1.0));
    REQUIRE(determinant(identity<double, 3>()) == Catch::Approx(1.0));
}

// ── Frobenius norm ─────────────────────────────────────────────────

TEST_CASE("frobenius norm of identity", "[tensor][norm]") {
    auto I = identity<double, 3>();
    REQUIRE(frobenius_norm(I) == Catch::Approx(std::sqrt(3.0)));
}

// ── Transpose ──────────────────────────────────────────────────────

TEST_CASE("transpose of rank-2 tensor", "[tensor][transpose]") {
    tensor<int, 2, 3> t;
    t(0, 1) = 5; t(1, 0) = 7;
    auto tt = transpose(t);
    REQUIRE(tt(1, 0) == 5);
    REQUIRE(tt(0, 1) == 7);
}

// ── Einstein summation (contraction) ───────────────────────────────

TEST_CASE("contract rank-2 tensors (matrix multiply)", "[tensor][contract]") {
    Index<'i'> i; Index<'j'> j; Index<'k'> k;

    tensor<double, 2, 2> A{1, 2, 3, 4};
    tensor<double, 2, 2> B{5, 6, 7, 8};

    // C = A * B via Einstein: C^i_k = A^i_j * B^j_k
    auto C = contract(bind(A, i, j), bind(B, j, k));

    // Manual: C(0,0)=1*5+2*7=19, C(0,1)=1*6+2*8=22
    //         C(1,0)=3*5+4*7=43, C(1,1)=3*6+4*8=50
    REQUIRE(C(0, 0) == Catch::Approx(19.0));
    REQUIRE(C(0, 1) == Catch::Approx(22.0));
    REQUIRE(C(1, 0) == Catch::Approx(43.0));
    REQUIRE(C(1, 1) == Catch::Approx(50.0));
}

TEST_CASE("contract rank-2 with rank-1 (matrix-vector)", "[tensor][contract]") {
    Index<'i'> i; Index<'j'> j;

    tensor<double, 2, 3> A;
    A(0, 0) = 1; A(0, 1) = 0; A(0, 2) = 0;
    A(1, 0) = 0; A(1, 1) = 2; A(1, 2) = 0;
    A(2, 0) = 0; A(2, 1) = 0; A(2, 2) = 3;

    tensor<double, 1, 3> v;
    v(0) = 1; v(1) = 1; v(2) = 1;

    auto result = contract(bind(A, i, j), bind(v, j));
    REQUIRE(result(0) == Catch::Approx(1.0));
    REQUIRE(result(1) == Catch::Approx(2.0));
    REQUIRE(result(2) == Catch::Approx(3.0));
}

TEST_CASE("contraction identity * A = A", "[tensor][contract]") {
    Index<'i'> i; Index<'j'> j; Index<'k'> k;

    auto I = identity<double, 3>();
    tensor<double, 2, 3> A{1, 2, 3, 4, 5, 6, 7, 8, 9};

    auto result = contract(bind(I, i, j), bind(A, j, k));
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            REQUIRE(result(r, c) == Catch::Approx(A(r, c)));
}

// ── Outer product ──────────────────────────────────────────────────

TEST_CASE("outer product of rank-1 tensors", "[tensor][outer]") {
    tensor<double, 1, 3> a;
    a(0) = 1; a(1) = 2; a(2) = 3;
    tensor<double, 1, 3> b;
    b(0) = 4; b(1) = 5; b(2) = 6;

    auto C = outer(a, b);
    REQUIRE(C(0, 0) == Catch::Approx(4.0));
    REQUIRE(C(0, 2) == Catch::Approx(6.0));
    REQUIRE(C(2, 1) == Catch::Approx(15.0));
}

// ── Symmetric tensor ───────────────────────────────────────────────

TEST_CASE("symmetric tensor storage size", "[tensor][symmetric]") {
    STATIC_REQUIRE(symmetric_tensor<double, 3>::num_stored == 6);
    STATIC_REQUIRE(symmetric_tensor<double, 4>::num_stored == 10);
}

TEST_CASE("symmetric tensor symmetry", "[tensor][symmetric]") {
    symmetric_tensor<double, 3> s;
    s(0, 1) = 5.0;
    REQUIRE(s(1, 0) == 5.0);  // automatic symmetry
    REQUIRE(s(0, 1) == 5.0);
}

TEST_CASE("symmetric tensor round-trip to full", "[tensor][symmetric]") {
    symmetric_tensor<double, 3> s;
    s(0, 0) = 1; s(0, 1) = 2; s(0, 2) = 3;
    s(1, 1) = 4; s(1, 2) = 5;
    s(2, 2) = 6;

    auto full = s.to_full();
    REQUIRE(full(0, 1) == full(1, 0));
    REQUIRE(full(0, 2) == full(2, 0));
    REQUIRE(full(1, 2) == full(2, 1));

    auto s2 = symmetric_tensor<double, 3>::from_full(full);
    REQUIRE(s2 == s);
}

TEST_CASE("symmetric tensor trace", "[tensor][symmetric]") {
    symmetric_tensor<double, 3> s;
    s(0, 0) = 1; s(1, 1) = 2; s(2, 2) = 3;
    REQUIRE(trace(s) == Catch::Approx(6.0));
}

// ── Antisymmetric tensor ───────────────────────────────────────────

TEST_CASE("antisymmetric tensor storage size", "[tensor][antisymmetric]") {
    STATIC_REQUIRE(antisymmetric_tensor<double, 3>::num_stored == 3);
    STATIC_REQUIRE(antisymmetric_tensor<double, 4>::num_stored == 6);
}

TEST_CASE("antisymmetric tensor skew symmetry", "[tensor][antisymmetric]") {
    antisymmetric_tensor<double, 3> a;
    a.set(0, 1, 5.0);
    REQUIRE(a(0, 1) == 5.0);
    REQUIRE(a(1, 0) == -5.0);
    REQUIRE(a(0, 0) == 0.0);  // diagonal always zero
}

TEST_CASE("antisymmetric to full round-trip", "[tensor][antisymmetric]") {
    antisymmetric_tensor<double, 3> a;
    a.set(0, 1, 1.0);
    a.set(0, 2, 2.0);
    a.set(1, 2, 3.0);

    auto full = a.to_full();
    REQUIRE(full(0, 1) == -full(1, 0));
    REQUIRE(full(0, 2) == -full(2, 0));
    REQUIRE(full(1, 2) == -full(2, 1));
    // Trace should be zero
    REQUIRE(trace(full) == Catch::Approx(0.0));
}

// ── Metric operations ──────────────────────────────────────────────

TEST_CASE("Euclidean metric raise/lower is identity", "[tensor][metric]") {
    auto g = euclidean_metric<double, 3>();
    tensor<double, 1, 3> v;
    v(0) = 1; v(1) = 2; v(2) = 3;

    auto lowered = lower(v, g);
    REQUIRE(lowered(0) == Catch::Approx(1.0));
    REQUIRE(lowered(1) == Catch::Approx(2.0));
    REQUIRE(lowered(2) == Catch::Approx(3.0));
}

TEST_CASE("Minkowski metric lower changes sign of time component", "[tensor][metric]") {
    auto eta = minkowski_metric<double>();
    tensor<double, 1, 4> v;
    v(0) = 1; v(1) = 2; v(2) = 3; v(3) = 4;  // (t, x, y, z)

    auto lowered = lower(v, eta);
    REQUIRE(lowered(0) == Catch::Approx(-1.0));  // time component flipped
    REQUIRE(lowered(1) == Catch::Approx(2.0));
    REQUIRE(lowered(2) == Catch::Approx(3.0));
    REQUIRE(lowered(3) == Catch::Approx(4.0));
}

TEST_CASE("Minkowski raise/lower round-trip", "[tensor][metric]") {
    auto eta = minkowski_metric<double>();
    // Minkowski is its own inverse: eta^ij = eta_ij
    tensor<double, 1, 4> v;
    v(0) = 5; v(1) = 6; v(2) = 7; v(3) = 8;

    auto lowered = lower(v, eta);
    auto raised = raise(lowered, eta);
    for (std::size_t i = 0; i < 4; ++i)
        REQUIRE(raised(i) == Catch::Approx(v(i)));
}

TEST_CASE("lower_first with identity is identity", "[tensor][metric]") {
    auto g = euclidean_metric<double, 3>();
    tensor<double, 2, 3> t{1, 2, 3, 4, 5, 6, 7, 8, 9};

    auto lowered = lower_first(t, g);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE(lowered(i, j) == Catch::Approx(t(i, j)));
}

// ── Identity tensor ────────────────────────────────────────────────

TEST_CASE("identity tensor properties", "[tensor][identity]") {
    auto I = identity<double, 3>();
    REQUIRE(I(0, 0) == 1.0);
    REQUIRE(I(1, 1) == 1.0);
    REQUIRE(I(2, 2) == 1.0);
    REQUIRE(I(0, 1) == 0.0);
    REQUIRE(I(1, 0) == 0.0);
}

// ── Equality ───────────────────────────────────────────────────────

TEST_CASE("tensor equality", "[tensor][equality]") {
    tensor<int, 2, 2> a{1, 2, 3, 4};
    tensor<int, 2, 2> b{1, 2, 3, 4};
    tensor<int, 2, 2> c{1, 2, 3, 5};
    REQUIRE(a == b);
    REQUIRE(a != c);
}
