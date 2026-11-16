// tensor, symmetric_tensor and antisymmetric_tensor: bounds, construction,
// contraction patterns, the norm of a complex tensor, and the API the sweep in
// #832 found untested (#857).
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <complex>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

#include <mtl/config.hpp>
#include <mtl/tensor/index.hpp>
#include <mtl/tensor/metric.hpp>
#include <mtl/tensor/properties.hpp>
#include <mtl/tensor/symmetric.hpp>
#include <mtl/tensor/tensor.hpp>

using namespace mtl::tensor;

TEST_CASE("tensor bounds are checked per index", "[tensor][857]") {
    tensor<double, 2, 3> t;
    t(1, 0) = 5.0;
    if constexpr (mtl::bounds_checking) {
        // t(0, 3) has flat offset 3, inside the 9 components: it passed and
        // aliased t(1, 0).
        CHECK_THROWS_AS(t(0, 3), std::out_of_range);
        CHECK_THROWS_AS((t[{3, 0}]), std::out_of_range);
        const auto& ct = t;
        CHECK_THROWS_AS(ct(2, 3), std::out_of_range);
        symmetric_tensor<double, 3> s;
        CHECK_THROWS_AS(s(0, 3), std::out_of_range);
        CHECK_THROWS_AS(std::as_const(s)(3, 0), std::out_of_range);
        antisymmetric_tensor<double, 3> a;
        CHECK_THROWS_AS(a(0, 3), std::out_of_range);
        CHECK_THROWS_AS(a.set(3, 1, 1.0), std::out_of_range);
    }
    CHECK(t(1, 0) == 5.0);
}

TEST_CASE("tensor initializer list: shorter is zero-filled, longer throws", "[tensor][857]") {
    tensor<double, 2, 2> a{1, 2};
    CHECK((a(0, 0) == 1 && a(0, 1) == 2 && a(1, 0) == 0 && a(1, 1) == 0));
    CHECK_THROWS_AS((tensor<double, 2, 2>{1, 2, 3, 4, 5}), std::invalid_argument);
}

TEST_CASE("contract with a transposed pattern", "[tensor][857]") {
    Index<'i'> i; Index<'j'> j; Index<'k'> k;
    const tensor<double, 2, 2> A{1, 2, 3, 4}, B{5, 6, 7, 8};
    // A(j, i) B(j, k): (A^T B)(i, k).
    const auto C = contract(bind(A, j, i), bind(B, j, k));
    CHECK((C(0, 0) == 1 * 5 + 3 * 7 && C(0, 1) == 1 * 6 + 3 * 8));
    CHECK((C(1, 0) == 2 * 5 + 4 * 7 && C(1, 1) == 2 * 6 + 4 * 8));
    // A(i, j) B(k, j): (A B^T)(i, k).
    const auto D = contract(bind(A, i, j), bind(B, k, j));
    CHECK((D(0, 0) == 1 * 5 + 2 * 6 && D(1, 0) == 3 * 5 + 4 * 6));
    // A(j, i) v(j): A^T v.
    const tensor<double, 1, 2> v{1, 10};
    const auto w = contract(bind(A, j, i), bind(v, j));
    CHECK((w(0) == 1 + 30 && w(1) == 2 + 40));
}

TEST_CASE("frobenius_norm of a complex tensor is a real norm", "[tensor][857]") {
    using C = std::complex<double>;
    tensor<C, 1, 2> z;
    z(0) = C(3, 4);   // |z0| = 5
    z(1) = C(0, 12);  // |z1| = 12
    const auto n = frobenius_norm(z);
    STATIC_REQUIRE(std::is_same_v<std::remove_const_t<decltype(n)>, double>);
    CHECK(n == 13.0);   // it was sqrt((3+4i)^2 + (12i)^2), not a norm
    CHECK(trace(tensor<C, 2, 2>{C(1, 1), C(0), C(0), C(2, -1)}) == C(3, 0));
}

TEST_CASE("tensor arithmetic and fill", "[tensor][857]") {
    tensor<double, 2, 2> a{4, 6, 8, 10}, b{1, 2, 3, 4};
    const auto d = a - b;
    CHECK((d(0, 0) == 3 && d(1, 1) == 6));
    const auto q = a / 2.0;
    CHECK((q(0, 1) == 3 && q(1, 0) == 4));
    a -= b;
    CHECK(a(1, 0) == 5);
    a *= 2.0;
    CHECK(a(1, 1) == 12);
    a.fill(7.0);
    CHECK((a(0, 0) == 7 && a(1, 1) == 7));
    a.set_to_zero();
    CHECK((a(0, 0) == 0 && a(1, 1) == 0));
    // Rank 0 and dimension 1.
    tensor<double, 0, 3> s;
    s() = 2.5;
    CHECK((tensor<double, 0, 3>::size() == 1 && s() == 2.5));
    tensor<double, 2, 1> one{4.0};
    CHECK((determinant(tensor<double, 2, 2>{2, 1, 1, 3}) == 5.0 && trace(one) == 4.0));
    // The rank-4 outer product.
    const tensor<double, 2, 2> m{1, 2, 3, 4};
    const auto o = outer(m, m);
    CHECK((o(1, 0, 0, 1) == 3 * 2 && o(1, 1, 1, 1) == 16));
}

TEST_CASE("symmetric and antisymmetric tensor operations", "[tensor][857]") {
    symmetric_tensor<double, 3> a, b;
    a(0, 1) = 2; a(2, 2) = 5;
    b(1, 0) = 1; b(2, 2) = 1;
    const auto d = a - b;
    CHECK((d(1, 0) == 1 && d(2, 2) == 4));
    const auto s = 3.0 * a;
    CHECK(s(1, 0) == 6);
    auto c = a;
    c -= b;
    c *= 2.0;
    CHECK((c(0, 1) == 2 && c(2, 2) == 8));
    CHECK(a == a);
    CHECK(a != b);
    antisymmetric_tensor<double, 3> x, y;
    x.set(0, 2, 4.0);
    y.set(2, 0, -4.0);   // the same tensor, set from the other side
    CHECK(x == y);
    CHECK(x(2, 0) == -4.0);
    CHECK(x(1, 1) == 0.0);
}

TEST_CASE("metrics and index raising in 3 dimensions", "[tensor][857]") {
    const auto g = euclidean_metric<double, 3>();
    const tensor<double, 1, 3> v{1, 2, 3};
    const auto lo = lower(v, g);
    const auto up = raise(lo, g);
    CHECK((up(0) == 1 && up(2) == 3));
    const tensor<double, 2, 3> t{1, 2, 3, 4, 5, 6, 7, 8, 9};
    CHECK(raise_first(t, g) == t);
    CHECK(raise_second(t, g) == t);
    CHECK(lower_second(t, g) == t);
}

TEST_CASE("is_symmetric with a tolerance", "[tensor][857]") {
    tensor<double, 2, 2> t{1, 2, 2.0 + 1e-9, 1};
    CHECK_FALSE(is_symmetric(t));
    CHECK(is_symmetric(t, 1e-8));
    CHECK_FALSE(is_symmetric(t, 1e-10));
}
