// Vector expression templates (#829): vec_vec_op_expr, vec_negate_expr,
// vec_scal_op_expr, vec_rscal_op_expr and the vec/operators.hpp overloads,
// tested directly rather than only through the operations that use them.
#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <type_traits>

#include <mtl/vec/dense_vector.hpp>
#include <mtl/vec/operators.hpp>
#include <mtl/vec/sparse_vector.hpp>
#include <mtl/traits/is_expression.hpp>

using namespace mtl;
using dvec = vec::dense_vector<double>;

TEST_CASE("vector expressions are lazy and evaluate element-wise", "[vec][expr][829]") {
    const dvec a{1.0, 2.0, 3.0};
    const dvec b{10.0, 20.0, 30.0};
    auto e = a + b;
    STATIC_REQUIRE(traits::is_expression_v<decltype(e)>);
    REQUIRE(e.size() == 3);
    REQUIRE(e(2) == 33.0);

    const dvec sum  = a + b;
    const dvec diff = b - a;
    const dvec neg  = -a;
    const dvec ls   = 2.0 * a;
    const dvec rs   = a * 3.0;
    const dvec dv   = b / 10.0;
    REQUIRE(sum(0) == 11.0);
    REQUIRE(diff(1) == 18.0);
    REQUIRE(neg(2) == -3.0);
    REQUIRE(ls(1) == 4.0);
    REQUIRE(rs(2) == 9.0);
    REQUIRE(dv(0) == 1.0);
}

TEST_CASE("vector expressions: aliasing the target", "[vec][expr][829]") {
    dvec x{1.0, 2.0, 3.0};
    const dvec y{1.0, 1.0, 1.0};
    x = x + y;   // element i reads only x(i), so it is safe in place
    REQUIRE(x(0) == 2.0);
    REQUIRE(x(2) == 4.0);
    x = -x;
    REQUIRE(x(1) == -3.0);
    x = 2.0 * x - x;
    REQUIRE(x(1) == -3.0);
}

TEST_CASE("vector expressions: empty and size-1 operands", "[vec][expr][829]") {
    const dvec e0, f0;
    const dvec s = e0 + f0;
    REQUIRE(s.size() == 0);
    const dvec one{4.0};
    const dvec r = -(one * 2.0) + one;
    REQUIRE(r.size() == 1);
    REQUIRE(r(0) == -4.0);
}

TEST_CASE("vector expressions: rvalue operands are stored by value", "[vec][expr][829]") {
    const dvec a{1.0, 2.0};
    const dvec b{3.0, 4.0};
    const dvec c{5.0, 6.0};
    auto e = a + (b + c);   // the inner expression is a temporary
    const dvec r = e;
    REQUIRE(r(0) == 9.0);
    REQUIRE(r(1) == 12.0);
    auto f = a + dvec{100.0, 200.0};   // a temporary vector, moved into f
    REQUIRE(f(1) == 202.0);
}

TEST_CASE("vector expressions: mixed value types", "[vec][expr][829]") {
    const vec::dense_vector<float> f{1.5f, 2.5f};
    const dvec d{0.25, 0.5};
    auto e = f + d;
    STATIC_REQUIRE(std::is_same_v<typename decltype(e)::value_type, double>);
    REQUIRE(e(0) == 1.75);
    const dvec r = 2 * d;   // an int scalar
    REQUIRE(r(1) == 1.0);

    using cvec = vec::dense_vector<std::complex<double>>;
    const cvec z{{1.0, 2.0}, {0.0, -1.0}};
    const cvec zz = z + z;
    REQUIRE(zz(0) == std::complex<double>(2.0, 4.0));
    const cvec nz = -z;
    REQUIRE(nz(1) == std::complex<double>(0.0, 1.0));
}

TEST_CASE("vector expressions assign into an external buffer", "[vec][expr][829]") {
    double buf[2] = {0, 0};
    vec::dense_vector<double> v(2, buf);
    const dvec a{1.0, 2.0};
    v = a + a;
    REQUIRE(buf[1] == 4.0);
    v += a;
    REQUIRE(buf[1] == 6.0);
    v -= -a;
    REQUIRE(buf[1] == 8.0);
}

TEST_CASE("vector expressions: a sparse operand", "[vec][expr][829]") {
    const dvec a{1.0, 2.0, 3.0, 4.0};
    vec::sparse_vector<double> s(4);
    s.insert(2, 10.0);
    const dvec r = a + s;   // absent entries read as zero
    REQUIRE(r(0) == 1.0);
    REQUIRE(r(2) == 13.0);
    const dvec d = s - a;
    REQUIRE(d(3) == -4.0);
    REQUIRE(d(2) == 7.0);
}
