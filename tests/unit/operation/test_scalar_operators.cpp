// MTL5 -- scalar*matrix, matrix*scalar, matrix/scalar and the vector equivalents
// accept any Scalar that is not a Collection (#538), not just is_arithmetic.
//
// The operators were gated on std::is_arithmetic_v<S>, which rejected every
// custom number type MTL5 exists to support (posit, cfloat, LNS, fixpnt) and
// std::complex too, even though all of them satisfy mtl::Scalar and mtl::Field.
// Found writing a Kalman filter over posits: `K = P H^T / S` did not compile.
//
// MTL5 stays Universal-free, so `real_wrapper` stands in for a posit: a class
// type that satisfies Scalar and Field but is not is_arithmetic. std::complex
// covers the standard-library case. The last test case checks that the guard
// ordering keeps matrix*matrix and matrix*vector resolving as before.
#include <catch2/catch_test_macros.hpp>

#include <complex>

#include <mtl/concepts/collection.hpp>
#include <mtl/concepts/scalar.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/operators.hpp>

using namespace mtl;

namespace {

struct real_wrapper {
    double v = 0.0;
    real_wrapper() = default;
    constexpr real_wrapper(double d) : v(d) {}
    friend real_wrapper operator+(real_wrapper a, real_wrapper b) { return a.v + b.v; }
    friend real_wrapper operator-(real_wrapper a, real_wrapper b) { return a.v - b.v; }
    friend real_wrapper operator*(real_wrapper a, real_wrapper b) { return a.v * b.v; }
    friend real_wrapper operator/(real_wrapper a, real_wrapper b) { return a.v / b.v; }
    friend real_wrapper operator-(real_wrapper a) { return -a.v; }
    real_wrapper& operator+=(real_wrapper o) { v += o.v; return *this; }
    real_wrapper& operator-=(real_wrapper o) { v -= o.v; return *this; }
    real_wrapper& operator*=(real_wrapper o) { v *= o.v; return *this; }
    real_wrapper& operator/=(real_wrapper o) { v /= o.v; return *this; }
    friend bool operator==(real_wrapper a, real_wrapper b) { return a.v == b.v; }
};

static_assert(!std::is_arithmetic_v<real_wrapper>);
static_assert(Scalar<real_wrapper> && Field<real_wrapper>);
static_assert(!Collection<real_wrapper>);

using cplx = std::complex<double>;

} // namespace

TEST_CASE("matrix scalar operators accept a non-builtin Scalar", "[operation][operators][scalar]") {
    mat::dense2D<real_wrapper> A(2, 2), B(2, 2);
    A(0, 0) = 1.0; A(0, 1) = 2.0; A(1, 0) = 3.0; A(1, 1) = 4.0;
    const real_wrapper alpha(0.5);

    B = alpha * A;
    CHECK(B(0, 1) == real_wrapper(1.0));
    CHECK(B(1, 1) == real_wrapper(2.0));

    B = A * alpha;
    CHECK(B(1, 0) == real_wrapper(1.5));

    B = A / alpha;
    CHECK(B(0, 0) == real_wrapper(2.0));
    CHECK(B(1, 1) == real_wrapper(8.0));
}

TEST_CASE("vector scalar operators accept a non-builtin Scalar", "[operation][operators][scalar]") {
    vec::dense_vector<real_wrapper> v(3), w(3);
    v(0) = 1.0; v(1) = 2.0; v(2) = 3.0;
    const real_wrapper alpha(0.5);

    w = alpha * v;
    CHECK(w(2) == real_wrapper(1.5));

    w = v * alpha;
    CHECK(w(0) == real_wrapper(0.5));

    w = v / alpha;
    CHECK(w(1) == real_wrapper(4.0));
}

TEST_CASE("scalar operators accept std::complex", "[operation][operators][scalar][complex]") {
    const cplx i(0.0, 1.0);

    mat::dense2D<cplx> Z(2, 2), Y(2, 2);
    Z(0, 0) = 1.0; Z(0, 1) = cplx(0, 2); Z(1, 0) = 3.0; Z(1, 1) = cplx(4, 4);
    Y = i * Z;
    CHECK(Y(0, 0) == cplx(0, 1));
    CHECK(Y(0, 1) == cplx(-2, 0));
    Y = Z * i;
    CHECK(Y(1, 1) == cplx(-4, 4));
    Y = Z / i;
    CHECK(Y(1, 0) == cplx(0, -3));

    vec::dense_vector<cplx> v(2), w(2);
    v(0) = 1.0; v(1) = cplx(0, 1);
    w = i * v;
    CHECK(w(1) == cplx(-1, 0));
    w = v * i;
    CHECK(w(0) == cplx(0, 1));
    w = v / i;
    CHECK(w(0) == cplx(0, -1));
}

TEST_CASE("container operands still resolve to the container operators",
          "[operation][operators][scalar]") {
    // The scalar gate is (!Collection<S> && Scalar<S>). If Scalar<S> were
    // evaluated for a container, checking Scalar<dense2D<T>> would recurse into
    // these operators and fail to compile -- so compiling at all is half the
    // test. The values confirm each call reached the right overload.
    mat::dense2D<double> A(2, 2), B(2, 2), C(2, 2);
    A(0, 0) = 1; A(0, 1) = 2; A(1, 0) = 3; A(1, 1) = 4;
    B(0, 0) = 1; B(0, 1) = 0; B(1, 0) = 0; B(1, 1) = 1;
    vec::dense_vector<double> x = {1.0, 1.0};

    C = A * B;                         // matrix * matrix
    CHECK(C(1, 0) == 3.0);
    vec::dense_vector<double> y = A * x;   // matrix * vector
    CHECK(y(1) == 7.0);
    C = 2 * A;                         // int literal scalar, unchanged
    CHECK(C(1, 1) == 8.0);
    C = 2.0 * (A + B);                 // scalar * expression template
    CHECK(C(0, 0) == 4.0);
    C = (A * 0.5) / 0.5;               // chained scalar expressions
    CHECK(C(0, 1) == 2.0);
}
