// The Scalar concept requires std::numeric_limits on the magnitude type (#838),
// and the concepts that had no direct test.
#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <cstdint>
#include <limits>
#include <string>

#include <mtl/mtl.hpp>
#include <mtl/array/ndarray.hpp>
#include <mtl/concepts/linear_operator.hpp>
#include <mtl/concepts/ndarray.hpp>
#include <mtl/concepts/preconditioner.hpp>
#include <mtl5_test/limits_of.hpp>

using namespace mtl;

namespace {

/// The arithmetic of a Scalar, and no numeric_limits.
struct no_limits {
    double v = 0.0;
    no_limits() = default;
    no_limits(double d) : v(d) {}
    friend no_limits operator+(no_limits a, no_limits b) { return a.v + b.v; }
    friend no_limits operator-(no_limits a, no_limits b) { return a.v - b.v; }
    friend no_limits operator*(no_limits a, no_limits b) { return a.v * b.v; }
    friend no_limits operator/(no_limits a, no_limits b) { return a.v / b.v; }
    friend no_limits operator-(no_limits a) { return -a.v; }
};

/// The same, with them.
struct with_limits {
    double v = 0.0;
    with_limits() = default;
    constexpr with_limits(double d) : v(d) {}
    friend with_limits operator+(with_limits a, with_limits b) { return a.v + b.v; }
    friend with_limits operator-(with_limits a, with_limits b) { return a.v - b.v; }
    friend with_limits operator*(with_limits a, with_limits b) { return a.v * b.v; }
    friend with_limits operator/(with_limits a, with_limits b) { return a.v / b.v; }
    friend with_limits operator-(with_limits a) { return -a.v; }
};

} // namespace

template <> class std::numeric_limits<with_limits> : public mtl5_test::limits_of<with_limits, double> {};

TEST_CASE("Scalar requires numeric_limits on the magnitude type", "[concepts][scalar][838]") {
    STATIC_REQUIRE(!HasNumericLimits<no_limits>);
    STATIC_REQUIRE(!Scalar<no_limits>);           // the arithmetic alone is not enough
    STATIC_REQUIRE(!Field<no_limits>);
    STATIC_REQUIRE(HasNumericLimits<with_limits>);
    STATIC_REQUIRE(Scalar<with_limits>);
    STATIC_REQUIRE(Field<with_limits>);
    // std::complex has no numeric_limits in the standard; its value type does.
    STATIC_REQUIRE(!std::numeric_limits<std::complex<double>>::is_specialized);
    STATIC_REQUIRE(Scalar<std::complex<double>>);
    STATIC_REQUIRE(Scalar<std::complex<float>>);
    STATIC_REQUIRE(Scalar<double> && Scalar<float> && Scalar<int> && Scalar<std::uint8_t>);
}

TEST_CASE("magnitude_t and is_complex_v ignore cv-qualifiers", "[concepts][scalar][838]") {
    STATIC_REQUIRE(std::is_same_v<magnitude_t<const std::complex<double>>, double>);
    STATIC_REQUIRE(std::is_same_v<magnitude_t<volatile std::complex<float>>, float>);
    STATIC_REQUIRE(std::is_same_v<magnitude_t<const double>, double>);
    STATIC_REQUIRE(is_complex_v<const std::complex<double>>);
    STATIC_REQUIRE(!is_complex_v<const double>);
    STATIC_REQUIRE(Scalar<const std::complex<double>>);   // failed the new rule without the fix
}

TEST_CASE("Magnitude finds abs through std and ADL", "[concepts][magnitude][838]") {
    STATIC_REQUIRE(Magnitude<double>);   // through std::abs, not C's ::abs(int)
    STATIC_REQUIRE(Magnitude<float>);
    STATIC_REQUIRE(Magnitude<std::complex<double>>);
    STATIC_REQUIRE(!Magnitude<with_limits>);   // no abs of its own
}

TEST_CASE("LinearOperator, Preconditioner, FieldVector, NdArray, MutableNdArray", "[concepts][838]") {
    using V = vec::dense_vector<double>;
    using M = mat::dense2D<double>;
    STATIC_REQUIRE(LinearOperator<M, V>);
    STATIC_REQUIRE(LinearOperator<mat::compressed2D<double>, V>);
    STATIC_REQUIRE(!LinearOperator<std::string, V>);

    STATIC_REQUIRE(Preconditioner<itl::pc::identity<M>, V>);
    STATIC_REQUIRE(!Preconditioner<M, V>);

    STATIC_REQUIRE(FieldVector<V>);
    STATIC_REQUIRE(FieldVector<vec::dense_vector<std::complex<double>>>);
    STATIC_REQUIRE(!FieldVector<vec::dense_vector<int>>);   // integers are a ring

    STATIC_REQUIRE(NdArray<array::ndarray<double, 2>>);
    STATIC_REQUIRE(MutableNdArray<array::ndarray<double, 3>>);
    STATIC_REQUIRE(!NdArray<V>);   // no rank, shape or strides
}
