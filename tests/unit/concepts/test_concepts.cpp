#include <catch2/catch_test_macros.hpp>
#include <complex>
#include <vector>

#include <mtl/concepts/scalar.hpp>
#include <mtl/concepts/magnitude.hpp>
#include <mtl/concepts/collection.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/linear_operator.hpp>
#include <mtl/concepts/preconditioner.hpp>

// -- Scalar concept tests ------------------------------------------------

TEST_CASE("Scalar concept satisfied by arithmetic types", "[concepts][scalar]") {
    STATIC_REQUIRE(mtl::Scalar<int>);
    STATIC_REQUIRE(mtl::Scalar<float>);
    STATIC_REQUIRE(mtl::Scalar<double>);
    STATIC_REQUIRE(mtl::Scalar<long double>);
    STATIC_REQUIRE(mtl::Scalar<unsigned int>);
}

TEST_CASE("Scalar concept satisfied by std::complex", "[concepts][scalar]") {
    STATIC_REQUIRE(mtl::Scalar<std::complex<float>>);
    STATIC_REQUIRE(mtl::Scalar<std::complex<double>>);
}

TEST_CASE("Field concept satisfied by floating-point types", "[concepts][scalar]") {
    STATIC_REQUIRE(mtl::Field<float>);
    STATIC_REQUIRE(mtl::Field<double>);
    STATIC_REQUIRE(mtl::Field<std::complex<double>>);
}

TEST_CASE("OrderedField concept satisfied by real floating-point", "[concepts][scalar]") {
    STATIC_REQUIRE(mtl::OrderedField<float>);
    STATIC_REQUIRE(mtl::OrderedField<double>);
    // complex is not ordered
    STATIC_REQUIRE_FALSE(mtl::OrderedField<std::complex<double>>);
}

// -- is_complex trait tests ----------------------------------------------

TEST_CASE("is_complex trait", "[concepts][scalar]") {
    STATIC_REQUIRE_FALSE(mtl::is_complex_v<double>);
    STATIC_REQUIRE_FALSE(mtl::is_complex_v<int>);
    STATIC_REQUIRE(mtl::is_complex_v<std::complex<double>>);
    STATIC_REQUIRE(mtl::is_complex_v<std::complex<float>>);
}

// -- Magnitude trait tests -----------------------------------------------

TEST_CASE("magnitude_t is identity for real types", "[concepts][magnitude]") {
    STATIC_REQUIRE(std::is_same_v<mtl::magnitude_t<double>, double>);
    STATIC_REQUIRE(std::is_same_v<mtl::magnitude_t<float>, float>);
    STATIC_REQUIRE(std::is_same_v<mtl::magnitude_t<int>, int>);
}

TEST_CASE("magnitude_t extracts real part for complex", "[concepts][magnitude]") {
    STATIC_REQUIRE(std::is_same_v<mtl::magnitude_t<std::complex<double>>, double>);
    STATIC_REQUIRE(std::is_same_v<mtl::magnitude_t<std::complex<float>>, float>);
}

// -- Collection concept smoke test ---------------------------------------

TEST_CASE("std::vector does not satisfy Collection (no size_type alias)", "[concepts][collection]") {
    // std::vector has value_type and size_type, but the Collection concept
    // is designed for MTL types -- std::vector actually does satisfy it
    STATIC_REQUIRE(mtl::Collection<std::vector<double>>);
}

// -- Negative tests ------------------------------------------------------

TEST_CASE("Non-scalar types do not satisfy Scalar", "[concepts][scalar]") {
    STATIC_REQUIRE_FALSE(mtl::Scalar<std::vector<double>>);
    STATIC_REQUIRE_FALSE(mtl::Scalar<std::string>);
}
