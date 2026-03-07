#include <catch2/catch_test_macros.hpp>
#include <mtl/vec/strided_vector_ref.hpp>
#include <mtl/mat/dense2D.hpp>
#include <vector>
#include <numeric>

using namespace mtl;

TEST_CASE("strided_vector_ref from raw pointer", "[vec][strided]") {
    double data[] = {10, 20, 30, 40, 50, 60};
    // stride=2: elements at indices 0, 2, 4 → values 10, 30, 50
    vec::strided_vector_ref<double> v(data, 3, 2);
    REQUIRE(v.size() == 3);
    REQUIRE(v.stride() == 2);
    REQUIRE(v(0) == 10.0);
    REQUIRE(v(1) == 30.0);
    REQUIRE(v(2) == 50.0);
}

TEST_CASE("strided_vector_ref operator[] access", "[vec][strided]") {
    double data[] = {1, 2, 3, 4, 5, 6};
    vec::strided_vector_ref<double> v(data, 3, 2);
    REQUIRE(v[0] == 1.0);
    REQUIRE(v[1] == 3.0);
    REQUIRE(v[2] == 5.0);
}

TEST_CASE("strided_vector_ref stride=1 is contiguous", "[vec][strided]") {
    double data[] = {1, 2, 3, 4};
    vec::strided_vector_ref<double> v(data, 4, 1);
    REQUIRE(v(0) == 1.0);
    REQUIRE(v(1) == 2.0);
    REQUIRE(v(2) == 3.0);
    REQUIRE(v(3) == 4.0);
}

TEST_CASE("strided_vector_ref modification through reference", "[vec][strided]") {
    double data[] = {1, 2, 3, 4, 5, 6};
    vec::strided_vector_ref<double> v(data, 3, 2);
    v(1) = 99.0;
    REQUIRE(data[2] == 99.0);  // element 1 at offset 2
    v[0] = 77.0;
    REQUIRE(data[0] == 77.0);
}

TEST_CASE("strided_vector_ref iterator-based iteration", "[vec][strided]") {
    double data[] = {10, 20, 30, 40, 50, 60};
    vec::strided_vector_ref<double> v(data, 3, 2);

    std::vector<double> collected;
    for (auto val : v) {
        collected.push_back(val);
    }
    REQUIRE(collected.size() == 3);
    REQUIRE(collected[0] == 10.0);
    REQUIRE(collected[1] == 30.0);
    REQUIRE(collected[2] == 50.0);
}

TEST_CASE("strided_vector_ref iterator arithmetic", "[vec][strided]") {
    double data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    vec::strided_vector_ref<double> v(data, 4, 2);

    auto it = v.begin();
    REQUIRE(*it == 1.0);
    ++it;
    REQUIRE(*it == 3.0);
    it += 2;
    REQUIRE(*it == 7.0);
    REQUIRE(v.end() - v.begin() == 4);
}

TEST_CASE("strided_vector_ref sub_vector extraction", "[vec][strided]") {
    double data[] = {10, 20, 30, 40, 50, 60, 70, 80};
    vec::strided_vector_ref<double> v(data, 4, 2);
    // v = {10, 30, 50, 70}
    auto sv = vec::sub_vector(v, 1, 3);
    REQUIRE(sv.size() == 2);
    REQUIRE(sv(0) == 30.0);
    REQUIRE(sv(1) == 50.0);
}

TEST_CASE("strided_vector_ref column extraction from row-major dense2D", "[vec][strided]") {
    // 3x4 row-major matrix:
    //   1  2  3  4
    //   5  6  7  8
    //   9 10 11 12
    mat::dense2D<double> A = {
        {1, 2, 3, 4},
        {5, 6, 7, 8},
        {9, 10, 11, 12}
    };

    // Column 1 = {2, 6, 10}, stride = num_cols = 4
    auto nrows = A.num_rows();
    auto ncols = A.num_cols();
    vec::strided_vector_ref<double> col1(A.data() + 1, nrows, ncols);
    REQUIRE(col1.size() == 3);
    REQUIRE(col1(0) == 2.0);
    REQUIRE(col1(1) == 6.0);
    REQUIRE(col1(2) == 10.0);

    // Column 0 = {1, 5, 9}
    vec::strided_vector_ref<double> col0(A.data(), nrows, ncols);
    REQUIRE(col0(0) == 1.0);
    REQUIRE(col0(1) == 5.0);
    REQUIRE(col0(2) == 9.0);

    // Modification through strided ref updates the matrix
    col1(0) = 99.0;
    REQUIRE(A(0, 1) == 99.0);
}

TEST_CASE("strided_vector_ref traits", "[vec][strided]") {
    using ref_t = vec::strided_vector_ref<double>;
    STATIC_REQUIRE(std::is_same_v<traits::category_t<ref_t>, tag::dense>);
}
