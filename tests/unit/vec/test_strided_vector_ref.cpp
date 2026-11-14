#include <catch2/catch_test_macros.hpp>
#include <mtl/vec/strided_vector_ref.hpp>
#include <mtl/mat/dense2D.hpp>
#include <vector>
#include <numeric>

using namespace mtl;

TEST_CASE("strided_vector_ref from raw pointer", "[vec][strided]") {
    double data[] = {10, 20, 30, 40, 50, 60};
    // stride=2: elements at indices 0, 2, 4 -> values 10, 30, 50
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

// -- #830: the iterator range, size 0 and 1, const access, expressions ---

#include <algorithm>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/dot.hpp>

TEST_CASE("strided_vector_ref sizes 0 and 1", "[vec][strided][830]") {
    double one = 4.0;
    vec::strided_vector_ref<double> e(&one, 0, 3);
    REQUIRE(e.begin() == e.end());
    REQUIRE(e.end() - e.begin() == 0);
    vec::strided_vector_ref<double> s(&one, 1, 3);
    REQUIRE(s.end() - s.begin() == 1);
    REQUIRE(*s.begin() == 4.0);
}

TEST_CASE("strided_vector_ref stride 0 views one element", "[vec][strided][830]") {
    double x = 2.0;
    const vec::strided_vector_ref<double> b(&x, 5, 0);
    REQUIRE(b.end() - b.begin() == 5);   // divided by the stride before #830
    double s = 0;
    for (double v : b) s += v;
    REQUIRE(s == 10.0);
}

TEST_CASE("strided_vector_ref over the last column of a row-major matrix", "[vec][strided][830]") {
    // Before #830 end() was data + 3 + 3 * 4, past one past the end of the
    // 12-element buffer. Now no pointer is formed beyond the last element read.
    mat::dense2D<double> A = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}};
    vec::strided_vector_ref<double> col3(A.data() + 3, 3, 4);
    std::vector<double> got(col3.begin(), col3.end());
    REQUIRE(got == std::vector<double>{4, 8, 12});
    REQUIRE(std::count(col3.begin(), col3.end(), 8.0) == 1);
    auto last = col3.end();
    --last;
    REQUIRE(*last == 12.0);
    REQUIRE(col3.begin()[2] == 12.0);
}

TEST_CASE("strided_vector_ref const iterators and const sub_vector", "[vec][strided][830]") {
    double data[] = {0, 1, 2, 3, 4, 5, 6, 7};
    const vec::strided_vector_ref<double> v(data, 4, 2);   // 0 2 4 6
    double s = 0;
    for (auto it = v.begin(); it != v.end(); ++it) s += *it;
    REQUIRE(s == 12.0);
    auto sub = vec::sub_vector(v, 1, 3);                   // 2 4, read-only
    STATIC_REQUIRE(std::is_same_v<decltype(sub), vec::strided_vector_ref<const double>>);
    REQUIRE(sub.size() == 2);
    REQUIRE(sub(0) == 2.0);
    REQUIRE(sub(1) == 4.0);
    auto tail = vec::sub_vector(v, 4, 4);                  // empty, at the end
    REQUIRE(tail.size() == 0);
    REQUIRE(tail.begin() == tail.end());
}

TEST_CASE("strided_vector_ref as an operand", "[vec][strided][830]") {
    mat::dense2D<double> A = {{1, 2}, {3, 4}, {5, 6}};
    const vec::strided_vector_ref<double> c0(A.data(), 3, 2), c1(A.data() + 1, 3, 2);
    REQUIRE(dot(c0, c1) == 1 * 2 + 3 * 4 + 5 * 6);
}
