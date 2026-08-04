#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/array/ndarray.hpp>
#include <mtl/array/slice.hpp>
#include <mtl/array/interop.hpp>

using namespace mtl;
using namespace mtl::array;

// -- Vector -> ndarray round-trip ------------------------------------

TEST_CASE("dense_vector to ndarray view (zero-copy)", "[interop][vec]") {
    vec::dense_vector<double> v = {1.0, 2.0, 3.0, 4.0};
    auto a = as_ndarray(v);

    REQUIRE(a.size() == 4);
    REQUIRE(a.is_view());
    REQUIRE(a.data() == v.data());  // zero-copy
    REQUIRE(a(0) == 1.0);
    REQUIRE(a(3) == 4.0);
}

TEST_CASE("mutation through ndarray view visible in vector", "[interop][vec]") {
    vec::dense_vector<int> v = {10, 20, 30};
    auto a = as_ndarray(v);

    a(1) = 99;
    REQUIRE(v(1) == 99);  // changed in original
}

TEST_CASE("const dense_vector to const ndarray view", "[interop][vec]") {
    const vec::dense_vector<double> v = {1.0, 2.0, 3.0};
    auto a = as_ndarray(v);

    REQUIRE(a.size() == 3);
    REQUIRE(a(0) == 1.0);
    // a(0) = 5.0;  // should not compile (const)
}

// -- Matrix -> ndarray round-trip ------------------------------------

TEST_CASE("dense2D to ndarray view (zero-copy)", "[interop][mat]") {
    mat::dense2D<double> m(3, 4);
    m(0, 0) = 1.0; m(1, 2) = 42.0; m(2, 3) = 99.0;

    auto a = as_ndarray(m);

    REQUIRE(a.extent(0) == 3);
    REQUIRE(a.extent(1) == 4);
    REQUIRE(a.is_view());
    REQUIRE(a.data() == m.data());
    REQUIRE(a(0, 0) == 1.0);
    REQUIRE(a(1, 2) == 42.0);
    REQUIRE(a(2, 3) == 99.0);
}

TEST_CASE("mutation through ndarray view visible in matrix", "[interop][mat]") {
    mat::dense2D<int> m(2, 3);
    auto a = as_ndarray(m);

    a(1, 1) = 77;
    REQUIRE(m(1, 1) == 77);
}

// -- ndarray -> vector round-trip ------------------------------------

TEST_CASE("ndarray to dense_vector view", "[interop][vec]") {
    ndarray<double, 1> a(shape<1>{5});
    for (std::size_t i = 0; i < 5; ++i) a(i) = static_cast<double>(i);

    auto v = as_vector(a);
    REQUIRE(v.size() == 5);
    REQUIRE(v(0) == 0.0);
    REQUIRE(v(4) == 4.0);
    REQUIRE(v.data() == a.data());  // zero-copy
}

TEST_CASE("ndarray to dense2D view", "[interop][mat]") {
    ndarray<double, 2> a({3, 4});
    a(1, 2) = 42.0;

    auto m = as_matrix(a);
    REQUIRE(m.num_rows() == 3);
    REQUIRE(m.num_cols() == 4);
    REQUIRE(m(1, 2) == 42.0);
    REQUIRE(m.data() == a.data());  // zero-copy
}

// -- Matrix -> ndarray -> slice row -> verify 1D view -----------------

TEST_CASE("matrix to ndarray, slice row, get vector view", "[interop][slice]") {
    mat::dense2D<int> m(3, 4);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            m(i, j) = static_cast<int>(i * 4 + j);

    auto a = as_ndarray(m);
    auto row1 = slice(a, 1, all);  // row 1 -> 1D view

    REQUIRE(row1.size() == 4);
    REQUIRE(row1(0) == 4);
    REQUIRE(row1(1) == 5);
    REQUIRE(row1(2) == 6);
    REQUIRE(row1(3) == 7);

    // Mutation through sliced view
    row1(2) = 99;
    REQUIRE(m(1, 2) == 99);
}

// -- Generic algorithms ---------------------------------------------

TEST_CASE("transform on ndarray", "[interop][algorithm]") {
    ndarray<int, 1> a(shape<1>{4});
    a(0) = 1; a(1) = 2; a(2) = 3; a(3) = 4;

    auto doubled = transform(a, [](int x) { return x * 2; });
    REQUIRE(doubled(0) == 2);
    REQUIRE(doubled(3) == 8);
}

TEST_CASE("transform_inplace on ndarray from vector", "[interop][algorithm]") {
    vec::dense_vector<double> v = {1.0, 4.0, 9.0, 16.0};
    auto a = as_ndarray(v);

    transform_inplace(a, [](double x) { return x + 1.0; });
    REQUIRE(v(0) == Catch::Approx(2.0));
    REQUIRE(v(1) == Catch::Approx(5.0));
    REQUIRE(v(3) == Catch::Approx(17.0));
}

TEST_CASE("reduce on ndarray", "[interop][algorithm]") {
    ndarray<int, 2> a({2, 3}, 5);
    int total = reduce(a, 0, std::plus<int>{});
    REQUIRE(total == 30);
}

TEST_CASE("reduce on vector via ndarray", "[interop][algorithm]") {
    vec::dense_vector<double> v = {1.0, 2.0, 3.0, 4.0};
    auto a = as_ndarray(v);

    double total = reduce(a, 0.0, std::plus<double>{});
    REQUIRE(total == Catch::Approx(10.0));
}

TEST_CASE("flatten 2D to 1D", "[interop][algorithm]") {
    ndarray<int, 2> a({2, 3});
    int k = 0;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            a(i, j) = k++;

    auto flat = flatten(a);
    REQUIRE(flat.size() == 6);
    for (int i = 0; i < 6; ++i)
        REQUIRE(flat(i) == i);
}

TEST_CASE("flatten matrix via ndarray", "[interop][algorithm]") {
    mat::dense2D<int> m(2, 3);
    m(0, 0) = 1; m(0, 1) = 2; m(0, 2) = 3;
    m(1, 0) = 4; m(1, 1) = 5; m(1, 2) = 6;

    auto a = as_ndarray(m);
    auto flat = flatten(a);
    REQUIRE(flat.size() == 6);
    REQUIRE(flat(0) == 1);
    REQUIRE(flat(5) == 6);
}

// ---------------------------------------------------------------------------
// #359: flatten inherited the memory-order bug through for_each_element, and
// as_matrix aliased an F-contiguous array as row-major dense2D -- silently the
// transpose. as_vector is unaffected: at rank 1 the C and F strides coincide.
// ---------------------------------------------------------------------------

TEST_CASE("flatten returns logical order for an F-contiguous array (#359)",
          "[array][interop][regression]") {
    ndarray<double, 2> a(shape<2>{2, 3});
    double v = 1.0;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j) a(i, j) = v++;

    // a   = [[1,2,3],[4,5,6]]        memory: 1 2 3 4 5 6
    // a^T = [[1,4],[2,5],[3,6]]      NumPy: a.T.flatten() == [1,4,2,5,3,6]
    auto t = a.transpose();
    auto f = flatten(t);

    REQUIRE(f.size() == 6);
    const double expected[6] = {1, 4, 2, 5, 3, 6};
    for (std::size_t i = 0; i < 6; ++i) {
        INFO("i = " << i);
        REQUIRE(f(i) == Catch::Approx(expected[i]));
    }

    // The C-contiguous case is unchanged.
    auto g = flatten(a);
    for (std::size_t i = 0; i < 6; ++i)
        REQUIRE(g(i) == Catch::Approx(static_cast<double>(i + 1)));
}

TEST_CASE("as_matrix aliases a C-contiguous ndarray correctly (#359)",
          "[array][interop][regression]") {
    // The positive direction. The negative one -- an F-contiguous array passed
    // to as_matrix -- is a precondition violation caught by assert in debug
    // builds, so it cannot be expressed as a runtime test here.
    ndarray<double, 2> a(shape<2>{2, 3});
    double v = 1.0;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j) a(i, j) = v++;

    REQUIRE(a.is_c_contiguous());
    auto M = as_matrix(a);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j) {
            INFO("i = " << i << ", j = " << j);
            REQUIRE(M(i, j) == Catch::Approx(a(i, j)));
        }

    // And the transpose is exactly the case the precondition now rejects.
    auto t = a.transpose();
    REQUIRE(t.is_contiguous());          // the old, too-weak predicate
    REQUIRE_FALSE(t.is_c_contiguous());  // the one as_matrix now requires
}
