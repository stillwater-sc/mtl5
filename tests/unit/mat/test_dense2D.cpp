#include <catch2/catch_test_macros.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/concepts/collection.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/math/identity.hpp>
#include <algorithm>
#include <numeric>

using namespace mtl;

// ── Dimension tests (preserved from skeleton) ───────────────────────────

TEST_CASE("Fixed matrix dimensions", "[mat][dimension]") {
    mtl::mat::fixed::dimensions<3, 4> d;
    REQUIRE(d.num_rows() == 3);
    REQUIRE(d.num_cols() == 4);
    REQUIRE(d.size() == 12);
    STATIC_REQUIRE(d.is_fixed);
}

TEST_CASE("Non-fixed matrix dimensions", "[mat][dimension]") {
    mtl::mat::non_fixed::dimensions d(5, 6);
    REQUIRE(d.num_rows() == 5);
    REQUIRE(d.num_cols() == 6);
    REQUIRE(d.size() == 30);
    STATIC_REQUIRE_FALSE(d.is_fixed);

    d.set_dimensions(10, 20);
    REQUIRE(d.num_rows() == 10);
    REQUIRE(d.num_cols() == 20);
}

TEST_CASE("Matrix parameter bundle compiles", "[mat][parameter]") {
    using default_params = mtl::mat::parameters<>;
    STATIC_REQUIRE(std::is_same_v<default_params::orientation, mtl::tag::row_major>);
    STATIC_REQUIRE(std::is_same_v<default_params::storage, mtl::tag::on_heap>);
    STATIC_REQUIRE_FALSE(default_params::is_fixed);
}

// ── Concept satisfaction ────────────────────────────────────────────────

TEST_CASE("dense2D satisfies Collection concept", "[mat][concept]") {
    STATIC_REQUIRE(Collection<dense2D<double>>);
    STATIC_REQUIRE(MutableCollection<dense2D<double>>);
}

TEST_CASE("dense2D satisfies Matrix concept", "[mat][concept]") {
    STATIC_REQUIRE(Matrix<dense2D<double>>);
    STATIC_REQUIRE(Matrix<dense2D<int>>);
    STATIC_REQUIRE(Matrix<dense2D<float>>);
}

// ── Constructor tests ───────────────────────────────────────────────────

TEST_CASE("Default construction creates empty matrix", "[mat][ctor]") {
    dense2D<double> m;
    REQUIRE(m.num_rows() == 0);
    REQUIRE(m.num_cols() == 0);
    REQUIRE(m.size() == 0);
}

TEST_CASE("(rows, cols) construction", "[mat][ctor]") {
    dense2D<double> m(3, 4);
    REQUIRE(m.num_rows() == 3);
    REQUIRE(m.num_cols() == 4);
    REQUIRE(m.size() == 12);

    // Should be zero-initialized
    for (std::size_t r = 0; r < 3; ++r)
        for (std::size_t c = 0; c < 4; ++c)
            REQUIRE(m(r, c) == 0.0);
}

TEST_CASE("Initializer list construction", "[mat][ctor]") {
    dense2D<int> m = {
        {1, 2, 3},
        {4, 5, 6}
    };
    REQUIRE(m.num_rows() == 2);
    REQUIRE(m.num_cols() == 3);
    REQUIRE(m(0, 0) == 1);
    REQUIRE(m(0, 2) == 3);
    REQUIRE(m(1, 0) == 4);
    REQUIRE(m(1, 2) == 6);
}

TEST_CASE("External pointer construction", "[mat][ctor]") {
    double data[6] = {1, 2, 3, 4, 5, 6};
    dense2D<double> m(2, 3, data);
    REQUIRE(m.num_rows() == 2);
    REQUIRE(m.num_cols() == 3);
    REQUIRE(m.data() == data);
    REQUIRE(m(0, 0) == 1.0);
    REQUIRE(m(0, 2) == 3.0);
    REQUIRE(m(1, 0) == 4.0);

    // Modifying through matrix modifies original data
    m(0, 0) = 99.0;
    REQUIRE(data[0] == 99.0);
}

// ── Copy / Move ─────────────────────────────────────────────────────────

TEST_CASE("Copy construction", "[mat][copy]") {
    dense2D<int> a = {{1, 2}, {3, 4}};
    dense2D<int> b(a);
    REQUIRE(b.num_rows() == 2);
    REQUIRE(b.num_cols() == 2);
    REQUIRE(b(0, 0) == 1);
    REQUIRE(b(1, 1) == 4);
    REQUIRE(b.data() != a.data());
}

TEST_CASE("Move construction", "[mat][move]") {
    dense2D<int> a = {{10, 20}, {30, 40}};
    auto* ptr = a.data();
    dense2D<int> b(std::move(a));
    REQUIRE(b.num_rows() == 2);
    REQUIRE(b.num_cols() == 2);
    REQUIRE(b.data() == ptr);
    REQUIRE(b(1, 1) == 40);
}

// ── Row-major element access ────────────────────────────────────────────

TEST_CASE("Row-major layout", "[mat][access][row_major]") {
    // Default is row_major
    dense2D<int> m(2, 3);
    // Set elements
    int val = 1;
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            m(r, c) = val++;

    // Verify memory layout is row-major: data = {1,2,3,4,5,6}
    REQUIRE(m.data()[0] == 1);  // (0,0)
    REQUIRE(m.data()[1] == 2);  // (0,1)
    REQUIRE(m.data()[2] == 3);  // (0,2)
    REQUIRE(m.data()[3] == 4);  // (1,0)
    REQUIRE(m.data()[4] == 5);  // (1,1)
    REQUIRE(m.data()[5] == 6);  // (1,2)
}

// ── Column-major element access ─────────────────────────────────────────

TEST_CASE("Column-major layout", "[mat][access][col_major]") {
    using col_params = mtl::mat::parameters<mtl::tag::col_major>;
    dense2D<int, col_params> m(2, 3);

    int val = 1;
    for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t c = 0; c < 3; ++c)
            m(r, c) = val++;

    // Column-major: data = {1,4,2,5,3,6} (columns stored contiguously)
    REQUIRE(m.data()[0] == 1);  // (0,0)
    REQUIRE(m.data()[1] == 4);  // (1,0)
    REQUIRE(m.data()[2] == 2);  // (0,1)
    REQUIRE(m.data()[3] == 5);  // (1,1)
    REQUIRE(m.data()[4] == 3);  // (0,2)
    REQUIRE(m.data()[5] == 6);  // (1,2)
}

// ── ldim correctness ────────────────────────────────────────────────────

TEST_CASE("ldim for row-major is num_cols", "[mat][ldim]") {
    dense2D<double> m(3, 5);
    REQUIRE(m.get_ldim() == 5);
}

TEST_CASE("ldim for col-major is num_rows", "[mat][ldim]") {
    using col_params = mtl::mat::parameters<mtl::tag::col_major>;
    dense2D<double, col_params> m(3, 5);
    REQUIRE(m.get_ldim() == 3);
}

// ── Bounds checking ─────────────────────────────────────────────────────

TEST_CASE("Bounds checking in debug builds", "[mat][access]") {
    if constexpr (mtl::bounds_checking) {
        dense2D<int> m(3, 4);
        REQUIRE_THROWS_AS(m(3, 0), std::out_of_range);
        REQUIRE_THROWS_AS(m(0, 4), std::out_of_range);
    }
}

// ── change_dim ──────────────────────────────────────────────────────────

TEST_CASE("change_dim resizes dynamic matrix", "[mat][resize]") {
    dense2D<double> m(3, 4);
    REQUIRE(m.size() == 12);

    m.change_dim(5, 6);
    REQUIRE(m.num_rows() == 5);
    REQUIRE(m.num_cols() == 6);
    REQUIRE(m.size() == 30);
}

// ── Fixed-size on stack ────────────────────────────────────────────────

TEST_CASE("Fixed-size matrix on stack", "[mat][fixed][stack]") {
    using fixed_params = mtl::mat::parameters<
        mtl::tag::row_major,
        mtl::detail::c_index,
        mtl::mat::fixed::dimensions<2, 3>,
        mtl::tag::on_stack
    >;
    dense2D<double, fixed_params> m;
    REQUIRE(m.num_rows() == 2);
    REQUIRE(m.num_cols() == 3);
    REQUIRE(m.size() == 6);

    m(0, 0) = 1.0; m(0, 1) = 2.0; m(0, 2) = 3.0;
    m(1, 0) = 4.0; m(1, 1) = 5.0; m(1, 2) = 6.0;
    REQUIRE(m(1, 2) == 6.0);
}

TEST_CASE("Fixed-size matrix copy", "[mat][fixed][stack]") {
    using fixed_params = mtl::mat::parameters<
        mtl::tag::row_major,
        mtl::detail::c_index,
        mtl::mat::fixed::dimensions<2, 2>,
        mtl::tag::on_stack
    >;
    dense2D<int, fixed_params> a;
    a(0, 0) = 1; a(0, 1) = 2;
    a(1, 0) = 3; a(1, 1) = 4;

    auto b = a;
    REQUIRE(b(0, 0) == 1);
    REQUIRE(b(1, 1) == 4);
    REQUIRE(b.data() != a.data());
}

// ── f_index support ─────────────────────────────────────────────────────

TEST_CASE("Fortran-style 1-based indexing", "[mat][f_index]") {
    using f_params = mtl::mat::parameters<
        mtl::tag::row_major,
        mtl::detail::f_index
    >;
    dense2D<int, f_params> m(2, 3);

    // With f_index, indices are 1-based
    m(1, 1) = 10;
    m(1, 3) = 30;
    m(2, 1) = 40;
    m(2, 3) = 60;

    REQUIRE(m(1, 1) == 10);
    REQUIRE(m(1, 3) == 30);
    REQUIRE(m(2, 1) == 40);
    REQUIRE(m(2, 3) == 60);

    // Internally stored row-major: m(1,1) is at offset 0
    REQUIRE(m.data()[0] == 10);
}

// ── Data pointer ────────────────────────────────────────────────────────

TEST_CASE("data() provides direct access", "[mat][data]") {
    dense2D<int> m = {{1, 2}, {3, 4}};
    int* p = m.data();
    REQUIRE(p != nullptr);
    REQUIRE(p[0] == 1);  // row-major: (0,0)
    REQUIRE(p[3] == 4);  // row-major: (1,1)
}

// ── Swap ────────────────────────────────────────────────────────────────

TEST_CASE("Swap dynamic matrices", "[mat][swap]") {
    dense2D<int> a = {{1, 2}, {3, 4}};
    dense2D<int> b = {{10, 20, 30}};

    a.swap(b);
    REQUIRE(a.num_rows() == 1);
    REQUIRE(a.num_cols() == 3);
    REQUIRE(a(0, 0) == 10);
    REQUIRE(b.num_rows() == 2);
    REQUIRE(b.num_cols() == 2);
    REQUIRE(b(0, 0) == 1);
}

// ── Traits ──────────────────────────────────────────────────────────────

TEST_CASE("dense2D category is dense", "[mat][traits]") {
    STATIC_REQUIRE(std::is_same_v<
        mtl::traits::category_t<dense2D<double>>,
        mtl::tag::dense
    >);
}

TEST_CASE("dense2D ashape is mat", "[mat][traits]") {
    STATIC_REQUIRE(std::is_same_v<
        mtl::ashape::ashape<dense2D<double>>::type,
        mtl::ashape::mat<double>
    >);
}

// ── Iteration ───────────────────────────────────────────────────────────

TEST_CASE("begin/end iteration over all elements", "[mat][iter]") {
    dense2D<int> m = {{1, 2}, {3, 4}};
    int sum = 0;
    for (auto it = m.begin(); it != m.end(); ++it)
        sum += *it;
    REQUIRE(sum == 10);
}

// ── Free functions ──────────────────────────────────────────────────────

TEST_CASE("Free function num_rows/num_cols/size", "[mat][free]") {
    dense2D<double> m(3, 5);
    REQUIRE(mtl::mat::num_rows(m) == 3);
    REQUIRE(mtl::mat::num_cols(m) == 5);
    REQUIRE(mtl::mat::size(m) == 15);
}
