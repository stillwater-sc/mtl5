#include <catch2/catch_test_macros.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/concepts/collection.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/math/identity.hpp>
#include <algorithm>
#include <numeric>
#include <vector>

using namespace mtl;

// -- Dimension tests (preserved from skeleton) ---------------------------

TEST_CASE("Fixed vector dimension", "[vec][dimension]") {
    mtl::vec::fixed::dimension<5> d;
    REQUIRE(d.size() == 5);
    STATIC_REQUIRE(d.is_fixed);
    STATIC_REQUIRE(d.value == 5);
}

TEST_CASE("Non-fixed vector dimension", "[vec][dimension]") {
    mtl::vec::non_fixed::dimension d(10);
    REQUIRE(d.size() == 10);
    STATIC_REQUIRE_FALSE(d.is_fixed);
    STATIC_REQUIRE(d.value == 0);

    d.set_size(20);
    REQUIRE(d.size() == 20);
}

TEST_CASE("Vector parameter bundle compiles", "[vec][parameter]") {
    using default_params = mtl::vec::parameters<>;
    STATIC_REQUIRE(std::is_same_v<default_params::orientation, mtl::tag::col_major>);
    STATIC_REQUIRE(std::is_same_v<default_params::storage, mtl::tag::on_heap>);
    STATIC_REQUIRE_FALSE(default_params::is_fixed);
}

// -- Concept satisfaction ------------------------------------------------

TEST_CASE("dense_vector satisfies Collection concept", "[vec][concept]") {
    STATIC_REQUIRE(Collection<dense_vector<double>>);
    STATIC_REQUIRE(MutableCollection<dense_vector<double>>);
}

TEST_CASE("dense_vector satisfies Vector concept", "[vec][concept]") {
    STATIC_REQUIRE(Vector<dense_vector<double>>);
    STATIC_REQUIRE(Vector<dense_vector<int>>);
    STATIC_REQUIRE(Vector<dense_vector<float>>);
}

// -- Constructor tests ---------------------------------------------------

TEST_CASE("Default construction creates empty vector", "[vec][ctor]") {
    dense_vector<double> v;
    REQUIRE(v.size() == 0);
    REQUIRE(v.empty());
}

TEST_CASE("Size construction", "[vec][ctor]") {
    dense_vector<double> v(10);
    REQUIRE(v.size() == 10);
    REQUIRE_FALSE(v.empty());

    for (std::size_t i = 0; i < v.size(); ++i) {
        REQUIRE(v(i) == 0.0);
    }
}

TEST_CASE("Size + fill value construction", "[vec][ctor]") {
    dense_vector<int> v(5, 42);
    REQUIRE(v.size() == 5);
    for (std::size_t i = 0; i < v.size(); ++i) {
        REQUIRE(v(i) == 42);
    }
}

TEST_CASE("Initializer list construction", "[vec][ctor]") {
    dense_vector<double> v = {1.0, 2.0, 3.0, 4.0};
    REQUIRE(v.size() == 4);
    REQUIRE(v(0) == 1.0);
    REQUIRE(v(1) == 2.0);
    REQUIRE(v(2) == 3.0);
    REQUIRE(v(3) == 4.0);
}

TEST_CASE("std::vector construction", "[vec][ctor]") {
    std::vector<int> sv = {10, 20, 30};
    dense_vector<int> v(sv);
    REQUIRE(v.size() == 3);
    REQUIRE(v(0) == 10);
    REQUIRE(v(1) == 20);
    REQUIRE(v(2) == 30);
}

TEST_CASE("External pointer construction", "[vec][ctor]") {
    double data[3] = {1.5, 2.5, 3.5};
    dense_vector<double> v(3, data);
    REQUIRE(v.size() == 3);
    REQUIRE(v.data() == data);
    REQUIRE(v(0) == 1.5);

    // Modifying through vector modifies original data
    v(0) = 99.0;
    REQUIRE(data[0] == 99.0);
}

// -- Copy / Move ---------------------------------------------------------

TEST_CASE("Copy construction", "[vec][copy]") {
    dense_vector<int> a = {1, 2, 3};
    dense_vector<int> b(a);
    REQUIRE(b.size() == 3);
    REQUIRE(b(0) == 1);
    REQUIRE(b.data() != a.data());  // deep copy
}

TEST_CASE("Move construction", "[vec][move]") {
    dense_vector<int> a = {10, 20, 30};
    auto* ptr = a.data();
    dense_vector<int> b(std::move(a));
    REQUIRE(b.size() == 3);
    REQUIRE(b.data() == ptr);
    REQUIRE(b(2) == 30);
    REQUIRE(a.empty());
}

TEST_CASE("Copy assignment", "[vec][copy]") {
    dense_vector<int> a = {5, 6, 7};
    dense_vector<int> b;
    b = a;
    REQUIRE(b.size() == 3);
    REQUIRE(b(0) == 5);
}

TEST_CASE("Move assignment", "[vec][move]") {
    dense_vector<int> a = {5, 6, 7};
    auto* ptr = a.data();
    dense_vector<int> b;
    b = std::move(a);
    REQUIRE(b.size() == 3);
    REQUIRE(b.data() == ptr);
}

// -- Element access ------------------------------------------------------

TEST_CASE("operator() and operator[]", "[vec][access]") {
    dense_vector<int> v = {10, 20, 30};
    REQUIRE(v(0) == 10);
    REQUIRE(v[1] == 20);

    v(2) = 99;
    REQUIRE(v[2] == 99);
}

TEST_CASE("Bounds checking in debug builds", "[vec][access]") {
    if constexpr (mtl::bounds_checking) {
        dense_vector<int> v(3);
        REQUIRE_THROWS_AS(v(3), std::out_of_range);
        REQUIRE_THROWS_AS(v(100), std::out_of_range);
    }
}

// -- Iteration -----------------------------------------------------------

TEST_CASE("begin/end iteration", "[vec][iter]") {
    dense_vector<int> v = {1, 2, 3, 4, 5};
    int sum = 0;
    for (auto it = v.begin(); it != v.end(); ++it)
        sum += *it;
    REQUIRE(sum == 15);
}

TEST_CASE("Range-based for", "[vec][iter]") {
    dense_vector<int> v = {10, 20, 30};
    int sum = 0;
    for (auto& x : v) sum += x;
    REQUIRE(sum == 60);
}

// -- Fill / set_to_zero --------------------------------------------------

TEST_CASE("vec::fill", "[vec][fill]") {
    dense_vector<double> v(5);
    mtl::vec::fill(v, 3.14);
    for (std::size_t i = 0; i < v.size(); ++i) {
        REQUIRE(v(i) == 3.14);
    }
}

// -- Orientation-aware num_rows/num_cols ---------------------------------

TEST_CASE("Column vector: num_rows=size, num_cols=1", "[vec][orientation]") {
    // Default is col_major
    dense_vector<double> v(5);
    REQUIRE(v.num_rows() == 5);
    REQUIRE(v.num_cols() == 1);
}

TEST_CASE("Row vector: num_rows=1, num_cols=size", "[vec][orientation]") {
    using row_params = mtl::vec::parameters<mtl::tag::row_major>;
    dense_vector<double, row_params> v(5);
    REQUIRE(v.num_rows() == 1);
    REQUIRE(v.num_cols() == 5);
}

// -- change_dim ----------------------------------------------------------

TEST_CASE("change_dim resizes dynamic vector", "[vec][resize]") {
    dense_vector<double> v(5, 1.0);
    REQUIRE(v.size() == 5);

    v.change_dim(10);
    REQUIRE(v.size() == 10);

    v.change_dim(3);
    REQUIRE(v.size() == 3);
}

// -- Fixed-size on stack ------------------------------------------------

TEST_CASE("Fixed-size vector on stack", "[vec][fixed][stack]") {
    using fixed_params = mtl::vec::parameters<
        mtl::tag::col_major,
        mtl::vec::fixed::dimension<3>,
        mtl::tag::on_stack
    >;
    dense_vector<double, fixed_params> v;
    REQUIRE(v.size() == 3);

    v(0) = 1.0; v(1) = 2.0; v(2) = 3.0;
    REQUIRE(v(0) == 1.0);
    REQUIRE(v(1) == 2.0);
    REQUIRE(v(2) == 3.0);
}

TEST_CASE("Fixed-size vector on stack: copy", "[vec][fixed][stack]") {
    using fixed_params = mtl::vec::parameters<
        mtl::tag::col_major,
        mtl::vec::fixed::dimension<3>,
        mtl::tag::on_stack
    >;
    dense_vector<double, fixed_params> a;
    a(0) = 10.0; a(1) = 20.0; a(2) = 30.0;

    auto b = a;
    REQUIRE(b(0) == 10.0);
    REQUIRE(b(2) == 30.0);
    REQUIRE(b.data() != a.data());
}

// -- Swap ----------------------------------------------------------------

TEST_CASE("Swap dynamic vectors", "[vec][swap]") {
    dense_vector<int> a = {1, 2, 3};
    dense_vector<int> b = {10, 20};

    a.swap(b);
    REQUIRE(a.size() == 2);
    REQUIRE(a(0) == 10);
    REQUIRE(b.size() == 3);
    REQUIRE(b(0) == 1);
}

// -- Traits --------------------------------------------------------------

TEST_CASE("dense_vector category is dense", "[vec][traits]") {
    STATIC_REQUIRE(std::is_same_v<
        mtl::traits::category_t<dense_vector<double>>,
        mtl::tag::dense
    >);
}

TEST_CASE("dense_vector ashape is cvec for col_major", "[vec][traits]") {
    STATIC_REQUIRE(std::is_same_v<
        mtl::ashape::ashape<dense_vector<double>>::type,
        mtl::ashape::cvec<double>
    >);
}

TEST_CASE("dense_vector ashape is rvec for row_major", "[vec][traits]") {
    using row_params = mtl::vec::parameters<mtl::tag::row_major>;
    STATIC_REQUIRE(std::is_same_v<
        mtl::ashape::ashape<dense_vector<double, row_params>>::type,
        mtl::ashape::rvec<double>
    >);
}

// -- Stride --------------------------------------------------------------

TEST_CASE("stride is always 1", "[vec][stride]") {
    STATIC_REQUIRE(dense_vector<double>::stride() == 1);
}

// -- Free functions ------------------------------------------------------

TEST_CASE("Free function size()", "[vec][free]") {
    dense_vector<int> v = {1, 2, 3, 4};
    REQUIRE(mtl::vec::size(v) == 4);
}

TEST_CASE("Free function num_rows/num_cols", "[vec][free]") {
    dense_vector<int> v(3);
    REQUIRE(mtl::vec::num_rows(v) == 3);
    REQUIRE(mtl::vec::num_cols(v) == 1);
}

// -- #829: external-pointer assignment, fixed heap vectors, (n, 0) --------

#include <mtl/vec/operators.hpp>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {
using fixed_heap_vec = vec::dense_vector<double,
    vec::parameters<tag::col_major, vec::fixed::dimension<3>, tag::on_heap>>;
}

TEST_CASE("dense_vector: copy assignment writes an external buffer", "[vec][dense_vector][829]") {
    double buf[3] = {1, 2, 3};
    vec::dense_vector<double> v(3, buf);
    const vec::dense_vector<double> w(3, 9.0);
    v = w;   // fails before #829: v detached into a private copy
    REQUIRE(v.data() == buf);
    REQUIRE(!v.owns_memory());
    REQUIRE(buf[0] == 9.0);
    REQUIRE(buf[2] == 9.0);

    v = vec::dense_vector<double>{4.0, 5.0, 6.0};   // move assignment
    REQUIRE(v.data() == buf);
    REQUIRE(buf[1] == 5.0);

    v = w + w;   // expression assignment, already write-through
    REQUIRE(buf[2] == 18.0);
}

TEST_CASE("dense_vector: assigning another size to an external vector throws", "[vec][dense_vector][829]") {
    double buf[3] = {1, 2, 3};
    vec::dense_vector<double> v(3, buf);
    const vec::dense_vector<double> w(4, 9.0);
    REQUIRE_THROWS_AS(v = w, std::domain_error);
    REQUIRE_THROWS_AS(v = w + w, std::domain_error);
    REQUIRE(v.size() == 3);
    REQUIRE(v.data() == buf);
    REQUIRE(buf[0] == 1.0);
    v.change_dim(4);   // an explicit resize still detaches
    REQUIRE(v.owns_memory());
    REQUIRE(v.size() == 4);
}

TEST_CASE("dense_vector: owning vectors still resize on assignment", "[vec][dense_vector][829]") {
    vec::dense_vector<double> v(2, 1.0);
    const vec::dense_vector<double> w(5, 2.0);
    v = w;
    REQUIRE(v.size() == 5);
    REQUIRE(v(4) == 2.0);
    v = vec::dense_vector<double>(1, 3.0);
    REQUIRE(v.size() == 1);
    v = w + w;
    REQUIRE(v.size() == 5);
    REQUIRE(v(0) == 4.0);
}

TEST_CASE("dense_vector: move leaves the source empty, self-move is a no-op", "[vec][dense_vector][829]") {
    vec::dense_vector<double> a{1.0, 2.0};
    vec::dense_vector<double> b(std::move(a));
    REQUIRE(b.size() == 2);
    REQUIRE(a.size() == 0);
    auto& rb = b;
    b = std::move(rb);
    REQUIRE(b.size() == 2);
    REQUIRE(b(1) == 2.0);
}

TEST_CASE("dense_vector: swap of external and owning vectors", "[vec][dense_vector][829]") {
    double buf[2] = {1, 2};
    vec::dense_vector<double> e(2, buf);
    vec::dense_vector<double> o{7.0, 8.0, 9.0};
    e.swap(o);
    REQUIRE(e.size() == 3);
    REQUIRE(e.owns_memory());
    REQUIRE(o.size() == 2);
    REQUIRE(o.data() == buf);
    std::swap(e, o);   // through the moves: the external vector is re-pointed back
    REQUIRE(e.data() == buf);
    REQUIRE(e.size() == 2);
    REQUIRE(o.size() == 3);
    REQUIRE(o(2) == 9.0);
    REQUIRE(buf[0] == 1.0);
}

TEST_CASE("dense_vector: fixed-size heap vectors always have storage", "[vec][dense_vector][829]") {
    fixed_heap_vec a;
    REQUIRE(a.size() == 3);
    REQUIRE(a.data() != nullptr);   // null before #829
    a(0) = 1.0;
    fixed_heap_vec b(std::move(a));
    REQUIRE(b(0) == 1.0);
    REQUIRE(a.data() != nullptr);
    a(2) = 5.0;                     // dereferenced null before #829
    fixed_heap_vec c;
    c = std::move(a);
    REQUIRE(c(2) == 5.0);
    REQUIRE(a.data() != nullptr);
    c.swap(b);
    REQUIRE(c(0) == 1.0);
    REQUIRE(b(2) == 5.0);
    REQUIRE_THROWS_AS(c.checked_change_dim(4), std::domain_error);
    REQUIRE_NOTHROW(c.checked_change_dim(3));
}

TEST_CASE("dense_vector: checked_change_dim on a dynamic vector", "[vec][dense_vector][829]") {
    vec::dense_vector<double> v(2, 1.0);
    v.checked_change_dim(6);
    REQUIRE(v.size() == 6);
}

TEST_CASE("dense_vector(n, 0) and fill(v, 0) take 0 as a value", "[vec][dense_vector][829]") {
    vec::dense_vector<double> v(4, 0);   // ambiguous before #829
    REQUIRE(v.size() == 4);
    REQUIRE(v.owns_memory());
    REQUIRE(v(3) == 0.0);
    vec::fill(v, 2);                     // did not compile before #829
    REQUIRE(v(0) == 2.0);
    STATIC_REQUIRE(!std::is_constructible_v<vec::dense_vector<double>, std::size_t, float*>);
}

TEST_CASE("dense_vector compound assignment with vectors and expressions", "[vec][dense_vector][829]") {
    vec::dense_vector<double> v{1.0, 2.0, 3.0};
    const vec::dense_vector<double> w{10.0, 20.0, 30.0};
    v += w;
    REQUIRE(v(2) == 33.0);
    v -= w;
    REQUIRE(v(2) == 3.0);
    v += w + w;
    REQUIRE(v(0) == 21.0);
    v -= 2.0 * w;
    REQUIRE(v(0) == 1.0);
    v *= 3.0;
    REQUIRE(v(1) == 6.0);
    v /= 2.0;
    REQUIRE(v(1) == 3.0);
}
