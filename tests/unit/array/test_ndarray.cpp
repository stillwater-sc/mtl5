#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <mtl/array/ndarray.hpp>
#include <array>
#include <complex>
#include <iterator>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>
#include <mtl/array/interop.hpp>
#include <mtl/config.hpp>
#include <mtl/array/shape.hpp>
#include <mtl/array/slice.hpp>
#include <mtl/array/broadcast.hpp>
#include <mtl/array/operations.hpp>
#include <mtl/concepts/ndarray.hpp>
#include <mtl/tag/sparsity.hpp>
#include <mtl/traits/category.hpp>

using namespace mtl;
using namespace mtl::array;

// -- Shape tests ----------------------------------------------------

TEST_CASE("shape construction and access", "[array][shape]") {
    shape<3> sh{4, 5, 6};
    REQUIRE(sh[0] == 4);
    REQUIRE(sh[1] == 5);
    REQUIRE(sh[2] == 6);
    REQUIRE(sh.total_size() == 120);
}

TEST_CASE("shape equality", "[array][shape]") {
    shape<2> a{3, 4};
    shape<2> b{3, 4};
    shape<2> c{4, 3};
    REQUIRE(a == b);
    REQUIRE(a != c);
}

// -- Stride tests ---------------------------------------------------

TEST_CASE("c-order strides", "[array][strides]") {
    shape<3> sh{2, 3, 4};
    auto strides = c_order_strides(sh);
    // Last dim stride=1, middle=4, first=12
    REQUIRE(strides[0] == 12);
    REQUIRE(strides[1] == 4);
    REQUIRE(strides[2] == 1);
}

TEST_CASE("f-order strides", "[array][strides]") {
    shape<3> sh{2, 3, 4};
    auto strides = f_order_strides(sh);
    // First dim stride=1, middle=2, last=6
    REQUIRE(strides[0] == 1);
    REQUIRE(strides[1] == 2);
    REQUIRE(strides[2] == 6);
}

// -- Construction tests ---------------------------------------------

TEST_CASE("ndarray default construction", "[array][ctor]") {
    ndarray<double, 2> a;
    REQUIRE(a.size() == 0);
    REQUIRE(a.empty());
}

TEST_CASE("ndarray shape construction", "[array][ctor]") {
    ndarray<double, 3> a(shape<3>{4, 5, 6});
    REQUIRE(a.size() == 120);
    REQUIRE(a.extent(0) == 4);
    REQUIRE(a.extent(1) == 5);
    REQUIRE(a.extent(2) == 6);
    REQUIRE(!a.empty());
}

TEST_CASE("ndarray initializer-list construction", "[array][ctor]") {
    ndarray<int, 2> a({3, 4});
    REQUIRE(a.size() == 12);
    REQUIRE(a.extent(0) == 3);
    REQUIRE(a.extent(1) == 4);
}

TEST_CASE("ndarray fill construction", "[array][ctor]") {
    ndarray<int, 2> a(shape<2>{2, 3}, 42);
    REQUIRE(a(0, 0) == 42);
    REQUIRE(a(1, 2) == 42);
}

// -- Concept satisfaction -------------------------------------------

TEST_CASE("ndarray satisfies NdArray concept", "[array][concept]") {
    STATIC_REQUIRE(Collection<ndarray<double, 2>>);
    STATIC_REQUIRE(MutableCollection<ndarray<double, 2>>);
    STATIC_REQUIRE(NdArray<ndarray<double, 2>>);
    STATIC_REQUIRE(MutableNdArray<ndarray<double, 2>>);
}

TEST_CASE("ndarray has dense category trait", "[array][trait]") {
    using cat = traits::category_t<ndarray<double, 3>>;
    STATIC_REQUIRE(std::is_same_v<cat, tag::dense>);
}

// -- Element access -------------------------------------------------

TEST_CASE("ndarray element access via operator()", "[array][access]") {
    ndarray<int, 2> a({3, 4});
    a(0, 0) = 1;
    a(1, 2) = 42;
    a(2, 3) = 99;
    REQUIRE(a(0, 0) == 1);
    REQUIRE(a(1, 2) == 42);
    REQUIRE(a(2, 3) == 99);
}

TEST_CASE("ndarray element access via array index", "[array][access]") {
    ndarray<int, 3> a({2, 3, 4});
    a[{1, 2, 3}] = 7;
    REQUIRE(a[{1, 2, 3}] == 7);
    REQUIRE(a(1, 2, 3) == 7);
}

TEST_CASE("ndarray C-order memory layout", "[array][layout]") {
    // In C-order, last index varies fastest
    ndarray<int, 2> a({2, 3});
    int counter = 0;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            a(i, j) = counter++;

    // Flat memory should be 0,1,2,3,4,5
    for (int k = 0; k < 6; ++k) {
        REQUIRE(a.flat(k) == k);
    }
}

TEST_CASE("ndarray F-order memory layout", "[array][layout]") {
    // In F-order, first index varies fastest
    ndarray<int, 2, f_order> a(shape<2>{2, 3});
    int counter = 0;
    for (std::size_t j = 0; j < 3; ++j)
        for (std::size_t i = 0; i < 2; ++i)
            a(i, j) = counter++;

    // Flat memory should be 0,1,2,3,4,5 (column-major: i varies first)
    for (int k = 0; k < 6; ++k) {
        REQUIRE(a.flat(k) == k);
    }
}

// -- View tests -----------------------------------------------------

TEST_CASE("ndarray view shares memory", "[array][view]") {
    ndarray<int, 2> a({3, 4}, 0);
    auto v = a.view();
    REQUIRE(v.is_view());
    REQUIRE(!a.is_view());
    REQUIRE(v.data() == a.data());

    // Mutation through view visible in original
    v(1, 2) = 99;
    REQUIRE(a(1, 2) == 99);
}

TEST_CASE("ndarray external pointer constructor", "[array][view]") {
    int data[] = {1, 2, 3, 4, 5, 6};
    ndarray<int, 2> a(data, shape<2>{2, 3});
    REQUIRE(a.is_view());
    REQUIRE(a(0, 0) == 1);
    REQUIRE(a(1, 2) == 6);

    // Modification goes through to original array
    a(0, 1) = 42;
    REQUIRE(data[1] == 42);
}

// -- Reshape and transpose ------------------------------------------

TEST_CASE("ndarray reshape", "[array][reshape]") {
    ndarray<int, 2> a({3, 4});
    for (int i = 0; i < 12; ++i) a.flat(i) = i;

    auto b = a.reshape(shape<1>{12});
    REQUIRE(b.size() == 12);
    REQUIRE(b(0) == 0);
    REQUIRE(b(11) == 11);
    REQUIRE(b.is_view());
}

TEST_CASE("ndarray transpose", "[array][transpose]") {
    ndarray<int, 2> a({2, 3});
    a(0, 0) = 1; a(0, 1) = 2; a(0, 2) = 3;
    a(1, 0) = 4; a(1, 1) = 5; a(1, 2) = 6;

    auto t = a.transpose();
    REQUIRE(t.extent(0) == 3);
    REQUIRE(t.extent(1) == 2);
    REQUIRE(t(0, 0) == 1);
    REQUIRE(t(1, 0) == 2);
    REQUIRE(t(2, 0) == 3);
    REQUIRE(t(0, 1) == 4);
    REQUIRE(t(2, 1) == 6);
}

// -- Slicing tests --------------------------------------------------

TEST_CASE("slice with all keeps full dimension", "[array][slice]") {
    ndarray<int, 2> a({3, 4}, 0);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            a(i, j) = static_cast<int>(i * 4 + j);

    auto row = slice(a, 1, all);  // row 1 -> 1D view
    REQUIRE(row.size() == 4);
    REQUIRE(row(0) == 4);
    REQUIRE(row(1) == 5);
    REQUIRE(row(2) == 6);
    REQUIRE(row(3) == 7);
}

TEST_CASE("slice with integer reduces rank", "[array][slice]") {
    ndarray<int, 3> a({2, 3, 4}, 0);
    a(1, 2, 3) = 42;

    auto s = slice(a, 1, all, all);  // fix first dim -> 2D
    REQUIRE(s.extent(0) == 3);
    REQUIRE(s.extent(1) == 4);
    REQUIRE(s(2, 3) == 42);
}

TEST_CASE("slice with range", "[array][slice]") {
    ndarray<int, 2> a({4, 6}, 0);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 6; ++j)
            a(i, j) = static_cast<int>(i * 6 + j);

    auto s = slice(a, range(1, 3), all);  // rows 1..2
    REQUIRE(s.extent(0) == 2);
    REQUIRE(s.extent(1) == 6);
    REQUIRE(s(0, 0) == 6);   // original row 1
    REQUIRE(s(1, 0) == 12);  // original row 2
}

TEST_CASE("slice mutation visible in original", "[array][slice]") {
    ndarray<int, 2> a({3, 4}, 0);
    auto row = slice(a, 1, all);
    row(2) = 99;
    REQUIRE(a(1, 2) == 99);
}

// -- Broadcasting tests ---------------------------------------------

TEST_CASE("broadcast shape computation", "[array][broadcast]") {
    shape<3> a{4, 1, 6};
    shape<3> b{1, 5, 6};
    shape<3> result;
    REQUIRE(broadcast_shape(a, b, result));
    REQUIRE(result[0] == 4);
    REQUIRE(result[1] == 5);
    REQUIRE(result[2] == 6);
}

TEST_CASE("broadcast shape incompatible", "[array][broadcast]") {
    shape<2> a{3, 4};
    shape<2> b{2, 4};
    shape<2> result;
    REQUIRE(!broadcast_shape(a, b, result));
}

TEST_CASE("element-wise addition same shape", "[array][broadcast]") {
    ndarray<int, 2> a({2, 3}, 1);
    ndarray<int, 2> b({2, 3}, 2);
    ndarray<int, 2> c = a + b;
    REQUIRE(c(0, 0) == 3);
    REQUIRE(c(1, 2) == 3);
}

TEST_CASE("element-wise multiplication with broadcasting", "[array][broadcast]") {
    // a is (3, 1), b is (1, 4) -> result is (3, 4)
    ndarray<int, 2> a(shape<2>{3, 1});
    ndarray<int, 2> b(shape<2>{1, 4});

    for (std::size_t i = 0; i < 3; ++i) a(i, 0) = static_cast<int>(i + 1);
    for (std::size_t j = 0; j < 4; ++j) b(0, j) = static_cast<int>(j + 1);

    ndarray<int, 2> c = a * b;
    REQUIRE(c.extent(0) == 3);
    REQUIRE(c.extent(1) == 4);
    // c(i,j) = (i+1) * (j+1)
    REQUIRE(c(0, 0) == 1);
    REQUIRE(c(0, 3) == 4);
    REQUIRE(c(2, 3) == 12);
}

// -- Reduction tests ------------------------------------------------

TEST_CASE("sum all elements", "[array][reduction]") {
    ndarray<int, 2> a({2, 3}, 5);
    REQUIRE(sum(a) == 30);
}

TEST_CASE("prod all elements", "[array][reduction]") {
    ndarray<int, 1> a(shape<1>{4});
    a(0) = 1; a(1) = 2; a(2) = 3; a(3) = 4;
    REQUIRE(prod(a) == 24);
}

TEST_CASE("min and max", "[array][reduction]") {
    ndarray<int, 1> a(shape<1>{5});
    a(0) = 3; a(1) = 1; a(2) = 4; a(3) = 1; a(4) = 5;
    REQUIRE(min(a) == 1);
    REQUIRE(max(a) == 5);
}

TEST_CASE("mean", "[array][reduction]") {
    ndarray<double, 1> a(shape<1>{4});
    a(0) = 1.0; a(1) = 2.0; a(2) = 3.0; a(3) = 4.0;
    REQUIRE(mean(a) == Catch::Approx(2.5));
}

TEST_CASE("sum along axis", "[array][reduction]") {
    ndarray<int, 2> a({2, 3});
    a(0, 0) = 1; a(0, 1) = 2; a(0, 2) = 3;
    a(1, 0) = 4; a(1, 1) = 5; a(1, 2) = 6;

    // Sum along axis 0 -> shape (3,)
    auto s0 = sum_axis(a, 0);
    REQUIRE(s0.extent(0) == 3);
    REQUIRE(s0(0) == 5);   // 1+4
    REQUIRE(s0(1) == 7);   // 2+5
    REQUIRE(s0(2) == 9);   // 3+6

    // Sum along axis 1 -> shape (2,)
    auto s1 = sum_axis(a, 1);
    REQUIRE(s1.extent(0) == 2);
    REQUIRE(s1(0) == 6);   // 1+2+3
    REQUIRE(s1(1) == 15);  // 4+5+6
}

TEST_CASE("mean along axis", "[array][reduction]") {
    ndarray<double, 2> a({2, 3});
    a(0, 0) = 1; a(0, 1) = 2; a(0, 2) = 3;
    a(1, 0) = 4; a(1, 1) = 5; a(1, 2) = 6;

    auto m = mean_axis(a, 0);
    REQUIRE(m(0) == Catch::Approx(2.5));
    REQUIRE(m(1) == Catch::Approx(3.5));
    REQUIRE(m(2) == Catch::Approx(4.5));
}

// -- Fill and compound assignment -----------------------------------

TEST_CASE("ndarray fill", "[array][fill]") {
    ndarray<int, 2> a({3, 4});
    a.fill(7);
    REQUIRE(a(0, 0) == 7);
    REQUIRE(a(2, 3) == 7);
}

TEST_CASE("ndarray compound assignment", "[array][ops]") {
    ndarray<int, 1> a(shape<1>{3});
    a(0) = 1; a(1) = 2; a(2) = 3;

    a += 10;
    REQUIRE(a(0) == 11);
    REQUIRE(a(2) == 13);

    a *= 2;
    REQUIRE(a(0) == 22);
    REQUIRE(a(2) == 26);
}

// -- 3D array test --------------------------------------------------

TEST_CASE("3D ndarray construction and access", "[array][3d]") {
    ndarray<double, 3> vol({4, 5, 6});
    REQUIRE(vol.size() == 120);
    REQUIRE(vol.is_contiguous());

    vol(2, 3, 4) = 3.14;
    REQUIRE(vol(2, 3, 4) == Catch::Approx(3.14));
}

// -- Contiguity -----------------------------------------------------

TEST_CASE("contiguity check", "[array][contiguous]") {
    ndarray<int, 2> a({3, 4});
    REQUIRE(a.is_contiguous());

    // A transpose of a C-order 3x4 is F-order contiguous (4x3 with strides {1, 4})
    auto t = a.transpose();
    REQUIRE(t.is_contiguous());  // contiguous in F-order

    // A sliced view with step > 1 is not contiguous
    ndarray<int, 3> vol({4, 5, 6});
    auto s = slice(vol, range(0, 4, 2), all, all);  // stride 2 on first dim
    REQUIRE(!s.is_contiguous());
}

// -- Copy semantics -------------------------------------------------

TEST_CASE("ndarray copy is deep", "[array][copy]") {
    ndarray<int, 1> a(shape<1>{3});
    a(0) = 1; a(1) = 2; a(2) = 3;

    ndarray<int, 1> b = a;
    b(0) = 99;
    REQUIRE(a(0) == 1);  // original unchanged
    REQUIRE(b(0) == 99);
}

// ---------------------------------------------------------------------------
// #359: reshape and flatten returned MEMORY order, not LOGICAL order, for an
// F-contiguous array -- which is what any transpose of a C-order array is.
//
// One root cause: is_contiguous() answers "contiguous in EITHER C or F order",
// which is the right question for order-INdependent work (fill, BLAS dispatch,
// a commutative reduction) and the wrong one for anything that walks memory as
// though it were logical order. Two callers used it as though it meant
// "memory order == logical order", which only holds for the C-contiguous case.
//
// NumPy is the reference: a.T.reshape(6) and a.T.flatten() are [1,4,2,5,3,6].
// ---------------------------------------------------------------------------

TEST_CASE("ndarray contiguity predicates distinguish C, F and own order (#359)",
          "[array][ndarray][regression]") {
    ndarray<double, 2> a(shape<2>{2, 3});
    double v = 1.0;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j) a(i, j) = v++;

    // A freshly built C-order array is contiguous in every sense.
    REQUIRE(a.is_contiguous());
    REQUIRE(a.is_c_contiguous());
    REQUIRE(a.is_contiguous_in_own_order());

    // Its transpose is F-contiguous but NOT C-contiguous. is_contiguous() is
    // still true -- that is the trap, and it is correct for what it means.
    auto t = a.transpose();
    REQUIRE(t.is_contiguous());
    REQUIRE_FALSE(t.is_c_contiguous());
    REQUIRE(t.is_f_contiguous());
    REQUIRE_FALSE(t.is_contiguous_in_own_order());   // Order is still c_order

    // Rank 1 has no distinction: the C and F strides coincide.
    ndarray<double, 1> u(shape<1>{4});
    REQUIRE(u.is_c_contiguous());
    REQUIRE(u.is_f_contiguous());
}

TEST_CASE("ndarray reshape refuses an F-contiguous c_order array (#359)",
          "[array][ndarray][regression]") {
    ndarray<double, 2> a(shape<2>{2, 3});
    double v = 1.0;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j) a(i, j) = v++;

    // Previously this returned a view that read the memory in C order and
    // silently produced [1 2 3 4 5 6] instead of the logical [1 4 2 5 3 6].
    auto t = a.transpose();
    REQUIRE_THROWS_AS(t.reshape(shape<1>{6}), std::runtime_error);

    // The C-contiguous case must still succeed, and still be a view.
    auto r = a.reshape(shape<1>{6});
    REQUIRE(r.size() == 6);
    for (std::size_t i = 0; i < 6; ++i)
        REQUIRE(r(i) == Catch::Approx(static_cast<double>(i + 1)));
    r(0) = 99.0;
    REQUIRE(a(0, 0) == Catch::Approx(99.0));   // aliases, not a copy
}

TEST_CASE("ndarray order-independent operations are unaffected (#359)",
          "[array][ndarray][regression]") {
    // Stated explicitly so the fix stays narrow: these visit every element and
    // do not care in what order, so they were correct before and after.
    ndarray<double, 2> a(shape<2>{2, 3});
    double v = 1.0;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j) a(i, j) = v++;

    auto t = a.transpose();
    REQUIRE(sum(t) == Catch::Approx(21.0));

    auto s1 = sum_axis(t, 1);
    REQUIRE(s1(0) == Catch::Approx(5.0));
    REQUIRE(s1(1) == Catch::Approx(7.0));
    REQUIRE(s1(2) == Catch::Approx(9.0));

    // Compound assignment through a transposed view must still reach every
    // element -- it now takes the strided path rather than the flat one.
    auto t2 = a.transpose();
    t2 += 10.0;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            REQUIRE(a(i, j) == Catch::Approx(static_cast<double>(i * 3 + j + 11)));
}

// -- Copying views and assigning to them (#829) ---------------------------

TEST_CASE("copying a strided view copies its elements", "[array][view][829]") {
    ndarray<int, 2> a({3, 4}, 0);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            a(i, j) = static_cast<int>(i * 4 + j);
    auto col = slice(a, all, 2);   // stride 4: reads elements 2, 6, 10
    const ndarray<int, 1> c(col);
    REQUIRE(!c.is_view());
    REQUIRE(c.is_contiguous());
    REQUIRE(c(0) == 2);
    REQUIRE(c(1) == 6);
    REQUIRE(c(2) == 10);   // read past the 3-element copy before #829

    const auto tv = a.transpose();   // an F-contiguous view
    const ndarray<int, 2> t(tv);     // a copy, not a move: it owns C-order storage
    REQUIRE(!t.is_view());
    REQUIRE(t.is_c_contiguous());
    REQUIRE(t(3, 2) == a(2, 3));
}

TEST_CASE("assigning to a view writes the viewed elements", "[array][view][829]") {
    ndarray<int, 2> a({3, 4}, 0);
    ndarray<int, 1> src({3}, 0);
    for (std::size_t i = 0; i < 3; ++i) src(i) = static_cast<int>(100 + i);

    auto col = slice(a, all, 1);
    col = src;   // detached before #829; a raw write would hit a(0, 1..3)
    REQUIRE(col.is_view());
    REQUIRE(a(0, 1) == 100);
    REQUIRE(a(1, 1) == 101);
    REQUIRE(a(2, 1) == 102);
    REQUIRE(a(0, 2) == 0);
    REQUIRE(a(0, 0) == 0);

    auto row = slice(a, 2, all);
    row = ndarray<int, 1>({4}, 7);   // move assignment
    REQUIRE(a(2, 0) == 7);
    REQUIRE(a(2, 3) == 7);
    REQUIRE(a(1, 1) == 101);

    const ndarray<int, 1> wrong({5}, 1);
    REQUIRE_THROWS_AS(row = wrong, std::domain_error);
    REQUIRE(a(2, 0) == 7);
}

TEST_CASE("assigning an overlapping view to a view", "[array][view][829]") {
    ndarray<int, 2> a({2, 2}, 0);
    a(0, 0) = 1; a(0, 1) = 2; a(1, 0) = 3; a(1, 1) = 4;
    auto v = a.view();
    v = a.transpose();   // reads all of the transpose before writing
    REQUIRE(a(0, 1) == 3);
    REQUIRE(a(1, 0) == 2);
    REQUIRE(a(1, 1) == 4);
}

TEST_CASE("assigning to an owning ndarray replaces it", "[array][829]") {
    ndarray<int, 2> a({2, 2}, 1);
    ndarray<int, 2> b({3, 5}, 2);
    a = b;
    REQUIRE(!a.is_view());
    REQUIRE(a.extent(1) == 5);
    REQUIRE(a(2, 4) == 2);
    a = ndarray<int, 2>({1, 1}, 9);
    REQUIRE(a.size() == 1);
    REQUIRE(a(0, 0) == 9);
}

TEST_CASE("assigning an owning array to a view of it", "[array][view][829]") {
    ndarray<int, 2> a({2, 2}, 0);
    a(0, 0) = 1; a(0, 1) = 2; a(1, 0) = 3; a(1, 1) = 4;
    auto v = a.transpose();
    v = a;   // v(i, j) = a(i, j) is a(j, i) = a(i, j): a becomes its transpose
    REQUIRE(a(0, 1) == 3);
    REQUIRE(a(1, 0) == 2);
}

TEST_CASE("move-assigning a view of an owning array to it", "[array][view][829]") {
    ndarray<int, 2> a({2, 3}, 0);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            a(i, j) = static_cast<int>(10 * i + j);
    a = a.transpose();   // released the storage the view points into before #829's review
    REQUIRE(!a.is_view());
    REQUIRE(a.extent(0) == 3);
    REQUIRE(a(2, 1) == 12);
    REQUIRE(a(0, 1) == 10);
}

TEST_CASE("a moved-from ndarray is empty", "[array][829]") {
    ndarray<int, 2> a({2, 2}, 1);
    ndarray<int, 2> b(std::move(a));
    REQUIRE(b(1, 1) == 1);
    REQUIRE(a.size() == 0);
    const ndarray<int, 2> c(a);   // read a null pointer through the old shape
    REQUIRE(c.size() == 0);
}

// ---------------------------------------------------------------------------
// #856: the order of iteration, checked preconditions, reductions, broadcast
// lifetimes, and move-assignment of views.

namespace {

template <typename A>
std::vector<double> elements_in_order(const A& a) {
    std::vector<double> v;
    a.for_each_element([&](const auto& x) { v.push_back(double(x)); });
    return v;
}

ndarray<double, 2, f_order> f_matrix(std::size_t r, std::size_t c) {
    ndarray<double, 2, f_order> a(shape<2>{r, c});
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j) a(i, j) = double(10 * i + j);
    return a;
}

struct ordered_t {   // totally ordered, no numeric_limits specialization
    int v;
    auto operator<=>(const ordered_t&) const = default;
};

} // namespace

TEST_CASE("an f_order array iterates in its own order, contiguous or not", "[array][856]") {
    auto f = f_matrix(2, 3);
    // Contiguous: first index fastest, as before.
    REQUIRE(elements_in_order(f) == std::vector<double>{0, 10, 1, 11, 2, 12});
    // A strided view of it iterates the same way. It took C order (0 2 10 12),
    // so flatten and reduce depended on whether the array was contiguous.
    auto v = slice(f, all, range(0, 3, 2));
    REQUIRE_FALSE(v.is_contiguous_in_own_order());
    REQUIRE(elements_in_order(v) == std::vector<double>{0, 10, 2, 12});
    const auto fl = flatten(v);
    REQUIRE((fl(0) == 0 && fl(1) == 10 && fl(2) == 2 && fl(3) == 12));
    // A non-commutative reduce agrees with the copy's.
    auto digits = [](double acc, double x) { return acc * 100 + x; };
    const ndarray<double, 2, f_order> copy(v);
    REQUIRE(copy.is_contiguous_in_own_order());
    REQUIRE(reduce(v, 0.0, digits) == reduce(copy, 0.0, digits));
    // flatten agrees with reshape, which reads in the array's own order (#359).
    const auto r = f.reshape(shape<1>{6});
    const auto ff = flatten(f);
    for (std::size_t i = 0; i < 6; ++i) REQUIRE(ff(i) == r(i));
    // iterate_indices visits the first index fastest too.
    std::vector<std::array<std::size_t, 2>> seen;
    f.iterate_indices([&](const std::array<std::size_t, 2>& idx) { seen.push_back(idx); });
    REQUIRE((seen[0] == std::array<std::size_t, 2>{0, 0} && seen[1] == std::array<std::size_t, 2>{1, 0}));
    // c_order arrays, and their transposes, are unchanged: last index fastest.
    ndarray<double, 2> c(shape<2>{2, 3});
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j) c(i, j) = double(10 * i + j);
    REQUIRE(elements_in_order(c) == std::vector<double>{0, 1, 2, 10, 11, 12});
    REQUIRE(elements_in_order(c.transpose()) == std::vector<double>{0, 10, 1, 11, 2, 12});
}

TEST_CASE("f_order content through sum_axis, transform, slice and broadcasting", "[array][856]") {
    const auto f = f_matrix(2, 3);
    const auto s0 = sum_axis(f, 0);
    REQUIRE((s0(0) == 10 && s0(1) == 12 && s0(2) == 14));
    const auto s1 = sum_axis(f, 1);
    REQUIRE((s1(0) == 3 && s1(1) == 33));
    const auto t = transform(f, [](double x) { return x + 1; });
    REQUIRE((t(1, 2) == 13 && t(0, 1) == 2));
    const auto row = slice(f, 1, all);
    REQUIRE((row(0) == 10 && row(2) == 12));
    const ndarray<double, 2> g = f + f;
    REQUIRE((g(1, 2) == 24 && g(0, 1) == 2));
    REQUIRE_THROWS_AS(sum_axis(f, 2), std::out_of_range);
}

TEST_CASE("reshape and shape lists check their sizes in every build", "[array][856]") {
    ndarray<double, 1> a(shape<1>{3});
    // An assert: under NDEBUG a size-100 view of a 3-element buffer.
    REQUIRE_THROWS_AS(a.reshape(shape<1>{100}), std::invalid_argument);
    REQUIRE_THROWS_AS(a.reshape(shape<2>{2, 2}), std::invalid_argument);
    REQUIRE(a.reshape(shape<2>{1, 3}).size() == 3);
    REQUIRE_THROWS_AS((shape<2>{1, 2, 3}), std::invalid_argument);
    REQUIRE_THROWS_AS((shape<3>{4, 5}), std::invalid_argument);
    REQUIRE_THROWS_AS((ndarray<double, 2>{2, 3, 4}), std::invalid_argument);
}

TEST_CASE("slice clamps ranges and checks steps and indices", "[array][856]") {
    ndarray<double, 2> a(shape<2>{3, 4});
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j) a(i, j) = double(10 * i + j);
    // range(2, 10) of 3 rows is row 2 (it had extent 8, past the buffer).
    auto s = slice(a, range(2, 10), all);
    REQUIRE(s.extent(0) == 1);
    REQUIRE(s(0, 3) == 23);
    REQUIRE(slice(a, range(5, 9), all).extent(0) == 0);   // starts past the end: empty
    REQUIRE(slice(a, range(3, 3), all).empty());
    auto st = slice(a, all, range(0, 4, 3));               // a step > 1
    REQUIRE((st.extent(1) == 2 && st(1, 1) == 13));
    REQUIRE_THROWS_AS(slice(a, range(0, 3, 0), all), std::invalid_argument);   // divided by zero
    REQUIRE_THROWS_AS(slice(a, 3, all), std::out_of_range);
    REQUIRE_THROWS_AS(slice(a, all, 4), std::out_of_range);
    REQUIRE_THROWS_AS(slice(a, -1, all), std::out_of_range);
    const auto& ca = a;
    REQUIRE_THROWS_AS(slice(ca, all, 4), std::out_of_range);
    REQUIRE(slice(ca, range(1, 99), 2).extent(0) == 2);

    // Rank 0: slice() takes no arguments (it did not compile).
    ndarray<double, 0> z(shape<0>{});
    z() = 7.0;
    auto zv = slice(z);
    REQUIRE((zv.size() == 1 && zv() == 7.0));
    std::size_t visits = 0;
    z.iterate_indices([&](const std::array<std::size_t, 0>&) { ++visits; });
    REQUIRE(visits == 1);
    REQUIRE(sum(z) == 7.0);
}

TEST_CASE("reductions: empty arrays, identities, custom types, const views", "[array][856]") {
    const ndarray<double, 1> e(shape<1>{0});
    REQUIRE(sum(e) == 0.0);
    REQUIRE(prod(e) == 1.0);
    // min and max returned DBL_MAX and -DBL_MAX; mean was 0/0.
    REQUIRE_THROWS_AS(min(e), std::invalid_argument);
    REQUIRE_THROWS_AS(max(e), std::invalid_argument);
    REQUIRE_THROWS_AS(mean(e), std::invalid_argument);
    const ndarray<int, 2> ez(shape<2>{3, 0});
    REQUIRE_THROWS_AS(mean_axis(ez, 1), std::invalid_argument);   // an integer division by zero
    const auto zs = sum_axis(ez, 1);
    REQUIRE((zs.size() == 3 && zs(1) == 0));

    // A type without numeric_limits: min was seeded with ordered_t{} = 0.
    ndarray<ordered_t, 1> o(shape<1>{3});
    o(0) = {5}; o(1) = {3}; o(2) = {9};
    REQUIRE(min(o).v == 3);
    REQUIRE(max(o).v == 9);
    ndarray<double, 1> neg(shape<1>{2}, -4.0);
    REQUIRE(max(neg) == -4.0);

    // A size-1 axis.
    ndarray<double, 2> col(shape<2>{3, 1});
    for (std::size_t i = 0; i < 3; ++i) col(i, 0) = double(i + 1);
    const auto s = sum_axis(col, 1);
    REQUIRE((s(0) == 1 && s(2) == 3));
    REQUIRE(mean_axis(col, 0)(0) == 2.0);

    // A read-only array: these did not compile (the accumulator was const).
    double buf[4] = {4, 1, 3, 2};
    const ndarray<const double, 1> c(static_cast<const double*>(buf), shape<1>{4});
    REQUIRE(sum(c) == 10.0);
    REQUIRE(prod(c) == 24.0);
    REQUIRE(min(c) == 1.0);
    REQUIRE(max(c) == 4.0);
    REQUIRE(mean(c) == 2.5);
    STATIC_REQUIRE(std::is_same_v<decltype(sum(c)), double>);
}

TEST_CASE("reshape and transpose of a const array are read-only views", "[array][856]") {
    ndarray<double, 2> a(shape<2>{2, 3}, 1.0);
    const auto& ca = a;
    STATIC_REQUIRE(std::is_same_v<decltype(ca.transpose()), ndarray<const double, 2>>);
    STATIC_REQUIRE(std::is_same_v<decltype(ca.reshape(shape<1>{6})), ndarray<const double, 1>>);
    STATIC_REQUIRE(std::is_same_v<decltype(a.transpose()), ndarray<double, 2>>);
    const auto t = ca.transpose();
    REQUIRE((t.data() == a.data() && t(2, 1) == 1.0));
    a.transpose()(2, 1) = 5.0;   // a mutable array still gives a mutable view
    REQUIRE(a(1, 2) == 5.0);
}

TEST_CASE("broadcast operands that are temporaries are kept alive", "[array][856]") {
    auto make = [] { return ndarray<double, 2>(shape<2>{2, 2}, 3.0); };
    ndarray<double, 2> b(shape<2>{2, 2}, 1.0);
    // Both held a const reference to a dead temporary (ASan: use after scope).
    auto e1 = make() + b;
    const ndarray<double, 2> r1 = e1;
    REQUIRE((r1(0, 0) == 4.0 && r1(1, 1) == 4.0));
    ndarray<double, 2> a(shape<2>{2, 3});
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j) a(i, j) = double(i * 3 + j);
    ndarray<double, 2> row(shape<2>{1, 2}, 10.0);
    auto e2 = a.transpose() - row;   // a 3x2 view minus a broadcast row
    const ndarray<double, 2> r2 = e2;
    REQUIRE((r2(2, 1) == 5.0 - 10.0 && r2(0, 0) == -10.0));
    const ndarray<double, 2> q = a / ndarray<double, 2>(shape<2>{1, 1}, 2.0);
    REQUIRE(q(1, 2) == 2.5);
    REQUIRE_THROWS_AS((a + ndarray<double, 2>(shape<2>{3, 2})), std::invalid_argument);
    if constexpr (mtl::bounds_checking) {
        REQUIRE_THROWS_AS((e1[{2, 0}]), std::out_of_range);
    }
}

TEST_CASE("an owning array move-assigned a view copies it and stays owning", "[array][856]") {
    double buf[4] = {1, 2, 3, 4};
    ndarray<double, 2> own(shape<2>{3, 3}, 0.0);
    ndarray<double, 2> view(buf, shape<2>{2, 2});
    own = std::move(view);   // was a hand-over: own became a view of buf
    REQUIRE_FALSE(own.is_view());
    REQUIRE((own.extent(0) == 2 && own(1, 1) == 4.0));
    own(0, 0) = 99.0;
    REQUIRE(buf[0] == 1.0);
    REQUIRE(view.is_view());   // the source is a view still, of buf
    REQUIRE(view(0, 1) == 2.0);

    ndarray<double, 2> a(shape<2>{2, 3});
    for (std::size_t i = 0; i < 6; ++i) a.data()[i] = double(i);
    ndarray<double, 2> b(shape<2>{1, 1});
    b = a.transpose();       // a prvalue view of another array
    REQUIRE_FALSE(b.is_view());
    REQUIRE((b.extent(0) == 3 && b(2, 1) == 5.0));
    b(0, 0) = -1.0;
    REQUIRE(a(0, 0) == 0.0);
}

TEST_CASE("copying and assigning transpose, reshape and nested views", "[array][856]") {
    ndarray<double, 2> a(shape<2>{3, 4});
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j) a(i, j) = double(10 * i + j);
    const auto tv = a.transpose();
    const ndarray<double, 2> t(tv);        // a copy of the view, not the view itself
    REQUIRE((!t.is_view() && t(3, 2) == 23.0));
    const auto rv = a.reshape(shape<1>{12});
    const ndarray<double, 1> r(rv);
    REQUIRE((!r.is_view() && r(5) == 11.0));
    auto outer = slice(a, range(1, 3), all);
    auto inner = slice(outer, all, range(1, 4, 2));   // a view of a view
    const ndarray<double, 2> ic = inner;
    REQUIRE((ic.extent(0) == 2 && ic.extent(1) == 2 && ic(1, 1) == 23.0 && ic(0, 0) == 11.0));

    // Column assignment, interleaved and overlapping, lands the right values.
    auto c0 = slice(a, all, 0);
    c0 = slice(a, all, 1);
    REQUIRE((a(0, 0) == 1.0 && a(2, 0) == 21.0 && a(2, 1) == 21.0));
    auto left = slice(a, all, range(0, 3));
    left = slice(a, all, range(1, 4));    // shifted by one column: overlapping
    REQUIRE((a(0, 0) == 1.0 && a(0, 1) == 2.0 && a(0, 2) == 3.0 && a(2, 2) == 23.0));
}

TEST_CASE("untested array API: views, flat access, fill, contiguity", "[array][856]") {
    ndarray<double, 2> a(shape<2>{2, 3}, 2.0);
    auto v = a.view();
    v += 1.0; v -= 0.5; v /= 2.5;
    REQUIRE(a(1, 2) == 1.0);
    REQUIRE(!a.empty());
    REQUIRE(ndarray<double, 2>(shape<2>{0, 3}).empty());
    a.flat(4) = 7.0;
    REQUIRE(a(1, 1) == 7.0);
    REQUIRE(std::distance(a.begin(), a.end()) == 6);
    // fill through a strided view takes the walk, not the flat path.
    auto col = slice(a, all, 2);
    REQUIRE_FALSE(col.is_contiguous());
    col.fill(-3.0);
    REQUIRE((a(0, 2) == -3.0 && a(1, 2) == -3.0 && a(0, 1) == 1.0));
    // A size-1 dimension's stride does not matter: a 1 x 3 row of a 3 x 4 array
    // is contiguous in both orders, as in NumPy.
    ndarray<double, 2> b(shape<2>{3, 4});
    const auto onerow = slice(b, range(1, 2), range(0, 3));
    REQUIRE(onerow.is_c_contiguous());
    REQUIRE(onerow.is_f_contiguous());
    REQUIRE(is_contiguous_f(shape<2>{1, 3}, std::array<std::size_t, 2>{99, 1}));
    REQUIRE_FALSE(is_contiguous_f(shape<2>{2, 3}, std::array<std::size_t, 2>{3, 1}));
    // Broadcast strides: an extent-1 axis that stays extent 1 keeps its stride.
    REQUIRE((broadcast_strides(shape<2>{1, 3}, std::array<std::size_t, 2>{3, 1}, shape<2>{1, 3})
             == std::array<std::size_t, 2>{3, 1}));
    REQUIRE((broadcast_strides(shape<2>{1, 3}, std::array<std::size_t, 2>{3, 1}, shape<2>{4, 3})
             == std::array<std::size_t, 2>{0, 1}));
}

TEST_CASE("ndarray of std::complex", "[array][856]") {
    using cd = std::complex<double>;
    ndarray<cd, 1> z(shape<1>{3});
    z(0) = {1, 1}; z(1) = {2, -1}; z(2) = {0, 3};
    REQUIRE(sum(z) == cd(3, 3));
    REQUIRE(prod(z) == cd(1, 1) * cd(2, -1) * cd(0, 3));
    REQUIRE(mean(z) == cd(1, 1));
    const auto w = transform(z, [](cd x) { return std::conj(x); });
    REQUIRE(w(2) == cd(0, -3));
}
