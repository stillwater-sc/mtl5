#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <mtl/array/ndarray.hpp>
#include <mtl/array/shape.hpp>
#include <mtl/array/slice.hpp>
#include <mtl/array/broadcast.hpp>
#include <mtl/array/operations.hpp>
#include <mtl/concepts/ndarray.hpp>
#include <mtl/tag/sparsity.hpp>
#include <mtl/traits/category.hpp>

using namespace mtl;
using namespace mtl::array;

// ── Shape tests ────────────────────────────────────────────────────

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

// ── Stride tests ───────────────────────────────────────────────────

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

// ── Construction tests ─────────────────────────────────────────────

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

// ── Concept satisfaction ───────────────────────────────────────────

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

// ── Element access ─────────────────────────────────────────────────

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

// ── View tests ─────────────────────────────────────────────────────

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

// ── Reshape and transpose ──────────────────────────────────────────

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

// ── Slicing tests ──────────────────────────────────────────────────

TEST_CASE("slice with all keeps full dimension", "[array][slice]") {
    ndarray<int, 2> a({3, 4}, 0);
    for (std::size_t i = 0; i < 3; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            a(i, j) = static_cast<int>(i * 4 + j);

    auto row = slice(a, 1, all);  // row 1 → 1D view
    REQUIRE(row.size() == 4);
    REQUIRE(row(0) == 4);
    REQUIRE(row(1) == 5);
    REQUIRE(row(2) == 6);
    REQUIRE(row(3) == 7);
}

TEST_CASE("slice with integer reduces rank", "[array][slice]") {
    ndarray<int, 3> a({2, 3, 4}, 0);
    a(1, 2, 3) = 42;

    auto s = slice(a, 1, all, all);  // fix first dim → 2D
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

// ── Broadcasting tests ─────────────────────────────────────────────

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
    // a is (3, 1), b is (1, 4) → result is (3, 4)
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

// ── Reduction tests ────────────────────────────────────────────────

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

    // Sum along axis 0 → shape (3,)
    auto s0 = sum_axis(a, 0);
    REQUIRE(s0.extent(0) == 3);
    REQUIRE(s0(0) == 5);   // 1+4
    REQUIRE(s0(1) == 7);   // 2+5
    REQUIRE(s0(2) == 9);   // 3+6

    // Sum along axis 1 → shape (2,)
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

// ── Fill and compound assignment ───────────────────────────────────

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

// ── 3D array test ──────────────────────────────────────────────────

TEST_CASE("3D ndarray construction and access", "[array][3d]") {
    ndarray<double, 3> vol({4, 5, 6});
    REQUIRE(vol.size() == 120);
    REQUIRE(vol.is_contiguous());

    vol(2, 3, 4) = 3.14;
    REQUIRE(vol(2, 3, 4) == Catch::Approx(3.14));
}

// ── Contiguity ─────────────────────────────────────────────────────

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

// ── Copy semantics ─────────────────────────────────────────────────

TEST_CASE("ndarray copy is deep", "[array][copy]") {
    ndarray<int, 1> a(shape<1>{3});
    a(0) = 1; a(1) = 2; a(2) = 3;

    ndarray<int, 1> b = a;
    b(0) = 99;
    REQUIRE(a(0) == 1);  // original unchanged
    REQUIRE(b(0) == 99);
}
