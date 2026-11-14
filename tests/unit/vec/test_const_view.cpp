// Read-only vector views over a caller's const buffer (tag::read_only):
// zero-copy input for callback interfaces such as rhs(t, const double* y, ...),
// with writes rejected at compile time.
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <utility>

#include <mtl/mtl.hpp>

using namespace mtl;
using cview = vec::const_vector_view<double>;

namespace {
template <typename V> concept element_writable = requires(V v) { v(0) = 1.0; };
template <typename V> concept index_writable   = requires(V v) { v[0] = 1.0; };
template <typename V> concept resizable        = requires(V v) { v.change_dim(3); };
template <typename V> concept scalable         = requires(V v) { v *= 2.0; };
template <typename V> concept expr_assignable  = requires(V v, const V& w) { v = w + w; };
template <typename V> concept data_writable    = requires(V v) { *v.data() = 1.0; };
template <typename V> concept iter_writable    = requires(V v) { *v.begin() = 1.0; };
template <typename V> concept fillable         = requires(V v) { vec::fill(v, 1.0); };
} // namespace

TEST_CASE("const_view: writing through a read-only vector does not compile", "[vec][const_view]") {
    STATIC_REQUIRE(cview::read_only);
    STATIC_REQUIRE(!element_writable<cview>);
    STATIC_REQUIRE(!index_writable<cview>);
    STATIC_REQUIRE(!resizable<cview>);
    STATIC_REQUIRE(!scalable<cview>);
    STATIC_REQUIRE(!expr_assignable<cview>);
    STATIC_REQUIRE(!data_writable<cview>);
    STATIC_REQUIRE(!iter_writable<cview>);
    STATIC_REQUIRE(!fillable<cview>);
    STATIC_REQUIRE(!std::is_copy_assignable_v<cview>);
    STATIC_REQUIRE(!std::is_move_assignable_v<cview>);
    STATIC_REQUIRE(!std::is_constructible_v<cview, std::size_t>);
    STATIC_REQUIRE(std::is_same_v<cview::reference, const double&>);
    STATIC_REQUIRE(std::is_same_v<cview::pointer, const double*>);
    // The mutable vector still cannot be built over a const buffer.
    STATIC_REQUIRE(!std::is_constructible_v<vec::dense_vector<double>, std::size_t, const double*>);
    // And it is unchanged.
    STATIC_REQUIRE(element_writable<vec::dense_vector<double>>);
    STATIC_REQUIRE(std::is_copy_assignable_v<vec::dense_vector<double>>);
}

TEST_CASE("const_view: reads a const buffer without copying it", "[vec][const_view]") {
    const double buf[4] = {1.0, 2.0, 3.0, 4.0};
    const auto y = vec::const_view(4, buf);
    STATIC_REQUIRE(std::is_same_v<std::remove_const_t<decltype(y)>, cview>);
    REQUIRE(y.size() == 4);
    REQUIRE(y.data() == buf);
    REQUIRE(!y.owns_memory());
    REQUIRE(y(2) == 3.0);
    REQUIRE(y[3] == 4.0);
    double s = 0;
    for (double v : y) s += v;
    REQUIRE(s == 10.0);

    cview z(4, buf);   // non-const object: still read-only
    REQUIRE(z(0) == 1.0);
    const cview c = z;   // a copy is another view of the same buffer
    REQUIRE(c.data() == buf);
    cview m(std::move(z));
    REQUIRE(m.data() == buf);
}

TEST_CASE("const_view: a view of a mutable buffer sees later writes", "[vec][const_view]") {
    double buf[3] = {1.0, 2.0, 3.0};
    const cview y(3, buf);
    buf[1] = 20.0;   // the producer writes; the view reads the same memory
    REQUIRE(y(1) == 20.0);

    vec::dense_vector<double> v{5.0, 6.0};
    const auto vv = vec::const_view(v);
    REQUIRE(vv.data() == v.data());
    v(0) = 50.0;
    REQUIRE(vv(0) == 50.0);
}

TEST_CASE("const_view works with the vector operations", "[vec][const_view]") {
    const double xb[3] = {3.0, 0.0, 4.0};
    const double yb[3] = {1.0, 2.0, 3.0};
    const auto x = vec::const_view(3, xb);
    const auto y = vec::const_view(3, yb);

    REQUIRE(two_norm(x) == 5.0);
    REQUIRE(one_norm(x) == 7.0);
    REQUIRE(infinity_norm(x) == 4.0);
    REQUIRE(dot(x, y) == 15.0);

    const vec::dense_vector<double> s = x + 2.0 * y;   // expressions read the views
    REQUIRE(s(2) == 10.0);
    vec::dense_vector<double> out(3, 0.0);
    out = x - y;
    REQUIRE(out(1) == -2.0);
    out += -y;
    REQUIRE(out(0) == 1.0);
    const vec::dense_vector<double> copy(x);   // an owning copy of a view
    REQUIRE(copy.owns_memory());
    REQUIRE(copy.data() != xb);
    REQUIRE(copy(2) == 4.0);
}

TEST_CASE("const_view as the input of a callback that writes its output in place", "[vec][const_view]") {
    // dy/dt = A y for the 1D Laplacian, the shape of an ODE integrator's rhs.
    const std::size_t n = 5;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << -2.0;
            if (i > 0)     ins[i][i - 1] << 1.0;
            if (i + 1 < n) ins[i][i + 1] << 1.0;
        }
    }
    auto rhs = [&](const double* y, double* dydt) {
        const auto Y = vec::const_view(n, y);
        vec::dense_vector<double> dY(n, dydt);
        dY = A * Y;
        dY *= 0.5;
    };
    const double y[n] = {0.0, 1.0, 4.0, 9.0, 16.0};
    double dydt[n] = {};
    rhs(y, dydt);
    REQUIRE(dydt[0] == 0.5);
    REQUIRE(dydt[2] == 1.0);   // (1 - 8 + 9) / 2
    REQUIRE(dydt[4] == -11.5); // (9 - 32) / 2
}

TEST_CASE("const_view as the right-hand side of cg", "[vec][const_view][itl]") {
    const std::size_t n = 4;
    mat::compressed2D<double> A(n, n);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 4.0;
            if (i > 0)     ins[i][i - 1] << -1.0;
            if (i + 1 < n) ins[i][i + 1] << -1.0;
        }
    }
    const double bb[n] = {1.0, 2.0, 3.0, 4.0};
    const auto b = vec::const_view(n, bb);
    vec::dense_vector<double> x(n, 0.0);
    itl::pc::identity<mat::compressed2D<double>> I(A);
    itl::basic_iteration<double> iter(b, 100, 1e-12);
    itl::cg(A, x, b, I, iter);
    const vec::dense_vector<double> r = b - A * x;
    REQUIRE(two_norm(r) < 1e-10);
}

TEST_CASE("const_view of an empty buffer", "[vec][const_view]") {
    const cview e;
    REQUIRE(e.size() == 0);
    const auto z = vec::const_view<double>(0, nullptr);
    REQUIRE(z.size() == 0);
    REQUIRE(two_norm(z) == 0.0);
}
