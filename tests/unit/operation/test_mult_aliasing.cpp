// mult(A, x, y) when y shares storage with x or A (#840). Every kernel writes y
// while it still reads its inputs -- the column-major gemv zeroes y first -- so
// an overlapping y is computed through a temporary, in every build. Before
// #840 a debug-only assert caught the same object, and a vector over part of
// the same buffer (easy with zero-copy views, #833/#835) went unseen.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

#include <mtl/mtl.hpp>

using namespace mtl;

namespace {

mat::dense2D<double> dense_operator(std::size_t n, bool col_major_values = false) {
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = double((i + 1) * 10 + (col_major_values ? j * 3 : j) % 7) / 8.0;
    return A;
}

std::vector<double> reference(const mat::dense2D<double>& A, const std::vector<double>& x) {
    std::vector<double> y(A.num_rows(), 0.0);
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t j = 0; j < A.num_cols(); ++j) y[i] += A(i, j) * x[j];
    return y;
}

bool near(double a, double b) { return a == b || std::abs(a - b) <= 1e-12 * (std::abs(a) + std::abs(b)); }

} // namespace

TEST_CASE("mult(A, x, x): y is the same vector as x", "[operation][mult][840]") {
    for (std::size_t n : {3u, 17u, 64u}) {
        const auto A = dense_operator(n);
        vec::dense_vector<double> x(n);
        std::vector<double> xs(n);
        for (std::size_t i = 0; i < n; ++i) xs[i] = x(i) = 1.0 + double(i % 5);
        const auto want = reference(A, xs);
        mult(A, x, x);   // aborted in a debug build, wrong under NDEBUG, before #840
        bool ok = true;
        for (std::size_t i = 0; i < n; ++i) ok = ok && near(x(i), want[i]);
        CHECK(ok);
    }
}

TEST_CASE("mult<Accumulator>(A, x, x)", "[operation][mult][840]") {
    const std::size_t n = 9;
    const auto A = dense_operator(n);
    vec::dense_vector<double> x(n);
    std::vector<double> xs(n);
    for (std::size_t i = 0; i < n; ++i) xs[i] = x(i) = double(i) - 3.5;
    const auto want = reference(A, xs);
    mult<double>(A, x, x);   // the accumulator path, aliased
    bool ok = true;
    for (std::size_t i = 0; i < n; ++i) ok = ok && near(x(i), want[i]);
    CHECK(ok);
}

TEST_CASE("mult(A, x, y): y over part of x's buffer", "[operation][mult][840]") {
    const std::size_t n = 16;
    const auto A = dense_operator(n);
    std::vector<double> buf(n + 4);
    for (std::size_t i = 0; i < buf.size(); ++i) buf[i] = double(i % 7) - 2.0;
    const std::vector<double> xs(buf.begin(), buf.begin() + n);
    const auto want = reference(A, xs);
    const auto x = vec::const_view(n, buf.data());           // reads buf[0, n)
    vec::dense_vector<double> y(n, buf.data() + 2);          // writes buf[2, n + 2)
    mult(A, x, y);   // not caught before #840: same_object saw two objects
    bool ok = true;
    for (std::size_t i = 0; i < n; ++i) ok = ok && near(buf[i + 2], want[i]);
    CHECK(ok);
    CHECK(y.data() == buf.data() + 2);   // still writes the caller's buffer
}

TEST_CASE("mult(A, x, y): y over part of A's storage", "[operation][mult][840]") {
    const std::size_t n = 6;
    auto A = dense_operator(n);
    const auto A0 = A;   // an owning copy, for the reference
    std::vector<double> xs(n, 1.0);
    const vec::dense_vector<double> x(n, 1.0);
    const auto want = reference(A0, xs);
    vec::dense_vector<double> y(n, A.data());   // the first row of A, row-major
    mult(A, x, y);
    bool ok = true;
    for (std::size_t i = 0; i < n; ++i) ok = ok && near(A.data()[i], want[i]);
    CHECK(ok);
}

TEST_CASE("mult(A, x, x) for compressed2D and its transpose", "[operation][mult][840]") {
    const std::size_t n = 8;
    mat::compressed2D<double> S(n, n);
    mat::dense2D<double> D(n, n);
    std::fill(D.begin(), D.end(), 0.0);
    {
        mat::inserter<mat::compressed2D<double>> ins(S);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << 2.0;                       D(i, i) = 2.0;
            if (i + 1 < n) { ins[i][i + 1] << -1.0; D(i, i + 1) = -1.0; }
            if (i > 0)     { ins[i][i - 1] << 0.5;  D(i, i - 1) = 0.5; }
        }
    }
    std::vector<double> xs(n);
    for (std::size_t i = 0; i < n; ++i) xs[i] = double(i + 1);

    vec::dense_vector<double> x(n);
    for (std::size_t i = 0; i < n; ++i) x(i) = xs[i];
    mult(S, x, x);
    const auto want = reference(D, xs);
    bool ok = true;
    for (std::size_t i = 0; i < n; ++i) ok = ok && near(x(i), want[i]);
    CHECK(ok);

    for (std::size_t i = 0; i < n; ++i) x(i) = xs[i];
    mult(trans(S), x, x);
    mat::dense2D<double> Dt(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) Dt(i, j) = D(j, i);
    const auto want_t = reference(Dt, xs);
    ok = true;
    for (std::size_t i = 0; i < n; ++i) ok = ok && near(x(i), want_t[i]);
    CHECK(ok);
}

TEST_CASE("storage_overlaps", "[operation][mult][840]") {
    std::vector<double> buf(10);
    const auto a = vec::const_view(4, buf.data());
    const auto b = vec::const_view(4, buf.data() + 3);
    const auto c = vec::const_view(4, buf.data() + 4);
    CHECK(detail::storage_overlaps(a, b));
    CHECK(!detail::storage_overlaps(a, c));   // adjacent, not overlapping
    CHECK(detail::storage_overlaps(a, a));
    const vec::strided_vector_ref<double> s(buf.data() + 1, 3, 4);   // 1, 5, 9
    CHECK(detail::storage_overlaps(s, c));
    CHECK(!detail::storage_overlaps(s, vec::const_view(1, buf.data())));
    const vec::dense_vector<double> empty;
    CHECK(!detail::storage_overlaps(empty, a));
}
