// mult(A, x, y) when y shares storage with x or A (#840). Every kernel writes y
// while it still reads its inputs -- the column-major gemv zeroes y first -- so
// an overlapping y is computed through a temporary, in every build. Before
// #840 a debug-only assert caught the same object, and a vector over part of
// the same buffer (easy with zero-copy views, #833/#835) went unseen.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <tuple>
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

TEST_CASE("mult with y over a compressed matrix's values or a view's base", "[operation][mult][840]") {
    // CodeRabbit's examples on #841: storage reached through ref_data() and
    // through a transposed view's base(), which have no data().
    auto make_S = [] {
        mat::compressed2D<double> S(2, 2);
        {   // the inserter finalizes S when it goes out of scope
            mat::inserter<mat::compressed2D<double>> ins(S);
            ins[0][0] << 1.0; ins[0][1] << 2.0;
            ins[1][0] << 4.0; ins[1][1] << 8.0;
        }
        return S;
    };
    const vec::dense_vector<double> x(2, 1.0);
    {
        auto S = make_S();
        vec::dense_vector<double> y(2, S.ref_data().data() + 2);   // the values 4, 8
        mult(S, x, y);
        CHECK(S.ref_data()[2] == 3.0);
        CHECK(S.ref_data()[3] == 12.0);   // 11 before the fix
    }
    {
        auto S = make_S();
        vec::dense_vector<double> y(2, S.ref_data().data());       // the values 1, 2
        mult(trans(S), x, y);
        CHECK(S.ref_data()[0] == 5.0);
        CHECK(S.ref_data()[1] == 10.0);   // {4, 8} before the fix
    }
    {
        mat::dense2D<double> A{{1, 2}, {3, 4}};
        vec::dense_vector<double> y(2, A.data() + 1);              // 2, 3
        mult<double>(trans(A), x, y);
        CHECK(A.data()[1] == 4.0);
        CHECK(A.data()[2] == 6.0);        // 8 before the fix
    }
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
    mat::dense2D<double> M(2, 2);
    const auto Mv = vec::const_view(2, M.data() + 2);
    CHECK(detail::storage_overlaps(Mv, trans(M)));   // through the view's base
    CHECK(!detail::storage_overlaps(vec::const_view(2, buf.data()), trans(M)));
}

// ---------------------------------------------------------------------------
// #855: matrix-matrix mult, mult_quad and symv, and the matrix-vector cases
// #840 left out: a C (or y) that shares storage with an input is computed
// through a temporary in every build. These were debug-only asserts that saw
// only the same object, and wrong under NDEBUG: mult(A, B, A) on a 3x3
// dense2D<double> differed from mult(A, B, R) in 9 of 9 entries.

namespace {

template <typename M>
M filled(std::size_t r, std::size_t c, int seed) {
    M A(r, c);
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j)
            A(i, j) = typename M::value_type((int(i * 7 + j * 3) + seed) % 11) / 4;
    return A;
}

template <typename M>
bool same(const M& A, const M& B) {
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t j = 0; j < A.num_cols(); ++j)
            if (A(i, j) != B(i, j)) return false;
    return true;
}

} // namespace

TEMPLATE_TEST_CASE("mult(A, B, A) and mult(A, B, B)", "[operation][mult][855]",
                   (mat::dense2D<double>),
                   (mat::dense2D<double, mat::parameters<tag::col_major>>),
                   (mat::dense2D<float>)) {
    using M = TestType;
    for (std::size_t n : {3u, 8u, 33u}) {
        INFO("n = " << n);
        auto A = filled<M>(n, n, 1);
        const auto B = filled<M>(n, n, 5);
        M R(n, n);
        mult(A, B, R);
        mult(A, B, A);
        CHECK(same(A, R));

        const auto A2 = filled<M>(n, n, 2);
        auto B2 = filled<M>(n, n, 9);
        M R2(n, n);
        mult(A2, B2, R2);
        mult(A2, B2, B2);
        CHECK(same(B2, R2));

        auto A3 = filled<M>(n, n, 4);
        M R3(n, n);
        mult(A3, A3, R3);
        mult(A3, A3, A3);   // C is both operands
        CHECK(same(A3, R3));

        auto A4 = filled<M>(n, n, 6);
        M R4(n, n);
        mult(trans(A4), B, R4);
        mult(trans(A4), B, A4);   // C behind A's transposed view
        CHECK(same(A4, R4));

        auto A5 = filled<M>(n, n, 7);
        M R5(n, n);
        mult<double>(A5, B, R5);
        mult<double>(A5, B, A5);   // the accumulator path
        CHECK(same(A5, R5));
    }
}

TEST_CASE("mult(A, B, C): C over part of A's buffer", "[operation][mult][855]") {
    // A is 2 x 4 over buf[0, 8); C is 2 x 2 over buf[4, 8), A's second row.
    // same_object saw two objects, even in a debug build.
    std::vector<double> buf(8);
    for (std::size_t i = 0; i < buf.size(); ++i) buf[i] = double(i % 5) - 1.5;
    mat::dense2D<double> Acopy(2, 4);
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 4; ++j) Acopy(i, j) = buf[i * 4 + j];
    const auto B = filled<mat::dense2D<double>>(4, 2, 3);
    mat::dense2D<double> R(2, 2);
    mult(Acopy, B, R);

    mat::dense2D<double> A(2, 4, buf.data());
    mat::dense2D<double> C(2, 2, buf.data() + 4);
    mult(A, B, C);
    CHECK(C.data() == buf.data() + 4);   // still writes the caller's buffer
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 2; ++j) CHECK(buf[4 + i * 2 + j] == R(i, j));
}

TEST_CASE("mult_quad: C over the bytes of A", "[operation][mult][855]") {
    // (u8, u8) -> u32: A is 2 x 4 uint8 over the first 8 bytes of C's buffer.
    mat::dense2D<std::uint32_t> C(2, 2);
    auto* bytes = reinterpret_cast<std::uint8_t*>(C.data());
    for (std::size_t k = 0; k < 8; ++k) bytes[k] = std::uint8_t(3 * k + 1);
    mat::dense2D<std::uint8_t> Acopy(2, 4);
    for (std::size_t k = 0; k < 8; ++k) Acopy(k / 4, k % 4) = bytes[k];
    mat::dense2D<std::uint8_t> B(4, 2);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 2; ++j) B(i, j) = std::uint8_t(i + 2 * j + 1);
    mat::dense2D<std::uint32_t> R(2, 2);
    mult_quad<std::uint32_t>(Acopy, B, R);

    mat::dense2D<std::uint8_t> A(2, 4, bytes);
    mult_quad<std::uint32_t>(A, B, C);
    CHECK(same(C, R));
}

TEST_CASE("symv with y aliasing x or A", "[operation][symv][855]") {
    for (bool col_major : {false, true}) {
        INFO("col_major = " << col_major);
        const std::size_t n = 7;
        auto run = [&](auto A) {
            for (std::size_t i = 0; i < n; ++i)
                for (std::size_t j = 0; j <= i; ++j) A(i, j) = A(j, i) = double(i + 2 * j) / 3 - 1;
            vec::dense_vector<double> x(n), want(n);
            for (std::size_t i = 0; i < n; ++i) want(i) = x(i) = double(i % 3) + 0.5;
            vec::dense_vector<double> x2(x), want2(x);
            symv(2.0, A, x, 0.5, want);
            symv(2.0, A, x, 0.5, x);
            for (std::size_t i = 0; i < n; ++i) CHECK(x(i) == want(i));
            symv<double>(2.0, A, x2, 0.5, want2);   // the accumulator path
            symv<double>(2.0, A, x2, 0.5, x2);
            for (std::size_t i = 0; i < n; ++i) CHECK(x2(i) == want2(i));
        };
        if (col_major) run(mat::dense2D<double, mat::parameters<tag::col_major>>(n, n));
        else           run(mat::dense2D<double>(n, n));
    }
}

TEMPLATE_TEST_CASE("mult(A, x, x): column-major A, float, and transposed A", "[operation][mult][855]",
                   (mat::dense2D<double, mat::parameters<tag::col_major>>),
                   (mat::dense2D<float>),
                   (mat::dense2D<float, mat::parameters<tag::col_major>>)) {
    using M = TestType;
    using T = typename M::value_type;
    for (std::size_t n : {3u, 17u, 64u}) {
        INFO("n = " << n);
        const auto A = filled<M>(n, n, 2);
        vec::dense_vector<T> x(n), want(n), wantT(n), wantAcc(n);
        for (std::size_t i = 0; i < n; ++i) x(i) = T(1) + T(i % 5) / T(4);
        const vec::dense_vector<T> x0(x);
        mult(A, x0, want);
        mult(trans(A), x0, wantT);
        mult<double>(trans(A), x0, wantAcc);

        mult(A, x, x);   // the gemv_colmajor zeroing path for a column-major A
        bool ok = true;
        for (std::size_t i = 0; i < n; ++i) ok = ok && x(i) == want(i);
        CHECK(ok);

        x = x0;
        mult(trans(A), x, x);
        ok = true;
        for (std::size_t i = 0; i < n; ++i) ok = ok && x(i) == wantT(i);
        CHECK(ok);

        x = x0;
        mult<double>(trans(A), x, x);   // accumulator with a transposed A
        ok = true;
        for (std::size_t i = 0; i < n; ++i) ok = ok && x(i) == wantAcc(i);
        CHECK(ok);
    }
}

TEST_CASE("storage_overlaps sees coordinate2D, ell_matrix and block_diagonal2D storage",
          "[operation][mult][855]") {
    mat::compressed2D<double> S(3, 3);
    {
        mat::inserter<mat::compressed2D<double>> ins(S);
        ins[0][0] << 1.0; ins[1][1] << 2.0; ins[2][0] << 3.0;
    }
    const mat::ell_matrix<double> E(S);
    CHECK(detail::storage_overlaps(vec::const_view(1, E.ref_data().data() + 1), E));
    CHECK(!detail::storage_overlaps(vec::const_view(1, S.ref_data().data()), E));

    mat::coordinate2D<double> K(3, 3);
    K.insert(0, 1, 4.0);
    K.insert(2, 2, 5.0);
    const double* v = &std::get<2>(K.ref_entries()[1]);
    CHECK(detail::storage_overlaps(vec::const_view(1, v), K));
    CHECK(!detail::storage_overlaps(vec::const_view(1, S.ref_data().data()), K));

    mat::block_diagonal2D<double> D({filled<mat::dense2D<double>>(2, 2, 1),
                                     filled<mat::dense2D<double>>(3, 3, 2)});
    CHECK(detail::storage_overlaps(vec::const_view(2, D.block(1).data() + 3), D));   // the second block
    CHECK(detail::storage_overlaps(D, vec::const_view(1, D.block(0).data())));
    CHECK(!detail::storage_overlaps(vec::const_view(1, S.ref_data().data()), D));
}
