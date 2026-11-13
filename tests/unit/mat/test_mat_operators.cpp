// The matrix expressions and operators of mat/operators.hpp, tested directly
// (#824): the lazy product, negation and right-scalar expressions, and the
// eager products at the edges (empty, 1x1, non-square, mixed value types,
// sparse operands).
#include <catch2/catch_test_macros.hpp>

#include <cstddef>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/mat/operators.hpp>
#include <mtl/mat/view/transposed_view.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/vec/dense_vector.hpp>

using namespace mtl;

namespace {

/// A(r, c) = r * cols + c + 1, so every entry differs.
template <typename T = double>
mat::dense2D<T> counting(std::size_t rows, std::size_t cols) {
    mat::dense2D<T> A(rows, cols);
    for (std::size_t r = 0; r < rows; ++r)
        for (std::size_t c = 0; c < cols; ++c) A(r, c) = static_cast<T>(r * cols + c + 1);
    return A;
}

/// The textbook product, for reference.
template <typename MA, typename MB>
mat::dense2D<double> naive(const MA& A, const MB& B) {
    mat::dense2D<double> C(A.num_rows(), B.num_cols());
    for (std::size_t r = 0; r < A.num_rows(); ++r)
        for (std::size_t c = 0; c < B.num_cols(); ++c) {
            double acc = 0;
            for (std::size_t k = 0; k < A.num_cols(); ++k)
                acc += static_cast<double>(A(r, k)) * static_cast<double>(B(k, c));
            C(r, c) = acc;
        }
    return C;
}

template <typename M>
void require_equal(const M& a, const mat::dense2D<double>& b) {
    REQUIRE(a.num_rows() == b.num_rows());
    REQUIRE(a.num_cols() == b.num_cols());
    for (std::size_t r = 0; r < b.num_rows(); ++r)
        for (std::size_t c = 0; c < b.num_cols(); ++c) {
            INFO("(" << r << ", " << c << ")");
            REQUIRE(static_cast<double>(a(r, c)) == b(r, c));
        }
}

/// The CRS matrix with A's nonzeros.
mat::compressed2D<double> to_crs(const mat::dense2D<double>& A) {
    mat::compressed2D<double> S(A.num_rows(), A.num_cols());
    {
        mat::compressed2D_inserter<double> ins(S, 4);
        for (std::size_t r = 0; r < A.num_rows(); ++r)
            for (std::size_t c = 0; c < A.num_cols(); ++c)
                if (A(r, c) != 0) ins[r][c] << A(r, c);
    }
    return S;
}

} // namespace

// -- The expressions ------------------------------------------------------

TEST_CASE("mat_mat_times_expr computes the product lazily", "[mat][expr][824]") {
    const auto A = counting(2, 3);
    const auto B = counting(3, 4);
    const mat::expr::mat_mat_times_expr<const mat::dense2D<double>&, const mat::dense2D<double>&> P(A, B);
    REQUIRE(P.num_rows() == 2);
    REQUIRE(P.num_cols() == 4);
    require_equal(P, naive(A, B));
    mat::dense2D<double> C(P);
    require_equal(C, naive(A, B));
}

TEST_CASE("mat_negate_expr and mat_rscal_op_expr", "[mat][expr][824]") {
    const auto A = counting(2, 3);
    const auto N = -A;
    REQUIRE(N.num_rows() == 2);
    REQUIRE(N.num_cols() == 3);
    REQUIRE(N(1, 2) == -6.0);
    const auto H = A / 2.0;
    REQUIRE(H.num_rows() == 2);
    REQUIRE(H(1, 2) == 3.0);
    const auto R = A * 3.0;   // matrix * scalar
    REQUIRE(R(0, 1) == 6.0);
    mat::dense2D<double> D(-(A / 2.0));
    REQUIRE(D(1, 0) == -2.0);
}

// -- Matrix * Matrix --------------------------------------------------------

TEST_CASE("Matrix * Matrix at the edges", "[mat][operators][824]") {
    SECTION("empty operands") {
        const mat::dense2D<double> A(0, 3), B(3, 2);
        const auto C = A * B;
        REQUIRE(C.num_rows() == 0);
        REQUIRE(C.num_cols() == 2);
        const mat::dense2D<double> E(2, 0), F(0, 3);
        const auto G = E * F;   // inner dimension 0: a 2x3 zero matrix
        require_equal(G, {{0, 0, 0}, {0, 0, 0}});
    }
    SECTION("1x1") {
        const mat::dense2D<double> A{{3}}, B{{4}};
        require_equal(A * B, {{12}});
    }
    SECTION("non-square") {
        const auto A = counting(3, 5);
        const auto B = counting(5, 2);
        require_equal(A * B, naive(A, B));
    }
    SECTION("mixed value types: float * double gives double") {
        const auto A = counting<float>(3, 4);
        const auto B = counting<double>(4, 3);
        const auto C = A * B;
        STATIC_REQUIRE(std::is_same_v<typename std::remove_cvref_t<decltype(C)>::value_type, double>);
        require_equal(C, naive(A, B));
    }
    SECTION("a view and an expression as operands") {
        const auto A = counting(3, 4);
        const auto At = trans(A);
        require_equal(A * At, naive(A, At));
        const auto S = A + A;
        require_equal(S * At, naive(S, At));
    }
}

TEST_CASE("Sparse operands of Matrix * Matrix", "[mat][operators][sparse][824]") {
    mat::dense2D<double> Ad{{1, 0, 2, 0}, {0, 0, 0, 3}, {4, 5, 0, 0}};   // 3x4
    const auto Bd = counting(4, 3);
    const auto S = to_crs(Ad);
    require_equal(S * Bd, naive(Ad, Bd));            // CRS kernel (#550)
    const auto Sb = to_crs(counting(3, 2));
    require_equal(Bd * Sb, naive(Bd, counting(3, 2)));   // dense * CRS
}

// -- Matrix * Vector --------------------------------------------------------

TEST_CASE("Matrix * Vector at the edges", "[mat][operators][824]") {
    SECTION("empty") {
        const mat::dense2D<double> A(0, 3);
        vec::dense_vector<double> x(3, 1.0);
        const auto y = A * x;
        REQUIRE(y.size() == 0);
        const mat::dense2D<double> B(2, 0);
        const vec::dense_vector<double> z(0);
        const auto w = B * z;
        REQUIRE(w.size() == 2);
        REQUIRE(w(0) == 0.0);
        REQUIRE(w(1) == 0.0);
    }
    SECTION("1x1") {
        const mat::dense2D<double> A{{3}};
        const vec::dense_vector<double> x(1, 4.0);
        REQUIRE((A * x)(0) == 12.0);
    }
    SECTION("mixed value types") {
        const auto A = counting<float>(2, 3);
        vec::dense_vector<double> x(3);
        x(0) = 1; x(1) = 0.5; x(2) = 0.25;
        const auto y = A * x;
        STATIC_REQUIRE(std::is_same_v<typename std::remove_cvref_t<decltype(y)>::value_type, double>);
        REQUIRE(y(0) == 1 + 1.0 + 0.75);
        REQUIRE(y(1) == 4 + 2.5 + 1.5);
    }
}

TEST_CASE("trans(compressed2D) * Vector: empty and non-square", "[mat][operators][sparse][824]") {
    SECTION("non-square") {
        mat::dense2D<double> Ad{{1, 0, 2, 0}, {0, 0, 0, 3}, {4, 5, 0, 0}};   // 3x4
        const auto S = to_crs(Ad);
        const mat::view::transposed_view<mat::compressed2D<double>> St(S);
        vec::dense_vector<double> x(3);
        x(0) = 1; x(1) = 2; x(2) = 3;
        const auto y = St * x;   // 4 entries: A^T x
        REQUIRE(y.size() == 4);
        REQUIRE(y(0) == 1 + 12.0);
        REQUIRE(y(1) == 15.0);
        REQUIRE(y(2) == 2.0);
        REQUIRE(y(3) == 6.0);
    }
    SECTION("empty") {
        const mat::compressed2D<double> S(0, 3);
        const mat::view::transposed_view<mat::compressed2D<double>> St(S);
        const vec::dense_vector<double> x(0);
        const auto y = St * x;
        REQUIRE(y.size() == 3);
        REQUIRE(y(0) == 0.0);
    }
}
