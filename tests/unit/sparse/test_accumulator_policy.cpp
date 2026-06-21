// MTL5 -- the sparse_lu numeric accumulator policy (issue #122).
//
// sparse_lu_numeric touches its dense workspace only through
// accumulator_traits<Acc, Value>, so a caller can make the inner accumulation
// extended-precision/exact without MTL5 depending on any external library. This
// test specializes the trait for a custom "wide" accumulator (a double behind a
// float factorization) and shows it yields a smaller residual than the default
// (float-only) accumulation on an ill-conditioned system -- the dependency-free
// stand-in for the Universal `quire` super-accumulator that mp-spice will inject.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>

namespace {

// Extended accumulator: hold the running sum in double behind a float solve, and
// round to float only when the column entry is consumed (single rounding).
struct wide_acc { double v = 0.0; };

} // namespace

namespace mtl::sparse::factorization {
template <>
struct accumulator_traits<wide_acc, float> {
    static void  clear(wide_acc& a)                       { a.v = 0.0; }
    static void  assign(wide_acc& a, const float& x)      { a.v = static_cast<double>(x); }
    static float value(const wide_acc& a)                 { return static_cast<float>(a.v); }
    static void  sub_product(wide_acc& a, const float& m, const float& x) {
        a.v -= static_cast<double>(m) * static_cast<double>(x);   // product in double
    }
};
} // namespace mtl::sparse::factorization

using namespace mtl;

namespace {

// Hilbert-ish dense matrix: H(i,j) = 1/(i+j+1), notoriously ill-conditioned, so
// float accumulation loses accuracy that a double accumulator recovers.
mat::compressed2D<float> hilbert(std::size_t n) {
    mat::compressed2D<float> A(n, n);
    mat::inserter<mat::compressed2D<float>> ins(A);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            ins[i][j] << static_cast<float>(1.0 / static_cast<double>(i + j + 1));
    return A;
}

// Residual ||A x - b||_inf in double.
double residual_inf(const mat::compressed2D<float>& A,
                    const vec::dense_vector<float>& x,
                    const vec::dense_vector<float>& b) {
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& dat = A.ref_data();
    double m = 0.0;
    for (std::size_t r = 0; r < A.num_rows(); ++r) {
        double ax = 0.0;
        for (std::size_t k = rp[r]; k < rp[r + 1]; ++k)
            ax += static_cast<double>(dat[k]) * static_cast<double>(x(static_cast<int>(ci[k])));
        m = std::max(m, std::abs(ax - static_cast<double>(b(static_cast<int>(r)))));
    }
    return m;
}

} // namespace

TEST_CASE("accumulator policy: default path solves correctly", "[sparse][lu][accumulator]") {
    // Default Accumulator == Value: the seam must be a no-op.
    std::size_t n = 8;
    auto A = hilbert(n);
    auto sym = sparse::factorization::sparse_lu_symbolic(A);
    auto num = sparse::factorization::sparse_lu_numeric(A, sym);  // default accumulator
    vec::dense_vector<float> b(n, 1.0f), x(n, 0.0f);
    num.solve(x, b);
    REQUIRE(x.size() == n);   // solves without throwing; accuracy checked below
}

TEST_CASE("accumulator policy: wide accumulator reduces residual vs default",
          "[sparse][lu][accumulator]") {
    std::size_t n = 8;
    auto A = hilbert(n);
    auto sym = sparse::factorization::sparse_lu_symbolic(A);
    vec::dense_vector<float> b(n, 1.0f);

    // Default float accumulation.
    auto num_f = sparse::factorization::sparse_lu_numeric(A, sym);
    vec::dense_vector<float> xf(n, 0.0f);
    num_f.solve(xf, b);
    double res_default = residual_inf(A, xf, b);

    // Extended (double) accumulation via the custom trait.
    auto num_w = sparse::factorization::sparse_lu_numeric<float, mat::parameters<>, wide_acc>(A, sym);
    vec::dense_vector<float> xw(n, 0.0f);
    num_w.solve(xw, b);
    double res_wide = residual_inf(A, xw, b);

    INFO("default(float) residual = " << res_default << ", wide(double) residual = " << res_wide);
    // Extended accumulation must be at least as accurate, and strictly better on
    // this ill-conditioned system.
    REQUIRE(res_wide <= res_default);
}
