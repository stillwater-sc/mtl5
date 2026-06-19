// MTL5 -- Sparse factorizations must work with custom number types whose
// abs()/sqrt() are found only via ADL (not std::). This guards against
// regressions to unqualified std::abs / std::sqrt in the sparse factorizations,
// which break custom field types (e.g. Universal's posit/cfloat) that are not
// implicitly convertible to a built-in float.
//
// `adl::real` deliberately mimics that situation: it is only EXPLICITLY
// convertible to double, and its abs()/sqrt() live in namespace `adl`, so the
// factorization code must call them in an ADL-friendly way. No external
// dependency is used (MTL5 must not depend on Universal).
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <compare>
#include <cstddef>
#include <limits>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>
#include <mtl/sparse/factorization/sparse_cholesky.hpp>
#include <mtl/sparse/factorization/sparse_qr.hpp>

namespace adl {

// A minimal ordered field wrapping double, EXPLICITLY convertible to double so
// that std::abs(real)/std::sqrt(real) do NOT compile -- abs()/sqrt() must be
// found by ADL in this namespace.
struct real {
    double v{};
    constexpr real() = default;
    constexpr real(double x) : v(x) {}            // enables real{0}, real{1}
    explicit constexpr operator double() const { return v; }

    constexpr real operator+(real o) const { return real{v + o.v}; }
    constexpr real operator-(real o) const { return real{v - o.v}; }
    constexpr real operator*(real o) const { return real{v * o.v}; }
    constexpr real operator/(real o) const { return real{v / o.v}; }
    constexpr real operator-() const { return real{-v}; }
    real& operator+=(real o) { v += o.v; return *this; }
    real& operator-=(real o) { v -= o.v; return *this; }
    real& operator*=(real o) { v *= o.v; return *this; }
    real& operator/=(real o) { v /= o.v; return *this; }

    constexpr auto operator<=>(const real&) const = default;
    constexpr bool operator==(const real&) const = default;
};

// abs/sqrt visible ONLY via ADL (intentionally not in std).
inline real abs(real x)  { return real{x.v < 0.0 ? -x.v : x.v}; }
inline real sqrt(real x) { return real{std::sqrt(x.v)}; }

} // namespace adl

// numeric_limits specialization (sparse_qr uses ::min()).
template <>
class std::numeric_limits<adl::real> {
public:
    static constexpr bool is_specialized = true;
    static constexpr adl::real min()     noexcept { return adl::real{std::numeric_limits<double>::min()}; }
    static constexpr adl::real max()     noexcept { return adl::real{std::numeric_limits<double>::max()}; }
    static constexpr adl::real lowest()  noexcept { return adl::real{std::numeric_limits<double>::lowest()}; }
    static constexpr adl::real epsilon() noexcept { return adl::real{std::numeric_limits<double>::epsilon()}; }
};

using namespace mtl;
using R = adl::real;

namespace {

double residual_inf(const mat::compressed2D<R>& A,
                    const vec::dense_vector<R>& x,
                    const vec::dense_vector<R>& b) {
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

TEST_CASE("sparse_lu works with a custom ADL-abs scalar type", "[sparse][custom][adl]") {
    // Unsymmetric tridiagonal.
    std::size_t n = 6;
    mat::compressed2D<R> A(n, n);
    {
        mat::inserter<mat::compressed2D<R>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            if (i > 0)     ins[i][i - 1] << R{-1.0};
            ins[i][i] << R{4.0};
            if (i + 1 < n) ins[i][i + 1] << R{-2.0};
        }
    }
    vec::dense_vector<R> b(n, R{1.0}), x(n, R{0.0});
    sparse::factorization::sparse_lu_solve(A, x, b);
    REQUIRE(residual_inf(A, x, b) < 1e-10);
}

TEST_CASE("sparse_cholesky works with a custom ADL-sqrt scalar type",
          "[sparse][custom][adl]") {
    // SPD tridiagonal: 4 on the diagonal, -1 off-diagonal.
    std::size_t n = 6;
    mat::compressed2D<R> A(n, n);
    {
        mat::inserter<mat::compressed2D<R>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            if (i > 0)     ins[i][i - 1] << R{-1.0};
            ins[i][i] << R{4.0};
            if (i + 1 < n) ins[i][i + 1] << R{-1.0};
        }
    }
    vec::dense_vector<R> b(n, R{1.0}), x(n, R{0.0});
    sparse::factorization::sparse_cholesky_solve(A, x, b);
    REQUIRE(residual_inf(A, x, b) < 1e-10);
}

TEST_CASE("sparse_qr works with a custom ADL-abs/sqrt scalar type",
          "[sparse][custom][adl]") {
    // Square nonsingular system (QR as a general solver).
    std::size_t n = 5;
    mat::compressed2D<R> A(n, n);
    {
        mat::inserter<mat::compressed2D<R>> ins(A);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << R{4.0};
            if (i > 0)     ins[i][i - 1] << R{-1.0};
            if (i + 1 < n) ins[i][i + 1] << R{-1.0};
        }
    }
    vec::dense_vector<R> b(n, R{1.0}), x(n, R{0.0});
    sparse::factorization::sparse_qr_solve(A, x, b);
    REQUIRE(residual_inf(A, x, b) < 1e-8);
}
