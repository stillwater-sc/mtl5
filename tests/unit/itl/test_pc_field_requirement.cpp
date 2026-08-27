// Every preconditioner and smoother requires a Field element type (#505).
//
// The companion to #503, which gated the Krylov solvers. These were left
// unconstrained, and they are a DIFFERENT entry point: the caller constructs one
// before any solver sees it, so `pc::ilu_0<int> M(A)` truncated its factorization
// at construction and the solver gate never fired.
//
// The `value_type(1) / A(i, i)` pattern in diagonal, jacobi, gauss_seidel and sor
// is the sharp end: on an integral type that is 0 whenever |A(i,i)| > 1, so the
// preconditioner is not merely inaccurate, it is the ZERO OPERATOR, and
// M.solve(x, b) returns zeros with no division by zero and no flag.
//
// Same test shape as itl/test_field_requirement.cpp, and for the same reasons: a
// requires-expression per type checks the constraint without instantiating a
// body, so all twelve fit in one cheap TU; and the POSITIVE half is asserted
// alongside the negative, because a constraint that rejected everything would
// pass a rejection-only test.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <complex>
#include <cstdint>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/scalar.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/pc/diagonal.hpp>
#include <mtl/itl/pc/block_diagonal.hpp>
#include <mtl/itl/pc/ic_0.hpp>
#include <mtl/itl/pc/ildl.hpp>
#include <mtl/itl/pc/ilu_0.hpp>
#include <mtl/itl/pc/ilut.hpp>
#include <mtl/itl/pc/ssor.hpp>
#include <mtl/itl/smoother/jacobi.hpp>
#include <mtl/itl/smoother/gauss_seidel.hpp>
#include <mtl/itl/smoother/sor.hpp>

using namespace mtl;

namespace {

/// Minimal custom Field type, standing in for a posit / cfloat / LNS.
struct emul {
    double x = 0.0;
    emul() = default;
    emul(double d) : x(d) {}
};
inline emul operator+(emul a, emul b) { return emul{a.x + b.x}; }
inline emul operator-(emul a, emul b) { return emul{a.x - b.x}; }
inline emul operator-(emul a)         { return emul{-a.x}; }
inline emul operator*(emul a, emul b) { return emul{a.x * b.x}; }
inline emul operator/(emul a, emul b) { return emul{a.x / b.x}; }

static_assert(Field<emul>, "the stand-in must actually be a Field");

// A preconditioner or smoother is a CLASS template, so "is it usable with T?" is
// asked by naming the type rather than by calling a function -- the constraint is
// checked when the specialization is named, and no member is instantiated.
#define MTL5_PC_MATRIX_SHAPE(QUALNAME, TRAIT)                                   \
    template <typename T>                                                       \
    concept TRAIT = requires { typename QUALNAME<mat::dense2D<T>>; };

#define MTL5_PC_VALUE_SHAPE(QUALNAME, TRAIT)                                    \
    template <typename T>                                                       \
    concept TRAIT = requires { typename QUALNAME<T>; };

// Storage shape: constructed from a Matrix.
MTL5_PC_MATRIX_SHAPE(itl::pc::identity,       IdentityOk)
MTL5_PC_MATRIX_SHAPE(itl::pc::diagonal,       DiagonalOk)
MTL5_PC_MATRIX_SHAPE(itl::pc::block_diagonal, BlockDiagonalOk)
MTL5_PC_MATRIX_SHAPE(itl::pc::ssor,           SsorOk)
MTL5_PC_MATRIX_SHAPE(itl::smoother::jacobi,       JacobiOk)
MTL5_PC_MATRIX_SHAPE(itl::smoother::gauss_seidel, GaussSeidelOk)
MTL5_PC_MATRIX_SHAPE(itl::smoother::sor,          SorOk)

// Storage shape: parameterised on the element Value directly.
MTL5_PC_VALUE_SHAPE(itl::pc::ic_0,  Ic0Ok)
MTL5_PC_VALUE_SHAPE(itl::pc::ildl,  IldlOk)
MTL5_PC_VALUE_SHAPE(itl::pc::ilu_0, Ilu0Ok)
MTL5_PC_VALUE_SHAPE(itl::pc::ilut,  IlutOk)

#define MTL5_ASSERT_PC_GATE(TRAIT)                                              \
    static_assert(TRAIT<float>,                #TRAIT " must accept float");             \
    static_assert(TRAIT<double>,               #TRAIT " must accept double");            \
    static_assert(TRAIT<std::complex<double>>, #TRAIT " must accept complex");           \
    static_assert(TRAIT<emul>,                 #TRAIT " must accept a custom Field");    \
    static_assert(!TRAIT<int>,                 #TRAIT " must reject int");               \
    static_assert(!TRAIT<std::int64_t>,        #TRAIT " must reject int64");

MTL5_ASSERT_PC_GATE(IdentityOk)
MTL5_ASSERT_PC_GATE(DiagonalOk)
MTL5_ASSERT_PC_GATE(BlockDiagonalOk)
MTL5_ASSERT_PC_GATE(SsorOk)
MTL5_ASSERT_PC_GATE(JacobiOk)
MTL5_ASSERT_PC_GATE(GaussSeidelOk)
MTL5_ASSERT_PC_GATE(SorOk)
MTL5_ASSERT_PC_GATE(Ic0Ok)
MTL5_ASSERT_PC_GATE(IldlOk)
MTL5_ASSERT_PC_GATE(Ilu0Ok)
MTL5_ASSERT_PC_GATE(IlutOk)

// The smoothers carry a compressed2D partial specialization with its own
// template head, so it can drift out of step with the primary. Checked
// separately rather than assumed to follow.
template <typename T>
concept JacobiSparseOk = requires { typename itl::smoother::jacobi<mat::compressed2D<T>>; };
template <typename T>
concept GaussSeidelSparseOk = requires { typename itl::smoother::gauss_seidel<mat::compressed2D<T>>; };
template <typename T>
concept SorSparseOk = requires { typename itl::smoother::sor<mat::compressed2D<T>>; };

MTL5_ASSERT_PC_GATE(JacobiSparseOk)
MTL5_ASSERT_PC_GATE(GaussSeidelSparseOk)
MTL5_ASSERT_PC_GATE(SorSparseOk)

// The four WRAPPER smoothers, which hold a gauss_seidel or sor member. A member
// being constrained does not constrain the enclosing template: naming
// backward_gauss_seidel<dense2D<int>> instantiates no member, so the inner gate
// never fires and the integer slipped straight through. Exactly the hole that
// block_diagonal has around lu_factor, and the reason each of these carries its
// own constraint rather than inheriting one (#505 review).
MTL5_PC_MATRIX_SHAPE(itl::smoother::backward_gauss_seidel,  BackwardGsOk)
MTL5_PC_MATRIX_SHAPE(itl::smoother::symmetric_gauss_seidel, SymmetricGsOk)
MTL5_PC_MATRIX_SHAPE(itl::smoother::backward_sor,           BackwardSorOk)
MTL5_PC_MATRIX_SHAPE(itl::smoother::symmetric_sor,          SymmetricSorOk)

MTL5_ASSERT_PC_GATE(BackwardGsOk)
MTL5_ASSERT_PC_GATE(SymmetricGsOk)
MTL5_ASSERT_PC_GATE(BackwardSorOk)
MTL5_ASSERT_PC_GATE(SymmetricSorOk)

} // namespace

TEST_CASE("every preconditioner and smoother requires a Field element type",
          "[itl][pc][smoother][concepts][field]") {
    // The claim is entirely in the static_asserts above. This case exists so the
    // check appears as a named ctest entry rather than only as a build that
    // happened to succeed.
    SUCCEED("all preconditioners and smoothers gate on Field");
}

TEST_CASE("the gated types still work on a real solve", "[itl][pc][smoother][field]") {
    // The negative half is compile-time; this is the positive half at RUNTIME --
    // a constraint that admitted a type but broke its behaviour would satisfy
    // every static_assert above.
    const std::size_t n = 8;
    mat::dense2D<double> A(n, n);
    for (std::size_t r = 0; r < n; ++r)
        for (std::size_t c = 0; c < n; ++c)
            A(r, c) = (r == c) ? 4.0 : (r + 1 == c || c + 1 == r ? -1.0 : 0.0);

    vec::dense_vector<double> b(n), x(n), y(n);
    for (std::size_t i = 0; i < n; ++i) { b(i) = 1.0 + 0.25 * static_cast<double>(i); x(i) = 0.0; }

    itl::pc::diagonal<mat::dense2D<double>> D(A);
    D.solve(x, b);
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(x(i) == b(i) / A(i, i));      // exact: the reciprocal diagonal

    itl::pc::identity<mat::dense2D<double>> I(A);
    I.solve(y, b);
    for (std::size_t i = 0; i < n; ++i)
        REQUIRE(y(i) == b(i));
}

TEST_CASE("a complex element type gets past naming and actually factorizes",
          "[itl][pc][field][complex]") {
    // `TRAIT<std::complex<double>>` above only proves the TYPE NAME is valid --
    // naming a specialization instantiates no member. ilut was the one type where
    // that gap was real: its drop tolerance was stored as Value, so
    // `abs(w[k]) < drop_tol` compared a double against a complex and factorize()
    // did not compile, while the static_assert happily claimed complex support
    // (#505 review). Nine of the ten were genuinely fine; this exercises the one
    // that was not, through the member that used to fail.
    using cx = std::complex<double>;
    const std::size_t n = 6;
    mat::compressed2D<cx> A(n, n);
    {
        mat::inserter<mat::compressed2D<cx>> ins(A, 3);
        for (std::size_t i = 0; i < n; ++i) {
            ins[i][i] << cx(4.0, 0.5);
            if (i + 1 < n) { ins[i][i + 1] << cx(-1.0, 0.25); ins[i + 1][i] << cx(-1.0, -0.25); }
        }
    }
    itl::pc::ilut<cx> M(A, 2, 1e-3);          // the threshold is REAL, not cx
    vec::dense_vector<cx> x(n), b(n);
    for (std::size_t i = 0; i < n; ++i) b(i) = cx(1.0, 0.0);
    M.solve(x, b);

    // A preconditioner is an approximate inverse, so assert it did something
    // finite and non-trivial rather than a specific value.
    for (std::size_t i = 0; i < n; ++i) {
        REQUIRE(std::isfinite(x(i).real()));
        REQUIRE(std::isfinite(x(i).imag()));
    }
    REQUIRE(std::abs(x(0)) > 0.0);
}
