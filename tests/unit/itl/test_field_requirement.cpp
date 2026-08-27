// Every Krylov solver requires a Field element type (#503).
//
// The solvers were unconstrained templates, so `dense_vector<int>` instantiated
// all of them -- and each computes its step lengths as ratios in the element
// type (`cg`'s beta = rho/rho_1 and alpha = rho/dot(p,q); eight such ratios in
// qmr). On an integral type that division truncates, so the iteration was
// nonsense from the first step and returned a confident wrong answer with no
// assertion and no flag. Same defect as `lu_factor(dense2D<int>)`, which #430
// fixed with FieldMatrix; this is the vector counterpart, one level up.
//
// WHY THIS IS NOT TEN compile_fail FILES. The convention in tests/unit/
// compile_fail/ is one translation unit per case, each a full compile driven by
// ctest -- ten of those to make one point is a lot of CI for little extra
// information. A requires-expression answers the same question at compile time
// for every solver in a single cheap TU: it runs overload resolution and checks
// the constraint WITHOUT instantiating the body, so it costs almost nothing and
// covers all eleven constrained entry points -- the ten public solvers plus
// gmres_inner. compile_fail/cg_integer_vector.cpp still exists alongside it,
// for the one thing this file cannot check -- that the DIAGNOSTIC names the
// reason rather than merely failing.
//
// The positive half matters as much as the negative: a constraint that rejects
// everything would pass a rejection-only test. So each solver is also asserted
// CALLABLE for double, for complex<double>, and for a custom Field type standing
// in for a Universal format -- the types this library exists to serve.
#include <catch2/catch_test_macros.hpp>

#include <complex>
#include <cstdint>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/concepts/vector.hpp>
#include <mtl/itl/iteration/basic_iteration.hpp>
#include <mtl/itl/pc/identity.hpp>
#include <mtl/itl/krylov/cg.hpp>
#include <mtl/itl/krylov/bicg.hpp>
#include <mtl/itl/krylov/bicgstab.hpp>
#include <mtl/itl/krylov/bicgstab_ell.hpp>
#include <mtl/itl/krylov/cgs.hpp>
#include <mtl/itl/krylov/gmres.hpp>
#include <mtl/itl/krylov/idr_s.hpp>
#include <mtl/itl/krylov/minres.hpp>
#include <mtl/itl/krylov/qmr.hpp>
#include <mtl/itl/krylov/tfqmr.hpp>

using namespace mtl;

namespace {

/// Minimal custom Field type, standing in for a posit / cfloat / LNS: a class
/// type (so no hardware path claims it) that divides.
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

static_assert(Field<emul>,          "the stand-in must actually be a Field");
static_assert(Field<double>);
static_assert(Field<float>);
static_assert(Field<std::complex<double>>);
static_assert(!Field<int>,          "integers are a ring, not a field");
static_assert(!Field<std::int32_t>);
static_assert(!Field<std::int64_t>);

static_assert(FieldVector<vec::dense_vector<double>>);
static_assert(FieldVector<vec::dense_vector<std::complex<double>>>);
static_assert(FieldVector<vec::dense_vector<emul>>);
static_assert(!FieldVector<vec::dense_vector<int>>);
static_assert(!FieldVector<vec::dense_vector<std::int64_t>>);

// Each trait below asks "is this solver callable with element type T?" -- the
// call is never evaluated and the body never instantiated, so only the template
// constraint decides.
//
// TWO MACROS RATHER THAN ONE VARIADIC, because four of the entry points take a
// trailing tuning argument -- gmres a restart, idr_s an s, bicgstab_ell an ell,
// and detail::gmres_inner a kmax -- while the rest take none, and the obvious
// `__VA_OPT__` spelling does not compile on MSVC without
// /Zc:preprocessor -- MSVC's default preprocessor is not conforming, and adding
// a build flag to accommodate a test is the wrong way round. Nothing else in the
// project uses __VA_OPT__, so this stays a portability rule the project already
// follows rather than a new exception.
#define MTL5_SOLVER_CALLABLE(NAME, TRAIT)                                       \
    template <typename T>                                                       \
    concept TRAIT = requires(mat::dense2D<T>& A, vec::dense_vector<T>& x,       \
                             const vec::dense_vector<T>& b,                     \
                             itl::pc::identity<mat::dense2D<T>>& M,             \
                             itl::basic_iteration<T>& it) {                     \
        itl::NAME(A, x, b, M, it);                                              \
    };

#define MTL5_SOLVER_CALLABLE_ARG(NAME, TRAIT, EXTRA)                            \
    template <typename T>                                                       \
    concept TRAIT = requires(mat::dense2D<T>& A, vec::dense_vector<T>& x,       \
                             const vec::dense_vector<T>& b,                     \
                             itl::pc::identity<mat::dense2D<T>>& M,             \
                             itl::basic_iteration<T>& it) {                     \
        itl::NAME(A, x, b, M, it, EXTRA);                                       \
    };

MTL5_SOLVER_CALLABLE(cg,       CgOk)
MTL5_SOLVER_CALLABLE(bicg,     BicgOk)
MTL5_SOLVER_CALLABLE(bicgstab, BicgstabOk)
MTL5_SOLVER_CALLABLE(cgs,      CgsOk)
MTL5_SOLVER_CALLABLE(minres,   MinresOk)
MTL5_SOLVER_CALLABLE(qmr,      QmrOk)
MTL5_SOLVER_CALLABLE(tfqmr,    TfqmrOk)
MTL5_SOLVER_CALLABLE_ARG(gmres,        GmresOk,       30)
MTL5_SOLVER_CALLABLE_ARG(idr_s,        IdrsOk,        4)
MTL5_SOLVER_CALLABLE_ARG(bicgstab_ell, BicgstabEllOk, 2)
// gmres_inner is the eleventh constrained entry point. It is in itl::detail
// rather than itl, but it is a real gate that can drift independently of the
// gmres wrapper around it, so it is covered here too -- the macro qualifies with
// `itl::`, so the detail:: prefix composes.
MTL5_SOLVER_CALLABLE_ARG(detail::gmres_inner, GmresInnerOk, 30)

#define MTL5_ASSERT_SOLVER_GATE(TRAIT)                                          \
    static_assert(TRAIT<float>,                     #TRAIT " must accept float");         \
    static_assert(TRAIT<double>,                    #TRAIT " must accept double");        \
    static_assert(TRAIT<std::complex<double>>,      #TRAIT " must accept complex");       \
    static_assert(TRAIT<emul>,                      #TRAIT " must accept a custom Field");\
    static_assert(!TRAIT<int>,                      #TRAIT " must reject int");           \
    static_assert(!TRAIT<std::int64_t>,             #TRAIT " must reject int64");

MTL5_ASSERT_SOLVER_GATE(CgOk)
MTL5_ASSERT_SOLVER_GATE(BicgOk)
MTL5_ASSERT_SOLVER_GATE(BicgstabOk)
MTL5_ASSERT_SOLVER_GATE(CgsOk)
MTL5_ASSERT_SOLVER_GATE(MinresOk)
MTL5_ASSERT_SOLVER_GATE(QmrOk)
MTL5_ASSERT_SOLVER_GATE(TfqmrOk)
MTL5_ASSERT_SOLVER_GATE(GmresOk)
MTL5_ASSERT_SOLVER_GATE(IdrsOk)
MTL5_ASSERT_SOLVER_GATE(BicgstabEllOk)
MTL5_ASSERT_SOLVER_GATE(GmresInnerOk)

} // namespace

TEST_CASE("every Krylov solver requires a Field element type", "[itl][concepts][field]") {
    // The claim is entirely in the static_asserts above -- if this TU compiled,
    // all eleven constrained entry points reject the integers and accept float,
    // double, complex and a custom Field. This case exists so the check appears
    // as a named ctest entry rather than only as a build that happened to
    // succeed.
    SUCCEED("all eleven constrained entry points gate on FieldVector");
}
