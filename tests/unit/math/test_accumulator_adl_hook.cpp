// MTL5 -- the ADL hook for super-accumulators (configuration 3).
//
// A number library supplies an exact accumulator by defining free functions
// next to the accumulator type (see `mtl::math::adl_accumulator`), so it never
// includes MTL5 and every downstream repository shares one definition instead of
// carrying its own accumulator_traits specialization. The accumulator here is a
// stand-in for a Universal quire: it lives in its own namespace, holds the sum
// in double (exact for these float fixtures), and counts hook calls so the tests
// can tell the hook path from the plain one.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <vector>

#include <mtl/math/accumulator_traits.hpp>
#include <mtl/operation/dot.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/sparse/factorization/sparse_cholesky.hpp>
#include <mtl/sparse/factorization/sparse_ldlt.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>
#include <mtl/sparse/factorization/sparse_qr.hpp>
#include <mtl/sparse/factorization/supernodal_ldlt.hpp>
#include <mtl/sparse/factorization/supernodal_lu.hpp>
#include <mtl/sparse/ordering/amd.hpp>

namespace numlib {

int g_hook_calls = 0;

// The accumulator a number library would ship. The hook functions are
// constrained to the element type the accumulator is for, as the contract
// recommends, so they cannot claim an unrelated accumulator_traits<A, V>.
struct exact_acc {
    double sum;
};

inline void accumulator_clear(exact_acc& a) { ++g_hook_calls; a.sum = 0.0; }
template <std::same_as<float> F>
void accumulator_assign(exact_acc& a, const F& v) {
    ++g_hook_calls;
    a.sum = static_cast<double>(v);
}
template <std::same_as<float> F>
void accumulator_add_product(exact_acc& a, const F& m, const F& v) {
    ++g_hook_calls;
    a.sum += static_cast<double>(m) * static_cast<double>(v);   // exact product
}
template <typename Result>
Result accumulator_value(const exact_acc& a, std::type_identity<Result>) {
    ++g_hook_calls;
    return static_cast<Result>(a.sum);
}
inline void accumulator_merge(exact_acc& a, const exact_acc& b) {
    ++g_hook_calls;
    a.sum += b.sum;
}

// The same accumulator without accumulator_merge: still a valid hook, but
// merge() is unavailable rather than silently falling back to something else.
struct mergeless_acc {
    double sum;
};
inline void accumulator_clear(mergeless_acc& a) { a.sum = 0.0; }
inline void accumulator_assign(mergeless_acc& a, const float& v) { a.sum = v; }
inline void accumulator_add_product(mergeless_acc& a, const float& m, const float& v) {
    a.sum += static_cast<double>(m) * static_cast<double>(v);
}
template <typename Result>
Result accumulator_value(const mergeless_acc& a, std::type_identity<Result>) {
    return static_cast<Result>(a.sum);
}

// An accumulator that provides the hook AND has an explicit specialization, as
// a pre-existing bridge header would: the specialization must win.
template <int N>
struct bridged_acc {
    double sum;
};
template <int N>
void accumulator_clear(bridged_acc<N>& a) { a.sum = 0.0; }
template <int N>
void accumulator_assign(bridged_acc<N>& a, const float& v) { a.sum = v; }
template <int N>
void accumulator_add_product(bridged_acc<N>& a, const float& m, const float& v) {
    a.sum += static_cast<double>(m) * static_cast<double>(v);
}
template <int N, typename Result>
Result accumulator_value(const bridged_acc<N>& a, std::type_identity<Result>) {
    return static_cast<Result>(a.sum);
}

} // namespace numlib

namespace mtl::math {
template <int N>
struct accumulator_traits<numlib::bridged_acc<N>, float> {
    static constexpr bool from_bridge = true;
    using Acc = numlib::bridged_acc<N>;
    static void clear(Acc& a) { a.sum = 0.0; }
    static void assign(Acc& a, const float& v) { a.sum = v; }
    template <typename Result = float>
    static Result value(const Acc& a) { return static_cast<Result>(a.sum); }
    static void add_product(Acc& a, const float& m, const float& v) { a.sum += m * v; }
};
} // namespace mtl::math

using mtl::math::accumulator_traits;
using mtl::math::adl_accumulator;
using numlib::exact_acc;

template <typename AT, typename Acc>
concept has_merge = requires(Acc& a, const Acc& b) { AT::merge(a, b); };

TEST_CASE("adl_accumulator: which accumulators provide the hook",
          "[math][accumulator]") {
    STATIC_REQUIRE(adl_accumulator<exact_acc, float>);
    STATIC_REQUIRE(adl_accumulator<numlib::mergeless_acc, float>);
    // The hook is constrained to float elements; a double element does not match.
    STATIC_REQUIRE_FALSE(adl_accumulator<exact_acc, double>);
    // Configurations 1 and 2 are untouched by the hook.
    STATIC_REQUIRE_FALSE(adl_accumulator<double, float>);
    STATIC_REQUIRE_FALSE(adl_accumulator<float, float>);
    STATIC_REQUIRE_FALSE(adl_accumulator<mtl::math::fma_accumulator<double>, float>);
}

TEST_CASE("accumulator_traits forwards to the ADL hook", "[math][accumulator]") {
    using AT = accumulator_traits<exact_acc, float>;
    numlib::g_hook_calls = 0;
    exact_acc a{42.0};
    AT::clear(a);                          REQUIRE(a.sum == 0.0);
    AT::assign(a, 1.0f);                   REQUIRE(a.sum == 1.0);
    AT::add_product(a, -1e8f, 1.0f);
    AT::add_product(a, 1e8f, 1.0f);
    // 1 - 1e8 + 1e8 cancels to 0 in float; the exact accumulator keeps the 1.
    REQUIRE(AT::value(a) == 1.0f);
    REQUIRE(AT::value<double>(a) == 1.0);
    STATIC_REQUIRE(std::is_same_v<decltype(AT::value(a)), float>);
    STATIC_REQUIRE(std::is_same_v<decltype(AT::value<double>(a)), double>);
    REQUIRE(numlib::g_hook_calls == 6);
}

TEST_CASE("accumulator_traits::merge combines partial accumulators",
          "[math][accumulator]") {
    SECTION("ADL hook") {
        using AT = accumulator_traits<exact_acc, float>;
        exact_acc lo{}, hi{};
        AT::clear(lo); AT::clear(hi);
        AT::add_product(lo, 1.0f, 1.0f);
        AT::add_product(lo, -1e8f, 1.0f);
        AT::add_product(hi, 1e8f, 1.0f);
        AT::merge(lo, hi);
        REQUIRE(AT::value(lo) == 1.0f);
    }
    SECTION("an accumulator without accumulator_merge has no merge()") {
        STATIC_REQUIRE(has_merge<accumulator_traits<exact_acc, float>, exact_acc>);
        STATIC_REQUIRE_FALSE(has_merge<accumulator_traits<numlib::mergeless_acc, float>,
                                       numlib::mergeless_acc>);
    }
    SECTION("configuration 1: plain add") {
        using AT = accumulator_traits<double, float>;
        double a = 0.0, b = 0.0;
        AT::add_product(a, 2.0f, 3.0f);
        AT::add_product(b, 4.0f, 5.0f);
        AT::merge(a, b);
        REQUIRE(a == 26.0);
    }
    SECTION("configuration 2: fma_accumulator") {
        using Acc = mtl::math::fma_accumulator<double>;
        using AT  = accumulator_traits<Acc, float>;
        Acc a{}, b{};
        AT::add_product(a, 2.0f, 3.0f);
        AT::add_product(b, 4.0f, 5.0f);
        AT::merge(a, b);
        REQUIRE(AT::value<double>(a) == 26.0);
    }
}

TEST_CASE("an explicit specialization still wins over the ADL hook",
          "[math][accumulator]") {
    STATIC_REQUIRE(adl_accumulator<numlib::bridged_acc<1>, float>);
    STATIC_REQUIRE(accumulator_traits<numlib::bridged_acc<1>, float>::from_bridge);
    // The sparse layer's inheriting alias resolves the same way.
    STATIC_REQUIRE(std::is_base_of_v<
        accumulator_traits<exact_acc, float>,
        mtl::sparse::factorization::accumulator_traits<exact_acc, float>>);
}

TEST_CASE("dot<Acc, Result> accumulates through the ADL hook", "[math][accumulator]") {
    mtl::vec::dense_vector<float> x(3), y(3, 1.0f);
    x(0) = 1.0f; x(1) = -1e8f; x(2) = 1e8f;
    numlib::g_hook_calls = 0;
    const float exact = mtl::dot<exact_acc, float>(x, y);
    REQUIRE(exact == 1.0f);
    REQUIRE(numlib::g_hook_calls > 0);
    // The float accumulator loses the 1 to cancellation.
    REQUIRE(mtl::dot<float, float>(x, y) == 0.0f);
}

namespace {

// 5-point 2-D Laplacian on a g x g grid: SPD, real fill and supernodes.
mtl::mat::compressed2D<float> laplacian2d(std::size_t g) {
    const std::size_t n = g * g;
    mtl::mat::compressed2D<float> A(n, n);
    mtl::mat::inserter<mtl::mat::compressed2D<float>> ins(A);
    auto id = [g](std::size_t r, std::size_t c) { return r * g + c; };
    for (std::size_t r = 0; r < g; ++r)
        for (std::size_t c = 0; c < g; ++c) {
            const std::size_t i = id(r, c);
            ins[i][i] << 4.0f;
            if (r + 1 < g) { ins[i][id(r + 1, c)] << -1.0f; ins[id(r + 1, c)][i] << -1.0f; }
            if (c + 1 < g) { ins[i][id(r, c + 1)] << -1.0f; ins[id(r, c + 1)][i] << -1.0f; }
        }
    return A;
}

} // namespace

TEST_CASE("sparse direct solvers factor and solve through the ADL hook",
          "[math][accumulator][sparse]") {
    using namespace mtl::sparse::factorization;
    const auto A = laplacian2d(5);
    const std::size_t n = A.num_cols();
    mtl::vec::dense_vector<float> b(n);
    for (std::size_t i = 0; i < n; ++i) b(i) = static_cast<float>(1 + i % 3);

    auto check = [&](auto&& solve_default, auto&& solve_hook) {
        mtl::vec::dense_vector<float> d(n, 0.f), h(n, 0.f);
        solve_default(d);
        numlib::g_hook_calls = 0;
        solve_hook(h);
        REQUIRE(numlib::g_hook_calls > 0);
        for (std::size_t i = 0; i < n; ++i)
            REQUIRE(std::abs(h(i) - d(i)) <= 1e-5f * std::abs(d(i)) + 1e-6f);
    };

    check([&](auto& x) { sparse_cholesky_solve(A, x, b); },
          [&](auto& x) { sparse_cholesky_solve<exact_acc>(A, x, b); });
    check([&](auto& x) { sparse_ldlt_solve(A, x, b); },
          [&](auto& x) { sparse_ldlt_solve<exact_acc>(A, x, b); });
    check([&](auto& x) { sparse_lu_solve(A, x, b); },
          [&](auto& x) { sparse_lu_solve<exact_acc>(A, x, b); });
    check([&](auto& x) { sparse_qr_solve(A, x, b); },
          [&](auto& x) { sparse_qr_solve<exact_acc>(A, x, b); });
    check([&](auto& x) { supernodal_ldlt_solve(A, x, b); },
          [&](auto& x) { supernodal_ldlt_solve<exact_acc>(A, x, b); });
    check([&](auto& x) { supernodal_lu_solve(A, x, b, mtl::sparse::ordering::amd{}); },
          [&](auto& x) { supernodal_lu_solve<exact_acc>(A, x, b, mtl::sparse::ordering::amd{}); });
}
