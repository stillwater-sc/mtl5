// MTL5 -- accumulator workspace scaling (#342).
//
// Several accumulator-aware kernels allocate a std::vector<Accumulator> whose
// length scales with the problem rather than with a register. That is free for
// configurations 1 and 2, where an accumulator is the size of the value type,
// but a configuration-3 super-accumulator (a Universal quire) is one to two
// orders of magnitude larger per element, so the workspace dominates.
//
// The decision recorded in accumulator_traits.hpp is to accept that cost and
// let the caller bound it through the accumulator's capacity bits, rather than
// restructure kernels for a type that cannot yet be instantiated here. These
// tests pin the scaling that decision rests on: if a kernel later grew a
// superlinear workspace, the policy would silently become wrong, and the
// footprint numbers quoted in the header and in #342 would be wrong with it.
//
// The measurement works by counting live instances of a stand-in accumulator,
// so it observes the real kernels without instrumenting library code.
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/math/accumulator_traits.hpp>

namespace {

/// Behaves like a double, but records the peak number of simultaneously live
/// instances. Every allocation a kernel makes is therefore visible.
struct counting_acc {
    double v{};
    static inline long long live = 0;
    static inline long long peak = 0;

    counting_acc()                        { bump(); }
    counting_acc(const counting_acc& o) : v(o.v) { bump(); }
    counting_acc(counting_acc&& o) noexcept : v(o.v) { bump(); }
    counting_acc& operator=(const counting_acc&) = default;
    counting_acc& operator=(counting_acc&&) noexcept = default;
    ~counting_acc() { --live; }

    static void bump() { ++live; peak = std::max(peak, live); }
    static void reset() { live = 0; peak = 0; }
};

}  // namespace

namespace mtl::math {
template <typename Value>
struct accumulator_traits<counting_acc, Value> {
    using Acc = counting_acc;
    static void clear(Acc& a) { a.v = 0.0; }
    static void assign(Acc& a, const Value& x) { a.v = static_cast<double>(x); }
    template <typename Result = Value>
    static Result value(const Acc& a) { return static_cast<Result>(a.v); }
    static void add_product(Acc& a, const Value& m, const Value& x) {
        a.v += static_cast<double>(m) * static_cast<double>(x);
    }
};
}  // namespace mtl::math

#include <mtl/sparse/factorization/sparse_lu.hpp>
#include <mtl/sparse/factorization/supernodal_ldlt.hpp>
#include <mtl/sparse/ordering/amd.hpp>
#include <mtl/sparse/ordering/colamd.hpp>

namespace {

mtl::mat::compressed2D<double> laplacian_2d(std::size_t g) {
    const std::size_t n = g * g;
    mtl::mat::compressed2D<double> A(n, n);
    mtl::mat::inserter<mtl::mat::compressed2D<double>> ins(A);
    auto id = [g](std::size_t r, std::size_t c) { return r * g + c; };
    for (std::size_t r = 0; r < g; ++r)
        for (std::size_t c = 0; c < g; ++c) {
            const std::size_t i = id(r, c);
            ins[i][i] << 4.0;
            if (r + 1 < g) { ins[i][id(r+1,c)] << -1.0; ins[id(r+1,c)][i] << -1.0; }
            if (c + 1 < g) { ins[i][id(r,c+1)] << -1.0; ins[id(r,c+1)][i] << -1.0; }
        }
    return A;
}

}  // namespace

TEST_CASE("sparse_lu accumulator workspace is O(n) (#342)", "[sparse][accumulator][workspace]") {
    namespace fact = mtl::sparse::factorization;
    namespace ord  = mtl::sparse::ordering;

    for (std::size_t g : {20u, 40u}) {
        const auto A = laplacian_2d(g);
        const std::size_t n = g * g;

        counting_acc::reset();
        {
            auto num = fact::sparse_lu_numeric<counting_acc>(
                A, fact::sparse_lu_symbolic(A, ord::colamd{}));
            (void)num;
        }

        INFO("n = " << n << ", peak accumulators = " << counting_acc::peak);
        REQUIRE(counting_acc::peak > 0);
        // Measured exactly n. A small multiple is allowed so incidental
        // temporaries do not make this brittle; the point is that it is linear,
        // not that it is exactly 1.0.
        REQUIRE(counting_acc::peak <= static_cast<long long>(4 * n));
    }
}

TEST_CASE("supernodal_ldlt accumulator workspace stays linear in n (#342)",
          "[sparse][accumulator][workspace]") {
    namespace fact = mtl::sparse::factorization;
    namespace ord  = mtl::sparse::ordering;

    for (std::size_t g : {20u, 40u}) {
        const auto A = laplacian_2d(g);
        const std::size_t n = g * g;

        counting_acc::reset();
        {
            auto num = fact::supernodal_ldlt_numeric<counting_acc>(
                A, fact::supernodal_ldlt_symbolic(A, ord::amd{}));
            (void)num;
        }

        INFO("n = " << n << ", peak accumulators = " << counting_acc::peak);
        REQUIRE(counting_acc::peak > 0);
        // The dense panel is m*w per supernode, so the multiplier tracks
        // supernode structure (measured 1.22-1.92 x n on this family) and is
        // matrix-dependent. The invariant the policy needs is that it does not
        // become superlinear.
        REQUIRE(counting_acc::peak <= static_cast<long long>(8 * n));
    }
}
