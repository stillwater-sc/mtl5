#pragma once
// The shapes an nc-model session runs -- shared by the offline sweep and the
// timing harness (#479).
//
// IT IS A SHARED HEADER RATHER THAN A COPIED FUNCTION for one reason: the sweep
// nominates the shapes and the harness measures them, so if the two derivations
// ever drifted the session would time shapes the sweep never examined and report
// them against a disagreement count computed for different ones. Nothing would
// look wrong -- both files would be internally consistent -- and the CSVs would
// not be comparable. Same failure mode as #470's two-arms-that-were-one-arm,
// reached through duplication instead of inference.
//
// WHY THE LIST IS SEARCHED, NOT LISTED AND NOT COMPUTED. A fixed list silently
// tests nothing -- every measurement in #426 was square, the one regime where
// `jc_nt` is structurally 1 and the effect under study cannot appear. But a list
// computed from #430's algebra (`m <= mc * T/2`, `n > nc`) fails the same way for
// a subtler reason, and the sweep's first draft did exactly that: every shape it
// produced came back `jc_nt == 1`. The constraint is right and the consequence is
// not, because `plan_gemm_grid` caps `mc` at `ceil(m/budget)` (#441) and prefers
// the larger `ic_nt` on ties, so jc parallelism appears only near `m ~ mr * T`
// rather than `m ~ mc * T/2` -- on the Xeon E5-2420, m = 6, not m = 96.
//
// So this asks `plan_gemm_grid` which shapes actually produce a jc partition and
// keeps those. A derived list is no better than a fixed one if it is derived from
// the wrong model of the grid.

#include <mtl/detail/gemm_blocked.hpp>
#include <mtl/simd/blocking.hpp>

#include <cstddef>
#include <vector>

namespace mtl::bench {

/// One problem to plan or measure.
struct nc_point {
    std::size_t m, n, k;
    unsigned    threads;
};

/// Shapes derived from this machine's own blocking, by SEARCHING the grid.
///
/// Includes negative controls unconditionally, so the record shows they were
/// checked rather than leaving their absence to be assumed:
///   square    -- jc_nt structurally 1 on most machines
///   tall/thin -- ic-dominated
///   T = 1     -- every balancing model must be a no-op
inline std::vector<nc_point> derive_nc_shapes(const mtl::simd::blocking_params& bp,
                                              unsigned tmax) {
    using mtl::detail::plan_gemm_grid;
    std::vector<nc_point> v;
    const std::size_t k = 1024;
    if (tmax == 0) tmax = 1;

    std::vector<std::size_t> jc_m;
    for (std::size_t m = bp.mr; m <= bp.mc * tmax && jc_m.size() < 3; m += bp.mr) {
        const auto g = plan_gemm_grid(m, bp.nc * 8, bp.mc, bp.nc, bp.mr, tmax);
        if (g.jc_nt >= 2) jc_m.push_back(m);
    }
    for (std::size_t m : jc_m)
        for (std::size_t mult : {2u, 3u, 5u, 7u, 8u})
            v.push_back({m, bp.nc * mult, k, tmax});

    for (std::size_t s : {std::size_t{1024}, std::size_t{2048}})
        v.push_back({s, s, k, tmax});
    v.push_back({8192, 64, k, tmax});
    for (std::size_t mult : {2u, 5u})
        v.push_back({64, bp.nc * mult, k, 1u});

    return v;
}

} // namespace mtl::bench
