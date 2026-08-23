#pragma once
// MTL5 -- the candidate `nc` models, as pure functions (#479, feeding #429).
//
// `nc` sets the jc BLOCK COUNT, `njb = ceil(n / nc)`, and the threaded nest
// hands those blocks to `jc_nt` teams round-robin. So `nc` decides how evenly
// the n dimension divides across teams, and a model that sizes it purely for
// cache capacity can leave the partition ragged -- which is the ic-side defect
// #408 measured at 1.41x critical path, on a loop that now has `balanced_mc`
// and a jc loop that has nothing.
//
// SIX MODELS, NOT ONE HYPOTHESIS. They are here as pure functions so the
// disagreement between them can be enumerated OFFLINE -- for a given machine's
// blocking parameters, most shapes give every model the same answer, and only
// the shapes where they differ are worth spending machine time on. That is what
// makes a six-model sweep across four machines affordable at all.
//
// NONE OF THESE IS WIRED INTO THE NEST. `gemm_blocked` still uses M0. Choosing
// between them is #479's job and needs >= 3 microarchitectures; applying the
// winner is #429. Shipping one unmeasured would repeat #426, where runtime cache
// detection was merged on the assumption that real sizes must beat constants
// tuned for a Haswell core, and then lost on every machine that ran it.
//
// M3 is included despite having been REJECTED on #426 as modelling the wrong
// cause -- the regression there was block-count imbalance, not capacity. It is
// cheap to evaluate, and the data should settle it rather than the argument.

#include <cstddef>

#include <mtl/detail/gemm_blocked.hpp>   // balanced_nc

namespace mtl::detail {

/// Candidate models for sizing `nc`. See the file header for why each exists.
enum class nc_model {
    m0_default,     ///< today: compile-time L3, no balancing. The baseline.
    m1_balanced,    ///< M0 + balanced_nc
    m2_detected,    ///< detected L3 + balanced_nc
    m3_per_team,    ///< (detected L3 / jc_nt) + balanced -- rejected on #426, measured anyway
    m4_per_sharer,  ///< (detected L3 / L3 sharing cores) + balanced
    m5_exact,       ///< pick nc so njb == jc_nt exactly, where n allows
};

/// Everything a model needs. Kept as one struct so adding a model cannot
/// silently change a signature the sweep and the harness both call.
struct nc_model_inputs {
    std::size_t n;                   ///< the problem's n
    std::size_t kc;                  ///< as the nest will step it
    std::size_t nr;                  ///< register tile width; nc stays a multiple
    std::size_t sdata;               ///< sizeof(element)
    std::size_t l3_default_bytes;    ///< hw_traits' compile-time figure
    std::size_t l3_detected_bytes;   ///< cache_info's, 0 if undetected
    std::size_t l3_sharing_cores;    ///< physical cores sharing one L3, 0 if unknown
    unsigned    jc_nt;               ///< teams across jc, from a first grid pass
};

/// The capacity-only part: `nc` such that the packed B panel `kc x nc` fills the
/// budget, rounded down to a whole number of `nr`-column panels. This is
/// `derive_blocking`'s own formula, factored out so every model shares it.
constexpr std::size_t nc_from_budget(std::size_t budget_bytes, std::size_t kc,
                                     std::size_t sdata, std::size_t nr) noexcept {
    if (kc == 0 || sdata == 0 || nr == 0) return nr ? nr : 1;
    const std::size_t cap = budget_bytes / (kc * sdata);
    const std::size_t nc = (cap / nr) * nr;
    return nc == 0 ? nr : nc;
}

/// `nc` under one model. Pure, so the offline sweep and the benchmark arm cannot
/// disagree about what a model means.
///
/// A model that would need a figure the machine did not report FALLS BACK to the
/// compile-time L3 rather than to zero. An undetectable machine must behave
/// exactly as M0 does, not produce a degenerate block size -- the same rule
/// `with_detected_caches` follows field by field.
inline std::size_t nc_for_model(nc_model model, const nc_model_inputs& in) noexcept {
    const std::size_t l3_det = in.l3_detected_bytes ? in.l3_detected_bytes
                                                    : in.l3_default_bytes;
    const unsigned    teams  = in.jc_nt ? in.jc_nt : 1u;
    const std::size_t shar   = in.l3_sharing_cores ? in.l3_sharing_cores : 1u;

    switch (model) {
        case nc_model::m0_default:
            return nc_from_budget(in.l3_default_bytes, in.kc, in.sdata, in.nr);

        case nc_model::m1_balanced:
            return balanced_nc(in.n, nc_from_budget(in.l3_default_bytes, in.kc, in.sdata, in.nr),
                               teams, in.nr);

        case nc_model::m2_detected:
            return balanced_nc(in.n, nc_from_budget(l3_det, in.kc, in.sdata, in.nr),
                               teams, in.nr);

        case nc_model::m3_per_team:
            return balanced_nc(in.n, nc_from_budget(l3_det / teams, in.kc, in.sdata, in.nr),
                               teams, in.nr);

        case nc_model::m4_per_sharer:
            return balanced_nc(in.n, nc_from_budget(l3_det / shar, in.kc, in.sdata, in.nr),
                               teams, in.nr);

        case nc_model::m5_exact: {
            // Exactly one jc block per team, when n is large enough to supply
            // them AND the result still fits the detected budget. Where it does
            // not fit, fall back to M2 rather than overflowing L3 -- an "exact"
            // partition that thrashes is not the thing being tested.
            const std::size_t cap = nc_from_budget(l3_det, in.kc, in.sdata, in.nr);
            if (teams <= 1 || in.n == 0) return cap;
            std::size_t nc = (in.n + teams - 1) / teams;         // njb == teams
            if (in.nr > 1) nc = ((nc + in.nr - 1) / in.nr) * in.nr;
            if (nc == 0 || nc > cap) return balanced_nc(in.n, cap, teams, in.nr);
            return nc;
        }
    }
    return nc_from_budget(in.l3_default_bytes, in.kc, in.sdata, in.nr);
}

/// Short stable token for CSV columns and command lines.
constexpr const char* nc_model_name(nc_model m) noexcept {
    switch (m) {
        case nc_model::m0_default:    return "m0_default";
        case nc_model::m1_balanced:   return "m1_balanced";
        case nc_model::m2_detected:   return "m2_detected";
        case nc_model::m3_per_team:   return "m3_per_team";
        case nc_model::m4_per_sharer: return "m4_per_sharer";
        case nc_model::m5_exact:      return "m5_exact";
    }
    return "unknown";
}

inline constexpr nc_model all_nc_models[] = {
    nc_model::m0_default, nc_model::m1_balanced, nc_model::m2_detected,
    nc_model::m3_per_team, nc_model::m4_per_sharer, nc_model::m5_exact,
};

} // namespace mtl::detail
