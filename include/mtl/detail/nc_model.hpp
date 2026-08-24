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
// SEVEN MODELS, NOT ONE HYPOTHESIS. They are here as pure functions so the
// disagreement between them can be enumerated OFFLINE -- for a given machine's
// blocking parameters, most shapes give every model the same answer, and only
// the shapes where they differ are worth spending machine time on. That is what
// makes a seven-model sweep across four machines affordable at all.
//
// THE DEFAULT IS NOW M6 (#429), AND THAT IS A MEASURED CHOICE. #479 timed all
// six candidates on four microarchitectures and two dtypes -- 840 arms, every
// one verified bit-identical to the baseline. What it found:
//
//   M6 = M1 + guard  DEFAULT. 42 of 44 arms taken, median x1.161, worst
//                    x1.0002, zero regressions. See `nc_for_plan`.
//   M1 balanced      median x1.152 but TWO regressions to x0.82-0.89, both on
//                    a Zen 4 whose 16 MB L3 cannot hold the grown packed-B set.
//   M2 detected L3   FALSIFIED. Up to 45% slower (x0.548), 37 regressions --
//                    and it already includes balanced_nc, so balancing does not
//                    rescue it. #426's lesson a second time: a larger cache
//                    block is not a faster one. This is why the tripwire in
//                    test_cache_info.cpp asserting l3 is NOT applied stays.
//   M5 exact         Same failure as M2, x0.548 worst, 39 regressions.
//   M4 per-sharer    The LARGEST wins measured anywhere here, median x1.763 --
//                    but confined to the jc-parallel shapes the sweep selects,
//                    and it costs ~1.2% on SQUARE GEMM, the common case.
//   M3 per-team      median x1.223, taxes T=1.
//
// M3 and M4 were rejected on #426 as "modelling the wrong cause -- the
// regression is block-count imbalance, not capacity". The measurement reverses
// that: M4's speedup does NOT track imbalance (r = 0.29) and does track `nc`
// being reduced. Capacity was not the wrong cause; on those shapes it is the
// dominant one. They are not the default only because they do not generalise.
//
// M0 IS KEPT AS THE PRE-#429 BASELINE so the experiment stays reproducible: an
// arm labelled "the old behaviour" has to BE the old behaviour, not a
// reconstruction of it. `MTL5_NC_MODEL=m0_default` restores it exactly.

#include <cstddef>
#include <cstdio>    // fprintf, for the selector's refusal message
#include <cstdlib>   // getenv/abort, for the harness arm's selector
#include <cstring>

namespace mtl::detail {

/// Choose the jc-block size that balances the threaded n-partition -- the jc-side
/// counterpart of `balanced_mc` (#479, feeding #429).
///
/// `nc_max` is an UPPER bound set by the L3 budget, so any smaller value is
/// still cache-legal and we are free to trade block size for a partition that
/// divides evenly across the teams. The threaded nest hands jc-blocks to the
/// `jc_nt` teams round-robin, so the most-loaded team gets `ceil(njb / jc_nt)`
/// blocks and the critical path is minimised when `njb` is a MULTIPLE of
/// `jc_nt`. This picks the largest `nc <= nc_max` achieving that.
///
/// WHY THE ic SIDE GOT THIS FIRST. `balanced_mc` exists because #408 measured
/// what its absence costs: an mc that moved 64 -> 60 took nib from 16 (exactly
/// 2.00 blocks per thread on 8 threads) to 18 (2.25), a 1.41x critical path that
/// turned a +21.5% single-thread win into a -7.4% eight-thread regression. The
/// jc loop has the same arithmetic and went without the repair for longer; this
/// is that repair, default since #429/#490 as `m6_guarded`.
///
/// IT DOES NOT LET THE DETECTED L3 BACK IN, and the reason is worth stating
/// because the old version of this comment implied the opposite. `nc` was
/// withheld from cache detection on the theory that an unbalanced partition was
/// the only hazard. #479 measured the pairing: detected L3 WITH balancing (M2)
/// still loses up to 45%, 37 regressions. Balancing does not rescue it, so the
/// theory was wrong rather than merely untested -- and `with_detected_caches`
/// keeps pinning `nc` to the compile-time derivation. See blocking.hpp.
///
/// IT LIVES HERE, NOT IN gemm_blocked.hpp, because it exists FOR the models
/// below: every one of M1..M6 is "some capacity budget, then this". Keeping it
/// beside them also breaks the include cycle the harness arm would otherwise
/// need -- `gemm_blocked` now includes this header to reach `nc_for_model`, so
/// this header can no longer include `gemm_blocked`.
///
/// Returns `nc_max` unchanged for the serial case, where there is nothing to
/// balance.
///
/// IT DOES NOT ROUND THE BALANCED `nc` TO A MULTIPLE OF `nr`, and that omission
/// is the whole lesson of #408 restated on this axis. Rounding `nc` after
/// balancing changes `njb` again and destroys the property just established:
/// the first draft of this function did round, and the pre-registered control
/// in test_nc_models caught it -- `m4_per_sharer` at n = 276681, jc_nt = 4 went
/// from a perfectly even partition to 1.0038, i.e. the balancing step made the
/// imbalance WORSE. That is #408's defect exactly: a register-tile quantity
/// perturbing a partition quantity.
///
/// `balanced_mc` takes the same position for the same reason -- it rounds to
/// whole `mr` panels only on the SERIAL path, where there is no partition to
/// perturb. A multiple of `nr` is an optimisation, not a requirement: the nest
/// computes `npanels = ceil(nci / NR)` and the micro-kernel's edge path already
/// handles a ragged final panel, since `n` is not a multiple of `nc` in general.
inline std::size_t balanced_nc(std::size_t n, std::size_t nc_max, unsigned jc_nt,
                               std::size_t nr = 1) {
    if (jc_nt <= 1 || n == 0 || nc_max == 0) {
        // Serial: nothing to balance, so take the largest cache-legal block and
        // round it to whole nr-column panels -- harmless here, and it removes the
        // ragged panel from every jc block rather than only the last.
        if (nr > 1 && nc_max >= nr) return (nc_max / nr) * nr;
        return nc_max;
    }
    std::size_t njb = (n + nc_max - 1) / nc_max;        // fewest blocks the bound allows
    njb = ((njb + jc_nt - 1) / jc_nt) * jc_nt;          // round up to a multiple of jc_nt
    if (njb == 0) return nc_max;
    const std::size_t nc = (n + njb - 1) / njb;         // cols per block; <= nc_max
    return nc == 0 ? nc_max : nc;
}

/// Candidate models for sizing `nc`. See the file header for why each exists.
enum class nc_model {
    m0_default,     ///< pre-#429: compile-time L3, no balancing. The baseline.
    m1_balanced,    ///< M0 + balanced_nc
    m2_detected,    ///< detected L3 + balanced_nc -- FALSIFIED, see below
    m3_per_team,    ///< (detected L3 / jc_nt) + balanced -- rejected on #426, measured anyway
    m4_per_sharer,  ///< (detected L3 / L3 sharing cores) + balanced
    m5_exact,       ///< pick nc so njb == jc_nt exactly, where n allows
    m6_guarded,     ///< M1, declined where the packed-B set would grow past L3.
                    ///< THE DEFAULT since #429. See nc_for_plan in gemm_blocked.hpp.
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
        case nc_model::m6_guarded:
            // Same arithmetic. M6's guard needs the jc_nt that the RE-PLAN
            // produces, and this function only sees the first pass's, so the
            // guard lives in `nc_for_plan`. Here M6 is M1.
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

/// M6's rule, as a named predicate: decline the balanced `nc` when it would BOTH
/// grow the resident packed-B set AND push it past L3 (#429, measured in #479).
///
/// BOTH TERMS ARE LOAD-BEARING, and the measurement says so rather than the
/// argument. Over 44 arms on four microarchitectures and two dtypes:
///
///   grew                  fires  4 -- catches both regressions, declines 2 GAINS
///   exceeds L3            fires 35 -- catches both, declines 33 gains
///   grew AND exceeds L3   fires  2 -- catches both, declines nothing else
///
/// The mechanism it encodes: `balanced_nc` lowers `nc`, which raises `njb`, which
/// can let `plan_gemm_grid` re-factor the grid and take `jc_nt` 2 -> 4. Each team
/// holds its own `kc x nc` packed-B panel, so the resident set moves 16 MB ->
/// 24 MB. An i7-12700K (25 MB L3) absorbs that and gains x1.02; a Ryzen 9 8945HS
/// (16 MB) does not and loses to x0.82 -- reproduced at 0.8181 / 0.8181 / 0.8203
/// across three independent sessions, the last with a pinned governor.
///
/// A set that merely EXCEEDS L3 is not the problem: where the set shrank, the
/// same Ryzen gained x1.06-1.74 whether it overflowed or not. It is the growth
/// across the boundary that costs.
constexpr bool nc_guard_declines(std::size_t packed_b_new, std::size_t packed_b_old,
                                 std::size_t l3_bytes) noexcept {
    return packed_b_new > packed_b_old && packed_b_new > l3_bytes;
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
        case nc_model::m6_guarded:    return "m6_guarded";
    }
    return "unknown";
}

inline constexpr nc_model all_nc_models[] = {
    nc_model::m0_default, nc_model::m1_balanced, nc_model::m2_detected,
    nc_model::m3_per_team, nc_model::m4_per_sharer, nc_model::m5_exact,
    nc_model::m6_guarded,
};

/// How many models there are. Derived, because a hardcoded copy of this number
/// silently EXCLUDED a model once: `bench_nc_models` and `sweep_nc_models` both
/// carried a literal `6`, and when M6 was added in #429 neither tool measured
/// it -- the shipped default was the one arm missing from the harness that
/// exists to check the default. Nothing failed; the CSVs just had six rows per
/// shape instead of seven.
inline constexpr std::size_t nc_model_count =
    sizeof(all_nc_models) / sizeof(all_nc_models[0]);

/// Parse a model name; `ok` reports whether it was recognised.
///
/// UNRECOGNISED NAMES ARE NOT SILENTLY M0. A typo'd `MTL5_NC_MODEL=m1_balnced`
/// that quietly ran the baseline would produce a CSV whose `model` column says
/// m1 and whose numbers are m0 -- the two-arms-that-are-secretly-one-arm failure
/// that #470 shipped with the quad kernel, where `gemm_i8_i32` and
/// `gemm_i8_i32_quad` benchmarked identically because inference had made them
/// the same kernel. It cost a benchmark round to notice. So the caller is told,
/// and `nc_model_selection` refuses to start.
inline nc_model nc_model_from_name(const char* s, bool& ok) noexcept {
    ok = true;
    if (s != nullptr)
        for (nc_model m : all_nc_models)
            if (std::strcmp(s, nc_model_name(m)) == 0) return m;
    ok = false;
    return nc_model::m0_default;
}

/// The model this process runs, from `MTL5_NC_MODEL`; M6 when unset (#429).
///
/// Read ONCE into a function-local static: the value is baked into the plan
/// every GEMM makes, so a mid-run change would split one CSV row's timings
/// across two models. Thread-safe by C++11 static initialisation.
///
/// An unset variable is the shipped configuration and the overwhelmingly common
/// case, so it costs one relaxed load after the first call. An unrecognised
/// value ABORTS rather than falling back -- see `nc_model_from_name`. This is a
/// benchmark selector; there is no production caller to keep running, and a
/// silent fallback here forges the provenance of every row that follows.
inline nc_model nc_model_selection() {
    static const nc_model sel = [] {
        const char* e = std::getenv("MTL5_NC_MODEL");
        if (e == nullptr || *e == '\0') return nc_model::m6_guarded;   // #429
        bool ok = false;
        const nc_model m = nc_model_from_name(e, ok);
        if (!ok) {
            std::fprintf(stderr,
                         "MTL5_NC_MODEL=\"%s\" is not a model name. Expected one of:", e);
            for (nc_model c : all_nc_models) std::fprintf(stderr, " %s", nc_model_name(c));
            std::fprintf(stderr,
                         "\nRefusing to fall back to m0_default: a run that silently used the\n"
                         "baseline while its output said otherwise is worse than no run.\n");
            std::abort();
        }
        return m;
    }();
    return sel;
}

} // namespace mtl::detail
