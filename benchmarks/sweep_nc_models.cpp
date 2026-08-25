// Offline sweep: where do the candidate `nc` models actually disagree? (#479)
//
// Six models across four machines is a combinatorial benchmark run, and most of
// it would measure nothing: for the majority of shapes every model computes the
// same `nc`, so timing them compares a binary against itself. #430's design note
// says to "enumerate shapes x models offline and measure only the shapes where
// models disagree", and this is that enumeration.
//
// It MEASURES NOTHING. Every number here comes from pure functions --
// `nc_for_model`, `plan_gemm_grid`, `grid_imbalance` -- so it runs in
// milliseconds, needs no quiet machine, and can be run on a laptop to plan a
// session on hardware that is scarce.
//
// WHY THE SHAPE LIST IS SEARCHED, NOT LISTED AND NOT COMPUTED. A fixed list
// silently tests nothing -- every measurement in #426 was square, the one regime
// where `jc_nt` is structurally 1 and the effect under study cannot appear. But
// a list computed from #430's algebra (`m <= mc * T/2`, `n > nc`) fails the same
// way for a subtler reason, and this sweep's first draft did exactly that: every
// shape it produced came back `jc_nt == 1`. The constraint is right and the
// consequence is not, because `plan_gemm_grid` caps `mc` at `ceil(m/budget)`
// (#441) and prefers the larger `ic_nt` on ties, so jc parallelism appears only
// near `m ~ mr * T` rather than `m ~ mc * T/2` -- on this Xeon, m = 6, not
// m = 96. So the sweep asks `plan_gemm_grid` which shapes actually produce a jc
// partition, and keeps those. See `derive_nc_shapes` in nc_shapes.hpp, which the
// timing harness shares so the shapes nominated here are the ones measured.
//
// THE TWO-PASS STRUCTURE is not an accident either. A model needs `jc_nt` to
// balance against, and `jc_nt` depends on `nc`, which is what the model
// computes. So: plan once with the baseline `nc` to learn the grid, ask the
// model for its `nc` given that `jc_nt`, then re-plan. That is the same shape
// `gemm_blocked` already uses for `mc` -- `plan_gemm_grid` then `balanced_mc`.

#include "nc_shapes.hpp"

#include <mtl/build_info.hpp>
#include <mtl/detail/gemm_blocked.hpp>
#include <mtl/detail/nc_model.hpp>
#include <mtl/detail/thread_pool.hpp>
#include <mtl/simd/blocking.hpp>
#include <mtl/util/cache_info.hpp>
#include <mtl/util/system_info.hpp>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <string>
#include <vector>

namespace {

struct outcome {
    std::size_t nc, nib, njb;
    unsigned    ic_nt, jc_nt;
    double      jc_imbalance;
    std::size_t packed_b;
};

/// Plan one (shape, model) pair. Two passes; see the file header.
outcome evaluate(mtl::detail::nc_model model, const mtl::bench::nc_point& p,
                 const mtl::simd::blocking_params& bp, std::size_t sdata,
                 std::size_t l3_default, std::size_t l3_detected,
                 std::size_t l3_sharers) {
    using namespace mtl::detail;
    // Pass 1: the grid the baseline nc produces, to learn jc_nt.
    const gemm_grid g0 = plan_gemm_grid(p.m, p.n, bp.mc, bp.nc, bp.mr, p.threads);

    nc_model_inputs in{p.n, bp.kc, bp.nr, sdata, l3_default, l3_detected,
                       l3_sharers, g0.jc_nt};
    const std::size_t nc = nc_for_model(model, in);

    // Pass 2: the grid that nc actually gives.
    const gemm_grid g = plan_gemm_grid(p.m, p.n, bp.mc, nc, bp.mr, p.threads);
    return outcome{nc, g.nib, g.njb, g.ic_nt, g.jc_nt,
                   grid_imbalance(g.njb, g.jc_nt),
                   packed_b_bytes(g.jc_nt, bp.kc, nc, bp.nr, sdata)};
}

// The shape derivation moved to benchmarks/nc_shapes.hpp so the TIMING harness
// (bench_nc_models) runs exactly the shapes this sweep nominates. Two copies
// that drifted would leave the session timing shapes the sweep never examined,
// reported against a disagreement count computed for different ones -- both
// files internally consistent, neither comparable to the other. See the header
// for why the list is searched rather than computed.

} // namespace

int main(int argc, char* argv[]) {
    std::string csv, dtype = "double";
    unsigned tmax = mtl::detail::thread_pool::instance().size();
    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* f) {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", f); std::exit(2); }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--csv"))          csv = need("--csv");
        else if (!std::strcmp(argv[i], "--dtype"))   dtype = need("--dtype");
        else if (!std::strcmp(argv[i], "--threads")) tmax = static_cast<unsigned>(std::atoi(need("--threads")));
        else if (!std::strcmp(argv[i], "--help")) {
            std::printf("Usage: sweep_nc_models [--csv f] [--dtype double|float] [--threads N]\n"
                        "Pure enumeration -- measures nothing. Lists the shapes where the\n"
                        "candidate nc models disagree, so a hardware session can measure\n"
                        "only those.\n");
            return 0;
        }
    }

    // Normalise BEFORE anything reports it. `derive_shapes` clamps its own copy
    // to 1, so a `--threads 0` run would emit rows saying threads=1 while the
    // header and the sidecar said 0 -- a committed CSV whose provenance
    // contradicts its own contents. Reject rather than silently clamp: 0 is not
    // a thread count anyone means.
    if (tmax == 0) {
        std::fprintf(stderr, "--threads must be >= 1 (got 0)\n");
        return 2;
    }

    const bool f32 = (dtype == "float");
    const std::size_t sdata = f32 ? sizeof(float) : sizeof(double);
    const mtl::simd::blocking_params bp =
        f32 ? mtl::simd::default_blocking<float> : mtl::simd::default_blocking<double>;

    const auto& ci = mtl::util::cached_cache_info();
    const std::size_t l3_default  = mtl::simd::default_hw_traits.l3_bytes;
    const std::size_t l3_detected = ci.l3_bytes;
    const std::size_t l3_sharers  = ci.l3_sharing_cores;

    std::printf("nc-model disagreement sweep (#479) -- nothing is measured here\n");
    std::printf("  dtype=%s  mr=%zu nr=%zu kc=%zu mc=%zu nc=%zu\n",
                dtype.c_str(), bp.mr, bp.nr, bp.kc, bp.mc, bp.nc);
    std::printf("  L3 default=%zu  detected=%zu  sharers=%zu  threads=%u\n\n",
                l3_default, l3_detected, l3_sharers, tmax);

    if (tmax == 1) {
        std::printf(
            "WARNING: --threads is 1.\n"
            "  At jc_nt == 1 every balancing model is a no-op BY CONSTRUCTION, so this\n"
            "  run cannot say anything about M1 vs M0 -- the pair #429 needs. The shape\n"
            "  search also finds no jc-parallel shapes, so only the negative controls\n"
            "  are enumerated. That is a valid control run and a useless planning run.\n"
            "  Re-run with --threads set to the count you intend to MEASURE at.\n\n");
    }

    const auto shapes = mtl::bench::derive_nc_shapes(bp, tmax);
    std::vector<std::string> rows;
    std::size_t disagreeing = 0;
    // Disagreement against the BASELINE, per model. The bare "N of M shapes
    // discriminate" line is not the answer and has actively misled once: at
    // --threads 1 it reads "5 of 5" while M1 equals M0 on every one of them,
    // because M2/M3/M4 differ and the total does not say which pair moved. The
    // question a planning run is asked is always about a specific pair.
    std::vector<std::size_t> pair_diff(mtl::detail::nc_model_count, 0);
    char buf[512];

    for (const auto& p : shapes) {
        outcome base{};
        bool first = true, differs = false;
        std::vector<outcome> outs;
        std::size_t mi = 0;
        for (auto m : mtl::detail::all_nc_models) {
            const outcome o = evaluate(m, p, bp, sdata, l3_default, l3_detected, l3_sharers);
            if (first) { base = o; first = false; }
            else if (o.nc != base.nc) { differs = true; ++pair_diff[mi]; }
            ++mi;
            outs.push_back(o);
            std::snprintf(buf, sizeof buf,
                          "%s,%s,%zu,%zu,%zu,%u,%zu,%zu,%zu,%u,%u,%.6f,%zu",
                          mtl::detail::nc_model_name(m), dtype.c_str(),
                          p.m, p.n, p.k, p.threads,
                          o.nc, o.nib, o.njb, o.ic_nt, o.jc_nt, o.jc_imbalance, o.packed_b);
            rows.emplace_back(buf);
        }
        if (differs) {
            ++disagreeing;
            std::printf("DISAGREE  m=%-6zu n=%-8zu k=%-5zu T=%u   nc:", p.m, p.n, p.k, p.threads);
            for (std::size_t i = 0; i < outs.size(); ++i)
                std::printf(" %s=%zu", mtl::detail::nc_model_name(mtl::detail::all_nc_models[i]),
                            outs[i].nc);
            std::printf("\n");
        }
    }

    std::printf("\n%zu of %zu shapes separate at least two models.\n\n",
                disagreeing, shapes.size());
    std::printf("Against the baseline M0, per model -- this is the number to read:\n");
    for (std::size_t i = 1; i < mtl::detail::nc_model_count; ++i)
        std::printf("  %-14s %2zu of %2zu%s\n",
                    mtl::detail::nc_model_name(mtl::detail::all_nc_models[i]),
                    pair_diff[i], shapes.size(),
                    i == 1 ? "   <- the pair #429 needs" : "");

    const std::size_t m1 = pair_diff[1];
    std::printf("\n");
    if (m1 == 0) {
        std::printf("M1 NEVER DIFFERS FROM M0 on this machine's derived shapes.\n"
                    "  That is a RESULT, not a failure: this machine cannot contribute to\n"
                    "  the M0-vs-M1 choice, and a throughput run here would compare a\n"
                    "  binary against itself. Worth knowing before booking the machine.\n");
    } else {
        std::printf("Measure the %zu shape%s where M1 differs from M0; the other %zu would\n"
                    "compare a binary against itself.\n",
                    m1, m1 == 1 ? "" : "s", shapes.size() - m1);
    }

    if (!csv.empty()) {
        std::ofstream out(csv);
        if (!out) { std::fprintf(stderr, "cannot write %s\n", csv.c_str()); return 1; }
        out << "model,dtype,m,n,k,threads,nc,nib,njb,ic_nt,jc_nt,jc_imbalance,packedB_bytes\n";
        for (const auto& r : rows) out << r << "\n";

        // Sidecar (#442, #477). This tool MEASURES NOTHING, so the machine-state
        // half of the contract -- governor, thermal headroom, competing load --
        // is irrelevant here and deliberately absent. The BUILD half is not
        // merely relevant, it is the whole story: every number in this CSV comes
        // from `default_blocking<T>`, which is derived from the compiled SIMD
        // width. The same source built for SSE4 and for AVX-512 produces
        // different `mr`, `nr`, `kc` and `nc`, and therefore a different answer
        // to "where do the models disagree". A reader who cannot recover
        // `cxx_flags` cannot tell which machine's question this CSV answers.
        // Path built through std::filesystem::path per the project rule. It is a
        // suffix, not a join, so this is about intent and consistency rather
        // than correctness; the real consolidation is the shared sidecar writer
        // #477 asks for, which would let all three producers share one.
        std::filesystem::path side = std::filesystem::path{csv};
        side += ".sysinfo";
        std::ofstream si{side};
        if (si) {
            si << "label=nc_model_sweep\n"
               << mtl::util::to_keyvals(mtl::util::identify())
               << "git_commit="       << mtl::build_git_commit << "\n"
               << "git_dirty="        << mtl::build_git_dirty  << "\n"
               << "cxx_flags="        << mtl::build_cxx_flags  << "\n"
               << "cmake_build_type=" << mtl::build_cmake_type << "\n"
               << "harness=sweep_nc_models\n"
               << "measures=nothing (pure enumeration)\n"
               << "dtype="            << dtype << "\n"
               << "threads="          << tmax << "\n"
               << "mr=" << bp.mr << " nr=" << bp.nr << " kc=" << bp.kc
               << " mc=" << bp.mc << " nc=" << bp.nc << "\n"
               << "l3_default_bytes="  << l3_default  << "\n"
               << "l3_detected_bytes=" << l3_detected << "\n"
               << "l3_sharing_cores="  << l3_sharers  << "\n"
               << "shapes_total=" << shapes.size() << "\n"
               << "shapes_separating_any=" << disagreeing << "\n";
            for (std::size_t i = 1; i < mtl::detail::nc_model_count; ++i)
                si << "shapes_differing_vs_m0_"
                   << mtl::detail::nc_model_name(mtl::detail::all_nc_models[i])
                   << "=" << pair_diff[i] << "\n";
        }
        // FAIL, do not report success. A silent sidecar failure produces exactly
        // the untraceable CSV #478 exists to reject -- and the tool would have
        // said "+ .sysinfo" while returning 0, so the next thing to notice would
        // be CI, on a file already committed. Check the final stream state too:
        // an ofstream that opened can still fail on a full disk mid-write.
        si.flush();
        if (!si) {
            std::fprintf(stderr,
                         "failed to write %s -- the CSV would be untraceable, so it is\n"
                         "not left behind. See benchmarks/check_sidecars.sh.\n",
                         side.string().c_str());
            std::error_code ec;
            std::filesystem::remove(csv, ec);
            std::filesystem::remove(side, ec);
            return 1;
        }
        std::printf("wrote %s (%zu rows) + %s\n", csv.c_str(), rows.size(),
                    side.filename().string().c_str());
    }
    return 0;
}
