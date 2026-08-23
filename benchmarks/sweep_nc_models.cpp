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
// partition, and keeps those. See derive_shapes.
//
// THE TWO-PASS STRUCTURE is not an accident either. A model needs `jc_nt` to
// balance against, and `jc_nt` depends on `nc`, which is what the model
// computes. So: plan once with the baseline `nc` to learn the grid, ask the
// model for its `nc` given that `jc_nt`, then re-plan. That is the same shape
// `gemm_blocked` already uses for `mc` -- `plan_gemm_grid` then `balanced_mc`.

#include <mtl/detail/gemm_blocked.hpp>
#include <mtl/detail/nc_model.hpp>
#include <mtl/detail/thread_pool.hpp>
#include <mtl/simd/blocking.hpp>
#include <mtl/util/cache_info.hpp>
#include <mtl/util/system_info.hpp>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct point {
    std::size_t m, n, k;
    unsigned    threads;
};

struct outcome {
    std::size_t nc, nib, njb;
    unsigned    ic_nt, jc_nt;
    double      jc_imbalance;
    std::size_t packed_b;
};

/// Plan one (shape, model) pair. Two passes; see the file header.
outcome evaluate(mtl::detail::nc_model model, const point& p,
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
                   packed_b_bytes(g.jc_nt, bp.kc, nc, sdata)};
}

/// Shapes derived from this machine's own blocking -- by SEARCHING the grid,
/// not by applying a formula to `mc`.
///
/// The first draft of this function did apply the formula from #430's design
/// note (`m <= mc * T/2`, `n > nc`) and every shape it produced came back with
/// `jc_nt == 1`, making the whole sweep vacuous. The note's algebra is right
/// about the constraint and wrong about the consequence, because
/// `plan_gemm_grid` does two further things: it caps `mc` at `ceil(m/budget)`
/// so the ic partition can fill the budget (#441), which drives `nib` back up to
/// the thread count; and on a tie it PREFERS the larger `ic_nt`, because a wider
/// ic team keeps the shared packed-B panel resident. Between them, jc
/// parallelism appears only for `m` far smaller than the formula suggests --
/// around `mr * T` rather than `mc * T/2`.
///
/// That is the same failure the note itself warns about ("a fixed shape list
/// silently tests nothing"), reached by a different route: a derived list is no
/// better than a fixed one if it is derived from the wrong model of the grid. So
/// this asks `plan_gemm_grid` directly and keeps what it says.
std::vector<point> derive_shapes(const mtl::simd::blocking_params& bp, unsigned tmax) {
    using mtl::detail::plan_gemm_grid;
    std::vector<point> v;
    const std::size_t k = 1024;
    if (tmax == 0) tmax = 1;

    // Search for m values that actually yield jc_nt >= 2 at this thread count,
    // then pair each with several n so njb varies against jc_nt.
    std::vector<std::size_t> jc_m;
    for (std::size_t m = bp.mr; m <= bp.mc * tmax && jc_m.size() < 3; m += bp.mr) {
        const auto g = plan_gemm_grid(m, bp.nc * 8, bp.mc, bp.nc, bp.mr, tmax);
        if (g.jc_nt >= 2) jc_m.push_back(m);
    }
    for (std::size_t m : jc_m)
        for (std::size_t mult : {2u, 3u, 5u, 7u, 8u})
            v.push_back({m, bp.nc * mult, k, tmax});

    // Negative controls, always included so the sweep records that they were
    // checked rather than leaving their absence to be assumed:
    //   square    -- jc_nt structurally 1 on most machines
    //   tall/thin -- ic-dominated
    //   T = 1     -- every balancing model must be a no-op
    for (std::size_t s : {std::size_t{1024}, std::size_t{2048}})
        v.push_back({s, s, k, tmax});
    v.push_back({8192, 64, k, tmax});
    for (std::size_t mult : {2u, 5u})
        v.push_back({64, bp.nc * mult, k, 1u});

    return v;
}

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

    const auto shapes = derive_shapes(bp, tmax);
    std::vector<std::string> rows;
    std::size_t disagreeing = 0;
    char buf[512];

    for (const auto& p : shapes) {
        outcome base{};
        bool first = true, differs = false;
        std::vector<outcome> outs;
        for (auto m : mtl::detail::all_nc_models) {
            const outcome o = evaluate(m, p, bp, sdata, l3_default, l3_detected, l3_sharers);
            if (first) { base = o; first = false; }
            else if (o.nc != base.nc) differs = true;
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

    std::printf("\n%zu of %zu shapes discriminate between the models.\n",
                disagreeing, shapes.size());
    if (disagreeing == 0)
        std::printf("On this machine no derived shape separates them -- measuring here\n"
                    "would compare a binary against itself. That is a result, not a\n"
                    "failure: it says this machine cannot contribute to the choice.\n");

    if (!csv.empty()) {
        std::ofstream out(csv);
        if (!out) { std::fprintf(stderr, "cannot write %s\n", csv.c_str()); return 1; }
        out << "model,dtype,m,n,k,threads,nc,nib,njb,ic_nt,jc_nt,jc_imbalance,packedB_bytes\n";
        for (const auto& r : rows) out << r << "\n";
        std::printf("wrote %s (%zu rows)\n", csv.c_str(), rows.size());
    }
    return 0;
}
