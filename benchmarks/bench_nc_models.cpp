// Time the candidate `nc` models against each other (#479, feeding #429).
//
// The offline sweep (sweep_nc_models) says WHERE the six models disagree. This
// measures what that disagreement is worth. Together they are the experiment
// #429 needs before `balanced_nc` can become the default -- shipping it on the
// argument that an even partition must be faster is how #426 shipped runtime
// cache detection, which then lost on every machine that ran it.
//
// ALL SIX ARMS RUN IN ONE PROCESS, interleaved within each round. Six processes
// would put the arms on six different thermal and scheduling states and make the
// machine, not the model, the thing under test. `gemm_blocked` takes the model
// as a per-call parameter for exactly this reason.
//
// THE NEGATIVE CONTROL IS THE POINT OF HALF THESE ROWS. The shape list includes
// shapes where every model computes the SAME `nc` -- square problems, T = 1 --
// and those arms are running byte-identical code. Whatever spread they show IS
// the noise floor of this session, measured rather than assumed. A "win" on a
// discriminating shape that is smaller than the spread on a non-discriminating
// one is not a win, and the summary says so. Without this, the first plausible
// 3% would be reported as a result; #430's committed A/B data has ratios in that
// range that nobody can now defend, because no control was recorded beside them.
//
// WHAT IS DELIBERATELY NOT HERE: a winner. This writes a CSV. Choosing needs
// >= 3 microarchitectures, and one machine's answer is how the Haswell-tuned
// constants got hardcoded in the first place.

#include "nc_shapes.hpp"

#include <mtl/build_info.hpp>
#include <mtl/detail/gemm_blocked.hpp>
#include <mtl/detail/nc_model.hpp>
#include <mtl/detail/thread_pool.hpp>
#include <mtl/simd/blocking.hpp>
#include <mtl/util/cache_info.hpp>
#include <mtl/util/system_info.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace {

using mtl::bench::nc_point;
using mtl::detail::nc_model;

/// Bit-exact fold of the result buffer.
///
/// XOR of the raw bit patterns, NOT a floating sum: `nc` regroups columns and
/// must not change any C element's FMA chain, so the arms have to agree BIT for
/// bit. A summed checksum would compare equal for two buffers that differ in a
/// low mantissa bit -- which is precisely the corruption a mis-sized packed-B
/// panel produces, and it corrupts silently rather than crashing.
template <typename T>
std::uint64_t fold(const std::vector<T>& v) {
    std::uint64_t h = 0;
    for (const T& x : v) {
        std::uint64_t bits = 0;
        std::memcpy(&bits, &x, sizeof(T) < sizeof(bits) ? sizeof(T) : sizeof(bits));
        h ^= bits + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    }
    return h;
}

struct arm_result {
    double        best_secs = 1e300;   // min across rounds; see run()
    std::uint64_t checksum  = 0;       // DIAGNOSTIC only -- see `exact`
    bool          exact     = true;    // element-wise equal to M0's result
    std::size_t   nc = 0, mc = 0, nib = 0, njb = 0;
    unsigned      ic_nt = 1, jc_nt = 1;
};

/// One shape, all six models, `rounds` times, interleaved.
///
/// MIN across rounds rather than mean. The quantity wanted is the machine's
/// capability at this blocking; every perturbation available on a shared box --
/// a migrated thread, a competing process, a DVFS excursion -- adds time and
/// none removes it, so the distribution is one-sided and the mean estimates the
/// interference as much as the kernel. The spread across rounds is reported
/// separately so a run whose minimum is not converged is visible rather than
/// quietly reported as a result.
template <typename T>
std::vector<arm_result> run(const nc_point& p, unsigned reps, unsigned rounds,
                            double& worst_spread) {
    const std::size_t m = p.m, n = p.n, k = p.k;
    std::vector<T> A(m * k), B(k * n), C(m * n);
    for (std::size_t i = 0; i < A.size(); ++i) A[i] = T((i * 37) % 101) - T(50);
    for (std::size_t i = 0; i < B.size(); ++i) B[i] = T((i * 53) % 97) - T(48);

    constexpr std::size_t NMODELS = 6;
    std::vector<arm_result> out(NMODELS);
    std::vector<double> first_secs(NMODELS, 0.0);

    // ---- pass 1: verify + warm, both untimed ------------------------------
    //
    // EVERY model runs once here, not just M0. Two jobs in one pass:
    //
    // (a) EXACT verification. The result is compared to M0's element by element,
    //     not by checksum: `fold` reduces a whole buffer to 64 bits, and equal
    //     checksums do not prove equal elements. A collision would set
    //     `checksum_ok=1` on a genuinely wrong arm and let its timings into the
    //     CSV. The comparison is cheap, certain, and outside the timed region,
    //     so there is no reason to accept a probabilistic answer. The checksum
    //     stays, as a diagnostic that survives into the CSV.
    //
    // (b) WARMUP FOR ALL SIX. First touch of C, the pool's threads, and each
    //     model's own packed-B buffer (they differ in size -- M2's nc is nearly
    //     twice M0's here) cost time that belongs to no arm in particular.
    //     Warming only M0, as the first draft did, handed the baseline a warm
    //     first round and every candidate a cold one.
    std::vector<T> ref;
    bool mismatch[NMODELS] = {};
    for (std::size_t mi = 0; mi < NMODELS; ++mi) {
        const nc_model mo = mtl::detail::all_nc_models[mi];
        std::fill(C.begin(), C.end(), T(0));
        mtl::detail::gemm_blocked<T>(m, n, k, T(1),
            A.data(), static_cast<std::ptrdiff_t>(k), 1,
            B.data(), static_cast<std::ptrdiff_t>(n), 1,
            T(0), C.data(), n, p.threads, mo);
        out[mi].checksum = fold(C);
        if (mi == 0) ref = C;                       // the baseline's exact result
        else mismatch[mi] = (std::memcmp(ref.data(), C.data(),
                                         ref.size() * sizeof(T)) != 0);
        const auto pl = mtl::detail::gemm_plan_for<T>(m, n, p.threads, mo);
        out[mi].nc = pl.nc;   out[mi].mc = pl.mc;
        out[mi].nib = pl.nib; out[mi].njb = pl.njb;
        out[mi].ic_nt = pl.ic_nt; out[mi].jc_nt = pl.jc_nt;
        out[mi].exact = !mismatch[mi];
    }

    // ---- pass 2: timing, with the model order rotated per round -----------
    //
    // COUNTERBALANCED, because the minimum is taken per model across rounds and
    // a fixed order gives every model the SAME position every time -- so a
    // position effect never averages out, it becomes a model effect. M0 ran
    // first in every round and M5 last, which is the worst arrangement
    // available: the baseline every ratio is divided by would have carried
    // whatever position 0 is worth on this machine.
    //
    // Rotating by the round index gives each model each position as evenly as
    // `rounds` allows. With rounds a multiple of 6 it is exact.
    for (unsigned r = 0; r < rounds; ++r) {
        for (std::size_t j = 0; j < NMODELS; ++j) {
            const std::size_t mi = (j + r) % NMODELS;
            const nc_model mo = mtl::detail::all_nc_models[mi];
            std::fill(C.begin(), C.end(), T(0));
            const auto t0 = std::chrono::steady_clock::now();
            for (unsigned rep = 0; rep < reps; ++rep)
                mtl::detail::gemm_blocked<T>(m, n, k, T(1),
                    A.data(), static_cast<std::ptrdiff_t>(k), 1,
                    B.data(), static_cast<std::ptrdiff_t>(n), 1,
                    T(0), C.data(), n, p.threads, mo);
            const auto t1 = std::chrono::steady_clock::now();
            const double secs =
                std::chrono::duration<double>(t1 - t0).count() / double(reps);
            if (secs < out[mi].best_secs) out[mi].best_secs = secs;
            if (r == 0) first_secs[mi] = secs;
        }
    }

    // How far the first round sat above the eventual minimum, worst arm. A large
    // value means the minimum is not converged and `--rounds` is too low.
    worst_spread = 0.0;
    for (std::size_t mi = 0; mi < NMODELS; ++mi)
        if (out[mi].best_secs > 0.0) {
            const double s = first_secs[mi] / out[mi].best_secs - 1.0;
            if (s > worst_spread) worst_spread = s;
        }
    return out;
}

} // namespace

int main(int argc, char* argv[]) {
    std::string csv, dtype = "double";
    unsigned tmax = mtl::detail::thread_pool::instance().size();
    unsigned reps = 3, rounds = 5;

    for (int i = 1; i < argc; ++i) {
        auto need = [&](const char* f) {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", f); std::exit(2); }
            return argv[++i];
        };
        if (!std::strcmp(argv[i], "--csv"))          csv    = need("--csv");
        else if (!std::strcmp(argv[i], "--dtype"))   dtype  = need("--dtype");
        else if (!std::strcmp(argv[i], "--threads")) tmax   = unsigned(std::atoi(need("--threads")));
        else if (!std::strcmp(argv[i], "--reps"))    reps   = unsigned(std::atoi(need("--reps")));
        else if (!std::strcmp(argv[i], "--rounds"))  rounds = unsigned(std::atoi(need("--rounds")));
        else if (!std::strcmp(argv[i], "--help")) {
            std::printf(
                "Usage: bench_nc_models [--csv f] [--dtype double|float]\n"
                "                       [--threads N] [--reps R] [--rounds K]\n\n"
                "Times the six candidate nc models against each other on the shapes\n"
                "sweep_nc_models nominates for THIS machine. All six arms run in one\n"
                "process, interleaved, so they share a thermal state.\n\n"
                "Run sweep_nc_models first: if it reports M1 differs from M0 on 0\n"
                "shapes, this machine cannot answer #429's question and this binary\n"
                "would compare arms that are byte-identical.\n");
            return 0;
        } else { std::fprintf(stderr, "unknown option: %s\n", argv[i]); return 2; }
    }

    // Same refusal as the sweep: 0 is not a thread count anyone means, and a run
    // that clamped silently would emit rows saying threads=1 under a header and
    // sidecar saying 0.
    if (tmax == 0) { std::fprintf(stderr, "--threads must be >= 1 (got 0)\n"); return 2; }
    if (reps == 0 || rounds == 0) {
        std::fprintf(stderr, "--reps and --rounds must be >= 1\n"); return 2;
    }
    if (dtype != "double" && dtype != "float") {
        std::fprintf(stderr, "--dtype must be double or float (got %s)\n", dtype.c_str());
        return 2;
    }

    const bool f32 = (dtype == "float");
    const std::size_t sdata = f32 ? sizeof(float) : sizeof(double);
    const mtl::simd::blocking_params bp =
        f32 ? mtl::simd::default_blocking<float> : mtl::simd::default_blocking<double>;
    const mtl::simd::blocking_params rbp =
        f32 ? mtl::simd::runtime_blocking<float>() : mtl::simd::runtime_blocking<double>();
    const auto& ci   = mtl::util::cached_cache_info();
    const unsigned pool = mtl::detail::thread_pool::instance().size();

    std::printf("nc-model timing harness (#479)\n");
    std::printf("  %s\n", mtl::util::to_string(mtl::util::identify()).c_str());
    std::printf("  dtype=%s  mr=%zu nr=%zu kc=%zu mc=%zu nc=%zu\n",
                dtype.c_str(), bp.mr, bp.nr, bp.kc, bp.mc, bp.nc);
    std::printf("  L3 default=%zu detected=%zu sharers=%zu   threads=%u (pool %u)"
                "  reps=%u rounds=%u\n\n",
                mtl::simd::default_hw_traits.l3_bytes, ci.l3_bytes,
                ci.l3_sharing_cores, tmax, pool, reps, rounds);

    // THE CAVEAT THIS MACHINE MAY CARRY. The models are evaluated against the
    // COMPILE-TIME kc, because that is what the offline sweep fed them and the
    // two must ask the same question. The nest steps the RUNTIME kc. Where they
    // differ -- a machine whose detected L1 moves kc; see the pin at
    // blocking.hpp:351 -- the panel actually built is not the one the models
    // reasoned about, and this machine's rows carry an asterisk. Reported, not
    // silently reconciled: papering over it here would make the CSV disagree
    // with the sweep that chose its shapes.
    const bool kc_split = (bp.kc != rbp.kc), mc_split = (bp.mc != rbp.mc);
    if (kc_split || mc_split) {
        std::printf("NOTE: compile-time and runtime blocking differ on this machine:\n");
        if (kc_split) std::printf("  kc: compile=%zu runtime=%zu\n", bp.kc, rbp.kc);
        if (mc_split) std::printf("  mc: compile=%zu runtime=%zu\n", bp.mc, rbp.mc);
        std::printf("  The models are evaluated at the COMPILE-TIME values, matching\n"
                    "  sweep_nc_models. The nest steps the runtime ones. Treat the\n"
                    "  capacity models (M2..M5) on this machine accordingly.\n\n");
    }
    if (tmax == 1) {
        std::printf("WARNING: --threads is 1. At jc_nt == 1 every balancing model is a\n"
                    "  no-op BY CONSTRUCTION, so M1 cannot differ from M0 and this run\n"
                    "  cannot address #429. Valid as a control; useless as an experiment.\n\n");
    }

    const auto shapes = mtl::bench::derive_nc_shapes(bp, tmax);
    std::vector<std::string> rows;
    char buf[768];

    std::size_t integrity_failures = 0, discriminating = 0;
    // THE NOISE FLOOR IS MEASURED PER ARM, NOT PER SHAPE, and the difference is
    // not cosmetic. The first draft called a SHAPE a control when every model
    // chose the same `nc` -- and on this hardware no shape qualifies, because
    // the detected L3 differs from the compile-time figure so M2 and M4 always
    // move. The control count was therefore zero, the spread printed 0.00%, and
    // "is the effect above the noise" became a comparison against nothing that
    // no run could fail. It read like a clean result.
    //
    // Any ARM whose `nc` equals M0's is running byte-identical code, whatever the
    // other four did, so its deviation from 1.0 is pure measurement noise. Those
    // arms exist on nearly every shape: on this Xeon's 1024^2 point M1 chose the
    // same nc = 2048 as M0 and still timed x1.0337. A 3% "win" elsewhere in the
    // same session is that number, not a finding.
    double      control_spread = 0.0;   // largest |ratio-1| among nc-identical arms
    std::size_t control_n      = 0;     // how many such arms -- 0 makes the above vacuous
    double      best_disc_gain = 0.0;   // largest M1-over-M0 gain where M1's nc DIFFERS
    std::size_t disc_m1_n      = 0;
    double      worst_convergence = 0.0;

    for (const nc_point& p : shapes) {
        double spread = 0.0;
        const auto arms = f32 ? run<float>(p, reps, rounds, spread)
                              : run<double>(p, reps, rounds, spread);
        if (spread > worst_convergence) worst_convergence = spread;

        const double flops = 2.0 * double(p.m) * double(p.n) * double(p.k);
        const double base  = arms[0].best_secs;

        // Does this shape actually separate anything? If every model chose the
        // same nc, the six arms just ran byte-identical code and their spread is
        // this session's noise floor.
        bool disc = false;
        for (std::size_t mi = 1; mi < arms.size(); ++mi)
            if (arms[mi].nc != arms[0].nc) disc = true;
        if (disc) ++discriminating;

        std::printf("m=%-6zu n=%-7zu k=%-5zu T=%u  %s\n", p.m, p.n, p.k, p.threads,
                    disc ? "[discriminates]" : "[control: all models agree]");

        for (std::size_t mi = 0; mi < arms.size(); ++mi) {
            const arm_result& a = arms[mi];
            const nc_model mo = mtl::detail::all_nc_models[mi];
            const double gf    = flops / a.best_secs / 1e9;
            const double ratio = base / a.best_secs;      // >1 means faster than M0

            // Every arm must have computed the same answer. `nc` groups columns;
            // it does not reorder any C element's FMA chain. A mismatch means the
            // timings are not comparable and the row is not a measurement.
            //
            // Decided by the ELEMENT-WISE comparison in run()'s first pass, not
            // by the checksum: equal 64-bit folds do not prove equal buffers, and
            // a collision would admit a wrong arm's timings to the CSV. The
            // checksum is carried as a diagnostic so two runs can be compared
            // after the fact.
            const bool ok = a.exact;
            if (!ok) ++integrity_failures;

            const bool same_nc = (a.nc == arms[0].nc);
            if (mi > 0 && same_nc) {          // byte-identical code: pure noise
                ++control_n;
                const double dev = ratio > 1.0 ? ratio - 1.0 : 1.0 - ratio;
                if (dev > control_spread) control_spread = dev;
            }
            if (mo == nc_model::m1_balanced && !same_nc) {
                ++disc_m1_n;
                if (ratio - 1.0 > best_disc_gain) best_disc_gain = ratio - 1.0;
            }

            const double jc_imb = mtl::detail::grid_imbalance(a.njb, a.jc_nt);
            const double ic_imb = mtl::detail::grid_imbalance(a.nib, a.ic_nt);
            const std::size_t pb = mtl::detail::packed_b_bytes(a.jc_nt, rbp.kc, a.nc, sdata);

            std::printf("    %-14s nc=%-6zu njb=%-4zu jc_nt=%u  jc_imb=%.3f  "
                        "%8.3f GF/s  x%.4f%s\n",
                        mtl::detail::nc_model_name(mo), a.nc, a.njb, a.jc_nt,
                        jc_imb, gf, ratio, ok ? "" : "  *** RESULT DIFFERS FROM M0 ***");

            std::snprintf(buf, sizeof buf,
                "%s,%s,%zu,%zu,%zu,%u,%u,%d,%zu,%zu,%zu,%zu,%u,%u,"
                "%.6f,%.6f,%zu,%zu,%u,%u,%.9f,%.4f,%.6f,%d,%llu",
                mtl::detail::nc_model_name(mo), dtype.c_str(),
                p.m, p.n, p.k, p.threads, pool, disc ? 1 : 0,
                a.nc, a.mc, a.nib, a.njb, a.ic_nt, a.jc_nt,
                ic_imb, jc_imb, pb, ci.l3_bytes, reps, rounds,
                a.best_secs, gf, ratio, ok ? 1 : 0,
                static_cast<unsigned long long>(a.checksum));
            rows.emplace_back(buf);
        }
    }

    std::printf("\n%zu of %zu shapes discriminate; %zu integrity failures.\n",
                discriminating, shapes.size(), integrity_failures);
    std::printf("Noise floor: %.2f%%  (worst deviation among %zu arm(s) whose nc "
                "equalled M0's,\n             i.e. byte-identical code)\n",
                control_spread * 100.0, control_n);
    std::printf("Best M1-over-M0 gain where M1's nc actually differed: %.2f%% "
                "(over %zu arm(s))\n", best_disc_gain * 100.0, disc_m1_n);
    std::printf("Worst first-round excess over the minimum: %.2f%%%s\n",
                worst_convergence * 100.0,
                worst_convergence > 0.25 ? "  (raise --rounds)" : "");

    // The comparison that decides whether this session says anything at all --
    // and the guard that keeps it from being vacuous. With no control arms there
    // is no floor to clear, so a zero spread is ABSENCE OF EVIDENCE and must not
    // read as a clean result; the first version of this check printed exactly
    // that and passed unconditionally.
    if (control_n == 0) {
        std::printf("\nNO CONTROL ARMS IN THIS SESSION.\n"
                    "  Every model chose an nc different from M0 on every shape, so\n"
                    "  nothing here ran byte-identical code and the noise floor is\n"
                    "  UNMEASURED -- not zero. Any ratio below is an upper bound on the\n"
                    "  effect, and the session cannot say the effect is real.\n");
    } else if (disc_m1_n == 0) {
        std::printf("\nM1 NEVER DIFFERED FROM M0 HERE.\n"
                    "  This machine cannot address #429's question at this thread count.\n"
                    "  Check sweep_nc_models before booking it again.\n");
    } else if (best_disc_gain <= control_spread) {
        std::printf("\nTHE EFFECT IS NOT ABOVE THIS SESSION'S NOISE FLOOR.\n"
                    "  Arms running byte-identical code varied by %.2f%%, and the best\n"
                    "  M1 gain was %.2f%%. Report this as an upper bound, not a result:\n"
                    "  on this machine the model choice is worth less than the\n"
                    "  measurement can see. A quieter box or more --rounds may help;\n"
                    "  a bigger number pulled from these rows will not.\n",
                    control_spread * 100.0, best_disc_gain * 100.0);
    }
    if (integrity_failures) {
        std::fprintf(stderr,
            "\n%zu arm(s) disagreed with M0 bit-for-bit. `nc` regroups columns and\n"
            "must not change any C element's FMA chain, so this is a defect in the\n"
            "blocking, not a tolerance question -- the timings above are not\n"
            "comparable. Not writing a CSV.\n", integrity_failures);
        return 1;
    }

    if (csv.empty()) return 0;

    std::ofstream out(csv);
    if (!out) { std::fprintf(stderr, "cannot write %s\n", csv.c_str()); return 1; }
    out << "model,dtype,m,n,k,threads,pool,discriminates,nc,mc,nib,njb,ic_nt,jc_nt,"
           "ic_imbalance,jc_imbalance,packedB_bytes,l3_detected,reps,rounds,"
           "best_seconds,gflops,ratio_vs_m0,exact_vs_m0,checksum\n";
    for (const std::string& r : rows) out << r << "\n";
    out.flush();
    if (!out) { std::fprintf(stderr, "failed writing %s\n", csv.c_str()); return 1; }

    // Sidecar (#442, #477). Unlike the sweep, this one MEASURES, so the
    // machine-state half of the contract matters and the runner script captures
    // it (governor, thermal, competing load) before invoking this binary. What
    // the binary owns is the BUILD half, which it alone can see.
    std::filesystem::path side{csv};
    side += ".sysinfo";
    std::ofstream si{side};
    if (si) {
        si << "label=nc_model_timing\n"
           << mtl::util::to_keyvals(mtl::util::identify())
           << "git_commit="       << mtl::build_git_commit << "\n"
           << "git_dirty="        << mtl::build_git_dirty  << "\n"
           << "cxx_flags="        << mtl::build_cxx_flags  << "\n"
           << "cmake_build_type=" << mtl::build_cmake_type << "\n"
           << "harness=bench_nc_models\n"
           << "measures=gemm throughput per nc model\n"
           << "dtype="   << dtype  << "\n"
           << "threads=" << tmax   << "\n"
           << "pool="    << pool   << "\n"
           << "reps="    << reps   << "\n"
           << "rounds="  << rounds << "\n"
           << "mr=" << bp.mr << " nr=" << bp.nr << " kc=" << bp.kc
           << " mc=" << bp.mc << " nc=" << bp.nc << "\n"
           << "runtime_kc=" << rbp.kc << " runtime_mc=" << rbp.mc << "\n"
           << "compile_runtime_blocking_split=" << ((kc_split || mc_split) ? 1 : 0) << "\n"
           << "l3_default_bytes="  << mtl::simd::default_hw_traits.l3_bytes << "\n"
           << "l3_detected_bytes=" << ci.l3_bytes << "\n"
           << "l3_sharing_cores="  << ci.l3_sharing_cores << "\n"
           << "shapes_total="      << shapes.size() << "\n"
           << "shapes_discriminating=" << discriminating << "\n"
           << "noise_floor="    << control_spread << "\n"
           << "noise_floor_arms=" << control_n << "\n"
           << "best_m1_gain="   << best_disc_gain << "\n"
           << "m1_differing_arms=" << disc_m1_n << "\n"
           << "worst_first_round_excess=" << worst_convergence << "\n"
           // Three states, not two: unmeasured is not the same as "no effect",
           // and a reader who sees only a 0/1 flag cannot tell them apart.
           << "effect_above_noise="
           << (control_n == 0 ? "unmeasured"
                              : (best_disc_gain > control_spread ? "1" : "0")) << "\n";
    }
    si.flush();
    if (!si) {
        std::fprintf(stderr,
                     "failed to write %s -- the CSV would be untraceable, so it is not\n"
                     "left behind. See benchmarks/check_sidecars.sh.\n",
                     side.string().c_str());
        std::error_code ec;
        std::filesystem::remove(csv, ec);
        std::filesystem::remove(side, ec);
        return 1;
    }
    std::printf("\nwrote %s (%zu rows) + %s\n", csv.c_str(), rows.size(),
                side.filename().string().c_str());
    return 0;
}
