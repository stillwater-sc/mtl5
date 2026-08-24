// Runtime cache detection and the blocking parameters derived from it (#222).
//
// Detection is best effort and machine-dependent, so these assert INVARIANTS and
// FALLBACK behaviour rather than specific sizes -- a test that hardcoded this
// machine's 15 MB L3 would fail on the next machine, which is the very coupling
// the change exists to remove. What is pinned:
//
//   * anything detected is physically plausible (powers of two, ordered by level)
//   * anything NOT detected leaves the compile-time default untouched
//   * the register tile mr x nr is IDENTICAL to default_blocking's, because the
//     micro-kernel is instantiated on that tile and a runtime value must never
//     be able to move it
//
// The detected values are reported via INFO so a failure elsewhere in the GEMM
// suite can be read against the hierarchy the run actually used.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <cstddef>

#include <mtl/util/cache_info.hpp>
#include <mtl/simd/blocking.hpp>

using namespace mtl;

namespace {

bool is_power_of_two(std::size_t v) { return v != 0 && (v & (v - 1)) == 0; }

} // namespace

TEST_CASE("detected cache sizes are plausible", "[util][cache][blocking]") {
    const util::cache_info c = util::detect_caches();

    INFO("l1d=" << c.l1d_bytes << " assoc=" << c.l1d_assoc
         << " l2=" << c.l2_bytes << " l3=" << c.l3_bytes
         << " line=" << c.line_bytes);

    // Each field is either "not detected" (0) or a believable figure. The bounds
    // are deliberately loose: they catch a misparse (an I-cache counted as L1d, a
    // K/M suffix dropped, a CPUID field shifted) without encoding one machine.
    if (c.line_bytes != 0) {
        REQUIRE(is_power_of_two(c.line_bytes));
        REQUIRE(c.line_bytes >= 16);
        REQUIRE(c.line_bytes <= 256);
    }
    if (c.l1d_bytes != 0) {
        REQUIRE(c.l1d_bytes >= 4u * 1024);
        REQUIRE(c.l1d_bytes <= 1024u * 1024);
    }
    if (c.l2_bytes != 0) {
        REQUIRE(c.l2_bytes >= 64u * 1024);
        REQUIRE(c.l2_bytes <= 128u * 1024 * 1024);
    }
    if (c.l3_bytes != 0) {
        REQUIRE(c.l3_bytes >= 256u * 1024);
        REQUIRE(c.l3_bytes <= 2048u * 1024 * 1024ull);
    }

    // Level ordering, where both levels were detected. A cache that is not larger
    // than the one above it means the levels were misidentified.
    if (c.l1d_bytes != 0 && c.l2_bytes != 0) REQUIRE(c.l2_bytes >= c.l1d_bytes);
    if (c.l2_bytes  != 0 && c.l3_bytes != 0) REQUIRE(c.l3_bytes >= c.l2_bytes);

    // Detection is a pure read of fixed hardware: two calls must agree, and the
    // cached accessor must agree with a fresh call.
    const util::cache_info again = util::detect_caches();
    REQUIRE(again.l1d_bytes  == c.l1d_bytes);
    REQUIRE(again.l2_bytes   == c.l2_bytes);
    REQUIRE(again.l3_bytes   == c.l3_bytes);
    REQUIRE(again.line_bytes == c.line_bytes);
    REQUIRE(util::cached_cache_info().l1d_bytes == c.l1d_bytes);
    REQUIRE(util::cached_cache_info().l3_bytes  == c.l3_bytes);
}

TEST_CASE("detected_hw_traits overrides only what was detected",
          "[util][cache][blocking]") {
    const util::cache_info  c  = util::detect_caches();
    const simd::hw_traits&  hw = simd::detected_hw_traits();
    const simd::hw_traits&  def = simd::default_hw_traits;

    // Cache fields: the detected value when there is one, the default otherwise.
    // Stated as an if/else on each field so an undetectable platform (non-x86
    // Windows, or a container without sysfs) is asserted to behave exactly as it
    // did before detection existed.
#if defined(MTL5_ENABLE_CACHE_DETECTION)
    if (c.l1d_bytes  != 0) REQUIRE(hw.l1_bytes   == c.l1d_bytes);
    else                   REQUIRE(hw.l1_bytes   == def.l1_bytes);
    if (c.l2_bytes   != 0) REQUIRE(hw.l2_bytes   == c.l2_bytes);
    else                   REQUIRE(hw.l2_bytes   == def.l2_bytes);
    if (c.line_bytes != 0) REQUIRE(hw.line_bytes == c.line_bytes);
    else                   REQUIRE(hw.line_bytes == def.line_bytes);
#else
    // Detection is opt-in and this build did not opt in, so the SHIPPED blocking
    // must be the compile-time defaults on every machine, whatever was detected
    // (#426 measured the detected figures losing by up to 45% on an i7-12700K).
    // Deliberately unconditional: this is the assertion that fails first if the
    // default is ever flipped back without the measurement to justify it.
    REQUIRE(hw.l1_bytes   == def.l1_bytes);
    REQUIRE(hw.l2_bytes   == def.l2_bytes);
    REQUIRE(hw.line_bytes == def.line_bytes);
    REQUIRE(hw.l1_assoc   == def.l1_assoc);
#endif

    // L3 is detected but NOT applied, whatever the machine reports: it feeds only
    // nc, and nc sets the jc block count that the threaded nest partitions
    // round-robin. Applying a detected 15 MB L3 on this class of machine measured
    // a 10-25% regression on wide/short threaded GEMM (njb 8 -> 5 across 2 teams).
    //
    // THE CONDITION THIS USED TO NAME HAS BEEN MET, AND THE ANSWER IS STILL NO.
    // The comment here read "if a future balanced_nc lets L3 back in, this is
    // what fails first" -- i.e. it treated the missing jc-side balancer as the
    // reason L3 was withheld. `balanced_nc` now exists and is the DEFAULT
    // (#429/#490), and the detected L3 is still not applied, because #479
    // measured that pairing directly:
    //
    //     M2 = detected L3 + balanced_nc   102 arms, worst x0.548, 37 regressions
    //     M5 = exact partition             102 arms, worst x0.548, 39 regressions
    //
    // Up to 45% slower on an i7-12700K, and M2 ALREADY INCLUDES the balancing --
    // so balancing does not rescue it. The premise was falsified, not deferred.
    //
    // So this assertion is no longer waiting on anything. Anyone reaching for it
    // needs a new measurement across >= 3 microarchitectures, not the argument
    // that the blocker is gone. See docs/performance/benchmarking-methodology.md
    // for the hypothesis register and what has already been ruled out.
    //
    // Unconditional, so it holds on a machine with a huge L3 as well as one with
    // none.
    REQUIRE(hw.l3_bytes == def.l3_bytes);

    // Non-cache fields are never touched: they select the register tile, which
    // belongs to the compiled micro-kernel.
    REQUIRE(hw.fma_latency   == def.fma_latency);
    REQUIRE(hw.fma_units     == def.fma_units);
    REQUIRE(hw.vec_registers == def.vec_registers);
    REQUIRE(hw.page_bytes    == def.page_bytes);
}

TEMPLATE_TEST_CASE("runtime_blocking keeps the compiled register tile",
                   "[util][cache][blocking]", float, double) {
    const simd::blocking_params& rbp = simd::runtime_blocking<TestType>();
    constexpr simd::blocking_params dbp = simd::default_blocking<TestType>;

    INFO("runtime  mr=" << rbp.mr << " nr=" << rbp.nr
         << " kc=" << rbp.kc << " mc=" << rbp.mc << " nc=" << rbp.nc);
    INFO("default  mr=" << dbp.mr << " nr=" << dbp.nr
         << " kc=" << dbp.kc << " mc=" << dbp.mc << " nc=" << dbp.nc);

    // The load-bearing invariant: gemm_blocked instantiates the micro-kernel on
    // default_blocking's mr/nr while taking kc/mc from here. If detection could
    // move mr or nr, the packed panels would no longer match the kernel.
    REQUIRE(rbp.mr == dbp.mr);
    REQUIRE(rbp.nr == dbp.nr);

    // nc must also be untouched by detection: it is the jc block count, and the
    // threaded nest's team partition is sensitive to it (#429). Note this does
    // NOT follow from withholding L3 -- nc = l3/(kc*sdata) and kc is detected, so
    // a machine with a larger L1 would move nc from an unchanged L3. This caught
    // exactly that on the ARM64 runners, where L1d is 64-128 KB against the 32 KB
    // default; runtime_blocking pins nc for that reason.
    REQUIRE(rbp.nc == dbp.nc);

    // Cache blocks stay well-formed whatever was detected.
    REQUIRE(rbp.kc >= 1);
    REQUIRE(rbp.mc >= 1);
    REQUIRE(rbp.nc >= rbp.nr);
    REQUIRE(rbp.nc % rbp.nr == 0);

    // Two calls return the same cached object.
    REQUIRE(&simd::runtime_blocking<TestType>() == &rbp);

#if !defined(MTL5_ENABLE_CACHE_DETECTION)
    // Detection opt-out (the shipped default): every parameter, not just the
    // register tile, must equal the compile-time derivation -- kc and mc
    // included. This is what makes the shipped GEMM byte-identical to pre-#426.
    REQUIRE(rbp.kc == dbp.kc);
    REQUIRE(rbp.mc == dbp.mc);
#endif
}

TEST_CASE("an undetected cache level keeps the compile-time default",
          "[util][cache][blocking]") {
    // Runs everywhere, unlike the host-dependent fallback branches above: feed
    // `with_detected_caches` hierarchies this machine does not have.
    const simd::hw_traits def = simd::default_hw_traits;

    SECTION("nothing detected at all -> the defaults, unchanged") {
        // A container without sysfs, or non-x86 Windows.
        const simd::hw_traits hw = simd::with_detected_caches(def, util::cache_info{});
        REQUIRE(hw.l1_bytes   == def.l1_bytes);
        REQUIRE(hw.l2_bytes   == def.l2_bytes);
        REQUIRE(hw.l3_bytes   == def.l3_bytes);
        REQUIRE(hw.line_bytes == def.line_bytes);
        REQUIRE(simd::derive_blocking<double>(4, hw).nc
                == simd::derive_blocking<double>(4, def).nc);
    }

    SECTION("no L3 reported -> nc must not collapse to nr") {
        // Apple M-series shape: a large L1/L2 and no L3 in the sysctl sense. Two
        // guards keep the zero away from nc = l3/(kc*sdata), where it would floor
        // at nr and make the jc loop re-stream A every nr columns: L3 is not
        // applied at all today, and the per-field 0 check would hold it back even
        // if it were. Asserted here so removing either one fails.
        util::cache_info m;
        m.l1d_bytes = 128u * 1024;
        m.l2_bytes  = 12u * 1024 * 1024;
        m.l3_bytes  = 0;                      // not detected
        m.line_bytes = 128;

        const simd::hw_traits hw = simd::with_detected_caches(def, m);
        REQUIRE(hw.l1_bytes   == 128u * 1024);       // detected values applied
        REQUIRE(hw.l2_bytes   == 12u * 1024 * 1024);
        REQUIRE(hw.line_bytes == 128);
        REQUIRE(hw.l3_bytes   == def.l3_bytes);      // undetected one held back

        const auto bp = simd::derive_blocking<double>(4, hw);
        REQUIRE(bp.nc > bp.nr);                      // the collapse this prevents
        REQUIRE(bp.kc > simd::derive_blocking<double>(4, def).kc);   // bigger L1 -> bigger kc
        REQUIRE(bp.mc > simd::derive_blocking<double>(4, def).mc);   // bigger L2 -> bigger mc
    }
}

TEST_CASE("derive_blocking tracks the cache figures it is given",
          "[util][cache][blocking]") {
    // The point of the change, exercised without depending on the host: a larger
    // L3 must produce a larger nc, and a larger L2 a larger mc. Pinned here
    // because the host-dependent tests above cannot assert a direction.
    simd::hw_traits small = simd::default_hw_traits;
    simd::hw_traits big   = simd::default_hw_traits;
    big.l3_bytes = small.l3_bytes * 2;

    const auto bp_small = simd::derive_blocking<double>(4, small);
    const auto bp_big   = simd::derive_blocking<double>(4, big);
    REQUIRE(bp_big.nc > bp_small.nc);
    REQUIRE(bp_big.mr == bp_small.mr);   // cache size must not move the tile
    REQUIRE(bp_big.nr == bp_small.nr);
    REQUIRE(bp_big.kc == bp_small.kc);   // nor the L1-sized block

    simd::hw_traits big_l2 = simd::default_hw_traits;
    big_l2.l2_bytes = small.l2_bytes * 2;
    REQUIRE(simd::derive_blocking<double>(4, big_l2).mc > bp_small.mc);
}

TEST_CASE("a shared cache contributes only its per-core share",
          "[util][cache][blocking]") {
    // #432: an Alder Lake E-cluster publishes ONE 2 MiB L2 for four cores. Taken
    // at face value it sized mc for roughly 4x the L2 a core actually has.
    const simd::hw_traits def = simd::default_hw_traits;

    util::cache_info cluster;
    cluster.l1d_bytes = 32u * 1024;  cluster.l1d_sharing_cores = 1;
    cluster.l2_bytes  = 2u * 1024 * 1024;
    cluster.l2_sharing_cores = 4;                    // four cores, one L2
    const simd::hw_traits hw = simd::with_detected_caches(def, cluster);
    REQUIRE(hw.l2_bytes == 512u * 1024);             // 2 MiB / 4, not 2 MiB
    REQUIRE(hw.l1_bytes == 32u * 1024);              // private: undivided

    // An SMT pair must NOT be discounted: siblings share L1d/L2, but the pinning
    // policy runs one thread per physical core and leaves the sibling idle. This
    // is why cache_info counts CORES rather than logical CPUs -- counting CPUs
    // would halve both on every hyperthreaded machine.
    util::cache_info smt;
    smt.l1d_bytes = 32u * 1024;   smt.l1d_sharing_cores = 1;   // 2 CPUs, 1 core
    smt.l2_bytes  = 256u * 1024;  smt.l2_sharing_cores  = 1;
    const simd::hw_traits hw_smt = simd::with_detected_caches(def, smt);
    REQUIRE(hw_smt.l1_bytes == 32u * 1024);
    REQUIRE(hw_smt.l2_bytes == 256u * 1024);

    // Unknown sharing (0) is treated as private rather than dividing by zero.
    util::cache_info unknown;
    unknown.l2_bytes = 1024u * 1024;                 // sharing_cores left 0
    REQUIRE(simd::with_detected_caches(def, unknown).l2_bytes == 1024u * 1024);
}

TEST_CASE("detection is reproducible and independent of the running core",
          "[util][cache][blocking]") {
    // The #432 defect: CPUID describes whichever core the thread is on, so on a
    // hybrid CPU the same binary reported a P-core or an E-core hierarchy run to
    // run. Detection must be a function of the machine (and the affinity mask),
    // not of where the scheduler put this thread.
    const util::cache_info a = util::detect_caches();
    const util::cache_info b = util::detect_caches();
    REQUIRE(a.l1d_bytes == b.l1d_bytes);
    REQUIRE(a.l2_bytes  == b.l2_bytes);
    REQUIRE(a.l3_bytes  == b.l3_bytes);
    REQUIRE(a.l1d_sharing_cores == b.l1d_sharing_cores);
    REQUIRE(a.l2_sharing_cores  == b.l2_sharing_cores);

    INFO("l1d=" << a.l1d_bytes << "/" << a.l1d_sharing_cores
         << " l2=" << a.l2_bytes << "/" << a.l2_sharing_cores
         << " l3=" << a.l3_bytes << "/" << a.l3_sharing_cores);

    // Sharing widens monotonically with level: a cache cannot be shared by fewer
    // cores than the one above it. This is a property of the topology itself, so
    // it holds on any machine. Deliberately NOT compared against
    // hardware_concurrency() -- that is a concurrency hint with no defined
    // relationship to cache topology, it varies with libstdc++ version and cgroup
    // limits, and comparing a sysfs physical-core count to it would pass here by
    // luck of toolchain rather than by being true.
    if (a.l1d_sharing_cores != 0 && a.l2_sharing_cores != 0)
        REQUIRE(a.l2_sharing_cores >= a.l1d_sharing_cores);
    if (a.l2_sharing_cores != 0 && a.l3_sharing_cores != 0)
        REQUIRE(a.l3_sharing_cores >= a.l2_sharing_cores);
}

TEST_CASE("a detected L1 moves nc even with L3 held fixed",
          "[util][cache][blocking]") {
    // Why runtime_blocking pins nc rather than merely withholding l3_bytes.
    // nc = round_down(l3/(kc*sdata), nr) and kc = (l1/2)/(nr*sdata), so a larger
    // L1 raises kc and LOWERS nc from an unchanged L3. Withholding L3 alone would
    // therefore still let the jc block count drift on any machine whose L1 is not
    // the 32 KB default -- which is every ARM64 target in CI, and is why this only
    // showed up there. Asserted on explicit traits so it runs on x86 as well.
    simd::hw_traits base = simd::default_hw_traits;
    simd::hw_traits big_l1 = base;
    big_l1.l1_bytes = base.l1_bytes * 4;          // 32 KB -> 128 KB, M-series shape
    REQUIRE(big_l1.l3_bytes == base.l3_bytes);    // L3 deliberately identical

    const auto bp_base = simd::derive_blocking<double>(4, base);
    const auto bp_l1   = simd::derive_blocking<double>(4, big_l1);
    REQUIRE(bp_l1.kc > bp_base.kc);               // bigger L1 -> bigger kc
    REQUIRE(bp_l1.nc < bp_base.nc);               // ... which drags nc down

    // And the pin holds regardless: runtime_blocking reports the compile-time nc.
    REQUIRE(simd::runtime_blocking<double>().nc == simd::default_blocking<double>.nc);
    REQUIRE(simd::runtime_blocking<float>().nc  == simd::default_blocking<float>.nc);
}

// --- detection level groups (#430 follow-up) --------------------------------
//
// L1 feeds kc and L2 feeds mc, and the four-machine A/B says the two do not
// behave alike: the Ryzen run moved mc alone (kc identical, mc 64 -> 256) at no
// single-thread cost, while both machines whose kc moved lost throughput. That
// was an inference across machines that each happened to vary something
// different; splitting the switch turns it into an experiment on one machine.
//
// These tests pin the SEPARATION -- that each group moves its own fields and
// leaves the other's alone -- which is the property the kconly/mconly arms rest
// on. If it broke, both arms would still run and would quietly measure the same
// thing.
TEST_CASE("detection levels are separable", "[util][cache_info][blocking]") {
    using mtl::simd::detect_levels;
    using mtl::simd::with_detected_caches;

    const simd::hw_traits def = simd::default_hw_traits;

    // A machine unlike the defaults in BOTH levels, so either leak is visible.
    util::cache_info c{};
    c.l1d_bytes = 4 * def.l1_bytes;
    c.l1d_assoc = def.l1_assoc + 4;
    c.l2_bytes  = 4 * def.l2_bytes;
    c.line_bytes = def.line_bytes;      // 64 on every machine measured so far
    c.l1d_sharing_cores = 1;
    c.l2_sharing_cores  = 1;

    SECTION("l1 only moves the kc inputs") {
        const auto hw = with_detected_caches(def, c, detect_levels::l1);
        REQUIRE(hw.l1_bytes == c.l1d_bytes);
        REQUIRE(hw.l1_assoc == c.l1d_assoc);
        REQUIRE(hw.l2_bytes == def.l2_bytes);      // mc input untouched
    }
    SECTION("l2 only moves the mc input") {
        const auto hw = with_detected_caches(def, c, detect_levels::l2);
        REQUIRE(hw.l2_bytes == c.l2_bytes);
        REQUIRE(hw.l1_bytes == def.l1_bytes);      // kc inputs untouched
        REQUIRE(hw.l1_assoc == def.l1_assoc);
    }
    SECTION("both is the union, and the default") {
        const auto both = with_detected_caches(def, c, detect_levels::both);
        REQUIRE(both.l1_bytes == c.l1d_bytes);
        REQUIRE(both.l2_bytes == c.l2_bytes);
        // The unparameterised call is what every pre-split caller compiles to.
        const auto implicit_both = with_detected_caches(def, c);
        REQUIRE(implicit_both.l1_bytes == both.l1_bytes);
        REQUIRE(implicit_both.l2_bytes == both.l2_bytes);
    }
    SECTION("none changes nothing") {
        const auto hw = with_detected_caches(def, c, detect_levels::none);
        REQUIRE(hw.l1_bytes == def.l1_bytes);
        REQUIRE(hw.l2_bytes == def.l2_bytes);
        REQUIRE(hw.l1_assoc == def.l1_assoc);
    }
    SECTION("the arms are not measuring the same thing") {
        // The point of the split: on a machine whose caches differ from the
        // model, kc-only and mc-only must produce DIFFERENT blocking. Same
        // hierarchy, one level each -- if these ever agreed, the experiment
        // would be a null run that looks like a result.
        const auto kc_only = mtl::simd::derive_blocking<double>(
            4, with_detected_caches(def, c, detect_levels::l1));
        const auto mc_only = mtl::simd::derive_blocking<double>(
            4, with_detected_caches(def, c, detect_levels::l2));
        REQUIRE(kc_only.kc != mc_only.kc);
        REQUIRE(kc_only.mc != mc_only.mc);
    }
}
