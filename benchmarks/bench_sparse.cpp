// MTL5 -- sparse triangular-solve scaling benchmark (#297 phase 2).
//
// Measures the benefit of the level-scheduled sparse triangular solves (sparse
// Cholesky / LDL^T / LU and the supernodal LDL^T / LU) delivered in #297. The
// WIN is in the SOLVE phase, so the protocol isolates it:
//   1. build the matrix (synthetic, or a .mtx via --file)
//   2. symbolic + numeric factor ONCE, untimed
//   3. time num.solve(x, b) -- median of N iterations (the repeated-solve /
//      transient model: one analyze+factor, many solves)
//   4. also time the numeric factor so the results doc can compute the
//      solve-count break-even (factor_time / solve_time)
//   5. report the relative residual ||Ax-b||/||b|| as a correctness gauge
//
// Thread count is a per-process axis: run under `MTL5_NUM_THREADS=T` (pinned to
// physical cores via benchmarks/run_scaling.sh) and compare T against T=1. The
// solves are bit-identical across thread counts by construction (proven in CI);
// here we measure how far level scheduling turns that into wall-clock speedup,
// which tracks the matrix's level structure (few wide levels scale, deep narrow
// levels are scheduling-limited).
//
// Usage:
//   bench_sparse                      # synthetic suite (default)
//   bench_sparse --csv out.csv        # also write a CSV scoreboard
//   bench_sparse --label native-t8    # backend label (run_scaling sets this)
//   bench_sparse --file A.mtx [B.mtx] # add SPD/general matrices from disk
//   bench_sparse --sizes 100,150      # 2-D grid side lengths (default 100,160)

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/io/matrix_market.hpp>
#include <mtl/sparse/ordering/amd.hpp>
#include <mtl/sparse/ordering/colamd.hpp>
#include <mtl/sparse/factorization/sparse_cholesky.hpp>
#include <mtl/sparse/factorization/sparse_ldlt.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>
#include <mtl/sparse/factorization/supernodal_ldlt.hpp>
#include <mtl/sparse/factorization/supernodal_lu.hpp>

#include "harness/timer.hpp"
#include "harness/reporter.hpp"

using mtl::mat::compressed2D;
using mtl::vec::dense_vector;
namespace fact = mtl::sparse::factorization;
namespace ord  = mtl::sparse::ordering;

// ---------------------------------------------------------------------------
// Matrix generators spanning level structures
// ---------------------------------------------------------------------------

// 2-D 5-point Laplacian on a g x g grid (n = g*g): SPD, MANY NARROW levels
// (deep dependency chain) -> scheduling-limited, the pessimistic case.
static compressed2D<double> laplacian_2d(std::size_t g) {
    const std::size_t n = g * g;
    compressed2D<double> A(n, n);
    mtl::mat::inserter<compressed2D<double>> ins(A);
    auto id = [g](std::size_t r, std::size_t c) { return r * g + c; };
    for (std::size_t r = 0; r < g; ++r)
        for (std::size_t c = 0; c < g; ++c) {
            const std::size_t i = id(r, c);
            ins[i][i] << 4.0;
            if (r + 1 < g) { ins[i][id(r + 1, c)] << -1.0; ins[id(r + 1, c)][i] << -1.0; }
            if (c + 1 < g) { ins[i][id(r, c + 1)] << -1.0; ins[id(r, c + 1)][i] << -1.0; }
        }
    return A;
}

// 3-D 7-point Laplacian on a g x g x g grid (n = g^3): SPD, WIDER levels than
// 2-D -> more parallelism per level.
static compressed2D<double> laplacian_3d(std::size_t g) {
    const std::size_t n = g * g * g;
    compressed2D<double> A(n, n);
    mtl::mat::inserter<compressed2D<double>> ins(A);
    auto id = [g](std::size_t x, std::size_t y, std::size_t z) { return (x * g + y) * g + z; };
    for (std::size_t x = 0; x < g; ++x)
        for (std::size_t y = 0; y < g; ++y)
            for (std::size_t z = 0; z < g; ++z) {
                const std::size_t i = id(x, y, z);
                ins[i][i] << 6.0;
                if (x + 1 < g) { ins[i][id(x+1,y,z)] << -1.0; ins[id(x+1,y,z)][i] << -1.0; }
                if (y + 1 < g) { ins[i][id(x,y+1,z)] << -1.0; ins[id(x,y+1,z)][i] << -1.0; }
                if (z + 1 < g) { ins[i][id(x,y,z+1)] << -1.0; ins[id(x,y,z+1)][i] << -1.0; }
            }
    return A;
}

// Arrow SPD: dense first row/col + heavy diagonal, n. FEW WIDE levels -> the
// optimistic (well-parallelizing) case for the solve.
static compressed2D<double> arrow_spd(std::size_t n) {
    compressed2D<double> A(n, n);
    mtl::mat::inserter<compressed2D<double>> ins(A);
    ins[0][0] << static_cast<double>(n);            // heavy pivot to stay SPD
    for (std::size_t i = 1; i < n; ++i) {
        ins[0][i] << 1.0; ins[i][0] << 1.0;
        ins[i][i] << 4.0;
    }
    return A;
}

// Unsymmetric, diagonally dominant banded matrix (nonsingular, stable LU):
// diag 8, sub -1/-0.5, super -2.
static compressed2D<double> unsym_banded(std::size_t n) {
    compressed2D<double> A(n, n);
    mtl::mat::inserter<compressed2D<double>> ins(A);
    for (std::size_t i = 0; i < n; ++i) {
        ins[i][i] << 8.0;
        if (i >= 1)    ins[i][i - 1] << -1.0;
        if (i >= 2)    ins[i][i - 2] << -0.5;
        if (i + 1 < n) ins[i][i + 1] << -2.0;
    }
    return A;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static double rel_residual(const compressed2D<double>& A,
                           const dense_vector<double>& x,
                           const dense_vector<double>& b) {
    const std::size_t n = b.size();
    const auto& starts = A.ref_major();
    const auto& idx    = A.ref_minor();
    const auto& data   = A.ref_data();
    double rn = 0.0, bn = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        double ax = 0.0;
        for (std::size_t k = starts[i]; k < starts[i + 1]; ++k)
            ax += data[k] * x(static_cast<int>(idx[k]));
        const double ri = ax - b(static_cast<int>(i));
        rn += ri * ri;
        bn += b(static_cast<int>(i)) * b(static_cast<int>(i));
    }
    return bn == 0.0 ? std::sqrt(rn) : std::sqrt(rn) / std::sqrt(bn);
}

static dense_vector<double> make_rhs(std::size_t n) {
    dense_vector<double> b(n);
    for (std::size_t i = 0; i < n; ++i)
        b(static_cast<int>(i)) = 0.5 + std::sin(0.9 * static_cast<double>(i));
    return b;
}

// Factor once (untimed), then time solve(); report residual + a CSV row whose
// "gflops" column carries the solve THROUGHPUT derived from the FACTOR nnz (so
// fill-in is counted, not just the input A). The throughput ratio across thread
// counts == the wall-clock speedup either way; using the factor nnz makes the
// absolute magnitude an honest solve rate. FactorFn(A) -> a factored solver
// exposing .solve(x, b); FlopsFn(num) -> the solve's flop estimate from its
// factors (~2 flops/entry x the triangular traversals the solve performs).
template <typename FactorFn, typename FlopsFn>
static void bench_case(mtl::bench::reporter& rep, const std::string& op,
                       const std::string& label, const compressed2D<double>& A,
                       FactorFn&& factor, FlopsFn&& solve_flops_of) {
    const std::size_t n = A.num_rows();
    dense_vector<double> b = make_rhs(n);
    dense_vector<double> x(n);

    auto num = factor(A);                          // symbolic + numeric, UNTIMED
    const double solve_flops = solve_flops_of(num);  // from the factors (with fill)
    num.solve(x, b);                               // warm + residual sample
    const double rr = rel_residual(A, x, b);

    auto t = mtl::bench::measure([&]{ num.solve(x, b); },
                                 op, label, n, solve_flops,
                                 /*warmup=*/3, /*iterations=*/25);
    rep.add(t);
    std::printf("  %-26s n=%-8zu solve=%9.1f us  Gflop/s=%6.2f  resid=%.2e\n",
                op.c_str(), n, t.median_ns / 1e3, t.gflops, rr);
}

// Solve-flop estimates from the factors (~2 flops per stored entry per
// triangular traversal). L-only methods do a forward + transpose solve of L
// (two traversals); LU does a forward L + back U (one each).
static const auto chol_flops = [](const auto& num) { return 4.0 * static_cast<double>(num.factor().nnz()); };
static const auto ltri_flops = [](const auto& num) { return 4.0 * static_cast<double>(num.factorL().nnz()); };
static const auto lu_flops   = [](const auto& num) {
    return 2.0 * static_cast<double>(num.factorL().nnz() + num.factorU().nnz());
};

// ---------------------------------------------------------------------------
// Suite
// ---------------------------------------------------------------------------

static void run_suite(mtl::bench::reporter& rep, const std::string& label,
                      const std::vector<std::size_t>& grid_sizes) {
    auto chol = [](const compressed2D<double>& A) {
        return fact::sparse_cholesky_numeric(A, fact::sparse_cholesky_symbolic(A, ord::amd{}));
    };
    auto ldlt = [](const compressed2D<double>& A) {
        return fact::sparse_ldlt_numeric(A, fact::sparse_ldlt_symbolic(A, ord::amd{}));
    };
    auto snldlt = [](const compressed2D<double>& A) {
        return fact::supernodal_ldlt_numeric(A, fact::supernodal_ldlt_symbolic(A, ord::amd{}));
    };
    auto lu = [](const compressed2D<double>& A) {
        return fact::sparse_lu_numeric(A, fact::sparse_lu_symbolic(A, ord::colamd{}));
    };
    auto snlu = [](const compressed2D<double>& A) {
        return fact::supernodal_lu_numeric(A, fact::supernodal_lu_symbolic_analyze(A, ord::colamd{}));
    };

    for (std::size_t g : grid_sizes) {
        // 2-D Laplacian (many narrow levels): the scheduling-limited case.
        auto L2 = laplacian_2d(g);
        const std::string t2 = "lap2d" + std::to_string(g);
        bench_case(rep, "chol "   + t2, label, L2, chol,   chol_flops);
        bench_case(rep, "ldlt "   + t2, label, L2, ldlt,   ltri_flops);
        bench_case(rep, "snldlt " + t2, label, L2, snldlt, ltri_flops);
        bench_case(rep, "lu "     + t2, label, L2, lu,     lu_flops);
        bench_case(rep, "snlu "   + t2, label, L2, snlu,   lu_flops);

        // 3-D Laplacian (wider levels than 2-D): more parallelism per level.
        // g3 chosen so n3 ~ n2 (comparable problem size) but bounded.
        const std::size_t g3 = std::max<std::size_t>(8, static_cast<std::size_t>(std::cbrt(
                                   static_cast<double>(g) * static_cast<double>(g))));
        auto L3 = laplacian_3d(g3);
        const std::string t3 = "lap3d" + std::to_string(g3);
        bench_case(rep, "chol "   + t3, label, L3, chol,   chol_flops);
        bench_case(rep, "snldlt " + t3, label, L3, snldlt, ltri_flops);
    }

    // Arrow SPD (one wide level): the optimistic solve-scaling case.
    {
        const std::size_t n = grid_sizes.back() * grid_sizes.back() * 2;
        auto AR = arrow_spd(n);
        bench_case(rep, "chol arrow", label, AR, chol, chol_flops);
    }

    // Unsymmetric banded (sequential recurrence): LU / supernodal LU on a
    // genuinely non-SPD system -- the no-parallelism reference.
    {
        const std::size_t n = grid_sizes.back() * grid_sizes.back() * 4;
        auto UB = unsym_banded(n);
        bench_case(rep, "lu banded",   label, UB, lu,   lu_flops);
        bench_case(rep, "snlu banded", label, UB, snlu, lu_flops);
    }
}

int main(int argc, char** argv) {
    std::string csv, label = "native";
    std::vector<std::size_t> grid_sizes;
    std::vector<std::string> files;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--csv") == 0 && i + 1 < argc)        csv = argv[++i];
        else if (std::strcmp(argv[i], "--label") == 0 && i + 1 < argc) label = argv[++i];
        else if (std::strcmp(argv[i], "--file") == 0 && i + 1 < argc)  files.push_back(argv[++i]);
        else if (std::strcmp(argv[i], "--sizes") == 0 && i + 1 < argc) {
            const char* s = argv[++i];
            std::size_t v = 0; bool any = false;
            for (const char* p = s; ; ++p) {
                if (*p >= '0' && *p <= '9') { v = v * 10 + std::size_t(*p - '0'); any = true; }
                else { if (any) grid_sizes.push_back(v); v = 0; any = false; if (!*p) break; }
            }
        }
    }
    if (grid_sizes.empty()) grid_sizes = {100, 160};   // n = 10000, 25600
    // Reject non-positive grid sizes: arrow_spd(0) / laplacian_*(0) would index
    // an empty matrix. A grid side < 2 has no interior structure worth timing.
    for (std::size_t g : grid_sizes)
        if (g < 2) { std::fprintf(stderr, "error: --sizes entries must be >= 2 (got %zu)\n", g); return 2; }

    std::printf("=== sparse triangular-solve scaling (label=%s) ===\n", label.c_str());
    mtl::bench::reporter rep;
    run_suite(rep, label, grid_sizes);

    // Optional real matrices from disk (SPD -> Cholesky; general -> LU). We do
    // not know SPD-ness a priori, so run LU (always valid) and, if it factors,
    // Cholesky too.
    for (const auto& f : files) {
        compressed2D<double> A;
        try { A = mtl::io::mm_read<double>(f); }
        catch (const std::exception& e) { std::fprintf(stderr, "  [skip %s: %s]\n", f.c_str(), e.what()); continue; }
        const std::string nm = "file:" + f;
        // LU throws for rectangular (symbolic) or singular (numeric) inputs;
        // isolate each factorization so one bad file doesn't drop the rest / CSV.
        try {
            bench_case(rep, "lu " + nm, label, A,
                [](const compressed2D<double>& M) {
                    return fact::sparse_lu_numeric(M, fact::sparse_lu_symbolic(M, ord::colamd{}));
                }, lu_flops);
        } catch (const std::exception& e) { std::fprintf(stderr, "  [%s LU failed: %s]\n", f.c_str(), e.what()); }
        try {
            bench_case(rep, "chol " + nm, label, A,
                [](const compressed2D<double>& M) {
                    return fact::sparse_cholesky_numeric(M, fact::sparse_cholesky_symbolic(M, ord::amd{}));
                }, chol_flops);
        } catch (const std::exception& e) { std::fprintf(stderr, "  [%s not SPD: %s]\n", f.c_str(), e.what()); }
    }

    if (!csv.empty()) { rep.write_csv(csv); std::printf("\nResults written to: %s\n", csv.c_str()); }
    return 0;
}
