// MTL5 Benchmark -- native LU vs SuiteSparse SuperLU scoreboard (Phase 0, #180)
//
// Establishes the performance scoreboard for the native supernodal-LU parity
// effort (docs/plans/native-superlu-performance.md). There is no native
// supernodal LU yet, so the native column here is MTL5's existing native
// general-unsymmetric solver -- native KLU (BTF + per-block Gilbert-Peierls LU,
// the same baseline bench_klu uses). For each matrix it reports, for native LU
// and (when built with MTL5_HAS_SUPERLU) external SuperLU:
//   - BTF block count and largest block (native pipeline; structural)
//   - factor+solve time
//   - factor fill (nnz of L+U) -- the memory proxy (peak-memory instrumentation
//     is deferred to a later phase; fill is what bench_klu tracks too)
//   - residual ||Ax-b||_inf / ||b||_inf
//   - the native / SuperLU ratio (the number Phase 5 drives to <= 1.5x)
//
// SuperLU is a supernodal (BLAS-3) solver while native LU is scalar
// non-supernodal, so the time ratio is expected to GROW with dense fill
// structure -- that kernel-class gap is exactly what this scoreboard quantifies
// and what the native supernodal LU (Phases 1-3) must close.
//
// Matrices: built-in 2D convection-diffusion grids (unsymmetric) by default,
// plus any Matrix Market files passed on the command line (e.g. SuiteSparse
// unsymmetric matrices fetched via benchmarks/fetch_superlu_matrices.sh).
//
// Usage:
//   bench_superlu                       # built-in conv-diffusion suite (32^2 .. 256^2)
//   bench_superlu A.mtx B.mtx ...       # those matrices instead
//   bench_superlu ext:Big.mtx           # external-only row: skip native (too
//                                       #   slow to finish on a huge dense block)
//   bench_superlu --csv out.csv [mtx]   # also write a CSV scoreboard

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/io/matrix_market.hpp>
#include <mtl/sparse/ordering/dulmage_mendelsohn.hpp>
#include <mtl/sparse/factorization/native_klu.hpp>

#ifdef MTL5_HAS_SUPERLU
#include <mtl/interface/superlu.hpp>
#endif

namespace {

using clk = std::chrono::steady_clock;
template <typename F>
double seconds(F&& f) {
    auto t0 = clk::now();
    f();
    return std::chrono::duration<double>(clk::now() - t0).count();
}

double rel_residual(const mtl::mat::compressed2D<double>& A,
                    const mtl::vec::dense_vector<double>& x,
                    const mtl::vec::dense_vector<double>& b) {
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& dat = A.ref_data();
    double r = 0.0, bn = 0.0;
    for (std::size_t i = 0; i < A.num_rows(); ++i) {
        double ax = 0.0;
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k)
            ax += dat[k] * x(static_cast<int>(ci[k]));
        double d = ax - b(static_cast<int>(i));
        r = std::max(r, std::abs(d));
        bn = std::max(bn, std::abs(b(static_cast<int>(i))));
    }
    return (bn > 0.0) ? r / bn : r;
}

// Unsymmetric 2D convection-diffusion on an N x N grid (n = N*N): 5-point
// central diffusion + first-order upwind advection with a constant wind
// (bx, by) > 0. The upwind term makes the west/north coupling differ from
// east/south, so the matrix is genuinely unsymmetric; it is a diagonally
// dominant M-matrix (strictly dominant at the Dirichlet boundary), hence
// nonsingular. This is the unsymmetric analogue of bench_klu's 2D Poisson.
mtl::mat::compressed2D<double> conv_diffusion_2d(std::size_t N,
                                                double bx = 1.0,
                                                double by = 1.0) {
    std::size_t n = N * N;
    mtl::mat::compressed2D<double> A(n, n);
    {
        mtl::mat::inserter<mtl::mat::compressed2D<double>> ins(A);
        auto id = [N](std::size_t r, std::size_t c) { return r * N + c; };
        for (std::size_t r = 0; r < N; ++r) {
            for (std::size_t c = 0; c < N; ++c) {
                std::size_t i = id(r, c);
                ins[i][i] << (4.0 + bx + by);
                if (c > 0)     ins[i][id(r, c - 1)] << -(1.0 + bx);  // west (upwind)
                if (c + 1 < N) ins[i][id(r, c + 1)] << -1.0;         // east
                if (r > 0)     ins[i][id(r - 1, c)] << -(1.0 + by);  // north (upwind)
                if (r + 1 < N) ins[i][id(r + 1, c)] << -1.0;         // south
            }
        }
    }
    return A;
}

struct Score {
    std::string name;
    std::size_t n = 0, nnz = 0;
    std::size_t nblocks = 0, largest_block = 0;
    double native_s = 0.0, native_resid = 0.0;
    std::size_t native_fill = 0;
    bool native_ok = false;
    bool native_skipped = false;   // external-only matrix (native too slow to run)
#ifdef MTL5_HAS_SUPERLU
    double ext_s = 0.0, ext_resid = 0.0;
    std::size_t ext_fill = 0;
    bool ext_ok = false;
#endif
};

Score score_matrix(const std::string& name,
                   const mtl::mat::compressed2D<double>& A,
                   bool run_native = true) {
    Score s;
    s.name = name;
    s.n = A.num_rows();
    s.nnz = A.nnz();

    // RHS: b = A * ones (exact solution all-ones).
    std::size_t n = s.n;
    mtl::vec::dense_vector<double> ones(n, 1.0), b(n, 0.0);
    {
        const auto& rp = A.ref_major();
        const auto& ci = A.ref_minor();
        const auto& dat = A.ref_data();
        for (std::size_t i = 0; i < n; ++i) {
            double acc = 0.0;
            for (std::size_t k = rp[i]; k < rp[i + 1]; ++k)
                acc += dat[k] * ones(static_cast<int>(ci[k]));
            b(static_cast<int>(i)) = acc;
        }
    }

    // Structural: BTF block stats (native pipeline).
    {
        auto btf = mtl::sparse::ordering::block_triangular_form(A);
        s.nblocks = btf.nblocks();
        for (std::size_t b2 = 0; b2 < btf.nblocks(); ++b2)
            s.largest_block = std::max(s.largest_block,
                                       btf.blocks[b2 + 1] - btf.blocks[b2]);
    }

    // Native LU (KLU: BTF + per-block GP-LU) factor + solve. Skipped for
    // external-only matrices known not to finish in reasonable time.
    if (!run_native) {
        s.native_skipped = true;
    } else {
        try {
            mtl::vec::dense_vector<double> x(n, 0.0);
            mtl::sparse::factorization::klu_numeric<double> fac;
            s.native_s = seconds([&] {
                fac = mtl::sparse::factorization::native_klu_factor(A);
                fac.solve(x, b);
            });
            for (const auto& blk : fac.block_numeric)
                s.native_fill += blk.L.nnz() + blk.U.nnz();
            s.native_resid = rel_residual(A, x, b);
            s.native_ok = true;
        } catch (const std::exception& e) {
            std::fprintf(stderr, "  [native LU failed on %s: %s]\n", name.c_str(), e.what());
        }
    }

#ifdef MTL5_HAS_SUPERLU
    try {
        mtl::vec::dense_vector<double> x(n, 0.0);
        // Construct the solver (CCS conversion) and solve, timed together to
        // match the native factor+solve measurement; read its factor nnz after.
        s.ext_s = seconds([&] {
            mtl::interface::superlu_solver solver(A);
            solver.solve(x, b);
            s.ext_fill = solver.factor_nnz();
        });
        s.ext_resid = rel_residual(A, x, b);
        s.ext_ok = true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "  [SuperLU failed on %s: %s]\n", name.c_str(), e.what());
    }
#endif
    return s;
}

void print_header() {
    std::printf("\n%-22s %9s %11s %8s %11s  %10s %12s",
                "matrix", "n", "nnz", "blocks", "maxblock",
                "native(s)", "native_fill");
#ifdef MTL5_HAS_SUPERLU
    std::printf("  %10s %12s %8s %8s", "SuperLU(s)", "SuperLU_fill", "t_ratio", "f_ratio");
#endif
    std::printf("\n%s\n", std::string(132, '-').c_str());
}

void print_row(const Score& s) {
    std::printf("%-22s %9zu %11zu %8zu %11zu  ",
                s.name.c_str(), s.n, s.nnz, s.nblocks, s.largest_block);
    if (s.native_ok)            std::printf("%10.3f %12zu", s.native_s, s.native_fill);
    else if (s.native_skipped)  std::printf("%10s %12s", "skip", "-");
    else                        std::printf("%10s %12s", "FAIL", "-");
#ifdef MTL5_HAS_SUPERLU
    if (s.ext_ok) {
        std::printf("  %10.3f %12zu", s.ext_s, s.ext_fill);
        if (s.native_ok && s.ext_s > 0.0)         std::printf(" %7.1fx", s.native_s / s.ext_s);
        else                                      std::printf(" %8s", "-");
        if (s.native_ok && s.ext_fill > 0)        std::printf(" %7.1fx",
                                                      double(s.native_fill) / double(s.ext_fill));
        else                                      std::printf(" %8s", "-");
    } else {
        std::printf("  %10s %12s %8s %8s", "FAIL", "-", "-", "-");
    }
#endif
    std::printf("\n");
}

// Returns false (with a message on stderr) if the CSV could not be written, so
// the caller never reports a CSV artifact that was silently dropped.
bool write_csv(const std::string& path, const std::vector<Score>& rows) {
    std::ofstream out(path);
    if (!out) {
        std::fprintf(stderr, "  [failed to open CSV output: %s]\n", path.c_str());
        return false;
    }
    // native_status is ok|skip|fail; numeric native fields are left EMPTY when
    // not ok, so "skipped/failed" is never confused with a real value of 0.
    out << "matrix,n,nnz,nblocks,largest_block,native_status,native_s,native_fill,native_resid";
#ifdef MTL5_HAS_SUPERLU
    out << ",ext_status,ext_s,ext_fill,ext_resid,time_ratio,fill_ratio";
#endif
    out << "\n";
    for (const auto& s : rows) {
        const char* nstatus = s.native_ok ? "ok" : (s.native_skipped ? "skip" : "fail");
        out << s.name << ',' << s.n << ',' << s.nnz << ',' << s.nblocks << ','
            << s.largest_block << ',' << nstatus << ',';
        if (s.native_ok) out << s.native_s << ',' << s.native_fill << ',' << s.native_resid;
        else             out << ",,";   // empty native_s, native_fill, native_resid
#ifdef MTL5_HAS_SUPERLU
        out << ',' << (s.ext_ok ? "ok" : "fail") << ',';
        if (s.ext_ok) out << s.ext_s << ',' << s.ext_fill << ',' << s.ext_resid;
        else          out << ",,";      // empty ext_s, ext_fill, ext_resid
        out << ',';
        if (s.native_ok && s.ext_ok && s.ext_s > 0.0) out << (s.native_s / s.ext_s);
        out << ',';
        if (s.native_ok && s.ext_ok && s.ext_fill > 0)
            out << (double(s.native_fill) / double(s.ext_fill));
        // else: empty ratios
#endif
        out << "\n";
    }
    out.flush();
    if (!out) {
        std::fprintf(stderr, "  [failed while writing CSV output: %s]\n", path.c_str());
        return false;
    }
    return true;
}

} // namespace

int main(int argc, char** argv) {
    std::string csv_path;
    std::vector<std::string> mtx_files;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--csv") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "bench_superlu: --csv requires a file path\n");
                return 2;
            }
            csv_path = argv[++i];
        }
        else { mtx_files.push_back(a); }
    }

#ifdef MTL5_HAS_SUPERLU
    std::printf("Native LU vs SuiteSparse SuperLU scoreboard (external SuperLU ENABLED)\n");
#else
    std::printf("Native LU scoreboard (external SuperLU NOT built; "
                "configure -DMTL5_WITH_SUPERLU=ON to compare)\n");
#endif

    std::vector<Score> rows;
    print_header();

    if (mtx_files.empty()) {
        // Built-in 2D convection-diffusion suite (unsymmetric).
        for (std::size_t N : {32u, 64u, 128u, 256u}) {
            auto A = conv_diffusion_2d(N);
            char nm[40]; std::snprintf(nm, sizeof nm, "convdiff_%zux%zu", N, N);
            rows.push_back(score_matrix(nm, A));
            print_row(rows.back());
        }
    } else {
        for (const auto& arg : mtx_files) {
            // "ext:<file>" marks a matrix external-only (skip native, too slow).
            bool run_native = true;
            std::string f = arg;
            if (f.rfind("ext:", 0) == 0) { run_native = false; f = f.substr(4); }

            mtl::mat::compressed2D<double> A;
            double load = 0.0;
            try { load = seconds([&] { A = mtl::io::mm_read<double>(f); }); }
            catch (const std::exception& e) {
                std::fprintf(stderr, "  [load failed %s: %s]\n", f.c_str(), e.what());
                continue;
            }
            // label = filename without directory or extension
            std::string nm = std::filesystem::path(f).stem().string();
            std::fprintf(stderr, "  [loaded %s in %.2fs]\n", nm.c_str(), load);
            rows.push_back(score_matrix(nm, A, run_native));
            print_row(rows.back());
        }
    }

    if (!csv_path.empty()) {
        if (!write_csv(csv_path, rows)) return 1;
        std::printf("\nCSV: %s\n", csv_path.c_str());
    }
    return 0;
}
