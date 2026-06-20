// MTL5 Benchmark -- native KLU vs SuiteSparse KLU scoreboard (Phase 0, #132)
//
// Establishes the performance scoreboard for the native-KLU parity effort
// (docs/plans/native-klu-performance.md). For each matrix it reports, for native
// KLU and (when built with MTL5_HAS_KLU) external SuiteSparse KLU:
//   - BTF block count and largest block (native only; structural)
//   - factor+solve time
//   - factor fill (nnz of L+U)
//   - residual ||Ax-b||_inf / ||b||_inf
//   - the native / external ratio (the number we are driving to <= 1.5x)
//
// Matrices: built-in 2D Poisson grids by default, plus any Matrix Market files
// passed on the command line (e.g. SuiteSparse circuit matrices fetched via
// benchmarks/fetch_klu_matrices.sh).
//
// Usage:
//   bench_klu                       # built-in 2D Poisson suite (32^2 .. 256^2)
//   bench_klu A.mtx B.mtx ...       # those matrices instead
//   bench_klu ext:Big.mtx           # external-only row: skip native (too slow
//                                   #   to finish, e.g. rajat30 before Phase 1)
//   bench_klu --csv out.csv [mtx]   # also write a CSV scoreboard

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/io/matrix_market.hpp>
#include <mtl/generators/poisson.hpp>
#include <mtl/sparse/ordering/dulmage_mendelsohn.hpp>
#include <mtl/sparse/factorization/native_klu.hpp>

#ifdef MTL5_HAS_KLU
#include <mtl/interface/klu.hpp>
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

struct Score {
    std::string name;
    std::size_t n = 0, nnz = 0;
    std::size_t nblocks = 0, largest_block = 0;
    double native_s = 0.0, native_resid = 0.0;
    std::size_t native_fill = 0;
    bool native_ok = false;
    bool native_skipped = false;   // external-only matrix (native too slow to run)
#ifdef MTL5_HAS_KLU
    double ext_s = 0.0, ext_resid = 0.0;
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

    // Native KLU factor + solve (skipped for external-only matrices that are
    // known not to finish in reasonable time, e.g. rajat30 pre-Phase-1).
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
            std::fprintf(stderr, "  [native KLU failed on %s: %s]\n", name.c_str(), e.what());
        }
    }

#ifdef MTL5_HAS_KLU
    try {
        mtl::vec::dense_vector<double> x(n, 0.0);
        s.ext_s = seconds([&] { mtl::interface::klu_solve(A, x, b); });
        s.ext_resid = rel_residual(A, x, b);
        s.ext_ok = true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "  [external KLU failed on %s: %s]\n", name.c_str(), e.what());
    }
#endif
    return s;
}

void print_header() {
    std::printf("\n%-22s %9s %11s %8s %11s  %10s %10s",
                "matrix", "n", "nnz", "blocks", "maxblock",
                "native(s)", "fill");
#ifdef MTL5_HAS_KLU
    std::printf("  %10s %8s", "KLU(s)", "ratio");
#endif
    std::printf("\n%s\n", std::string(110, '-').c_str());
}

void print_row(const Score& s) {
    std::printf("%-22s %9zu %11zu %8zu %11zu  ",
                s.name.c_str(), s.n, s.nnz, s.nblocks, s.largest_block);
    if (s.native_ok)            std::printf("%10.3f %10zu", s.native_s, s.native_fill);
    else if (s.native_skipped)  std::printf("%10s %10s", "skip", "-");
    else                        std::printf("%10s %10s", "FAIL", "-");
#ifdef MTL5_HAS_KLU
    if (s.ext_ok) {
        std::printf("  %10.3f", s.ext_s);
        if (s.native_ok && s.ext_s > 0.0) std::printf(" %7.1fx", s.native_s / s.ext_s);
        else                              std::printf(" %8s", "-");
    } else {
        std::printf("  %10s %8s", "FAIL", "-");
    }
#endif
    std::printf("\n");
}

void write_csv(const std::string& path, const std::vector<Score>& rows) {
    std::ofstream out(path);
    // native_status is ok|skip|fail; numeric native fields are left EMPTY when
    // not ok, so "skipped/failed" is never confused with a real value of 0.
    out << "matrix,n,nnz,nblocks,largest_block,native_status,native_s,native_fill,native_resid";
#ifdef MTL5_HAS_KLU
    out << ",ext_status,ext_s,ext_resid,ratio";
#endif
    out << "\n";
    for (const auto& s : rows) {
        const char* nstatus = s.native_ok ? "ok" : (s.native_skipped ? "skip" : "fail");
        out << s.name << ',' << s.n << ',' << s.nnz << ',' << s.nblocks << ','
            << s.largest_block << ',' << nstatus << ',';
        if (s.native_ok) out << s.native_s << ',' << s.native_fill << ',' << s.native_resid;
        else             out << ",,";   // empty native_s, native_fill, native_resid
#ifdef MTL5_HAS_KLU
        out << ',' << (s.ext_ok ? "ok" : "fail") << ',';
        if (s.ext_ok) out << s.ext_s << ',' << s.ext_resid;
        else          out << ",";       // empty ext_s, ext_resid
        out << ',';
        if (s.native_ok && s.ext_ok && s.ext_s > 0.0) out << (s.native_s / s.ext_s);
        // else: empty ratio
#endif
        out << "\n";
    }
}

} // namespace

int main(int argc, char** argv) {
    std::string csv_path;
    std::vector<std::string> mtx_files;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--csv" && i + 1 < argc) { csv_path = argv[++i]; }
        else { mtx_files.push_back(a); }
    }

#ifdef MTL5_HAS_KLU
    std::printf("Native KLU vs SuiteSparse KLU scoreboard (external KLU ENABLED)\n");
#else
    std::printf("Native KLU scoreboard (external KLU NOT built; "
                "configure -DMTL5_WITH_SUITESPARSE_KLU=ON to compare)\n");
#endif

    std::vector<Score> rows;
    print_header();

    if (mtx_files.empty()) {
        // Built-in 2D Poisson suite.
        for (std::size_t N : {32u, 64u, 128u, 256u}) {
            auto A = mtl::generators::poisson2d_dirichlet<double>(N, N);
            char nm[32]; std::snprintf(nm, sizeof nm, "poisson_%zux%zu", N, N);
            rows.push_back(score_matrix(nm, A));
            print_row(rows.back());
        }
    } else {
        for (const auto& arg : mtx_files) {
            // "ext:<file>" marks a matrix external-only (skip native, which is
            // too slow to finish, e.g. rajat30 before Phase 1).
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

    if (!csv_path.empty()) { write_csv(csv_path, rows); std::printf("\nCSV: %s\n", csv_path.c_str()); }
    return 0;
}
