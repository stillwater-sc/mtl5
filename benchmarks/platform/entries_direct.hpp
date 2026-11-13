#pragma once
// mtl5_bench -- the direct solvers one phase at a time, and the reference
// libraries beside them (#810).
//
// WHY PHASES. The `direct` one-shot entries (entries_sparse.hpp) time ordering,
// symbolic analysis, numeric factorization and the solve together, as a solve
// on a matrix seen for the first time. A library that is faster only at
// ordering would look like a faster factorization there. Each solver here has
// three entries:
//   <solver>.analyze   ordering + symbolic factorization
//   <solver>.factor    numeric factorization, on the analysis made in make()
//   <solver>.solve     forward and back substitution, on that factor
// The earlier phases run once in make() or in the untimed setup(), so each
// entry times one phase. Every check is the normwise backward error of a full
// solve: the analyze and factor entries finish the solve in check().
//
// TWO PROBLEMS.
//   area direct     the 2D grid of the one-shot entries (medium 96^2)
//   area direct3d   a 3D grid (medium 20^3 = 8000 unknowns): 7-point Laplacian
//                   (SPD) and convection-diffusion (unsymmetric). A 2D grid
//                   leaves only small supernodes; on a 3D grid they are large,
//                   so the supernodal kernels and their BLAS calls show.
// native_klu and sparse_qr are not here: KLU has no phase API of its own, and
// sparse_qr already takes seconds on the 2D grid.
//
// THE REFERENCES. Registered only in a build configured with
// -DMTL5_BENCH_REFERENCES=ON (platform.sh's blas variant), so the other
// variants' cards do not change. There, an entry whose library the build did
// not find is listed as skipped, with the reason.
//   ref_cholmod      CHOLMOD, supernodal L L^T (forced), AMD ordering (forced:
//                    CHOLMOD's default may also try METIS), through
//                    mtl/interface/cholmod.hpp. Counterpart of sparse_cholesky
//                    and supernodal_ldlt.
//   ref_pardiso_spd  MKL PARDISO, mtype 2 (real SPD), minimum degree ordering.
//   ref_pardiso_lu   MKL PARDISO, mtype 11 (real unsymmetric), minimum degree
//                    on A + A^T, MKL's default scaling and matching.
//                    Counterpart of sparse_lu and supernodal_lu, which order
//                    the columns with COLAMD.
// Each reference also has a one-shot entry in `direct` (ref_cholmod,
// ref_pardiso_spd, ref_pardiso_lu), the counterpart of the one-shot entries.
// Format conversion: CHOLMOD's analyze includes the binding's CRS-to-CCS copy;
// PARDISO's CSR arrays are built once in make(). Both are O(nnz), small beside
// the analysis.
//
// THREADS. CHOLMOD takes the run's thread count (Common->nthreads_max, from
// CHOLMOD 5 on) and calls the build's BLAS; PARDISO takes it through
// MKL_Set_Num_Threads.
//
// WORK MODEL, the one entries_sparse.hpp uses, from the AMD-ordered Cholesky
// pattern of A, whose column counts are c_j and nnz(L) = sum c_j:
//   factor   sum c_j^2, doubled for the unsymmetric solvers (as the one-shots)
//   solve    4 nnz(L) flops and 16 nnz(L) bytes: L and L^T, or L and U at
//            about nnz(L) each, each pass reading every value once. Values
//            only: index storage differs by format (one index per entry in
//            column-compressed form, one per supernode row in supernodal
//            form), so counting it would charge the formats differently.
//            Counted with an index per entry, CHOLMOD's supernodal solve
//            read 164% of the bandwidth roofline.
//   analyze  none: timed only, reported as n/a

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include <mtl/detail/thread_pool.hpp>
#include <mtl/sparse/factorization/sparse_cholesky.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>
#include <mtl/sparse/factorization/supernodal_ldlt.hpp>
#include <mtl/sparse/factorization/supernodal_lu.hpp>
#include <mtl/sparse/ordering/amd.hpp>
#include <mtl/sparse/ordering/colamd.hpp>
#include <mtl/vec/dense_vector.hpp>

#ifdef MTL5_HAS_CHOLMOD
#include <mtl/interface/cholmod.hpp>
#endif

#include <benchmarks/platform/entries_sparse.hpp>
#ifdef MTL5_BENCH_REFERENCES
#include <benchmarks/platform/mkl_pardiso.hpp>
#endif
#include <benchmarks/platform/registry.hpp>

namespace mtl::bench::platform::direct_entries {

using sparse_entries::csr;
namespace fac = mtl::sparse::factorization;
namespace ord = mtl::sparse::ordering;
using dvec = vec::dense_vector<double>;

/// Grid side of the 3D problems. Medium is 20^3 (8000 unknowns, top separator
/// 400 columns, against 96 for the 2D grid). The up-looking sparse_cholesky
/// sets the limit: on an i7-12700K one factorization takes 7.3 s at 20^3 and
/// ran for over ten minutes at 32^3, and an entry runs at least five times
/// (#810). Large is 24^3 for the same reason.
inline std::size_t grid3_direct(size_class c) {
    return c == size_class::small ? 10 : c == size_class::medium ? 20 : 24;
}

/// 7-point stencil on a k^3 grid: diagonal 6, neighbours -1 - c (west, south,
/// down) and -1 + c (east, north, up). c = 0 is the SPD Laplacian; c = 0.4 the
/// unsymmetric convection-diffusion of convdiff2d, weakly diagonally dominant
/// and irreducible, so nonsingular.
inline csr grid3d(std::size_t k, double c) {
    const std::size_t n = k * k * k;
    csr A(n, n);
    {
        mat::inserter<csr> ins(A);
        for (std::size_t z = 0; z < k; ++z)
            for (std::size_t y = 0; y < k; ++y)
                for (std::size_t x = 0; x < k; ++x) {
                    const std::size_t r = (z * k + y) * k + x;
                    ins[r][r] << 6.0;
                    if (x > 0) ins[r][r - 1] << -1.0 - c;
                    if (x + 1 < k) ins[r][r + 1] << -1.0 + c;
                    if (y > 0) ins[r][r - k] << -1.0 - c;
                    if (y + 1 < k) ins[r][r + k] << -1.0 + c;
                    if (z > 0) ins[r][r - k * k] << -1.0 - c;
                    if (z + 1 < k) ins[r][r + k * k] << -1.0 + c;
                }
    }
    return A;
}

// -- adapters: one interface over every solver -------------------------------------
//
// reset_analysis() / reset_factor() run in the untimed setup(), so a phase's
// timing never includes freeing the previous sample's result.

/// An MTL5 solver: X_symbolic(A, ordering), X_numeric(A, sym), num.solve(x, b).
template <typename Sym, typename Num, auto Analyze, auto Factor>
struct mtl5_adapter {
    std::optional<Sym> sym;
    std::optional<Num> num;
    explicit mtl5_adapter(const csr&) {}
    static std::string unavailable(unsigned) { return {}; }
    void reset_analysis() { num.reset(); sym.reset(); }
    void reset_factor() { num.reset(); }
    void analyze(const csr& A) { sym.emplace(Analyze(A)); }
    void factor(const csr& A) { num.emplace(Factor(A, *sym)); }
    void solve(dvec& x, const dvec& b) { num->solve(x, b); }
};

inline auto chol_sym(const csr& A) { return fac::sparse_cholesky_symbolic(A, ord::amd{}); }
inline auto chol_num(const csr& A, const fac::cholesky_symbolic& s) { return fac::sparse_cholesky_numeric(A, s); }
inline auto sldlt_sym(const csr& A) { return fac::supernodal_ldlt_symbolic(A, ord::amd{}); }
inline auto sldlt_num(const csr& A, const fac::supernodal_symbolic& s) { return fac::supernodal_ldlt_numeric(A, s); }
inline auto lu_sym(const csr& A) { return fac::sparse_lu_symbolic(A, ord::colamd{}); }
inline auto lu_num(const csr& A, const fac::lu_symbolic& s) { return fac::sparse_lu_numeric(A, s); }
inline auto slu_sym(const csr& A) { return fac::supernodal_lu_symbolic_analyze(A, ord::colamd{}); }
inline auto slu_num(const csr& A, const fac::supernodal_lu_symbolic& s) { return fac::supernodal_lu_numeric(A, s); }

using cholesky_adapter = mtl5_adapter<decltype(chol_sym(std::declval<const csr&>())),
                                      decltype(chol_num(std::declval<const csr&>(), std::declval<const fac::cholesky_symbolic&>())),
                                      chol_sym, chol_num>;
using sn_ldlt_adapter = mtl5_adapter<decltype(sldlt_sym(std::declval<const csr&>())),
                                     decltype(sldlt_num(std::declval<const csr&>(), std::declval<const fac::supernodal_symbolic&>())),
                                     sldlt_sym, sldlt_num>;
using lu_adapter = mtl5_adapter<decltype(lu_sym(std::declval<const csr&>())),
                                decltype(lu_num(std::declval<const csr&>(), std::declval<const fac::lu_symbolic&>())),
                                lu_sym, lu_num>;
using sn_lu_adapter = mtl5_adapter<decltype(slu_sym(std::declval<const csr&>())),
                                   decltype(slu_num(std::declval<const csr&>(), std::declval<const fac::supernodal_lu_symbolic&>())),
                                   slu_sym, slu_num>;

#ifdef MTL5_HAS_CHOLMOD
// cholmod_common::nthreads_max arrived in CHOLMOD 5 (SuiteSparse 7); an older
// CHOLMOD (Ubuntu 22.04's is 3) has no OpenMP team of its own and threads only
// through the BLAS, which platform.sh pins. A template, so the requires-check
// is a SFINAE test and not a hard error on the missing member.
template <typename Common>
void set_cholmod_threads(Common& c, int n) {
    if constexpr (requires { c.nthreads_max; }) c.nthreads_max = n;
}

struct cholmod_adapter {
    std::optional<mtl::interface::cholmod_solver> s;
    unsigned threads;
    explicit cholmod_adapter(const csr&)
        : threads(static_cast<unsigned>(mtl::detail::thread_pool::instance().size())) {}
    static std::string unavailable(unsigned) { return {}; }
    void reset_analysis() {
        s.emplace();
        auto& c = s->common();
        c.supernodal = CHOLMOD_SUPERNODAL;
        c.nmethods = 1;
        c.method[0].ordering = CHOLMOD_AMD;
        set_cholmod_threads(c, static_cast<int>(threads));
    }
    void reset_factor() {}
    void analyze(const csr& A) { s->analyze(A); }
    void factor(const csr&) { s->factorize(); }
    void solve(dvec& x, const dvec& b) { s->solve(x, b); }
};
#else
struct cholmod_adapter {
    explicit cholmod_adapter(const csr&) {}
    static std::string unavailable(unsigned) { return "CHOLMOD not found when this build was configured"; }
    void reset_analysis() {}
    void reset_factor() {}
    void analyze(const csr&) {}
    void factor(const csr&) {}
    void solve(dvec&, const dvec&) {}
};
#endif

#ifdef MTL5_BENCH_REFERENCES
template <int Mtype>
struct pardiso_adapter {
    std::unique_ptr<pardiso_handle> h;
    explicit pardiso_adapter(const csr& A) {
        const auto& lib = mkl_pardiso::instance(pool_threads());
        if (lib.unavailable().empty()) h = std::make_unique<pardiso_handle>(lib, A, Mtype);
    }
    static unsigned pool_threads() {
        return static_cast<unsigned>(mtl::detail::thread_pool::instance().size());
    }
    static std::string unavailable(unsigned threads) { return mkl_pardiso::instance(threads).unavailable(); }
    void reset_analysis() { h->release(); }
    void reset_factor() {}
    void analyze(const csr&) { h->analyze(); }
    void factor(const csr&) { h->factor(); }
    void solve(dvec& x, const dvec& b) { h->solve(&x(0), &b(0)); }
};
#endif

// -- the entries ----------------------------------------------------------------

enum class phase { analyze, factor, solve, all };

inline const char* suffix(phase p) {
    switch (p) {
    case phase::analyze: return ".analyze";
    case phase::factor:  return ".factor";
    case phase::solve:   return ".solve";
    case phase::all:     return "";
    }
    return "";
}

/// sum c_j and sum c_j^2 over the AMD-ordered Cholesky column counts of A.
inline std::pair<double, double> cholesky_counts(const csr& A) {
    const auto sym = fac::sparse_cholesky_symbolic(A, ord::amd{});
    double s = 0, s2 = 0;
    for (std::size_t c : sym.col_counts) { s += double(c); s2 += double(c) * double(c); }
    return {s, s2};
}

template <typename Adapter, bool Spd, int Dim, phase Ph>
problem make_phase(size_class c) {
    const std::size_t k = Dim == 2 ? sparse_entries::grid_direct(c) : grid3_direct(c);
    csr A = Dim == 2 ? (Spd ? sparse_entries::poisson2d(k) : sparse_entries::convdiff2d(k))
                     : grid3d(k, Spd ? 0.0 : 0.4);
    const std::size_t n = A.num_rows();
    const auto [nnz_l, work] = cholesky_counts(A);
    const std::string shape = std::to_string(k) + (Dim == 2 ? "^2" : "^3") + " grid, nnz " +
                              std::to_string(A.nnz());
    double flops = 0, bytes = 0;
    if constexpr (Ph == phase::factor || Ph == phase::all) {
        flops = work * (Spd ? 1.0 : 2.0);
        bytes = double(A.nnz()) * (sizeof(double) + sizeof(std::size_t));
    } else if constexpr (Ph == phase::solve) {
        flops = 4.0 * nnz_l;
        bytes = 16.0 * nnz_l;
    }

    struct S {
        csr A;
        dvec b, x;
        Adapter ad;
        S(csr M, dvec bb, std::size_t m) : A(std::move(M)), b(std::move(bb)), x(m), ad(A) {}
    };
    auto s = std::make_shared<S>(std::move(A), vec_of<double>(n, 300 + Dim * 10 + Spd), n);
    if constexpr (Ph == phase::factor || Ph == phase::solve) {
        s->ad.reset_analysis();
        s->ad.analyze(s->A);
    }
    if constexpr (Ph == phase::solve) s->ad.factor(s->A);

    problem p{shape, flops, bytes,
              [s] {
                  if constexpr (Ph == phase::analyze || Ph == phase::all) s->ad.reset_analysis();
                  else if constexpr (Ph == phase::factor) s->ad.reset_factor();
              },
              [s] {
                  if constexpr (Ph == phase::analyze) s->ad.analyze(s->A);
                  else if constexpr (Ph == phase::factor) s->ad.factor(s->A);
                  else if constexpr (Ph == phase::solve) s->ad.solve(s->x, s->b);
                  else {
                      s->ad.analyze(s->A);
                      s->ad.factor(s->A);
                      s->ad.solve(s->x, s->b);
                  }
              },
              {}};
    p.check = [s] {
        if constexpr (Ph == phase::analyze) s->ad.factor(s->A);
        if constexpr (Ph == phase::analyze || Ph == phase::factor) s->ad.solve(s->x, s->b);
        return sparse_entries::backward_error(s->A, s->x, s->b);
    };
    return p;
}

/// One solver's entries on one grid: its three phases, and for a reference on
/// the 2D grid also its one-shot. Skipped, with the reason, if its library is
/// absent.
template <typename Adapter, bool Spd, int Dim>
void add_solver(registry& r, const std::string& name, bool one_shot) {
    const std::string why = Adapter::unavailable(
        static_cast<unsigned>(mtl::detail::thread_pool::instance().size()));
    const char* area = Dim == 2 ? "direct" : "direct3d";
    auto add = [&](const std::string& op, std::function<problem(size_class)> make) {
        if (why.empty()) r.add(area, op, "f64", std::move(make));
        else r.skip(area, op, "f64", why);
    };
    if (one_shot && Dim == 2) add(name, make_phase<Adapter, Spd, Dim, phase::all>);
    add(name + suffix(phase::analyze), make_phase<Adapter, Spd, Dim, phase::analyze>);
    add(name + suffix(phase::factor), make_phase<Adapter, Spd, Dim, phase::factor>);
    add(name + suffix(phase::solve), make_phase<Adapter, Spd, Dim, phase::solve>);
}

/// Every `direct` row first, then every `direct3d` row, so each area's rows
/// stay together on the card.
template <int Dim>
void register_grid(registry& r) {
    add_solver<cholesky_adapter, true, Dim>(r, "sparse_cholesky", false);
    add_solver<sn_ldlt_adapter, true, Dim>(r, "supernodal_ldlt", false);
    add_solver<lu_adapter, false, Dim>(r, "sparse_lu", false);
    add_solver<sn_lu_adapter, false, Dim>(r, "supernodal_lu", false);
#ifdef MTL5_BENCH_REFERENCES
    add_solver<cholmod_adapter, true, Dim>(r, "ref_cholmod", true);
    add_solver<pardiso_adapter<2>, true, Dim>(r, "ref_pardiso_spd", true);
    add_solver<pardiso_adapter<11>, false, Dim>(r, "ref_pardiso_lu", true);
#endif
}

inline void register_all(registry& r) {
    register_grid<2>(r);
    register_grid<3>(r);
}

} // namespace mtl::bench::platform::direct_entries
