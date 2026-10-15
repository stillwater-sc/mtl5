// Accumulator policy for the SOLVE phase of the sparse direct solvers (#535).
//
// The numeric factorizations take a leading `Accumulator` (#530/#531/#534/#542);
// this covers the solves that follow them: the five level-scheduled triangular
// kernels every solve() delegates to, qr_numeric::solve's inline reductions, the
// solve<Acc>() members and the one-shot *_solve<Acc> wrappers.
//
//   * The default `void` and an explicit `Acc == Value` are bit-identical to each
//     other and to the serial dense_* reference kernels.
//   * A wider `Acc` really accumulates in `Acc`: a float row reduction that
//     cancels catastrophically in float (1 - 1e8 + 1e8 == 0) recovers the exact
//     answer (1) with a double accumulator. 1e8 is exact in float but its float
//     spacing is 8, so the raw path rounds 1 - 1e8 to -1e8.
//   * On a real system a wider / FMA accumulator agrees with the default.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/sparse/util/csc.hpp>
#include <mtl/sparse/factorization/level_schedule.hpp>
#include <mtl/sparse/factorization/triangular_solve.hpp>
#include <mtl/sparse/factorization/sparse_cholesky.hpp>
#include <mtl/sparse/factorization/sparse_ldlt.hpp>
#include <mtl/sparse/factorization/sparse_lu.hpp>
#include <mtl/sparse/factorization/sparse_qr.hpp>
#include <mtl/sparse/factorization/supernodal_lu.hpp>
#include <mtl/sparse/factorization/supernodal_ldlt.hpp>
#include <mtl/sparse/ordering/amd.hpp>

using namespace mtl;
using namespace mtl::sparse::factorization;
using mtl::sparse::util::csc_matrix;

namespace {

// Build a CSC matrix from (row, col, value) triplets listed column by column in
// the storage order the kernel expects (lower: diagonal first; upper: last).
csc_matrix<float> csc3(std::vector<std::vector<std::pair<std::size_t, float>>> cols) {
    csc_matrix<float> M;
    M.nrows = M.ncols = cols.size();
    M.col_ptr.assign(1, 0);
    for (const auto& c : cols) {
        for (const auto& [r, v] : c) { M.row_ind.push_back(r); M.values.push_back(v); }
        M.col_ptr.push_back(M.row_ind.size());
    }
    return M;
}

// x2 = 1 - 1e8*x0 - (-1e8)*x1 with x0 = x1 = 1: cancels to 0 in float.
csc_matrix<float> cancelling_lower() {
    return csc3({{{0, 1.f}, {2, 1e8f}}, {{1, 1.f}, {2, -1e8f}}, {{2, 1.f}}});
}
// x0 = 1 - 1e8*x1 - (-1e8)*x2 with x1 = x2 = 1 (L^T gathers column 0 downward).
csc_matrix<float> cancelling_lower_for_transpose() {
    return csc3({{{0, 1.f}, {1, 1e8f}, {2, -1e8f}}, {{1, 1.f}}, {{2, 1.f}}});
}
// x0 = 1 - 1e8*x2 - (-1e8)*x1: the upper gather runs k decreasing.
csc_matrix<float> cancelling_upper() {
    return csc3({{{0, 1.f}}, {{0, -1e8f}, {1, 1.f}}, {{0, 1e8f}, {2, 1.f}}});
}

// Random lower-triangular L (diagonal first), and its transpose as an upper U
// (diagonal last) -- the layouts the level-scheduled kernels expect.
csc_matrix<float> random_lower(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> val(-0.5f, 0.5f);
    std::bernoulli_distribution keep(0.15);
    csc_matrix<float> L;
    L.nrows = L.ncols = n;
    L.col_ptr.assign(1, 0);
    for (std::size_t j = 0; j < n; ++j) {
        L.row_ind.push_back(j); L.values.push_back(2.0f + val(rng));
        for (std::size_t i = j + 1; i < n; ++i)
            if (keep(rng)) { L.row_ind.push_back(i); L.values.push_back(val(rng)); }
        L.col_ptr.push_back(L.row_ind.size());
    }
    return L;
}

csc_matrix<float> transpose(const csc_matrix<float>& L) {
    const std::size_t n = L.ncols;
    std::vector<std::vector<std::pair<std::size_t, float>>> cols(n);
    for (std::size_t j = 0; j < n; ++j)              // increasing j -> increasing row in U
        for (std::size_t p = L.col_ptr[j]; p < L.col_ptr[j + 1]; ++p)
            cols[L.row_ind[p]].push_back({j, L.values[p]});
    return csc3(std::move(cols));
}

std::vector<float> ones(std::size_t n) { return std::vector<float>(n, 1.0f); }

// 5-point 2-D Laplacian on a g x g grid: SPD, real fill and supernodes.
mat::compressed2D<float> laplacian2d(std::size_t g) {
    const std::size_t n = g * g;
    mat::compressed2D<float> A(n, n);
    mat::inserter<mat::compressed2D<float>> ins(A);
    auto id = [g](std::size_t r, std::size_t c) { return r * g + c; };
    for (std::size_t r = 0; r < g; ++r)
        for (std::size_t c = 0; c < g; ++c) {
            const std::size_t i = id(r, c);
            ins[i][i] << 4.0f;
            if (r + 1 < g) { ins[i][id(r + 1, c)] << -1.0f; ins[id(r + 1, c)][i] << -1.0f; }
            if (c + 1 < g) { ins[i][id(r, c + 1)] << -1.0f; ins[id(r, c + 1)][i] << -1.0f; }
        }
    return A;
}

vec::dense_vector<float> rhs(std::size_t n) {
    vec::dense_vector<float> b(n);
    for (std::size_t i = 0; i < n; ++i)
        b(i) = static_cast<float>(0.5 + std::sin(0.9 * static_cast<double>(i)));
    return b;
}

// ||A x - b||_inf / ||b||_inf, computed in double.
double rel_residual(const mat::compressed2D<float>& A,
                    const vec::dense_vector<float>& x,
                    const vec::dense_vector<float>& b) {
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();
    const auto& dat = A.ref_data();
    double r = 0.0, bn = 0.0;
    for (std::size_t i = 0; i < A.num_rows(); ++i) {
        double ax = 0.0;
        for (std::size_t k = rp[i]; k < rp[i + 1]; ++k)
            ax += static_cast<double>(dat[k]) * static_cast<double>(x(ci[k]));
        r  = std::max(r, std::abs(ax - static_cast<double>(b(i))));
        bn = std::max(bn, std::abs(static_cast<double>(b(i))));
    }
    return r / bn;
}

// The solve<Acc>() contract on any factorization object.
template <typename Fac>
void require_solve_policy(const Fac& fac, const mat::compressed2D<float>& A) {
    const std::size_t n = A.num_cols();
    const auto b = rhs(n);
    vec::dense_vector<float> x_def(n, 0.f), x_val(n, 0.f), x_wide(n, 0.f), x_fma(n, 0.f);

    fac.solve(x_def, b);                                          // void
    fac.template solve<float>(x_val, b);                          // Acc == Value
    fac.template solve<double>(x_wide, b);                        // wider
    fac.template solve<math::fma_accumulator<double>>(x_fma, b);  // fused

    for (std::size_t i = 0; i < n; ++i) REQUIRE(x_val(i) == x_def(i));
    REQUIRE(rel_residual(A, x_def, b) < 1e-5);
    REQUIRE(rel_residual(A, x_wide, b) < 1e-5);
    REQUIRE(rel_residual(A, x_fma, b) < 1e-5);
    for (std::size_t i = 0; i < n; ++i) {
        const double scale = 1.0 + std::abs(static_cast<double>(x_def(i)));
        REQUIRE(std::abs(static_cast<double>(x_wide(i) - x_def(i))) < 1e-4 * scale);
        REQUIRE(std::abs(static_cast<double>(x_fma(i) - x_def(i))) < 1e-4 * scale);
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Level-scheduled kernels
// ---------------------------------------------------------------------------

TEST_CASE("level-scheduled solves: void and Acc == Value are bit-identical to the dense reference",
          "[sparse][accumulator][level_schedule]") {
    const auto L = random_lower(300, 535);
    const auto U = transpose(L);
    std::vector<float> b(300);
    for (std::size_t i = 0; i < b.size(); ++i) b[i] = static_cast<float>(std::cos(0.3 * double(i)));

    auto check = [&](auto&& ref, auto&& run) {
        std::vector<float> r = b, d = b, v = b;
        ref(r);
        run.template operator()<void>(d);
        run.template operator()<float>(v);
        REQUIRE(d == r);
        REQUIRE(v == r);
    };

    const auto fs = build_lower_solve_schedule(L);
    check([&](auto& x) { dense_lower_solve(L, x); },
          [&]<typename A>(auto& x) { level_scheduled_lower_solve<A>(L, fs, x); });

    const auto ts = build_lower_transpose_solve_schedule(L);
    check([&](auto& x) { dense_lower_transpose_solve(L, x); },
          [&]<typename A>(auto& x) { level_scheduled_lower_transpose_solve<A>(L, ts, x); });

    const auto us = build_upper_solve_schedule(U);
    check([&](auto& x) { dense_upper_solve(U, x); },
          [&]<typename A>(auto& x) { level_scheduled_upper_solve<A>(U, us, x); });

    const auto ufs = build_unit_lower_solve_schedule(L);
    check([&](auto& x) { dense_unit_lower_solve(L, x); },
          [&]<typename A>(auto& x) { level_scheduled_unit_lower_solve<A>(L, ufs, x); });

    const auto uts = build_unit_lower_transpose_solve_schedule(L);
    check([&](auto& x) { dense_unit_lower_transpose_solve(L, x); },
          [&]<typename A>(auto& x) { level_scheduled_unit_lower_transpose_solve<A>(L, uts, x); });
}

TEST_CASE("level-scheduled solves: a wider accumulator carries the row reduction",
          "[sparse][accumulator][level_schedule]") {
    SECTION("lower") {
        const auto L = cancelling_lower();
        const auto s = build_lower_solve_schedule(L);
        auto raw = ones(3), wide = ones(3), fma = ones(3);
        level_scheduled_lower_solve(L, s, raw);
        level_scheduled_lower_solve<double>(L, s, wide);
        level_scheduled_lower_solve<math::fma_accumulator<double>>(L, s, fma);
        REQUIRE(raw[2] == 0.0f);
        REQUIRE(wide[2] == 1.0f);
        REQUIRE(fma[2] == 1.0f);
    }
    SECTION("unit lower") {
        const auto L = cancelling_lower();   // stored unit diagonal is skipped
        const auto s = build_unit_lower_solve_schedule(L);
        auto raw = ones(3), wide = ones(3);
        level_scheduled_unit_lower_solve(L, s, raw);
        level_scheduled_unit_lower_solve<double>(L, s, wide);
        REQUIRE(raw[2] == 0.0f);
        REQUIRE(wide[2] == 1.0f);
    }
    SECTION("lower transpose") {
        const auto L = cancelling_lower_for_transpose();
        const auto s = build_lower_transpose_solve_schedule(L);
        auto raw = ones(3), wide = ones(3);
        level_scheduled_lower_transpose_solve(L, s, raw);
        level_scheduled_lower_transpose_solve<double>(L, s, wide);
        REQUIRE(raw[0] == 0.0f);
        REQUIRE(wide[0] == 1.0f);
    }
    SECTION("unit lower transpose") {
        const auto L = cancelling_lower_for_transpose();
        const auto s = build_unit_lower_transpose_solve_schedule(L);
        auto raw = ones(3), wide = ones(3);
        level_scheduled_unit_lower_transpose_solve(L, s, raw);
        level_scheduled_unit_lower_transpose_solve<double>(L, s, wide);
        REQUIRE(raw[0] == 0.0f);
        REQUIRE(wide[0] == 1.0f);
    }
    SECTION("upper") {
        const auto U = cancelling_upper();
        const auto s = build_upper_solve_schedule(U);
        auto raw = ones(3), wide = ones(3);
        level_scheduled_upper_solve(U, s, raw);
        level_scheduled_upper_solve<double>(U, s, wide);
        REQUIRE(raw[0] == 0.0f);
        REQUIRE(wide[0] == 1.0f);
    }
}

// ---------------------------------------------------------------------------
// qr_numeric::solve -- its reductions are inline, not in level_schedule.hpp
// ---------------------------------------------------------------------------

namespace {
// A hand-built qr_numeric (3x3, identity column order): R and the Householder
// data are public, so each solve-time reduction can be driven in isolation.
qr_numeric<float> qr3(csc_matrix<float> R) {
    qr_numeric<float> q;
    q.R = std::move(R);
    q.V.assign(3, {});
    q.V_idx.assign(3, {});
    q.beta.assign(3, 0.0f);            // beta == 0: H_k skipped
    q.symbolic.nrows = q.symbolic.ncols = 3;
    q.symbolic.col_perm = {0, 1, 2};
    q.symbolic.col_pinv = {0, 1, 2};
    return q;
}
} // namespace

TEST_CASE("qr_numeric::solve: R back-substitution accumulates in Acc",
          "[sparse][accumulator][qr]") {
    const auto q = qr3(cancelling_upper());
    vec::dense_vector<float> b(3, 1.0f), raw(3, 0.f), val(3, 0.f), wide(3, 0.f);
    q.solve(raw, b);
    q.solve<float>(val, b);
    q.solve<double>(wide, b);
    for (std::size_t i = 0; i < 3; ++i) REQUIRE(val(i) == raw(i));
    REQUIRE(raw(0) == 0.0f);
    REQUIRE(wide(0) == 1.0f);
    REQUIRE(wide(1) == 1.0f);
    REQUIRE(wide(2) == 1.0f);
}

TEST_CASE("qr_numeric::solve: the v^T w dot of Q^T b accumulates in Acc",
          "[sparse][accumulator][qr]") {
    // R = I; one reflector with v = (1, 1e8, -1e8), beta = 1, applied to w = 1s.
    // v^T w = 1 + 1e8 - 1e8: 0 in float, 1 in double. With dot = 0 the reflector
    // is a no-op (x0 = 1); with dot = 1 it gives x0 = 1 - 1*1*1 = 0.
    auto q = qr3(csc3({{{0, 1.f}}, {{1, 1.f}}, {{2, 1.f}}}));
    q.V[0] = {1.0f, 1e8f, -1e8f};
    q.V_idx[0] = {0, 1, 2};
    q.beta[0] = 1.0f;
    vec::dense_vector<float> b(3, 1.0f), raw(3, 0.f), wide(3, 0.f);
    q.solve(raw, b);
    q.solve<double>(wide, b);
    REQUIRE(raw(0) == 1.0f);
    REQUIRE(wide(0) == 0.0f);
}

// ---------------------------------------------------------------------------
// solve<Acc>() on every sparse direct factorization
// ---------------------------------------------------------------------------

TEST_CASE("solve<Acc>: sparse_cholesky", "[sparse][accumulator][cholesky]") {
    const auto A = laplacian2d(8);
    require_solve_policy(sparse_cholesky_numeric(A, sparse_cholesky_symbolic(A)), A);
}

TEST_CASE("solve<Acc>: sparse_ldlt", "[sparse][accumulator][ldlt]") {
    const auto A = laplacian2d(8);
    require_solve_policy(sparse_ldlt_numeric(A, sparse_ldlt_symbolic(A)), A);
}

TEST_CASE("solve<Acc>: sparse_lu", "[sparse][accumulator][lu]") {
    const auto A = laplacian2d(8);
    require_solve_policy(sparse_lu_numeric(A, sparse_lu_symbolic(A)), A);
}

TEST_CASE("solve<Acc>: sparse_qr", "[sparse][accumulator][qr]") {
    const auto A = laplacian2d(8);
    require_solve_policy(sparse_qr_numeric(A, sparse_qr_symbolic(A)), A);
}

TEST_CASE("solve<Acc>: supernodal_lu", "[sparse][accumulator][supernodal]") {
    const auto A = laplacian2d(8);
    require_solve_policy(supernodal_lu_numeric(A, supernodal_lu_symbolic_analyze(A, sparse::ordering::amd{})), A);
}

TEST_CASE("solve<Acc>: supernodal_ldlt", "[sparse][accumulator][supernodal]") {
    const auto A = laplacian2d(8);
    require_solve_policy(supernodal_ldlt_numeric(A, supernodal_ldlt_symbolic(A)), A);
}

// ---------------------------------------------------------------------------
// One-shot wrappers forward Acc to the factorization AND the solve
// ---------------------------------------------------------------------------

TEST_CASE("one-shot *_solve<Acc>: default unchanged, wider accumulator solves",
          "[sparse][accumulator]") {
    const auto A = laplacian2d(6);
    const std::size_t n = A.num_cols();
    const auto b = rhs(n);

    auto run = [&](auto&& solve_default, auto&& solve_value, auto&& solve_wide) {
        vec::dense_vector<float> d(n, 0.f), v(n, 0.f), w(n, 0.f);
        solve_default(d);
        solve_value(v);
        solve_wide(w);
        for (std::size_t i = 0; i < n; ++i) REQUIRE(v(i) == d(i));
        REQUIRE(rel_residual(A, d, b) < 1e-5);
        REQUIRE(rel_residual(A, w, b) < 1e-5);
    };

    run([&](auto& x) { sparse_cholesky_solve(A, x, b); },
        [&](auto& x) { sparse_cholesky_solve<float>(A, x, b); },
        [&](auto& x) { sparse_cholesky_solve<double>(A, x, b); });
    run([&](auto& x) { sparse_ldlt_solve(A, x, b); },
        [&](auto& x) { sparse_ldlt_solve<float>(A, x, b); },
        [&](auto& x) { sparse_ldlt_solve<double>(A, x, b); });
    run([&](auto& x) { sparse_lu_solve(A, x, b); },
        [&](auto& x) { sparse_lu_solve<float>(A, x, b); },
        [&](auto& x) { sparse_lu_solve<double>(A, x, b); });
    run([&](auto& x) { sparse_qr_solve(A, x, b); },
        [&](auto& x) { sparse_qr_solve<float>(A, x, b); },
        [&](auto& x) { sparse_qr_solve<double>(A, x, b); });
    run([&](auto& x) { supernodal_ldlt_solve(A, x, b); },
        [&](auto& x) { supernodal_ldlt_solve<float>(A, x, b); },
        [&](auto& x) { supernodal_ldlt_solve<double>(A, x, b); });
    run([&](auto& x) { supernodal_lu_solve(A, x, b, sparse::ordering::amd{}); },
        [&](auto& x) { supernodal_lu_solve<float>(A, x, b, sparse::ordering::amd{}); },
        [&](auto& x) { supernodal_lu_solve<double>(A, x, b, sparse::ordering::amd{}); });
}
