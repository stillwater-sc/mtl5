// bench_mult_ikj -- the dense i-k-j GEMM (#549) and the CRS x dense product (#550)
//
// One source, built twice: once against the headers before #552 and once
// against the headers after it. `mtl::mult` is the only arm whose code differs
// between the two builds. The hand-rolled loops are identical in both builds and
// act as controls: if they move between builds, the comparison is noise.
//
// Shapes model a filter covariance update with an 81-dimensional error state:
// a 15-dimensional IMU block followed by 11 six-dimensional clone poses, and a
// 15-row measurement Jacobian. These are the dimensions in PR #552; the matrix
// VALUES are random, since the time a product takes does not depend on them.
//
//   F  = Phi (+) I        transition: dense 15x15 IMU block, identity elsewhere
//   H                     measurement Jacobian: the IMU block plus the clones
//                         that observe the feature track (6 columns each)
//   I - K H               Joseph factor: identity plus the dense columns of K H
//
// Output: one line per (case, arm), `case,arm,nnz,us_per_product`, as the
// minimum over rounds of the mean time per product within a round.
//
// Build (no BLAS, no MTL5_NATIVE_FAST_GEMM -- the configuration where the
// generic kernel is what runs, which is the one #552 changes):
//   g++ -std=c++20 -O3 -DNDEBUG -I<mtl5>/include bench_mult_ikj.cpp
// Run single-threaded and pinned:
//   MTL5_NUM_THREADS=1 taskset -c 4 ./a.out
// docs/performance/ikj-gemm-and-crs-dense.md reports the results; the figures
// on that page come from fig_mult_ikj.cpp.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/vec/dense_vector.hpp>

using dense = mtl::mat::dense2D<double>;
using crs   = mtl::mat::compressed2D<double>;

namespace {

constexpr std::size_t kState = 81;   // 15 IMU + 11 clones x 6
constexpr std::size_t kImu   = 15;
constexpr std::size_t kMeas  = 15;   // rows of H

std::mt19937_64 rng(20261011);

dense random_dense(std::size_t m, std::size_t n) {
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    dense A(m, n);
    for (std::size_t i = 0; i < m; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = u(rng);
    return A;
}

crs to_crs(const dense& A) {
    crs S(A.num_rows(), A.num_cols());
    {
        mtl::mat::inserter<crs> ins(S);
        for (std::size_t i = 0; i < A.num_rows(); ++i)
            for (std::size_t j = 0; j < A.num_cols(); ++j)
                if (A(i, j) != 0.0)
                    ins[i][j] << A(i, j);
    }
    return S;
}

/// F = Phi (+) I: a dense IMU block, the identity on every clone.
dense transition() {
    dense F(kState, kState);
    for (std::size_t i = 0; i < kState; ++i)
        for (std::size_t j = 0; j < kState; ++j)
            F(i, j) = (i == j) ? 1.0 : 0.0;
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::size_t i = 0; i < kImu; ++i)
        for (std::size_t j = 0; j < kImu; ++j)
            F(i, j) = u(rng);
    return F;
}

/// The columns of the error state a measurement touches: the IMU block and the
/// first `clones` clone poses.
std::vector<std::size_t> observed_columns(std::size_t clones) {
    std::vector<std::size_t> cols;
    for (std::size_t j = 0; j < kImu + 6 * clones; ++j)
        cols.push_back(j);
    return cols;
}

/// H: dense in the observed columns, zero elsewhere.
dense jacobian(std::size_t clones) {
    dense H(kMeas, kState);
    for (std::size_t i = 0; i < kMeas; ++i)
        for (std::size_t j = 0; j < kState; ++j)
            H(i, j) = 0.0;
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    for (std::size_t i = 0; i < kMeas; ++i)
        for (std::size_t j : observed_columns(clones))
            H(i, j) = u(rng);
    return H;
}

/// I - K H: K H is non-zero only in H's non-zero columns.
dense joseph(const dense& K, const dense& H) {
    dense KH(kState, kState);
    mtl::mult(K, H, KH);
    dense J(kState, kState);
    for (std::size_t i = 0; i < kState; ++i)
        for (std::size_t j = 0; j < kState; ++j)
            J(i, j) = (i == j ? 1.0 : 0.0) - KH(i, j);
    return J;
}

// ---- controls: identical code in both builds ------------------------------

/// The hand-rolled row-major i-k-j loop on raw pointers, skipping zero a(r,k):
/// what the caller wrote before using mtl::mult.
void hand_ikj(const dense& A, const dense& B, dense& C) {
    const std::size_t m = A.num_rows(), n = B.num_cols(), kd = A.num_cols();
    const double* a = A.data();
    const double* b = B.data();
    double* c = C.data();
    for (std::size_t r = 0; r < m; ++r) {
        double* crow = c + r * n;
        for (std::size_t j = 0; j < n; ++j)
            crow[j] = 0.0;
        for (std::size_t k = 0; k < kd; ++k) {
            const double ark = a[r * kd + k];
            if (ark == 0.0)
                continue;
            const double* brow = b + k * n;
            for (std::size_t j = 0; j < n; ++j)
                crow[j] += ark * brow[j];
        }
    }
}

/// The hand-rolled i-j-k dot product on raw pointers: the loop order the
/// generic kernel used before #549, without MTL5's element accessors.
void hand_ijk(const dense& A, const dense& B, dense& C) {
    const std::size_t m = A.num_rows(), n = B.num_cols(), kd = A.num_cols();
    const double* a = A.data();
    const double* b = B.data();
    double* c = C.data();
    for (std::size_t r = 0; r < m; ++r)
        for (std::size_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (std::size_t k = 0; k < kd; ++k)
                acc += a[r * kd + k] * b[k * n + j];
            c[r * n + j] = acc;
        }
}

/// The only sparse route before #550: one sparse matvec per column of B.
void crs_by_columns(const crs& A, const dense& B, dense& C) {
    const std::size_t kd = B.num_rows(), m = C.num_rows();
    mtl::vec::dense_vector<double> x(kd), y(m);
    for (std::size_t j = 0; j < B.num_cols(); ++j) {
        for (std::size_t k = 0; k < kd; ++k)
            x(k) = B(k, j);
        mtl::mult(A, x, y);
        for (std::size_t i = 0; i < m; ++i)
            C(i, j) = y(i);
    }
}

// ---- timing ---------------------------------------------------------------

volatile double sink = 0.0;

/// Mean microseconds per call within a round of at least `min_ns`, minimum
/// over `rounds`.
double time_us(const std::function<void()>& f, double& checksum, int rounds = 21,
               double min_ns = 2e6) {
    using clk = std::chrono::steady_clock;
    // Calibrate reps so one round lasts at least min_ns.
    std::size_t reps = 1;
    for (;;) {
        auto t0 = clk::now();
        for (std::size_t i = 0; i < reps; ++i) f();
        const double ns = std::chrono::duration<double, std::nano>(clk::now() - t0).count();
        if (ns >= min_ns) break;
        reps *= 2;
    }
    double best = 1e300;
    for (int r = 0; r < rounds; ++r) {
        auto t0 = clk::now();
        for (std::size_t i = 0; i < reps; ++i) f();
        const double ns = std::chrono::duration<double, std::nano>(clk::now() - t0).count();
        best = std::min(best, ns / static_cast<double>(reps));
    }
    sink = sink + checksum;
    return best / 1000.0;
}

double sum(const dense& C) {
    double s = 0.0;
    for (std::size_t i = 0; i < C.num_rows(); ++i)
        for (std::size_t j = 0; j < C.num_cols(); ++j)
            s += C(i, j);
    return s;
}

bool same_bits(const dense& a, const dense& b) {
    return a.num_rows() == b.num_rows() && a.num_cols() == b.num_cols() &&
           std::memcmp(a.data(), b.data(), a.num_rows() * a.num_cols() * sizeof(double)) == 0;
}

std::size_t nnz_of(const dense& A) {
    std::size_t n = 0;
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t j = 0; j < A.num_cols(); ++j)
            n += A(i, j) != 0.0;
    return n;
}

void report(const std::string& name, const std::string& arm, std::size_t nnz, double us) {
    std::printf("%s,%s,%zu,%.3f\n", name.c_str(), arm.c_str(), nnz, us);
    std::fflush(stdout);
}

/// Dense product: mtl::mult against the two hand-rolled loops.
void dense_case(const std::string& name, const dense& A, const dense& B) {
    dense C(A.num_rows(), B.num_cols()), R(A.num_rows(), B.num_cols());
    double cs = 0.0;
    mtl::mult(A, B, R);
    report(name, "mtl_mult", nnz_of(A), time_us([&] { mtl::mult(A, B, C); }, cs));
    cs = sum(C);
    report(name, "hand_ikj", nnz_of(A), time_us([&] { hand_ikj(A, B, C); }, cs));
    report(name, "hand_ijk", nnz_of(A), time_us([&] { hand_ijk(A, B, C); }, cs));
    // mtl::mult's result must not depend on the build: print its bits' checksum.
    std::printf("#check,%s,mtl_mult_sum,%.17g\n", name.c_str(), sum(R));
}

/// Sparse left operand: mtl::mult on the CRS matrix, the column-by-column
/// matvec, mtl::mult on the same matrix stored dense, and the hand loop.
void sparse_case(const std::string& name, const dense& Ad, const dense& B) {
    const crs A = to_crs(Ad);
    const std::size_t nz = nnz_of(Ad);
    dense C(Ad.num_rows(), B.num_cols()), D(Ad.num_rows(), B.num_cols());
    double cs = 0.0;
    report(name, "mtl_mult_crs", nz, time_us([&] { mtl::mult(A, B, C); }, cs));
    crs_by_columns(A, B, D);
    std::printf("#check,%s,crs_equals_columns,%d\n", name.c_str(), same_bits(C, D) ? 1 : 0);
    report(name, "crs_by_columns", nz, time_us([&] { crs_by_columns(A, B, D); }, cs));
    report(name, "mtl_mult_dense", nz, time_us([&] { mtl::mult(Ad, B, D); }, cs));
    std::printf("#check,%s,crs_equals_dense,%d\n", name.c_str(), same_bits(C, D) ? 1 : 0);
    report(name, "hand_ikj", nz, time_us([&] { hand_ikj(Ad, B, D); }, cs));
    // The per-call cost the caller pays to build the CRS operand each time.
    report(name, "crs_build", nz, time_us([&] { const crs T = to_crs(Ad); cs += T.nnz(); }, cs));
}

} // namespace

int main() {
    rng.seed(20261011);
    const dense P = random_dense(kState, kState);
    const dense Ht = random_dense(kState, kMeas);
    const dense K = random_dense(kState, kMeas);
    const dense Hd = random_dense(kMeas, kState);
    std::printf("case,arm,nnz,us\n");

    // The four dense products of the PR's table.
    dense_case("P*P", P, P);
    dense_case("P*Ht", P, Ht);
    dense_case("K*H", K, Hd);

    // Square sweep: where the i-k-j form's advantage comes from and how it scales.
    for (std::size_t n : {8, 16, 32, 64, 128, 256}) {
        const dense A = random_dense(n, n), B = random_dense(n, n);
        dense_case("sq" + std::to_string(n), A, B);
    }

    // Sparse left operands.
    sparse_case("F*P", transition(), P);
    for (std::size_t clones : {1, 3, 6, 11}) {
        const dense H = jacobian(clones);
        sparse_case("J" + std::to_string(clones) + "*P", joseph(K, H), P);
    }
    return 0;
}
