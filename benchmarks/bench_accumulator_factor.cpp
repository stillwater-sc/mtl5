// bench_accumulator_factor -- the cost of the accumulator paths of lu_factor and
// cholesky_factor (#585, #594, #596)
//
// With Accumulator = void, lu_factor and cholesky_factor run the native
// right-looking factorization (or LAPACK, when MTL5 is built with it). Any
// non-void Accumulator takes the left-looking path instead, which forms every
// entry of the factor as one reduction and rounds it once. This measures what
// that costs, per accumulation configuration:
//
//   void       the native right-looking path
//   plain      Accumulator = double: the left-looking path, plain arithmetic
//   fma        math::fma_accumulator<double>
//   comp       math::compensated_accumulator<double> (Dot2, #615)
//   exact      math::exact_accumulator<double> (Kulisch, #612)
//
// The matrix is a random dense symmetric positive definite double matrix; the
// time does not depend on its values. Output: one line per (factorization, n),
// the fastest of at least 5 runs (and at least 0.2 s of them) in milliseconds,
// then each column relative to void.
// docs/architecture/accuracy-architecture.md reports the results.
//
// Build without BLAS/LAPACK, so that void is MTL5's own right-looking code:
//   g++ -std=c++20 -O3 -DNDEBUG -I<mtl5>/include -I<build>/include bench_accumulator_factor.cpp
// Run single-threaded:
//   MTL5_NUM_THREADS=1 ./a.out

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <mtl/mtl.hpp>

namespace {

using M = mtl::mat::dense2D<double>;

M random_spd(std::size_t n) {
    std::mt19937_64 gen(1);
    std::uniform_real_distribution<double> u(-1.0, 1.0);
    M B(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) B(i, j) = u(gen);
    M A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double s = 0;
            for (std::size_t k = 0; k < n; ++k) s += B(i, k) * B(j, k);
            A(i, j) = s + (i == j ? static_cast<double>(n) : 0.0);
        }
    return A;
}

volatile double sink = 0;   // keeps the factors observable, so no work is elided

/// f() factors a fresh copy and returns a checksum of the factor; it is added
/// to `sink` after the stop timestamp.
template <typename F>
double best_ms(F f) {
    double best = 1e300, total = 0;
    for (int r = 0; r < 5 || total < 200.0; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        const double check = f();
        const auto dt = std::chrono::steady_clock::now() - t0;
        sink = sink + check;
        const double ms = std::chrono::duration<double, std::milli>(dt).count();
        best = std::min(best, ms);
        total += ms;
    }
    return best;
}

[[noreturn]] void fail(const char* what) {
    std::fprintf(stderr, "%s failed on an SPD matrix\n", what);
    std::exit(1);
}

double checksum(const M& A) {
    double s = 0;
    for (std::size_t i = 0; i < A.num_rows(); ++i) s += A(i, i) + A(i, 0);
    return s;
}

template <typename Acc>
double lu_ms(const M& A0) {
    return best_ms([&] {
        M A(A0);
        std::vector<std::size_t> p;
        if (mtl::lu_factor<Acc>(A, p) != 0) fail("lu_factor");
        return checksum(A) + static_cast<double>(p.back());
    });
}

template <typename Acc>
double cholesky_ms(const M& A0) {
    return best_ms([&] {
        M A(A0);
        if (mtl::cholesky_factor<Acc>(A) != 0) fail("cholesky_factor");
        return checksum(A);
    });
}

void row(const char* op, std::size_t n, const double (&t)[5]) {
    std::printf("%-9s %5zu  %9.3f %9.3f %9.3f %9.3f %9.3f   %5.1f %5.1f %5.1f %6.1f\n", op, n,
                t[0], t[1], t[2], t[3], t[4], t[1] / t[0], t[2] / t[0], t[3] / t[0], t[4] / t[0]);
}

} // namespace

int main() {
    using namespace mtl::math;
    std::printf("%-9s %5s  %9s %9s %9s %9s %9s   %5s %5s %5s %6s\n", "factor", "n", "void",
                "plain", "fma", "comp", "exact", "plain", "fma", "comp", "exact");
    std::printf("%-9s %5s  %49s   %s\n", "", "", "milliseconds", "relative to void");
    for (std::size_t n : {100, 200, 400}) {
        const M A = random_spd(n);
        const double lu[5] = {lu_ms<void>(A), lu_ms<double>(A), lu_ms<fma_accumulator<double>>(A),
                              lu_ms<compensated_accumulator<double>>(A),
                              lu_ms<exact_accumulator<double>>(A)};
        row("lu", n, lu);
        const double ch[5] = {cholesky_ms<void>(A), cholesky_ms<double>(A),
                              cholesky_ms<fma_accumulator<double>>(A),
                              cholesky_ms<compensated_accumulator<double>>(A),
                              cholesky_ms<exact_accumulator<double>>(A)};
        row("cholesky", n, ch);
    }
    return 0;
}
