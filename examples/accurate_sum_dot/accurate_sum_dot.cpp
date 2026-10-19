// MTL5 -- Accurate sum and dot product: reproducing Ogita, Rump and Oishi (#611)
//
// T. Ogita, S. M. Rump and S. Oishi, "Accurate sum and dot product", SIAM J.
// Sci. Comput. 26(6), 2005. Section 6 of the paper measures its algorithms on
// test data from its own generator, GenDot (Algorithm 6.1). This example runs
// the same experiments with MTL5's implementations and an exact reference:
//
//   Figures 6.1-6.4   relative error of a dot product against its condition
//                     number, for Dot, Dot2 and DotK, with the error bounds
//                     (5.8) and (5.11); in double (n = 100 and n = 2000) and,
//                     beyond the paper, in float
//   Table 6.2         correct decimal digits of the summation algorithms at
//                     condition numbers 1e7, 1e14, 1e21 and 1e28
//   Tables 6.4, 6.5   computing time of the summation and dot product
//                     algorithms, relative to the plain loop
//
// The algorithms:
//
//   dot, sum       the plain recursive loop: mtl::dot<T, T>, mtl::sum
//   dot_fma        the loop with each product fused into the sum:
//                  mtl::dot<math::fma_accumulator<T>, T>
//   dot2           Dot2 (Algorithm 5.3): mtl::dot<math::compensated_accumulator<T>, T>
//   dotK           DotK (Algorithm 5.10), K = 3..7: mtl::dot_k<K>
//   sum2, sum3     Sum2 and SumK (Algorithms 4.4, 4.8): mtl::sum_k<2>, mtl::sum_k<3>
//   exact          the exact sum rounded once: mtl::dot<math::exact_accumulator<T>, T>
//
// and, local to this example, the paper's comparison algorithms:
//
//   kahan_babuska  Kahan's compensated summation, the "K-B" column of Table 6.2
//   sum_xblas      Algorithm 4.6, XBLAS's BLAS_dsum_x
//   dot_xblas      Algorithm 5.6, XBLAS's BLAS_ddot_x
//
// The XBLAS algorithms are double-double arithmetic written out with
// math::two_sum, math::fast_two_sum and math::two_product, so no double-double
// type is needed.
//
// Errors. The relative error |res - x.y| / |x.y| is computed exactly: res is
// subtracted inside an exact accumulator holding x.y, and only the difference is
// rounded. As in the paper, a relative error above 2 is set to 2.
//
// Differences from the paper:
//   - TwoProduct is the FMA version (Algorithm 3.5) wherever the hardware has
//     FMA, so Dot2 and DotK cost fewer flops than the paper counts for
//     Dekker's TwoProduct, and their time ratios are lower than its "theor." rows.
//   - Times are relative to MTL5's plain loop, mtl::dot<T, T>, not to a BLAS
//     DDOT; the default mtl::dot path (SIMD, or BLAS when MTL5 is built with
//     it) is timed as a row of its own.
//   - The dot product sweeps use the paper's 1000 samples; Table 6.2 uses its
//     lengths but 4 sums per length and condition number rather than some
//     1000. The whole run takes under a minute in a Release build; --samples
//     scales every count.
//   - Table 6.2's rows are the requested condition numbers. GenDot typically
//     achieves several times the requested one (its Table 6.1), and each
//     decade costs a correct digit, so the example also prints the median
//     achieved condition number of each row.
//
// Usage (the CMake target and binary are prefixed "example_"):
//
//   example_accurate_sum_dot [--out DIR] [--samples N] [--quick]
//
//   --out DIR      where to write the CSV files (default: the current directory)
//   --samples N    the dot products per accuracy sweep (default 1000, the paper's)
//   --quick        a small run, for a smoke test
//
// Build in Release for meaningful timings.
//
// Outputs, all CSV with a header row:
//
//   dot_double_n100.csv    Figures 6.1-6.3: one row per sample, the achieved
//   dot_double_n2000.csv   condition number, each algorithm's relative error,
//   dot_float_n100.csv     and each algorithm's bound
//   sum_digits.csv         Table 6.2: one row per sum
//   timing.csv             Tables 6.4 and 6.5: one row per algorithm and length
//
// plot_accurate_sum_dot.py, next to this file, draws the figures from them;
// the results are discussed in docs/examples/accurate-sum-dot.md.

#include <mtl/mtl.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace {

using namespace mtl;

template <typename T>
using dvec = vec::dense_vector<T>;

// ---------------------------------------------------------------------------
// The paper's constants and bounds
// ---------------------------------------------------------------------------

/// eps, the relative rounding error unit: 2^-53 in double, 2^-24 in float.
template <typename T>
const double eps = std::ldexp(1.0, -std::numeric_limits<T>::digits);

/// gamma_n = n eps / (1 - n eps).
double gamma_n(double n, double e) { return n * e / (1.0 - n * e); }

// Relative-error bounds as functions of the length n, the condition number
// cond = 2 |x|.|y| / |x.y| and eps. NaN means the paper proves none.
using bound_fn = double (*)(double n, double cond, double e);

/// The recursive dot product: |res - x.y| <= gamma_n |x|.|y| (Higham, (2.4) for
/// sums), so the relative error is at most gamma_n cond / 2.
double bound_dot(double n, double cond, double e) { return gamma_n(n, e) * cond / 2; }

/// (5.8), Dot2: eps + gamma_n^2 cond / 2.
double bound_dot2(double n, double cond, double e) {
    return e + std::pow(gamma_n(n, e), 2) * cond / 2;
}

/// (5.11), DotK: eps + 2 gamma_{4n-2}^2 + gamma_{4n-2}^K cond / 2.
template <int K>
double bound_dot_k(double n, double cond, double e) {
    const double g = gamma_n(4 * n - 2, e);
    return e + 2 * g * g + std::pow(g, K) * cond / 2;
}

/// The exact sum rounded once to nearest.
double bound_exact(double, double, double e) { return e; }

double no_bound(double, double, double) { return std::numeric_limits<double>::quiet_NaN(); }

// ---------------------------------------------------------------------------
// The paper's comparison algorithms
// ---------------------------------------------------------------------------

/// Kahan's compensated summation: the running error c is subtracted from the
/// next summand. Backward stable, (1.2), so its error still grows with cond.
template <typename T>
T sum_kahan_babuska(const dvec<T>& p) {
    T s = 0, c = 0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        const T y = p(i) - c;
        const T t = s + y;
        c = (t - s) - y;
        s = t;
    }
    return s;
}

/// Algorithm 4.6, SumXBLAS: a double-double running sum [s, t].
template <typename T>
T sum_xblas(const dvec<T>& p) {
    T s = 0, t = 0;
    for (std::size_t i = 0; i < p.size(); ++i) {
        const auto t12 = math::two_sum(s, p(i));
        const auto st = math::fast_two_sum(t12.hi, t12.lo + t);
        s = st.hi;
        t = st.lo;
    }
    return s;
}

/// Algorithm 5.6, DotXBLAS: a double-double running dot product [s, t].
template <typename T>
T dot_xblas(const dvec<T>& x, const dvec<T>& y) {
    T s = 0, t = 0;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const auto hr = math::two_product(x(i), y(i));
        const auto s12 = math::two_sum(s, hr.hi);
        const auto t12 = math::two_sum(t, hr.lo);
        const auto u = math::fast_two_sum(s12.hi, s12.lo + t12.hi);
        const auto st = math::fast_two_sum(u.hi, t12.lo + u.lo);
        s = st.hi;
        t = st.lo;
    }
    return s;
}

// ---------------------------------------------------------------------------
// The algorithms under test
// ---------------------------------------------------------------------------

template <typename T>
T dot_plain(const dvec<T>& x, const dvec<T>& y) { return mtl::dot<T, T>(x, y); }
template <typename T>
T dot_default(const dvec<T>& x, const dvec<T>& y) { return mtl::dot(x, y); }
template <typename T>
T dot_fma(const dvec<T>& x, const dvec<T>& y) {
    return mtl::dot<math::fma_accumulator<T>, T>(x, y);
}
template <typename T>
T dot_compensated(const dvec<T>& x, const dvec<T>& y) {
    return mtl::dot<math::compensated_accumulator<T>, T>(x, y);
}
template <typename T, int K>
T dot_k(const dvec<T>& x, const dvec<T>& y) { return mtl::dot_k<K>(x, y); }
template <typename T>
T dot_exact(const dvec<T>& x, const dvec<T>& y) {
    return mtl::dot<math::exact_accumulator<T>, T>(x, y);
}

template <typename T>
T sum_plain(const dvec<T>& p) { return mtl::sum(p); }
template <typename T, int K>
T sum_k(const dvec<T>& p) { return mtl::sum_k<K>(p); }
template <typename T>
T sum_compensated(const dvec<T>& p) {
    return mtl::sum<math::compensated_accumulator<T>>(p);
}

template <typename T>
struct dot_algorithm {
    const char* name;
    T (*run)(const dvec<T>&, const dvec<T>&);
    bound_fn bound;
};

template <typename T>
struct sum_algorithm {
    const char* name;
    T (*run)(const dvec<T>&);
};

/// The dot products of the accuracy sweeps. Dot2 is the compensated
/// accumulator; mtl::dot_k<2> computes the same bits.
template <typename T>
std::vector<dot_algorithm<T>> accuracy_dots(int k_max) {
    std::vector<dot_algorithm<T>> a = {
        {"dot", dot_plain<T>, bound_dot},
        {"dot_fma", dot_fma<T>, bound_dot},
        {"dot2", dot_compensated<T>, bound_dot2},
        {"dot_xblas", dot_xblas<T>, no_bound},
        {"dot3", dot_k<T, 3>, bound_dot_k<3>},
        {"dot4", dot_k<T, 4>, bound_dot_k<4>},
    };
    if (k_max >= 5) a.push_back({"dot5", dot_k<T, 5>, bound_dot_k<5>});
    if (k_max >= 6) a.push_back({"dot6", dot_k<T, 6>, bound_dot_k<6>});
    if (k_max >= 7) a.push_back({"dot7", dot_k<T, 7>, bound_dot_k<7>});
    a.push_back({"exact", dot_exact<T>, bound_exact});
    return a;
}

// ---------------------------------------------------------------------------
// Exact errors
// ---------------------------------------------------------------------------

/// |res - v| / |v|, where `ref` holds v exactly; the difference is formed
/// exactly and rounded once. Clipped to 2, as in the paper.
template <typename T>
double relative_error(const math::exact_accumulator<T>& ref, double abs_ref, T res) {
    auto d = ref;
    accumulator_add_product(d, -res, T(1));
    const double err = std::fabs(accumulator_value(d, std::type_identity<double>{})) / abs_ref;
    return std::min(err, 2.0);
}

/// The paper's correct decimal digits, -log10 of the relative error, set to 0
/// when negative. An exact result is reported as 16, the paper's maximum.
double correct_digits(double rel) {
    if (rel == 0) return 16.0;
    return std::clamp(-std::log10(rel), 0.0, 16.0);
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

double median(std::vector<double> v) {
    if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
    const auto mid = v.begin() + static_cast<std::ptrdiff_t>(v.size() / 2);
    std::nth_element(v.begin(), mid, v.end());
    return *mid;
}

/// The median absolute deviation, the paper's spread measure: median |v - median v|.
double mad(const std::vector<double>& v) {
    const double m = median(v);
    std::vector<double> d;
    for (double x : v) d.push_back(std::fabs(x - m));
    return median(d);
}

std::ofstream open_csv(const std::filesystem::path& path) {
    std::ofstream f(path);
    if (!f) {
        std::cerr << "cannot write " << path.string() << '\n';
        std::exit(1);
    }
    f << std::scientific << std::setprecision(6);
    return f;
}

// ---------------------------------------------------------------------------
// Figures 6.1-6.4: relative error against condition number
// ---------------------------------------------------------------------------

/// `samples` GenDot products of length n with anticipated condition numbers
/// spread evenly in exponent over [1, 10^max_log10]. Writes one CSV row per
/// sample and prints the median relative error per band of condition numbers.
template <typename T>
void dot_sweep(const char* label, std::size_t n, int samples, double max_log10, int k_max,
               const std::filesystem::path& csv_path) {
    const auto algs = accuracy_dots<T>(k_max);
    const double e = eps<T>;
    const double nd = static_cast<double>(n);
    auto csv = open_csv(csv_path);
    csv << "sample,n,cond_requested,cond";
    for (const auto& a : algs) csv << ",err_" << a.name;
    for (const auto& a : algs) csv << ",bound_" << a.name;
    csv << '\n';

    // Bands of condition numbers: [1e0, 1e10), [1e10, 1e20), ...
    const double band = max_log10 > 60 ? 10.0 : 5.0;
    const int bands = static_cast<int>(std::ceil((max_log10 + 2) / band));
    std::vector<std::vector<std::vector<double>>> errs(
        bands, std::vector<std::vector<double>>(algs.size()));
    std::vector<std::size_t> violations(algs.size(), 0);

    std::mt19937_64 gen(611);
    int used = 0;
    for (int k = 0; k < samples; ++k) {
        const double c =
            std::pow(10.0, samples > 1 ? max_log10 * k / (samples - 1) : 0.0);
        const auto g = generators::gen_dot<T>(n, c, gen);
        if (!std::isfinite(g.cond)) continue;   // x.y == 0: no relative error
        ++used;
        math::exact_accumulator<T> ref{};
        for (std::size_t i = 0; i < n; ++i) accumulator_add_product(ref, g.x(i), g.y(i));
        const double abs_ref = std::fabs(accumulator_value(ref, std::type_identity<double>{}));
        const int b = std::clamp(static_cast<int>(std::log10(g.cond) / band), 0, bands - 1);

        csv << k << ',' << n << ',' << c << ',' << g.cond;
        std::vector<double> bounds;
        for (std::size_t a = 0; a < algs.size(); ++a) {
            const double err = relative_error(ref, abs_ref, algs[a].run(g.x, g.y));
            const double bound = algs[a].bound(nd, g.cond, e);
            // A clipped error of 2 is below any bound of 2 or more.
            if (err > bound) ++violations[a];
            errs[b][a].push_back(err);
            bounds.push_back(bound);
            csv << ',' << err;
        }
        for (double bound : bounds) csv << ',' << bound;
        csv << '\n';
    }

    std::cout << "\n" << label << ": n = " << n << ", " << used
              << " GenDot samples, median relative error by condition number\n";
    std::cout << "  " << std::left << std::setw(17) << "cond" << std::right;
    for (const auto& a : algs) std::cout << std::setw(10) << a.name;
    std::cout << "\n";
    for (int b = 0; b < bands; ++b) {
        if (errs[b][0].empty()) continue;
        std::cout << "  1e" << std::left << std::setw(3) << static_cast<int>(b * band)
                  << "- 1e" << std::setw(7) << static_cast<int>((b + 1) * band) << std::right
                  << std::scientific << std::setprecision(1);
        for (std::size_t a = 0; a < algs.size(); ++a)
            std::cout << std::setw(10) << median(errs[b][a]);
        std::cout << std::defaultfloat << "  (" << errs[b][0].size() << ")\n";
    }
    std::cout << "  bound violations:";
    for (std::size_t a = 0; a < algs.size(); ++a)
        if (!std::isnan(algs[a].bound(nd, 1.0, e)))
            std::cout << ' ' << algs[a].name << ' ' << violations[a];
    std::cout << "\n  -> " << csv_path.string() << "\n";
}

// ---------------------------------------------------------------------------
// Table 6.2: correct digits of the summation algorithms
// ---------------------------------------------------------------------------

std::vector<sum_algorithm<double>> summation_algorithms() {
    return {
        {"sum", sum_plain<double>},
        {"kahan_babuska", sum_kahan_babuska<double>},
        {"sum2", sum_k<double, 2>},
        {"sum_xblas", sum_xblas<double>},
        {"sum3", sum_k<double, 3>},
    };
}

void sum_digits(const std::vector<std::size_t>& lengths, int samples,
                const std::filesystem::path& csv_path) {
    const auto algs = summation_algorithms();
    const double conds[] = {1e7, 1e14, 1e21, 1e28};
    auto csv = open_csv(csv_path);
    csv << "cond_requested,n,sample,cond";
    for (const auto& a : algs) csv << ",digits_" << a.name;
    csv << '\n';

    std::cout << "\nTable 6.2: correct decimal digits of summation, double, n = "
              << lengths.front() << ".." << lengths.back() << " (" << lengths.size()
              << " lengths), " << samples
              << " sums per length; median (median absolute deviation)\n";
    std::cout << "  cond   achieved";
    for (const auto& a : algs) std::cout << std::setw(15) << a.name;
    std::cout << "\n" << std::fixed << std::setprecision(1);

    std::mt19937_64 gen(4);
    for (double c : conds) {
        std::vector<std::vector<double>> digits(algs.size());
        std::vector<double> achieved;
        for (std::size_t n : lengths) {
            for (int k = 0; k < samples; ++k) {
                const auto g = generators::gen_sum<double>(n, c, gen);
                if (!std::isfinite(g.cond)) continue;
                math::exact_accumulator<double> ref{};
                for (std::size_t i = 0; i < n; ++i) accumulator_add_product(ref, g.p(i), 1.0);
                const double abs_ref =
                    std::fabs(accumulator_value(ref, std::type_identity<double>{}));
                achieved.push_back(g.cond);
                csv << c << ',' << n << ',' << k << ',' << g.cond;
                for (std::size_t a = 0; a < algs.size(); ++a) {
                    const double d = correct_digits(relative_error(ref, abs_ref, algs[a].run(g.p)));
                    digits[a].push_back(d);
                    csv << ',' << d;
                }
                csv << '\n';
            }
        }
        std::cout << "  1e" << std::setw(2) << static_cast<int>(std::lround(std::log10(c)))
                  << "   " << std::scientific << std::setprecision(1) << median(achieved)
                  << std::fixed;
        for (const auto& d : digits)
            std::cout << std::setw(8) << median(d) << " (" << mad(d) << ")";
        std::cout << "\n";
    }
    std::cout << std::defaultfloat;
    std::cout << "  'achieved' is the median achieved condition number; GenDot typically\n"
                 "  overshoots the requested one, and each decade costs a digit.\n"
                 "  paper (DSum, K-B, Sum2, SumXBLAS, Sum3):\n"
                 "  1e 7   8.9 9.8 16.0 16.0 16.0;  1e14   2.0 2.9 16.0 16.0 16.0\n"
                 "  1e21   0.0 0.0 10.3 11.1 16.0;  1e28   0.0 0.0  3.8  4.3 16.0\n";
    std::cout << "  -> " << csv_path.string() << "\n";
}

// ---------------------------------------------------------------------------
// Tables 6.4 and 6.5: computing time
// ---------------------------------------------------------------------------

volatile double sink = 0;   // keeps the results of the timed calls alive

template <typename F>
double call_opaque(void* f) { return static_cast<double>((*static_cast<F*>(f))()); }

/// Nanoseconds per call of f: each trial repeats f until about 2 ms have
/// passed, and the fastest of 7 trials is kept. f is called through a volatile
/// function pointer, so the compiler cannot see that every call computes the
/// same value from the same inputs and hoist it out of the loop.
template <typename F>
double ns_per_call(F f) {
    using clock = std::chrono::steady_clock;
    double (*volatile call)(void*) = &call_opaque<F>;
    void* volatile ctx = &f;
    std::size_t reps = 1;
    for (;;) {   // calibrate
        const auto t0 = clock::now();
        for (std::size_t r = 0; r < reps; ++r) sink = sink + call(ctx);
        const auto dt = std::chrono::duration<double>(clock::now() - t0).count();
        if (dt > 2e-3 || reps > (std::size_t{1} << 30)) break;
        reps *= 2;
    }
    double best = std::numeric_limits<double>::infinity();
    for (int trial = 0; trial < 7; ++trial) {
        const auto t0 = clock::now();
        for (std::size_t r = 0; r < reps; ++r) sink = sink + call(ctx);
        const auto dt = std::chrono::duration<double, std::nano>(clock::now() - t0).count();
        best = std::min(best, dt / static_cast<double>(reps));
    }
    return best;
}

void timing(const std::vector<std::size_t>& lengths, const std::filesystem::path& csv_path) {
    auto csv = open_csv(csv_path);
    csv << "table,n,algorithm,ns_per_call,ratio\n";
    // The paper's measured ratios (environment I, then II) and its flop-count
    // ratio ("theor.", with Dekker's TwoProduct).
    struct row { const char* name; const char* paper; };

    // sum2 and sum3 are mtl::sum_k, which copies its input to a workspace;
    // sum2_acc is Sum2 streamed through the compensated accumulator.
    auto sums = summation_algorithms();
    sums.insert(sums.begin() + 3, {"sum2_acc", sum_compensated<double>});
    const row sum_paper[] = {{"sum", "1"}, {"kahan_babuska", "4.3 / 3.4 / 4"},
                             {"sum2", "7.1 / 5.1 / 7"}, {"sum2_acc", "-"},
                             {"sum_xblas", "15.5 / 8.3 / 10"}, {"sum3", "16.7 / 11.3 / 13"}};
    const std::vector<dot_algorithm<double>> dots = {
        {"dot", dot_plain<double>, no_bound},
        {"dot_default", dot_default<double>, no_bound},
        {"dot_fma", dot_fma<double>, no_bound},
        {"dot2", dot_compensated<double>, no_bound},
        {"dot2_k", dot_k<double, 2>, no_bound},
        {"dot_xblas", dot_xblas<double>, no_bound},
        {"dot3", dot_k<double, 3>, no_bound},
        {"dot4", dot_k<double, 4>, no_bound},
        {"dot5", dot_k<double, 5>, no_bound},
        {"exact", dot_exact<double>, no_bound},
    };
    const char* dot_paper[] = {"1", "-", "-", "13.5 / 22.7 / 12.5", "-", "18.5 / 38.3 / 18.5",
                               "21.3 / 38.0 / 18.5", "28.2 / 48.7 / 24.5",
                               "35.1 / 59.0 / 30.5", "-"};

    std::cout << "\nTables 6.4 and 6.5: time relative to the plain loop, double, cond 1e20"
#ifndef NDEBUG
              << "\n  (NDEBUG is not defined: this is not an optimised build, and the"
                 " ratios mean little)"
#endif
              << "\n";
    std::cout << "  " << std::left << std::setw(14) << "algorithm" << std::right;
    for (std::size_t n : lengths) std::cout << std::setw(10) << ("n=" + std::to_string(n));
    std::cout << "    paper (env I / env II / theor.)\n";

    std::mt19937_64 gen(65);
    std::vector<dvec<double>> ps;
    std::vector<generators::ill_conditioned_dot<double>> ds;
    for (std::size_t n : lengths) {
        ps.push_back(generators::gen_sum<double>(n, 1e20, gen).p);
        ds.push_back(generators::gen_dot<double>(n, 1e20, gen));
    }

    std::cout << std::fixed << std::setprecision(1);
    std::vector<double> sum_base(lengths.size()), dot_base(lengths.size());
    for (std::size_t a = 0; a < sums.size(); ++a) {
        std::cout << "  " << std::left << std::setw(14) << sums[a].name << std::right;
        for (std::size_t j = 0; j < lengths.size(); ++j) {
            const auto& p = ps[j];
            const double t = ns_per_call([&] { return sums[a].run(p); });
            if (a == 0) sum_base[j] = t;
            csv << "6.4," << lengths[j] << ',' << sums[a].name << ',' << t << ','
                << t / sum_base[j] << '\n';
            std::cout << std::setw(10) << t / sum_base[j];
        }
        std::cout << "    " << sum_paper[a].paper << "\n";
    }
    for (std::size_t a = 0; a < dots.size(); ++a) {
        std::cout << "  " << std::left << std::setw(14) << dots[a].name << std::right;
        for (std::size_t j = 0; j < lengths.size(); ++j) {
            const auto& d = ds[j];
            const double t = ns_per_call([&] { return dots[a].run(d.x, d.y); });
            if (a == 0) dot_base[j] = t;
            csv << "6.5," << lengths[j] << ',' << dots[a].name << ',' << t << ','
                << t / dot_base[j] << '\n';
            std::cout << std::setw(10) << t / dot_base[j];
        }
        std::cout << "    " << dot_paper[a] << "\n";
    }
    std::cout << std::defaultfloat;
    std::cout << "  plain loop: sum ";
    for (std::size_t j = 0; j < lengths.size(); ++j)
        std::cout << (j ? ", " : "") << std::setprecision(3) << sum_base[j] / 1e3 << " us";
    std::cout << "; dot ";
    for (std::size_t j = 0; j < lengths.size(); ++j)
        std::cout << (j ? ", " : "") << dot_base[j] / 1e3 << " us";
    std::cout << "\n  -> " << csv_path.string() << "\n";
}

void usage(const char* argv0) {
    std::cerr << "usage: " << argv0 << " [--out DIR] [--samples N] [--quick]\n";
}

} // namespace

int main(int argc, char** argv) {
    std::filesystem::path out = ".";
    int samples = 1000;
    bool quick = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--out" && i + 1 < argc) {
            out = argv[++i];
        } else if (a == "--samples" && i + 1 < argc) {
            samples = std::max(1, std::atoi(argv[++i]));
        } else if (a == "--quick") {
            quick = true;
        } else {
            usage(argv[0]);
            return a == "--help" || a == "-h" ? 0 : 1;
        }
    }
    if (quick) samples = std::min(samples, 50);
    std::filesystem::create_directories(out);

    std::cout << "Accurate sum and dot product (Ogita, Rump and Oishi, 2005), MTL5\n"
              << "eps = 2^-53 in double, 2^-24 in float; relative errors above 2 are set to 2\n";

    // Figures 6.1-6.3 (n = 100, cond up to 1e120), Figure 6.4 (n = 2000, K up
    // to 7), and the same sweep in float, as far as float's exponent range
    // lets GenDot go.
    dot_sweep<double>("Figures 6.1-6.3, double", 100, samples, 120, 7,
                      out / "dot_double_n100.csv");
    dot_sweep<double>("Figure 6.4, double", 2000, quick ? 10 : samples, 120, 7,
                      out / "dot_double_n2000.csv");
    dot_sweep<float>("Dot products in float", 100, samples, 36, 4, out / "dot_float_n100.csv");

    // Table 6.2: the paper's lengths, n = 100..5000 in steps of 100, with
    // samples / 250 sums per length and condition number (the paper ran some
    // 1000).
    std::vector<std::size_t> lengths;
    for (std::size_t n = 100; n <= 5000; n += quick ? 900 : 100) lengths.push_back(n);
    sum_digits(lengths, quick ? 2 : std::max(1, samples / 250), out / "sum_digits.csv");

    timing(quick ? std::vector<std::size_t>{100, 1000}
                 : std::vector<std::size_t>{100, 1000, 5000},
           out / "timing.csv");
    return 0;
}
