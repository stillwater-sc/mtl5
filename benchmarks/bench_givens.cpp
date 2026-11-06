// MTL5 Benchmark -- the cost of math::givens_rotation (#697)
//
// Times N calls of givens_rotation against the plain formula
// r = sqrt(a*a + b*b), c = a/r, s = b/r on the same inputs, for float and
// double, in two magnitude bands:
//
//   ordinary  |a|, |b| in [1, 2): inside the fast-path interval, where
//             givens_rotation computes the plain formula
//   extreme   |a|, |b| near 2^600 (float: 2^80): outside it, where
//             givens_rotation scales by 2^-e and the plain formula overflows
//
// The inputs cycle through a 1024-entry table, so the loop is bound by the
// arithmetic, not by memory. Each result is summed into a checksum that is
// printed, so the calls cannot be optimized away.
//
// Usage: bench_givens [N]   (default N = 20000000)
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <mtl/math/pow2_scaling.hpp>

namespace {

template <typename T>
struct inputs {
    std::vector<T> a, b;
};

template <typename T>
inputs<T> make_inputs(int exponent) {
    std::mt19937_64 rng(697);
    std::uniform_real_distribution<double> u(1.0, 2.0);
    inputs<T> in;
    for (int i = 0; i < 1024; ++i) {
        in.a.push_back(std::ldexp(T(u(rng)), exponent));
        in.b.push_back(std::ldexp(T(u(rng)), exponent));
    }
    return in;
}

template <typename T, typename F>
double time_loop(const inputs<T>& in, long n, F f, double& checksum) {
    T acc(0);
    const auto t0 = std::chrono::steady_clock::now();
    for (long i = 0; i < n; ++i) acc += f(in.a[i & 1023], in.b[i & 1023]);
    const auto t1 = std::chrono::steady_clock::now();
    checksum += static_cast<double>(acc);
    return std::chrono::duration<double>(t1 - t0).count();
}

template <typename T>
void run(const char* type, int extreme_exponent, long n, double& checksum) {
    auto givens = [](T a, T b) {
        const auto g = mtl::math::givens_rotation(a, b);
        return g.c + g.s + g.r;
    };
    auto plain = [](T a, T b) {
        const T r = std::sqrt(a * a + b * b);
        return a / r + b / r + r;
    };
    for (auto [band, e] : {std::pair{"ordinary", 0}, std::pair{"extreme", extreme_exponent}}) {
        const auto in = make_inputs<T>(e);
        const double tg = time_loop(in, n, givens, checksum);
        const double tp = time_loop(in, n, plain, checksum);
        std::printf("%-6s %-8s givens_rotation %7.3f s   plain %7.3f s   ratio %5.2f\n",
                    type, band, tg, tp, tg / tp);
    }
}

} // namespace

int main(int argc, char** argv) {
    const long n = argc > 1 ? std::atol(argv[1]) : 20'000'000L;
    std::printf("bench_givens: %ld calls per row\n", n);
    double checksum = 0;
    run<double>("double", 600, n, checksum);
    run<float>("float", 80, n, checksum);
    std::printf("checksum %g\n", checksum);   // the extreme plain rows are inf
    return 0;
}
