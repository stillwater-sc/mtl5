// mult<Accumulator>(A, B, C) through the packed-panel kernel (#587).
//
// The accumulator GEMM used to be the dot-product form of `mult_generic_rows`,
// reading B(k, c) down a column for every C(r, c). It now packs a panel of B's
// columns and sweeps it with one accumulator per column. The claim that makes
// that safe is BIT-IDENTITY with the old kernel, which stays in the tree as
// `detail::mult_generic`: each C(r, c) still sees clear, then add_product for
// k = 0, 1, ..., K-1 in ascending order, then one value<Result>. So every
// assertion here is `==` on the bits, never a tolerance -- a tolerance would pass
// a kernel that had reordered the sum, which is the failure being watched for.
//
// The operands are random reals, so every product and every partial sum rounds
// and a reordering would show. The order-sensitive mock accumulator (Kahan
// compensation, through the ADL hook) makes that sharper still: its result
// depends on the exact sequence of terms.
//
// Built with -ffp-contract=off (tests/unit/CMakeLists.txt): the packed kernel
// vectorizes across columns, and where the target has FMA a vectorizing
// compiler may contract acc += a*b in one kernel and not in the other.
//
// Sets MTL5_NUM_THREADS before the pool's first use, so `mult` partitions rows.
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <type_traits>

#include <mtl/mat/dense2D.hpp>
#include <mtl/math/accumulator_traits.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/detail/thread_pool.hpp>

using namespace mtl;

namespace {

const int g_set_threads = [] {
#if defined(_WIN32)
    _putenv_s("MTL5_NUM_THREADS", "4");
#else
    setenv("MTL5_NUM_THREADS", "4", /*overwrite=*/1);
#endif
    return 0;
}();

} // namespace

namespace kahan_ns {

/// An order-sensitive accumulator through the ADL hook: Kahan-compensated
/// double summation. 16 bytes, so it takes the narrower panel.
struct kahan {
    double sum;
    double c;
};

inline void accumulator_clear(kahan& a) { a.sum = 0.0; a.c = 0.0; }
inline void accumulator_assign(kahan& a, double v) { a.sum = v; a.c = 0.0; }
inline void accumulator_add_product(kahan& a, double m, double v) {
    const double y = m * v - a.c;
    const double t = a.sum + y;
    a.c = (t - a.sum) - y;
    a.sum = t;
}
template <typename R>
R accumulator_value(const kahan& a, std::type_identity<R>) { return static_cast<R>(a.sum); }

} // namespace kahan_ns

static_assert(math::adl_accumulator<kahan_ns::kahan, double>);
static_assert(detail::acc_panel_width<double> == 8);
static_assert(detail::acc_panel_width<kahan_ns::kahan> == 4);

// Which accumulators mult<Acc> sends to the packed kernel. A large accumulator
// stands in for a quire: compute-bound, so it keeps the dot-product loop.
struct big_acc { double limbs[8]; };
using dmat = mat::dense2D<double>;
static_assert(detail::acc_panel_gemm_v<double, dmat, dmat>);
static_assert(detail::acc_panel_gemm_v<kahan_ns::kahan, dmat, dmat>);
static_assert(!detail::acc_panel_gemm_v<big_acc, dmat, dmat>);

namespace {

using colmaj = mat::parameters<tag::col_major>;

template <typename M>
void fill(M& A, std::uint64_t seed) {
    std::mt19937_64 g(seed);
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    using T = typename M::value_type;
    for (std::size_t r = 0; r < A.num_rows(); ++r)
        for (std::size_t c = 0; c < A.num_cols(); ++c) {
            if constexpr (std::is_integral_v<T>)
                A(r, c) = static_cast<T>(static_cast<int>(d(g) * 100.0));
            else
                A(r, c) = static_cast<T>(d(g));
        }
}

template <typename T>
bool same_bits(const T& a, const T& b) {
    return std::memcmp(&a, &b, sizeof(T)) == 0;
}

/// mult<Acc> against the reference kernel, and the serial packed kernel against
/// it too, for one shape and one combination of operand layouts.
template <typename Acc, typename TA, typename TB, typename TC,
          typename PA = mat::parameters<>, typename PB = mat::parameters<>,
          typename PC = mat::parameters<>>
void check(std::size_t M, std::size_t N, std::size_t K) {
    mat::dense2D<TA, PA> A(M, K);
    mat::dense2D<TB, PB> B(K, N);
    fill(A, 1 + M * 131 + K);
    fill(B, 2 + N * 137 + K);
    mat::dense2D<TC, PC> Cref(M, N), C(M, N), Cser(M, N);

    detail::mult_generic<Acc>(A, B, Cref);   // the old dot-product kernel
    mult<Acc>(A, B, C);                      // dispatch: packed, threaded when large
    detail::mult_acc_panel_rows<Acc>(A, B, Cser, 0, M);   // packed, one thread

    INFO("M=" << M << " N=" << N << " K=" << K);
    for (std::size_t r = 0; r < M; ++r)
        for (std::size_t c = 0; c < N; ++c) {
            INFO("C(" << r << ", " << c << ")");
            REQUIRE(same_bits(C(r, c), Cref(r, c)));
            REQUIRE(same_bits(Cser(r, c), Cref(r, c)));
        }
}

/// Shapes: square, rectangular both ways, widths that leave a ragged last
/// panel (1, 7, 9, 13 against a panel of 8 or 4), K = 0, and empty C.
template <typename Acc, typename TA, typename TB, typename TC,
          typename PA = mat::parameters<>, typename PB = mat::parameters<>,
          typename PC = mat::parameters<>>
void check_shapes() {
    constexpr std::size_t shapes[][3] = {
        {64, 64, 64}, {33, 17, 50}, {17, 33, 5}, {5, 1, 9}, {9, 7, 1}, {3, 9, 13},
        {12, 13, 0}, {0, 8, 4}, {8, 0, 4}, {0, 0, 0},
        {256, 64, 64},   // large enough to split across the pool
    };
    for (const auto& s : shapes)
        check<Acc, TA, TB, TC, PA, PB, PC>(s[0], s[1], s[2]);
}

} // namespace

TEST_CASE("the pool is threaded, so mult partitions rows", "[operation][mult][accumulator][mt]") {
    REQUIRE(detail::thread_pool::instance().size() == 4);
}

TEST_CASE("packed accumulator GEMM is bit-identical: double", "[operation][mult][accumulator][mt]") {
    check_shapes<double, double, double, double>();
}

TEST_CASE("packed accumulator GEMM is bit-identical: fma_accumulator<double>",
          "[operation][mult][accumulator][mt]") {
    check_shapes<math::fma_accumulator<double>, double, double, double>();
}

TEST_CASE("packed accumulator GEMM is bit-identical: float operands, wider accumulators",
          "[operation][mult][accumulator][mt]") {
    check_shapes<double, float, float, float>();
    check_shapes<double, float, float, double>();
    check_shapes<long double, float, float, float>();   // 16 bytes on x86: the narrow panel
}

TEST_CASE("packed accumulator GEMM is bit-identical: an order-sensitive ADL accumulator",
          "[operation][mult][accumulator][mt]") {
    check_shapes<kahan_ns::kahan, double, double, double>();
}

TEST_CASE("packed accumulator GEMM is bit-identical: int32 on int8 operands",
          "[operation][mult][accumulator][mt]") {
    check_shapes<std::int32_t, std::int8_t, std::int8_t, std::int32_t>();
}

TEST_CASE("packed accumulator GEMM is bit-identical: column-major and mixed layouts",
          "[operation][mult][accumulator][mt]") {
    check_shapes<double, double, double, double, colmaj, colmaj, colmaj>();
    check_shapes<double, double, double, double, mat::parameters<>, colmaj, mat::parameters<>>();
    check_shapes<double, double, double, double, colmaj, mat::parameters<>, colmaj>();
    check_shapes<math::fma_accumulator<double>, double, double, double,
                 colmaj, colmaj, mat::parameters<>>();
}
