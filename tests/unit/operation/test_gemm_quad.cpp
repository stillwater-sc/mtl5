// The VNNI / SDOT GEMM: the blocked nest driving the QUAD micro-kernel.
// #451 phase 5.
//
// What is new here, against the widen-on-load integer GEMM (#468):
//
//   widen-on-load   int8 promoted into int32 lanes, one k value per multiply-add
//   quad            four k values per instruction, from quad-interleaved panels
//
// The second is what `vpdpbusd` and `SDOT` actually are, and #468's measurement
// is why it exists: on a machine WITHOUT the instruction, narrowing a GEMM's
// operand buys nothing at all (Xeon E5-2420 v2, n=512 -- gemm_i8_i32 13.3 GOP/s
// against gemm_i32's 13.5 and fp32's 18.8), because a GEMM packs each operand
// once and reads it from cache O(n) times after, so its width in memory stops
// mattering. Everything an int8 GEMM can offer over fp32 therefore has to come
// from the instruction.
//
// CORRECTNESS IS ABSOLUTE HERE, not statistical. Integer arithmetic wraps mod
// 2^32 and wrapping addition is associative, so every rearrangement this nest
// performs -- splitting k across pc blocks, splitting m across threads, summing
// four products in one instruction instead of four, accumulating in a register
// tile -- must give the BIT-IDENTICAL answer a naive triple loop gives. Not
// close: identical. A float GEMM could not make that claim, and the quad kernel
// is exactly the kind of change (a different summation grouping) whose effect a
// tolerance-based test would hide.
//
// The pairing rules are the hardware's and are re-checked at this level: VNNI
// implements unsigned x signed natively, symmetric i8 x i8 only from AVX10.2,
// and `(i8, u8)` does not exist. Unlike a dot product, a GEMM is NOT symmetric
// in its operands, so there is no argument swap that rescues that last case --
// which is why mult_quad rejects it rather than reordering.
#define MTL5_NATIVE_FAST_GEMM 1

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <mtl/detail/gemm_quad_microkernel.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/parameter.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/simd/batch.hpp>
#include <mtl/tag/orientation.hpp>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace {

using rowmaj = mtl::mat::parameters<mtl::tag::row_major>;
using colmaj = mtl::mat::parameters<mtl::tag::col_major>;

/// Exact reference, computed in 64 bits and reduced to the accumulator width.
template <typename TC, typename MA, typename MB>
TC ref_cell(const MA& A, const MB& B, std::size_t i, std::size_t j, std::size_t K) {
    std::uint64_t acc = 0;
    for (std::size_t k = 0; k < K; ++k)
        acc += static_cast<std::uint64_t>(static_cast<std::uint32_t>(static_cast<TC>(A(i, k)))) *
               static_cast<std::uint32_t>(static_cast<TC>(B(k, j)));
    return static_cast<TC>(static_cast<std::uint32_t>(acc));
}

template <typename T> T gen_a(std::size_t i, std::size_t k) {
    if constexpr (std::is_signed_v<T>) return static_cast<T>((i * 7 + k * 3) % 251 - 125);
    else                               return static_cast<T>((i * 7 + k * 3) % 251);
}
template <typename T> T gen_b(std::size_t k, std::size_t j) {
    if constexpr (std::is_signed_v<T>) return static_cast<T>((k * 5 + j * 2) % 241 - 120);
    else                               return static_cast<T>((k * 5 + j * 2) % 241);
}

// Shapes straddling the register tile (mr x nr), the k-group of four, the cache
// blocks and the thread partition, plus ragged ones that leave partial panels
// everywhere. K values 1..3 exercise a panel that is ENTIRELY k-padding beyond
// the first group, which the pairwise kernel never had to handle.
const std::size_t kShapes[][3] = {
    {1, 1, 1}, {1, 1, 2}, {1, 1, 3}, {1, 1, 4}, {1, 1, 5},
    {1, 8, 1}, {8, 1, 1}, {4, 4, 4}, {7, 5, 3}, {13, 11, 7},
    {16, 16, 16}, {17, 16, 16}, {16, 17, 16}, {16, 16, 17}, {16, 16, 18},
    {16, 16, 19}, {33, 17, 9}, {64, 64, 64}, {65, 63, 31}, {128, 96, 48},
    {96, 128, 64}, {67, 53, 41},
};

/// A `Matrix` with NO `orientation` alias -- which the concept does not require,
/// and which a sparse matrix and a transposed view both lack in this library.
/// `interface::is_row_major_v` has no default, so naming it outside a discarded
/// branch is a hard error for such a type; this pins that `mult_quad` reaches
/// its documented generic fallback instead of failing to compile.
template <typename M>
concept HasOrientation = requires { typename M::orientation; };

template <typename T>
struct no_orientation_matrix {
    using value_type = T;
    using size_type  = std::size_t;
    std::size_t r_, c_;
    std::vector<T> d_;
    no_orientation_matrix(std::size_t r, std::size_t c) : r_(r), c_(c), d_(r * c) {}
    std::size_t size() const { return d_.size(); }
    std::size_t num_rows() const { return r_; }
    std::size_t num_cols() const { return c_; }
    T&       operator()(std::size_t i, std::size_t j)       { return d_[i * c_ + j]; }
    const T& operator()(std::size_t i, std::size_t j) const { return d_[i * c_ + j]; }
};

/// Run a shape sweep for one operand pairing.
template <typename TA, typename TB>
void sweep() {
    using TC = mtl::simd::quad_accumulator_t<TA, TB>;
    for (auto& s : kShapes) {
        const std::size_t M = s[0], N = s[1], K = s[2];
        mtl::mat::dense2D<TA, rowmaj> A(M, K);
        mtl::mat::dense2D<TB, rowmaj> B(K, N);
        mtl::mat::dense2D<TC, rowmaj> C(M, N);
        for (std::size_t i = 0; i < M; ++i)
            for (std::size_t k = 0; k < K; ++k) A(i, k) = gen_a<TA>(i, k);
        for (std::size_t k = 0; k < K; ++k)
            for (std::size_t j = 0; j < N; ++j) B(k, j) = gen_b<TB>(k, j);

        mtl::mult_quad<TC>(A, B, C);

        bool all = true;
        for (std::size_t i = 0; i < M && all; ++i)
            for (std::size_t j = 0; j < N && all; ++j)
                all = (C(i, j) == ref_cell<TC>(A, B, i, j, K));
        INFO("M=" << M << " N=" << N << " K=" << K);
        CHECK(all);
    }
}

} // namespace

TEST_CASE("the quad kernel is selected by the (accumulator, operand, operand) triple",
          "[operation][gemm][quad]") {
    using mtl::detail::is_quad_gemm;
    // The three pairings the hardware op accepts, with their accumulators.
    STATIC_REQUIRE(is_quad_gemm<std::int32_t,  std::int8_t,  std::int8_t>());
    STATIC_REQUIRE(is_quad_gemm<std::int32_t,  std::uint8_t, std::int8_t>());
    STATIC_REQUIRE(is_quad_gemm<std::uint32_t, std::uint8_t, std::uint8_t>());

    // (i8, u8) is absent on purpose -- see the file header.
    STATIC_REQUIRE_FALSE(is_quad_gemm<std::int32_t, std::int8_t, std::uint8_t>());
    // Right pairing, wrong accumulator signedness.
    STATIC_REQUIRE_FALSE(is_quad_gemm<std::uint32_t, std::int8_t, std::int8_t>());
    STATIC_REQUIRE_FALSE(is_quad_gemm<std::int32_t, std::uint8_t, std::uint8_t>());
    // Not 8-bit operands at all: these stay on the widening or same-type kernel.
    STATIC_REQUIRE_FALSE(is_quad_gemm<std::int32_t, std::int16_t, std::int16_t>());
    STATIC_REQUIRE_FALSE(is_quad_gemm<std::int32_t, std::int32_t, std::int32_t>());
    STATIC_REQUIRE_FALSE(is_quad_gemm<double, float, float>());
}

TEST_CASE("i8 x i8 -> i32 quad GEMM is exact at every shape", "[operation][gemm][quad]") {
    sweep<std::int8_t, std::int8_t>();
}

TEST_CASE("u8 x i8 -> i32 quad GEMM is exact at every shape", "[operation][gemm][quad]") {
    // VNNI's NATIVE shape -- unsigned activations against signed weights -- and
    // the pairing the instruction was designed around. It is also the one the
    // widen-on-load path CANNOT take: batch::load_widen requires matching
    // signedness, so before this kernel a mixed pair fell to the generic loop.
    sweep<std::uint8_t, std::int8_t>();
}

TEST_CASE("u8 x u8 -> u32 quad GEMM is exact at every shape", "[operation][gemm][quad]") {
    sweep<std::uint8_t, std::uint8_t>();
}

TEST_CASE("the quad nest agrees with the triple loop bit for bit",
          "[operation][gemm][quad]") {
    // The claim a float GEMM cannot make. Four products summed by ONE
    // instruction is a different grouping from four separate adds, and on
    // integers that must be unobservable, because addition mod 2^32 is
    // associative. Not "within tolerance" -- identical.
    constexpr std::size_t M = 67, N = 53, K = 41;
    mtl::mat::dense2D<std::uint8_t, rowmaj> A(M, K);
    mtl::mat::dense2D<std::int8_t, rowmaj> B(K, N);
    mtl::mat::dense2D<std::int32_t, rowmaj> C(M, N), Cref(M, N);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k) A(i, k) = gen_a<std::uint8_t>(i, k);
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < N; ++j) B(k, j) = gen_b<std::int8_t>(k, j);

    mtl::mult_quad<std::int32_t>(A, B, C);
    mtl::detail::mult_generic<std::int32_t>(A, B, Cref);   // the naive triple loop

    bool all = true;
    for (std::size_t i = 0; i < M && all; ++i)
        for (std::size_t j = 0; j < N && all; ++j) all = (C(i, j) == Cref(i, j));
    CHECK(all);
}

TEST_CASE("the quad kernel agrees with the widen-on-load kernel exactly",
          "[operation][gemm][quad]") {
    // The two arms the benchmark compares. They must differ ONLY in speed: the
    // same operands, the same accumulator, two summation groupings that are
    // equal mod 2^32. If this ever fails, a throughput comparison between them
    // is measuring two different computations.
    constexpr std::size_t M = 61, N = 47, K = 39;
    mtl::mat::dense2D<std::int8_t, rowmaj> A(M, K), B(K, N);
    mtl::mat::dense2D<std::int32_t, rowmaj> Cq(M, N), Cw(M, N);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k) A(i, k) = gen_a<std::int8_t>(i, k);
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < N; ++j) B(k, j) = gen_b<std::int8_t>(k, j);

    mtl::mult_quad<std::int32_t>(A, B, Cq);   // quad-interleaved, four k per instruction
    mtl::mult<std::int32_t>(A, B, Cw);        // widen-on-load, one k per multiply-add

    bool all = true;
    for (std::size_t i = 0; i < M && all; ++i)
        for (std::size_t j = 0; j < N && all; ++j) all = (Cq(i, j) == Cw(i, j));
    CHECK(all);
}

TEST_CASE("mult is NOT silently rerouted through the quad kernel",
          "[operation][gemm][quad]") {
    // Deliberate: both kernels are correct for (i8,i8), so letting `mult` choose
    // would mean the same call measures a different thing on different machines,
    // and nothing in a timing distinguishes vpdpbusd from its decomposition. The
    // default flips later, with a within-machine measurement behind it. This
    // test is what would notice the flip happening by accident.
    //
    // THIS IS NOT HYPOTHETICAL. The kernel selection was briefly inferred from
    // the element types, and since (i8,i8) is valid input to both, `mult`'s
    // widening path was silently rerouted -- caught only because the benchmark's
    // two int8 arms started reporting the same number to three digits. They had
    // become the same kernel, and the arm meant to be the control had stopped
    // being one. The selection is now an explicit template argument defaulting
    // to `widening`; these checks pin that default.
    //
    // Checked STRUCTURALLY, because the two kernels agree bit for bit (above) --
    // a result genuinely cannot tell them apart, which is exactly what made the
    // defect survive a full green test suite.
    using Fn = void (*)(std::size_t, std::size_t, std::size_t, std::int32_t,
                        const std::int8_t*, std::ptrdiff_t, std::ptrdiff_t,
                        const std::int8_t*, std::ptrdiff_t, std::ptrdiff_t,
                        std::int32_t, std::int32_t*, std::size_t, unsigned);
    const Fn defaulted = &mtl::detail::gemm_blocked<std::int32_t, std::int8_t, std::int8_t>;
    const Fn widening  = &mtl::detail::gemm_blocked<std::int32_t, std::int8_t, std::int8_t,
                                                    mtl::detail::gemm_kernel::widening>;
    const Fn quad      = &mtl::detail::gemm_blocked<std::int32_t, std::int8_t, std::int8_t,
                                                    mtl::detail::gemm_kernel::quad>;
    CHECK(defaulted == widening);   // the default is the kernel that was already there
    CHECK(defaulted != quad);       // ... and asking for the other one is deliberate

    // The widening path also requires MATCHING SIGNEDNESS, so a (u8,i8) pair
    // reaching `mult` takes neither fast kernel: it must fall to the
    // accumulator-aware generic loop, and still be right.
    constexpr std::size_t M = 5, N = 4, K = 6;
    mtl::mat::dense2D<std::uint8_t, rowmaj> A(M, K);
    mtl::mat::dense2D<std::int8_t, rowmaj> B(K, N);
    mtl::mat::dense2D<std::int32_t, rowmaj> C(M, N);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k) A(i, k) = gen_a<std::uint8_t>(i, k);
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < N; ++j) B(k, j) = gen_b<std::int8_t>(k, j);

    mtl::mult<std::int32_t>(A, B, C);

    bool all = true;
    for (std::size_t i = 0; i < M && all; ++i)
        for (std::size_t j = 0; j < N && all; ++j)
            all = (C(i, j) == ref_cell<std::int32_t>(A, B, i, j, K));
    CHECK(all);
}

TEST_CASE("column-major C still gets the right answer", "[operation][gemm][quad]") {
    // Col-major C is computed as C^T = B^T A^T, which SWAPS the operand order --
    // and a GEMM is not symmetric in its operands, so the swapped pair must
    // itself be one the instruction has. The symmetric pairings survive the
    // swap; (u8,i8) becomes (i8,u8), which does not exist, so that combination
    // takes the generic loop. Both must be correct, which is what this checks.
    constexpr std::size_t M = 33, N = 21, K = 17;
    {
        mtl::mat::dense2D<std::int8_t, rowmaj> A(M, K), B(K, N);
        mtl::mat::dense2D<std::int32_t, colmaj> C(M, N);
        for (std::size_t i = 0; i < M; ++i)
            for (std::size_t k = 0; k < K; ++k) A(i, k) = gen_a<std::int8_t>(i, k);
        for (std::size_t k = 0; k < K; ++k)
            for (std::size_t j = 0; j < N; ++j) B(k, j) = gen_b<std::int8_t>(k, j);
        mtl::mult_quad<std::int32_t>(A, B, C);
        bool all = true;
        for (std::size_t i = 0; i < M && all; ++i)
            for (std::size_t j = 0; j < N && all; ++j)
                all = (C(i, j) == ref_cell<std::int32_t>(A, B, i, j, K));
        INFO("symmetric pairing, swapped by the col-major path");
        CHECK(all);
    }
    {
        mtl::mat::dense2D<std::uint8_t, rowmaj> A(M, K);
        mtl::mat::dense2D<std::int8_t, rowmaj> B(K, N);
        mtl::mat::dense2D<std::int32_t, colmaj> C(M, N);
        for (std::size_t i = 0; i < M; ++i)
            for (std::size_t k = 0; k < K; ++k) A(i, k) = gen_a<std::uint8_t>(i, k);
        for (std::size_t k = 0; k < K; ++k)
            for (std::size_t j = 0; j < N; ++j) B(k, j) = gen_b<std::int8_t>(k, j);
        mtl::mult_quad<std::int32_t>(A, B, C);   // -> generic loop, still correct
        bool all = true;
        for (std::size_t i = 0; i < M && all; ++i)
            for (std::size_t j = 0; j < N && all; ++j)
                all = (C(i, j) == ref_cell<std::int32_t>(A, B, i, j, K));
        INFO("asymmetric pairing: the swap has no instruction, so the generic loop runs");
        CHECK(all);
    }
}

TEST_CASE("a matrix without an orientation alias falls back, not fails to compile",
          "[operation][gemm][quad]") {
    // `mult_quad` documents a generic fallback for matrices that are not dense
    // and contiguous -- "always correct, never merely fast". That promise was
    // false for any Matrix lacking `orientation`: the layout test named
    // `is_row_major_v<MC>` at function scope, outside any discarded branch, so
    // the fallback path could not be reached because naming it was already an
    // error. The gate now tests `orientation` before using it.
    STATIC_REQUIRE(mtl::Matrix<no_orientation_matrix<std::uint8_t>>);
    STATIC_REQUIRE_FALSE(HasOrientation<no_orientation_matrix<std::uint8_t>>);
    STATIC_REQUIRE(HasOrientation<mtl::mat::dense2D<std::uint8_t, rowmaj>>);   // for contrast

    constexpr std::size_t M = 5, N = 4, K = 6;
    no_orientation_matrix<std::uint8_t> A(M, K);
    no_orientation_matrix<std::int8_t>  B(K, N);
    no_orientation_matrix<std::int32_t> C(M, N);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k) A(i, k) = gen_a<std::uint8_t>(i, k);
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < N; ++j) B(k, j) = gen_b<std::int8_t>(k, j);

    mtl::mult_quad<std::int32_t>(A, B, C);   // compiles, and is right

    bool all = true;
    for (std::size_t i = 0; i < M && all; ++i)
        for (std::size_t j = 0; j < N && all; ++j)
            all = (C(i, j) == ref_cell<std::int32_t>(A, B, i, j, K));
    CHECK(all);
}

TEST_CASE("column-major OPERANDS pack through their strides", "[operation][gemm][quad]") {
    // Generic (rs, cs) means orientation is handled by the pack, not by a
    // special case in the kernel. Same logical product, operands stored the
    // other way round.
    constexpr std::size_t M = 29, N = 23, K = 19;
    mtl::mat::dense2D<std::uint8_t, colmaj> A(M, K);
    mtl::mat::dense2D<std::int8_t, colmaj> B(K, N);
    mtl::mat::dense2D<std::int32_t, rowmaj> C(M, N);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k) A(i, k) = gen_a<std::uint8_t>(i, k);
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < N; ++j) B(k, j) = gen_b<std::int8_t>(k, j);

    mtl::mult_quad<std::int32_t>(A, B, C);

    bool all = true;
    for (std::size_t i = 0; i < M && all; ++i)
        for (std::size_t j = 0; j < N && all; ++j)
            all = (C(i, j) == ref_cell<std::int32_t>(A, B, i, j, K));
    CHECK(all);
}

TEST_CASE("full-range operands stay exact inside the accumulator headroom",
          "[operation][gemm][quad]") {
    // The headroom claim, measured rather than asserted. u8 x i8 into int32 is
    // the pairing the instruction exists for: a product needs ~15 bits, leaving
    // ~16, so of the order of 10^4-10^5 terms fit. At the extreme operands
    // (255 x 127) and K = 5000 the true sum is 161 925 000 -- comfortably under
    // 2^31, so nothing wraps and the result is the exact product-sum.
    //
    // The NEXT test takes the same operands past that boundary; the two together
    // are the claim, one on each side of it.
    constexpr std::size_t M = 5, N = 5, K = 5000;
    mtl::mat::dense2D<std::uint8_t, rowmaj> A(M, K);
    mtl::mat::dense2D<std::int8_t, rowmaj> B(K, N);
    mtl::mat::dense2D<std::int32_t, rowmaj> C(M, N);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k) A(i, k) = 255;
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < N; ++j) B(k, j) = 127;

    mtl::mult_quad<std::int32_t>(A, B, C);

    bool all = true, any_negative = false;
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            all = all && (C(i, j) == ref_cell<std::int32_t>(A, B, i, j, K));
            any_negative = any_negative || (C(i, j) < 0);
        }
    CHECK(all);
    CHECK_FALSE(any_negative);   // i.e. it really did NOT wrap, unlike the next
}

TEST_CASE("the quad GEMM wraps past the accumulator, exactly", "[operation][gemm][quad]") {
    // Past the headroom: 255 * 127 * 100000 ~ 3.2e9 > 2^31, so the int32
    // accumulator goes negative. The answer must still be the true sum mod 2^32.
    constexpr std::size_t M = 3, N = 3, K = 100000;
    mtl::mat::dense2D<std::uint8_t, rowmaj> A(M, K);
    mtl::mat::dense2D<std::int8_t, rowmaj> B(K, N);
    mtl::mat::dense2D<std::int32_t, rowmaj> C(M, N);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k) A(i, k) = 255;
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < N; ++j) B(k, j) = 127;

    mtl::mult_quad<std::int32_t>(A, B, C);

    bool all = true, any_negative = false;
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            all = all && (C(i, j) == ref_cell<std::int32_t>(A, B, i, j, K));
            any_negative = any_negative || (C(i, j) < 0);
        }
    CHECK(all);
    CHECK(any_negative);      // i.e. it really did wrap, so the check has teeth
}

TEST_CASE("threading changes nothing at all", "[operation][gemm][quad]") {
    // The nest partitions m across threads and packs B cooperatively within a
    // team. Every C macro-block receives the same instructions in the same order
    // whichever thread runs it, and the pack is pure data movement, so the
    // threaded result must be BIT-IDENTICAL -- for any grid shape.
    constexpr std::size_t M = 129, N = 97, K = 83;
    mtl::mat::dense2D<std::uint8_t, rowmaj> A(M, K);
    mtl::mat::dense2D<std::int8_t, rowmaj> B(K, N);
    mtl::mat::dense2D<std::int32_t, rowmaj> C1(M, N), Cn(M, N);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k) A(i, k) = gen_a<std::uint8_t>(i, k);
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < N; ++j) B(k, j) = gen_b<std::int8_t>(k, j);

    // Serial reference, straight through the nest.
    mtl::detail::gemm_blocked<std::int32_t, std::uint8_t, std::int8_t,
                              mtl::detail::gemm_kernel::quad>(
        M, N, K, 1, A.data(), static_cast<std::ptrdiff_t>(K), 1,
        B.data(), static_cast<std::ptrdiff_t>(N), 1, 0, C1.data(), N, 1);

    for (unsigned nt : {2u, 3u, 4u, 8u}) {
        for (std::size_t t = 0; t < M * N; ++t) Cn.data()[t] = 0;
        mtl::detail::gemm_blocked<std::int32_t, std::uint8_t, std::int8_t,
                              mtl::detail::gemm_kernel::quad>(
            M, N, K, 1, A.data(), static_cast<std::ptrdiff_t>(K), 1,
            B.data(), static_cast<std::ptrdiff_t>(N), 1, 0, Cn.data(), N, nt);
        bool all = true;
        for (std::size_t t = 0; t < M * N && all; ++t) all = (C1.data()[t] == Cn.data()[t]);
        INFO("nthreads = " << nt);
        CHECK(all);
    }
}

TEST_CASE("alpha and beta are honoured by the quad nest", "[operation][gemm][quad]") {
    // alpha cannot be folded into the packed A panel the way the pairwise nest
    // folds it -- the panel holds BYTES, so alpha * A would be truncated before
    // it ever reached the accumulator. The quad kernel scales the int32 tile on
    // the way out instead, which is exact and applies per kc block, summing to
    // alpha times the total. beta pre-scales C as usual.
    constexpr std::size_t M = 21, N = 19, K = 13;
    const std::int32_t alpha = 3, beta = 5;
    mtl::mat::dense2D<std::uint8_t, rowmaj> A(M, K);
    mtl::mat::dense2D<std::int8_t, rowmaj> B(K, N);
    std::vector<std::int32_t> C(M * N), want(M * N);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k) A(i, k) = gen_a<std::uint8_t>(i, k);
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < N; ++j) B(k, j) = gen_b<std::int8_t>(k, j);
    for (std::size_t t = 0; t < M * N; ++t) C[t] = static_cast<std::int32_t>(t % 17) - 8;

    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            const std::uint32_t prod =
                static_cast<std::uint32_t>(ref_cell<std::int32_t>(A, B, i, j, K));
            want[i * N + j] = static_cast<std::int32_t>(
                static_cast<std::uint32_t>(beta) * static_cast<std::uint32_t>(C[i * N + j]) +
                static_cast<std::uint32_t>(alpha) * prod);
        }

    mtl::detail::gemm_blocked<std::int32_t, std::uint8_t, std::int8_t,
                              mtl::detail::gemm_kernel::quad>(
        M, N, K, alpha, A.data(), static_cast<std::ptrdiff_t>(K), 1,
        B.data(), static_cast<std::ptrdiff_t>(N), 1, beta, C.data(), N, 1);

    bool all = true;
    for (std::size_t t = 0; t < M * N && all; ++t) all = (C[t] == want[t]);
    CHECK(all);
}

TEST_CASE("alpha survives a kc-split k", "[operation][gemm][quad]") {
    // The alpha test above uses a k that fits ONE kc block, so it cannot show
    // the property the per-block scaling actually rests on: alpha is applied to
    // each block's int32 tile separately, and the blocks must sum to alpha times
    // the whole product. That is distributivity mod 2^32, and it is only
    // observable once the pc loop splits k.
    const std::size_t KC = mtl::simd::runtime_blocking<std::int32_t>().kc;
    const std::size_t K = 2 * KC + 7;
    constexpr std::size_t M = 9, N = 11;
    const std::int32_t alpha = 3, beta = 5;
    mtl::mat::dense2D<std::uint8_t, rowmaj> A(M, K);
    mtl::mat::dense2D<std::int8_t, rowmaj> B(K, N);
    std::vector<std::int32_t> C(M * N), want(M * N);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k) A(i, k) = gen_a<std::uint8_t>(i, k);
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < N; ++j) B(k, j) = gen_b<std::int8_t>(k, j);
    for (std::size_t t = 0; t < M * N; ++t) C[t] = static_cast<std::int32_t>(t % 17) - 8;

    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            const std::uint32_t prod =
                static_cast<std::uint32_t>(ref_cell<std::int32_t>(A, B, i, j, K));
            want[i * N + j] = static_cast<std::int32_t>(
                static_cast<std::uint32_t>(beta) * static_cast<std::uint32_t>(C[i * N + j]) +
                static_cast<std::uint32_t>(alpha) * prod);
        }

    mtl::detail::gemm_blocked<std::int32_t, std::uint8_t, std::int8_t,
                              mtl::detail::gemm_kernel::quad>(
        M, N, K, alpha, A.data(), static_cast<std::ptrdiff_t>(K), 1,
        B.data(), static_cast<std::ptrdiff_t>(N), 1, beta, C.data(), N, 1);

    bool all = true;
    for (std::size_t t = 0; t < M * N && all; ++t) all = (C[t] == want[t]);
    INFO("K=" << K << " over kc=" << KC << ", alpha=" << alpha << " beta=" << beta);
    CHECK(all);
}

TEST_CASE("a kc-split k reaches the same answer", "[operation][gemm][quad]") {
    // k longer than one kc block forces the pc loop to split it, so the panel is
    // packed and padded to a multiple of four MORE THAN ONCE -- each block
    // independently. The padding of an interior block is what a naive "pad the
    // whole k once" implementation would get wrong.
    const std::size_t KC = mtl::simd::runtime_blocking<std::int32_t>().kc;
    const std::size_t K = 2 * KC + 7;              // several blocks, ragged last one
    constexpr std::size_t M = 9, N = 11;
    mtl::mat::dense2D<std::uint8_t, rowmaj> A(M, K);
    mtl::mat::dense2D<std::int8_t, rowmaj> B(K, N);
    mtl::mat::dense2D<std::int32_t, rowmaj> C(M, N);
    for (std::size_t i = 0; i < M; ++i)
        for (std::size_t k = 0; k < K; ++k) A(i, k) = gen_a<std::uint8_t>(i, k);
    for (std::size_t k = 0; k < K; ++k)
        for (std::size_t j = 0; j < N; ++j) B(k, j) = gen_b<std::int8_t>(k, j);

    mtl::mult_quad<std::int32_t>(A, B, C);

    bool all = true;
    for (std::size_t i = 0; i < M && all; ++i)
        for (std::size_t j = 0; j < N && all; ++j)
            all = (C(i, j) == ref_cell<std::int32_t>(A, B, i, j, K));
    INFO("K=" << K << " over kc=" << KC);
    CHECK(all);
}
