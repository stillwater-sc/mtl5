// Run-time selection of the interleave width (#778). With MTL5_INTERLEAVE_RUNTIME
// the generic gemv, trmv and forward substitution pick their width for float
// and double on first use. Every width must give the one-row loop's bits, so
// this forces each width the dispatch can pick and compares the public
// functions with the W = 1 kernels; it also checks the table and when
// selection is off.
//
// Built with -ffp-contract=off (tests/unit/CMakeLists.txt), as
// op_test_trmv_trsv_interleaved is.
#undef MTL5_INTERLEAVE_ROWS
#undef MTL5_INTERLEAVE_ROWS_F32
#undef MTL5_INTERLEAVE_ROWS_F64
#ifndef MTL5_INTERLEAVE_RUNTIME
#define MTL5_INTERLEAVE_RUNTIME
#endif

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include <mtl/interface/interleave_rows.hpp>
#include <mtl/interface/interleave_runtime.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/lower_trisolve.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/trmv.hpp>
#include <mtl/vec/dense_vector.hpp>

using namespace mtl;

namespace {

/// A number type with its own width: the specialization turns run-time
/// selection off for it.
struct fixed_width_tag {};

} // namespace

template <>
struct mtl::interface::interleave_rows<fixed_width_tag, mtl::interface::kernel::gemv> {
    static constexpr std::size_t value = 3;
};

namespace {

template <typename T>
mat::dense2D<T> lower(std::size_t n) {
    mat::dense2D<T> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = static_cast<T>(static_cast<double>((7 * i + 3 * j) % 11) / 13.0 - 0.4);
    for (std::size_t i = 0; i < n; ++i) A(i, i) = static_cast<T>(2.0 + static_cast<double>(i % 5) / 7.0);
    return A;
}

template <typename T>
vec::dense_vector<T> vec_of(std::size_t n) {
    vec::dense_vector<T> v(n);
    for (std::size_t i = 0; i < n; ++i) v(i) = static_cast<T>(static_cast<double>((5 * i) % 9) / 9.0 - 0.3);
    return v;
}

template <typename V>
void require_same(const V& u, const V& w) {
    REQUIRE(u.size() == w.size());
    for (std::size_t i = 0; i < u.size(); ++i) {
        INFO("i = " << i);
        REQUIRE(u(i) == w(i));
    }
}

/// The public mult_generic, trmv and lower_trisolve against the W = 1 kernels,
/// at n = 29: a remainder for every width in runtime_interleave_widths.
template <typename T>
void check_public_matches_one_row() {
    constexpr std::size_t n = 29;
    const auto A = lower<T>(n);
    const auto x0 = vec_of<T>(n);
    vec::dense_vector<T> y(n), z(n);
    detail::mult_generic(A, x0, y);
    detail::mult_rows_interleaved<1>(A, x0, z, 0, n);
    require_same(y, z);
    // A BLAS build sends float and double trmv to BLAS.
    if constexpr (!detail::trmv_dispatches_blas_v<void, mat::dense2D<T>, vec::dense_vector<T>>)
    for (bool upper : {false, true}) {
        auto p = x0, q = x0;
        trmv(A, p, upper);
        if (upper) detail::trmv_upper_rows<1>(A, q, std::size_t{0}, false);
        else       detail::trmv_lower_rows<1>(A, q, n, false);
        require_same(p, q);
    }
    auto u = x0, w = x0;
    lower_trisolve(A, u);
    detail::lower_trisolve_rows<1>(A, w, w, false);
    require_same(u, w);
    auto s = x0;
    vec::dense_vector<T> t(n);
    lower_trisolve(A, t, x0);
    detail::lower_trisolve_rows<1>(A, s, x0, false);
    require_same(t, s);
}

/// Undo force_interleave_rows however a test case exits.
struct unforce {
    ~unforce() { interface::force_interleave_rows(0); }
};

constexpr bool runtime_target =
#if defined(__x86_64__) || defined(_M_X64) || defined(__aarch64__) || defined(_M_ARM64)
    true;
#else
    false;
#endif

} // namespace

TEST_CASE("run-time selection covers float and double only", "[operation][778]") {
    using namespace mtl::interface;
    STATIC_REQUIRE(interleave_rows_at_runtime_v<float, kernel::gemv> == runtime_target);
    STATIC_REQUIRE(interleave_rows_at_runtime_v<double, kernel::trmv> == runtime_target);
    STATIC_REQUIRE(interleave_rows_at_runtime_v<const double&, kernel::lower_trisolve> == runtime_target);
    STATIC_REQUIRE_FALSE(interleave_rows_at_runtime_v<long double, kernel::gemv>);
    STATIC_REQUIRE_FALSE(interleave_rows_at_runtime_v<std::int32_t, kernel::gemv>);
    // A specialization fixes the width and keeps it fixed.
    STATIC_REQUIRE_FALSE(interleave_rows_at_runtime_v<fixed_width_tag, kernel::gemv>);
    STATIC_REQUIRE(interleave_rows_v<fixed_width_tag, kernel::gemv> == 3);
}

TEST_CASE("every width run-time selection can pick gives the one-row loop's bits", "[operation][778]") {
    unforce guard;
    for (std::size_t w : interface::runtime_interleave_widths) {
        INFO("W = " << w);
        interface::force_interleave_rows(w);
        if constexpr (runtime_target) {
            REQUIRE(interface::selected_interleave_rows<float, interface::kernel::gemv>() == w);
            REQUIRE(interface::selected_interleave_rows<double, interface::kernel::lower_trisolve>() == w);
        }
        // long double never selects at run time: forcing doesn't reach it.
        REQUIRE(interface::selected_interleave_rows<long double, interface::kernel::trmv>()
                == interface::interleave_rows_v<long double, interface::kernel::trmv>);
        check_public_matches_one_row<float>();
        check_public_matches_one_row<double>();
        check_public_matches_one_row<long double>();
    }
    interface::force_interleave_rows(0);
    check_public_matches_one_row<float>();
    check_public_matches_one_row<double>();
}

TEST_CASE("force_interleave_rows takes only the compiled widths", "[operation][778]") {
    unforce guard;
    REQUIRE_THROWS_AS(interface::force_interleave_rows(3), std::invalid_argument);
    REQUIRE_THROWS_AS(interface::force_interleave_rows(16), std::invalid_argument);
    REQUIRE_NOTHROW(interface::force_interleave_rows(12));
    REQUIRE_NOTHROW(interface::force_interleave_rows(0));
}

TEST_CASE("the unforced choice is the table's or the compile-time default", "[operation][778]") {
    using namespace mtl::interface;
    force_interleave_rows(0);
    const interleave_choice c = interleave_table(runtime_interleave_cpu());
    INFO("cpu = " << to_string(runtime_interleave_cpu()));
    const auto expect = [](std::size_t table, std::size_t dflt) { return table ? table : dflt; };
    if constexpr (runtime_target) {
        REQUIRE(selected_interleave_rows<double, kernel::gemv>()
                == expect(c.width[0][0], interleave_rows_v<double, kernel::gemv>));
        REQUIRE(selected_interleave_rows<double, kernel::trmv>()
                == expect(c.width[0][1], interleave_rows_v<double, kernel::trmv>));
        REQUIRE(selected_interleave_rows<float, kernel::lower_trisolve>()
                == expect(c.width[1][2], interleave_rows_v<float, kernel::lower_trisolve>));
    }
    // Every entry is a width the dispatch compiles.
    for (auto cpu : {interleave_cpu::unknown, interleave_cpu::alder_lake, interleave_cpu::ivy_bridge,
                     interleave_cpu::zen4, interleave_cpu::cortex_a78ae, interleave_cpu::cortex_a78c}) {
        const interleave_choice e = interleave_table(cpu);
        for (const auto& type : e.width)
            for (std::uint8_t w : type) {
                INFO(to_string(cpu) << ": " << int(w));
                REQUIRE((w == 0 || is_runtime_interleave_width(w)));
            }
    }
    STATIC_REQUIRE(interleave_table(interleave_cpu::unknown).width[0][0] == 0);
}
