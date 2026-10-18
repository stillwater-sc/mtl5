// MTL5 -- Accumulator policy for the multigrid transfers and cycle (#585).
//
//   reach     -- restrict / prolongate<Accumulator> see one product per stored
//                transfer entry; the cycle's residual one per stored A entry;
//                the void default sees none;
//   identity  -- a plain element-type accumulator gives the void transfers'
//                bits (built with -ffp-contract=off for that reason);
//   use       -- an accumulated V-cycle converges as the void one does.
//
// The void transfers now start their sums from math::zero<Value>() rather than
// Value{}. That is byte-identical for every built-in type, which is all this
// file can instantiate (MTL5 is Universal-free); it matters for an LNS, whose
// value-initialized all-zero encoding is 1.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <type_traits>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/itl/smoother/gauss_seidel.hpp>
#include <mtl/itl/mg/restriction.hpp>
#include <mtl/itl/mg/prolongation.hpp>
#include <mtl/itl/mg/multigrid.hpp>

using namespace mtl;

namespace counting {

struct count_acc { double s; };
inline std::size_t products = 0;

inline void accumulator_clear(count_acc& a) { a.s = 0.0; }
template <typename V>
void accumulator_assign(count_acc& a, const V& v) { a.s = static_cast<double>(v); }
template <typename V>
void accumulator_add_product(count_acc& a, const V& m, const V& v) {
    ++products;
    a.s += static_cast<double>(m) * static_cast<double>(v);
}
template <typename R>
R accumulator_value(const count_acc& a, std::type_identity<R>) { return static_cast<R>(a.s); }

} // namespace counting

namespace {

mat::compressed2D<double> poisson_1d(std::size_t n) {
    const double h2_inv = static_cast<double>((n + 1) * (n + 1));
    mat::compressed2D<double> A(n, n);
    mat::inserter<mat::compressed2D<double>> ins(A);
    for (std::size_t i = 0; i < n; ++i) {
        ins[i][i] << 2.0 * h2_inv;
        if (i > 0)     ins[i][i - 1] << -1.0 * h2_inv;
        if (i + 1 < n) ins[i][i + 1] << -1.0 * h2_inv;
    }
    return A;
}

vec::dense_vector<double> wave(std::size_t n) {
    vec::dense_vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i) v(i) = std::sin(0.9 * double(i) + 0.3) + 1.7;
    return v;
}

bool same_bits(const vec::dense_vector<double>& x, const vec::dense_vector<double>& y) {
    if (x.size() != y.size()) return false;
    for (std::size_t i = 0; i < x.size(); ++i)
        if (std::memcmp(&x(i), &y(i), sizeof(double)) != 0) return false;
    return true;
}

/// A three-level hierarchy on 15 -> 7 -> 3, Gauss-Seidel smoothing.
template <typename Acc>
double vcycle_residual(int cycles) {
    auto A0 = poisson_1d(15), A1 = poisson_1d(7), A2 = poisson_1d(3);
    std::vector<mat::compressed2D<double>> levels = {A0, A1, A2};
    std::vector<mat::compressed2D<double>> R = {itl::mg::make_restriction_1d(15),
                                                itl::mg::make_restriction_1d(7)};
    std::vector<mat::compressed2D<double>> P = {itl::mg::make_prolongation_1d(7),
                                                itl::mg::make_prolongation_1d(3)};
    auto sf = [](const mat::compressed2D<double>& A) {
        return itl::smoother::gauss_seidel<mat::compressed2D<double>>(A);
    };
    auto cs = [&A2](vec::dense_vector<double>& x, const vec::dense_vector<double>& b) {
        itl::smoother::gauss_seidel<mat::compressed2D<double>> gs(A2);
        for (int i = 0; i < 50; ++i) gs(x, b);
    };
    itl::mg::multigrid<double, Acc> mg(levels, R, P, sf, cs, 2, 2);
    vec::dense_vector<double> b(15, 1.0), x(15, 0.0);
    for (int c = 0; c < cycles; ++c) mg.vcycle(x, b);
    auto Ax = A0 * x;
    vec::dense_vector<double> r(15);
    for (std::size_t i = 0; i < 15; ++i) r(i) = b(i) - Ax(i);
    return mtl::two_norm(r) / mtl::two_norm(b);
}

} // namespace

TEST_CASE("multigrid: the accumulator reaches the transfers and the residual (#585)",
          "[itl][mg][multigrid][accumulator]") {
    const auto R = itl::mg::make_restriction_1d(15);
    const auto P = itl::mg::make_prolongation_1d(7);

    counting::products = 0;
    (void)itl::mg::restrict<counting::count_acc>(R, wave(15));
    CHECK(counting::products == R.nnz());

    counting::products = 0;
    (void)itl::mg::prolongate<counting::count_acc>(P, wave(7));
    CHECK(counting::products == P.nnz());

    // One V-cycle on 15 -> 7 -> 3: the residual and both transfers run once on
    // each level above the coarsest. The smoothers and the coarse solve are the
    // caller's void ones, so this is the whole count.
    counting::products = 0;
    (void)vcycle_residual<counting::count_acc>(1);
    const std::size_t expected = poisson_1d(15).nnz() + poisson_1d(7).nnz()
        + itl::mg::make_restriction_1d(15).nnz() + itl::mg::make_restriction_1d(7).nnz()
        + itl::mg::make_prolongation_1d(7).nnz() + itl::mg::make_prolongation_1d(3).nnz();
    CHECK(counting::products == expected);

    counting::products = 0;
    (void)itl::mg::restrict(R, wave(15));
    (void)itl::mg::prolongate(P, wave(7));
    (void)vcycle_residual<void>(1);
    CHECK(counting::products == 0);
}

TEST_CASE("multigrid: a plain element-type accumulator gives the void transfers' bits (#585)",
          "[itl][mg][multigrid][accumulator]") {
    const auto R = itl::mg::make_restriction_1d(31);
    const auto P = itl::mg::make_prolongation_1d(15);
    CHECK(same_bits(itl::mg::restrict<double>(R, wave(31)), itl::mg::restrict(R, wave(31))));
    CHECK(same_bits(itl::mg::prolongate<double>(P, wave(15)), itl::mg::prolongate(P, wave(15))));
}

TEST_CASE("multigrid: an accumulated V-cycle converges (#585)", "[itl][mg][multigrid][accumulator]") {
    const double v = vcycle_residual<void>(20);
    const double a = vcycle_residual<double>(20);
    INFO("void " << v << ", double accumulator " << a);
    CHECK(v < 1e-6);
    CHECK(a < 1e-6);
}
