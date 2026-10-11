// Row-major i-k-j generic GEMM (#549) and the CRS x dense matrix product (#550).
//
// Both kernels exist for speed only: the i-k-j form vectorizes where the i-j-k
// dot product walks B down a column, and the CRS product skips A's zeros instead
// of binary-searching every A(r, k). Neither may change a result. Each builds
// every C(r, c) as zero + a_0*b_0 + a_1*b_1 + ... in ascending k with the same
// generic_fma, so the tests compare BITS, not values within a tolerance -- a
// tolerance would pass a kernel that reordered the sum, which is the failure
// being watched for. The bit compare also pins -0 and the sign of infinities.
// A NaN must be a NaN in the same place, but its payload is not compared: IEEE
// leaves which operand's NaN an addition propagates unspecified, and compilers
// commute the operands of +, so two builds of the SAME loop can differ there.
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include <mtl/detail/wrapping_arithmetic.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/operation/mult.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/vec/dense_vector.hpp>

using namespace mtl;

namespace {

/// The i-j-k dot product the generic kernel used before #549: the reference.
template <typename T>
mat::dense2D<T> ijk(const mat::dense2D<T>& A, const mat::dense2D<T>& B) {
    mat::dense2D<T> C(A.num_rows(), B.num_cols());
    for (std::size_t r = 0; r < A.num_rows(); ++r)
        for (std::size_t c = 0; c < B.num_cols(); ++c) {
            T acc = math::zero<T>();
            for (std::size_t k = 0; k < A.num_cols(); ++k)
                acc = detail::generic_fma<T>(acc, A(r, k), B(k, c));
            C(r, c) = acc;
        }
    return C;
}

template <typename T>
bool same_bits(const mat::dense2D<T>& a, const mat::dense2D<T>& b) {
    if (a.num_rows() != b.num_rows() || a.num_cols() != b.num_cols())
        return false;
    for (std::size_t i = 0; i < a.num_rows(); ++i)
        for (std::size_t j = 0; j < a.num_cols(); ++j) {
            const T x = a(i, j), y = b(i, j);
            if (std::isnan(x) || std::isnan(y)) {
                if (!(std::isnan(x) && std::isnan(y)))
                    return false;
            } else if (std::memcmp(&x, &y, sizeof(T)) != 0) {
                return false;
            }
        }
    return true;
}

template <typename T>
mat::dense2D<T> random(std::size_t r, std::size_t c, std::mt19937& g, double zero_fraction = 0.0) {
    std::normal_distribution<double> n(0.0, 1.0);
    std::uniform_real_distribution<double> u(0.0, 1.0);
    mat::dense2D<T> m(r, c);
    for (std::size_t i = 0; i < r; ++i)
        for (std::size_t j = 0; j < c; ++j)
            m(i, j) = u(g) < zero_fraction ? T(0) : static_cast<T>(n(g));
    return m;
}

/// The CRS form of a dense matrix (its nonzeros, row by row, ascending column).
template <typename T>
mat::compressed2D<T> crs_of(const mat::dense2D<T>& d) {
    std::vector<std::size_t> starts(d.num_rows() + 1, 0), cols;
    std::vector<T> vals;
    for (std::size_t i = 0; i < d.num_rows(); ++i) {
        for (std::size_t j = 0; j < d.num_cols(); ++j)
            if (d(i, j) != T(0)) {
                cols.push_back(j);
                vals.push_back(d(i, j));
            }
        starts[i + 1] = cols.size();
    }
    return mat::compressed2D<T>(d.num_rows(), d.num_cols(), vals.size(), starts.data(), cols.data(), vals.data());
}

/// A class-typed scalar (no BLAS, no SIMD lane) wrapping a double.
struct emul {
    double v = 0.0;
    emul() = default;
    emul(double x) : v(x) {}
    friend emul operator+(emul a, emul b) { return {a.v + b.v}; }
    friend emul operator-(emul a, emul b) { return {a.v - b.v}; }
    friend emul operator*(emul a, emul b) { return {a.v * b.v}; }
    friend emul operator/(emul a, emul b) { return {a.v / b.v}; }
    friend emul operator-(emul a) { return {-a.v}; }
    emul& operator+=(emul b) { v += b.v; return *this; }
    friend bool operator==(emul a, emul b) { return a.v == b.v; }
    friend bool operator!=(emul a, emul b) { return a.v != b.v; }
};

} // namespace

TEST_CASE("generic GEMM: the i-k-j kernel is bit-identical to the i-j-k dot product", "[mult][ikj][549]") {
    std::mt19937 g(549);
    const std::size_t shapes[][3] = {{1, 1, 1}, {1, 7, 1}, {7, 1, 9}, {3, 5, 4}, {81, 81, 81},
                                     {81, 81, 15}, {81, 15, 81}, {15, 81, 81}, {64, 1, 64}};
    for (const auto& s : shapes) {
        INFO(s[0] << "x" << s[1] << " * " << s[1] << "x" << s[2]);
        {
            const auto A = random<double>(s[0], s[1], g, 0.3), B = random<double>(s[1], s[2], g);
            mat::dense2D<double> C(s[0], s[2]);
            detail::mult_generic(A, B, C);
            REQUIRE(same_bits(C, ijk(A, B)));
#if !defined(MTL5_HAS_BLAS) && !defined(MTL5_NATIVE_FAST_GEMM)
            // The dispatcher reaches this kernel only when neither BLAS nor the
            // native blocked GEMM claims dense2D<double>; those sum in their own order.
            mat::dense2D<double> D(s[0], s[2]);
            mult(A, B, D);
            REQUIRE(same_bits(D, C));
#endif
        }
        {
            const auto A = random<float>(s[0], s[1], g), B = random<float>(s[1], s[2], g);
            mat::dense2D<float> C(s[0], s[2]);
            detail::mult_generic(A, B, C);
            REQUIRE(same_bits(C, ijk(A, B)));
        }
    }
}

TEST_CASE("generic GEMM: i-k-j keeps -0, inf and NaN where the dot product puts them", "[mult][ikj][549]") {
    const double inf = std::numeric_limits<double>::infinity(), nan = std::numeric_limits<double>::quiet_NaN();
    mat::dense2D<double> A{{-0.0, 1.0, 0.0}, {inf, 0.0, -1.0}, {0.0, -0.0, 2.0}};
    mat::dense2D<double> B{{0.0, -0.0, 1.0}, {-0.0, nan, 2.0}, {1.0, 0.0, -inf}};
    mat::dense2D<double> C(3, 3);
    detail::mult_generic(A, B, C);
    REQUIRE(same_bits(C, ijk(A, B)));
}

TEST_CASE("generic GEMM: a class-typed scalar takes the i-k-j path and the same values", "[mult][ikj][549]") {
    std::mt19937 g(5490);
    std::normal_distribution<double> n;
    mat::dense2D<emul> A(9, 11), B(11, 6);
    for (std::size_t i = 0; i < 9; ++i)
        for (std::size_t j = 0; j < 11; ++j)
            A(i, j) = emul(n(g));
    for (std::size_t i = 0; i < 11; ++i)
        for (std::size_t j = 0; j < 6; ++j)
            B(i, j) = emul(n(g));
    mat::dense2D<emul> C(9, 6);
    detail::mult_generic(A, B, C);
    const auto R = ijk(A, B);
    for (std::size_t i = 0; i < 9; ++i)
        for (std::size_t j = 0; j < 6; ++j)
            REQUIRE(std::memcmp(&C(i, j).v, &R(i, j).v, sizeof(double)) == 0);
}

TEST_CASE("generic GEMM: a non-row-major operand still takes the dot-product path", "[mult][ikj][549]") {
    std::mt19937 g(54900);
    const auto A = random<double>(6, 5, g), Bt = random<double>(7, 5, g);
    mat::dense2D<double> C(6, 7);
    mult(A, trans(Bt), C);  // a transposed view has no row-major storage
    mat::dense2D<double> B(5, 7);
    for (std::size_t i = 0; i < 5; ++i)
        for (std::size_t j = 0; j < 7; ++j)
            B(i, j) = Bt(j, i);
    REQUIRE(same_bits(C, ijk(A, B)));
}

TEST_CASE("CRS x dense matrix product equals the dense product, bit for bit", "[mult][crs][550]") {
    std::mt19937 g(550);
    const std::size_t shapes[][3] = {{1, 1, 1}, {5, 5, 5}, {81, 81, 81}, {87, 81, 81}, {81, 81, 15}, {30, 1, 4}};
    for (const double zeros : {0.0, 0.5, 0.95, 1.0}) {
        for (const auto& s : shapes) {
            INFO(s[0] << "x" << s[1] << " * " << s[1] << "x" << s[2] << ", zero fraction " << zeros);
            const auto Ad = random<double>(s[0], s[1], g, zeros), B = random<double>(s[1], s[2], g);
            const auto A = crs_of(Ad);
            mat::dense2D<double> C(s[0], s[2]);
            mult(A, B, C);
            // The dense product's order with A's zero terms (exact zeros) left out.
            REQUIRE(same_bits(C, ijk(Ad, B)));
            // And the column-by-column sparse matvec, the only route before #550.
            for (std::size_t j = 0; j < s[2]; ++j) {
                vec::dense_vector<double> x(s[1]), y(s[0]);
                for (std::size_t k = 0; k < s[1]; ++k)
                    x(k) = B(k, j);
                mult(A, x, y);
                for (std::size_t i = 0; i < s[0]; ++i) {
                    const double cij = C(i, j), yi = y(i);
                    REQUIRE(std::memcmp(&cij, &yi, sizeof(double)) == 0);
                }
            }
        }
    }
}

TEST_CASE("CRS x dense matrix product overwrites C", "[mult][crs][550]") {
    mat::dense2D<double> Ad{{0.0, 2.0}, {0.0, 0.0}};
    mat::dense2D<double> B{{1.0, 1.0}, {3.0, 4.0}};
    mat::dense2D<double> C{{9.0, 9.0}, {9.0, 9.0}};
    mult(crs_of(Ad), B, C);
    REQUIRE(C(0, 0) == 6.0);
    REQUIRE(C(0, 1) == 8.0);
    REQUIRE(C(1, 0) == 0.0);  // an empty row is zero, not left over
    REQUIRE(C(1, 1) == 0.0);
}
