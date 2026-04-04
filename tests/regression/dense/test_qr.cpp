#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/qr.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/generators/frank.hpp>
#include <mtl/generators/kahan.hpp>

using namespace mtl;

namespace {

mat::dense2D<double> copy_matrix(const mat::dense2D<double>& A) {
    auto n = A.num_rows(), m = A.num_cols();
    mat::dense2D<double> B(n, m);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < m; ++j)
            B(i, j) = A(i, j);
    return B;
}

double factorization_error(const mat::dense2D<double>& A,
                           const mat::dense2D<double>& Q,
                           const mat::dense2D<double>& R) {
    auto QR = Q * R;
    double err = 0.0;
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (std::size_t j = 0; j < A.num_cols(); ++j) {
            double d = QR(i, j) - A(i, j);
            err += d * d;
        }
    return std::sqrt(err) / frobenius_norm(A);
}

double orthogonality_error(const mat::dense2D<double>& Q) {
    auto n = Q.num_cols();
    auto QtQ = trans(Q) * Q;
    double err = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            double expected = (i == j) ? 1.0 : 0.0;
            double d = QtQ(i, j) - expected;
            err += d * d;
        }
    return std::sqrt(err);
}

void report(const char* test, const char* matrix, int n,
            const char* metric, double value, double tol) {
    std::cout << std::left << std::setw(10) << test
              << std::setw(8) << matrix
              << "  n=" << std::setw(6) << n
              << "  " << std::setw(14) << metric << "="
              << std::scientific << std::setprecision(3) << value
              << "  tol=" << tol
              << (value < tol ? "  PASS" : "  FAIL")
              << std::defaultfloat << std::endl;
}

} // anonymous namespace

TEST_CASE("QR regression: Frank matrix", "[regression][dense][qr]") {
    auto n = GENERATE(100, 500);
    auto A = generators::frank<double>(n);
    auto Acopy = copy_matrix(A);

    vec::dense_vector<double> tau(n);
    int info = qr_factor(Acopy, tau);
    REQUIRE(info == 0);

    auto Q = qr_extract_Q(Acopy, tau);
    auto R = qr_extract_R(Acopy);

    double tol = double(n) * std::numeric_limits<double>::epsilon();
    double fe = factorization_error(A, Q, R);
    double oe = orthogonality_error(Q);
    report("QR", "Frank", n, "||QR-A||/||A||", fe, tol);
    report("QR", "Frank", n, "||Q'Q-I||     ", oe, tol);
    REQUIRE(fe < tol);
    REQUIRE(oe < tol);
}

TEST_CASE("QR regression: Kahan matrix", "[regression][dense][qr]") {
    auto n = GENERATE(100, 500);
    auto A = generators::kahan<double>(n);
    auto Acopy = copy_matrix(A);

    vec::dense_vector<double> tau(n);
    int info = qr_factor(Acopy, tau);
    REQUIRE(info == 0);

    auto Q = qr_extract_Q(Acopy, tau);
    auto R = qr_extract_R(Acopy);

    double tol = double(n) * 100.0 * std::numeric_limits<double>::epsilon();
    double fe = factorization_error(A, Q, R);
    double oe = orthogonality_error(Q);
    report("QR", "Kahan", n, "||QR-A||/||A||", fe, tol);
    report("QR", "Kahan", n, "||Q'Q-I||     ", oe, tol);
    REQUIRE(fe < tol);
    REQUIRE(oe < tol);
}

TEST_CASE("QR regression: solve via QR", "[regression][dense][qr]") {
    auto n = GENERATE(100, 500);
    auto A = generators::frank<double>(n);
    vec::dense_vector<double> x_exact(n, 1.0);
    auto b = A * x_exact;

    auto Acopy = copy_matrix(A);
    vec::dense_vector<double> tau(n);
    qr_factor(Acopy, tau);

    vec::dense_vector<double> x(n);
    qr_solve(Acopy, tau, x, b);

    auto r = A * x;
    double res = 0.0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
        double d = r(i) - b(i);
        res += d * d;
    }
    res = std::sqrt(res) / (frobenius_norm(A) * two_norm(x));
    double tol = double(n) * std::numeric_limits<double>::epsilon();
    report("QR-solve", "Frank", n, "backward_err  ", res, tol);
    REQUIRE(res < tol);
}
