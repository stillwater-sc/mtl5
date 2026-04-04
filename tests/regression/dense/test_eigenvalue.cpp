#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/eigenvalue_symmetric.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/generators/randsym.hpp>
#include <mtl/generators/wilkinson.hpp>

using namespace mtl;

namespace {

void report_trace(const char* matrix, int n, double trace_A, double trace_eig, double tol) {
    double err = std::abs(trace_eig - trace_A);
    std::cout << std::left << std::setw(12) << matrix
              << "  n=" << std::setw(6) << n
              << "  tr(A)=" << std::scientific << std::setprecision(6) << trace_A
              << "  sum(eig)=" << trace_eig
              << "  |diff|=" << std::setprecision(3) << err
              << "  tol=" << tol
              << (err < tol ? "  PASS" : "  FAIL")
              << std::defaultfloat << std::endl;
}

} // anonymous namespace

TEST_CASE("Symmetric eigenvalue regression: randsym", "[regression][dense][eigenvalue]") {
    auto n = GENERATE(100, 500);
    auto A = generators::randsym<double>(static_cast<std::size_t>(n), 100.0);

    auto eigenvalues = eigenvalue_symmetric(A);
    REQUIRE(eigenvalues.size() == static_cast<std::size_t>(n));

    // Eigenvalues should be sorted ascending
    for (std::size_t i = 1; i < eigenvalues.size(); ++i)
        REQUIRE(eigenvalues(i) >= eigenvalues(i - 1) - 1e-10);

    double trace_A = 0.0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i)
        trace_A += A(i, i);

    double trace_eig = 0.0;
    for (std::size_t i = 0; i < eigenvalues.size(); ++i)
        trace_eig += eigenvalues(i);

    double tol = double(n) * 100.0 * std::numeric_limits<double>::epsilon() * std::abs(trace_A);
    tol = std::max(tol, 1e-8);
    report_trace("randsym", n, trace_A, trace_eig, tol);
    REQUIRE(std::abs(trace_eig - trace_A) < tol);
}

TEST_CASE("Symmetric eigenvalue regression: Wilkinson", "[regression][dense][eigenvalue]") {
    auto n = GENERATE(101, 501);
    auto A = generators::wilkinson<double>(n);

    auto eigenvalues = eigenvalue_symmetric(A);
    REQUIRE(eigenvalues.size() == static_cast<std::size_t>(n));

    double trace_A = 0.0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i)
        trace_A += A(i, i);

    double trace_eig = 0.0;
    for (std::size_t i = 0; i < eigenvalues.size(); ++i)
        trace_eig += eigenvalues(i);

    double tol = double(n) * 100.0 * std::numeric_limits<double>::epsilon() * std::abs(trace_A);
    tol = std::max(tol, 1e-8);
    report_trace("Wilkinson", n, trace_A, trace_eig, tol);
    REQUIRE(std::abs(trace_eig - trace_A) < tol);
}

// NOTE: SVD regression test omitted — native SVD has accuracy issues beyond
// n~20 (see known limitation). Tracked separately for investigation.
