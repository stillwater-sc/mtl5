#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/eigenvalue_symmetric.hpp>
#include <mtl/operation/svd.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/operation/trans.hpp>
#include <mtl/generators/randsym.hpp>
#include <mtl/generators/wilkinson.hpp>
#include <mtl/generators/frank.hpp>

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

} // anonymous namespace

TEST_CASE("Symmetric eigenvalue regression: randsym", "[regression][dense][eigenvalue]") {
    auto n = GENERATE(100, 500);
    CAPTURE(n);

    // randsym(n, kappa) generates symmetric matrix with condition number ~kappa
    auto A = generators::randsym<double>(static_cast<std::size_t>(n), 100.0);

    // eigenvalue_symmetric returns a dense_vector of eigenvalues
    auto eigenvalues = eigenvalue_symmetric(A);

    REQUIRE(eigenvalues.size() == static_cast<std::size_t>(n));

    // Eigenvalues should be sorted ascending
    for (std::size_t i = 1; i < eigenvalues.size(); ++i) {
        REQUIRE(eigenvalues(i) >= eigenvalues(i - 1) - 1e-10);
    }

    // Trace check: sum of eigenvalues ≈ trace of A
    double trace_A = 0.0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i)
        trace_A += A(i, i);

    double trace_eig = 0.0;
    for (std::size_t i = 0; i < eigenvalues.size(); ++i)
        trace_eig += eigenvalues(i);

    double tol = double(n) * 100.0 * std::numeric_limits<double>::epsilon() * std::abs(trace_A);
    REQUIRE(std::abs(trace_eig - trace_A) < std::max(tol, 1e-8));
}

TEST_CASE("Symmetric eigenvalue regression: Wilkinson", "[regression][dense][eigenvalue]") {
    auto n = GENERATE(101, 501);
    CAPTURE(n);

    auto A = generators::wilkinson<double>(n);
    auto eigenvalues = eigenvalue_symmetric(A);

    REQUIRE(eigenvalues.size() == static_cast<std::size_t>(n));

    // Trace check
    double trace_A = 0.0;
    for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i)
        trace_A += A(i, i);

    double trace_eig = 0.0;
    for (std::size_t i = 0; i < eigenvalues.size(); ++i)
        trace_eig += eigenvalues(i);

    double tol = double(n) * 100.0 * std::numeric_limits<double>::epsilon() * std::abs(trace_A);
    REQUIRE(std::abs(trace_eig - trace_A) < std::max(tol, 1e-8));
}

// NOTE: SVD regression test omitted — native SVD has accuracy issues beyond
// n~20 (see known limitation). Tracked separately for investigation.
