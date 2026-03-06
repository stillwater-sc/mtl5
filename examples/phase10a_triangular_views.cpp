// phase10a_triangular_views.cpp - Triangular Views and the LU Decomposition
//
// This example demonstrates:
//   1. upper(), lower(), strict_upper(), strict_lower() views
//   2. The identity A = lower(A) + strict_upper(A) = upper(A) + strict_lower(A)
//   3. MATLAB-compatible triu(A,k) and tril(A,k) with diagonal offsets
//   4. How triangular views feed directly into trisolve operations
//   5. Extracting L and U from a packed LU factorization using views
//
// Views are non-owning and zero-cost: they don't copy data, they just
// filter which elements are visible.  This makes them ideal for working
// with packed factorizations where L and U share one matrix.

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace mtl;

void print_matrix(const std::string& name, auto const& M,
                  std::size_t rows, std::size_t cols) {
    std::cout << name << ":\n";
    for (std::size_t i = 0; i < rows; ++i) {
        std::cout << "  [";
        for (std::size_t j = 0; j < cols; ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(4) << std::setw(8) << M(i, j);
        }
        std::cout << "]\n";
    }
    std::cout << "\n";
}

void print_vector(const std::string& name, const vec::dense_vector<double>& v) {
    std::cout << "  " << name << " = [";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(6) << v(i);
    }
    std::cout << "]\n";
}

int main() {
    std::cout << "=============================================================\n";
    std::cout << " Phase 10A: Triangular Views\n";
    std::cout << "=============================================================\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // Part 1: The Four Triangular Views
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== Part 1: The Four Triangular Views ===\n\n";

    const std::size_t n = 4;
    mat::dense2D<double> A(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            A(i, j) = static_cast<double>(i * n + j + 1);

    print_matrix("A (original)", A, n, n);

    // upper(A): elements on and above the diagonal
    auto U = upper(A);
    print_matrix("upper(A) - includes diagonal", U, n, n);

    // lower(A): elements on and below the diagonal
    auto L = lower(A);
    print_matrix("lower(A) - includes diagonal", L, n, n);

    // strict_upper(A): elements strictly above the diagonal
    auto SU = strict_upper(A);
    print_matrix("strict_upper(A) - excludes diagonal", SU, n, n);

    // strict_lower(A): elements strictly below the diagonal
    auto SL = strict_lower(A);
    print_matrix("strict_lower(A) - excludes diagonal", SL, n, n);

    // ══════════════════════════════════════════════════════════════════════
    // Part 2: Decomposition Identity
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== Part 2: A = lower(A) + strict_upper(A) ===\n\n";

    std::cout << "  Verifying element-wise: L(i,j) + SU(i,j) == A(i,j)\n";
    double max_err = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            max_err = std::max(max_err, std::abs(L(i, j) + SU(i, j) - A(i, j)));
    std::cout << "  Max error: " << std::scientific << max_err << "\n\n";

    std::cout << "  Similarly: upper(A) + strict_lower(A) == A\n";
    max_err = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            max_err = std::max(max_err, std::abs(U(i, j) + SL(i, j) - A(i, j)));
    std::cout << "  Max error: " << std::scientific << max_err << "\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // Part 3: MATLAB-Compatible triu/tril with Offsets
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== Part 3: triu(A, k) and tril(A, k) ===\n\n";

    std::cout << "  triu(A, k) keeps elements on diagonal k and above.\n";
    std::cout << "  tril(A, k) keeps elements on diagonal k and below.\n\n";

    // triu(A, 0) = upper(A)
    auto T0 = triu(A, 0);
    print_matrix("triu(A, 0) - same as upper(A)", T0, n, n);

	// triu(A, 1) = 1st superdiagonal and above
	auto T1 = triu(A, 1);
	print_matrix("triu(A, 1) - 1st superdiagonal and above - same as strict_upper(A)", T1, n, n);

    // triu(A, 2) = 2nd superdiagonal and above
    auto T2 = triu(A, 2);
    print_matrix("triu(A, 2) - 2nd superdiagonal and above", T2, n, n);

    // tril(A, -1) = strict lower
    auto TL1 = tril(A, -1);
    print_matrix("tril(A, -1) - same as strict_lower(A)", TL1, n, n);

    // ══════════════════════════════════════════════════════════════════════
    // Part 4: Triangular Views with LU Factorization
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== Part 4: Extracting L and U from Packed LU ===\n\n";

    // Build a well-conditioned SPD matrix for LU
    mat::dense2D<double> B(n, n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < n; ++j)
            B(i, j) = 1.0 / (1.0 + std::abs(static_cast<double>(i) - static_cast<double>(j)));
        B(i, i) += 2.0;
    }

    vec::dense_vector<double> x_true = {1.0, 2.0, 3.0, 4.0};
    auto b = B * x_true;

    print_matrix("B (original)", B, n, n);

    // LU factorization: L and U are packed in one matrix
    mat::dense2D<double> LU(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            LU(i, j) = B(i, j);

    std::vector<std::size_t> pivot;
    lu_factor(LU, pivot);

    std::cout << "After LU factorization (L\\U packed in one matrix):\n";
    print_matrix("LU (packed)", LU, n, n);

    // Extract U (upper triangle including diagonal)
    auto LU_upper = upper(LU);
    print_matrix("upper(LU) = U", LU_upper, n, n);

    // Extract L (strict lower triangle - diagonal of L is implicitly 1)
    auto LU_lower = strict_lower(LU);
    std::cout << "strict_lower(LU) = L (without unit diagonal):\n";
    print_matrix("strict_lower(LU)", LU_lower, n, n);
    std::cout << "  (The actual L has 1s on the diagonal - unit lower triangular)\n\n";

    // ── Solve using the factorization ───────────────────────────────────
    std::cout << "- Solving B*x = b using packed LU -\n";
    vec::dense_vector<double> x(n);
    lu_solve(LU, pivot, x, b);
    print_vector("x_computed", x);
    print_vector("x_true    ", x_true);

    double norm_err = 0.0;
    for (std::size_t i = 0; i < n; ++i)
        norm_err += (x(i) - x_true(i)) * (x(i) - x_true(i));
    std::cout << "  ||x - x_true|| = " << std::scientific << std::sqrt(norm_err) << "\n\n";

    // ══════════════════════════════════════════════════════════════════════
    // Part 5: Direct Trisolve with Views
    // ══════════════════════════════════════════════════════════════════════
    std::cout << "=== Part 5: Trisolve with Triangular Views ===\n\n";

    // Build a known lower triangular system: L*x = b
    mat::dense2D<double> L_full(3, 3);
    L_full(0, 0) = 2.0; L_full(0, 1) = 0.0; L_full(0, 2) = 0.0;
    L_full(1, 0) = 3.0; L_full(1, 1) = 4.0; L_full(1, 2) = 0.0;
    L_full(2, 0) = 1.0; L_full(2, 1) = 5.0; L_full(2, 2) = 6.0;

    // Even though L_full is already lower triangular, the view enforces
    // the contract at the type level - upper entries are guaranteed zero.
    auto L_view = lower(L_full);

    vec::dense_vector<double> b3 = {2.0, 11.0, 29.0};
    vec::dense_vector<double> x3(3);
    lower_trisolve(L_view, x3, b3);

    std::cout << "  Lower triangular system L*x = b:\n";
    print_matrix("  L", L_view, 3, 3);
    print_vector("b", b3);
    print_vector("x (forward substitution)", x3);
    std::cout << "  Expected: x = [1, 2, 3]\n\n";

    // ── Takeaways ────────────────────────────────────────────────────────
    std::cout << "=== Takeaways ===\n\n";
    std::cout << "  1. Views are ZERO-COST: no data copied, just filtered access\n";
    std::cout << "  2. upper(A) + strict_lower(A) = A  (decomposition identity)\n";
    std::cout << "  3. triu(A,k) / tril(A,k) generalize with diagonal offset k\n";
    std::cout << "  4. LU stores L\\U packed - views extract each part cleanly\n";
    std::cout << "  5. Views satisfy the Matrix concept - pass them to any solver\n";

    return EXIT_SUCCESS;
}
