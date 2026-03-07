// phase8b_expression_concepts.cpp - Expression Template Mechanics
//
// This example demonstrates:
//   1. Lazy capture: expressions hold references, not computed values
//   2. Type inspection: expression types vs concrete types
//   3. Nested expression trees: composing multiple operations
//   4. Three ways to materialize an expression
//   5. Fused accumulation: evaluate and add in one pass
//   6. Compound assignment operators (+=, -=)
//   7. Vector expressions
//   8. Mixed eager/lazy: matrix-vector multiply + vector add
//   9. Unary operations: negation and scalar division
//
// Expression templates defer computation until assignment, building
// a compile-time tree that the compiler can optimize into a single
// fused loop - no temporary allocations.

#include <mtl/mtl.hpp>
#include <iostream>
#include <iomanip>
#include <typeinfo>

using namespace mtl;

void print_matrix(const std::string& name, auto const& M,
                  std::size_t rows, std::size_t cols) {
    std::cout << name << " (" << rows << "x" << cols << "):\n";
    for (std::size_t i = 0; i < rows; ++i) {
        std::cout << "  [";
        for (std::size_t j = 0; j < cols; ++j) {
            if (j > 0) std::cout << ", ";
            std::cout << std::fixed << std::setprecision(2) << std::setw(7) << M(i, j);
        }
        std::cout << "]\n";
    }
}

void print_vector(const std::string& name, auto const& v, std::size_t n) {
    std::cout << name << " (" << n << "): [";
    for (std::size_t i = 0; i < n; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << std::fixed << std::setprecision(2) << std::setw(7) << v(i);
    }
    std::cout << "]\n";
}

int main() {
    std::cout << "=============================================================\n";
    std::cout << " Phase 8B: Expression Template Mechanics - A Guided Tour\n";
    std::cout << "=============================================================\n\n";

    const std::size_t n = 3;

    // Set up small matrices for clear output
    mat::dense2D<double> A(n, n), B(n, n), C(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j) {
            A(i, j) = static_cast<double>(i * n + j + 1);  // 1..9
            B(i, j) = static_cast<double>((i + j) * 2);    // 0,2,4,...
            C(i, j) = 10.0 * (i + 1);                      // row-constant
        }

    std::cout << "Source matrices:\n";
    print_matrix("A", A, n, n);
    print_matrix("B", B, n, n);
    print_matrix("C", C, n, n);
    std::cout << "\n";

    // ======================================================================
    // Part 1: Lazy Capture - Expressions Hold References
    // ======================================================================
    std::cout << "=== Part 1: Lazy Capture ===\n";
    std::cout << "auto expr = A + B;  // no computation yet!\n";
    std::cout << "The expression stores references to A and B.\n";
    std::cout << "Modifying A changes the expression's result.\n\n";

    auto expr1 = A + B;

    // Evaluate before modification
    mat::dense2D<double> before = expr1;
    std::cout << "Before modifying A:\n";
    print_matrix("A + B", before, n, n);

    // Modify A(0,0) and re-evaluate the SAME expression
    double old_val = A(0, 0);
    A(0, 0) = 999.0;
    mat::dense2D<double> after = expr1;
    std::cout << "\nAfter setting A(0,0) = 999:\n";
    print_matrix("A + B (same expr)", after, n, n);

    bool lazy_ok = (after(0, 0) == 999.0 + B(0, 0));
    std::cout << "Expression reflects modification: " << (lazy_ok ? "PASS" : "FAIL") << "\n";
    A(0, 0) = old_val;  // restore
    std::cout << "\n";

    // ======================================================================
    // Part 2: Type Inspection
    // ======================================================================
    std::cout << "=== Part 2: Type Inspection ===\n";
    std::cout << "Expression types differ from concrete matrix types.\n";
    std::cout << "The is_expression_v trait distinguishes them.\n\n";

    auto expr_add  = A + B;
    auto expr_scal = 2.0 * A;
    auto expr_neg  = -A;

    std::cout << "Type of dense2D<double>: " << typeid(A).name() << "\n";
    std::cout << "Type of A + B:           " << typeid(expr_add).name() << "\n";
    std::cout << "Type of 2.0 * A:         " << typeid(expr_scal).name() << "\n";
    std::cout << "Type of -A:              " << typeid(expr_neg).name() << "\n\n";

    std::cout << "is_expression_v<dense2D<double>>: "
              << std::boolalpha << traits::is_expression_v<mat::dense2D<double>> << "\n";
    std::cout << "is_expression_v<decltype(A+B)>:   "
              << std::boolalpha << traits::is_expression_v<decltype(expr_add)> << "\n";
    std::cout << "is_expression_v<decltype(2*A)>:   "
              << std::boolalpha << traits::is_expression_v<decltype(expr_scal)> << "\n";
    std::cout << "is_expression_v<decltype(-A)>:    "
              << std::boolalpha << traits::is_expression_v<decltype(expr_neg)> << "\n\n";

    // ======================================================================
    // Part 3: Nested Expression Trees
    // ======================================================================
    std::cout << "=== Part 3: Nested Expression Trees ===\n";
    std::cout << "Expressions compose: 2.0*A + B - C builds a tree:\n";
    std::cout << "          (-)\n";
    std::cout << "         /   \\\n";
    std::cout << "       (+)    C\n";
    std::cout << "      /   \\\n";
    std::cout << "   (2*A)   B\n\n";

    auto nested = 2.0 * A + B - C;
    std::cout << "is_expression_v<decltype(nested)>: "
              << std::boolalpha << traits::is_expression_v<decltype(nested)> << "\n";

    mat::dense2D<double> R3 = nested;
    print_matrix("2.0*A + B - C", R3, n, n);

    // Verify element-by-element
    bool nested_ok = true;
    for (std::size_t i = 0; i < n && nested_ok; ++i)
        for (std::size_t j = 0; j < n && nested_ok; ++j) {
            double expected = 2.0 * A(i, j) + B(i, j) - C(i, j);
            if (std::abs(R3(i, j) - expected) > 1e-12) nested_ok = false;
        }
    std::cout << "Correctness: " << (nested_ok ? "PASS" : "FAIL") << "\n\n";

    // ======================================================================
    // Part 4: Three Ways to Materialize
    // ======================================================================
    std::cout << "=== Part 4: Three Ways to Materialize ===\n\n";

    auto expr4 = 2.0 * A + B;

    // Way 1: Direct assignment
    std::cout << "1) Assignment:    dense2D<double> R = expr;\n";
    mat::dense2D<double> R4a = expr4;
    print_matrix("R4a", R4a, n, n);

    // Way 2: evaluate()
    std::cout << "\n2) evaluate():    auto R = evaluate(expr);\n";
    auto R4b = evaluate(expr4);
    print_matrix("R4b", R4b, n, n);

    // Way 3: fused_assign()
    std::cout << "\n3) fused_assign(): fused_assign(R, expr);\n";
    mat::dense2D<double> R4c(n, n);
    fused_assign(R4c, expr4);
    print_matrix("R4c", R4c, n, n);

    // All three should match
    bool mat_ok = true;
    for (std::size_t i = 0; i < n && mat_ok; ++i)
        for (std::size_t j = 0; j < n && mat_ok; ++j) {
            if (R4a(i,j) != R4b(i,j) || R4b(i,j) != R4c(i,j))
                mat_ok = false;
        }
    std::cout << "\nAll three methods agree: " << (mat_ok ? "PASS" : "FAIL") << "\n\n";

    // ======================================================================
    // Part 5: Fused Accumulation
    // ======================================================================
    std::cout << "=== Part 5: Fused Plus-Assign ===\n";
    std::cout << "fused_plus_assign(C_acc, 2.0*A + B) adds the expression\n";
    std::cout << "result to C_acc in a single pass - no temporary.\n\n";

    mat::dense2D<double> C_acc(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            C_acc(i, j) = 100.0;

    std::cout << "Before: C_acc = 100.0 everywhere\n";
    fused_plus_assign(C_acc, 2.0 * A + B);
    print_matrix("After fused_plus_assign", C_acc, n, n);

    bool fpa_ok = true;
    for (std::size_t i = 0; i < n && fpa_ok; ++i)
        for (std::size_t j = 0; j < n && fpa_ok; ++j) {
            double expected = 100.0 + 2.0 * A(i,j) + B(i,j);
            if (std::abs(C_acc(i,j) - expected) > 1e-12) fpa_ok = false;
        }
    std::cout << "Correctness: " << (fpa_ok ? "PASS" : "FAIL") << "\n\n";

    // ======================================================================
    // Part 6: Compound Assignment Operators
    // ======================================================================
    std::cout << "=== Part 6: Compound Assignment (+=, -=) ===\n";
    std::cout << "C += A + B  and  C -= A  also accept expressions.\n\n";

    mat::dense2D<double> D(n, n);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = 0; j < n; ++j)
            D(i, j) = 0.0;

    D += A + B;
    print_matrix("D after D += A + B", D, n, n);

    D -= 2.0 * B;
    print_matrix("D after D -= 2*B", D, n, n);

    // Should equal A + B - 2*B = A - B
    bool comp_ok = true;
    for (std::size_t i = 0; i < n && comp_ok; ++i)
        for (std::size_t j = 0; j < n && comp_ok; ++j) {
            double expected = A(i,j) - B(i,j);
            if (std::abs(D(i,j) - expected) > 1e-12) comp_ok = false;
        }
    std::cout << "D == A - B: " << (comp_ok ? "PASS" : "FAIL") << "\n\n";

    // ======================================================================
    // Part 7: Vector Expressions
    // ======================================================================
    std::cout << "=== Part 7: Vector Expressions ===\n";
    std::cout << "Vectors enjoy the same lazy expression machinery.\n\n";

    vec::dense_vector<double> u(4), v(4);
    for (std::size_t i = 0; i < 4; ++i) {
        u(i) = static_cast<double>(i + 1);       // 1, 2, 3, 4
        v(i) = static_cast<double>(10 * (i + 1)); // 10, 20, 30, 40
    }

    print_vector("u", u, 4);
    print_vector("v", v, 4);

    vec::dense_vector<double> w = 2.0 * u + v;
    print_vector("w = 2*u + v", w, 4);

    bool vec_ok = true;
    for (std::size_t i = 0; i < 4; ++i) {
        double expected = 2.0 * u(i) + v(i);
        if (std::abs(w(i) - expected) > 1e-12) vec_ok = false;
    }
    std::cout << "Correctness: " << (vec_ok ? "PASS" : "FAIL") << "\n\n";

    // ======================================================================
    // Part 8: Mixed Eager/Lazy - Matrix-Vector Multiply + Vector Add
    // ======================================================================
    std::cout << "=== Part 8: Mixed Eager/Lazy ===\n";
    std::cout << "Matrix-vector multiply (A*x) is EAGER - it returns a\n";
    std::cout << "concrete dense_vector immediately. But the subsequent\n";
    std::cout << "vector addition is lazy. So in y = A*x + b, the matvec\n";
    std::cout << "is computed first, then the add is fused into assignment.\n\n";

    mat::dense2D<double> M(3, 3);
    vec::dense_vector<double> x(3), b(3);
    M(0,0) = 1; M(0,1) = 0; M(0,2) = 0;
    M(1,0) = 0; M(1,1) = 2; M(1,2) = 0;
    M(2,0) = 0; M(2,1) = 0; M(2,2) = 3;
    x(0) = 1; x(1) = 2; x(2) = 3;
    b(0) = 10; b(1) = 20; b(2) = 30;

    print_matrix("M (diagonal)", M, 3, 3);
    print_vector("x", x, 3);
    print_vector("b", b, 3);

    vec::dense_vector<double> y = M * x + b;
    print_vector("y = M*x + b", y, 3);

    // M*x = [1, 4, 9], + b = [11, 24, 39]
    bool mixed_ok = (std::abs(y(0) - 11.0) < 1e-12 &&
                     std::abs(y(1) - 24.0) < 1e-12 &&
                     std::abs(y(2) - 39.0) < 1e-12);
    std::cout << "Correctness: " << (mixed_ok ? "PASS" : "FAIL") << "\n";

    // Type check: M*x is concrete, M*x + b is an expression
    auto matvec = M * x;
    auto matvec_plus_b = M * x + b;
    std::cout << "is_expression_v<decltype(M*x)>:     "
              << std::boolalpha << traits::is_expression_v<decltype(matvec)> << "  (eager)\n";
    std::cout << "is_expression_v<decltype(M*x + b)>: "
              << std::boolalpha << traits::is_expression_v<decltype(matvec_plus_b)> << " (lazy add)\n\n";

    // ======================================================================
    // Part 9: Unary Operations - Negation and Scalar Division
    // ======================================================================
    std::cout << "=== Part 9: Unary Operations ===\n\n";

    // Negation
    mat::dense2D<double> neg_A = -A;
    print_matrix("-A", neg_A, n, n);

    bool neg_ok = true;
    for (std::size_t i = 0; i < n && neg_ok; ++i)
        for (std::size_t j = 0; j < n && neg_ok; ++j)
            if (std::abs(neg_A(i,j) + A(i,j)) > 1e-12) neg_ok = false;
    std::cout << "-A == -(A): " << (neg_ok ? "PASS" : "FAIL") << "\n\n";

    // Scalar division
    mat::dense2D<double> half_A = A / 2.0;
    print_matrix("A / 2.0", half_A, n, n);

    bool div_ok = true;
    for (std::size_t i = 0; i < n && div_ok; ++i)
        for (std::size_t j = 0; j < n && div_ok; ++j)
            if (std::abs(half_A(i,j) - A(i,j) / 2.0) > 1e-12) div_ok = false;
    std::cout << "A/2.0 == A*0.5: " << (div_ok ? "PASS" : "FAIL") << "\n\n";

    // Combined: -A / 2.0 + B
    auto expr9 = -A / 2.0 + B;
    std::cout << "is_expression_v<decltype(-A/2.0 + B)>: "
              << std::boolalpha << traits::is_expression_v<decltype(expr9)> << "\n";
    mat::dense2D<double> R9 = expr9;
    print_matrix("-A/2.0 + B", R9, n, n);

    bool comb_ok = true;
    for (std::size_t i = 0; i < n && comb_ok; ++i)
        for (std::size_t j = 0; j < n && comb_ok; ++j) {
            double expected = -A(i,j) / 2.0 + B(i,j);
            if (std::abs(R9(i,j) - expected) > 1e-12) comb_ok = false;
        }
    std::cout << "Correctness: " << (comb_ok ? "PASS" : "FAIL") << "\n\n";

    // ======================================================================
    // Key Takeaways
    // ======================================================================
    std::cout << "=== Key Takeaways ===\n";
    std::cout << "1. Expression templates defer computation until assignment.\n";
    std::cout << "   auto expr = A + B creates a lightweight proxy, not a matrix.\n";
    std::cout << "2. Expressions store references to lvalue operands - modifying\n";
    std::cout << "   the source changes the expression result (lazy semantics).\n";
    std::cout << "3. Nested expressions compose into trees that the compiler\n";
    std::cout << "   can optimize into a single fused loop.\n";
    std::cout << "4. Three materialization methods: assignment, evaluate(),\n";
    std::cout << "   fused_assign(). All produce the same result.\n";
    std::cout << "5. fused_plus_assign() accumulates without temporaries.\n";
    std::cout << "6. Matrix-matrix and matrix-vector multiplies stay EAGER\n";
    std::cout << "   (avoiding O(n^3)-per-element recomputation and aliasing).\n";
    std::cout << "   Element-wise ops (+, -, scalar*, /) are LAZY.\n";
    std::cout << "7. is_expression_v<T> distinguishes expressions from concrete\n";
    std::cout << "   types at compile time - useful for overload resolution.\n";

    return EXIT_SUCCESS;
}
