#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <mtl/mat/dense2D.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/mat/operators.hpp>
#include <mtl/vec/operators.hpp>
#include <mtl/operation/lazy.hpp>
#include <mtl/operation/fuse.hpp>
#include <mtl/traits/is_expression.hpp>
#include <mtl/concepts/matrix.hpp>
#include <mtl/concepts/vector.hpp>

using namespace mtl;
using Catch::Matchers::WithinAbs;

// -- Static assertions: expression types satisfy concepts ----------------

TEST_CASE("expression types satisfy Matrix concept", "[expr][concepts]") {
    dense2D<double> a = {{1.0, 2.0}, {3.0, 4.0}};
    dense2D<double> b = {{5.0, 6.0}, {7.0, 8.0}};

    auto sum_expr = a + b;
    STATIC_REQUIRE(Matrix<decltype(sum_expr)>);
    STATIC_REQUIRE(traits::is_expression_v<decltype(sum_expr)>);

    auto neg_expr = -a;
    STATIC_REQUIRE(Matrix<decltype(neg_expr)>);
    STATIC_REQUIRE(traits::is_expression_v<decltype(neg_expr)>);

    auto scal_expr = 2.0 * a;
    STATIC_REQUIRE(Matrix<decltype(scal_expr)>);
    STATIC_REQUIRE(traits::is_expression_v<decltype(scal_expr)>);

    auto div_expr = a / 2.0;
    STATIC_REQUIRE(Matrix<decltype(div_expr)>);
    STATIC_REQUIRE(traits::is_expression_v<decltype(div_expr)>);

    // Matrix multiply is eager (returns dense2D), not lazy
    auto mul_result = a * b;
    STATIC_REQUIRE(Matrix<decltype(mul_result)>);
    STATIC_REQUIRE_FALSE(traits::is_expression_v<decltype(mul_result)>);
}

TEST_CASE("expression types satisfy Vector concept", "[expr][concepts]") {
    dense_vector<double> u = {1.0, 2.0, 3.0};
    dense_vector<double> v = {4.0, 5.0, 6.0};

    auto sum_expr = u + v;
    STATIC_REQUIRE(Vector<decltype(sum_expr)>);
    STATIC_REQUIRE(traits::is_expression_v<decltype(sum_expr)>);

    auto neg_expr = -u;
    STATIC_REQUIRE(Vector<decltype(neg_expr)>);
    STATIC_REQUIRE(traits::is_expression_v<decltype(neg_expr)>);

    auto scal_expr = 3.0 * u;
    STATIC_REQUIRE(Vector<decltype(scal_expr)>);
    STATIC_REQUIRE(traits::is_expression_v<decltype(scal_expr)>);
}

TEST_CASE("concrete types are not expressions", "[expr][concepts]") {
    STATIC_REQUIRE_FALSE(traits::is_expression_v<dense2D<double>>);
    STATIC_REQUIRE_FALSE(traits::is_expression_v<dense_vector<double>>);
}

// -- Lazy evaluation: expressions reflect source changes -----------------

TEST_CASE("lazy evaluation reflects source changes", "[expr][lazy]") {
    dense2D<double> a = {{1.0, 2.0}, {3.0, 4.0}};
    dense2D<double> b = {{10.0, 20.0}, {30.0, 40.0}};
    auto expr = a + b;

    // Before modification
    REQUIRE(expr(0, 0) == 11.0);

    // Modify source after creating expression
    a(0, 0) = 100.0;
    REQUIRE(expr(0, 0) == 110.0);  // expression reflects the change
}

TEST_CASE("lazy vector evaluation reflects source changes", "[expr][lazy]") {
    dense_vector<double> u = {1.0, 2.0, 3.0};
    dense_vector<double> v = {10.0, 20.0, 30.0};
    auto expr = u + v;

    REQUIRE(expr(0) == 11.0);
    u(0) = 100.0;
    REQUIRE(expr(0) == 110.0);
}

// -- Nested expressions --------------------------------------------------

TEST_CASE("nested matrix expressions: (A + B) - C", "[expr][nested]") {
    dense2D<double> a = {{1.0, 2.0}, {3.0, 4.0}};
    dense2D<double> b = {{5.0, 6.0}, {7.0, 8.0}};
    dense2D<double> c = {{2.0, 3.0}, {4.0, 5.0}};

    dense2D<double> result = (a + b) - c;
    REQUIRE(result(0, 0) == 4.0);   // 1+5-2
    REQUIRE(result(0, 1) == 5.0);   // 2+6-3
    REQUIRE(result(1, 0) == 6.0);   // 3+7-4
    REQUIRE(result(1, 1) == 7.0);   // 4+8-5
}

TEST_CASE("nested matrix expressions: 2*A + B", "[expr][nested]") {
    dense2D<double> a = {{1.0, 2.0}, {3.0, 4.0}};
    dense2D<double> b = {{10.0, 20.0}, {30.0, 40.0}};

    dense2D<double> result = 2.0 * a + b;
    REQUIRE(result(0, 0) == 12.0);   // 2*1+10
    REQUIRE(result(0, 1) == 24.0);   // 2*2+20
    REQUIRE(result(1, 0) == 36.0);   // 2*3+30
    REQUIRE(result(1, 1) == 48.0);   // 2*4+40
}

TEST_CASE("nested vector expressions: 2*u + v", "[expr][nested]") {
    dense_vector<double> u = {1.0, 2.0, 3.0};
    dense_vector<double> v = {10.0, 20.0, 30.0};

    dense_vector<double> result = 2.0 * u + v;
    REQUIRE(result(0) == 12.0);
    REQUIRE(result(1) == 24.0);
    REQUIRE(result(2) == 36.0);
}

// -- Assignment triggers evaluation --------------------------------------

TEST_CASE("assignment to dense2D triggers evaluation", "[expr][assign]") {
    dense2D<double> a = {{1.0, 2.0}, {3.0, 4.0}};
    dense2D<double> b = {{5.0, 6.0}, {7.0, 8.0}};

    dense2D<double> c = a + b;
    REQUIRE(c(0, 0) == 6.0);
    REQUIRE(c(0, 1) == 8.0);
    REQUIRE(c(1, 0) == 10.0);
    REQUIRE(c(1, 1) == 12.0);

    // Modifying source after materialization has no effect
    a(0, 0) = 999.0;
    REQUIRE(c(0, 0) == 6.0);
}

TEST_CASE("assignment to dense_vector triggers evaluation", "[expr][assign]") {
    dense_vector<double> u = {1.0, 2.0, 3.0};
    dense_vector<double> v = {4.0, 5.0, 6.0};

    dense_vector<double> w = u + v;
    REQUIRE(w(0) == 5.0);
    REQUIRE(w(1) == 7.0);
    REQUIRE(w(2) == 9.0);
}

// -- Compound assignment from expressions --------------------------------

TEST_CASE("matrix += expression", "[expr][compound]") {
    dense2D<double> c = {{1.0, 2.0}, {3.0, 4.0}};
    dense2D<double> a = {{10.0, 20.0}, {30.0, 40.0}};
    dense2D<double> b = {{5.0, 6.0}, {7.0, 8.0}};

    c += a + b;
    REQUIRE(c(0, 0) == 16.0);   // 1 + (10+5)
    REQUIRE(c(0, 1) == 28.0);   // 2 + (20+6)
    REQUIRE(c(1, 0) == 40.0);   // 3 + (30+7)
    REQUIRE(c(1, 1) == 52.0);   // 4 + (40+8)
}

TEST_CASE("matrix -= expression", "[expr][compound]") {
    dense2D<double> c = {{100.0, 200.0}, {300.0, 400.0}};
    dense2D<double> a = {{1.0, 2.0}, {3.0, 4.0}};
    dense2D<double> b = {{5.0, 6.0}, {7.0, 8.0}};

    c -= a + b;
    REQUIRE(c(0, 0) == 94.0);
    REQUIRE(c(1, 1) == 388.0);
}

TEST_CASE("vector += expression", "[expr][compound]") {
    dense_vector<double> w = {1.0, 2.0, 3.0};
    dense_vector<double> u = {10.0, 20.0, 30.0};
    dense_vector<double> v = {5.0, 6.0, 7.0};

    w += u + v;
    REQUIRE(w(0) == 16.0);
    REQUIRE(w(1) == 28.0);
    REQUIRE(w(2) == 40.0);
}

TEST_CASE("vector -= expression", "[expr][compound]") {
    dense_vector<double> w = {100.0, 200.0, 300.0};
    dense_vector<double> u = {1.0, 2.0, 3.0};
    dense_vector<double> v = {4.0, 5.0, 6.0};

    w -= u + v;
    REQUIRE(w(0) == 95.0);
    REQUIRE(w(1) == 193.0);
    REQUIRE(w(2) == 291.0);
}

// -- Matrix multiply (eager) ---------------------------------------------

TEST_CASE("matrix multiply expression", "[expr][matmul]") {
    dense2D<double> A = {{1.0, 2.0}, {3.0, 4.0}};
    dense2D<double> B = {{5.0, 6.0}, {7.0, 8.0}};

    dense2D<double> C = A * B;
    REQUIRE(C(0, 0) == 19.0);
    REQUIRE(C(0, 1) == 22.0);
    REQUIRE(C(1, 0) == 43.0);
    REQUIRE(C(1, 1) == 50.0);
}

TEST_CASE("non-square matrix multiply expression", "[expr][matmul]") {
    dense2D<double> A = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    dense2D<double> B = {{1.0, 0.0}, {0.0, 1.0}, {1.0, 1.0}};

    dense2D<double> C = A * B;
    REQUIRE(C.num_rows() == 2);
    REQUIRE(C.num_cols() == 2);
    REQUIRE(C(0, 0) == 4.0);
    REQUIRE(C(0, 1) == 5.0);
    REQUIRE(C(1, 0) == 10.0);
    REQUIRE(C(1, 1) == 11.0);
}

// -- Vector expressions --------------------------------------------------

TEST_CASE("vector expression: 2*u + v", "[expr][vec]") {
    dense_vector<double> u = {1.0, 2.0, 3.0};
    dense_vector<double> v = {10.0, 20.0, 30.0};

    dense_vector<double> w = 2.0 * u + v;
    REQUIRE(w(0) == 12.0);
    REQUIRE(w(1) == 24.0);
    REQUIRE(w(2) == 36.0);
}

TEST_CASE("vector negation expression", "[expr][vec]") {
    dense_vector<double> u = {1.0, -2.0, 3.0};
    dense_vector<double> w = -u;
    REQUIRE(w(0) == -1.0);
    REQUIRE(w(1) == 2.0);
    REQUIRE(w(2) == -3.0);
}

TEST_CASE("vector division expression", "[expr][vec]") {
    dense_vector<double> u = {4.0, 6.0, 8.0};
    dense_vector<double> w = u / 2.0;
    REQUIRE(w(0) == 2.0);
    REQUIRE(w(1) == 3.0);
    REQUIRE(w(2) == 4.0);
}

// -- Backward compatibility: auto c = a + b; c(0,0) works ---------------

TEST_CASE("auto captures expression with read-only access", "[expr][compat]") {
    dense2D<double> a = {{1.0, 2.0}, {3.0, 4.0}};
    dense2D<double> b = {{5.0, 6.0}, {7.0, 8.0}};

    auto c = a + b;  // c is an expression, not a dense2D
    REQUIRE(c(0, 0) == 6.0);
    REQUIRE(c(0, 1) == 8.0);
    REQUIRE(c(1, 0) == 10.0);
    REQUIRE(c(1, 1) == 12.0);
}

TEST_CASE("auto captures vector expression with read-only access", "[expr][compat]") {
    dense_vector<double> u = {1.0, 2.0, 3.0};
    dense_vector<double> v = {4.0, 5.0, 6.0};

    auto w = u + v;
    REQUIRE(w(0) == 5.0);
    REQUIRE(w(1) == 7.0);
    REQUIRE(w(2) == 9.0);
}

// -- evaluate() materializes to concrete type ----------------------------

TEST_CASE("evaluate materializes matrix expression", "[expr][evaluate]") {
    dense2D<double> a = {{1.0, 2.0}, {3.0, 4.0}};
    dense2D<double> b = {{5.0, 6.0}, {7.0, 8.0}};

    auto expr = a + b;
    auto concrete = evaluate(expr);
    STATIC_REQUIRE(std::is_same_v<decltype(concrete), dense2D<double>>);
    REQUIRE(concrete(0, 0) == 6.0);
    REQUIRE(concrete(1, 1) == 12.0);

    // After materialization, source changes don't affect concrete
    a(0, 0) = 999.0;
    REQUIRE(concrete(0, 0) == 6.0);
}

TEST_CASE("evaluate materializes vector expression", "[expr][evaluate]") {
    dense_vector<double> u = {1.0, 2.0, 3.0};
    dense_vector<double> v = {4.0, 5.0, 6.0};

    auto expr = u + v;
    auto concrete = evaluate(expr);
    STATIC_REQUIRE(std::is_same_v<decltype(concrete), dense_vector<double>>);
    REQUIRE(concrete(0) == 5.0);
    REQUIRE(concrete(2) == 9.0);
}

TEST_CASE("evaluate passes through concrete types", "[expr][evaluate]") {
    dense2D<double> a = {{1.0, 2.0}, {3.0, 4.0}};
    const auto& ref = evaluate(a);
    REQUIRE(&ref == &a);  // no copy, returns reference
}

// -- y = A*x + b pattern (eager matvec + lazy vec add) -------------------

TEST_CASE("y = A*x + b pattern", "[expr][pattern]") {
    dense2D<double> A = {{1.0, 0.0}, {0.0, 1.0}};
    dense_vector<double> x = {3.0, 4.0};
    dense_vector<double> b = {1.0, 2.0};

    // A*x is eager (returns dense_vector), then + b is lazy expression
    dense_vector<double> y = A * x + b;
    REQUIRE(y(0) == 4.0);
    REQUIRE(y(1) == 6.0);
}

TEST_CASE("y = A*x + b with non-identity matrix", "[expr][pattern]") {
    dense2D<double> A = {{2.0, 1.0}, {1.0, 3.0}};
    dense_vector<double> x = {1.0, 2.0};
    dense_vector<double> b = {10.0, 20.0};

    dense_vector<double> y = A * x + b;
    REQUIRE(y(0) == 14.0);   // 2*1+1*2 + 10
    REQUIRE(y(1) == 27.0);   // 1*1+3*2 + 20
}

// -- fused_assign and fused_plus_assign ----------------------------------

TEST_CASE("fused_assign evaluates expression into target", "[expr][fuse]") {
    dense2D<double> a = {{1.0, 2.0}, {3.0, 4.0}};
    dense2D<double> b = {{5.0, 6.0}, {7.0, 8.0}};
    dense2D<double> c;

    fused_assign(c, a + b);
    REQUIRE(c(0, 0) == 6.0);
    REQUIRE(c(1, 1) == 12.0);
}

TEST_CASE("fused_plus_assign adds expression to target", "[expr][fuse]") {
    dense2D<double> c = {{1.0, 2.0}, {3.0, 4.0}};
    dense2D<double> a = {{10.0, 20.0}, {30.0, 40.0}};
    dense2D<double> b = {{5.0, 6.0}, {7.0, 8.0}};

    fused_plus_assign(c, a + b);
    REQUIRE(c(0, 0) == 16.0);
    REQUIRE(c(1, 1) == 52.0);
}

// -- Operator= from expression -------------------------------------------

TEST_CASE("operator= from matrix expression", "[expr][assign]") {
    dense2D<double> a = {{1.0, 2.0}, {3.0, 4.0}};
    dense2D<double> b = {{5.0, 6.0}, {7.0, 8.0}};

    dense2D<double> c;
    c = a + b;
    REQUIRE(c.num_rows() == 2);
    REQUIRE(c.num_cols() == 2);
    REQUIRE(c(0, 0) == 6.0);
    REQUIRE(c(1, 1) == 12.0);
}

TEST_CASE("operator= from vector expression", "[expr][assign]") {
    dense_vector<double> u = {1.0, 2.0, 3.0};
    dense_vector<double> v = {4.0, 5.0, 6.0};

    dense_vector<double> w;
    w = u + v;
    REQUIRE(w.size() == 3);
    REQUIRE(w(0) == 5.0);
    REQUIRE(w(2) == 9.0);
}

// -- Expression with scalar on right -------------------------------------

TEST_CASE("matrix * scalar on right returns expression", "[expr][scal]") {
    dense2D<double> m = {{1.0, 2.0}, {3.0, 4.0}};
    dense2D<double> r = m * 3.0;
    REQUIRE(r(0, 0) == 3.0);
    REQUIRE(r(1, 1) == 12.0);
}

TEST_CASE("vector * scalar on right returns expression", "[expr][scal]") {
    dense_vector<double> v = {2.0, 4.0, 6.0};
    dense_vector<double> r = v * 0.5;
    REQUIRE(r(0) == 1.0);
    REQUIRE(r(1) == 2.0);
    REQUIRE(r(2) == 3.0);
}
