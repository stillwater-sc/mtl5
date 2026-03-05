# Phase 8: Expression Templates for MTL5

## Context

MTL5 phases 1-7 are complete (31/31 tests passing). All operators currently use **eager evaluation** — every `A + B`, `s * A`, `A * B` immediately materializes a concrete `dense2D` or `dense_vector`. Phase 8 introduces **expression templates** to eliminate unnecessary temporaries. For example, `C = 2.0 * A + B` currently creates 2 temporaries; with expression templates it creates zero — a single fused loop evaluates directly into `C`.

**Design choice**: C++20 concepts replace MTL4's CRTP hierarchy. Expression types simply satisfy existing `Matrix`/`Vector` concepts. No inheritance hierarchy needed.

---

## Implementation Steps

### Step 1: Create `traits/is_expression.hpp` (new file)

**Path**: `include/mtl/traits/is_expression.hpp`

Simple trait to disambiguate expression types from concrete types in assignment overloads:
```cpp
template <typename T> struct is_expression : std::false_type {};
template <typename T> inline constexpr bool is_expression_v = is_expression<T>::value;
```

### Step 2: Create matrix expression classes (new files)

All in `include/mtl/mat/expr/`. Each class satisfies `Matrix` concept by providing `value_type`, `size_type`, `num_rows()`, `num_cols()`, `size()`, `operator()(r,c)`.

| File | Class | Stores | `operator()(r,c)` returns |
|------|-------|--------|---------------------------|
| `mat_mat_op_expr.hpp` | `mat_mat_op_expr<E1,E2,SF>` | `const E1&`, `const E2&` | `SF::apply(e1(r,c), e2(r,c))` |
| `mat_scal_op_expr.hpp` | `mat_scal_op_expr<S,M,SF>` + `mat_rscal_op_expr<M,S,SF>` | scalar by value, matrix by ref | `SF::apply(s, m(r,c))` |
| `mat_negate_expr.hpp` | `mat_negate_expr<E>` | `const E&` | `-e(r,c)` |

Reuse existing functors: `functor::scalar::plus`, `minus`, `times`, `divide` (in `include/mtl/functor/scalar/`).

Each expression type gets `is_expression`, `category` (→ `tag::dense`), and `ashape` (→ `mat<V>`) trait specializations.

### Step 3: Populate `mat_mat_times_expr.hpp` (existing stub)

**Path**: `include/mtl/mat/expr/mat_mat_times_expr.hpp`

Stores `const E1&` and `const E2&`. `operator()(r,c)` computes inner product (O(K) per element). This is the same total complexity as eager when assigned element-by-element. For performance-critical code, users should continue using `mult(A, B, C)`.

### Step 4: Create vector expression classes (new files)

All in `include/mtl/vec/expr/`. Same pattern as matrix, but satisfy `Vector` concept (`operator()(i)`, `size()`).

| File | Class | Description |
|------|-------|-------------|
| `vec_vec_op_expr.hpp` | `vec_vec_op_expr<E1,E2,SF>` | Binary element-wise |
| `vec_scal_op_expr.hpp` | `vec_scal_op_expr<S,V,SF>` + `vec_rscal_op_expr<V,S,SF>` | Scalar-vector ops |
| `vec_negate_expr.hpp` | `vec_negate_expr<E>` | Unary negation |

### Step 5: Replace `mat/operators.hpp`

**Path**: `include/mtl/mat/operators.hpp`

Change element-wise operators to return expression types:
- `operator+(M1, M2)` → `mat_mat_op_expr<M1, M2, plus>`
- `operator-(M1, M2)` → `mat_mat_op_expr<M1, M2, minus>`
- `operator-(M)` → `mat_negate_expr<M>`
- `s * M`, `M * s` → `mat_scal_op_expr<S, M, times>`
- `M / s` → `mat_rscal_op_expr<M, S, divide>`
- `M1 * M2` (matmul) → `mat_mat_times_expr<M1, M2>`

**Keep eager**:
- `Matrix * Vector` → concrete `dense_vector` (stays eager)
- `compressed2D * dense_vector` specialization (stays eager, optimized CRS)
- `transposed_view<compressed2D> * dense_vector` specialization (stays eager)

### Step 6: Replace `vec/operators.hpp`

**Path**: `include/mtl/vec/operators.hpp`

Same pattern: return vector expression types for `+`, `-`, `-v`, `s*v`, `v/s`.

### Step 7: Add expression assignment to `dense2D`

**Path**: `include/mtl/mat/dense2D.hpp`

Add after existing copy/move (lines 122-139):
```cpp
// Expression template constructor
template <typename Expr>
    requires (Matrix<Expr> && traits::is_expression_v<Expr>
              && std::convertible_to<typename Expr::value_type, Value>)
dense2D(const Expr& expr) : dense2D(expr.num_rows(), expr.num_cols()) {
    for (size_type r = 0; r < num_rows(); ++r)
        for (size_type c = 0; c < num_cols(); ++c)
            (*this)(r, c) = static_cast<Value>(expr(r, c));
}

// Expression assignment
template <typename Expr>
    requires (Matrix<Expr> && traits::is_expression_v<Expr>)
dense2D& operator=(const Expr& expr) { /* change_dim + eval loop */ }

// Compound from expressions
template <typename Expr> requires (...) dense2D& operator+=(const Expr&);
template <typename Expr> requires (...) dense2D& operator-=(const Expr&);
```

The `is_expression_v` constraint prevents ambiguity with the defaulted `dense2D(const dense2D&)` and `operator=(const dense2D&)`.

### Step 8: Add expression assignment to `dense_vector`

**Path**: `include/mtl/vec/dense_vector.hpp`

Same pattern: add expression constructor, `operator=`, `operator+=`, `operator-=`.

### Step 9: Implement `operation/lazy.hpp` (existing stub)

**Path**: `include/mtl/operation/lazy.hpp`

Provides `evaluate(expr)` — forces materialization of an expression into a concrete type:
- Matrix expression → `dense2D<value_type>`
- Vector expression → `dense_vector<value_type>`
- Concrete type → pass-through (returns const ref)

### Step 10: Implement `operation/fuse.hpp` (existing stub)

**Path**: `include/mtl/operation/fuse.hpp`

Provides `fused_assign(dest, expr)` and `fused_plus_assign(dest, expr)` for explicit evaluation into pre-allocated targets.

### Step 11: Update `mat/expr/mat_expr.hpp`, `dmat_expr.hpp`, `smat_expr.hpp`, `vec/expr/vec_expr.hpp` (existing stubs)

Replace empty stubs with documentation headers explaining that C++20 concepts replaced the CRTP hierarchy. Include the new expression headers from them for backward compatibility.

### Step 12: Update `mtl.hpp`

Add includes for new expression headers, `lazy.hpp`, `fuse.hpp`.

### Step 13: Write tests

**Path**: `tests/unit/operation/test_expression_templates.cpp`

Test cases:
- Static assertions: expression types satisfy `Matrix`/`Vector` concepts, `is_expression_v` is true
- Lazy evaluation: modify source after creating expression, verify expression reflects change
- Nested expressions: `(A + B) - C`, `2.0 * A + B`
- Assignment to concrete types triggers evaluation: `dense2D<double> C = A + B;`
- Compound assignment: `C += A + B;`
- Matrix multiply expression: `dense2D<double> C = A * B;`
- Vector expressions: `dense_vector<double> w = 2.0 * u + v;`
- Backward compat: `auto c = a + b; c(0,0)` — works (read-only access on expression)
- `evaluate()` materializes to concrete type
- `y = A*x + b` pattern (eager matvec + lazy vec add)

---

## Backward Compatibility

**Safe**: All existing test patterns use `auto c = a + b; REQUIRE(c(i,j) == ...);` — read-only access through `operator()`, which expression types provide.

**Breaking**: `auto c = a + b; c(0,0) = 5.0;` would fail (expression returns by value). No existing tests do this.

**ADL works**: Expression types in `mtl::mat::expr` — ADL searches enclosing `mtl::mat` where operators live. Nested expressions like `(a+b)+c` resolve correctly.

---

## Key Files

| File | Action |
|------|--------|
| `include/mtl/traits/is_expression.hpp` | **Create** |
| `include/mtl/mat/expr/mat_mat_op_expr.hpp` | **Create** |
| `include/mtl/mat/expr/mat_scal_op_expr.hpp` | **Create** |
| `include/mtl/mat/expr/mat_negate_expr.hpp` | **Create** |
| `include/mtl/mat/expr/mat_mat_times_expr.hpp` | **Replace stub** |
| `include/mtl/vec/expr/vec_vec_op_expr.hpp` | **Create** |
| `include/mtl/vec/expr/vec_scal_op_expr.hpp` | **Create** |
| `include/mtl/vec/expr/vec_negate_expr.hpp` | **Create** |
| `include/mtl/mat/expr/mat_expr.hpp` | **Replace stub** (documentation) |
| `include/mtl/mat/expr/dmat_expr.hpp` | **Replace stub** (includes) |
| `include/mtl/mat/expr/smat_expr.hpp` | **Replace stub** (includes) |
| `include/mtl/vec/expr/vec_expr.hpp` | **Replace stub** (includes) |
| `include/mtl/mat/operators.hpp` | **Rewrite** (expr returns) |
| `include/mtl/vec/operators.hpp` | **Rewrite** (expr returns) |
| `include/mtl/mat/dense2D.hpp` | **Modify** (add expr ctor/assign) |
| `include/mtl/vec/dense_vector.hpp` | **Modify** (add expr ctor/assign) |
| `include/mtl/operation/lazy.hpp` | **Replace stub** |
| `include/mtl/operation/fuse.hpp` | **Replace stub** |
| `include/mtl/mtl.hpp` | **Modify** (add includes) |
| `tests/unit/operation/test_expression_templates.cpp` | **Create** |

---

## Verification

```bash
cd /home/stillwater/dev/stillwater/clones/mtl5
cmake -B build
cmake --build build -j$(nproc)
ctest --test-dir build               # All 31 existing tests must pass
ctest --test-dir build -R expression  # New expression template tests must pass
```
