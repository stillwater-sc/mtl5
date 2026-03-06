# Phase 8 Examples: Expression Template Demonstrations

## Context

Phase 8 (expression templates) is fully implemented and passing 32/32 tests. The user wants two pedagogical examples following the existing naming convention (`phase8a_*`, `phase8b_*`). One must demonstrate the **cache-efficiency performance advantage** of expression templates at L3 cache boundary sizes. The other should teach the **concepts and mechanics** of expression templates.

---

## Example A: `phase8a_expression_benchmark.cpp` — Cache-Efficient Expression Templates

**Goal**: Demonstrate measurable speedup from expression templates by choosing matrix sizes that span the L3 cache boundary.

**Key insight**: `C = 2.0*A + 3.0*B` involves 3 N×N matrices (A, B, C). With expression templates, only these 3 matrices need to be in cache during the single fused pass. With eager evaluation, 2 extra temporaries are created (one for `2.0*A`, one for `3.0*B`), requiring 5 matrices — potentially exceeding L3 capacity.

**Matrix sizes**: Sweep from N=200 to N=1500 in steps of ~100. For a typical 8-12 MB L3 cache:
- 3 matrices of N=700 doubles = 3 × 700² × 8 ≈ 11.2 MB → fits in L3 (lazy)
- 5 matrices of N=700 doubles = 5 × 700² × 8 ≈ 18.7 MB → exceeds L3 (eager)

This creates a clear performance crossover point.

**Structure**:
1. Header: explain what expression templates are and the cache hypothesis
2. For each matrix size N:
   - **Lazy path** (expression templates): `dense2D<double> C = 2.0 * A + 3.0 * B;` — compiler sees through expression tree, single fused loop, 3 matrices total
   - **Eager path** (forced temporaries): `auto t1 = evaluate(2.0 * A); auto t2 = evaluate(3.0 * B); dense2D<double> C = t1 + t2; evaluate()` forces materialization — 5 matrices total
   - Time both, report ratio
3. Summary table: N, lazy time, eager time, speedup, memory footprint (lazy vs eager)
4. Conclude with the cache-boundary explanation

**Timing approach**: `std::chrono::steady_clock`, multiple iterations per size, `volatile double sink` to prevent dead-code elimination. Warm-up pass before timing.

**Files used**:
- `include/mtl/mtl.hpp` (main include)
- `include/mtl/operation/lazy.hpp` — `evaluate()` to force eager materialization
- `include/mtl/mat/operators.hpp` — lazy operators
- `include/mtl/mat/dense2D.hpp` — expression constructor

---

## Example B: `phase8b_expression_concepts.cpp` — Expression Template Mechanics

**Goal**: A guided tour through expression template concepts with clear explanations and small matrices.

**Structure** (9 parts):

1. **Lazy capture**: `auto expr = A + B;` — show that modifying A changes the expression result (proves laziness)
2. **Type inspection**: Print `typeid` of expressions vs concrete types, check `is_expression_v`
3. **Nested expressions**: `auto expr = 2.0 * A + B - C;` — expression tree composition
4. **Three ways to materialize**:
   - Assignment: `dense2D<double> result = expr;`
   - `evaluate()`: `auto result = evaluate(expr);`
   - `fused_assign()`: `fused_assign(result, expr);`
5. **Fused evaluation**: `fused_plus_assign(C, 2.0 * A + B);` — accumulate without temporary
6. **Compound assignment**: `C += A + B;` and `C -= A;`
7. **Vector expressions**: `dense_vector<double> w = 2.0 * u + v;`
8. **Mixed eager/lazy**: `y = A*x + b` — matvec is eager, vector add is lazy
9. **Unary operations**: negation `-A`, scalar division `A / 2.0`

Each part includes:
- Code doing the operation
- Printed explanation of what happens under the hood
- Verification that results are correct

**Files used**:
- `include/mtl/mtl.hpp`
- `include/mtl/traits/is_expression.hpp` — `is_expression_v` trait
- `include/mtl/operation/lazy.hpp` — `evaluate()`
- `include/mtl/operation/fuse.hpp` — `fused_assign()`, `fused_plus_assign()`

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `examples/phase8a_expression_benchmark.cpp` | **Create** |
| `examples/phase8b_expression_concepts.cpp` | **Create** |
| `examples/CMakeLists.txt` | **Modify** — add two targets |

---

## Verification

```bash
cd /home/stillwater/dev/stillwater/clones/mtl5
cmake -B build
cmake --build build -j$(nproc)

# Run all existing tests (must still pass — 32/32)
ctest --test-dir build

# Run the new examples
./build/examples/phase8a_expression_benchmark
./build/examples/phase8b_expression_concepts
```
