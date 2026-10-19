# Exact Accumulator

`mtl::math::exact_accumulator<T>` sums products of IEEE `float` or `double` values **without rounding**, then rounds the total once. It is a Kulisch accumulator: a long fixed-point integer that covers every product two finite `T` can form, with 64 spare bits for carries. It gives MTL5 a correctly rounded dot product, and an exact reference to measure every other accumulation strategy against, without depending on an external number library.

Header: `include/mtl/math/exact_accumulator.hpp` (also pulled in by `mtl/mtl.hpp`).

```cpp
#include <mtl/mtl.hpp>

using exact = mtl::math::exact_accumulator<double>;

mtl::vec::dense_vector<double> x(4), y(4, 1.0);
x(0) = 1e16; x(1) = 1.0; x(2) = -1e16; x(3) = 0.5;

double d = mtl::dot<exact, double>(x, y);   // 1.5: x . y, correctly rounded
```

## Where it fits: three accumulation configurations

Every MTL5 operation that forms a sum of products takes an `Accumulator` template parameter and reaches it only through `math::accumulator_traits` (`include/mtl/math/accumulator_traits.hpp`). There are three configurations:

| Configuration | Accumulator | Rounding events per term | Result |
|---|---|---|---|
| 1. Plain | `T`, or a wider built-in such as `double` over `float` | 2: the product, then the sum | Error grows with the length and condition of the sum |
| 2. Fused | `math::fma_accumulator<T>` | 1: `fma(m, v, acc)` | The product rounding is gone; the sum still rounds every term |
| 2c. Compensated | `math::compensated_accumulator<T>` | 0 lost: each rounding error is kept in a second `T` | As if in twice the precision of `T`, then rounded once ([Error-Free Transformations](error-free-transformations.md#the-compensated-accumulator-sum2--dot2)) |
| 3. Exact | `math::exact_accumulator<T>`, or a posit quire from a number library | **0**, then one at the end | The correctly rounded value of the exact sum |

So the precision of a computation splits into three independent choices:

- the **input** type, which is what is stored;
- the **accumulator**, where each reduction lives;
- the **output** type, which is what the accumulator is rounded to (`accumulator_value(acc, std::type_identity<R>{})`).

An exact accumulator makes the middle choice drop out of the error analysis: what is left is the rounding of the inputs and one rounding on output.

The exact accumulator plugs in through the **ADL hook**: the free functions below live next to the type, in `mtl::math`, and `accumulator_traits` finds them by argument-dependent lookup. That is the same route a number library uses for its own quire, so the exact accumulator is also the worked example of that protocol.

| Function | Effect |
|---|---|
| `accumulator_clear(acc)` | `acc = 0` |
| `accumulator_assign(acc, v)` | `acc = v`, exactly |
| `accumulator_add_product(acc, m, v)` | `acc += m * v`, exactly |
| `accumulator_value(acc, std::type_identity<R>{})` | the sum rounded once to `R` (`float` or `double`) |
| `accumulator_merge(acc, other)` | `acc += other`, exactly |

## How it works

### Layout

```cpp
template <typename T>            // float or double
struct exact_accumulator {
    std::array<std::uint64_t, limbs> w;   // the sum: two's complement, little-endian limbs
    std::uint8_t special;                 // +inf, -inf and NaN seen (flags)
};
```

Bit 0 of `w` has weight `2^lsb`, where `lsb` is the exponent of the smallest product of two subnormals. The top of the array is set by the largest product of two finite values, plus 64 carry bits:

| `T` | `lsb` | bits | limbs | size |
|---|---|---|---|---|
| `float` | −298 | 618 | 10 | 80 bytes |
| `double` | −2148 | 4260 | 67 | 536 bytes |

The 64 carry bits mean that up to 2^64 maximal products can be added before the integer could wrap.

The struct is an aggregate and trivially default constructible, so `exact_accumulator<double> acc{}` is zero, and the kernels can hold vectors of them like any value type.

### Adding a product

`accumulator_add_product(acc, m, v)`:

1. **Decomposes** each operand into an odd integer significand and a binary exponent, `|m| = M · 2^e`, exactly (`frexp` / `ldexp`).
2. **Multiplies** the two 53-bit (or 24-bit) significands into a 128-bit integer, in portable 64-bit arithmetic: no `__int128`, no `long double`.
3. **Adds or subtracts** that 128-bit value at bit position `e_m + e_v − lsb`. It spans at most three limbs; the carry or borrow propagates past them only when it has to.

No step rounds. A product too large for a `double`, such as `1e300 · 1e300`, is held exactly and can be cancelled by a later term.

### Rounding out

`accumulator_value<R>` makes the only rounding of the whole computation:

- It negates a negative two's-complement sum to get the magnitude.
- It finds the leading one bit.
- It keeps as many bits as `R` has at that magnitude: `digits`, or fewer once the result is subnormal.
- It rounds to nearest, ties to even, using the round bit and a sticky bit over everything below it.
- It overflows to a signed infinity when the sum is beyond `R`'s range, and underflows to zero below half the smallest subnormal.

Because the sum is exact, `R` can be narrower than the accumulator's range type without double rounding. A `float` result read from an `exact_accumulator<double>` is the correctly rounded `float`, not the `float` nearest to a rounded `double`.

An exactly zero sum rounds to **+0**, whatever the signs of the terms that cancelled.

### Non-finite values

The finite sum and the special state are kept apart, and the result follows IEEE arithmetic:

| Terms seen | Result |
|---|---|
| a NaN operand, or `inf · 0` | NaN |
| infinite products of one sign | that infinity |
| infinite products of both signs | NaN |
| finite only | the rounded exact sum |

`accumulator_clear` and `accumulator_assign` reset the special state; `accumulator_merge` combines it.

### Merging and determinism

`accumulator_merge` adds two finite parts limb by limb with carry, which is exact. So a reduction split into partial accumulators gives the **same bits for every partition**: any number of threads, any chunking, any order. The threaded reductions in `dot` and the norms (#582) split only when the accumulator can merge; with this one they can, and the parallel result is the sequential result.

That is a property no rounding accumulator has. With configurations 1 and 2, the partition is part of the answer.

## How the rest of MTL5 treats it

- **Every accumulator-aware operation accepts it**: `dot`, `gemm`, the norms, `sum` and `trace`, the dense and sparse factorizations and their solves, the incomplete factorizations, `spgemm`, `svd`, the Krylov solvers and eigensolvers, multigrid (the #585 programme).
- **BLAS is skipped.** A non-void accumulator routes an operation to the native kernels (`interface::accumulator_allows_blas_v`), because a BLAS library accumulates in its own precision.
- **Three-factor terms are fed whole.** The LDLᵀ factorizations sum `L(i,k)·d_k·L(j,k)`. For an exact accumulator over an IEEE type, `math::splits_product_terms_v` holds, and the kernels split the product they form into `hi + lo` (`math::two_product`) so that no part of the term is rounded before it arrives (#602).
- **Workspace.** A kernel that keeps one accumulator per output entry, such as `sparse_lu` or the supernodal panels, holds Θ(n) of them, and an `exact_accumulator<double>` is 67 times the size of a `double`. The workspace note in `accumulator_traits.hpp` gives the measured scaling.

## Accuracy and cost

**Accuracy.** Over the accumulation matrix of #586, the exact accumulator is the configuration whose error does not grow with the condition number of the sum. Examples, all in float:

- **dot:** on ill-conditioned dot products (the Ogita–Rump–Oishi GenDot construction), every result is the correctly rounded value.
- **`spgemm`:** exact is correctly rounded at every entry; plain misses 584 of 1589 entries.
- **`sum`:** exact is correctly rounded on all 20 cancelling test sums; plain on none.
- **Factorizations:** where the algorithm keeps each entry's whole reduction in one accumulator (the left-looking and up-looking forms), the residual on Hilbert matrices drops by 2–8× for IC(0) and ILU(0), and by ≥ 4.3× for dense LU.

The measurements and their assertions are in the `[586:*]` tests.

**Cost.** About **16 ns per term** for `double` on an Intel i7-12700K, against 0.4 ns for a plain loop: roughly 35–40× (serial loop, `-O3`). Most of the time goes into decomposing the operands and placing the product, not the carry. The exact accumulator is a reference and an accuracy tool, not a fast path.

## Limits

- **`float` and `double` only** (`math::exact_accumulable`: IEEE binary, at most 53 significand bits). `long double` is excluded: it is 64, 80 or 128 bits depending on the platform, and its significand does not fit the 64-bit decomposition. A `float` may be added into an `exact_accumulator<double>`; a `double` into an `exact_accumulator<float>` is rejected at compile time.
- **The result type** `R` is `float` or `double`.
- **Size and speed** as above; choose it where accuracy, a reference value, or reproducibility across thread counts is the point.
- **Not a quire for other number systems.** A posit or LNS library supplies its own accumulator through the same ADL hook; MTL5 stays free of those libraries.

## Testing

- `tests/unit/math/test_exact_accumulator.cpp` covers the protocol the type satisfies, a single product against the IEEE product (20,000 random pairs), exact cancellation, intermediates beyond the `double` range, ties to even, rounding a `double`-range sum to `float` without double rounding, subnormals, overflow, non-finite inputs, an exact merge under every partition, and a dot product through `mtl::dot`.
- `tests/unit/operation/exact_accumulator.hpp` names it `exact_test::exact_acc` for the `[586:*]` measurement tests, and adds `err`, which measures a computed result's error exactly.

## Related

- [#586](https://github.com/stillwater-sc/mtl5/issues/586): the accumulation matrix, configurations 1–3 across MTL5's operations
- [The Accuracy Architecture](accuracy-architecture.md): which algorithm forms can use an exact accumulator (#596)
- [Accumulation Configurations](../algorithms/accumulation-configurations.md): what it buys across the library, measured (#592)
- [Error-Free Transformations](error-free-transformations.md): `two_sum`, `split`, `two_product` and the rest, checked exact against this accumulator
- [#602](https://github.com/stillwater-sc/mtl5/issues/602): three-factor terms and `math::two_product`
- [#611](https://github.com/stillwater-sc/mtl5/issues/611): reproducing Ogita, Rump and Oishi, "Accurate sum and dot product" (2005), with this accumulator as the reference
- U. Kulisch, *Computer Arithmetic and Validity: Theory, Implementation, and Applications*, de Gruyter, 2nd ed. 2013
