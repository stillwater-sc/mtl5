# Error-Free Transformations

An error-free transformation turns one rounded floating-point operation into a pair `(hi, lo)`: `hi` is the rounded result and `hi + lo` is the **exact** result. The rounding error of a sum or product of two floating-point numbers is itself a floating-point number, so nothing is lost. The program just has to keep `lo` instead of discarding it.

Header: `include/mtl/math/error_free.hpp` (also pulled in by `mtl/mtl.hpp` and by `math/accumulator_traits.hpp`).

```cpp
#include <mtl/math/error_free.hpp>

using namespace mtl::math;

auto [s, e] = two_sum(1e16, 1.0);        // s = 1e16, e = 1: the 1 that 1e16 + 1 drops
auto [p, f] = two_product(0.1, 0.1);     // p = fl(0.1 * 0.1), p + f = 0.1 * 0.1 exactly
```

## The transformations

| Function | Identity | Cost | Precondition | Due to |
|---|---|---|---|---|
| `two_sum(a, b)` | `a + b == hi + lo` | 6 flops | none | Knuth |
| `fast_two_sum(a, b)` | `a + b == hi + lo` | 3 flops | `\|a\| >= \|b\|` | Dekker |
| `split(a)` | `a == hi + lo`, each half a significand | 4 flops | `\|a\| <= 2^(max_exponent − s − 1)` | Veltkamp |
| `two_product_dekker(a, b)` | `a * b == hi + lo` | 17 flops | none beyond the product's own range | Dekker |
| `two_product(a, b)` | `a * b == hi + lo` | 2 flops, one an FMA | none beyond the product's own range | (FMA) |

Each returns a `math::exact_pair<T>{hi, lo}`, with `hi` bitwise the result of the plain operation. `split_product<T>`, the name `two_product` used to return (#602), is an alias for it. The functions accept any binary IEEE type, `math::binary_floating_point<T>`. `two_product` keeps its original constraint, any `std::is_floating_point_v<T>`.

These are the building blocks of Ogita, Rump and Oishi, "Accurate sum and dot product", *SIAM J. Sci. Comput.* 26(6), 2005, which MTL5 is reproducing under [#611](https://github.com/stillwater-sc/mtl5/issues/611):

- **Sum2 and Dot2** run a loop of `two_sum` (and `two_product`), carrying the `lo` parts in a second, ordinary accumulator. The result is as accurate as if the sum were computed in twice the working precision and then rounded.
- **SumK and DotK** apply the transformation K − 1 times, for K-fold precision.

### `two_sum` and `fast_two_sum`

```cpp
x = a + b;   z = x - a;   y = (a - (x - z)) + (b - z);     // two_sum
x = a + b;   y = (a - x) + b;                               // fast_two_sum
```

`fast_two_sum` is half the cost but needs the larger operand first. More precisely, it needs exponent(a) ≥ exponent(b). The function does not check this: ordering the operands is the caller's job, and checking would remove the saving. With the operands the wrong way round the error can come out wrong. For `p` the significand width, `1 + (2^p + 2)` rounds (a tie, to even) to `2^p + 4`, so its error is −1, but `fast_two_sum(1, 2^p + 2)` returns −2.

The result `lo` is always below half an ulp of `hi`, so `fl(hi + lo) == hi`.

### `split`

For a significand of `p` bits, let `s = ⌈p/2⌉`; then `c = (2^s + 1)·a`, `hi = c − (c − a)`, `lo = a − hi`. Then `a == hi + lo` exactly, with `hi` at most `p − s` significant bits and `lo` at most `s − 1`:

| `T` | `p` | `s` | bits in `hi` | bits in `lo` | domain |
|---|---|---|---|---|---|
| `float` | 24 | 12 | ≤ 12 | ≤ 11 | `\|a\| ≤ 2^115` |
| `double` | 53 | 27 | ≤ 26 | ≤ 26 | `\|a\| ≤ 2^996` |

Every product of two halves therefore fits in `p` bits, which is what Dekker's product needs.

The domain bound, 2^(max_exponent − s − 1), keeps `c` from overflowing. The bound is a power of two because `max / (2^s + 1)` rounds up in double, and `c` overflows at that value. Just above the bound the result is not guaranteed; far enough above it (beyond about `max / (2^s + 1)`) `c` overflows and `hi` is NaN, as in the paper. Splitting a scaled-down copy does not rescue the top of the range: there `hi` can round up to `2^max_exponent`.

### `two_product_dekker` and `two_product`

Dekker's product splits both operands and recovers the error from the four half-products:

```cpp
x = a * b;
[a1, a2] = split(a);   [b1, b2] = split(b);
y = a2*b2 - (((x - a1*b1) - a2*b1) - a1*b2);
```

An operand beyond `split`'s domain is scaled down by 2^(s+1) and the other operand is scaled up by 2^(s+1). Both scalings are exact and leave the product unchanged. If both operands are beyond the domain, the product overflows. Within a factor of about 1 + 2^-25 of `max`, the half-product `a1·b1` can overflow even though `a·b` does not: the halves round away from zero, and `max · 1` is the simplest case. So a product above 2^(max_exponent − 1) is formed at half scale: the larger operand is halved, the error of `x/2` is computed, and the result is doubled. Every one of these scalings is exact. So `two_product_dekker` is exact for every product that is finite and does not underflow.

With a fused multiply-add, the error is a single operation: `y = fma(a, b, −x)`. The two forms return the same pair wherever both are exact; the test suite checks this. `two_product` is the one to use. `two_product_dekker` is the paper's form and the fallback for a target without an FMA.

### Where exactness ends

- **Overflow.** A non-finite `hi` gets `lo = 0`. An overflowing sum or product is then the infinity, or NaN, it would have been without the transformation, rather than `inf − inf = NaN`.
- **Underflow, for the products.** When `a·b` is so small that its rounding error falls below the smallest subnormal, `lo` is that error rounded, not the error itself. Sums have no such limit: the error of a sum of two floating-point numbers is always representable.
- **Rounding mode.** These identities are for round-to-nearest, the IEEE default.

## Strict IEEE semantics

Each transformation depends on every operation being rounded exactly as written. Three compiler behaviours break that.

**Value-unsafe optimization: `-ffast-math`, `/fp:fast`.** Reassociation is allowed to simplify `(a − (x − z)) + (b − z)` with `z = x − a` to `0`, and `lo` is gone. MTL5 does not compile its error-free code with these flags, and code that uses these functions must not be built with them either. Under GCC and Clang `-ffast-math` is opt-in. MSVC's default, `/fp:precise`, is correct.

**FMA contraction: `-ffp-contract=fast`.** A compiler that may contract is allowed to rewrite `c − (c − a)`, where `c = factor·a`, as `fma(factor, a, −a)`. That form uses the unrounded product, so `split` returns `hi ≈ a` and `lo ≈ 0`: the halves are no longer half a significand, and Dekker's product is no longer exact. This is not hypothetical.

- **GCC** contracts by default (`-ffp-contract=fast`) wherever the target has an FMA, which means ARM64, or x86-64 with `-march=native`. Built that way without a guard, MTL5's test of `split` sees a 24-bit `hi` for `float`, where 12 is the limit.
- **Clang** contracts within a single expression by default (`-ffp-contract=on`). It honours `#pragma clang fp contract(off)` under that default and under `-ffp-contract=fast-honor-pragmas`. Plain `-ffp-contract=fast` overrides the pragma in Clang, so it is **not supported** for code that uses `split` or `two_product_dekker`.

So `split` and `two_product_dekker` switch contraction off locally:

- GCC: a `#pragma GCC optimize("fp-contract=off")` region around the two functions, honoured under any `-ffp-contract`.
- Clang (and Clang-based compilers such as icx and Clang-CL): `#pragma clang fp contract(off)` in each function body.
- MSVC: from Visual Studio 2022, MSVC contracts only under `/fp:contract` or `/fp:fast`. Older versions could contract under `/fp:precise` with `/arch:AVX2`, and need `/fp:strict` there.

`two_sum` and `fast_two_sum` contain no multiplications, so there is nothing to contract. `two_product` uses `std::fma` explicitly, so contraction cannot change it either.

The GCC region has a cost. GCC does not inline a function compiled with different optimization options into its caller, so `split` and `two_product_dekker` remain out-of-line calls from code built with default options. They are the reference algorithms; `two_product` is the fast path and is not affected.

**Extended-precision intermediates.** On 32-bit x86 without SSE2, x87 registers hold 64-bit significands, and the double roundings break the identities. Every platform MTL5 targets uses SSE2 or NEON, so this does not arise there.

## Where MTL5 uses them

- **Three-factor terms** (#602). The LDLᵀ factorizations sum terms `L(i,k)·d_k·L(j,k)`, and the accumulator protocol takes two factors. When `math::splits_product_terms_v<Acc, Value>` holds, which means an exact or a wider accumulator, the kernels feed it `two_product(L, d)` as two products, so the whole term reaches the accumulator. See `accumulator_traits.hpp` and [Exact Accumulator](exact-accumulator.md).
- **Next under #611:**
  - a compensated (Sum2/Dot2) accumulation configuration, between `fma_accumulator` and the exact accumulator;
  - `sum_k` and `dot_k`;
  - the paper's test-data generators;
  - an example that reproduces the paper's results.

## Testing

`tests/unit/math/test_error_free.cpp` checks each transformation in `float` and `double` against `math::exact_accumulator`. For example, `a + b − hi − lo` is accumulated exactly and must be zero in every limb. Over random operands spanning each function's domain, the test also checks:

- `hi` is the plain result, and `fl(hi + lo) == hi`;
- `split`'s bit widths, including at the edge of its domain;
- Dekker's product gives the FMA product's pair: bit for bit over the random operands, and by value (a zero `lo` may differ in sign) for operands up to `max` and for products just below `max`, which take the rescaling and half-scale paths;
- `fast_two_sum`'s precondition counterexample;
- the non-finite cases.

The test target is built with the most aggressive contraction setting the header supports, the opposite of the bit-reproducibility tests: **`-ffp-contract=fast`** on GCC and **`-ffp-contract=fast-honor-pragmas`** on Clang, or `-ffp-contract=on` where Clang lacks that mode (Apple Clang). On the ARM64 CI runners, which have an FMA, the `split` checks fail if the header's contraction guard stops working.

## Related

- [Exact Accumulator](exact-accumulator.md): the reference these are checked against, and configuration 3 of the accumulation protocol
- [#596](https://github.com/stillwater-sc/mtl5/issues/596): the accuracy-architecture document; it places compensated accumulation between configurations 2 and 3
- [#602](https://github.com/stillwater-sc/mtl5/issues/602): three-factor terms and `two_product`
- [#611](https://github.com/stillwater-sc/mtl5/issues/611): reproducing Ogita, Rump and Oishi
- T. Ogita, S. M. Rump, S. Oishi, "Accurate sum and dot product", *SIAM J. Sci. Comput.* 26(6):1955–1988, 2005
- T. J. Dekker, "A floating-point technique for extending the available precision", *Numer. Math.* 18:224–242, 1971
- D. E. Knuth, *The Art of Computer Programming*, Vol. 2, *Seminumerical Algorithms*, 3rd ed., Addison-Wesley, 1998
