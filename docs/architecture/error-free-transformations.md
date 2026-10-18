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

## The compensated accumulator: Sum2 / Dot2

`math::compensated_accumulator<T>` (in `accumulator_traits.hpp`) is Ogita, Rump and Oishi's Sum2 and Dot2 as an accumulation configuration, **2c**, between `fma_accumulator` and the exact accumulator. Any accumulator-aware MTL5 operation takes it:

```cpp
using comp = mtl::math::compensated_accumulator<double>;
double d = mtl::dot<comp, double>(x, y);       // Dot2
mtl::mult<comp>(A, B, C);                      // each C(i,j) a Dot2
```

Its state is two `T`s, `sum` and `err`. Each term is processed in four steps:

1. `two_product` splits `m·v` into its rounded value `h` and that rounding's error.
2. `two_sum` adds `h` to `sum`, which yields a second error.
3. Both errors go into `err`.
4. The round-out is `sum + err`.

The paper's error bound for Dot2 is

```
|res − x·y| ≤ u·|x·y| + γ_n²·Σ|x_i·y_i|,     u = 2^−p, the paper's eps;  γ_n = n·u / (1 − n·u),
```

where configurations 1 and 2 have `γ_n·Σ|x_i·y_i|`. The error therefore stays at `u` until the condition number approaches `1/u`, and only then grows, as `u²·cond`. The `[586:dot]` battery asserts this bound on every instance. The data come from `generators::gen_dot`, the paper's GenDot ([Ill-Conditioned Sums and Dot Products](../generators/ill-conditioned.md)). The table gives the geometric mean of the relative error over 40 instances of `n = 100` per anticipated condition number; the achieved condition number is typically about 10× higher, as in the paper:

| `T` | cond | plain | fma | **compensated** | exact |
|---|---|---|---|---|---|
| float | 1e4 | 2.8e-3 | 2.9e-3 | **1.3e-8** | 1.3e-8 |
| float | 1e8 | 9.6 | 11 | **1.7e-6** | 1.5e-8 |
| float | 1e12 | 1.0e5 | 1.1e5 | **1.5e-2** | 1.6e-8 |
| double | 1e8 | 2.8e-8 | 3.8e-8 | **2.9e-17** | 2.9e-17 |
| double | 1e16 | 4.8 | 4.3 | **9.5e-16** | 3.5e-17 |
| double | 1e24 | 1.0e8 | 8.5e7 | **1.5e-8** | 2.8e-17 |

How it differs from the other configurations:

- **Size.** It is 2 words, 16 bytes for `double`, against 536 bytes for the exact accumulator. A kernel that keeps one accumulator per output entry therefore keeps 2n words, not 67n.
- **Cost.** It takes about 10 flops per term: 2 for `two_product`, 6 for `two_sum` and 2 for the errors. There is no limb arithmetic.
- **Merge.** Two partials combine as a `two_sum` of the sums plus both errors. The result stays within the bound but is not exact. A threaded reduction is therefore deterministic for a given thread count, but not bit-identical across thread counts. The exact accumulator's merge, by contrast, is the same for every partition.
- **Three-factor terms.** It keeps a product's rounding error, so `splits_product_terms_v` holds for it, and the LDLᵀ kernels feed it whole terms. On the perturbed-Hilbert ensemble of `test_three_factor_terms.cpp`, ILDL with it lands on the exact accumulator's mean residual to two digits.
- **Widths.** A `Value` wider than `T` keeps its low part in `err` on `assign`. A `Result` wider than `T` gets `sum + err` added in its own precision.
- **Scope.** It works for built-in binary floating `T` only, and it needs the strict IEEE semantics above.

## K-fold precision: `sum_k` and `dot_k`

`include/mtl/operation/sum_dot_k.hpp` implements the paper's K-fold algorithms. The algorithm, proposition and bound numbers below are taken from the paper.

```cpp
double s = mtl::sum_k<3>(x);       // the sum, as if in 3-fold precision, rounded once
double d = mtl::dot_k<4>(x, y);    // the dot product, as if in 4-fold precision
```

- **VecSum** (Algorithm 4.3) runs `two_sum` down a vector. The exact sum does not change. The last entry becomes the rounded sum and the others become the rounding errors. When the sum cancels, those errors are small, so one pass improves the condition number of what is left by about a factor of `u`.
- **SumK** (Algorithm 4.8) makes K − 1 VecSum passes, then sums plainly. `sum_k<1>` is the recursive sum, and `sum_k<2>` is Sum2, the same bits as `mtl::sum<compensated_accumulator<T>>`.
- **DotK** (Algorithm 5.10) collects each `two_product`'s error and the running sum's `two_sum` errors into 2n terms, then runs SumK over them with K − 1. As in the paper, `dot_k<2>` is Dot2 (Algorithm 5.3), the same bits as `mtl::dot<compensated_accumulator<T>, T>`.

The bounds use `s` for the exact result, `S = Σ|pᵢ|` (or `|x|·|y|`), `u` for the unit roundoff (the paper's `eps`) and `γ_m = m·u/(1 − m·u)`:

| | bound | paper |
|---|---|---|
| `sum_k<2>` | `u·|s| + γ²_{n−1}·S` | Proposition 4.5 |
| `sum_k<K>`, K ≥ 3 | `(u + 3γ²_{n−1})·|s| + γ^K_{2n−2}·S` | Proposition 4.10 |
| `dot_k<2>` | `u·|x·y| + γ_n²·|x|·|y|` | Proposition 5.5 |
| `dot_k<K>`, K ≥ 3 | `(u + 2γ²_{4n−2})·|x·y| + γ^K_{4n−2}·|x|·|y|` | Proposition 5.11 |

So the relative error is `u` plus about `(φu)^K` times the condition number, and each extra pass buys another factor of `u`. `test_sum_dot_k.cpp` asserts every bound for K = 1..5, on cancelling sums and dot products in `float` and `double`. It also shows the K-fold behaviour. The table gives the worst relative error over 20 dot products of length 200:

| `T` | cond | K = 2 | K = 3 | K = 4 |
|---|---|---|---|---|
| float | 2^48 | 17 | 3.0e-6 | 5.0e-8 |
| double | 2^106 | 8.0 | 1.5e-15 | 6.8e-17 |

**SumK and DotK are not accumulators.** They keep every error term, n words for a sum and 2n for a dot product, and make K − 1 passes over them. They therefore work on whole vectors, not through `accumulator_traits`. Only K = 2 streams, and it is the compensated accumulator. In the language of the accuracy-architecture document (#596): DotK applies where a reduction's terms can be stored, as in a `dot` or a row of a `gemv`. It does not apply to the inner reduction of a factorization.

**Cost.** `sum_k<K>` takes `(6K − 5)(n − 1)` flops (Proposition 4.10). `dot_k<K>` with the FMA `two_product` takes about `(12K − 14)·n`: 8n for the products and the running sum, then SumK over 2n terms. Each call makes one allocation of n or 2n values.

## Where MTL5 uses them

- **Three-factor terms** (#602). The LDLᵀ factorizations sum terms `L(i,k)·d_k·L(j,k)`, and the accumulator protocol takes two factors. When `math::splits_product_terms_v<Acc, Value>` holds, which means an exact, a compensated or a wider accumulator, the kernels feed it `two_product(L, d)` as two products, so the whole term reaches the accumulator. See `accumulator_traits.hpp` and [Exact Accumulator](exact-accumulator.md).
- **The compensated accumulator** (configuration 2c, above).
- **`sum_k` and `dot_k`** (above).
- **The paper's test data** (GenDot, GenSum): `generators::gen_dot` and `generators::gen_sum`, see [Ill-Conditioned Sums and Dot Products](../generators/ill-conditioned.md).
- **The paper's experiments**: `examples/accurate_sum_dot/accurate_sum_dot.cpp` runs Section 6 of the paper (Figures 6.1–6.4, Tables 6.2, 6.4 and 6.5) and writes the results as CSV.

## Testing

`tests/unit/operation/test_ogita_rump_oishi.cpp` asserts the paper's theorems over the paper's own test data. The battery is GenDot and GenSum with `n` ∈ {50, 200, 1000}, anticipated condition numbers 1..1e100 in `double` and 1..1e30 in `float`, and three seeds each. Each test case is one numbered result, checked on every instance:

| Result | What it bounds | MTL5 code checked |
|---|---|---|
| Theorem 3.4 | TwoSum and TwoProduct are error-free, and `\|y\| ≤ eps·\|x\|` | `two_sum`, `two_product`, `two_product_dekker` |
| (2.4) | the recursive sum | `sum_k<1>` |
| Proposition 4.5 | Sum2 | `sum_k<2>`, `sum<compensated_accumulator>` |
| Corollary 4.7 | Sum2's computable bound `err`, which must enclose the exact sum (decided exactly) | Sum2 with the paper's appended statements |
| (4.14), Lemma 4.9 | each VecSum pass keeps the sum and shrinks `Σ\|p\|` | `detail::vec_sum`, 5 passes |
| Proposition 4.10 | SumK | `sum_k<3..6>` |
| Proposition 5.5, Corollary 5.7 | Dot2, absolute and relative | `dot_k<2>`, `dot<compensated_accumulator>` |
| Proposition 5.11, (5.11) | DotK, absolute and relative | `dot_k<3..6>` |

The tests assert the inequalities, never individual values. The exact quantities (`s`, `S`, `|x|·|y|` and every error) come from `math::exact_accumulator`.

`tests/unit/math/test_compensated_accumulator.cpp` checks that `mtl::dot` with the compensated accumulator gives the bits of the paper's Dot2 loop, written out with Dekker's TwoProduct. It also checks the cancellations that plain and FMA lose, the merge, wider initial values and results, and non-finite inputs. Its accuracy is checked in the #586 matrix.

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
