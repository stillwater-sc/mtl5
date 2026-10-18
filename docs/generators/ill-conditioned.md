# Ill-Conditioned Sums and Dot Products

`mtl::generators::gen_dot` and `mtl::generators::gen_sum` build vectors whose dot product or sum has a chosen condition number, up to hundreds of bits of cancellation. They are the test-data generators of Ogita, Rump and Oishi, "Accurate sum and dot product", *SIAM J. Sci. Comput.* 26(6), 2005, Section 6. They are what MTL5 uses to measure its accumulation configurations and its K-fold algorithms against the paper.

Header: `include/mtl/generators/ill_conditioned.hpp` (also pulled in by `mtl/generators/generators.hpp` and `mtl/mtl.hpp`).

```cpp
#include <mtl/generators/ill_conditioned.hpp>

auto d = mtl::generators::gen_dot<double>(100, 1e30, std::uint64_t{42});
// d.x, d.y: dense_vector<double> of length 100
// d.d:      x . y, rounded to nearest
// d.cond:   the achieved condition number, 2 |x|.|y| / |x.y|

auto s = mtl::generators::gen_sum<double>(200, 1e30, std::uint64_t{42});
// s.p:      dense_vector<double> of length 200
// s.s:      sum p, rounded to nearest
// s.cond:   the achieved condition number, sum |p_i| / |sum p_i|
```

Each generator has two overloads: one takes a seed, and the other takes a `std::mt19937_64&` so that a battery can draw a sequence of instances from one stream.

## The condition numbers

The paper defines the condition number of a dot product in (5.7) and that of a sum in (4.9):

| | condition number | |
|---|---|---|
| dot product | `cond(xᵀy) = 2·|x|ᵀ|y| / |xᵀy|` | (5.7) |
| sum | `cond(Σpᵢ) = Σ|pᵢ| / |Σpᵢ|` | (4.9) |

The error bounds of every algorithm in [Error-Free Transformations](../architecture/error-free-transformations.md) are in terms of these. Both generators report the **achieved** condition number, computed exactly, and the paper plots its errors against that value rather than the anticipated one.

## `gen_dot`: GenDot (Algorithm 6.1)

For a length `n ≥ 6` and an anticipated condition number `c`, with `b = log2(c)` and `n2 = round(n/2)`:

1. **First half.** `x(i)` and `y(i)` are random in `(−1, 1)·2^e`, with exponents `e` drawn from `0..b/2`. The first exponent is set to `round(b/2) + 1` and the last to 0, so that both ends of the range occur.
2. **Second half.** The exponents fall linearly from `b/2` to 0. Each `x(i)` is random at that exponent. `y(i)` is chosen so that the exact running dot product is cancelled down to a random value at that exponent:
   `y(i) = (r·2^e − DotExact(x, y)) / x(i)`.
3. **Permute.** `x` and `y` are permuted together.

This follows the paper's Matlab code step by step. The only external routine it needs is `DotExact`, "a floating point number r ∈ F near to the exact value of the dot product". MTL5 uses `math::exact_accumulator`, which computes the exact sum and rounds it once to nearest. So no substitute is involved, and the generator is defined for the types the exact accumulator covers: `float` and `double`.

**How close the achieved condition number is.** As the paper reports in Table 6.1, the achieved condition number is typically about 10× the anticipated one, with a wide spread. This is the ratio of achieved to anticipated over 500 draws with `n = 100..500`:

| | min | median | max | range of `c` |
|---|---|---|---|---|
| MTL5, double | 0.020 | 11.2 | 4.9e3 | 1 .. 1e100 |
| MTL5, float | 0.38 | 15.7 | 3.7e3 | 1 .. 1e30 |
| paper, Table 6.1 | 1.5e-3 | 10.0 | 1.9e6 | 1 .. 1e100, 50 000 draws |

## `gen_sum`: the paper's ill-conditioned sums

The paper builds an ill-conditioned sum of length 2m from a dot product of length m: "using Algorithm 3.3 (TwoProduct) and randomly permuting the summands". `gen_sum(n, c)` does the following for an even `n ≥ 12`:

1. It draws `gen_dot(n/2, c)`.
2. It splits each product `x(i)·y(i)` exactly into `hi + lo` with `math::two_product`.
3. It permutes the `n` summands.

The exact sum is therefore `x·y`, and `Σ|pᵢ|` equals `|x|ᵀ|y|` to within the rounding of the `hi` parts. The sum's condition number is consequently half the dot product's.

## Range and portability

- **Range.** The entries reach `2^(b/2 + 1)` and the products `2^(b + 2)`, so `c` must stay below about `2^(max_exponent − 4)`: about 2e37 in `float` and 1e307 in `double`. This is asserted.
- **Portable draws.** Every random number is taken from the raw `std::mt19937_64` stream. The standard specifies that stream bit for bit; it does not specify `<random>`'s distributions, whose output differs between standard libraries. So a seed gives the same vectors with every compiler and standard library. The paper's `rand` is uniform on (0, 1); here it is a 53-bit uniform on [0, 1).
- **Arithmetic.** The construction uses `T` arithmetic: for `float`, the cancellation step is done in `float`. The paper works in double only.

## Testing

`tests/unit/generators/test_ill_conditioned.cpp` checks:

- reproducibility from a seed;
- that `d` is the exact dot product rounded to nearest;
- that `cond` is the exact (5.7);
- that the median achieved-to-anticipated ratio lies within [1, 100];
- that `gen_sum`'s summands add up exactly to the GenDot product drawn from the same seed, with half its condition number.

The `[586:dot]`, `[586:gemm]` and `[586:determinism]` batteries in `tests/unit/operation/test_accumulation_configs_mt.cpp` draw their data from `gen_dot`.

## Related

- [Error-Free Transformations](../architecture/error-free-transformations.md): `two_sum`, `two_product`, the compensated accumulator, `sum_k` and `dot_k`, and their bounds
- [Exact Accumulator](../architecture/exact-accumulator.md): the `DotExact` behind GenDot
- [#611](https://github.com/stillwater-sc/mtl5/issues/611): reproducing Ogita, Rump and Oishi
