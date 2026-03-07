# Phase 12: Element-Wise Transcendental Functions

## Context

MTL5 Phases 1-11 are complete (61 tests passing). MTL5 currently has only 4 element-wise math operations (`abs`, `sqrt`, `conj`, `negate`), while MTL4 has 18+. Phase 12 adds the full set of transcendental and special functions needed for scientific computing — signal processing, control theory, neural networks, physics simulations. All 27 functions follow the exact same pattern as the existing `abs.hpp`/`sqrt.hpp`, making this highly mechanical.

Also fixes an inconsistency: `sqrt.hpp`, `conj.hpp`, `negate.hpp` are missing matrix overloads.

## Canonical Patterns

**Operation** (from `include/mtl/operation/abs.hpp`): `template <Vector V> auto func(const V& v)` + `template <Matrix M> auto func(const M& m)`, using ADL-friendly `using std::func;` inside loop body.

**Scalar functor** (from `include/mtl/functor/scalar/sqrt.hpp`): struct with `result_type`, static `apply()`, `operator()`.

## 27 Functions in 4 Groups

### Group A — Exponential/Logarithmic (7)
`exp`, `log`, `exp2`, `log2`, `log10`, `cbrt`, `pow` (binary: scalar exponent)

### Group B — Trigonometric (6)
`sin`, `cos`, `tan`, `asin`, `acos`, `atan`

### Group C — Hyperbolic (6)
`sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`

### Group D — Rounding & Special (8)
`ceil`, `floor`, `round`, `signum` (custom), `erf`, `erfc`, `real`, `imag`

## Special Cases

- **`pow`** — Binary: `pow(v, exponent)` with `Scalar` constrained exponent
- **`signum`** — Manual: `(x>0)?1:(x<0)?-1:0`
- **`real`/`imag`** — Return `magnitude_t<T>`, use `constexpr if` for real vs complex dispatch
- **`erf`/`erfc`** — Real types only (no complex `std::erf`)

## Files to Create (61 new files)

### Operation headers (28 in `include/mtl/operation/`)
- 27 individual: `exp.hpp`, `log.hpp`, `exp2.hpp`, `log2.hpp`, `log10.hpp`, `cbrt.hpp`, `pow.hpp`, `sin.hpp`, `cos.hpp`, `tan.hpp`, `asin.hpp`, `acos.hpp`, `atan.hpp`, `sinh.hpp`, `cosh.hpp`, `tanh.hpp`, `asinh.hpp`, `acosh.hpp`, `atanh.hpp`, `ceil.hpp`, `floor.hpp`, `round.hpp`, `signum.hpp`, `erf.hpp`, `erfc.hpp`, `real.hpp`, `imag.hpp`
- 1 umbrella: `transcendental.hpp`

### Scalar functors (27 in `include/mtl/functor/scalar/`)
Same 27 names as individual operations above.

### Tests (4 in `tests/unit/operation/`)
- `test_transcendental_exp_log.cpp` — exp, log, exp2, log2, log10, cbrt, pow (known values, vector+matrix)
- `test_transcendental_trig.cpp` — sin, cos, tan, asin, acos, atan + `sin²+cos²=1` identity
- `test_transcendental_hyp.cpp` — sinh, cosh, tanh, asinh, acosh, atanh + `cosh²-sinh²=1` identity
- `test_transcendental_special.cpp` — ceil, floor, round, signum, erf, erfc, real, imag

### Examples (2 in `examples/`)
- `phase12a_signal_processing.cpp` — Sine wave generation, exponential decay, phase angles, discretization
- `phase12b_activation_functions.cpp` — Sigmoid, tanh, softmax, GELU from transcendental building blocks

## Files to Modify (4)

- `include/mtl/operation/sqrt.hpp` — Add matrix overload
- `include/mtl/operation/conj.hpp` — Add matrix overload
- `include/mtl/operation/negate.hpp` — Add matrix overload
- `include/mtl/mtl.hpp` — Add includes for all new operation + functor headers

## Implementation Order

1. Fix `sqrt.hpp`, `conj.hpp`, `negate.hpp` (add matrix overloads)
2. Group A: 7 operations + 7 functors + test
3. Group B: 6 operations + 6 functors + test
4. Group C: 6 operations + 6 functors + test
5. Group D: 8 operations + 8 functors + test
6. `transcendental.hpp` umbrella + update `mtl.hpp`
7. Examples 12a + 12b
8. Full build + full test suite

## Cross-Platform Notes

- ALL `<cmath>` standard functions only — no POSIX/GNU extensions
- `exp10` excluded (GNU extension)
- `std::numbers::pi`/`std::numbers::e` for constants (C++20), never `M_PI`/`M_E`
- `signum` manually implemented

## Verification

```bash
cd /home/stillwater/dev/stillwater/clones/mtl5
cmake -B build && cmake --build build -j$(nproc) && ctest --test-dir build
```
