# QA sweep config for MTL5

Read by the `/qa-sweep` skill and the `qa-reviewer` agent, which are synced
from [stillwater-sc/claude-plugins](https://github.com/stillwater-sc/claude-plugins/tree/main/plugins/qa-sweep).
This file is MTL5's own; edit it here.

## Repository

`stillwater-sc/mtl5`

## Project

MTL5 is a C++20 header-only linear algebra library: dense and sparse matrices
and vectors, expression templates, factorizations, sparse direct and iterative
solvers, and optional BLAS/LAPACK dispatch, generic over custom number types.
`CLAUDE.md` describes the layout, the namespaces, the cross-platform rules and
the dispatch, and the findings should be judged against it.

## Build and test

At most `-j4`. Build directories: `build-ci/` and `build-asan/`.

```bash
cmake --preset ci -DMTL5_BUILD_EXAMPLES=OFF
cmake --build build-ci -j4
ctest --test-dir build-ci -j4 --output-on-failure
```

The examples are left out on purpose. Tier 1 CI builds them on eight
platforms for every PR and every push to `main`, so a break shows there; the
sweep's build time goes to the tests and the sanitizer pass instead.

A test's name is its target's: `<prefix>_test_<stem>`, where the prefix
follows the directory under `tests/unit/` (`op` for `operation/`, `iface` for
`interface/`, `concept` for `concepts/`).

## Docs-only paths

`docs/`, `docs-site/`, and Markdown files and images outside `.claude/`.

## Areas

| # | Area | Headers | Test selection |
|---|---|---|---|
| 0 | Factorizations and solvers | `include/mtl/operation/`: `lu*`, `qr`, `lq`, `cholesky`, `ldlt*`, `svd`, `schur`, `hessenberg`, `eigenvalue*`, `householder`, `givens`, `sylvester`, `expm`, `inv`, `*trisolve`, `trsv`, `trsm` | `-R "$FACT"` |
| 1 | Kernels and element-wise operations | the rest of `include/mtl/operation/` | `-R '^op_test_' -E "$FACT"` |
| 2 | Sparse direct solvers | `include/mtl/sparse/` | `-R '^sparse_'` |
| 3 | Iterative solvers | `include/mtl/itl/` | `-R '^itl_'` |
| 4 | Matrix types | `include/mtl/mat/` | `-R '^mat_'` |
| 5 | Vector types | `include/mtl/vec/` | `-R '^vec_'` |
| 6 | External library bindings | `include/mtl/interface/` | `-R '^iface_'` |
| 7 | Recursion, threading, SIMD | `include/mtl/recursion/`, `parallel/`, `simd/` | `-R '^(recursion\|parallel\|simd)_'` |
| 8 | I/O and generators | `include/mtl/io/`, `generators/` | `-R '^(io\|generators)_'` |
| 9 | Foundations | `include/mtl/concepts/`, `traits/`, `tag/`, `math/`, `functor/`, `detail/`, `util/` | `-R '^(concept\|math\|functor\|detail\|util)_'` |
| 10 | Arrays and tensors | `include/mtl/array/`, `tensor/` | `-R '^(array\|tensor)_'` |

The `\|` in the patterns is Markdown escaping; the regex alternation is a
plain `|`, as in `^(io|generators)_`. Areas 0 and 1 split the operation tests
with one pattern, so the two sets are disjoint:

```bash
FACT='^op_test_(lu|qr|lq|cholesky|ldlt|svd|schur|hessenberg|eigen|householder|givens|sylvester|expm|inv|trisolve|trsv|trsm|factorization)'
```

Shell variables don't carry over between commands, so set `FACT` in the same
command as the `ctest` that uses it.

An area's tests are its tests under `tests/unit/` and, for areas 0, 2 and 3,
`tests/regression/`. The regression tests are for reading only: they are off
by default (`MTL5_BUILD_REGRESSION_TESTS`) and too slow to run under the
sanitizers.

## Sanitizers

AddressSanitizer and UndefinedBehaviorSanitizer, over the area's unit tests:

```bash
cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_STANDARD=20 \
  -DMTL5_BUILD_TESTS=ON -DMTL5_BUILD_EXAMPLES=OFF \
  -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined"
ctest --test-dir build-asan -N <selection>   # lists the area's tests, which are the targets to build
cmake --build build-asan -j4 --target <the listed names>
ctest --test-dir build-asan <selection> --output-on-failure
```

The first frame to report is the first one inside `include/mtl/`.

## Review focus

- **Shapes and aliasing:** empty, 1×1 and non-square matrices; aliasing
  between arguments, such as `A = A * B` and `x = trans(A) * x`.
- **Generic scalars:** code that takes any `Scalar` but assumes
  `float`/`double`: `std::abs`/`std::sqrt` instead of the ADL forms,
  `math::zero<T>()`/`math::one<T>()` bypassed, `numeric_limits` taken as
  defined for posits, LNS and other custom types.
- **Dispatch:** a BLAS, LAPACK, SIMD or threaded path that disagrees with the
  generic one for `dense2D<float/double>`: layouts, leading dimensions,
  transposes, `info` codes ignored. Also the reverse: a case that should
  dispatch but doesn't.
- **Expression templates:** views and expressions that outlive their
  operands, and expressions evaluated early into temporaries.
- **Sparse:** dense-sized work or allocation, quadratic searches, inserters
  used without reservation, and CRS/CCS index ordering at the solver
  bindings.
- **Cross-platform rules** in `CLAUDE.md`: `M_PI` and other POSIX constants,
  POSIX-only headers without guards, hardcoded paths, `long` as a size.
- **Tests:** a public function, overload or `if constexpr` branch that no
  test under `tests/unit/` or `tests/regression/` reaches.
