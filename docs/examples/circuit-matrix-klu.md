# Native KLU on Circuit-Simulation Matrices

This example solves real circuit-simulation matrices from the
[SuiteSparse Matrix Collection](https://sparse.tamu.edu/) with MTL5's native KLU
sparse direct solver, in `double` precision. Where the optional SuiteSparse KLU
binding is enabled, it cross-checks the native result against the external
solver.

Source: `examples/phase15_sparse_direct/circuit_matrix_klu.cpp`.

## Why KLU for circuits

Circuit simulators assemble Modified Nodal Analysis (MNA) matrices that are
**unsymmetric** and typically **reducible** — after a permutation they become
block upper triangular with many small diagonal blocks. KLU (Davis &
Palamadai Natarajan, *ACM TOMS* Algorithm 907) exploits exactly this:

1. Permute to **Block Triangular Form** (BTF, via Dulmage–Mendelsohn): only the
   diagonal blocks need factorization.
2. Factor each diagonal block with **Gilbert–Peierls left-looking LU** with
   threshold partial pivoting.
3. Solve by **block back-substitution**.

MTL5's `sparse::factorization::native_klu` is a header-only, value-type-generic
implementation of this algorithm.

## Getting a matrix

Matrices are fetched on demand and never committed:

```bash
# small/medium demo
examples/phase15_sparse_direct/fetch_circuit_matrices.sh

# also the very large stress target (Freescale/circuit5M, hundreds of MB)
examples/phase15_sparse_direct/fetch_circuit_matrices.sh all
```

Targets:

| Matrix | Size | Role |
|--------|------|------|
| [`Rajat/rajat30`](https://sparse.tamu.edu/Rajat/rajat30) | ~644K × 644K, ~6.2M nnz | runnable demo |
| [`Freescale/circuit5M`](https://sparse.tamu.edu/Freescale/circuit5M) | ~5.56M × 5.56M, ~59.5M nnz | very large stress/scaling target |

## Running

```bash
# Configure with the external KLU binding for the cross-check (optional)
cmake -B build -DMTL5_WITH_SUITESPARSE_KLU=ON
cmake --build build --target example_circuit_matrix_klu -j4

# No argument -> built-in synthetic block-triangular matrix (no download needed)
./build/examples/example_circuit_matrix_klu

# ...or on a real circuit matrix
./build/examples/example_circuit_matrix_klu \
    examples/phase15_sparse_direct/data/rajat30/rajat30.mtx
```

## What it does

- Loads the matrix with `mtl::io::mm_read<double>` (Matrix Market `coordinate
  real general`/`symmetric`), or builds a small synthetic block-triangular
  circuit matrix when no file is given.
- Builds a reproducible right-hand side `b = A · 1`, so the exact solution is
  all-ones.
- Solves with `native_klu_factor` / `klu_numeric::solve`, reporting the BTF
  block count, factor/solve time, and residual `‖Ax − b‖∞`.
- When built with `-DMTL5_WITH_SUITESPARSE_KLU=ON`, also solves with
  `mtl::interface::klu_solve` and reports `max|native − external|`.

## Notes

- `mm_read` builds the matrix in memory before compressing; `circuit5M` is large,
  so expect significant memory/time. The v1 native KLU uses natural per-block
  ordering — a per-block COLAMD ordering is a planned improvement.
- On stiff circuit matrices, very low precision can hit singular diagonal blocks;
  in `double` these matrices factor cleanly. Mixed-precision experiments live in
  the `mp-spice` sister project.
