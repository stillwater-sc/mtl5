## jetson-orin -- native, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 64.3 GFLOP/s, f32 128.5 GFLOP/s |
| ... hardware, NEON | f64 64.3, f32 128.5 GFLOP/s |
| ... this build, scalar | f64 32.2, f32 32.1 GFLOP/s |
| bandwidth | read 28.0, update 34.5, triad 36.4 GB/s (ceiling 36.4) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 85% | 129% | 0 | 40% |
| blas2 | 10 | 55% | 79% | 0 | 17% |
| blas3 | 10 | 5% | 23% | 0 | 17% |
| factor | 12 | 2% | 21% | 0 | 23% |
| eigen | 8 | 3% | 8% | 0 | 17% |
| elementwise | 5 | 3% | 7% | 0 | 17% |
| sparse | 5 | 24% | 89% | 0 | 17% |
| direct | 19 | 0% | 4% | 0 | 17% |
| ordering | 4 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 1% | 4% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 22% | 27% | 0 | 17% |
| mg | 1 | 19% | 19% | 0 | 18% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 15% | 32% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 71% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.262 ms | 3.94 | 31.49 | 86% | ok | 2.7x (45%) |
| blas1 | dot | f32 | n=8388608 | 2.193 ms | 7.65 | 30.60 | 84% | ok | 3.0x (50%) |
| blas1 | axpy | f64 | n=8388608 | 5.032 ms | 3.33 | 40.01 | 110% | ok | 2.4x (41%) |
| blas1 | axpy | f32 | n=8388608 | 2.455 ms | 6.83 | 41.00 | 112% | ok | 2.4x (40%) |
| blas1 | scale | f64 | n=8388608 | 2.849 ms | 2.94 | 47.11 | 129% | ok | 2.2x (37%) |
| blas1 | two_norm | f64 | n=8388608 | 2.174 ms | 7.72 | 30.87 | 85% | ok | 2.6x (43%) |
| blas1 | one_norm | f64 | n=8388608 | 12.502 ms | 0.67 | 5.37 | 15% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.754 ms | 0.45 | 3.58 | 10% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.503 ms | 0.67 | 5.37 | 15% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 192.600 ms | 0.96 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 5.235 ms | 6.41 | 25.65 | 70% | ok | 2.5x (42%) |
| blas2 | gemv | f32 | n=4096 | 2.344 ms | 14.31 | 28.64 | 79% | ok | 3.3x (55%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 6.415 ms | 5.23 | 20.93 | 57% | ok | 2.5x (42%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.309 ms | 10.14 | 20.29 | 56% | ok | 1.9x (32%) |
| blas2 | symv | f64 | n=4096 | 21.335 ms | 1.57 | 3.15 | 9% | ok | 1.0x (17%) |
| blas2 | trmv | f64 | n=4096 | 10.952 ms | 1.53 | 6.13 | 17% | ok | 0.8x (14%) |
| blas2 | trmv | f32 | n=4096 | 6.971 ms | 2.41 | 4.82 | 13% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 8.864 ms | 1.89 | 7.58 | 21% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 6.811 ms | 2.46 | 4.93 | 14% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 13.352 ms | 2.51 | 20.11 | 55% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 22.134 ms | 12.13 | 0.28 | 19% | ok | 3.4x (57%) |
| blas3 | gemm | f32 | 512x512x512 | 9.060 ms | 29.63 | 0.35 | 23% | ok | 4.2x (70%) |
| blas3 | gemmt | f64 | 512x512x512 | 48.192 ms | 2.79 | 0.13 | 4% | ok | 1.0x (17%) |
| blas3 | symm | f64 | 512x512x512 | 75.880 ms | 3.54 | 0.08 | 6% | ok | 1.0x (16%) |
| blas3 | syrk | f64 | 512x512x512 | 49.623 ms | 2.70 | 0.08 | 4% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 512x512x512 | 93.478 ms | 2.87 | 0.07 | 4% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 512x512x512 | 39.084 ms | 3.43 | 0.13 | 5% | ok | 1.0x (16%) |
| blas3 | trsm | f64 | 512x512x512 | 38.347 ms | 3.50 | 0.14 | 5% | ok | 1.0x (16%) |
| blas3 | trsm | f32 | 512x512x512 | 19.024 ms | 7.06 | 0.14 | 5% | ok | 1.0x (16%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.276 ms | 0.57 | 4.58 | 13% | ok | 1.1x (18%) |
| factor | lu | f64 | 512x512 | 24.863 ms | 3.60 | 0.17 | 6% | ok | 1.4x (24%) |
| factor | lu | f32 | 512x512 | 14.721 ms | 6.08 | 0.14 | 5% | ok | 1.4x (23%) |
| factor | cholesky | f64 | 512x512 | 31.670 ms | 1.41 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 30.170 ms | 1.48 | 0.07 | 1% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 34.397 ms | 1.30 | 0.12 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 87.109 ms | 0.51 | 0.05 | 1% | ok | 1.0x (16%) |
| factor | qr | f64 | 512x512 | 210.233 ms | 0.85 | 0.02 | 1% | ok | 1.6x (27%) |
| factor | lq | f64 | 512x512 | 59.527 ms | 3.01 | 0.07 | 5% | ok | 1.6x (26%) |
| factor | hessenberg | f64 | 512x512 | 314.040 ms | 1.42 | 0.01 | 2% | ok | 1.5x (25%) |
| factor | householder | f64 | 512x512 | 0.542 ms | 1.94 | 7.74 | 21% | ok | 3.6x (60%) |
| factor | inv | f64 | 512x512 | 184.292 ms | 1.46 | 0.02 | 2% | ok | 1.0x (17%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 19.616 ms | 4.56 | 0.21 | 7% | ok | 1.3x (21%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.329 ms | 0.36 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 144.826 ms | 1.04 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 897.995 ms | 0.17 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 98.682 ms | 1.70 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 213.486 ms | 1.96 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 357.991 ms | 1.03 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 49.132 ms | 5.12 | 0.02 | 8% | ok | 2.1x (35%) |
| eigen | sylvester | f64 | 256x256 | 501.776 ms | 2.01 | 0.00 | 3% | ok | 1.1x (18%) |
| elementwise | exp | f64 | n=8388608 | 75.695 ms | 0.11 | 1.77 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 107.680 ms | 0.08 | 1.25 | 3% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 177.880 ms | 0.05 | 0.75 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 54.992 ms | 0.15 | 2.44 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 126.548 ms | 0.07 | 1.06 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.841 ms | 3.11 | 32.36 | 89% | ok | 2.7x (46%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.060 ms | 0.86 | 8.90 | 24% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.708 ms | 8.90 | 18.70 | 51% | ok | 3.6x (61%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.781 ms | 0.16 | 0.55 | 2% | ok | 1.0x (16%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.499 ms | 0.17 | 1.45 | 4% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 276.926 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 101.710 ms | 0.11 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.184 ms | 0.30 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 131.171 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 186.795 ms | 0.12 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 62.187 ms | 0.36 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 289.082 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.122 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.228 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.129 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 66.844 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 8.783 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 265.576 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.165 ms | 0.35 | 1.40 | 4% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 18.018 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.488 ms | 0.68 | 0.04 | 1% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.119 ms | 0.36 | 1.43 | 4% | ok | 1.0x (17%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.631 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 112.346 ms | 0.20 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.582 ms | 0.10 | 0.40 | 1% | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 14.607 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 161.405 ms | 0.14 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.573 ms | 0.10 | 0.40 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 30.992 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 25332.009 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 9.995 ms | 0.34 | 1.35 | 4% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 64.466 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 277.169 ms | 1.11 | 0.00 | 2% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.010 ms | 0.34 | 1.35 | 4% | ok | 1.0x (17%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.145 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1913.317 ms | 0.32 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.797 ms | 0.09 | 0.37 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 41.775 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 3162.686 ms | 0.20 | 0.00 | 0% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 37.052 ms | 0.09 | 0.36 | 1% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 105.993 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 147.645 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.024 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.662 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.838 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.635 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.713 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.026 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.960 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.767 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.517 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.468 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 67.983 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.166 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.704 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 45.380 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.557 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.264 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.090 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.327 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.000 ms | 0.87 | 9.78 | 27% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.736 ms | 0.70 | 7.85 | 22% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.529 ms | 0.58 | 6.48 | 18% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 42.374 ms | 0.91 | 6.86 | 19% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 350.851 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 411.172 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 195.807 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 21.031 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 12.498 ms | 1.34 | 5.37 | 15% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.894 ms | 20.82 | 0.33 | 32% | ok | 5.8x (97%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 5.603 ms | 2.99 | 3.18 | 9% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.587 ms | 3.00 | 12.01 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.739 ms | 6.12 | 12.25 | n/a | ok | 1.0x (16%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.671 ms | 2.96 | 2.96 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.758 ms | 9.55 | 9.55 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 8.983 ms | 29.88 | 0.35 | n/a | ok | 4.2x (71%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 7.075 ms | 37.94 | 0.30 | n/a | ok | 5.8x (97%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.484 ms | 48.94 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.479 ms | 48.99 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 10.299 ms | 26.06 | 0.15 | n/a | ok | 5.8x (97%) |

