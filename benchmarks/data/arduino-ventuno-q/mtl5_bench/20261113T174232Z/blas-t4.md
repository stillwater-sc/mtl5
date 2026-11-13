## arduino-ventuno-q -- blas, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bf315e4604cf` |
| FMA peak (ceiling) | f64 67.6 GFLOP/s, f32 134.9 GFLOP/s |
| ... hardware, NEON | f64 67.6, f32 134.9 GFLOP/s |
| ... this build, scalar | f64 33.7, f32 33.8 GFLOP/s |
| bandwidth | read 23.0, update 24.5, triad 22.5 GB/s (ceiling 24.5) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 99% | 109% | 0 | 28% |
| blas2 | 10 | 64% | 107% | 0 | 37% |
| blas3 | 10 | 75% | 81% | 0 | 88% |
| factor | 12 | 18% | 38% | 0 | 44% |
| eigen | 8 | 11% | 28% | 0 | 33% |
| elementwise | 5 | 6% | 8% | 0 | 26% |
| sparse | 5 | 57% | 93% | 0 | 28% |
| direct | 23 | 1% | 11% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 28% |
| direct3d | 15 | 2% | 24% | 0 | 27% |
| krylov | 13 | n/a | n/a | 0 | 27% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 45% | 61% | 0 | 27% |
| mg | 1 | 37% | 37% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 27% |
| mixed | 3 | 31% | 38% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 53% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.554 ms | 3.02 | 24.17 | 99% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.815 ms | 5.96 | 23.84 | 97% | ok | 1.6x (40%) |
| blas1 | axpy | f64 | n=8388608 | 8.272 ms | 2.03 | 24.34 | 99% | ok | 0.9x (24%) |
| blas1 | axpy | f32 | n=8388608 | 4.091 ms | 4.10 | 24.60 | 100% | ok | 1.0x (24%) |
| blas1 | scale | f64 | n=8388608 | 5.019 ms | 1.67 | 26.74 | 109% | ok | 1.1x (27%) |
| blas1 | two_norm | f64 | n=8388608 | 2.619 ms | 6.40 | 25.62 | 105% | ok | 1.2x (29%) |
| blas1 | one_norm | f64 | n=8388608 | 7.690 ms | 1.09 | 8.73 | 36% | ok | 1.1x (27%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.473 ms | 0.54 | 4.34 | 18% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.449 ms | 1.13 | 9.01 | 37% | ok | 1.1x (28%) |
| blas1 | dot_k3 | f64 | n=8388608 | 170.810 ms | 1.08 | 0.79 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 5.111 ms | 6.56 | 26.27 | 107% | ok | 1.8x (44%) |
| blas2 | gemv | f32 | n=4096 | 2.596 ms | 12.92 | 25.86 | 106% | ok | 1.5x (37%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 8.592 ms | 3.91 | 15.63 | 64% | ok | 1.1x (27%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.938 ms | 6.80 | 13.60 | 55% | ok | 0.8x (19%) |
| blas2 | symv | f64 | n=4096 | 6.494 ms | 5.17 | 10.34 | 42% | ok | 1.8x (44%) |
| blas2 | trmv | f64 | n=4096 | 3.660 ms | 4.58 | 18.36 | 75% | ok | 1.7x (43%) |
| blas2 | trmv | f32 | n=4096 | 2.204 ms | 7.61 | 15.24 | 62% | ok | 1.5x (37%) |
| blas2 | trsv | f64 | n=4096 | 5.994 ms | 2.80 | 11.21 | 46% | ok | 1.1x (27%) |
| blas2 | trsv | f32 | n=4096 | 3.078 ms | 5.45 | 10.91 | 45% | ok | 1.0x (26%) |
| blas2 | ger | f64 | n=4096 | 14.703 ms | 2.28 | 18.26 | 75% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 5.010 ms | 53.58 | 1.26 | 79% | ok | 3.5x (88%) |
| blas3 | gemm | f32 | 512x512x512 | 2.442 ms | 109.93 | 1.29 | 81% | ok | 3.5x (88%) |
| blas3 | gemmt | f64 | 512x512x512 | 3.412 ms | 39.33 | 1.84 | 58% | ok | 3.3x (83%) |
| blas3 | symm | f64 | 512x512x512 | 5.106 ms | 52.58 | 1.23 | 78% | ok | 3.5x (88%) |
| blas3 | syrk | f64 | 512x512x512 | 3.262 ms | 41.15 | 1.29 | 61% | ok | 3.0x (76%) |
| blas3 | syr2k | f64 | 512x512x512 | 5.739 ms | 46.77 | 1.10 | 69% | ok | 3.3x (84%) |
| blas3 | trmm | f64 | 512x512x512 | 2.516 ms | 53.34 | 2.08 | 79% | ok | 3.6x (90%) |
| blas3 | trsm | f64 | 512x512x512 | 2.657 ms | 50.51 | 1.97 | 75% | ok | 3.7x (91%) |
| blas3 | trsm | f32 | 512x512x512 | 1.446 ms | 92.79 | 1.81 | 69% | ok | 3.5x (87%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 29.514 ms | 0.18 | 1.44 | 6% | ok | 0.9x (22%) |
| factor | lu | f64 | 512x512 | 4.106 ms | 21.79 | 1.02 | 32% | ok | 2.2x (56%) |
| factor | lu | f32 | 512x512 | 3.135 ms | 28.54 | 0.67 | 21% | ok | 1.8x (44%) |
| factor | cholesky | f64 | 512x512 | 2.978 ms | 15.02 | 1.41 | 22% | ok | 1.8x (45%) |
| factor | cholesky | f32 | 512x512 | 2.003 ms | 22.34 | 1.05 | 17% | ok | 1.6x (40%) |
| factor | ldlt | f64 | 512x512 | 20.205 ms | 2.21 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 60.592 ms | 0.74 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 9.641 ms | 18.56 | 0.44 | 27% | ok | 2.0x (51%) |
| factor | lq | f64 | 512x512 | 36.377 ms | 4.92 | 0.12 | 7% | ok | 1.7x (42%) |
| factor | hessenberg | f64 | 512x512 | 199.692 ms | 2.24 | 0.02 | 3% | ok | 1.8x (44%) |
| factor | householder | f64 | 512x512 | 0.446 ms | 2.35 | 9.40 | 38% | ok | 3.5x (87%) |
| factor | inv | f64 | 512x512 | 116.185 ms | 2.31 | 0.04 | 3% | ok | 1.1x (28%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.381 ms | 12.12 | 0.57 | 18% | ok | 1.4x (35%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 5.379 ms | 4.16 | 0.19 | 6% | ok | 1.4x (34%) |
| eigen | eigen_symmetric | f64 | 256x256 | 87.133 ms | 1.73 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 539.104 ms | 0.28 | 0.00 | 0% | ok | 1.2x (29%) |
| eigen | eigenvalue | f64 | 256x256 | 37.224 ms | 4.51 | 0.03 | 7% | ok | 1.2x (30%) |
| eigen | schur | f64 | 256x256 | 47.925 ms | 8.75 | 0.02 | 13% | ok | 1.3x (33%) |
| eigen | svd | f64 | 256x256 | 19.602 ms | 18.83 | 0.05 | 28% | ok | 1.5x (38%) |
| eigen | expm | f64 | 256x256 | 23.763 ms | 10.59 | 0.04 | 16% | ok | 1.6x (40%) |
| eigen | sylvester | f64 | 256x256 | 136.135 ms | 7.39 | 0.01 | 11% | ok | 1.3x (32%) |
| elementwise | exp | f64 | n=8388608 | 81.017 ms | 0.10 | 1.66 | 7% | ok | 1.0x (26%) |
| elementwise | log | f64 | n=8388608 | 98.493 ms | 0.09 | 1.36 | 6% | ok | 1.1x (27%) |
| elementwise | sin | f64 | n=8388608 | 134.180 ms | 0.06 | 1.00 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 66.690 ms | 0.13 | 2.01 | 8% | ok | 1.0x (26%) |
| elementwise | erf | f64 | n=8388608 | 107.022 ms | 0.08 | 1.25 | 5% | ok | 1.0x (26%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.200 ms | 2.18 | 22.70 | 93% | ok | 1.4x (34%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.953 ms | 1.34 | 13.94 | 57% | ok | 1.1x (28%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.128 ms | 6.83 | 14.37 | 59% | ok | 2.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.584 ms | 0.28 | 0.99 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.951 ms | 0.22 | 1.86 | 8% | ok | 1.0x (25%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 163.315 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 52.088 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.461 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 93.877 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 128.229 ms | 0.17 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.730 ms | 0.56 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 185.655 ms | 0.01 | 0.00 | 0% | ok | 1.1x (27%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.782 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.088 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.633 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.837 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.336 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 155.405 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.304 ms | 0.33 | 1.31 | 5% | ok | 0.8x (20%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.645 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.353 ms | 1.08 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.156 ms | 0.35 | 1.40 | 6% | ok | 0.8x (20%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.060 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 80.173 ms | 0.28 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.706 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.238 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 109.607 ms | 0.20 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.694 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 14.788 ms | 0.76 | 0.05 | 1% | ok | 1.1x (28%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 3.503 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 9.942 ms | 1.13 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.099 ms | 0.69 | 2.75 | 11% | ok | 1.1x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 18.952 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 14667.743 ms | 0.02 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.976 ms | 0.42 | 1.69 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 41.845 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 187.033 ms | 1.65 | 0.00 | 2% | ok | 1.1x (27%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.789 ms | 0.43 | 1.73 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.881 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1583.298 ms | 0.39 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.524 ms | 0.12 | 0.47 | 2% | ok | 1.0x (25%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 26.417 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2308.999 ms | 0.27 | 0.00 | 0% | ok | 1.1x (26%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.194 ms | 0.12 | 0.48 | 2% | ok | 1.0x (25%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 6.695 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 28.052 ms | 11.00 | 0.03 | 16% | ok | 1.6x (39%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 2.286 ms | 1.47 | 5.90 | 24% | ok | 1.1x (26%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.027 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 96.507 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 49.051 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.637 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.606 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.856 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.863 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 58.505 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.967 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 34.039 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.683 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.890 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.356 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.102 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.644 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.746 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.852 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.779 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.065 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.700 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.973 ms | 1.33 | 14.86 | 61% | ok | 1.1x (27%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 4.286 ms | 0.61 | 6.84 | 28% | ok | 1.1x (27%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.677 ms | 0.98 | 10.96 | 45% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.201 ms | 1.19 | 9.03 | 37% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 209.027 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 214.362 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 135.143 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 12.006 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.220 ms | 2.32 | 9.29 | 38% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.781 ms | 21.00 | 0.33 | 31% | ok | 3.8x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.655 ms | 6.32 | 6.71 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.805 ms | 5.98 | 23.92 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.370 ms | 12.25 | 24.49 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.269 ms | 5.13 | 5.13 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.040 ms | 16.14 | 16.14 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.538 ms | 23.26 | 0.27 | n/a | ok | 2.1x (53%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.680 ms | 40.18 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.140 ms | 52.23 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.138 ms | 52.24 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 9.723 ms | 27.61 | 0.16 | n/a | ok | 4.0x (99%) |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

