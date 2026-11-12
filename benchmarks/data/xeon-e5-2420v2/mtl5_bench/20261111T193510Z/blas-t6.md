## xeon-e5-2420v2 -- blas, 6 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 58.9 GFLOP/s, f32 117.8 GFLOP/s |
| ... hardware, SSE2 | f64 58.9, f32 117.8 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.6 GFLOP/s |
| bandwidth | read 11.3, update 10.6, triad 8.4 GB/s (ceiling 11.3) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 54% | 107% | 0 | 82% |
| blas2 | 10 | 104% | 108% | 0 | 20% |
| blas3 | 10 | 126% | 158% | 0 | 75% |
| factor | 12 | 18% | 38% | 0 | 24% |
| eigen | 8 | 9% | 24% | 0 | 17% |
| elementwise | 5 | 7% | 13% | 0 | 17% |
| sparse | 5 | 68% | 104% | 0 | 17% |
| direct | 7 | 0% | 1% | 0 | 17% |
| ordering | 4 | n/a | n/a | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 72% | 83% | 0 | 17% |
| mg | 1 | 74% | 74% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 22% | 67% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 90% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 32.728 ms | 0.51 | 4.10 | 36% | ok | 5.5x (92%) |
| blas1 | dot | f32 | n=8388608 | 5.537 ms | 3.03 | 12.12 | 107% | ok | 4.9x (82%) |
| blas1 | axpy | f64 | n=8388608 | 38.283 ms | 0.44 | 5.26 | 46% | ok | 5.6x (93%) |
| blas1 | axpy | f32 | n=8388608 | 16.522 ms | 1.02 | 6.09 | 54% | ok | 5.5x (92%) |
| blas1 | scale | f64 | n=8388608 | 12.140 ms | 0.69 | 11.06 | 98% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 32.883 ms | 0.51 | 2.04 | 18% | ok | 5.6x (93%) |
| blas1 | one_norm | f64 | n=8388608 | 9.749 ms | 0.86 | 6.88 | 61% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.000 ms | 0.65 | 5.16 | 46% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.777 ms | 0.86 | 6.86 | 61% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 279.926 ms | 0.66 | 0.48 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 11.306 ms | 2.97 | 11.88 | 105% | ok | 1.3x (21%) |
| blas2 | gemv | f32 | n=4096 | 5.465 ms | 6.14 | 12.29 | 108% | ok | 1.1x (19%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 10.927 ms | 3.07 | 12.29 | 108% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.698 ms | 5.89 | 11.78 | 104% | ok | 1.2x (19%) |
| blas2 | symv | f64 | n=4096 | 5.786 ms | 5.80 | 11.61 | 102% | ok | 1.2x (20%) |
| blas2 | trmv | f64 | n=4096 | 5.728 ms | 2.93 | 11.73 | 104% | ok | 1.3x (22%) |
| blas2 | trmv | f32 | n=4096 | 2.963 ms | 5.66 | 11.34 | 100% | ok | 1.2x (21%) |
| blas2 | trsv | f64 | n=4096 | 7.405 ms | 2.27 | 9.07 | 80% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 3.515 ms | 4.77 | 9.55 | 84% | ok | 1.0x (16%) |
| blas2 | ger | f64 | n=4096 | 27.907 ms | 1.20 | 9.62 | 85% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 2.875 ms | 93.37 | 2.19 | 158% | ok | 4.9x (81%) |
| blas3 | gemm | f32 | 512x512x512 | 1.806 ms | 148.61 | 1.74 | 126% | ok | 4.5x (75%) |
| blas3 | gemmt | f64 | 512x512x512 | 2.546 ms | 52.72 | 2.47 | 89% | ok | 3.4x (57%) |
| blas3 | symm | f64 | 512x512x512 | 2.900 ms | 92.56 | 2.17 | 157% | ok | 4.9x (81%) |
| blas3 | syrk | f64 | 512x512x512 | 2.519 ms | 53.27 | 1.66 | 90% | ok | 3.1x (52%) |
| blas3 | syr2k | f64 | 512x512x512 | 3.711 ms | 72.34 | 1.70 | 123% | ok | 4.1x (68%) |
| blas3 | trmm | f64 | 512x512x512 | 1.608 ms | 83.49 | 3.26 | 142% | ok | 4.6x (77%) |
| blas3 | trsm | f64 | 512x512x512 | 1.767 ms | 75.98 | 2.97 | 129% | ok | 4.6x (77%) |
| blas3 | trsm | f32 | 512x512x512 | 1.242 ms | 108.03 | 2.11 | 92% | ok | 4.3x (71%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.589 ms | 0.14 | 1.10 | 10% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 5.537 ms | 16.16 | 0.76 | 27% | ok | 1.7x (28%) |
| factor | lu | f32 | 512x512 | 3.901 ms | 22.94 | 0.54 | 19% | ok | 1.6x (27%) |
| factor | cholesky | f64 | 512x512 | 4.081 ms | 10.96 | 1.03 | 19% | ok | 1.4x (24%) |
| factor | cholesky | f32 | 512x512 | 3.108 ms | 14.39 | 0.67 | 12% | ok | 1.3x (21%) |
| factor | ldlt | f64 | 512x512 | 25.485 ms | 1.76 | 0.16 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 133.535 ms | 0.34 | 0.03 | 1% | ok | 1.1x (18%) |
| factor | qr | f64 | 512x512 | 12.973 ms | 13.79 | 0.32 | 23% | ok | 1.5x (26%) |
| factor | lq | f64 | 512x512 | 51.643 ms | 3.47 | 0.08 | 6% | ok | 1.4x (24%) |
| factor | hessenberg | f64 | 512x512 | 595.367 ms | 0.75 | 0.01 | 1% | ok | 1.1x (18%) |
| factor | householder | f64 | 512x512 | 0.974 ms | 1.08 | 4.30 | 38%\* | ok | 3.3x (55%) |
| factor | inv | f64 | 512x512 | 131.199 ms | 2.05 | 0.03 | 3% | ok | 1.0x (16%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 8.487 ms | 10.54 | 0.49 | 18% | ok | 1.5x (25%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 5.791 ms | 3.86 | 0.18 | 7% | ok | 1.1x (18%) |
| eigen | eigen_symmetric | f64 | 256x256 | 170.646 ms | 0.88 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 999.865 ms | 0.15 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 56.535 ms | 2.97 | 0.02 | 5% | ok | 0.9x (16%) |
| eigen | schur | f64 | 256x256 | 75.059 ms | 5.59 | 0.01 | 9% | ok | 1.0x (16%) |
| eigen | svd | f64 | 256x256 | 26.503 ms | 13.93 | 0.04 | 24% | ok | 1.1x (19%) |
| eigen | expm | f64 | 256x256 | 27.353 ms | 9.20 | 0.04 | 16% | ok | 1.3x (22%) |
| eigen | sylvester | f64 | 256x256 | 196.965 ms | 5.11 | 0.01 | 9% | ok | 1.0x (16%) |
| elementwise | exp | f64 | n=8388608 | 159.123 ms | 0.05 | 0.84 | 7% | ok | 0.9x (16%) |
| elementwise | log | f64 | n=8388608 | 146.509 ms | 0.06 | 0.92 | 8% | ok | 1.0x (16%) |
| elementwise | sin | f64 | n=8388608 | 171.699 ms | 0.05 | 0.78 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 89.941 ms | 0.09 | 1.49 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 2476.149 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.320 ms | 1.13 | 11.74 | 104% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.087 ms | 0.85 | 8.82 | 78% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.345 ms | 3.69 | 7.76 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.262 ms | 0.20 | 0.71 | 6%\* | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.831 ms | 0.34 | 2.85 | 25% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 275.330 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 69.596 ms | 0.16 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.254 ms | 0.43 | 0.03 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 116.281 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 153.847 ms | 0.15 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 48.136 ms | 0.46 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 231.070 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.903 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.201 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.615 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.831 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 232.458 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 397.041 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 239.140 ms | 0.00 | 0.00 | n/a | ok | 0.5x (9%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 102.078 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 121.437 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 78.925 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 514.809 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 174.287 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.537 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 136.001 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 138.021 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 173.402 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 248.610 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.129 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.442 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 33.100 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 2.869 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 2.712 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| pc | ssor | f64 | 128^2 grid | 0.077 ms | 0.00 | 0.00 | n/a | ok | 1.5x (26%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 3.518 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.597 ms | 0.73 | 8.15 | 72% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.119 ms | 0.84 | 9.40 | 83% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.804 ms | 0.69 | 7.71 | 68% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 34.676 ms | 1.11 | 8.38 | 74% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 142.441 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 147.324 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 142.578 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.322 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 189.054 ms | 0.09 | 0.35 | 3% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 20.551 ms | 13.06 | 0.20 | 22% | ok | 5.5x (92%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.340 ms | 7.17 | 7.62 | 67% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.459 ms | 2.60 | 10.39 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.148 ms | 5.33 | 10.66 | n/a | ok | 1.0x (16%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.006 ms | 3.35 | 3.35 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.991 ms | 3.36 | 3.36 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 10.410 ms | 25.79 | 0.30 | n/a | ok | 5.5x (92%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 8.237 ms | 32.59 | 0.25 | n/a | ok | 5.6x (93%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.345 ms | 61.79 | 0.36 | n/a | ok | 5.4x (91%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.343 ms | 61.81 | 0.36 | n/a | ok | 5.4x (90%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.104 ms | 20.48 | 0.12 | n/a | ok | 5.4x (90%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

