## arduino-ventuno-q -- native, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 67.5 GFLOP/s, f32 135.5 GFLOP/s |
| ... hardware, NEON | f64 67.5, f32 135.5 GFLOP/s |
| ... this build, scalar | f64 33.7, f32 33.9 GFLOP/s |
| bandwidth | read 23.6, update 25.1, triad 22.9 GB/s (ceiling 25.1) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 90% | 107% | 0 | 27% |
| blas2 | 10 | 52% | 72% | 0 | 27% |
| blas3 | 10 | 9% | 16% | 0 | 29% |
| factor | 12 | 2% | 18% | 0 | 28% |
| eigen | 8 | 4% | 9% | 0 | 28% |
| elementwise | 5 | 5% | 8% | 0 | 27% |
| sparse | 5 | 54% | 70% | 0 | 27% |
| direct | 7 | 0% | 1% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 28% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 28% |
| smoother | 3 | 51% | 59% | 0 | 27% |
| mg | 1 | 33% | 33% | 0 | 23% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 30% | 37% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 55% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.958 ms | 2.82 | 22.53 | 90% | ok | 0.9x (24%) |
| blas1 | dot | f32 | n=8388608 | 3.072 ms | 5.46 | 21.85 | 87% | ok | 1.5x (37%) |
| blas1 | axpy | f64 | n=8388608 | 8.226 ms | 2.04 | 24.48 | 98% | ok | 0.9x (24%) |
| blas1 | axpy | f32 | n=8388608 | 4.217 ms | 3.98 | 23.87 | 95% | ok | 0.9x (23%) |
| blas1 | scale | f64 | n=8388608 | 5.022 ms | 1.67 | 26.72 | 107% | ok | 1.1x (27%) |
| blas1 | two_norm | f64 | n=8388608 | 2.859 ms | 5.87 | 23.47 | 94% | ok | 1.1x (28%) |
| blas1 | one_norm | f64 | n=8388608 | 7.692 ms | 1.09 | 8.72 | 35% | ok | 1.1x (27%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.459 ms | 0.54 | 4.34 | 17% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.897 ms | 1.06 | 8.50 | 34% | ok | 1.1x (26%) |
| blas1 | dot_k3 | f64 | n=8388608 | 178.871 ms | 1.03 | 0.75 | 3% | ok | 1.1x (27%) |
| blas2 | gemv | f64 | n=4096 | 7.760 ms | 4.32 | 17.31 | 69% | ok | 1.2x (30%) |
| blas2 | gemv | f32 | n=4096 | 3.957 ms | 8.48 | 16.97 | 68% | ok | 1.7x (44%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 8.003 ms | 4.19 | 16.78 | 67% | ok | 1.2x (31%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.188 ms | 6.47 | 12.94 | 52% | ok | 0.8x (21%) |
| blas2 | symv | f64 | n=4096 | 15.842 ms | 2.12 | 4.24 | 17% | ok | 1.1x (27%) |
| blas2 | trmv | f64 | n=4096 | 5.749 ms | 2.92 | 11.68 | 47% | ok | 0.9x (22%) |
| blas2 | trmv | f32 | n=4096 | 6.104 ms | 2.75 | 5.50 | 22% | ok | 1.1x (28%) |
| blas2 | trsv | f64 | n=4096 | 5.354 ms | 3.13 | 12.55 | 50% | ok | 0.9x (23%) |
| blas2 | trsv | f32 | n=4096 | 19.346 ms | 0.87 | 1.74 | 7% | ok | 1.1x (27%) |
| blas2 | ger | f64 | n=4096 | 14.897 ms | 2.25 | 18.02 | 72% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 27.034 ms | 9.93 | 0.23 | 15% | ok | 2.0x (50%) |
| blas3 | gemm | f32 | 512x512x512 | 12.044 ms | 22.29 | 0.26 | 16% | ok | 2.1x (52%) |
| blas3 | gemmt | f64 | 512x512x512 | 35.650 ms | 3.76 | 0.18 | 6% | ok | 1.2x (29%) |
| blas3 | symm | f64 | 512x512x512 | 54.711 ms | 4.91 | 0.11 | 7% | ok | 1.0x (24%) |
| blas3 | syrk | f64 | 512x512x512 | 35.106 ms | 3.82 | 0.12 | 6% | ok | 1.2x (31%) |
| blas3 | syr2k | f64 | 512x512x512 | 78.992 ms | 3.40 | 0.08 | 5% | ok | 1.0x (25%) |
| blas3 | trmm | f64 | 512x512x512 | 22.576 ms | 5.95 | 0.23 | 9% | ok | 1.1x (29%) |
| blas3 | trsm | f64 | 512x512x512 | 22.810 ms | 5.88 | 0.23 | 9% | ok | 1.2x (29%) |
| blas3 | trsm | f32 | 512x512x512 | 10.830 ms | 12.39 | 0.24 | 9% | ok | 1.1x (28%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 28.360 ms | 0.19 | 1.50 | 6% | ok | 1.1x (28%) |
| factor | lu | f64 | 512x512 | 109.537 ms | 0.82 | 0.04 | 1% | ok | 0.2x (6%) |
| factor | lu | f32 | 512x512 | 91.533 ms | 0.98 | 0.02 | 1% | ok | 0.1x (4%) |
| factor | cholesky | f64 | 512x512 | 19.947 ms | 2.24 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 17.565 ms | 2.55 | 0.12 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 19.210 ms | 2.33 | 0.22 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 61.333 ms | 0.73 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 213.764 ms | 0.84 | 0.02 | 1% | ok | 1.2x (29%) |
| factor | lq | f64 | 512x512 | 138.218 ms | 1.29 | 0.03 | 2% | ok | 0.4x (11%) |
| factor | hessenberg | f64 | 512x512 | 447.266 ms | 1.00 | 0.01 | 1% | ok | 0.8x (19%) |
| factor | householder | f64 | 512x512 | 0.936 ms | 1.12 | 4.48 | 18% | ok | 1.7x (41%) |
| factor | inv | f64 | 512x512 | 220.125 ms | 1.22 | 0.02 | 2% | ok | 0.7x (16%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 109.426 ms | 0.82 | 0.04 | 1% | ok | 0.2x (5%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.739 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.841 ms | 1.74 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 525.379 ms | 0.29 | 0.00 | 0% | ok | 1.1x (28%) |
| eigen | eigenvalue | f64 | 256x256 | 60.025 ms | 2.80 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 134.076 ms | 3.13 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.438 ms | 1.80 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 42.850 ms | 5.87 | 0.02 | 9% | ok | 1.6x (39%) |
| eigen | sylvester | f64 | 256x256 | 320.048 ms | 3.15 | 0.00 | 5% | ok | 1.2x (30%) |
| elementwise | exp | f64 | n=8388608 | 81.364 ms | 0.10 | 1.65 | 7% | ok | 1.1x (27%) |
| elementwise | log | f64 | n=8388608 | 100.498 ms | 0.08 | 1.34 | 5% | ok | 1.1x (28%) |
| elementwise | sin | f64 | n=8388608 | 135.677 ms | 0.06 | 0.99 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 65.958 ms | 0.13 | 2.03 | 8% | ok | 1.1x (27%) |
| elementwise | erf | f64 | n=8388608 | 107.588 ms | 0.08 | 1.25 | 5% | ok | 1.1x (27%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.555 ms | 1.68 | 17.51 | 70% | ok | 1.1x (27%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.990 ms | 1.32 | 13.68 | 55% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.449 ms | 6.49 | 13.65 | 54% | ok | 1.9x (47%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.795 ms | 0.28 | 0.98 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 9.907 ms | 0.20 | 1.68 | 7% | ok | 0.9x (22%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 163.561 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 53.182 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.388 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 91.661 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 125.560 ms | 0.18 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.527 ms | 0.57 | 0.02 | 1% | ok | 1.1x (27%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 181.881 ms | 0.01 | 0.00 | 0% | ok | 1.1x (28%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.049 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.033 ms | 0.00 | 0.00 | n/a | ok | 1.1x (26%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.630 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.715 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.123 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 97.680 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.571 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.093 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.451 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.546 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.815 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 56.981 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.116 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.528 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.188 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 36.045 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.209 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.099 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.374 ms | 0.00 | 0.00 | n/a | ok | 1.2x (30%) |
| pc | ilut | f64 | 128^2 grid | 27.633 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.797 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.675 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ssor | f64 | 128^2 grid | 0.062 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.552 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.977 ms | 1.32 | 14.83 | 59% | ok | 1.1x (27%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.288 ms | 1.14 | 12.82 | 51% | ok | 1.1x (27%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.472 ms | 0.59 | 6.56 | 26% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.349 ms | 1.09 | 8.22 | 33% | ok | 0.9x (23%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 209.007 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 243.701 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 133.262 ms | 0.00 | 0.00 | n/a | ok | 1.2x (31%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.968 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.221 ms | 2.32 | 9.29 | 37% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 13.140 ms | 20.43 | 0.32 | 30% | ok | 3.7x (93%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.657 ms | 6.31 | 6.71 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.811 ms | 5.97 | 23.87 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.373 ms | 12.22 | 24.45 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.269 ms | 5.13 | 5.13 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.036 ms | 16.20 | 16.20 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 12.256 ms | 21.90 | 0.26 | n/a | ok | 2.2x (55%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 7.445 ms | 36.06 | 0.28 | n/a | ok | 3.5x (89%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.901 ms | 45.49 | 0.27 | n/a | ok | 3.4x (86%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.889 ms | 45.58 | 0.27 | n/a | ok | 3.4x (86%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 10.483 ms | 25.61 | 0.15 | n/a | ok | 3.7x (92%) |

