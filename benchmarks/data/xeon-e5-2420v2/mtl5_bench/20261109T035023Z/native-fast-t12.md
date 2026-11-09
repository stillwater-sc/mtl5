## Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz -- native-fast, 12 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-42-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `73d11400a0e3` |
| FMA peak (ceiling) | f64 58.9 GFLOP/s, f32 117.9 GFLOP/s |
| ... hardware, SSE2 | f64 58.9, f32 117.9 GFLOP/s |
| ... this build, SSE4 | f64 58.9, f32 117.6 GFLOP/s |
| bandwidth | read 11.3, update 10.5, triad 8.4 GB/s (ceiling 11.3) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 98% | 106% | 0 | 9% |
| blas2 | 6 | 84% | 107% | 0 | 8% |
| blas3 | 10 | 25% | 75% | 0 | 17% |
| factor | 12 | 3% | 14% | 0 | 9% |
| eigen | 8 | 3% | 11% | 0 | 8% |
| elementwise | 5 | 7% | 13% | 0 | 8% |
| sparse | 5 | 56% | 102% | 0 | 9% |
| direct | 7 | 0% | 1% | 0 | 8% |
| ordering | 4 | n/a | n/a | 0 | 8% |
| krylov | 13 | n/a | n/a | 0 | 8% |
| pc | 7 | n/a | n/a | 0 | 8% |
| smoother | 3 | 71% | 81% | 0 | 9% |
| mg | 1 | 70% | 70% | 0 | 9% |
| sparse_eig | 4 | n/a | n/a | 0 | 9% |
| mixed | 3 | 45% | 79% | 0 | 8% |
| integer | 9 | n/a | n/a | 0 | 41% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 11.165 ms | 1.50 | 12.02 | 106% | ok | 1.6x (13%) |
| blas1 | dot | f32 | n=8388608 | 5.580 ms | 3.01 | 12.03 | 106% | ok | 1.1x (9%) |
| blas1 | axpy | f64 | n=8388608 | 18.153 ms | 0.92 | 11.09 | 98% | ok | 1.1x (9%) |
| blas1 | axpy | f32 | n=8388608 | 8.691 ms | 1.93 | 11.58 | 103% | ok | 1.1x (9%) |
| blas1 | scale | f64 | n=8388608 | 12.399 ms | 0.68 | 10.83 | 96% | ok | 1.1x (9%) |
| blas1 | two_norm | f64 | n=8388608 | 5.582 ms | 3.01 | 12.02 | 106% | ok | 1.1x (9%) |
| blas1 | one_norm | f64 | n=8388608 | 21.395 ms | 0.39 | 3.14 | 28% | ok | 0.5x (4%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.463 ms | 0.62 | 4.98 | 44% | ok | 1.0x (8%) |
| blas1 | sum | f64 | n=8388608 | 10.048 ms | 0.83 | 6.68 | 59% | ok | 1.0x (8%) |
| blas1 | dot_k3 | f64 | n=8388608 | 280.320 ms | 0.66 | 0.48 | 4% | ok | 1.0x (8%) |
| blas2 | gemv | f64 | n=4096 | 11.159 ms | 3.01 | 12.03 | 107% | ok | 1.2x (10%) |
| blas2 | gemv | f32 | n=4096 | 5.611 ms | 5.98 | 11.97 | 106% | ok | 1.2x (10%) |
| blas2 | symv | f64 | n=4096 | 16.717 ms | 2.01 | 4.02 | 36% | ok | 1.0x (8%) |
| blas2 | trmv | f64 | n=4096 | 7.089 ms | 2.37 | 9.48 | 84% | ok | 1.0x (8%) |
| blas2 | trsv | f64 | n=4096 | 7.136 ms | 2.35 | 9.41 | 83% | ok | 1.0x (8%) |
| blas2 | ger | f64 | n=4096 | 34.334 ms | 0.98 | 7.82 | 69% | ok | 0.8x (7%) |
| blas3 | gemm | f64 | 512x512x512 | 6.438 ms | 41.70 | 0.98 | 71% | ok | 4.5x (37%) |
| blas3 | gemm | f32 | 512x512x512 | 3.049 ms | 88.04 | 1.03 | 75% | ok | 4.6x (38%) |
| blas3 | gemmt | f64 | 512x512x512 | 6.806 ms | 19.72 | 0.92 | 33% | ok | 2.6x (21%) |
| blas3 | symm | f64 | 512x512x512 | 7.594 ms | 35.35 | 0.83 | 60% | ok | 3.9x (33%) |
| blas3 | syrk | f64 | 512x512x512 | 82.434 ms | 1.63 | 0.05 | 3% | ok | 1.0x (8%) |
| blas3 | syr2k | f64 | 512x512x512 | 67.088 ms | 4.00 | 0.09 | 7% | ok | 1.0x (8%) |
| blas3 | trmm | f64 | 512x512x512 | 8.963 ms | 14.97 | 0.58 | 25% | ok | 2.0x (17%) |
| blas3 | trsm | f64 | 512x512x512 | 9.364 ms | 14.33 | 0.56 | 24% | ok | 2.0x (17%) |
| blas3 | trsm | f32 | 512x512x512 | 5.049 ms | 26.58 | 0.52 | 23% | ok | 1.9x (16%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.710 ms | 0.14 | 1.10 | 10% | ok | 1.0x (9%) |
| factor | lu | f64 | 512x512 | 25.881 ms | 3.46 | 0.16 | 6% | ok | 1.2x (10%) |
| factor | lu | f32 | 512x512 | 15.390 ms | 5.81 | 0.14 | 5% | ok | 1.1x (9%) |
| factor | cholesky | f64 | 512x512 | 50.825 ms | 0.88 | 0.08 | 1% | ok | 0.5x (4%) |
| factor | cholesky | f32 | 512x512 | 24.991 ms | 1.79 | 0.08 | 2% | ok | 1.0x (8%) |
| factor | ldlt | f64 | 512x512 | 26.832 ms | 1.67 | 0.16 | 3% | ok | 1.0x (8%) |
| factor | ldlt_bk | f64 | 512x512 | 139.721 ms | 0.32 | 0.03 | 1% | ok | 1.0x (9%) |
| factor | qr | f64 | 512x512 | 397.471 ms | 0.45 | 0.01 | 1% | ok | 1.3x (11%) |
| factor | lq | f64 | 512x512 | 54.070 ms | 3.31 | 0.08 | 6% | ok | 1.4x (12%) |
| factor | hessenberg | f64 | 512x512 | 500.747 ms | 0.89 | 0.01 | 2% | ok | 1.3x (11%) |
| factor | householder | f64 | 512x512 | 2.636 ms | 0.40 | 1.59 | 14%\* | ok | 1.3x (11%) |
| factor | inv | f64 | 512x512 | 199.453 ms | 1.35 | 0.02 | 2% | ok | 1.0x (8%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 21.839 ms | 4.10 | 0.19 | 7% | ok | 1.1x (9%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 62.756 ms | 0.36 | 0.02 | 1% | ok | 1.0x (8%) |
| eigen | eigen_symmetric | f64 | 256x256 | 178.976 ms | 0.84 | 0.01 | 1% | ok | 1.0x (8%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1020.022 ms | 0.15 | 0.00 | 0% | ok | 1.0x (8%) |
| eigen | eigenvalue | f64 | 256x256 | 107.862 ms | 1.56 | 0.01 | 3% | ok | 1.0x (8%) |
| eigen | schur | f64 | 256x256 | 269.317 ms | 1.56 | 0.00 | 3% | ok | 1.0x (8%) |
| eigen | svd | f64 | 256x256 | 268.422 ms | 1.38 | 0.00 | 2% | ok | 1.0x (8%) |
| eigen | expm | f64 | 256x256 | 38.904 ms | 6.47 | 0.03 | 11% | ok | 1.5x (12%) |
| eigen | sylvester | f64 | 256x256 | 590.308 ms | 1.71 | 0.00 | 3% | ok | 1.0x (8%) |
| elementwise | exp | f64 | n=8388608 | 170.874 ms | 0.05 | 0.79 | 7% | ok | 0.9x (8%) |
| elementwise | log | f64 | n=8388608 | 144.267 ms | 0.06 | 0.93 | 8% | ok | 1.0x (8%) |
| elementwise | sin | f64 | n=8388608 | 176.593 ms | 0.05 | 0.76 | 7% | ok | 1.0x (8%) |
| elementwise | sqrt | f64 | n=8388608 | 93.034 ms | 0.09 | 1.44 | 13% | ok | 1.0x (8%) |
| elementwise | erf | f64 | n=8388608 | 2553.402 ms | 0.00 | 0.05 | 0% | ok | 1.1x (9%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.368 ms | 1.11 | 11.50 | 102% | ok | 1.1x (9%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 4.306 ms | 0.61 | 6.32 | 56% | ok | 1.0x (8%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.754 ms | 3.56 | 7.49 | 66% | ok | 1.3x (11%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 19.547 ms | 0.17 | 0.59 | 5%\* | ok | 1.7x (14%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 6.083 ms | 0.33 | 2.74 | 24% | ok | 1.0x (8%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 245.467 ms | 0.05 | 0.00 | 0% | ok | 1.0x (8%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 72.722 ms | 0.15 | 0.01 | 0% | ok | 1.0x (8%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 27.621 ms | 0.41 | 0.03 | 1% | ok | 1.0x (8%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 120.587 ms | 0.19 | 0.01 | 0% | ok | 1.0x (8%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 161.130 ms | 0.14 | 0.00 | 0% | ok | 1.0x (8%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 50.815 ms | 0.44 | 0.01 | 1% | ok | 1.0x (8%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 236.222 ms | 0.01 | 0.00 | 0% | ok | 1.0x (8%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.413 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 27.104 ms | 0.00 | 0.00 | n/a | ok | 0.5x (4%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.644 ms | 0.00 | 0.00 | n/a | ok | 1.0x (9%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 41.798 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 83.190 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 117.464 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 66.618 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 33.176 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 39.129 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 38.379 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 68.705 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 77.051 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 39.212 ms | 0.00 | 0.00 | n/a | ok | 1.1x (9%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 45.162 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 43.389 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 46.756 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 56.278 ms | 0.00 | 0.00 | n/a | ok | 1.1x (9%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.136 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.725 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | ilut | f64 | 128^2 grid | 35.056 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | ic_0 | f64 | 128^2 grid | 2.996 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | ildl | f64 | 128^2 grid | 2.849 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | ssor | f64 | 128^2 grid | 0.077 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.260 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.632 ms | 0.72 | 8.08 | 71% | ok | 1.0x (9%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.185 ms | 0.82 | 9.21 | 81% | ok | 1.0x (9%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.185 ms | 0.63 | 7.01 | 62% | ok | 1.0x (8%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 36.516 ms | 1.05 | 7.96 | 70% | ok | 1.0x (9%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 144.512 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 185.478 ms | 0.00 | 0.00 | n/a | ok | 1.1x (9%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 144.212 ms | 0.00 | 0.00 | n/a | ok | 1.1x (9%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 16.669 ms | 0.00 | 0.00 | n/a | ok | 1.0x (8%) |
| mixed | dot | f32>f64 | n=8388608 | 7.512 ms | 2.23 | 8.93 | 79% | ok | 1.0x (8%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.251 ms | 21.91 | 0.34 | 37% | ok | 5.2x (44%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 3.537 ms | 4.74 | 5.04 | 45% | ok | 0.7x (6%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.208 ms | 2.70 | 10.81 | n/a | ok | 1.0x (8%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.091 ms | 5.43 | 10.85 | n/a | ok | 1.0x (8%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.168 ms | 7.74 | 7.74 | n/a | ok | 1.0x (8%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.720 ms | 9.76 | 9.76 | n/a | ok | 1.0x (8%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 3.922 ms | 68.45 | 0.80 | n/a | ok | 5.0x (42%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.450 ms | 60.33 | 0.47 | n/a | ok | 5.2x (43%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.377 ms | 61.32 | 0.36 | n/a | ok | 5.2x (43%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.417 ms | 78.55 | 0.46 | n/a | ok | 4.9x (41%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 2.965 ms | 90.54 | 0.53 | n/a | ok | 5.9x (49%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

