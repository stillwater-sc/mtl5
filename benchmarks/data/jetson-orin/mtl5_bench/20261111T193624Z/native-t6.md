## jetson-orin -- native, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 64.2 GFLOP/s, f32 128.6 GFLOP/s |
| ... hardware, NEON | f64 64.2, f32 128.6 GFLOP/s |
| ... this build, scalar | f64 32.1, f32 32.1 GFLOP/s |
| bandwidth | read 29.3, update 35.0, triad 33.1 GB/s (ceiling 35.0) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 91% | 124% | 0 | 41% |
| blas2 | 10 | 51% | 59% | 0 | 17% |
| blas3 | 10 | 6% | 18% | 0 | 17% |
| factor | 12 | 2% | 21% | 0 | 21% |
| eigen | 8 | 3% | 8% | 0 | 17% |
| elementwise | 5 | 4% | 7% | 0 | 17% |
| sparse | 5 | 26% | 96% | 0 | 17% |
| direct | 7 | 0% | 1% | 0 | 17% |
| ordering | 4 | n/a | n/a | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 22% | 28% | 0 | 17% |
| mg | 1 | 20% | 20% | 0 | 18% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 15% | 32% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 49% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.235 ms | 3.96 | 31.70 | 91% | ok | 2.7x (46%) |
| blas1 | dot | f32 | n=8388608 | 2.228 ms | 7.53 | 30.12 | 86% | ok | 3.0x (49%) |
| blas1 | axpy | f64 | n=8388608 | 4.956 ms | 3.39 | 40.62 | 116% | ok | 2.5x (41%) |
| blas1 | axpy | f32 | n=8388608 | 2.387 ms | 7.03 | 42.17 | 121% | ok | 2.5x (42%) |
| blas1 | scale | f64 | n=8388608 | 3.088 ms | 2.72 | 43.47 | 124% | ok | 2.1x (34%) |
| blas1 | two_norm | f64 | n=8388608 | 2.040 ms | 8.22 | 32.90 | 94% | ok | 2.7x (46%) |
| blas1 | one_norm | f64 | n=8388608 | 12.503 ms | 0.67 | 5.37 | 15% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.750 ms | 0.45 | 3.58 | 10% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.501 ms | 0.67 | 5.37 | 15% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 190.977 ms | 0.97 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 7.589 ms | 4.42 | 17.69 | 51% | ok | 2.4x (39%) |
| blas2 | gemv | f32 | n=4096 | 3.336 ms | 10.06 | 20.13 | 58% | ok | 3.5x (59%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 6.523 ms | 5.14 | 20.59 | 59% | ok | 2.4x (40%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.425 ms | 9.80 | 19.60 | 56% | ok | 1.8x (31%) |
| blas2 | symv | f64 | n=4096 | 21.274 ms | 1.58 | 3.16 | 9% | ok | 1.0x (17%) |
| blas2 | trmv | f64 | n=4096 | 12.114 ms | 1.38 | 5.55 | 16% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 26.515 ms | 0.63 | 1.27 | 4% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 11.893 ms | 1.41 | 5.65 | 16% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 26.892 ms | 0.62 | 1.25 | 4% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 13.390 ms | 2.51 | 20.05 | 57% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 22.877 ms | 11.73 | 0.28 | 18% | ok | 3.3x (55%) |
| blas3 | gemm | f32 | 512x512x512 | 12.835 ms | 20.91 | 0.25 | 16% | ok | 3.0x (50%) |
| blas3 | gemmt | f64 | 512x512x512 | 47.045 ms | 2.85 | 0.13 | 4% | ok | 1.0x (17%) |
| blas3 | symm | f64 | 512x512x512 | 75.587 ms | 3.55 | 0.08 | 6% | ok | 1.0x (17%) |
| blas3 | syrk | f64 | 512x512x512 | 50.245 ms | 2.67 | 0.08 | 4% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 512x512x512 | 89.460 ms | 3.00 | 0.07 | 5% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 512x512x512 | 37.897 ms | 3.54 | 0.14 | 6% | ok | 1.0x (17%) |
| blas3 | trsm | f64 | 512x512x512 | 37.858 ms | 3.55 | 0.14 | 6% | ok | 1.0x (17%) |
| blas3 | trsm | f32 | 512x512x512 | 19.200 ms | 6.99 | 0.14 | 5% | ok | 1.0x (17%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.045 ms | 0.59 | 4.70 | 13% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 26.452 ms | 3.38 | 0.16 | 5% | ok | 1.4x (23%) |
| factor | lu | f32 | 512x512 | 16.575 ms | 5.40 | 0.13 | 4% | ok | 1.2x (21%) |
| factor | cholesky | f64 | 512x512 | 31.715 ms | 1.41 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 30.058 ms | 1.49 | 0.07 | 1% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 34.470 ms | 1.30 | 0.12 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 84.535 ms | 0.53 | 0.05 | 1% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 208.698 ms | 0.86 | 0.02 | 1% | ok | 1.6x (27%) |
| factor | lq | f64 | 512x512 | 61.325 ms | 2.92 | 0.07 | 5% | ok | 1.5x (25%) |
| factor | hessenberg | f64 | 512x512 | 323.191 ms | 1.38 | 0.01 | 2% | ok | 1.5x (25%) |
| factor | householder | f64 | 512x512 | 0.564 ms | 1.86 | 7.43 | 21% | ok | 3.6x (59%) |
| factor | inv | f64 | 512x512 | 181.972 ms | 1.48 | 0.02 | 2% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 26.621 ms | 3.36 | 0.16 | 5% | ok | 1.2x (19%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.449 ms | 0.36 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 144.619 ms | 1.04 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 948.658 ms | 0.16 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 98.211 ms | 1.71 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 213.744 ms | 1.96 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 358.073 ms | 1.03 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 49.406 ms | 5.09 | 0.02 | 8% | ok | 2.1x (35%) |
| eigen | sylvester | f64 | 256x256 | 506.498 ms | 1.99 | 0.00 | 3% | ok | 1.1x (18%) |
| elementwise | exp | f64 | n=8388608 | 75.046 ms | 0.11 | 1.79 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 107.136 ms | 0.08 | 1.25 | 4% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 177.662 ms | 0.05 | 0.76 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 54.767 ms | 0.15 | 2.45 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 126.009 ms | 0.07 | 1.07 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.815 ms | 3.21 | 33.43 | 96% | ok | 2.8x (47%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.052 ms | 0.86 | 8.92 | 26% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 5.868 ms | 7.14 | 15.00 | 43% | ok | 3.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.594 ms | 0.16 | 0.56 | 2% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.300 ms | 0.18 | 1.47 | 4% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 277.521 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 99.182 ms | 0.11 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.549 ms | 0.30 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 132.037 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 187.813 ms | 0.12 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 61.753 ms | 0.36 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 288.691 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.094 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.176 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.142 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 69.732 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 105.972 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 148.578 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 77.882 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.592 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.852 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.556 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.189 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.904 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.968 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.746 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.652 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.451 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 69.056 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.167 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.815 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 45.276 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.484 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.222 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.091 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.228 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.975 ms | 0.88 | 9.86 | 28% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.736 ms | 0.70 | 7.85 | 22% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 7.713 ms | 0.34 | 3.80 | 11% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 42.355 ms | 0.91 | 6.86 | 20% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 351.027 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 411.028 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 193.037 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.853 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 12.498 ms | 1.34 | 5.37 | 15% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.944 ms | 20.74 | 0.32 | 32% | ok | 5.8x (97%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 4.572 ms | 3.67 | 3.90 | 11% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.617 ms | 2.99 | 11.95 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.668 ms | 6.29 | 12.58 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.643 ms | 2.97 | 2.97 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.758 ms | 9.54 | 9.54 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 12.789 ms | 20.99 | 0.25 | n/a | ok | 3.0x (50%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 14.067 ms | 19.08 | 0.15 | n/a | ok | 2.9x (49%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.506 ms | 48.75 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.507 ms | 48.75 | 0.29 | n/a | ok | 5.8x (96%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 10.320 ms | 26.01 | 0.15 | n/a | ok | 5.8x (97%) |

