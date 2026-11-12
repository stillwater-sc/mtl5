## jetson-orin -- native, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 64.3 GFLOP/s, f32 128.6 GFLOP/s |
| ... hardware, NEON | f64 64.3, f32 128.6 GFLOP/s |
| ... this build, scalar | f64 32.2, f32 32.1 GFLOP/s |
| bandwidth | read 31.2, update 34.2, triad 32.7 GB/s (ceiling 34.2) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 93% | 138% | 0 | 40% |
| blas2 | 10 | 43% | 59% | 0 | 18% |
| blas3 | 10 | 6% | 19% | 0 | 17% |
| factor | 12 | 2% | 23% | 0 | 21% |
| eigen | 8 | 3% | 8% | 0 | 17% |
| elementwise | 5 | 4% | 7% | 0 | 17% |
| sparse | 5 | 26% | 98% | 0 | 18% |
| direct | 7 | 0% | 1% | 0 | 17% |
| ordering | 4 | n/a | n/a | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 23% | 29% | 0 | 17% |
| mg | 1 | 20% | 20% | 0 | 18% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 16% | 32% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 70% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.225 ms | 3.97 | 31.77 | 93% | ok | 2.7x (46%) |
| blas1 | dot | f32 | n=8388608 | 2.123 ms | 7.90 | 31.61 | 93% | ok | 2.8x (47%) |
| blas1 | axpy | f64 | n=8388608 | 4.928 ms | 3.40 | 40.85 | 120% | ok | 2.5x (41%) |
| blas1 | axpy | f32 | n=8388608 | 2.460 ms | 6.82 | 40.92 | 120% | ok | 2.4x (40%) |
| blas1 | scale | f64 | n=8388608 | 2.842 ms | 2.95 | 47.22 | 138% | ok | 2.2x (37%) |
| blas1 | two_norm | f64 | n=8388608 | 2.042 ms | 8.21 | 32.86 | 96% | ok | 2.7x (46%) |
| blas1 | one_norm | f64 | n=8388608 | 12.501 ms | 0.67 | 5.37 | 16% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.750 ms | 0.45 | 3.58 | 10% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 16% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 192.670 ms | 0.96 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 6.720 ms | 4.99 | 19.98 | 58% | ok | 3.2x (54%) |
| blas2 | gemv | f32 | n=4096 | 4.572 ms | 7.34 | 14.69 | 43% | ok | 2.7x (45%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 6.814 ms | 4.92 | 19.71 | 58% | ok | 2.3x (39%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.418 ms | 7.60 | 15.20 | 44% | ok | 1.4x (24%) |
| blas2 | symv | f64 | n=4096 | 21.300 ms | 1.58 | 3.15 | 9% | ok | 1.0x (17%) |
| blas2 | trmv | f64 | n=4096 | 12.384 ms | 1.35 | 5.42 | 16% | ok | 1.1x (18%) |
| blas2 | trmv | f32 | n=4096 | 26.602 ms | 0.63 | 1.26 | 4% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 12.352 ms | 1.36 | 5.44 | 16% | ok | 1.1x (18%) |
| blas2 | trsv | f32 | n=4096 | 27.011 ms | 0.62 | 1.24 | 4% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 13.386 ms | 2.51 | 20.06 | 59% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 21.511 ms | 12.48 | 0.29 | 19% | ok | 3.5x (58%) |
| blas3 | gemm | f32 | 512x512x512 | 12.534 ms | 21.42 | 0.25 | 17% | ok | 3.0x (51%) |
| blas3 | gemmt | f64 | 512x512x512 | 48.401 ms | 2.77 | 0.13 | 4% | ok | 1.0x (17%) |
| blas3 | symm | f64 | 512x512x512 | 75.648 ms | 3.55 | 0.08 | 6% | ok | 1.0x (17%) |
| blas3 | syrk | f64 | 512x512x512 | 51.121 ms | 2.63 | 0.08 | 4% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 512x512x512 | 92.848 ms | 2.89 | 0.07 | 4% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 512x512x512 | 37.946 ms | 3.54 | 0.14 | 6% | ok | 1.0x (17%) |
| blas3 | trsm | f64 | 512x512x512 | 37.885 ms | 3.54 | 0.14 | 6% | ok | 1.0x (17%) |
| blas3 | trsm | f32 | 512x512x512 | 19.232 ms | 6.98 | 0.14 | 5% | ok | 1.0x (17%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.271 ms | 0.57 | 4.58 | 13% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 26.334 ms | 3.40 | 0.16 | 5% | ok | 1.4x (23%) |
| factor | lu | f32 | 512x512 | 16.065 ms | 5.57 | 0.13 | 4% | ok | 1.3x (21%) |
| factor | cholesky | f64 | 512x512 | 31.719 ms | 1.41 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 30.064 ms | 1.49 | 0.07 | 1% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 34.421 ms | 1.30 | 0.12 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 83.966 ms | 0.53 | 0.05 | 1% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 207.576 ms | 0.86 | 0.02 | 1% | ok | 1.6x (27%) |
| factor | lq | f64 | 512x512 | 62.030 ms | 2.88 | 0.07 | 4% | ok | 1.5x (25%) |
| factor | hessenberg | f64 | 512x512 | 328.244 ms | 1.36 | 0.01 | 2% | ok | 1.4x (24%) |
| factor | householder | f64 | 512x512 | 0.545 ms | 1.93 | 7.70 | 23% | ok | 3.7x (62%) |
| factor | inv | f64 | 512x512 | 182.168 ms | 1.47 | 0.02 | 2% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 26.828 ms | 3.34 | 0.16 | 5% | ok | 1.1x (19%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 60.956 ms | 0.37 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 144.409 ms | 1.05 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 943.243 ms | 0.16 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 98.212 ms | 1.71 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 213.956 ms | 1.96 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 358.137 ms | 1.03 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 49.477 ms | 5.09 | 0.02 | 8% | ok | 2.1x (35%) |
| eigen | sylvester | f64 | 256x256 | 501.273 ms | 2.01 | 0.00 | 3% | ok | 1.1x (18%) |
| elementwise | exp | f64 | n=8388608 | 75.684 ms | 0.11 | 1.77 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 107.914 ms | 0.08 | 1.24 | 4% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 178.207 ms | 0.05 | 0.75 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 54.978 ms | 0.15 | 2.44 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 126.559 ms | 0.07 | 1.06 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.815 ms | 3.21 | 33.39 | 98% | ok | 2.8x (47%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.041 ms | 0.86 | 8.95 | 26% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 5.871 ms | 7.13 | 15.00 | 44% | ok | 3.0x (50%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.596 ms | 0.16 | 0.56 | 2% | ok | 1.0x (16%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.127 ms | 0.18 | 1.50 | 4% | ok | 1.1x (18%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 277.400 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 99.100 ms | 0.11 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.440 ms | 0.30 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 130.002 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 184.649 ms | 0.12 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 61.853 ms | 0.36 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 284.997 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.034 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.290 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.131 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 65.511 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 105.716 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 147.976 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 77.725 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.555 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.723 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.495 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.453 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.137 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.823 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.689 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.497 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.433 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 69.021 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.166 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.838 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 45.108 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.471 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.236 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.090 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.231 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.964 ms | 0.88 | 9.89 | 29% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.741 ms | 0.70 | 7.84 | 23% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 7.664 ms | 0.34 | 3.83 | 11% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 42.363 ms | 0.91 | 6.86 | 20% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 352.401 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 412.038 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 190.563 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.881 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 12.499 ms | 1.34 | 5.37 | 16% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.939 ms | 20.75 | 0.32 | 32% | ok | 5.8x (97%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 4.571 ms | 3.67 | 3.90 | 11% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.588 ms | 3.00 | 12.01 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.692 ms | 6.23 | 12.47 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.693 ms | 2.95 | 2.95 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.758 ms | 9.54 | 9.54 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 9.005 ms | 29.81 | 0.35 | n/a | ok | 4.2x (70%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 7.087 ms | 37.88 | 0.30 | n/a | ok | 5.8x (97%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.485 ms | 48.94 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.486 ms | 48.93 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 10.322 ms | 26.01 | 0.15 | n/a | ok | 5.8x (97%) |

