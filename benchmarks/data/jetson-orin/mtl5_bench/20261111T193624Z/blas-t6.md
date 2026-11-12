## jetson-orin -- blas, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 64.3 GFLOP/s, f32 128.5 GFLOP/s |
| ... hardware, NEON | f64 64.3, f32 128.5 GFLOP/s |
| ... this build, scalar | f64 32.1, f32 32.1 GFLOP/s |
| bandwidth | read 27.5, update 34.4, triad 35.0 GB/s (ceiling 35.0) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 91% | 135% | 0 | 40% |
| blas2 | 10 | 60% | 85% | 0 | 43% |
| blas3 | 10 | 72% | 86% | 0 | 85% |
| factor | 12 | 11% | 24% | 0 | 26% |
| eigen | 8 | 7% | 19% | 0 | 20% |
| elementwise | 5 | 4% | 7% | 0 | 17% |
| sparse | 5 | 26% | 97% | 0 | 18% |
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
| blas1 | dot | f64 | n=8388608 | 4.224 ms | 3.97 | 31.77 | 91% | ok | 2.7x (46%) |
| blas1 | dot | f32 | n=8388608 | 2.229 ms | 7.53 | 30.10 | 86% | ok | 3.0x (49%) |
| blas1 | axpy | f64 | n=8388608 | 5.016 ms | 3.34 | 40.14 | 115% | ok | 2.4x (41%) |
| blas1 | axpy | f32 | n=8388608 | 2.467 ms | 6.80 | 40.80 | 117% | ok | 2.4x (40%) |
| blas1 | scale | f64 | n=8388608 | 2.847 ms | 2.95 | 47.15 | 135% | ok | 2.2x (37%) |
| blas1 | two_norm | f64 | n=8388608 | 2.030 ms | 8.26 | 33.06 | 94% | ok | 2.8x (46%) |
| blas1 | one_norm | f64 | n=8388608 | 12.499 ms | 0.67 | 5.37 | 15% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.749 ms | 0.45 | 3.58 | 10% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 15% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 192.127 ms | 0.96 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 4.617 ms | 7.27 | 29.08 | 83% | ok | 2.7x (46%) |
| blas2 | gemv | f32 | n=4096 | 2.267 ms | 14.80 | 29.61 | 85% | ok | 2.6x (43%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 6.357 ms | 5.28 | 21.12 | 60% | ok | 2.0x (34%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.727 ms | 9.00 | 18.02 | 51% | ok | 1.6x (26%) |
| blas2 | symv | f64 | n=4096 | 5.102 ms | 6.58 | 13.17 | 38% | ok | 3.1x (52%) |
| blas2 | trmv | f64 | n=4096 | 2.876 ms | 5.83 | 23.36 | 67% | ok | 3.0x (50%) |
| blas2 | trmv | f32 | n=4096 | 1.434 ms | 11.70 | 23.41 | 67% | ok | 3.4x (57%) |
| blas2 | trsv | f64 | n=4096 | 8.581 ms | 1.96 | 7.83 | 22% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 4.852 ms | 3.46 | 6.92 | 20% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 13.371 ms | 2.51 | 20.08 | 57% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 4.830 ms | 55.58 | 1.30 | 86% | ok | 5.5x (92%) |
| blas3 | gemm | f32 | 512x512x512 | 2.498 ms | 107.46 | 1.26 | 84% | ok | 5.2x (87%) |
| blas3 | gemmt | f64 | 512x512x512 | 4.092 ms | 32.80 | 1.54 | 51% | ok | 4.0x (67%) |
| blas3 | symm | f64 | 512x512x512 | 4.835 ms | 55.52 | 1.30 | 86% | ok | 5.5x (92%) |
| blas3 | syrk | f64 | 512x512x512 | 3.615 ms | 37.13 | 1.16 | 58% | ok | 4.0x (66%) |
| blas3 | syr2k | f64 | 512x512x512 | 5.996 ms | 44.77 | 1.05 | 70% | ok | 4.7x (78%) |
| blas3 | trmm | f64 | 512x512x512 | 2.602 ms | 51.57 | 2.01 | 80% | ok | 5.2x (87%) |
| blas3 | trsm | f64 | 512x512x512 | 2.920 ms | 45.97 | 1.80 | 72% | ok | 5.1x (85%) |
| blas3 | trsm | f32 | 512x512x512 | 1.575 ms | 85.21 | 1.66 | 66% | ok | 4.8x (80%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.907 ms | 0.54 | 4.29 | 12% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 5.757 ms | 15.54 | 0.73 | 24% | ok | 2.4x (40%) |
| factor | lu | f32 | 512x512 | 4.578 ms | 19.55 | 0.46 | 15% | ok | 1.8x (31%) |
| factor | cholesky | f64 | 512x512 | 4.185 ms | 10.69 | 1.00 | 17% | ok | 1.9x (32%) |
| factor | cholesky | f32 | 512x512 | 3.200 ms | 13.98 | 0.66 | 11% | ok | 1.6x (26%) |
| factor | ldlt | f64 | 512x512 | 32.242 ms | 1.39 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 84.295 ms | 0.53 | 0.05 | 1% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 13.805 ms | 12.96 | 0.30 | 20% | ok | 2.1x (34%) |
| factor | lq | f64 | 512x512 | 61.544 ms | 2.91 | 0.07 | 5% | ok | 1.5x (25%) |
| factor | hessenberg | f64 | 512x512 | 321.510 ms | 1.39 | 0.01 | 2% | ok | 1.5x (25%) |
| factor | householder | f64 | 512x512 | 0.551 ms | 1.90 | 7.61 | 22% | ok | 3.6x (60%) |
| factor | inv | f64 | 512x512 | 164.905 ms | 1.63 | 0.03 | 3% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.070 ms | 5.57 | 0.26 | 9% | ok | 1.1x (18%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 7.029 ms | 3.18 | 0.15 | 5% | ok | 1.5x (25%) |
| eigen | eigen_symmetric | f64 | 256x256 | 144.870 ms | 1.04 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 896.998 ms | 0.17 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 56.313 ms | 2.98 | 0.02 | 5% | ok | 1.1x (19%) |
| eigen | schur | f64 | 256x256 | 74.029 ms | 5.67 | 0.01 | 9% | ok | 1.2x (20%) |
| eigen | svd | f64 | 256x256 | 29.852 ms | 12.36 | 0.04 | 19% | ok | 1.4x (24%) |
| eigen | expm | f64 | 256x256 | 37.402 ms | 6.73 | 0.03 | 10% | ok | 1.5x (25%) |
| eigen | sylvester | f64 | 256x256 | 210.516 ms | 4.78 | 0.01 | 7% | ok | 1.2x (20%) |
| elementwise | exp | f64 | n=8388608 | 75.698 ms | 0.11 | 1.77 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 108.208 ms | 0.08 | 1.24 | 4% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 178.168 ms | 0.05 | 0.75 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 54.951 ms | 0.15 | 2.44 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 126.871 ms | 0.07 | 1.06 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.803 ms | 3.26 | 33.90 | 97% | ok | 2.9x (48%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.044 ms | 0.86 | 8.95 | 26% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 5.799 ms | 7.22 | 15.18 | 43% | ok | 3.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.444 ms | 0.16 | 0.56 | 2% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.149 ms | 0.18 | 1.49 | 4% | ok | 1.1x (18%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 279.868 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 101.525 ms | 0.11 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.355 ms | 0.30 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 131.874 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 190.694 ms | 0.12 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 62.017 ms | 0.36 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 287.133 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.169 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.264 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.141 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 66.082 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 105.481 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 147.175 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 77.779 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.437 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.526 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.489 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.280 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.988 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.724 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.602 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.476 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.157 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 68.825 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.166 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.684 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 44.739 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.465 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.202 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.090 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 4.071 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.023 ms | 0.87 | 9.70 | 28% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.736 ms | 0.70 | 7.85 | 22% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.508 ms | 0.58 | 6.51 | 19% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 42.160 ms | 0.91 | 6.89 | 20% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 351.904 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 357.190 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 194.961 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.975 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 12.499 ms | 1.34 | 5.37 | 15% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.966 ms | 20.70 | 0.32 | 32% | ok | 5.8x (97%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 5.578 ms | 3.01 | 3.20 | 9% | ok | 0.8x (14%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.564 ms | 3.02 | 12.06 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.660 ms | 6.31 | 12.62 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.645 ms | 2.97 | 2.97 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.758 ms | 9.54 | 9.54 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 12.835 ms | 20.91 | 0.25 | n/a | ok | 3.0x (49%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 14.088 ms | 19.05 | 0.15 | n/a | ok | 2.9x (49%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.504 ms | 48.77 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.510 ms | 48.72 | 0.29 | n/a | ok | 5.8x (96%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 10.327 ms | 25.99 | 0.15 | n/a | ok | 5.8x (97%) |

