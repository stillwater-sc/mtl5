## arduino-ventuno-q -- native-fast, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON_WITHOUT_AES |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 67.5 GFLOP/s, f32 135.6 GFLOP/s |
| ... hardware, NEON | f64 67.5, f32 135.6 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 67.5, f32 135.5 GFLOP/s |
| bandwidth | read 23.1, update 24.8, triad 22.8 GB/s (ceiling 24.8) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 91% | 106% | 0 | 27% |
| blas2 | 10 | 56% | 82% | 0 | 28% |
| blas3 | 10 | 18% | 51% | 0 | 40% |
| factor | 12 | 2% | 18% | 0 | 27% |
| eigen | 8 | 4% | 11% | 0 | 28% |
| elementwise | 5 | 5% | 8% | 0 | 28% |
| sparse | 5 | 55% | 71% | 0 | 27% |
| direct | 7 | 0% | 1% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 28% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 28% |
| smoother | 3 | 51% | 60% | 0 | 27% |
| mg | 1 | 33% | 33% | 0 | 22% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 36% | 73% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 87% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.969 ms | 2.81 | 22.49 | 91% | ok | 1.0x (24%) |
| blas1 | dot | f32 | n=8388608 | 3.100 ms | 5.41 | 21.65 | 87% | ok | 0.9x (23%) |
| blas1 | axpy | f64 | n=8388608 | 8.309 ms | 2.02 | 24.23 | 98% | ok | 0.9x (23%) |
| blas1 | axpy | f32 | n=8388608 | 4.312 ms | 3.89 | 23.34 | 94% | ok | 0.9x (23%) |
| blas1 | scale | f64 | n=8388608 | 5.083 ms | 1.65 | 26.40 | 106% | ok | 1.0x (26%) |
| blas1 | two_norm | f64 | n=8388608 | 2.872 ms | 5.84 | 23.36 | 94% | ok | 1.1x (27%) |
| blas1 | one_norm | f64 | n=8388608 | 7.655 ms | 1.10 | 8.77 | 35% | ok | 1.1x (27%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.425 ms | 0.54 | 4.35 | 18% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.311 ms | 1.15 | 9.18 | 37% | ok | 1.1x (28%) |
| blas1 | dot_k3 | f64 | n=8388608 | 178.051 ms | 1.04 | 0.75 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 6.915 ms | 4.85 | 19.42 | 78% | ok | 1.2x (30%) |
| blas2 | gemv | f32 | n=4096 | 3.282 ms | 10.22 | 20.46 | 82% | ok | 1.2x (31%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 29.809 ms | 1.13 | 4.50 | 18% | ok | 1.7x (42%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 12.554 ms | 2.67 | 5.35 | 22% | ok | 2.0x (49%) |
| blas2 | symv | f64 | n=4096 | 15.526 ms | 2.16 | 4.33 | 17% | ok | 1.1x (28%) |
| blas2 | trmv | f64 | n=4096 | 4.822 ms | 3.48 | 13.93 | 56% | ok | 1.0x (26%) |
| blas2 | trmv | f32 | n=4096 | 2.432 ms | 6.90 | 13.81 | 56% | ok | 1.1x (27%) |
| blas2 | trsv | f64 | n=4096 | 4.867 ms | 3.45 | 13.80 | 56% | ok | 1.0x (25%) |
| blas2 | trsv | f32 | n=4096 | 2.453 ms | 6.84 | 13.69 | 55% | ok | 1.1x (27%) |
| blas2 | ger | f64 | n=4096 | 15.059 ms | 2.23 | 17.83 | 72% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 7.809 ms | 34.38 | 0.81 | 51% | ok | 3.6x (90%) |
| blas3 | gemm | f32 | 512x512x512 | 4.209 ms | 63.77 | 0.75 | 47% | ok | 3.3x (82%) |
| blas3 | gemmt | f64 | 512x512x512 | 11.703 ms | 11.47 | 0.54 | 17% | ok | 1.5x (38%) |
| blas3 | symm | f64 | 512x512x512 | 8.687 ms | 30.90 | 0.72 | 46% | ok | 3.3x (83%) |
| blas3 | syrk | f64 | 512x512x512 | 42.008 ms | 3.20 | 0.10 | 5% | ok | 1.0x (25%) |
| blas3 | syr2k | f64 | 512x512x512 | 78.979 ms | 3.40 | 0.08 | 5% | ok | 1.0x (26%) |
| blas3 | trmm | f64 | 512x512x512 | 11.071 ms | 12.12 | 0.47 | 18% | ok | 1.6x (40%) |
| blas3 | trsm | f64 | 512x512x512 | 11.199 ms | 11.98 | 0.47 | 18% | ok | 1.6x (40%) |
| blas3 | trsm | f32 | 512x512x512 | 6.745 ms | 19.90 | 0.39 | 15% | ok | 1.3x (32%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 28.290 ms | 0.19 | 1.50 | 6% | ok | 1.1x (27%) |
| factor | lu | f64 | 512x512 | 112.439 ms | 0.80 | 0.04 | 1% | ok | 0.2x (5%) |
| factor | lu | f32 | 512x512 | 97.358 ms | 0.92 | 0.02 | 1% | ok | 0.1x (3%) |
| factor | cholesky | f64 | 512x512 | 20.268 ms | 2.21 | 0.21 | 3% | ok | 1.1x (27%) |
| factor | cholesky | f32 | 512x512 | 17.564 ms | 2.55 | 0.12 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 20.545 ms | 2.18 | 0.20 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 60.987 ms | 0.73 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 184.780 ms | 0.97 | 0.02 | 1% | ok | 1.4x (34%) |
| factor | lq | f64 | 512x512 | 131.583 ms | 1.36 | 0.03 | 2% | ok | 0.5x (12%) |
| factor | hessenberg | f64 | 512x512 | 436.050 ms | 1.03 | 0.01 | 2% | ok | 0.8x (20%) |
| factor | householder | f64 | 512x512 | 0.939 ms | 1.12 | 4.46 | 18% | ok | 1.7x (42%) |
| factor | inv | f64 | 512x512 | 229.129 ms | 1.17 | 0.02 | 2% | ok | 0.6x (16%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 107.501 ms | 0.83 | 0.04 | 1% | ok | 0.2x (5%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.696 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.836 ms | 1.74 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 526.929 ms | 0.29 | 0.00 | 0% | ok | 1.2x (29%) |
| eigen | eigenvalue | f64 | 256x256 | 60.071 ms | 2.79 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 133.863 ms | 3.13 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.711 ms | 1.79 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 32.484 ms | 7.75 | 0.03 | 11% | ok | 1.5x (37%) |
| eigen | sylvester | f64 | 256x256 | 313.948 ms | 3.21 | 0.01 | 5% | ok | 1.1x (28%) |
| elementwise | exp | f64 | n=8388608 | 80.754 ms | 0.10 | 1.66 | 7% | ok | 1.1x (27%) |
| elementwise | log | f64 | n=8388608 | 98.588 ms | 0.09 | 1.36 | 5% | ok | 1.1x (28%) |
| elementwise | sin | f64 | n=8388608 | 133.527 ms | 0.06 | 1.01 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 64.853 ms | 0.13 | 2.07 | 8% | ok | 1.1x (28%) |
| elementwise | erf | f64 | n=8388608 | 105.065 ms | 0.08 | 1.28 | 5% | ok | 1.1x (28%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.542 ms | 1.70 | 17.66 | 71% | ok | 1.1x (27%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.994 ms | 1.31 | 13.65 | 55% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.347 ms | 6.60 | 13.87 | 56% | ok | 1.9x (47%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.909 ms | 0.27 | 0.97 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 9.458 ms | 0.21 | 1.76 | 7% | ok | 0.9x (22%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 164.559 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 55.491 ms | 0.20 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.452 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 92.784 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 126.331 ms | 0.18 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.130 ms | 0.57 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 186.887 ms | 0.01 | 0.00 | 0% | ok | 1.1x (27%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.685 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 11.836 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.628 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.403 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.203 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 97.389 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.261 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 25.966 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.258 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.594 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.125 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.170 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.264 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.672 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.379 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.984 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.589 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.100 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.415 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.345 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.778 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ildl | f64 | 128^2 grid | 2.669 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ssor | f64 | 128^2 grid | 0.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.627 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.973 ms | 1.33 | 14.86 | 60% | ok | 1.1x (26%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.299 ms | 1.14 | 12.76 | 51% | ok | 1.1x (27%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.677 ms | 0.98 | 10.96 | 44% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.592 ms | 1.08 | 8.17 | 33% | ok | 0.9x (22%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 209.923 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 243.995 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 150.429 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.799 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 3.687 ms | 4.55 | 18.20 | 73% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 11.186 ms | 24.00 | 0.37 | 36% | ok | 3.7x (93%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.380 ms | 7.05 | 7.49 | 30% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.793 ms | 6.01 | 24.02 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.360 ms | 12.34 | 24.67 | n/a | ok | 1.0x (26%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.100 ms | 7.99 | 7.99 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 3.812 ms | 4.40 | 4.40 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 5.006 ms | 53.63 | 0.63 | n/a | ok | 3.5x (87%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.816 ms | 55.74 | 0.44 | n/a | ok | 3.5x (87%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.705 ms | 57.06 | 0.33 | n/a | ok | 3.5x (88%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 6.002 ms | 44.72 | 0.26 | n/a | ok | 3.9x (98%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.234 ms | 20.28 | 0.12 | n/a | ok | 3.8x (96%) |

