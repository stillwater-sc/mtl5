## arduino-ventuno-q -- native-fast, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON_WITHOUT_AES |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 67.8 GFLOP/s, f32 135.6 GFLOP/s |
| ... hardware, NEON | f64 67.8, f32 135.5 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 67.4, f32 135.6 GFLOP/s |
| bandwidth | read 23.0, update 24.8, triad 22.5 GB/s (ceiling 24.8) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 90% | 109% | 0 | 28% |
| blas2 | 10 | 57% | 90% | 0 | 27% |
| blas3 | 10 | 18% | 51% | 0 | 40% |
| factor | 12 | 2% | 16% | 0 | 28% |
| eigen | 8 | 4% | 12% | 0 | 28% |
| elementwise | 5 | 5% | 8% | 0 | 27% |
| sparse | 5 | 55% | 87% | 0 | 28% |
| direct | 7 | 0% | 1% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 29% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 52% | 59% | 0 | 27% |
| mg | 1 | 33% | 33% | 0 | 22% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 35% | 74% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 85% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 6.001 ms | 2.80 | 22.37 | 90% | ok | 1.0x (24%) |
| blas1 | dot | f32 | n=8388608 | 3.137 ms | 5.35 | 21.40 | 86% | ok | 0.9x (23%) |
| blas1 | axpy | f64 | n=8388608 | 8.208 ms | 2.04 | 24.53 | 99% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 4.249 ms | 3.95 | 23.69 | 96% | ok | 0.9x (23%) |
| blas1 | scale | f64 | n=8388608 | 4.971 ms | 1.69 | 27.00 | 109% | ok | 1.1x (27%) |
| blas1 | two_norm | f64 | n=8388608 | 2.872 ms | 5.84 | 23.37 | 94% | ok | 1.1x (28%) |
| blas1 | one_norm | f64 | n=8388608 | 7.510 ms | 1.12 | 8.94 | 36% | ok | 1.1x (29%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.554 ms | 0.54 | 4.31 | 17% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.808 ms | 1.07 | 8.60 | 35% | ok | 1.1x (28%) |
| blas1 | dot_k3 | f64 | n=8388608 | 179.772 ms | 1.03 | 0.75 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 6.591 ms | 5.09 | 20.38 | 82% | ok | 1.3x (31%) |
| blas2 | gemv | f32 | n=4096 | 2.997 ms | 11.20 | 22.40 | 90% | ok | 1.4x (34%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 22.955 ms | 1.46 | 5.85 | 24% | ok | 2.1x (54%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 12.495 ms | 2.69 | 5.37 | 22% | ok | 2.1x (51%) |
| blas2 | symv | f64 | n=4096 | 15.586 ms | 2.15 | 4.31 | 17% | ok | 1.1x (27%) |
| blas2 | trmv | f64 | n=4096 | 4.778 ms | 3.51 | 14.06 | 57% | ok | 1.0x (26%) |
| blas2 | trmv | f32 | n=4096 | 2.543 ms | 6.60 | 13.21 | 53% | ok | 1.0x (25%) |
| blas2 | trsv | f64 | n=4096 | 4.783 ms | 3.51 | 14.04 | 57% | ok | 1.0x (25%) |
| blas2 | trsv | f32 | n=4096 | 2.412 ms | 6.96 | 13.92 | 56% | ok | 1.1x (27%) |
| blas2 | ger | f64 | n=4096 | 14.838 ms | 2.26 | 18.10 | 73% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 7.818 ms | 34.34 | 0.80 | 51% | ok | 3.6x (90%) |
| blas3 | gemm | f32 | 512x512x512 | 4.213 ms | 63.71 | 0.75 | 47% | ok | 3.3x (82%) |
| blas3 | gemmt | f64 | 512x512x512 | 10.402 ms | 12.90 | 0.60 | 19% | ok | 1.7x (43%) |
| blas3 | symm | f64 | 512x512x512 | 8.675 ms | 30.94 | 0.73 | 46% | ok | 3.4x (84%) |
| blas3 | syrk | f64 | 512x512x512 | 36.562 ms | 3.67 | 0.11 | 5% | ok | 1.1x (28%) |
| blas3 | syr2k | f64 | 512x512x512 | 77.579 ms | 3.46 | 0.08 | 5% | ok | 1.0x (26%) |
| blas3 | trmm | f64 | 512x512x512 | 10.954 ms | 12.25 | 0.48 | 18% | ok | 1.6x (40%) |
| blas3 | trsm | f64 | 512x512x512 | 11.394 ms | 11.78 | 0.46 | 17% | ok | 1.6x (40%) |
| blas3 | trsm | f32 | 512x512x512 | 6.689 ms | 20.06 | 0.39 | 15% | ok | 1.3x (32%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 29.173 ms | 0.18 | 1.46 | 6% | ok | 1.1x (27%) |
| factor | lu | f64 | 512x512 | 109.071 ms | 0.82 | 0.04 | 1% | ok | 0.2x (6%) |
| factor | lu | f32 | 512x512 | 93.127 ms | 0.96 | 0.02 | 1% | ok | 0.1x (4%) |
| factor | cholesky | f64 | 512x512 | 20.069 ms | 2.23 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 17.536 ms | 2.55 | 0.12 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 20.330 ms | 2.20 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 60.247 ms | 0.74 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 198.816 ms | 0.90 | 0.02 | 1% | ok | 1.3x (31%) |
| factor | lq | f64 | 512x512 | 126.414 ms | 1.42 | 0.03 | 2% | ok | 0.5x (12%) |
| factor | hessenberg | f64 | 512x512 | 440.059 ms | 1.02 | 0.01 | 1% | ok | 0.8x (20%) |
| factor | householder | f64 | 512x512 | 1.033 ms | 1.02 | 4.06 | 16% | ok | 1.5x (38%) |
| factor | inv | f64 | 512x512 | 224.090 ms | 1.20 | 0.02 | 2% | ok | 0.6x (16%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 108.003 ms | 0.83 | 0.04 | 1% | ok | 0.2x (5%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.471 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.506 ms | 1.75 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 525.246 ms | 0.29 | 0.00 | 0% | ok | 1.2x (29%) |
| eigen | eigenvalue | f64 | 256x256 | 59.907 ms | 2.80 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 133.826 ms | 3.13 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.719 ms | 1.79 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 31.967 ms | 7.87 | 0.03 | 12% | ok | 1.5x (38%) |
| eigen | sylvester | f64 | 256x256 | 312.866 ms | 3.22 | 0.01 | 5% | ok | 1.1x (28%) |
| elementwise | exp | f64 | n=8388608 | 81.760 ms | 0.10 | 1.64 | 7% | ok | 1.1x (27%) |
| elementwise | log | f64 | n=8388608 | 100.712 ms | 0.08 | 1.33 | 5% | ok | 1.1x (28%) |
| elementwise | sin | f64 | n=8388608 | 135.791 ms | 0.06 | 0.99 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 67.033 ms | 0.13 | 2.00 | 8% | ok | 1.1x (27%) |
| elementwise | erf | f64 | n=8388608 | 107.826 ms | 0.08 | 1.24 | 5% | ok | 1.1x (27%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.265 ms | 2.07 | 21.52 | 87% | ok | 1.3x (32%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.987 ms | 1.32 | 13.71 | 55% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.438 ms | 6.51 | 13.68 | 55% | ok | 1.9x (47%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.814 ms | 0.28 | 0.97 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 9.438 ms | 0.21 | 1.76 | 7% | ok | 0.9x (22%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 164.695 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 55.546 ms | 0.20 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.573 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 92.323 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 126.886 ms | 0.18 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.188 ms | 0.57 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 185.829 ms | 0.01 | 0.00 | 0% | ok | 1.1x (27%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.659 ms | 0.00 | 0.00 | n/a | ok | 1.1x (29%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.239 ms | 0.00 | 0.00 | n/a | ok | 1.2x (29%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.636 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 39.374 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.234 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 97.497 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.728 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.066 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.456 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.710 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.007 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.268 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.429 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.708 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.446 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.720 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.625 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.102 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.437 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilut | f64 | 128^2 grid | 27.683 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.782 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.695 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.571 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.017 ms | 1.30 | 14.54 | 59% | ok | 1.0x (26%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.278 ms | 1.15 | 12.88 | 52% | ok | 1.1x (27%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.662 ms | 0.98 | 11.02 | 44% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.794 ms | 1.07 | 8.12 | 33% | ok | 0.9x (22%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 210.080 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 244.974 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 149.863 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.893 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 3.680 ms | 4.56 | 18.24 | 74% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 11.197 ms | 23.97 | 0.37 | 35% | ok | 3.7x (93%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.357 ms | 7.12 | 7.56 | 31% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.784 ms | 6.03 | 24.11 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.361 ms | 12.33 | 24.66 | n/a | ok | 1.0x (26%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.111 ms | 7.95 | 7.95 | n/a | ok | 1.1x (27%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 3.810 ms | 4.40 | 4.40 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 5.067 ms | 52.97 | 0.62 | n/a | ok | 3.4x (85%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.867 ms | 55.15 | 0.43 | n/a | ok | 3.4x (85%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.289 ms | 62.59 | 0.37 | n/a | ok | 3.9x (97%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 6.509 ms | 41.24 | 0.24 | n/a | ok | 3.6x (91%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.316 ms | 20.16 | 0.12 | n/a | ok | 3.8x (96%) |

