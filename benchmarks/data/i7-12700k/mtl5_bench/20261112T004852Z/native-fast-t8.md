## i7-12700k -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 600.7 GFLOP/s, f32 1201.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.7, f32 1201.3 GFLOP/s |
| ... this build, AVX2 | f64 600.7, f32 1201.6 GFLOP/s |
| bandwidth | read 83.1, update 78.4, triad 76.8 GB/s (ceiling 83.1) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 87% | 143% | 0 | 22% |
| blas2 | 10 | 47% | 147% | 0 | 13% |
| blas3 | 10 | 12% | 63% | 0 | 17% |
| factor | 12 | 1% | 12% | 0 | 17% |
| eigen | 8 | 1% | 4% | 0 | 15% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 276% | 0 | 13% |
| direct | 7 | 0% | 0% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 27% | 37% | 0 | 13% |
| mg | 1 | 23% | 23% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 49% | 69% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 65% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.418 ms | 11.83 | 94.64 | 114% | ok | 2.6x (32%) |
| blas1 | dot | f32 | n=8388608 | 0.566 ms | 29.65 | 118.60 | 143% | ok | 3.0x (37%) |
| blas1 | axpy | f64 | n=8388608 | 2.768 ms | 6.06 | 72.73 | 87% | ok | 1.7x (22%) |
| blas1 | axpy | f32 | n=8388608 | 1.365 ms | 12.29 | 73.73 | 89% | ok | 1.7x (21%) |
| blas1 | scale | f64 | n=8388608 | 1.899 ms | 4.42 | 70.68 | 85% | ok | 1.8x (23%) |
| blas1 | two_norm | f64 | n=8388608 | 0.572 ms | 29.33 | 117.32 | 141% | ok | 3.7x (46%) |
| blas1 | one_norm | f64 | n=8388608 | 3.977 ms | 2.11 | 16.87 | 20% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.937 ms | 1.21 | 9.67 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.922 ms | 2.14 | 17.11 | 21% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 69.246 ms | 2.67 | 1.94 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.404 ms | 23.91 | 95.67 | 115% | ok | 2.4x (30%) |
| blas2 | gemv | f32 | n=4096 | 0.550 ms | 60.97 | 122.01 | 147% | ok | 3.0x (38%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.204 ms | 15.22 | 60.92 | 73% | ok | 4.6x (57%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.934 ms | 35.92 | 71.87 | 86% | ok | 4.7x (59%) |
| blas2 | symv | f64 | n=4096 | 6.146 ms | 5.46 | 10.93 | 13% | ok | 1.0x (12%) |
| blas2 | trmv | f64 | n=4096 | 1.756 ms | 9.55 | 38.25 | 46% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 0.851 ms | 19.70 | 39.45 | 47% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 1.788 ms | 9.38 | 37.56 | 45% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.887 ms | 18.90 | 37.85 | 46% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.180 ms | 4.67 | 37.40 | 45% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 1.148 ms | 233.83 | 5.48 | 39% | ok | 3.3x (42%) |
| blas3 | gemm | f32 | 512x512x512 | 0.356 ms | 753.52 | 8.83 | 63% | ok | 5.3x (67%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.267 ms | 105.90 | 4.96 | 18% | ok | 2.1x (26%) |
| blas3 | symm | f64 | 512x512x512 | 1.131 ms | 237.33 | 5.56 | 40% | ok | 3.6x (45%) |
| blas3 | syrk | f64 | 512x512x512 | 10.586 ms | 12.68 | 0.40 | 2% | ok | 0.9x (12%) |
| blas3 | syr2k | f64 | 512x512x512 | 16.417 ms | 16.35 | 0.38 | 3% | ok | 1.2x (15%) |
| blas3 | trmm | f64 | 512x512x512 | 2.111 ms | 63.58 | 2.48 | 11% | ok | 1.3x (16%) |
| blas3 | trsm | f64 | 512x512x512 | 2.138 ms | 62.79 | 2.45 | 10% | ok | 1.3x (16%) |
| blas3 | trsm | f32 | 512x512x512 | 0.963 ms | 139.40 | 2.72 | 12% | ok | 1.4x (17%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.864 ms | 0.41 | 3.30 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 6.475 ms | 13.82 | 0.65 | 2% | ok | 2.1x (26%) |
| factor | lu | f32 | 512x512 | 3.489 ms | 25.64 | 0.60 | 2% | ok | 1.3x (17%) |
| factor | cholesky | f64 | 512x512 | 6.901 ms | 6.48 | 0.61 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 6.430 ms | 6.96 | 0.33 | 1% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 7.053 ms | 6.34 | 0.59 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 44.122 ms | 1.01 | 0.10 | 0% | ok | 1.1x (13%) |
| factor | qr | f64 | 512x512 | 128.318 ms | 1.39 | 0.03 | 0% | ok | 1.7x (21%) |
| factor | lq | f64 | 512x512 | 15.085 ms | 11.86 | 0.28 | 2% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 162.011 ms | 2.76 | 0.03 | 0% | ok | 1.6x (19%) |
| factor | householder | f64 | 512x512 | 0.414 ms | 2.53 | 10.12 | 12%\* | ok | 3.8x (47%) |
| factor | inv | f64 | 512x512 | 52.499 ms | 5.11 | 0.08 | 1% | ok | 1.0x (13%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 5.283 ms | 16.94 | 0.79 | 3% | ok | 0.9x (12%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 17.170 ms | 1.30 | 0.06 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 61.094 ms | 2.47 | 0.02 | 0% | ok | 1.1x (14%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 770.545 ms | 0.20 | 0.00 | 0% | ok | 1.4x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 31.096 ms | 5.40 | 0.03 | 1% | ok | 1.0x (12%) |
| eigen | schur | f64 | 256x256 | 81.369 ms | 5.15 | 0.01 | 1% | ok | 1.1x (14%) |
| eigen | svd | f64 | 256x256 | 92.046 ms | 4.01 | 0.01 | 1% | ok | 1.2x (15%) |
| eigen | expm | f64 | 256x256 | 10.200 ms | 24.67 | 0.10 | 4% | ok | 1.3x (16%) |
| eigen | sylvester | f64 | 256x256 | 190.317 ms | 5.29 | 0.01 | 1% | ok | 1.5x (19%) |
| elementwise | exp | f64 | n=8388608 | 38.232 ms | 0.22 | 3.51 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 45.405 ms | 0.18 | 2.96 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 58.664 ms | 0.14 | 2.29 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.181 ms | 0.30 | 4.76 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 200.699 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.119 ms | 22.07 | 229.57 | 276% ! | ok | 6.2x (78%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.914 ms | 2.86 | 29.80 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.394 ms | 30.05 | 63.17 | 76% | ok | 2.9x (37%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.691 ms | 0.69 | 2.45 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.407 ms | 1.43 | 11.83 | 14%\* | ok | 0.9x (12%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 72.156 ms | 0.16 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 27.833 ms | 0.40 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 9.507 ms | 1.18 | 0.08 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.276 ms | 0.57 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 54.950 ms | 0.41 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 17.167 ms | 1.30 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 88.296 ms | 0.02 | 0.00 | 0% | ok | 1.1x (14%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.371 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.947 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.224 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.939 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.182 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 28.053 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 20.836 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.379 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.622 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.526 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 20.084 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.552 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.403 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.478 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 14.785 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 15.729 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 17.830 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.033 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.528 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 11.869 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.057 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.994 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.699 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.966 ms | 2.71 | 30.35 | 37% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.285 ms | 2.04 | 22.82 | 27% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.482 ms | 1.77 | 19.78 | 24% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.891 ms | 2.58 | 19.52 | 23% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 27.827 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 39.117 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 38.610 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.495 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 1.879 ms | 8.93 | 35.72 | 43% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 0.910 ms | 294.83 | 4.61 | 49% | ok | 6.7x (83%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.310 ms | 54.15 | 57.53 | 69%\* | ok | 1.1x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.845 ms | 9.09 | 36.38 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.735 ms | 22.84 | 45.68 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.270 ms | 62.21 | 62.21 | n/a | ok | 1.1x (14%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.264 ms | 63.51 | 63.51 | n/a | ok | 1.1x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 0.768 ms | 349.35 | 4.09 | n/a | ok | 5.2x (65%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 0.820 ms | 327.49 | 2.56 | n/a | ok | 6.9x (86%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 0.801 ms | 335.01 | 1.96 | n/a | ok | 7.0x (87%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.331 ms | 809.98 | 4.75 | n/a | ok | 6.3x (79%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.302 ms | 888.07 | 5.20 | n/a | ok | 6.2x (78%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

