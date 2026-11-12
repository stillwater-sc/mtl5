## xeon-e5-2420v2 -- native, 6 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 59.0 GFLOP/s, f32 118.1 GFLOP/s |
| ... hardware, SSE2 | f64 59.0, f32 118.1 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.5 GFLOP/s |
| bandwidth | read 11.2, update 10.5, triad 8.4 GB/s (ceiling 11.2) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 55% | 108% | 0 | 83% |
| blas2 | 10 | 88% | 109% | 0 | 17% |
| blas3 | 10 | 8% | 39% | 0 | 17% |
| factor | 12 | 3% | 38% | 0 | 20% |
| eigen | 8 | 3% | 11% | 0 | 17% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 69% | 105% | 0 | 17% |
| direct | 7 | 0% | 1% | 0 | 17% |
| ordering | 4 | n/a | n/a | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 16% |
| smoother | 3 | 73% | 83% | 0 | 17% |
| mg | 1 | 74% | 74% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 22% | 68% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 90% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 32.893 ms | 0.51 | 4.08 | 36% | ok | 5.6x (93%) |
| blas1 | dot | f32 | n=8388608 | 5.533 ms | 3.03 | 12.13 | 108% | ok | 5.0x (83%) |
| blas1 | axpy | f64 | n=8388608 | 38.924 ms | 0.43 | 5.17 | 46% | ok | 5.6x (93%) |
| blas1 | axpy | f32 | n=8388608 | 16.377 ms | 1.02 | 6.15 | 55% | ok | 5.6x (93%) |
| blas1 | scale | f64 | n=8388608 | 12.079 ms | 0.69 | 11.11 | 99% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 32.925 ms | 0.51 | 2.04 | 18% | ok | 5.6x (94%) |
| blas1 | one_norm | f64 | n=8388608 | 9.769 ms | 0.86 | 6.87 | 61% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 12.997 ms | 0.65 | 5.16 | 46% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.761 ms | 0.86 | 6.88 | 61% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 280.150 ms | 0.66 | 0.48 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 10.963 ms | 3.06 | 12.25 | 109% | ok | 1.2x (20%) |
| blas2 | gemv | f32 | n=4096 | 5.509 ms | 6.09 | 12.19 | 109% | ok | 1.5x (25%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 10.956 ms | 3.06 | 12.26 | 109% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.503 ms | 6.10 | 12.20 | 109% | ok | 1.2x (20%) |
| blas2 | symv | f64 | n=4096 | 16.195 ms | 2.07 | 4.15 | 37% | ok | 1.0x (17%) |
| blas2 | trmv | f64 | n=4096 | 7.049 ms | 2.38 | 9.53 | 85% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 4.725 ms | 3.55 | 7.11 | 63% | ok | 1.0x (16%) |
| blas2 | trsv | f64 | n=4096 | 6.824 ms | 2.46 | 9.84 | 88% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 4.622 ms | 3.63 | 7.27 | 65% | ok | 0.9x (16%) |
| blas2 | ger | f64 | n=4096 | 27.852 ms | 1.20 | 9.64 | 86% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 11.545 ms | 23.25 | 0.54 | 39% | ok | 5.6x (93%) |
| blas3 | gemm | f32 | 512x512x512 | 6.427 ms | 41.77 | 0.49 | 35% | ok | 5.4x (90%) |
| blas3 | gemmt | f64 | 512x512x512 | 29.382 ms | 4.57 | 0.21 | 8% | ok | 1.0x (16%) |
| blas3 | symm | f64 | 512x512x512 | 60.597 ms | 4.43 | 0.10 | 8% | ok | 1.0x (16%) |
| blas3 | syrk | f64 | 512x512x512 | 38.015 ms | 3.53 | 0.11 | 6% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 512x512x512 | 65.983 ms | 4.07 | 0.10 | 7% | ok | 1.0x (16%) |
| blas3 | trmm | f64 | 512x512x512 | 32.931 ms | 4.08 | 0.16 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f64 | 512x512x512 | 32.632 ms | 4.11 | 0.16 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f32 | 512x512x512 | 16.407 ms | 8.18 | 0.16 | 7% | ok | 1.0x (16%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.690 ms | 0.14 | 1.10 | 10% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 23.852 ms | 3.75 | 0.18 | 6% | ok | 1.3x (22%) |
| factor | lu | f32 | 512x512 | 13.991 ms | 6.40 | 0.15 | 5% | ok | 1.2x (20%) |
| factor | cholesky | f64 | 512x512 | 25.283 ms | 1.77 | 0.17 | 3% | ok | 1.0x (16%) |
| factor | cholesky | f32 | 512x512 | 23.800 ms | 1.88 | 0.09 | 2% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 25.103 ms | 1.78 | 0.17 | 3% | ok | 1.0x (16%) |
| factor | ldlt_bk | f64 | 512x512 | 139.070 ms | 0.32 | 0.03 | 1% | ok | 1.1x (18%) |
| factor | qr | f64 | 512x512 | 323.622 ms | 0.55 | 0.01 | 1% | ok | 1.6x (27%) |
| factor | lq | f64 | 512x512 | 51.363 ms | 3.48 | 0.08 | 6% | ok | 1.4x (24%) |
| factor | hessenberg | f64 | 512x512 | 417.343 ms | 1.07 | 0.01 | 2% | ok | 1.5x (25%) |
| factor | householder | f64 | 512x512 | 0.985 ms | 1.07 | 4.26 | 38%\* | ok | 3.3x (54%) |
| factor | inv | f64 | 512x512 | 143.911 ms | 1.87 | 0.03 | 3% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 19.420 ms | 4.61 | 0.22 | 8% | ok | 1.1x (19%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 63.741 ms | 0.35 | 0.02 | 1% | ok | 1.0x (16%) |
| eigen | eigen_symmetric | f64 | 256x256 | 172.253 ms | 0.88 | 0.01 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 996.068 ms | 0.15 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 107.492 ms | 1.56 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 264.391 ms | 1.59 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 302.720 ms | 1.22 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 37.541 ms | 6.70 | 0.03 | 11% | ok | 2.3x (38%) |
| eigen | sylvester | f64 | 256x256 | 582.347 ms | 1.73 | 0.00 | 3% | ok | 1.0x (17%) |
| elementwise | exp | f64 | n=8388608 | 151.636 ms | 0.06 | 0.89 | 8% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 142.557 ms | 0.06 | 0.94 | 8% | ok | 1.1x (18%) |
| elementwise | sin | f64 | n=8388608 | 172.453 ms | 0.05 | 0.78 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 92.119 ms | 0.09 | 1.46 | 13% | ok | 1.0x (16%) |
| elementwise | erf | f64 | n=8388608 | 2481.949 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.303 ms | 1.14 | 11.82 | 105% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.112 ms | 0.84 | 8.75 | 78% | ok | 1.0x (16%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.353 ms | 3.69 | 7.76 | 69% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.053 ms | 0.20 | 0.72 | 6%\* | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.770 ms | 0.35 | 2.89 | 26% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 273.553 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 69.419 ms | 0.16 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.388 ms | 0.42 | 0.03 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 115.723 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 153.331 ms | 0.15 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 48.018 ms | 0.47 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 231.533 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.909 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.274 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.632 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.676 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 233.678 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 394.637 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 126.593 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 101.729 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 121.185 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 79.198 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 520.399 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 174.551 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.725 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 137.763 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 139.507 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 171.042 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 251.140 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.129 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.471 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | ilut | f64 | 128^2 grid | 33.093 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 2.919 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | ildl | f64 | 128^2 grid | 2.767 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | ssor | f64 | 128^2 grid | 0.079 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.182 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.587 ms | 0.73 | 8.18 | 73% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.153 ms | 0.83 | 9.30 | 83% | ok | 1.0x (16%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.808 ms | 0.69 | 7.70 | 69% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 34.898 ms | 1.10 | 8.33 | 74% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 146.722 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 188.201 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 146.308 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.820 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 188.901 ms | 0.09 | 0.36 | 3% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 20.564 ms | 13.05 | 0.20 | 22% | ok | 5.5x (92%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.336 ms | 7.18 | 7.63 | 68% | ok | 1.0x (16%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.485 ms | 2.59 | 10.35 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.108 ms | 5.40 | 10.79 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.081 ms | 3.30 | 3.30 | n/a | ok | 1.0x (16%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.987 ms | 3.36 | 3.36 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 10.387 ms | 25.84 | 0.30 | n/a | ok | 5.6x (93%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 8.249 ms | 32.54 | 0.25 | n/a | ok | 5.5x (91%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.348 ms | 61.73 | 0.36 | n/a | ok | 5.5x (91%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.350 ms | 61.71 | 0.36 | n/a | ok | 5.4x (91%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.148 ms | 20.42 | 0.12 | n/a | ok | 5.4x (90%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

