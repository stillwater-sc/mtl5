## xeon-e5-2420v2 -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 21.1 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 21.1 GFLOP/s |
| ... this build, scalar | f64 0.1, f32 0.7 GFLOP/s |
| bandwidth | read 10.2, update 10.3, triad 8.2 GB/s (ceiling 10.3) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 24% | 104% | 0 |
| blas2 | 10 | 92% | 104% | 0 |
| blas3 | 10 | 162% | 181% | 0 |
| factor | 12 | 53% | 92% | 0 |
| eigen | 8 | 50% | 118% | 0 |
| elementwise | 5 | 9% | 14% | 0 |
| sparse | 5 | 56% | 109% | 0 |
| direct | 7 | 2% | 4% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 80% | 92% | 0 |
| mg | 1 | 79% | 79% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 22% | 75% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 182.373 ms | 0.09 | 0.74 | 7% | ok |
| blas1 | dot | f32 | n=8388608 | 26.991 ms | 0.62 | 2.49 | 24% | ok |
| blas1 | axpy | f64 | n=8388608 | 211.121 ms | 0.08 | 0.95 | 9% | ok |
| blas1 | axpy | f32 | n=8388608 | 91.224 ms | 0.18 | 1.10 | 11% | ok |
| blas1 | scale | f64 | n=8388608 | 12.487 ms | 0.67 | 10.75 | 104% | ok |
| blas1 | two_norm | f64 | n=8388608 | 186.301 ms | 0.09 | 0.36 | 3% | ok |
| blas1 | one_norm | f64 | n=8388608 | 9.742 ms | 0.86 | 6.89 | 67% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 13.071 ms | 0.64 | 5.13 | 50% | ok |
| blas1 | sum | f64 | n=8388608 | 9.740 ms | 0.86 | 6.89 | 67% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 277.765 ms | 0.66 | 0.48 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 14.383 ms | 2.33 | 9.34 | 91% | ok |
| blas2 | gemv | f32 | n=4096 | 6.270 ms | 5.35 | 10.71 | 104% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 12.972 ms | 2.59 | 10.35 | 100% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.622 ms | 5.07 | 10.14 | 98% | ok |
| blas2 | symv | f64 | n=4096 | 7.009 ms | 4.79 | 9.58 | 93% | ok |
| blas2 | trmv | f64 | n=4096 | 7.619 ms | 2.20 | 8.82 | 86% | ok |
| blas2 | trmv | f32 | n=4096 | 3.723 ms | 4.51 | 9.02 | 88% | ok |
| blas2 | trsv | f64 | n=4096 | 7.563 ms | 2.22 | 8.88 | 86% | ok |
| blas2 | trsv | f32 | n=4096 | 3.534 ms | 4.75 | 9.50 | 92% | ok |
| blas2 | ger | f64 | n=4096 | 28.726 ms | 1.17 | 9.35 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 14.096 ms | 19.04 | 0.45 | 181% | ok |
| blas3 | gemm | f32 | 512x512x512 | 8.550 ms | 31.40 | 0.37 | 149% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 8.897 ms | 15.09 | 0.71 | 144% | ok |
| blas3 | symm | f64 | 512x512x512 | 14.236 ms | 18.86 | 0.44 | 180% | ok |
| blas3 | syrk | f64 | 512x512x512 | 7.899 ms | 16.99 | 0.53 | 162% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 14.970 ms | 17.93 | 0.42 | 171% | ok |
| blas3 | trmm | f64 | 512x512x512 | 7.486 ms | 17.93 | 0.70 | 171% | ok |
| blas3 | trsm | f64 | 512x512x512 | 8.187 ms | 16.39 | 0.64 | 156% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.534 ms | 24.25 | 0.47 | 115% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.995 ms | 0.14 | 1.09 | 11% | ok |
| factor | lu | f64 | 512x512 | 9.244 ms | 9.68 | 0.45 | 92% | ok |
| factor | lu | f32 | 512x512 | 6.510 ms | 13.75 | 0.32 | 65% | ok |
| factor | cholesky | f64 | 512x512 | 5.818 ms | 7.69 | 0.72 | 73% | ok |
| factor | cholesky | f32 | 512x512 | 4.006 ms | 11.17 | 0.52 | 53% | ok |
| factor | ldlt | f64 | 512x512 | 25.980 ms | 1.72 | 0.16 | 16% | ok |
| factor | ldlt_bk | f64 | 512x512 | 150.052 ms | 0.30 | 0.03 | 3% | ok |
| factor | qr | f64 | 512x512 | 20.094 ms | 8.91 | 0.21 | 85% | ok |
| factor | lq | f64 | 512x512 | 74.552 ms | 2.40 | 0.06 | 23% | ok |
| factor | hessenberg | f64 | 512x512 | 638.219 ms | 0.70 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.275 ms | 0.32 | 1.28 | 12%\* | ok |
| factor | inv | f64 | 512x512 | 131.013 ms | 2.05 | 0.03 | 20% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.818 ms | 6.98 | 0.33 | 67% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 6.216 ms | 3.60 | 0.17 | 34% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 173.988 ms | 0.87 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 985.877 ms | 0.15 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 53.600 ms | 3.13 | 0.02 | 30% | ok |
| eigen | schur | f64 | 256x256 | 72.201 ms | 5.81 | 0.01 | 55% | ok |
| eigen | svd | f64 | 256x256 | 29.866 ms | 12.36 | 0.04 | 118% | ok |
| eigen | expm | f64 | 256x256 | 35.951 ms | 7.00 | 0.03 | 67% | ok |
| eigen | sylvester | f64 | 256x256 | 190.908 ms | 5.27 | 0.01 | 50% | ok |
| elementwise | exp | f64 | n=8388608 | 151.697 ms | 0.06 | 0.88 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 141.023 ms | 0.06 | 0.95 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 170.507 ms | 0.05 | 0.79 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 90.926 ms | 0.09 | 1.48 | 14% | ok |
| elementwise | erf | f64 | n=8388608 | 2480.361 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.419 ms | 1.08 | 11.26 | 109% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.068 ms | 0.85 | 8.87 | 86% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.215 ms | 2.75 | 5.79 | 56% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.219 ms | 0.20 | 0.71 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.743 ms | 0.35 | 2.90 | 28% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 273.354 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 70.781 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.259 ms | 0.43 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 116.137 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 156.041 ms | 0.14 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 48.483 ms | 0.46 | 0.02 | 4% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 231.371 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.920 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.187 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.634 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 43.240 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 232.919 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 393.891 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 126.478 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 102.557 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 121.421 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 79.411 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 518.317 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 174.188 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 90.586 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 135.698 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 137.360 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 170.407 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 248.892 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.128 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.436 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 32.890 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.826 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.723 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.077 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 3.423 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.539 ms | 0.74 | 8.29 | 80% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.082 ms | 0.85 | 9.52 | 92% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.797 ms | 0.69 | 7.72 | 75% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.525 ms | 1.08 | 8.18 | 79% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 141.712 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 148.924 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 144.331 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.539 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 187.653 ms | 0.09 | 0.36 | 3% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 114.436 ms | 2.35 | 0.04 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.304 ms | 7.28 | 7.74 | 75% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.501 ms | 2.58 | 10.32 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.157 ms | 5.31 | 10.63 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.982 ms | 3.37 | 3.37 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 5.097 ms | 3.29 | 3.29 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 57.575 ms | 4.66 | 0.05 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 45.060 ms | 5.96 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 23.740 ms | 11.31 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 23.451 ms | 11.45 | 0.07 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 70.283 ms | 3.82 | 0.02 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

