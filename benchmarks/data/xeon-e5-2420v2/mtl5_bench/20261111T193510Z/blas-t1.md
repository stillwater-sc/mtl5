## xeon-e5-2420v2 -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 21.0 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 21.0 GFLOP/s |
| ... this build, scalar | f64 0.1, f32 0.7 GFLOP/s |
| bandwidth | read 10.3, update 10.2, triad 8.1 GB/s (ceiling 10.3) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 24% | 104% | 0 |
| blas2 | 10 | 94% | 105% | 0 |
| blas3 | 10 | 162% | 183% | 0 |
| factor | 12 | 54% | 92% | 0 |
| eigen | 8 | 51% | 119% | 0 |
| elementwise | 5 | 9% | 14% | 0 |
| sparse | 5 | 56% | 108% | 0 |
| direct | 7 | 2% | 4% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 79% | 92% | 0 |
| mg | 1 | 79% | 79% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 23% | 73% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 180.998 ms | 0.09 | 0.74 | 7% | ok |
| blas1 | dot | f32 | n=8388608 | 27.095 ms | 0.62 | 2.48 | 24% | ok |
| blas1 | axpy | f64 | n=8388608 | 213.023 ms | 0.08 | 0.95 | 9% | ok |
| blas1 | axpy | f32 | n=8388608 | 91.531 ms | 0.18 | 1.10 | 11% | ok |
| blas1 | scale | f64 | n=8388608 | 12.619 ms | 0.66 | 10.64 | 104% | ok |
| blas1 | two_norm | f64 | n=8388608 | 183.955 ms | 0.09 | 0.36 | 4% | ok |
| blas1 | one_norm | f64 | n=8388608 | 9.705 ms | 0.86 | 6.91 | 67% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 13.055 ms | 0.64 | 5.14 | 50% | ok |
| blas1 | sum | f64 | n=8388608 | 9.760 ms | 0.86 | 6.88 | 67% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 277.771 ms | 0.66 | 0.48 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 14.263 ms | 2.35 | 9.41 | 92% | ok |
| blas2 | gemv | f32 | n=4096 | 6.237 ms | 5.38 | 10.77 | 105% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 12.939 ms | 2.59 | 10.38 | 101% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.616 ms | 5.07 | 10.15 | 99% | ok |
| blas2 | symv | f64 | n=4096 | 6.948 ms | 4.83 | 9.67 | 94% | ok |
| blas2 | trmv | f64 | n=4096 | 7.578 ms | 2.21 | 8.86 | 86% | ok |
| blas2 | trmv | f32 | n=4096 | 3.689 ms | 4.55 | 9.10 | 89% | ok |
| blas2 | trsv | f64 | n=4096 | 7.464 ms | 2.25 | 9.00 | 88% | ok |
| blas2 | trsv | f32 | n=4096 | 3.456 ms | 4.85 | 9.72 | 95% | ok |
| blas2 | ger | f64 | n=4096 | 28.043 ms | 1.20 | 9.57 | 93% | ok |
| blas3 | gemm | f64 | 512x512x512 | 14.028 ms | 19.14 | 0.45 | 183% | ok |
| blas3 | gemm | f32 | 512x512x512 | 8.149 ms | 32.94 | 0.39 | 157% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 8.734 ms | 15.37 | 0.72 | 147% | ok |
| blas3 | symm | f64 | 512x512x512 | 14.089 ms | 19.05 | 0.45 | 182% | ok |
| blas3 | syrk | f64 | 512x512x512 | 7.878 ms | 17.04 | 0.53 | 162% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 15.034 ms | 17.85 | 0.42 | 170% | ok |
| blas3 | trmm | f64 | 512x512x512 | 7.459 ms | 17.99 | 0.70 | 172% | ok |
| blas3 | trsm | f64 | 512x512x512 | 8.166 ms | 16.44 | 0.64 | 157% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.319 ms | 25.23 | 0.49 | 120% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.615 ms | 0.14 | 1.10 | 11% | ok |
| factor | lu | f64 | 512x512 | 9.239 ms | 9.68 | 0.45 | 92% | ok |
| factor | lu | f32 | 512x512 | 6.431 ms | 13.91 | 0.33 | 66% | ok |
| factor | cholesky | f64 | 512x512 | 5.891 ms | 7.59 | 0.71 | 72% | ok |
| factor | cholesky | f32 | 512x512 | 3.928 ms | 11.39 | 0.53 | 54% | ok |
| factor | ldlt | f64 | 512x512 | 25.363 ms | 1.76 | 0.17 | 17% | ok |
| factor | ldlt_bk | f64 | 512x512 | 146.733 ms | 0.30 | 0.03 | 3% | ok |
| factor | qr | f64 | 512x512 | 19.966 ms | 8.96 | 0.21 | 85% | ok |
| factor | lq | f64 | 512x512 | 74.673 ms | 2.40 | 0.06 | 23% | ok |
| factor | hessenberg | f64 | 512x512 | 630.391 ms | 0.71 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.230 ms | 0.33 | 1.30 | 13%\* | ok |
| factor | inv | f64 | 512x512 | 129.311 ms | 2.08 | 0.03 | 20% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.848 ms | 6.96 | 0.33 | 66% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 6.239 ms | 3.59 | 0.17 | 34% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 174.177 ms | 0.87 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 999.183 ms | 0.15 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 53.177 ms | 3.15 | 0.02 | 30% | ok |
| eigen | schur | f64 | 256x256 | 72.095 ms | 5.82 | 0.01 | 55% | ok |
| eigen | svd | f64 | 256x256 | 29.500 ms | 12.51 | 0.04 | 119% | ok |
| eigen | expm | f64 | 256x256 | 35.728 ms | 7.04 | 0.03 | 67% | ok |
| eigen | sylvester | f64 | 256x256 | 189.501 ms | 5.31 | 0.01 | 51% | ok |
| elementwise | exp | f64 | n=8388608 | 150.786 ms | 0.06 | 0.89 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 141.186 ms | 0.06 | 0.95 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 172.071 ms | 0.05 | 0.78 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 90.985 ms | 0.09 | 1.48 | 14% | ok |
| elementwise | erf | f64 | n=8388608 | 2476.708 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.458 ms | 1.06 | 11.08 | 108% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.097 ms | 0.85 | 8.79 | 86% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.204 ms | 2.75 | 5.79 | 56% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.224 ms | 0.20 | 0.71 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.776 ms | 0.35 | 2.88 | 28% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 273.021 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 69.556 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.648 ms | 0.42 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 119.204 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 155.239 ms | 0.14 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 48.043 ms | 0.47 | 0.02 | 4% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 232.139 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.135 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.372 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.625 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 41.388 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 229.514 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 396.802 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 126.671 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 101.910 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 121.217 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 78.757 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 516.955 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 176.476 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.619 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 139.590 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 140.118 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 170.930 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 248.486 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.128 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.456 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 33.044 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.915 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.936 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.119 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 3.389 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.623 ms | 0.72 | 8.09 | 79% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.123 ms | 0.84 | 9.39 | 92% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.828 ms | 0.68 | 7.66 | 75% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.710 ms | 1.08 | 8.14 | 79% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 142.049 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 148.131 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 144.134 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.513 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 188.768 ms | 0.09 | 0.36 | 3% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 113.542 ms | 2.36 | 0.04 | 23% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.367 ms | 7.09 | 7.53 | 73% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.620 ms | 2.53 | 10.14 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.112 ms | 5.39 | 10.78 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.015 ms | 3.35 | 3.35 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.999 ms | 3.36 | 3.36 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 57.681 ms | 4.65 | 0.05 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 45.933 ms | 5.84 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 23.593 ms | 11.38 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 23.568 ms | 11.39 | 0.07 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 70.878 ms | 3.79 | 0.02 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

