## Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-42-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `73d11400a0e3` |
| FMA peak (ceiling) | f64 9.9 GFLOP/s, f32 19.5 GFLOP/s |
| ... hardware, SSE2 | f64 8.0, f32 14.8 GFLOP/s |
| ... this build, SSE4 | f64 9.9, f32 19.5 GFLOP/s |
| bandwidth | read 10.0, update 10.0, triad 8.0 GB/s (ceiling 10.0) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 102% | 110% | 0 |
| blas2 | 6 | 94% | 101% | 0 |
| blas3 | 10 | 74% | 98% | 0 |
| factor | 12 | 17% | 39% | 0 |
| eigen | 8 | 16% | 44% | 0 |
| elementwise | 5 | 9% | 14% | 0 |
| sparse | 5 | 56% | 104% | 0 |
| direct | 7 | 2% | 4% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 78% | 90% | 0 |
| mg | 1 | 78% | 78% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 74% | 89% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 17.391 ms | 0.96 | 7.72 | 77% | ok |
| blas1 | dot | f32 | n=8388608 | 6.172 ms | 2.72 | 10.87 | 108% | ok |
| blas1 | axpy | f64 | n=8388608 | 19.382 ms | 0.87 | 10.39 | 103% | ok |
| blas1 | axpy | f32 | n=8388608 | 9.143 ms | 1.83 | 11.01 | 110% | ok |
| blas1 | scale | f64 | n=8388608 | 13.137 ms | 0.64 | 10.22 | 102% | ok |
| blas1 | two_norm | f64 | n=8388608 | 6.156 ms | 2.73 | 10.90 | 109% | ok |
| blas1 | one_norm | f64 | n=8388608 | 10.097 ms | 0.83 | 6.65 | 66% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 13.581 ms | 0.62 | 4.94 | 49% | ok |
| blas1 | sum | f64 | n=8388608 | 10.074 ms | 0.83 | 6.66 | 66% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 279.873 ms | 0.66 | 0.48 | 7% | ok |
| blas2 | gemv | f64 | n=4096 | 13.337 ms | 2.52 | 10.07 | 100% | ok |
| blas2 | gemv | f32 | n=4096 | 6.590 ms | 5.09 | 10.19 | 101% | ok |
| blas2 | symv | f64 | n=4096 | 16.680 ms | 2.01 | 4.03 | 40% | ok |
| blas2 | trmv | f64 | n=4096 | 7.101 ms | 2.36 | 9.46 | 94% | ok |
| blas2 | trsv | f64 | n=4096 | 7.091 ms | 2.37 | 9.47 | 94% | ok |
| blas2 | ger | f64 | n=4096 | 28.529 ms | 1.18 | 9.41 | 94% | ok |
| blas3 | gemm | f64 | 512x512x512 | 28.668 ms | 9.36 | 0.22 | 94% | ok |
| blas3 | gemm | f32 | 512x512x512 | 14.044 ms | 19.11 | 0.22 | 98% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 17.554 ms | 7.65 | 0.36 | 77% | ok |
| blas3 | symm | f64 | 512x512x512 | 29.721 ms | 9.03 | 0.21 | 91% | ok |
| blas3 | syrk | f64 | 512x512x512 | 82.373 ms | 1.63 | 0.05 | 16% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 66.080 ms | 4.06 | 0.10 | 41% | ok |
| blas3 | trmm | f64 | 512x512x512 | 18.295 ms | 7.34 | 0.29 | 74% | ok |
| blas3 | trsm | f64 | 512x512x512 | 19.126 ms | 7.02 | 0.27 | 71% | ok |
| blas3 | trsm | f32 | 512x512x512 | 9.501 ms | 14.13 | 0.28 | 72% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 40.070 ms | 0.13 | 1.06 | 11% | ok |
| factor | lu | f64 | 512x512 | 31.290 ms | 2.86 | 0.13 | 29% | ok |
| factor | lu | f32 | 512x512 | 16.977 ms | 5.27 | 0.12 | 27% | ok |
| factor | cholesky | f64 | 512x512 | 27.160 ms | 1.65 | 0.15 | 17% | ok |
| factor | cholesky | f32 | 512x512 | 25.112 ms | 1.78 | 0.08 | 9% | ok |
| factor | ldlt | f64 | 512x512 | 26.609 ms | 1.68 | 0.16 | 17% | ok |
| factor | ldlt_bk | f64 | 512x512 | 143.159 ms | 0.31 | 0.03 | 3% | ok |
| factor | qr | f64 | 512x512 | 524.296 ms | 0.34 | 0.01 | 3% | ok |
| factor | lq | f64 | 512x512 | 76.016 ms | 2.35 | 0.06 | 24% | ok |
| factor | hessenberg | f64 | 512x512 | 637.825 ms | 0.70 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.365 ms | 0.31 | 1.25 | 12%\* | ok |
| factor | inv | f64 | 512x512 | 200.749 ms | 1.34 | 0.02 | 13% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 23.019 ms | 3.89 | 0.18 | 39% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 63.668 ms | 0.35 | 0.02 | 4% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 179.221 ms | 0.84 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1037.015 ms | 0.15 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 107.641 ms | 1.56 | 0.01 | 16% | ok |
| eigen | schur | f64 | 256x256 | 267.883 ms | 1.57 | 0.00 | 16% | ok |
| eigen | svd | f64 | 256x256 | 269.510 ms | 1.37 | 0.00 | 14% | ok |
| eigen | expm | f64 | 256x256 | 57.453 ms | 4.38 | 0.02 | 44% | ok |
| eigen | sylvester | f64 | 256x256 | 593.183 ms | 1.70 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 156.544 ms | 0.05 | 0.86 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 144.499 ms | 0.06 | 0.93 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 176.626 ms | 0.05 | 0.76 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 92.843 ms | 0.09 | 1.45 | 14% | ok |
| elementwise | erf | f64 | n=8388608 | 2799.777 ms | 0.00 | 0.05 | 0% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.597 ms | 1.01 | 10.49 | 104% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 4.281 ms | 0.61 | 6.36 | 63% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.696 ms | 2.67 | 5.61 | 56% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 32.682 ms | 0.10 | 0.35 | 4%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 6.036 ms | 0.33 | 2.76 | 27% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 246.604 ms | 0.05 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 72.679 ms | 0.15 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 27.540 ms | 0.41 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 120.700 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 160.899 ms | 0.14 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 51.642 ms | 0.43 | 0.01 | 4% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 232.798 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.443 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.741 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.657 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 41.641 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 82.764 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 114.845 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 66.817 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 33.552 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 39.374 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 38.276 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 67.700 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 76.907 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 43.064 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 45.014 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 43.152 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 46.333 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 62.590 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.137 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.728 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 34.412 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.016 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.886 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.077 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.245 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.722 ms | 0.70 | 7.88 | 78% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.252 ms | 0.80 | 9.02 | 90% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.977 ms | 0.66 | 7.37 | 73% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 37.260 ms | 1.03 | 7.80 | 78% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 146.073 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 198.300 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 162.211 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 16.832 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 7.535 ms | 2.23 | 8.91 | 89% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 64.258 ms | 4.18 | 0.07 | 42% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.400 ms | 6.99 | 7.43 | 74% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.190 ms | 2.71 | 10.84 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.101 ms | 5.41 | 10.82 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.175 ms | 7.71 | 7.71 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.735 ms | 9.67 | 9.67 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 19.562 ms | 13.72 | 0.16 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 22.994 ms | 11.67 | 0.09 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 22.744 ms | 11.80 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 16.758 ms | 16.02 | 0.09 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 17.379 ms | 15.45 | 0.09 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

