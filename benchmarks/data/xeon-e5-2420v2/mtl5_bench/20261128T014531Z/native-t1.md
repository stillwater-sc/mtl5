## xeon-e5-2420v2 -- native, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 10.6 GFLOP/s, f32 21.0 GFLOP/s |
| ... hardware, SSE2 | f64 10.6, f32 21.0 GFLOP/s |
| ... this build, scalar | f64 0.1, f32 0.7 GFLOP/s |
| bandwidth | read 10.2, update 10.2, triad 8.1 GB/s (ceiling 10.2) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 24% | 106% | 0 |
| blas2 | 10 | 92% | 101% | 0 |
| blas3 | 10 | 38% | 39% | 0 |
| factor | 12 | 17% | 39% | 0 |
| eigen | 8 | 15% | 28% | 0 |
| elementwise | 5 | 9% | 15% | 0 |
| sparse | 5 | 57% | 105% | 0 |
| direct | 19 | 3% | 36% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 19% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 80% | 87% | 0 |
| mg | 1 | 79% | 79% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 22% | 75% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 183.083 ms | 0.09 | 0.73 | 7% | ok |
| blas1 | dot | f32 | n=8388608 | 27.800 ms | 0.60 | 2.41 | 24% | ok |
| blas1 | axpy | f64 | n=8388608 | 212.486 ms | 0.08 | 0.95 | 9% | ok |
| blas1 | axpy | f32 | n=8388608 | 91.388 ms | 0.18 | 1.10 | 11% | ok |
| blas1 | scale | f64 | n=8388608 | 12.402 ms | 0.68 | 10.82 | 106% | ok |
| blas1 | two_norm | f64 | n=8388608 | 184.651 ms | 0.09 | 0.36 | 4% | ok |
| blas1 | one_norm | f64 | n=8388608 | 9.703 ms | 0.86 | 6.92 | 68% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 13.040 ms | 0.64 | 5.15 | 50% | ok |
| blas1 | sum | f64 | n=8388608 | 9.744 ms | 0.86 | 6.89 | 67% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 276.407 ms | 0.67 | 0.49 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 13.546 ms | 2.48 | 9.91 | 97% | ok |
| blas2 | gemv | f32 | n=4096 | 8.528 ms | 3.93 | 7.87 | 77% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 13.248 ms | 2.53 | 10.14 | 99% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.505 ms | 5.16 | 10.32 | 101% | ok |
| blas2 | symv | f64 | n=4096 | 12.117 ms | 2.77 | 5.54 | 54% | ok |
| blas2 | trmv | f64 | n=4096 | 7.177 ms | 2.34 | 9.36 | 92% | ok |
| blas2 | trmv | f32 | n=4096 | 4.623 ms | 3.63 | 7.27 | 71% | ok |
| blas2 | trsv | f64 | n=4096 | 6.955 ms | 2.41 | 9.66 | 94% | ok |
| blas2 | trsv | f32 | n=4096 | 4.414 ms | 3.80 | 7.61 | 74% | ok |
| blas2 | ger | f64 | n=4096 | 28.484 ms | 1.18 | 9.43 | 92% | ok |
| blas3 | gemm | f64 | 512x512x512 | 65.157 ms | 4.12 | 0.10 | 39% | ok |
| blas3 | gemm | f32 | 512x512x512 | 34.578 ms | 7.76 | 0.09 | 37% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 36.231 ms | 3.70 | 0.17 | 35% | ok |
| blas3 | symm | f64 | 512x512x512 | 66.903 ms | 4.01 | 0.09 | 38% | ok |
| blas3 | syrk | f64 | 512x512x512 | 38.166 ms | 3.52 | 0.11 | 33% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 66.084 ms | 4.06 | 0.10 | 38% | ok |
| blas3 | trmm | f64 | 512x512x512 | 33.020 ms | 4.06 | 0.16 | 39% | ok |
| blas3 | trsm | f64 | 512x512x512 | 33.867 ms | 3.96 | 0.15 | 38% | ok |
| blas3 | trsm | f32 | 512x512x512 | 16.359 ms | 8.20 | 0.16 | 39% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 40.518 ms | 0.13 | 1.05 | 10% | ok |
| factor | lu | f64 | 512x512 | 30.984 ms | 2.89 | 0.14 | 27% | ok |
| factor | lu | f32 | 512x512 | 16.232 ms | 5.51 | 0.13 | 26% | ok |
| factor | cholesky | f64 | 512x512 | 25.150 ms | 1.78 | 0.17 | 17% | ok |
| factor | cholesky | f32 | 512x512 | 23.524 ms | 1.90 | 0.09 | 9% | ok |
| factor | ldlt | f64 | 512x512 | 25.880 ms | 1.73 | 0.16 | 16% | ok |
| factor | ldlt_bk | f64 | 512x512 | 13.324 ms | 3.36 | 0.31 | 32% | ok |
| factor | qr | f64 | 512x512 | 503.185 ms | 0.36 | 0.01 | 3% | ok |
| factor | lq | f64 | 512x512 | 73.786 ms | 2.43 | 0.06 | 23% | ok |
| factor | hessenberg | f64 | 512x512 | 610.949 ms | 0.73 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.304 ms | 0.32 | 1.27 | 12%\* | ok |
| factor | inv | f64 | 512x512 | 153.051 ms | 1.75 | 0.03 | 17% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 21.591 ms | 4.14 | 0.19 | 39% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 62.575 ms | 0.36 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 170.164 ms | 0.89 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 963.095 ms | 0.16 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 107.434 ms | 1.56 | 0.01 | 15% | ok |
| eigen | schur | f64 | 256x256 | 267.341 ms | 1.57 | 0.00 | 15% | ok |
| eigen | svd | f64 | 256x256 | 304.776 ms | 1.21 | 0.00 | 11% | ok |
| eigen | expm | f64 | 256x256 | 84.617 ms | 2.97 | 0.01 | 28% | ok |
| eigen | sylvester | f64 | 256x256 | 624.945 ms | 1.61 | 0.00 | 15% | ok |
| elementwise | exp | f64 | n=8388608 | 151.144 ms | 0.06 | 0.89 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 140.475 ms | 0.06 | 0.96 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 172.044 ms | 0.05 | 0.78 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 89.931 ms | 0.09 | 1.49 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 2470.273 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.544 ms | 1.03 | 10.70 | 105% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.097 ms | 0.85 | 8.79 | 86% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.067 ms | 2.78 | 5.84 | 57% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.633 ms | 0.20 | 0.69 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.861 ms | 0.34 | 2.84 | 28% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 40.547 ms | 0.28 | 0.02 | 3% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 69.023 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.743 ms | 0.45 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 115.273 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 131.357 ms | 0.17 | 0.01 | 2% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 47.941 ms | 0.47 | 0.02 | 4% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 81.691 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.898 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.398 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 125.382 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.527 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 41.598 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.176 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.540 ms | 0.33 | 0.02 | 3% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.814 ms | 0.93 | 3.71 | 36%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.279 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.663 ms | 0.88 | 0.06 | 8% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.821 ms | 0.92 | 3.68 | 36%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.384 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 110.366 ms | 0.20 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.879 ms | 0.19 | 0.78 | 8%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.771 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 112.525 ms | 0.20 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.888 ms | 0.19 | 0.78 | 8%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 13.009 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 633.912 ms | 0.49 | 0.00 | 5% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 6.806 ms | 0.49 | 1.98 | 19%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.932 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 205.685 ms | 1.50 | 0.00 | 14% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 6.796 ms | 0.50 | 1.98 | 19%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.481 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1729.147 ms | 0.36 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.661 ms | 0.10 | 0.41 | 4%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 18.061 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1519.532 ms | 0.41 | 0.00 | 4% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.459 ms | 0.10 | 0.42 | 4%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 231.567 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 406.632 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 135.926 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 109.289 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 129.005 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 81.666 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 533.424 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 175.409 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.799 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 202.377 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 192.997 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 212.933 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 308.674 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.170 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.442 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 33.292 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.930 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.801 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.098 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.144 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.585 ms | 0.73 | 8.18 | 80% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.293 ms | 0.79 | 8.91 | 87% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.904 ms | 0.67 | 7.51 | 73% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.939 ms | 1.07 | 8.09 | 79% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 140.637 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 145.145 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 132.673 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.668 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 186.719 ms | 0.09 | 0.36 | 4% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 113.890 ms | 2.36 | 0.04 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.325 ms | 7.22 | 7.67 | 75% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.458 ms | 2.60 | 10.39 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.145 ms | 5.33 | 10.67 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.986 ms | 3.36 | 3.36 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.974 ms | 3.37 | 3.37 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 57.734 ms | 4.65 | 0.05 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 45.169 ms | 5.94 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 23.659 ms | 11.35 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 23.606 ms | 11.37 | 0.07 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 70.389 ms | 3.81 | 0.02 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

