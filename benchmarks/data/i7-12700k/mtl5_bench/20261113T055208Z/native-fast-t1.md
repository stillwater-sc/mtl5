## i7-12700k -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `8867b44b3f52` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, AVX2 | f64 78.3, f32 156.6 GFLOP/s |
| bandwidth | read 37.7, update 41.5, triad 31.5 GB/s (ceiling 41.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 88% | 104% | 0 |
| blas2 | 10 | 94% | 97% | 0 |
| blas3 | 10 | 62% | 90% | 0 |
| factor | 12 | 8% | 24% | 0 |
| eigen | 8 | 6% | 27% | 0 |
| elementwise | 5 | 7% | 11% | 0 |
| sparse | 5 | 51% | 89% | 0 |
| direct | 19 | 1% | 29% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 12 | 4% | 23% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 54% | 72% | 0 |
| mg | 1 | 49% | 49% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 84% | 133% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 3.685 ms | 4.55 | 36.43 | 88% | ok |
| blas1 | dot | f32 | n=8388608 | 1.693 ms | 9.91 | 39.64 | 95% | ok |
| blas1 | axpy | f64 | n=8388608 | 4.751 ms | 3.53 | 42.38 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 2.342 ms | 7.16 | 42.98 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 3.452 ms | 2.43 | 38.88 | 94% | ok |
| blas1 | two_norm | f64 | n=8388608 | 2.195 ms | 7.64 | 30.57 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.007 ms | 2.09 | 16.75 | 40% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.028 ms | 1.19 | 9.55 | 23% | ok |
| blas1 | sum | f64 | n=8388608 | 3.957 ms | 2.12 | 16.96 | 41% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 70.420 ms | 2.62 | 1.91 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.400 ms | 9.87 | 39.49 | 95% | ok |
| blas2 | gemv | f32 | n=4096 | 1.682 ms | 19.95 | 39.91 | 96% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.456 ms | 9.71 | 38.85 | 94% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.674 ms | 20.05 | 40.11 | 97% | ok |
| blas2 | symv | f64 | n=4096 | 6.174 ms | 5.43 | 10.88 | 26% | ok |
| blas2 | trmv | f64 | n=4096 | 1.784 ms | 9.40 | 37.65 | 91% | ok |
| blas2 | trmv | f32 | n=4096 | 0.854 ms | 19.65 | 39.34 | 95% | ok |
| blas2 | trsv | f64 | n=4096 | 1.823 ms | 9.21 | 36.86 | 89% | ok |
| blas2 | trsv | f32 | n=4096 | 0.870 ms | 19.29 | 38.61 | 93% | ok |
| blas2 | ger | f64 | n=4096 | 7.168 ms | 4.68 | 37.46 | 90% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.825 ms | 70.18 | 1.64 | 90% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.897 ms | 141.50 | 1.66 | 90% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.649 ms | 50.66 | 2.37 | 65% | ok |
| blas3 | symm | f64 | 512x512x512 | 4.060 ms | 66.12 | 1.55 | 84% | ok |
| blas3 | syrk | f64 | 512x512x512 | 9.715 ms | 13.82 | 0.43 | 18% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 20.012 ms | 13.41 | 0.31 | 17% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.862 ms | 46.89 | 1.83 | 60% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.919 ms | 45.97 | 1.80 | 59% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.375 ms | 97.61 | 1.91 | 62% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.076 ms | 0.41 | 3.25 | 8% | ok |
| factor | lu | f64 | 512x512 | 10.862 ms | 8.24 | 0.39 | 11% | ok |
| factor | lu | f32 | 512x512 | 3.452 ms | 25.92 | 0.61 | 17% | ok |
| factor | cholesky | f64 | 512x512 | 7.093 ms | 6.31 | 0.59 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.540 ms | 6.84 | 0.32 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.197 ms | 6.22 | 0.58 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 46.554 ms | 0.96 | 0.09 | 1% | ok |
| factor | qr | f64 | 512x512 | 225.622 ms | 0.79 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 23.336 ms | 7.67 | 0.18 | 10% | ok |
| factor | hessenberg | f64 | 512x512 | 260.291 ms | 1.72 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.610 ms | 0.65 | 2.60 | 6%\* | ok |
| factor | inv | f64 | 512x512 | 55.017 ms | 4.88 | 0.08 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 4.813 ms | 18.59 | 0.87 | 24% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 16.896 ms | 1.32 | 0.06 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 56.892 ms | 2.65 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 750.458 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 30.715 ms | 5.46 | 0.03 | 7% | ok |
| eigen | schur | f64 | 256x256 | 79.512 ms | 5.28 | 0.01 | 7% | ok |
| eigen | svd | f64 | 256x256 | 91.650 ms | 4.03 | 0.01 | 5% | ok |
| eigen | expm | f64 | 256x256 | 11.862 ms | 21.22 | 0.09 | 27% | ok |
| eigen | sylvester | f64 | 256x256 | 204.765 ms | 4.92 | 0.01 | 6% | ok |
| elementwise | exp | f64 | n=8388608 | 39.254 ms | 0.21 | 3.42 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 46.764 ms | 0.18 | 2.87 | 7% | ok |
| elementwise | sin | f64 | n=8388608 | 59.693 ms | 0.14 | 2.25 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 28.811 ms | 0.29 | 4.66 | 11% | ok |
| elementwise | erf | f64 | n=8388608 | 200.870 ms | 0.04 | 0.67 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.733 ms | 3.57 | 37.13 | 89% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.919 ms | 2.85 | 29.63 | 71% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.139 ms | 10.12 | 21.27 | 51% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.867 ms | 0.67 | 2.36 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.264 ms | 1.59 | 13.17 | 32%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 72.752 ms | 0.15 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 27.439 ms | 0.41 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 9.097 ms | 1.23 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.386 ms | 0.57 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 55.193 ms | 0.41 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.915 ms | 1.32 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 98.650 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.493 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.078 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.231 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 16.661 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.335 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 70.063 ms | 0.16 | 0.01 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.253 ms | 2.99 | 11.95 | 29%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.461 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.304 ms | 2.60 | 0.17 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.272 ms | 2.78 | 11.14 | 27%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.133 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.049 ms | 0.64 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.654 ms | 1.16 | 4.63 | 11%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.923 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 49.325 ms | 0.45 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.644 ms | 1.17 | 4.70 | 11%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 9.500 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 5302.965 ms | 0.06 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.439 ms | 2.34 | 9.36 | 23%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 18.670 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 70.765 ms | 4.36 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.656 ms | 2.03 | 8.14 | 20%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.921 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 526.101 ms | 1.17 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.545 ms | 0.45 | 1.79 | 4%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.603 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 656.274 ms | 0.94 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.498 ms | 0.45 | 1.80 | 4%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.213 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 28.430 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 20.882 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.435 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.689 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.589 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 20.307 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.657 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.467 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.617 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 14.962 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 16.001 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 18.340 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.035 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.574 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.083 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.105 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.029 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.023 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.742 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.976 ms | 2.68 | 30.05 | 72% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.308 ms | 2.00 | 22.42 | 54% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.509 ms | 1.73 | 19.44 | 47% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.371 ms | 2.68 | 20.23 | 49% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 28.965 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 39.320 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 35.086 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.589 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 1.931 ms | 8.69 | 34.75 | 84% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 5.994 ms | 44.78 | 0.70 | 57% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.323 ms | 51.98 | 55.23 | 133%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.899 ms | 8.84 | 35.34 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.769 ms | 21.83 | 43.66 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.288 ms | 58.18 | 58.18 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.277 ms | 60.56 | 60.56 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 5.139 ms | 52.23 | 0.61 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.639 ms | 47.61 | 0.37 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.611 ms | 47.84 | 0.28 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 2.100 ms | 127.83 | 0.75 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 1.905 ms | 140.90 | 0.83 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

