## i7-12700k -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.5 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.5 GFLOP/s |
| ... this build, AVX2 | f64 78.3, f32 156.5 GFLOP/s |
| bandwidth | read 37.8, update 41.5, triad 31.4 GB/s (ceiling 41.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 88% | 103% | 0 |
| blas2 | 10 | 93% | 96% | 0 |
| blas3 | 10 | 67% | 90% | 0 |
| factor | 12 | 8% | 23% | 0 |
| eigen | 8 | 7% | 28% | 0 |
| elementwise | 5 | 7% | 11% | 0 |
| sparse | 5 | 49% | 89% | 0 |
| direct | 19 | 1% | 29% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 4% | 22% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 54% | 72% | 0 |
| mg | 1 | 49% | 49% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 84% | 132% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 3.690 ms | 4.55 | 36.37 | 88% | ok |
| blas1 | dot | f32 | n=8388608 | 1.704 ms | 9.84 | 39.38 | 95% | ok |
| blas1 | axpy | f64 | n=8388608 | 4.795 ms | 3.50 | 41.99 | 101% | ok |
| blas1 | axpy | f32 | n=8388608 | 2.348 ms | 7.14 | 42.87 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 3.463 ms | 2.42 | 38.76 | 93% | ok |
| blas1 | two_norm | f64 | n=8388608 | 2.214 ms | 7.58 | 30.30 | 73% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.036 ms | 2.08 | 16.63 | 40% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.039 ms | 1.19 | 9.53 | 23% | ok |
| blas1 | sum | f64 | n=8388608 | 3.967 ms | 2.11 | 16.92 | 41% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 71.017 ms | 2.60 | 1.89 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.410 ms | 9.84 | 39.38 | 95% | ok |
| blas2 | gemv | f32 | n=4096 | 1.689 ms | 19.87 | 39.75 | 96% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.471 ms | 9.67 | 38.69 | 93% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.680 ms | 19.98 | 39.97 | 96% | ok |
| blas2 | symv | f64 | n=4096 | 2.942 ms | 11.41 | 22.84 | 55% | ok |
| blas2 | trmv | f64 | n=4096 | 1.793 ms | 9.36 | 37.47 | 90% | ok |
| blas2 | trmv | f32 | n=4096 | 0.868 ms | 19.32 | 38.68 | 93% | ok |
| blas2 | trsv | f64 | n=4096 | 1.818 ms | 9.23 | 36.95 | 89% | ok |
| blas2 | trsv | f32 | n=4096 | 0.878 ms | 19.12 | 38.28 | 92% | ok |
| blas2 | ger | f64 | n=4096 | 7.161 ms | 4.69 | 37.49 | 90% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.803 ms | 70.59 | 1.65 | 90% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.898 ms | 141.43 | 1.66 | 90% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.624 ms | 51.14 | 2.40 | 65% | ok |
| blas3 | symm | f64 | 512x512x512 | 4.042 ms | 66.41 | 1.56 | 85% | ok |
| blas3 | syrk | f64 | 512x512x512 | 2.573 ms | 52.16 | 1.63 | 67% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 5.020 ms | 53.47 | 1.25 | 68% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.693 ms | 49.84 | 1.95 | 64% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.775 ms | 48.37 | 1.89 | 62% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.411 ms | 95.15 | 1.86 | 61% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.152 ms | 0.40 | 3.23 | 8% | ok |
| factor | lu | f64 | 512x512 | 11.040 ms | 8.10 | 0.38 | 10% | ok |
| factor | lu | f32 | 512x512 | 3.288 ms | 27.22 | 0.64 | 17% | ok |
| factor | cholesky | f64 | 512x512 | 7.088 ms | 6.31 | 0.59 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.568 ms | 6.81 | 0.32 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.220 ms | 6.20 | 0.58 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 4.413 ms | 10.14 | 0.95 | 13% | ok |
| factor | qr | f64 | 512x512 | 228.000 ms | 0.78 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 23.240 ms | 7.70 | 0.18 | 10% | ok |
| factor | hessenberg | f64 | 512x512 | 261.854 ms | 1.71 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.630 ms | 0.64 | 2.57 | 6%\* | ok |
| factor | inv | f64 | 512x512 | 55.026 ms | 4.88 | 0.08 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 4.882 ms | 18.33 | 0.86 | 23% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 16.850 ms | 1.33 | 0.06 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 56.752 ms | 2.66 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 749.812 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 30.672 ms | 5.47 | 0.03 | 7% | ok |
| eigen | schur | f64 | 256x256 | 79.932 ms | 5.25 | 0.01 | 7% | ok |
| eigen | svd | f64 | 256x256 | 95.131 ms | 3.88 | 0.01 | 5% | ok |
| eigen | expm | f64 | 256x256 | 11.624 ms | 21.65 | 0.09 | 28% | ok |
| eigen | sylvester | f64 | 256x256 | 176.795 ms | 5.69 | 0.01 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 39.508 ms | 0.21 | 3.40 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 45.718 ms | 0.18 | 2.94 | 7% | ok |
| elementwise | sin | f64 | n=8388608 | 60.050 ms | 0.14 | 2.24 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 29.117 ms | 0.29 | 4.61 | 11% | ok |
| elementwise | erf | f64 | n=8388608 | 202.424 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.738 ms | 3.55 | 36.91 | 89% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.913 ms | 2.87 | 29.83 | 72% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.324 ms | 9.68 | 20.36 | 49% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.866 ms | 0.67 | 2.36 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.311 ms | 1.53 | 12.70 | 31%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 15.256 ms | 0.73 | 0.05 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 27.034 ms | 0.41 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.904 ms | 1.26 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.283 ms | 0.56 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.035 ms | 0.51 | 0.02 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.744 ms | 1.34 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 25.733 ms | 0.07 | 0.01 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.460 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.030 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 58.605 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.193 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.591 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.158 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 13.490 ms | 0.83 | 0.05 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.252 ms | 3.00 | 12.00 | 29%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.126 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.481 ms | 2.50 | 0.16 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.272 ms | 2.78 | 11.13 | 27%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.118 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.931 ms | 0.62 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.646 ms | 1.17 | 4.68 | 11%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.485 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.129 ms | 0.59 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.650 ms | 1.16 | 4.65 | 11%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.932 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 206.638 ms | 1.49 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.480 ms | 2.28 | 9.10 | 22%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.641 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 67.636 ms | 4.56 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.695 ms | 1.99 | 7.95 | 19%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.960 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 528.105 ms | 1.17 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.547 ms | 0.45 | 1.79 | 4%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.348 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 694.318 ms | 0.89 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.558 ms | 0.45 | 1.78 | 4%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.218 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 40.782 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 21.030 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.446 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.745 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.537 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 19.906 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.783 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.474 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.344 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 14.068 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 13.198 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 14.492 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.034 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.596 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.046 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.083 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.027 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.025 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.704 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.984 ms | 2.66 | 29.82 | 72% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.311 ms | 2.00 | 22.37 | 54% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.511 ms | 1.73 | 19.41 | 47% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.334 ms | 2.68 | 20.28 | 49% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 28.548 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 29.312 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.676 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.594 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 1.924 ms | 8.72 | 34.87 | 84% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 6.001 ms | 44.73 | 0.70 | 57% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.325 ms | 51.59 | 54.82 | 132%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.912 ms | 8.78 | 35.11 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.760 ms | 22.08 | 44.16 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.293 ms | 57.21 | 57.21 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.280 ms | 59.87 | 59.87 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 3.955 ms | 67.87 | 0.80 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.578 ms | 48.12 | 0.38 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.556 ms | 48.31 | 0.28 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 2.098 ms | 127.97 | 0.75 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 1.877 ms | 143.00 | 0.84 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

