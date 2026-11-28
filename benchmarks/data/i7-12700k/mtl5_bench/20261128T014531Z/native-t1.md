## i7-12700k -- native, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.5 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.5 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.3 GFLOP/s |
| bandwidth | read 33.0, update 38.4, triad 32.4 GB/s (ceiling 38.4) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 44% | 98% | 0 |
| blas2 | 10 | 79% | 95% | 0 |
| blas3 | 10 | 17% | 18% | 0 |
| factor | 12 | 8% | 18% | 0 |
| eigen | 8 | 6% | 12% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 49% | 95% | 0 |
| direct | 19 | 1% | 32% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 24% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 62% | 77% | 0 |
| mg | 1 | 56% | 56% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 35% | 83% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.629 ms | 2.98 | 23.84 | 62% | ok |
| blas1 | dot | f32 | n=8388608 | 4.243 ms | 3.95 | 15.82 | 41% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.422 ms | 2.61 | 31.35 | 82% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.719 ms | 3.55 | 21.33 | 56% | ok |
| blas1 | scale | f64 | n=8388608 | 3.577 ms | 2.35 | 37.52 | 98% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.880 ms | 3.44 | 13.75 | 36% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.018 ms | 2.09 | 16.70 | 43% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.048 ms | 1.19 | 9.52 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 4.005 ms | 2.09 | 16.76 | 44% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 77.381 ms | 2.38 | 1.73 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.698 ms | 9.07 | 36.31 | 95% | ok |
| blas2 | gemv | f32 | n=4096 | 2.743 ms | 12.23 | 24.48 | 64% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 4.767 ms | 7.04 | 28.17 | 73% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.221 ms | 15.11 | 30.23 | 79% | ok |
| blas2 | symv | f64 | n=4096 | 3.206 ms | 10.47 | 20.96 | 55% | ok |
| blas2 | trmv | f64 | n=4096 | 2.012 ms | 8.34 | 33.38 | 87% | ok |
| blas2 | trmv | f32 | n=4096 | 1.426 ms | 11.77 | 23.56 | 61% | ok |
| blas2 | trsv | f64 | n=4096 | 1.980 ms | 8.47 | 33.93 | 88% | ok |
| blas2 | trsv | f32 | n=4096 | 1.393 ms | 12.04 | 24.11 | 63% | ok |
| blas2 | ger | f64 | n=4096 | 7.732 ms | 4.34 | 34.73 | 90% | ok |
| blas3 | gemm | f64 | 512x512x512 | 20.079 ms | 13.37 | 0.31 | 17% | ok |
| blas3 | gemm | f32 | 512x512x512 | 10.366 ms | 25.90 | 0.30 | 17% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 12.559 ms | 10.69 | 0.50 | 14% | ok |
| blas3 | symm | f64 | 512x512x512 | 19.782 ms | 13.57 | 0.32 | 17% | ok |
| blas3 | syrk | f64 | 512x512x512 | 13.629 ms | 9.85 | 0.31 | 13% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 24.378 ms | 11.01 | 0.26 | 14% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.562 ms | 14.04 | 0.55 | 18% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.633 ms | 13.93 | 0.54 | 18% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.193 ms | 25.85 | 0.50 | 17% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.245 ms | 0.40 | 3.21 | 8% | ok |
| factor | lu | f64 | 512x512 | 11.646 ms | 7.68 | 0.36 | 10% | ok |
| factor | lu | f32 | 512x512 | 4.299 ms | 20.81 | 0.49 | 13% | ok |
| factor | cholesky | f64 | 512x512 | 7.331 ms | 6.10 | 0.57 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.633 ms | 6.75 | 0.32 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.438 ms | 6.02 | 0.56 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 4.766 ms | 9.39 | 0.88 | 12% | ok |
| factor | qr | f64 | 512x512 | 226.778 ms | 0.79 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 25.845 ms | 6.92 | 0.16 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 267.932 ms | 1.67 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.623 ms | 0.65 | 2.58 | 7%\* | ok |
| factor | inv | f64 | 512x512 | 54.149 ms | 4.96 | 0.08 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.192 ms | 14.45 | 0.68 | 18% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 18.133 ms | 1.23 | 0.06 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 58.921 ms | 2.56 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 760.796 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 33.007 ms | 5.08 | 0.03 | 6% | ok |
| eigen | schur | f64 | 256x256 | 82.922 ms | 5.06 | 0.01 | 6% | ok |
| eigen | svd | f64 | 256x256 | 111.897 ms | 3.30 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 25.816 ms | 9.75 | 0.04 | 12% | ok |
| eigen | sylvester | f64 | 256x256 | 185.805 ms | 5.42 | 0.01 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 39.713 ms | 0.21 | 3.38 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 46.125 ms | 0.18 | 2.91 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 60.698 ms | 0.14 | 2.21 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 29.268 ms | 0.29 | 4.59 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 203.750 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.749 ms | 3.50 | 36.36 | 95% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.922 ms | 2.84 | 29.53 | 77% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.668 ms | 8.97 | 18.86 | 49% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 5.105 ms | 0.64 | 2.25 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.345 ms | 1.49 | 12.38 | 32%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 15.190 ms | 0.74 | 0.05 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 24.993 ms | 0.45 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.382 ms | 1.33 | 0.09 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.098 ms | 0.56 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.619 ms | 0.50 | 0.02 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 17.878 ms | 1.25 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 25.331 ms | 0.07 | 0.01 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.350 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.976 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 58.048 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.185 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.318 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.100 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 12.742 ms | 0.88 | 0.06 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.247 ms | 3.06 | 12.25 | 32%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.074 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.068 ms | 2.75 | 0.18 | 4% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.251 ms | 3.01 | 12.04 | 31%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.137 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.484 ms | 0.58 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.779 ms | 0.97 | 3.88 | 10%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.598 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.795 ms | 0.58 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.645 ms | 1.17 | 4.69 | 12%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.949 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 204.523 ms | 1.51 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.485 ms | 2.27 | 9.08 | 24%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.696 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 60.140 ms | 5.13 | 0.01 | 7% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.516 ms | 2.22 | 8.89 | 23%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.886 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 558.692 ms | 1.10 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.596 ms | 0.44 | 1.77 | 5%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.289 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 456.249 ms | 1.35 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.575 ms | 0.44 | 1.78 | 5%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 31.303 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 35.026 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 23.113 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.624 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 15.174 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 14.149 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.853 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.531 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.371 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 18.215 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.759 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 16.222 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 19.198 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.042 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.562 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 11.639 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.089 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.023 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.028 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.722 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.992 ms | 2.64 | 29.58 | 77% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.222 ms | 2.14 | 23.99 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.565 ms | 1.67 | 18.74 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.497 ms | 2.85 | 21.53 | 56% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 30.670 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 31.428 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.944 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 5.023 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.937 ms | 3.40 | 13.59 | 35% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 39.031 ms | 6.88 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.559 ms | 30.00 | 31.88 | 83%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.311 ms | 7.26 | 29.04 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.997 ms | 16.83 | 33.65 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.726 ms | 9.72 | 9.72 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.725 ms | 9.73 | 9.73 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.608 ms | 15.25 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.837 ms | 12.88 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.252 ms | 26.18 | 0.15 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.236 ms | 26.22 | 0.15 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 26.867 ms | 9.99 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

