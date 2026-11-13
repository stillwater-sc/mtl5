## arduino-ventuno-q -- native, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bf315e4604cf` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.6 GFLOP/s |
| ... this build, scalar | f64 8.4, f32 8.4 GFLOP/s |
| bandwidth | read 22.6, update 24.6, triad 23.0 GB/s (ceiling 24.6) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 86% | 104% | 0 |
| blas2 | 10 | 60% | 70% | 0 |
| blas3 | 10 | 30% | 32% | 0 |
| factor | 12 | 12% | 32% | 0 |
| eigen | 8 | 15% | 22% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 30% | 68% | 0 |
| direct | 19 | 2% | 7% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 9% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 57% | 0 |
| mg | 1 | 37% | 37% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 34% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.694 ms | 2.95 | 23.57 | 96% | ok |
| blas1 | dot | f32 | n=8388608 | 4.516 ms | 3.71 | 14.86 | 60% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.893 ms | 2.13 | 25.51 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.946 ms | 4.25 | 25.51 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 5.345 ms | 1.57 | 25.11 | 102% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.163 ms | 5.30 | 21.22 | 86% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.422 ms | 1.00 | 7.97 | 32% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.478 ms | 0.48 | 3.84 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.395 ms | 1.00 | 7.99 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 185.531 ms | 0.99 | 0.72 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 7.975 ms | 4.21 | 16.84 | 68% | ok |
| blas2 | gemv | f32 | n=4096 | 5.670 ms | 5.92 | 11.84 | 48% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 9.945 ms | 3.37 | 13.50 | 55% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.524 ms | 7.42 | 14.84 | 60% | ok |
| blas2 | symv | f64 | n=4096 | 16.991 ms | 1.97 | 3.95 | 16% | ok |
| blas2 | trmv | f64 | n=4096 | 4.451 ms | 3.77 | 15.09 | 61% | ok |
| blas2 | trmv | f32 | n=4096 | 4.769 ms | 3.52 | 7.04 | 29% | ok |
| blas2 | trsv | f64 | n=4096 | 4.495 ms | 3.73 | 14.95 | 61% | ok |
| blas2 | trsv | f32 | n=4096 | 4.667 ms | 3.59 | 7.20 | 29% | ok |
| blas2 | ger | f64 | n=4096 | 15.628 ms | 2.15 | 17.18 | 70% | ok |
| blas3 | gemm | f64 | 512x512x512 | 52.490 ms | 5.11 | 0.12 | 30% | ok |
| blas3 | gemm | f32 | 512x512x512 | 24.708 ms | 10.86 | 0.13 | 32% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 41.638 ms | 3.22 | 0.15 | 19% | ok |
| blas3 | symm | f64 | 512x512x512 | 53.414 ms | 5.03 | 0.12 | 30% | ok |
| blas3 | syrk | f64 | 512x512x512 | 40.993 ms | 3.27 | 0.10 | 19% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 82.131 ms | 3.27 | 0.08 | 19% | ok |
| blas3 | trmm | f64 | 512x512x512 | 26.397 ms | 5.08 | 0.20 | 30% | ok |
| blas3 | trsm | f64 | 512x512x512 | 26.419 ms | 5.08 | 0.20 | 30% | ok |
| blas3 | trsm | f32 | 512x512x512 | 12.478 ms | 10.76 | 0.21 | 32% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 30.482 ms | 0.17 | 1.39 | 6% | ok |
| factor | lu | f64 | 512x512 | 25.147 ms | 3.56 | 0.17 | 21% | ok |
| factor | lu | f32 | 512x512 | 13.002 ms | 6.88 | 0.16 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 20.988 ms | 2.13 | 0.20 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 19.659 ms | 2.28 | 0.11 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 22.868 ms | 1.96 | 0.18 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 68.502 ms | 0.65 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 253.324 ms | 0.71 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 61.263 ms | 2.92 | 0.07 | 17% | ok |
| factor | hessenberg | f64 | 512x512 | 352.471 ms | 1.27 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.578 ms | 0.67 | 2.66 | 11% | ok |
| factor | inv | f64 | 512x512 | 151.004 ms | 1.78 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.755 ms | 5.34 | 0.25 | 32% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.846 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.512 ms | 1.55 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 604.445 ms | 0.25 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.596 ms | 2.48 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 154.111 ms | 2.72 | 0.01 | 16% | ok |
| eigen | svd | f64 | 256x256 | 231.034 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 67.405 ms | 3.73 | 0.02 | 22% | ok |
| eigen | sylvester | f64 | 256x256 | 370.149 ms | 2.72 | 0.00 | 16% | ok |
| elementwise | exp | f64 | n=8388608 | 81.088 ms | 0.10 | 1.66 | 7% | ok |
| elementwise | log | f64 | n=8388608 | 107.523 ms | 0.08 | 1.25 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 151.125 ms | 0.06 | 0.89 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 70.578 ms | 0.12 | 1.90 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 117.999 ms | 0.07 | 1.14 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.636 ms | 1.60 | 16.65 | 68% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.184 ms | 1.20 | 12.47 | 51% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.049 ms | 3.48 | 7.31 | 30% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.154 ms | 0.25 | 0.87 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.416 ms | 0.24 | 1.98 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 182.274 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 65.264 ms | 0.17 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.381 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 102.380 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 138.618 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.075 ms | 0.52 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 203.343 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.428 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.453 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.695 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.824 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.888 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 174.088 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.799 ms | 0.42 | 1.68 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.782 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.494 ms | 0.97 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.759 ms | 0.43 | 1.72 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.427 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 87.924 ms | 0.25 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.798 ms | 0.13 | 0.52 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 10.036 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 118.813 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.856 ms | 0.13 | 0.52 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 21.626 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 16544.452 ms | 0.02 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.465 ms | 0.45 | 1.81 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 46.634 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 207.680 ms | 1.49 | 0.00 | 9% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.692 ms | 0.44 | 1.75 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.780 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1743.039 ms | 0.35 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.635 ms | 0.12 | 0.47 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 29.727 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2412.884 ms | 0.26 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.296 ms | 0.12 | 0.48 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.967 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 105.197 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 54.203 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 29.085 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.642 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 34.299 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 63.732 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 64.504 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.691 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 37.443 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.821 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.787 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 47.419 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.114 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.941 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 30.788 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.167 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.993 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.064 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.766 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.111 ms | 1.24 | 13.90 | 57% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.497 ms | 1.05 | 11.74 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.951 ms | 0.89 | 9.94 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.139 ms | 1.20 | 9.04 | 37% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 234.544 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 269.512 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 151.096 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.309 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 8.093 ms | 2.07 | 8.29 | 34% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 48.612 ms | 5.52 | 0.09 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.945 ms | 5.70 | 6.05 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.847 ms | 5.89 | 23.57 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.395 ms | 12.03 | 24.05 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.653 ms | 4.59 | 4.59 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.170 ms | 14.34 | 14.34 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 25.505 ms | 10.52 | 0.12 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 26.393 ms | 10.17 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 20.320 ms | 13.21 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 20.323 ms | 13.21 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 38.520 ms | 6.97 | 0.04 | n/a | ok |

