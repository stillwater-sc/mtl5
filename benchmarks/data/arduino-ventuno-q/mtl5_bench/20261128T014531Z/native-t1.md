## arduino-ventuno-q -- native, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.6 GFLOP/s |
| ... this build, scalar | f64 8.4, f32 8.4 GFLOP/s |
| bandwidth | read 22.6, update 24.8, triad 23.1 GB/s (ceiling 24.8) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 85% | 104% | 0 |
| blas2 | 10 | 59% | 71% | 0 |
| blas3 | 10 | 30% | 34% | 0 |
| factor | 12 | 13% | 32% | 0 |
| eigen | 8 | 15% | 22% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 29% | 66% | 0 |
| direct | 19 | 2% | 7% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 3% | 10% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 47% | 56% | 0 |
| mg | 1 | 36% | 36% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 34% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.692 ms | 2.95 | 23.58 | 95% | ok |
| blas1 | dot | f32 | n=8388608 | 4.320 ms | 3.88 | 15.53 | 63% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.783 ms | 2.16 | 25.87 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.941 ms | 4.26 | 25.54 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 5.367 ms | 1.56 | 25.01 | 101% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.177 ms | 5.28 | 21.12 | 85% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.330 ms | 1.01 | 8.06 | 32% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.336 ms | 0.48 | 3.87 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.190 ms | 1.02 | 8.19 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 182.980 ms | 1.01 | 0.73 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 7.869 ms | 4.26 | 17.06 | 69% | ok |
| blas2 | gemv | f32 | n=4096 | 5.756 ms | 5.83 | 11.66 | 47% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 10.178 ms | 3.30 | 13.19 | 53% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.619 ms | 7.26 | 14.54 | 59% | ok |
| blas2 | symv | f64 | n=4096 | 11.933 ms | 2.81 | 5.63 | 23% | ok |
| blas2 | trmv | f64 | n=4096 | 4.503 ms | 3.73 | 14.92 | 60% | ok |
| blas2 | trmv | f32 | n=4096 | 3.282 ms | 5.11 | 10.24 | 41% | ok |
| blas2 | trsv | f64 | n=4096 | 4.417 ms | 3.80 | 15.21 | 61% | ok |
| blas2 | trsv | f32 | n=4096 | 3.060 ms | 5.48 | 10.98 | 44% | ok |
| blas2 | ger | f64 | n=4096 | 15.305 ms | 2.19 | 17.54 | 71% | ok |
| blas3 | gemm | f64 | 512x512x512 | 52.611 ms | 5.10 | 0.12 | 30% | ok |
| blas3 | gemm | f32 | 512x512x512 | 24.271 ms | 11.06 | 0.13 | 33% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 39.929 ms | 3.36 | 0.16 | 20% | ok |
| blas3 | symm | f64 | 512x512x512 | 55.273 ms | 4.86 | 0.11 | 29% | ok |
| blas3 | syrk | f64 | 512x512x512 | 41.676 ms | 3.22 | 0.10 | 19% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 80.714 ms | 3.33 | 0.08 | 20% | ok |
| blas3 | trmm | f64 | 512x512x512 | 25.739 ms | 5.21 | 0.20 | 31% | ok |
| blas3 | trsm | f64 | 512x512x512 | 26.778 ms | 5.01 | 0.20 | 30% | ok |
| blas3 | trsm | f32 | 512x512x512 | 11.886 ms | 11.29 | 0.22 | 34% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 25.037 ms | 0.21 | 1.70 | 7% | ok |
| factor | lu | f64 | 512x512 | 24.220 ms | 3.69 | 0.17 | 22% | ok |
| factor | lu | f32 | 512x512 | 12.985 ms | 6.89 | 0.16 | 21% | ok |
| factor | cholesky | f64 | 512x512 | 20.809 ms | 2.15 | 0.20 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 19.627 ms | 2.28 | 0.11 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 22.550 ms | 1.98 | 0.19 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 11.040 ms | 4.05 | 0.38 | 24% | ok |
| factor | qr | f64 | 512x512 | 248.744 ms | 0.72 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 60.532 ms | 2.96 | 0.07 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 347.219 ms | 1.29 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.552 ms | 0.68 | 2.70 | 11% | ok |
| factor | inv | f64 | 512x512 | 145.325 ms | 1.85 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.705 ms | 5.36 | 0.25 | 32% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 40.156 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.449 ms | 1.55 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 603.687 ms | 0.25 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.188 ms | 2.50 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 150.159 ms | 2.79 | 0.01 | 17% | ok |
| eigen | svd | f64 | 256x256 | 229.721 ms | 1.61 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 66.694 ms | 3.77 | 0.02 | 22% | ok |
| eigen | sylvester | f64 | 256x256 | 371.237 ms | 2.71 | 0.00 | 16% | ok |
| elementwise | exp | f64 | n=8388608 | 91.073 ms | 0.09 | 1.47 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 112.422 ms | 0.07 | 1.19 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 150.112 ms | 0.06 | 0.89 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 67.721 ms | 0.12 | 1.98 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 114.388 ms | 0.07 | 1.17 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.650 ms | 1.59 | 16.50 | 66% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.109 ms | 1.24 | 12.91 | 52% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.301 ms | 3.40 | 7.16 | 29% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 12.865 ms | 0.25 | 0.89 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 9.148 ms | 0.22 | 1.82 | 7% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 37.773 ms | 0.30 | 0.02 | 2% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 59.861 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.194 ms | 0.43 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 102.849 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 122.367 ms | 0.18 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.326 ms | 0.52 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 73.567 ms | 0.03 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.700 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.352 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 125.568 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.569 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 43.544 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.482 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 29.946 ms | 0.37 | 0.02 | 2% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.830 ms | 0.41 | 1.65 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.997 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.714 ms | 0.96 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.761 ms | 0.43 | 1.72 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.445 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 88.946 ms | 0.25 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.986 ms | 0.13 | 0.51 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 8.697 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 101.496 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.917 ms | 0.13 | 0.51 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 11.657 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 675.668 ms | 0.46 | 0.00 | 3% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.780 ms | 0.43 | 1.73 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 27.765 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 182.771 ms | 1.69 | 0.00 | 10% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.696 ms | 0.44 | 1.75 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.613 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1709.264 ms | 0.36 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 30.791 ms | 0.11 | 0.44 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.599 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1198.718 ms | 0.51 | 0.00 | 3% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 31.200 ms | 0.11 | 0.43 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.837 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 116.755 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 53.794 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 29.014 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.814 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 34.272 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 63.010 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 64.093 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.843 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 39.372 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.956 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 34.812 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 40.974 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.115 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 5.016 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 30.640 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.135 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 3.012 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.076 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.768 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.123 ms | 1.23 | 13.81 | 56% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.515 ms | 1.04 | 11.66 | 47% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.979 ms | 0.88 | 9.85 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.149 ms | 1.20 | 9.04 | 36% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 234.106 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 233.699 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 147.335 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.333 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 8.025 ms | 2.09 | 8.36 | 34% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 48.627 ms | 5.52 | 0.09 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.938 ms | 5.71 | 6.07 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.867 ms | 5.85 | 23.41 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.410 ms | 11.90 | 23.81 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.650 ms | 4.60 | 4.60 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.157 ms | 14.51 | 14.51 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 24.149 ms | 11.12 | 0.13 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 26.332 ms | 10.19 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 20.317 ms | 13.21 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 20.319 ms | 13.21 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 38.514 ms | 6.97 | 0.04 | n/a | ok |

