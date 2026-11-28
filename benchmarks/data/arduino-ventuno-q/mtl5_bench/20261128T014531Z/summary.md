# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,4, CPUs 0,1,2,3, clock policy governor performance on the A78C policies; performance 902-2112 MHz on cpus 0,1; performance 940-2361 MHz on cpus 2,3, 2026-11-28T02:05:40Z.

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

## arduino-ventuno-q -- native, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 67.6 GFLOP/s, f32 135.0 GFLOP/s |
| ... hardware, NEON | f64 67.6, f32 135.0 GFLOP/s |
| ... this build, scalar | f64 33.7, f32 33.7 GFLOP/s |
| bandwidth | read 23.1, update 24.6, triad 22.7 GB/s (ceiling 24.6) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 98% | 104% | 0 | 26% |
| blas2 | 10 | 63% | 80% | 0 | 27% |
| blas3 | 10 | 8% | 17% | 0 | 28% |
| factor | 12 | 6% | 38% | 0 | 35% |
| eigen | 8 | 4% | 10% | 0 | 28% |
| elementwise | 5 | 5% | 8% | 0 | 28% |
| sparse | 5 | 56% | 92% | 0 | 27% |
| direct | 19 | 1% | 6% | 0 | 27% |
| ordering | 5 | n/a | n/a | 0 | 28% |
| direct3d | 12 | 2% | 7% | 0 | 27% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 52% | 62% | 0 | 28% |
| mg | 1 | 37% | 37% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 31% | 38% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 51% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.570 ms | 3.01 | 24.10 | 98% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.793 ms | 6.01 | 24.03 | 98% | ok | 1.5x (39%) |
| blas1 | axpy | f64 | n=8388608 | 8.189 ms | 2.05 | 24.58 | 100% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 4.001 ms | 4.19 | 25.16 | 102% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 5.426 ms | 1.55 | 24.74 | 100% | ok | 1.0x (25%) |
| blas1 | two_norm | f64 | n=8388608 | 2.624 ms | 6.39 | 25.57 | 104% | ok | 1.2x (30%) |
| blas1 | one_norm | f64 | n=8388608 | 7.316 ms | 1.15 | 9.17 | 37% | ok | 1.1x (28%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.443 ms | 0.54 | 4.35 | 18% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.900 ms | 1.06 | 8.50 | 34% | ok | 1.0x (26%) |
| blas1 | dot_k3 | f64 | n=8388608 | 179.235 ms | 1.03 | 0.75 | 3% | ok | 1.0x (26%) |
| blas2 | gemv | f64 | n=4096 | 7.008 ms | 4.79 | 19.16 | 78% | ok | 1.1x (28%) |
| blas2 | gemv | f32 | n=4096 | 3.407 ms | 9.85 | 19.71 | 80% | ok | 1.7x (42%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.868 ms | 4.26 | 17.07 | 69% | ok | 1.3x (32%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.757 ms | 5.83 | 11.66 | 47% | ok | 0.8x (20%) |
| blas2 | symv | f64 | n=4096 | 11.007 ms | 3.05 | 6.10 | 25% | ok | 1.1x (27%) |
| blas2 | trmv | f64 | n=4096 | 4.347 ms | 3.86 | 15.45 | 63% | ok | 1.0x (26%) |
| blas2 | trmv | f32 | n=4096 | 3.038 ms | 5.52 | 11.06 | 45% | ok | 1.1x (27%) |
| blas2 | trsv | f64 | n=4096 | 4.438 ms | 3.78 | 15.14 | 61% | ok | 1.0x (25%) |
| blas2 | trsv | f32 | n=4096 | 2.929 ms | 5.73 | 11.47 | 47% | ok | 1.0x (26%) |
| blas2 | ger | f64 | n=4096 | 14.816 ms | 2.26 | 18.12 | 74% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 26.844 ms | 10.00 | 0.23 | 15% | ok | 2.0x (49%) |
| blas3 | gemm | f32 | 512x512x512 | 11.584 ms | 23.17 | 0.27 | 17% | ok | 2.1x (52%) |
| blas3 | gemmt | f64 | 512x512x512 | 38.165 ms | 3.52 | 0.16 | 5% | ok | 1.0x (26%) |
| blas3 | symm | f64 | 512x512x512 | 48.034 ms | 5.59 | 0.13 | 8% | ok | 1.2x (29%) |
| blas3 | syrk | f64 | 512x512x512 | 37.120 ms | 3.62 | 0.11 | 5% | ok | 1.1x (28%) |
| blas3 | syr2k | f64 | 512x512x512 | 80.018 ms | 3.35 | 0.08 | 5% | ok | 1.0x (25%) |
| blas3 | trmm | f64 | 512x512x512 | 23.600 ms | 5.69 | 0.22 | 8% | ok | 1.1x (27%) |
| blas3 | trsm | f64 | 512x512x512 | 24.701 ms | 5.43 | 0.21 | 8% | ok | 1.1x (27%) |
| blas3 | trsm | f32 | 512x512x512 | 10.756 ms | 12.48 | 0.24 | 9% | ok | 1.1x (28%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 29.315 ms | 0.18 | 1.45 | 6% | ok | 0.9x (21%) |
| factor | lu | f64 | 512x512 | 17.438 ms | 5.13 | 0.24 | 8% | ok | 1.4x (35%) |
| factor | lu | f32 | 512x512 | 9.057 ms | 9.88 | 0.23 | 7% | ok | 1.4x (36%) |
| factor | cholesky | f64 | 512x512 | 18.565 ms | 2.41 | 0.23 | 4% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 17.621 ms | 2.54 | 0.12 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 20.319 ms | 2.20 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 10.254 ms | 4.36 | 0.41 | 6% | ok | 1.1x (27%) |
| factor | qr | f64 | 512x512 | 149.322 ms | 1.20 | 0.03 | 2% | ok | 1.7x (42%) |
| factor | lq | f64 | 512x512 | 36.533 ms | 4.90 | 0.11 | 7% | ok | 1.7x (41%) |
| factor | hessenberg | f64 | 512x512 | 201.343 ms | 2.22 | 0.02 | 3% | ok | 1.7x (43%) |
| factor | householder | f64 | 512x512 | 0.445 ms | 2.36 | 9.42 | 38% | ok | 3.5x (87%) |
| factor | inv | f64 | 512x512 | 129.347 ms | 2.08 | 0.03 | 3% | ok | 1.1x (28%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.349 ms | 7.25 | 0.34 | 11% | ok | 1.4x (34%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.842 ms | 0.62 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 87.157 ms | 1.73 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 532.789 ms | 0.28 | 0.00 | 0% | ok | 1.1x (28%) |
| eigen | eigenvalue | f64 | 256x256 | 60.400 ms | 2.78 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 134.226 ms | 3.12 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.387 ms | 1.80 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 38.012 ms | 6.62 | 0.03 | 10% | ok | 1.8x (44%) |
| eigen | sylvester | f64 | 256x256 | 317.491 ms | 3.17 | 0.00 | 5% | ok | 1.2x (29%) |
| elementwise | exp | f64 | n=8388608 | 80.557 ms | 0.10 | 1.67 | 7% | ok | 1.1x (28%) |
| elementwise | log | f64 | n=8388608 | 100.216 ms | 0.08 | 1.34 | 5% | ok | 1.1x (28%) |
| elementwise | sin | f64 | n=8388608 | 134.889 ms | 0.06 | 1.00 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 65.869 ms | 0.13 | 2.04 | 8% | ok | 1.0x (26%) |
| elementwise | erf | f64 | n=8388608 | 106.680 ms | 0.08 | 1.26 | 5% | ok | 1.1x (27%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.207 ms | 2.17 | 22.55 | 92% | ok | 1.4x (34%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.973 ms | 1.33 | 13.80 | 56% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.201 ms | 6.75 | 14.20 | 58% | ok | 2.0x (50%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.735 ms | 0.28 | 0.98 | 4% | ok | 1.1x (27%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 9.309 ms | 0.22 | 1.79 | 7% | ok | 1.0x (25%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 34.171 ms | 0.33 | 0.02 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 54.081 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.278 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 94.582 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 110.984 ms | 0.20 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.135 ms | 0.57 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 66.742 ms | 0.03 | 0.00 | 0% | ok | 1.1x (28%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.824 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.418 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 111.425 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.517 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.067 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.019 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 26.840 ms | 0.42 | 0.03 | 1% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.925 ms | 0.39 | 1.57 | 6% | ok | 1.0x (24%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.077 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.659 ms | 1.05 | 0.07 | 2% | ok | 1.1x (27%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.953 ms | 0.39 | 1.55 | 6% | ok | 0.9x (23%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.991 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 80.134 ms | 0.28 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.675 ms | 0.13 | 0.53 | 2% | ok | 1.1x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.897 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 93.920 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.708 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 10.626 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 615.140 ms | 0.50 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.979 ms | 0.42 | 1.69 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 24.813 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 161.732 ms | 1.91 | 0.01 | 3% | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.925 ms | 0.43 | 1.70 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.596 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1612.240 ms | 0.38 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.523 ms | 0.11 | 0.46 | 2% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 14.851 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1143.888 ms | 0.54 | 0.00 | 1% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 30.327 ms | 0.11 | 0.44 | 2% | ok | 1.0x (26%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.380 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 108.438 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.642 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.405 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.495 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.997 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 55.864 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.446 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.179 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 35.583 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.677 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 31.845 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 37.742 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.104 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.565 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.147 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.883 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.776 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.075 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.592 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.934 ms | 1.35 | 15.16 | 62% | ok | 1.1x (27%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.282 ms | 1.15 | 12.85 | 52% | ok | 1.1x (28%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.692 ms | 0.97 | 10.90 | 44% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.293 ms | 1.19 | 9.00 | 37% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 212.407 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 211.906 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 134.729 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 12.134 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| mixed | dot | f32>f64 | n=8388608 | 7.260 ms | 2.31 | 9.24 | 38% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.814 ms | 20.95 | 0.33 | 31% | ok | 3.8x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.635 ms | 6.37 | 6.77 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.832 ms | 5.92 | 23.69 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.400 ms | 11.98 | 23.97 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.272 ms | 5.13 | 5.13 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.041 ms | 16.12 | 16.12 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.814 ms | 22.72 | 0.27 | n/a | ok | 2.0x (51%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.679 ms | 40.19 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.139 ms | 52.23 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.139 ms | 52.23 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 9.741 ms | 27.56 | 0.16 | n/a | ok | 4.0x (99%) |

## arduino-ventuno-q -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON: -march=armv8.2-a+dotprod+fp16+rcpc -mtune=cortex-a78c]`, SIMD NEON_WITHOUT_AES |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.6 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 16.8, f32 33.6 GFLOP/s |
| bandwidth | read 21.9, update 24.6, triad 22.9 GB/s (ceiling 24.6) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 94% | 104% | 0 |
| blas2 | 10 | 65% | 72% | 0 |
| blas3 | 10 | 49% | 58% | 0 |
| factor | 12 | 12% | 32% | 0 |
| eigen | 8 | 15% | 32% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 29% | 66% | 0 |
| direct | 19 | 2% | 7% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 3% | 10% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 56% | 0 |
| mg | 1 | 36% | 36% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 67% | 133% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.768 ms | 2.91 | 23.27 | 94% | ok |
| blas1 | dot | f32 | n=8388608 | 2.887 ms | 5.81 | 23.25 | 94% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.822 ms | 2.14 | 25.74 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.981 ms | 4.21 | 25.29 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 5.325 ms | 1.58 | 25.21 | 102% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.148 ms | 5.33 | 21.32 | 87% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.342 ms | 1.01 | 8.04 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.688 ms | 0.47 | 3.79 | 15% | ok |
| blas1 | sum | f64 | n=8388608 | 8.360 ms | 1.00 | 8.03 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 198.566 ms | 0.93 | 0.68 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 8.337 ms | 4.02 | 16.11 | 65% | ok |
| blas2 | gemv | f32 | n=4096 | 4.037 ms | 8.31 | 16.63 | 68% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.657 ms | 4.38 | 17.54 | 71% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.946 ms | 8.50 | 17.02 | 69% | ok |
| blas2 | symv | f64 | n=4096 | 12.781 ms | 2.63 | 5.26 | 21% | ok |
| blas2 | trmv | f64 | n=4096 | 4.637 ms | 3.62 | 14.49 | 59% | ok |
| blas2 | trmv | f32 | n=4096 | 2.709 ms | 6.19 | 12.40 | 50% | ok |
| blas2 | trsv | f64 | n=4096 | 4.988 ms | 3.36 | 13.47 | 55% | ok |
| blas2 | trsv | f32 | n=4096 | 2.781 ms | 6.03 | 12.08 | 49% | ok |
| blas2 | ger | f64 | n=4096 | 15.192 ms | 2.21 | 17.67 | 72% | ok |
| blas3 | gemm | f64 | 512x512x512 | 28.214 ms | 9.51 | 0.22 | 57% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.784 ms | 19.47 | 0.23 | 58% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 17.792 ms | 7.54 | 0.35 | 45% | ok |
| blas3 | symm | f64 | 512x512x512 | 28.978 ms | 9.26 | 0.22 | 55% | ok |
| blas3 | syrk | f64 | 512x512x512 | 16.448 ms | 8.16 | 0.25 | 49% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 32.565 ms | 8.24 | 0.19 | 49% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.845 ms | 7.52 | 0.29 | 45% | ok |
| blas3 | trsm | f64 | 512x512x512 | 17.859 ms | 7.52 | 0.29 | 45% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.511 ms | 15.77 | 0.31 | 47% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 26.559 ms | 0.20 | 1.60 | 6% | ok |
| factor | lu | f64 | 512x512 | 24.268 ms | 3.69 | 0.17 | 22% | ok |
| factor | lu | f32 | 512x512 | 13.027 ms | 6.87 | 0.16 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 22.518 ms | 1.99 | 0.19 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 20.995 ms | 2.13 | 0.10 | 6% | ok |
| factor | ldlt | f64 | 512x512 | 21.346 ms | 2.10 | 0.20 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 11.218 ms | 3.99 | 0.37 | 24% | ok |
| factor | qr | f64 | 512x512 | 245.601 ms | 0.73 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 61.009 ms | 2.93 | 0.07 | 17% | ok |
| factor | hessenberg | f64 | 512x512 | 339.084 ms | 1.32 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.503 ms | 0.70 | 2.79 | 11% | ok |
| factor | inv | f64 | 512x512 | 145.137 ms | 1.85 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.721 ms | 5.35 | 0.25 | 32% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 40.080 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 99.948 ms | 1.51 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 592.748 ms | 0.25 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.539 ms | 2.48 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 151.822 ms | 2.76 | 0.01 | 16% | ok |
| eigen | svd | f64 | 256x256 | 230.148 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 47.449 ms | 5.30 | 0.02 | 32% | ok |
| eigen | sylvester | f64 | 256x256 | 362.468 ms | 2.78 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 85.933 ms | 0.10 | 1.56 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 106.521 ms | 0.08 | 1.26 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 147.828 ms | 0.06 | 0.91 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 72.154 ms | 0.12 | 1.86 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 113.827 ms | 0.07 | 1.18 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.664 ms | 1.57 | 16.36 | 66% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.133 ms | 1.23 | 12.77 | 52% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.170 ms | 3.44 | 7.23 | 29% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.022 ms | 0.25 | 0.88 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.786 ms | 0.23 | 1.89 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 38.233 ms | 0.29 | 0.02 | 2% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 58.497 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.282 ms | 0.43 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 104.224 ms | 0.21 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 122.165 ms | 0.18 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 42.441 ms | 0.53 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 78.265 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.409 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.427 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 125.477 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.570 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.130 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.602 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 30.390 ms | 0.37 | 0.02 | 2% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.811 ms | 0.42 | 1.67 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.085 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.913 ms | 0.94 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.728 ms | 0.44 | 1.75 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.571 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 88.226 ms | 0.25 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.836 ms | 0.13 | 0.52 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 8.762 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 102.838 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.836 ms | 0.13 | 0.52 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 12.336 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 671.465 ms | 0.46 | 0.00 | 3% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.418 ms | 0.45 | 1.82 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.065 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 187.508 ms | 1.65 | 0.00 | 10% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.338 ms | 0.46 | 1.84 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.588 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1764.353 ms | 0.35 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.755 ms | 0.11 | 0.45 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.750 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1205.520 ms | 0.51 | 0.00 | 3% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.205 ms | 0.12 | 0.46 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 71.219 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 117.239 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 54.550 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 28.972 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.719 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 34.136 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 62.431 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.570 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.755 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 39.294 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.788 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 34.588 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 40.873 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.109 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 5.069 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 31.106 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.113 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.996 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.076 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.762 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.108 ms | 1.24 | 13.91 | 56% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.495 ms | 1.05 | 11.76 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.972 ms | 0.88 | 9.87 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.404 ms | 1.19 | 8.97 | 36% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 236.557 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 234.194 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 144.755 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.256 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.045 ms | 4.15 | 16.59 | 67% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 41.560 ms | 6.46 | 0.10 | 38% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.751 ms | 22.33 | 23.72 | 133% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.829 ms | 5.93 | 23.73 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.402 ms | 11.97 | 23.94 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.699 ms | 24.00 | 24.00 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.534 ms | 10.94 | 10.94 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.374 ms | 15.45 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 16.675 ms | 16.10 | 0.13 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 16.605 ms | 16.17 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.608 ms | 74.40 | 0.44 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.182 ms | 20.36 | 0.12 | n/a | ok |

## arduino-ventuno-q -- native-fast, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON: -march=armv8.2-a+dotprod+fp16+rcpc -mtune=cortex-a78c]`, SIMD NEON_WITHOUT_AES |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 67.6 GFLOP/s, f32 134.9 GFLOP/s |
| ... hardware, NEON | f64 67.6, f32 134.9 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 67.5, f32 134.9 GFLOP/s |
| bandwidth | read 22.8, update 24.8, triad 22.8 GB/s (ceiling 24.8) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 97% | 109% | 0 | 27% |
| blas2 | 10 | 57% | 88% | 0 | 26% |
| blas3 | 10 | 37% | 55% | 0 | 77% |
| factor | 12 | 6% | 35% | 0 | 35% |
| eigen | 8 | 4% | 13% | 0 | 28% |
| elementwise | 5 | 5% | 9% | 0 | 27% |
| sparse | 5 | 56% | 91% | 0 | 27% |
| direct | 19 | 1% | 6% | 0 | 27% |
| ordering | 5 | n/a | n/a | 0 | 26% |
| direct3d | 12 | 2% | 7% | 0 | 27% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 51% | 62% | 0 | 27% |
| mg | 1 | 36% | 36% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 73% | 98% | 0 | 27% |
| integer | 9 | n/a | n/a | 0 | 96% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.569 ms | 3.01 | 24.10 | 97% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.802 ms | 5.99 | 23.95 | 96% | ok | 1.0x (26%) |
| blas1 | axpy | f64 | n=8388608 | 8.181 ms | 2.05 | 24.61 | 99% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 3.941 ms | 4.26 | 25.54 | 103% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 4.956 ms | 1.69 | 27.08 | 109% | ok | 1.1x (27%) |
| blas1 | two_norm | f64 | n=8388608 | 2.636 ms | 6.36 | 25.46 | 102% | ok | 1.2x (30%) |
| blas1 | one_norm | f64 | n=8388608 | 7.558 ms | 1.11 | 8.88 | 36% | ok | 1.1x (28%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.719 ms | 0.53 | 4.27 | 17% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.897 ms | 1.06 | 8.50 | 34% | ok | 1.1x (26%) |
| blas1 | dot_k3 | f64 | n=8388608 | 182.617 ms | 1.01 | 0.73 | 3% | ok | 1.1x (27%) |
| blas2 | gemv | f64 | n=4096 | 6.603 ms | 5.08 | 20.34 | 82% | ok | 1.3x (32%) |
| blas2 | gemv | f32 | n=4096 | 3.058 ms | 10.97 | 21.95 | 88% | ok | 1.3x (33%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.653 ms | 4.38 | 17.55 | 71% | ok | 1.0x (25%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.792 ms | 7.00 | 14.01 | 56% | ok | 0.8x (21%) |
| blas2 | symv | f64 | n=4096 | 11.017 ms | 3.05 | 6.10 | 25% | ok | 1.2x (29%) |
| blas2 | trmv | f64 | n=4096 | 4.708 ms | 3.56 | 14.27 | 57% | ok | 1.0x (25%) |
| blas2 | trmv | f32 | n=4096 | 2.764 ms | 6.07 | 12.15 | 49% | ok | 1.0x (25%) |
| blas2 | trsv | f64 | n=4096 | 4.855 ms | 3.46 | 13.84 | 56% | ok | 1.0x (26%) |
| blas2 | trsv | f32 | n=4096 | 2.640 ms | 6.35 | 12.72 | 51% | ok | 1.1x (26%) |
| blas2 | ger | f64 | n=4096 | 14.516 ms | 2.31 | 18.50 | 74% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 7.395 ms | 36.30 | 0.85 | 54% | ok | 3.8x (95%) |
| blas3 | gemm | f32 | 512x512x512 | 3.585 ms | 74.88 | 0.88 | 55% | ok | 3.8x (96%) |
| blas3 | gemmt | f64 | 512x512x512 | 5.918 ms | 22.68 | 1.06 | 34% | ok | 3.0x (75%) |
| blas3 | symm | f64 | 512x512x512 | 8.170 ms | 32.86 | 0.77 | 49% | ok | 3.5x (89%) |
| blas3 | syrk | f64 | 512x512x512 | 5.317 ms | 25.24 | 0.79 | 37% | ok | 3.1x (77%) |
| blas3 | syr2k | f64 | 512x512x512 | 10.502 ms | 25.56 | 0.60 | 38% | ok | 3.1x (78%) |
| blas3 | trmm | f64 | 512x512x512 | 7.609 ms | 17.64 | 0.69 | 26% | ok | 2.3x (59%) |
| blas3 | trsm | f64 | 512x512x512 | 7.841 ms | 17.12 | 0.67 | 25% | ok | 2.3x (57%) |
| blas3 | trsm | f32 | 512x512x512 | 3.585 ms | 37.44 | 0.73 | 28% | ok | 2.4x (59%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 22.576 ms | 0.24 | 1.88 | 8% | ok | 1.2x (29%) |
| factor | lu | f64 | 512x512 | 17.931 ms | 4.99 | 0.23 | 7% | ok | 1.4x (34%) |
| factor | lu | f32 | 512x512 | 9.035 ms | 9.90 | 0.23 | 7% | ok | 1.4x (36%) |
| factor | cholesky | f64 | 512x512 | 20.056 ms | 2.23 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 18.771 ms | 2.38 | 0.11 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 19.004 ms | 2.35 | 0.22 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 10.467 ms | 4.27 | 0.40 | 6% | ok | 1.1x (27%) |
| factor | qr | f64 | 512x512 | 148.349 ms | 1.21 | 0.03 | 2% | ok | 1.7x (41%) |
| factor | lq | f64 | 512x512 | 36.818 ms | 4.86 | 0.11 | 7% | ok | 1.7x (41%) |
| factor | hessenberg | f64 | 512x512 | 203.969 ms | 2.19 | 0.02 | 3% | ok | 1.7x (42%) |
| factor | householder | f64 | 512x512 | 0.482 ms | 2.18 | 8.70 | 35% | ok | 3.1x (78%) |
| factor | inv | f64 | 512x512 | 134.285 ms | 2.00 | 0.03 | 3% | ok | 1.1x (27%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.097 ms | 7.40 | 0.35 | 11% | ok | 1.4x (35%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.600 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 87.093 ms | 1.73 | 0.01 | 3% | ok | 1.1x (29%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 530.936 ms | 0.28 | 0.00 | 0% | ok | 1.1x (28%) |
| eigen | eigenvalue | f64 | 256x256 | 60.655 ms | 2.77 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 135.534 ms | 3.09 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.929 ms | 1.79 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 28.610 ms | 8.80 | 0.04 | 13% | ok | 1.7x (41%) |
| eigen | sylvester | f64 | 256x256 | 317.061 ms | 3.17 | 0.00 | 5% | ok | 1.1x (29%) |
| elementwise | exp | f64 | n=8388608 | 74.927 ms | 0.11 | 1.79 | 7% | ok | 1.1x (29%) |
| elementwise | log | f64 | n=8388608 | 100.766 ms | 0.08 | 1.33 | 5% | ok | 1.1x (26%) |
| elementwise | sin | f64 | n=8388608 | 137.336 ms | 0.06 | 0.98 | 4% | ok | 1.1x (27%) |
| elementwise | sqrt | f64 | n=8388608 | 58.829 ms | 0.14 | 2.28 | 9% | ok | 1.2x (31%) |
| elementwise | erf | f64 | n=8388608 | 104.297 ms | 0.08 | 1.29 | 5% | ok | 1.1x (27%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.211 ms | 2.16 | 22.49 | 91% | ok | 1.4x (34%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.945 ms | 1.35 | 14.00 | 56% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.232 ms | 6.72 | 14.13 | 57% | ok | 2.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.935 ms | 0.27 | 0.96 | 4% | ok | 1.1x (27%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.953 ms | 0.22 | 1.86 | 7% | ok | 1.0x (25%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 34.856 ms | 0.32 | 0.02 | 0% | ok | 1.1x (27%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 52.965 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.512 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 95.065 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 112.752 ms | 0.20 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 38.611 ms | 0.58 | 0.02 | 1% | ok | 1.1x (27%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 71.022 ms | 0.03 | 0.00 | 0% | ok | 1.1x (28%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.156 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.999 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 113.310 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.522 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.861 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.095 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 27.206 ms | 0.41 | 0.03 | 1% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.245 ms | 0.34 | 1.35 | 5% | ok | 0.8x (20%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.152 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.836 ms | 1.03 | 0.07 | 2% | ok | 1.1x (27%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.149 ms | 0.35 | 1.41 | 6% | ok | 0.8x (20%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.028 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 80.411 ms | 0.28 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.580 ms | 0.14 | 0.54 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 8.021 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 95.279 ms | 0.23 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.578 ms | 0.14 | 0.54 | 2% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 10.933 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 616.709 ms | 0.50 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.968 ms | 0.42 | 1.69 | 7% | ok | 0.9x (23%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 25.278 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 169.083 ms | 1.83 | 0.01 | 3% | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.666 ms | 0.44 | 1.76 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.691 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1583.355 ms | 0.39 | 0.00 | 1% | ok | 1.1x (28%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.702 ms | 0.12 | 0.47 | 2% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 15.143 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1137.605 ms | 0.54 | 0.00 | 1% | ok | 1.1x (26%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.087 ms | 0.12 | 0.46 | 2% | ok | 1.0x (25%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 63.985 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 107.797 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.528 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.088 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.390 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.489 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 55.127 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.360 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.404 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 35.656 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.523 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 31.608 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 37.363 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.102 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.622 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 28.181 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.895 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.751 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.076 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.604 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.901 ms | 1.38 | 15.43 | 62% | ok | 1.1x (28%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.315 ms | 1.13 | 12.67 | 51% | ok | 1.1x (27%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.711 ms | 0.97 | 10.82 | 44% | ok | 1.1x (27%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.529 ms | 1.18 | 8.94 | 36% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 211.846 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 210.583 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 131.181 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.759 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 3.693 ms | 4.54 | 18.17 | 73% | ok | 1.1x (27%) |
| mixed | gemm | f32>f64 | 512x512x512 | 10.526 ms | 25.50 | 0.40 | 38% | ok | 3.9x (99%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.733 ms | 22.90 | 24.33 | 98% | ok | 1.0x (26%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.800 ms | 5.99 | 23.97 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.377 ms | 12.19 | 24.38 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.679 ms | 24.70 | 24.70 | n/a | ok | 1.0x (26%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.373 ms | 12.22 | 12.22 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 4.468 ms | 60.08 | 0.70 | n/a | ok | 3.9x (97%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.255 ms | 63.08 | 0.49 | n/a | ok | 3.9x (98%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.193 ms | 64.02 | 0.38 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.944 ms | 284.50 | 1.67 | n/a | ok | 3.8x (96%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.340 ms | 80.36 | 0.47 | n/a | ok | 3.9x (99%) |

## arduino-ventuno-q -- blas, 1 thread, class medium

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
| bandwidth | read 22.0, update 24.3, triad 22.7 GB/s (ceiling 24.3) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 87% | 105% | 0 |
| blas2 | 10 | 59% | 98% | 0 |
| blas3 | 10 | 83% | 92% | 0 |
| factor | 12 | 41% | 58% | 0 |
| eigen | 8 | 34% | 75% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 30% | 67% | 0 |
| direct | 23 | 2% | 11% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 15 | 7% | 42% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 57% | 0 |
| mg | 1 | 37% | 37% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 34% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.674 ms | 2.96 | 23.65 | 97% | ok |
| blas1 | dot | f32 | n=8388608 | 4.295 ms | 3.91 | 15.62 | 64% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.875 ms | 2.13 | 25.56 | 105% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.983 ms | 4.21 | 25.27 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 5.394 ms | 1.56 | 24.88 | 102% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.180 ms | 5.28 | 21.10 | 87% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.084 ms | 1.04 | 8.30 | 34% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.233 ms | 0.49 | 3.89 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.594 ms | 0.98 | 7.81 | 32% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 189.987 ms | 0.97 | 0.71 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 9.058 ms | 3.70 | 14.82 | 61% | ok |
| blas2 | gemv | f32 | n=4096 | 3.770 ms | 8.90 | 17.81 | 73% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 9.354 ms | 3.59 | 14.36 | 59% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.687 ms | 9.10 | 18.21 | 75% | ok |
| blas2 | symv | f64 | n=4096 | 11.760 ms | 2.85 | 5.71 | 23% | ok |
| blas2 | trmv | f64 | n=4096 | 6.313 ms | 2.66 | 10.64 | 44% | ok |
| blas2 | trmv | f32 | n=4096 | 3.300 ms | 5.08 | 10.18 | 42% | ok |
| blas2 | trsv | f64 | n=4096 | 6.410 ms | 2.62 | 10.48 | 43% | ok |
| blas2 | trsv | f32 | n=4096 | 3.254 ms | 5.16 | 10.32 | 42% | ok |
| blas2 | ger | f64 | n=4096 | 11.241 ms | 2.99 | 23.89 | 98% | ok |
| blas3 | gemm | f64 | 512x512x512 | 17.656 ms | 15.20 | 0.36 | 90% | ok |
| blas3 | gemm | f32 | 512x512x512 | 8.697 ms | 30.87 | 0.36 | 92% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 11.378 ms | 11.80 | 0.55 | 70% | ok |
| blas3 | symm | f64 | 512x512x512 | 17.927 ms | 14.97 | 0.35 | 89% | ok |
| blas3 | syrk | f64 | 512x512x512 | 9.793 ms | 13.71 | 0.43 | 82% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 19.234 ms | 13.96 | 0.33 | 83% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.071 ms | 14.80 | 0.58 | 88% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.652 ms | 13.91 | 0.54 | 83% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.002 ms | 26.83 | 0.52 | 80% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 26.369 ms | 0.20 | 1.61 | 7% | ok |
| factor | lu | f64 | 512x512 | 9.221 ms | 9.70 | 0.45 | 58% | ok |
| factor | lu | f32 | 512x512 | 5.546 ms | 16.13 | 0.38 | 48% | ok |
| factor | cholesky | f64 | 512x512 | 5.334 ms | 8.39 | 0.79 | 50% | ok |
| factor | cholesky | f32 | 512x512 | 3.234 ms | 13.83 | 0.65 | 41% | ok |
| factor | ldlt | f64 | 512x512 | 21.156 ms | 2.11 | 0.20 | 13% | ok |
| factor | ldlt_bk | f64 | 512x512 | 11.206 ms | 3.99 | 0.37 | 24% | ok |
| factor | qr | f64 | 512x512 | 19.746 ms | 9.06 | 0.21 | 54% | ok |
| factor | lq | f64 | 512x512 | 60.999 ms | 2.93 | 0.07 | 17% | ok |
| factor | hessenberg | f64 | 512x512 | 349.319 ms | 1.28 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.531 ms | 0.69 | 2.74 | 11% | ok |
| factor | inv | f64 | 512x512 | 133.507 ms | 2.01 | 0.03 | 12% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 10.059 ms | 8.90 | 0.42 | 53% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 7.380 ms | 3.03 | 0.14 | 18% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.350 ms | 1.55 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 592.400 ms | 0.25 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 45.026 ms | 3.73 | 0.02 | 22% | ok |
| eigen | schur | f64 | 256x256 | 63.109 ms | 6.65 | 0.02 | 40% | ok |
| eigen | svd | f64 | 256x256 | 29.286 ms | 12.60 | 0.04 | 75% | ok |
| eigen | expm | f64 | 256x256 | 37.469 ms | 6.72 | 0.03 | 40% | ok |
| eigen | sylvester | f64 | 256x256 | 173.872 ms | 5.79 | 0.01 | 34% | ok |
| elementwise | exp | f64 | n=8388608 | 84.180 ms | 0.10 | 1.59 | 7% | ok |
| elementwise | log | f64 | n=8388608 | 106.726 ms | 0.08 | 1.26 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 150.104 ms | 0.06 | 0.89 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 73.027 ms | 0.11 | 1.84 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 119.235 ms | 0.07 | 1.13 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.680 ms | 1.56 | 16.21 | 67% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.113 ms | 1.24 | 12.88 | 53% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.054 ms | 3.47 | 7.30 | 30% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 12.746 ms | 0.26 | 0.90 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 9.403 ms | 0.21 | 1.77 | 7% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 37.264 ms | 0.30 | 0.02 | 2% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 65.869 ms | 0.17 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.047 ms | 0.43 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 100.606 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 119.084 ms | 0.19 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 42.367 ms | 0.53 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 75.572 ms | 0.03 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.380 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.181 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 125.735 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.576 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.652 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.509 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 29.072 ms | 0.38 | 0.03 | 2% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.830 ms | 0.41 | 1.65 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.085 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.874 ms | 0.94 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.721 ms | 0.44 | 1.76 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.432 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 86.356 ms | 0.26 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 6.027 ms | 0.13 | 0.50 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 8.668 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 101.053 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 6.002 ms | 0.13 | 0.50 | 2% | ok |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 16.534 ms | 0.68 | 0.04 | 4% | ok |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 3.875 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 11.029 ms | 1.01 | 0.07 | 6% | ok |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.169 ms | 0.65 | 2.59 | 11% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 11.855 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 660.135 ms | 0.47 | 0.00 | 3% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.642 ms | 0.44 | 1.76 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 27.697 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 183.185 ms | 1.68 | 0.00 | 10% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.543 ms | 0.45 | 1.79 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.460 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1717.525 ms | 0.36 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 31.484 ms | 0.11 | 0.43 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.501 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1228.798 ms | 0.50 | 0.00 | 3% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 31.636 ms | 0.11 | 0.43 | 2% | ok |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 7.454 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 44.098 ms | 7.00 | 0.02 | 42% | ok |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 2.411 ms | 1.40 | 5.59 | 23% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 71.109 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 117.778 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 54.065 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 29.236 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.942 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 34.514 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 62.977 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.633 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.997 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 39.699 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.937 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 34.787 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 41.100 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.112 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.992 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 30.929 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.189 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 3.062 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.076 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.972 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.114 ms | 1.24 | 13.88 | 57% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.498 ms | 1.05 | 11.74 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.961 ms | 0.88 | 9.90 | 41% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.055 ms | 1.20 | 9.07 | 37% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 232.173 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 233.908 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 143.967 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.346 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 8.035 ms | 2.09 | 8.35 | 34% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 48.590 ms | 5.52 | 0.09 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.939 ms | 5.71 | 6.07 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.860 ms | 5.87 | 23.47 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.415 ms | 11.86 | 23.72 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.651 ms | 4.59 | 4.59 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.149 ms | 14.60 | 14.60 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 24.681 ms | 10.88 | 0.13 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 26.370 ms | 10.18 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 20.306 ms | 13.22 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 20.310 ms | 13.22 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 38.466 ms | 6.98 | 0.04 | n/a | ok |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

## arduino-ventuno-q -- blas, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 67.6 GFLOP/s, f32 134.9 GFLOP/s |
| ... hardware, NEON | f64 67.6, f32 134.9 GFLOP/s |
| ... this build, scalar | f64 33.9, f32 33.7 GFLOP/s |
| bandwidth | read 22.5, update 24.6, triad 22.4 GB/s (ceiling 24.6) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 97% | 115% | 0 | 28% |
| blas2 | 10 | 65% | 118% | 0 | 36% |
| blas3 | 10 | 74% | 81% | 0 | 88% |
| factor | 12 | 18% | 36% | 0 | 43% |
| eigen | 8 | 11% | 28% | 0 | 33% |
| elementwise | 5 | 5% | 8% | 0 | 27% |
| sparse | 5 | 58% | 92% | 0 | 28% |
| direct | 23 | 1% | 11% | 0 | 27% |
| ordering | 5 | n/a | n/a | 0 | 28% |
| direct3d | 15 | 3% | 24% | 0 | 27% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 52% | 60% | 0 | 27% |
| mg | 1 | 37% | 37% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 32% | 38% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 52% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.563 ms | 3.02 | 24.12 | 98% | ok | 1.0x (25%) |
| blas1 | dot | f32 | n=8388608 | 2.812 ms | 5.97 | 23.86 | 97% | ok | 1.5x (38%) |
| blas1 | axpy | f64 | n=8388608 | 8.075 ms | 2.08 | 24.93 | 102% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 4.207 ms | 3.99 | 23.92 | 97% | ok | 0.9x (24%) |
| blas1 | scale | f64 | n=8388608 | 4.773 ms | 1.76 | 28.12 | 115% | ok | 1.1x (28%) |
| blas1 | two_norm | f64 | n=8388608 | 2.620 ms | 6.40 | 25.61 | 104% | ok | 1.2x (30%) |
| blas1 | one_norm | f64 | n=8388608 | 7.368 ms | 1.14 | 9.11 | 37% | ok | 1.1x (27%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.414 ms | 0.54 | 4.35 | 18% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.385 ms | 1.14 | 9.09 | 37% | ok | 1.2x (29%) |
| blas1 | dot_k3 | f64 | n=8388608 | 182.180 ms | 1.01 | 0.74 | 3% | ok | 1.0x (26%) |
| blas2 | gemv | f64 | n=4096 | 5.807 ms | 5.78 | 23.13 | 94% | ok | 1.6x (39%) |
| blas2 | gemv | f32 | n=4096 | 2.611 ms | 12.85 | 25.71 | 105% | ok | 1.4x (36%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 8.415 ms | 3.99 | 15.96 | 65% | ok | 1.1x (28%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.814 ms | 6.97 | 13.95 | 57% | ok | 0.8x (19%) |
| blas2 | symv | f64 | n=4096 | 6.410 ms | 5.24 | 10.48 | 43% | ok | 1.8x (46%) |
| blas2 | trmv | f64 | n=4096 | 3.638 ms | 4.61 | 18.46 | 75% | ok | 1.7x (43%) |
| blas2 | trmv | f32 | n=4096 | 2.166 ms | 7.74 | 15.50 | 63% | ok | 1.5x (38%) |
| blas2 | trsv | f64 | n=4096 | 5.983 ms | 2.80 | 11.23 | 46% | ok | 1.1x (27%) |
| blas2 | trsv | f32 | n=4096 | 3.140 ms | 5.34 | 10.70 | 44% | ok | 1.0x (26%) |
| blas2 | ger | f64 | n=4096 | 9.252 ms | 3.63 | 29.02 | 118% | ok | 1.2x (30%) |
| blas3 | gemm | f64 | 512x512x512 | 5.022 ms | 53.45 | 1.25 | 79% | ok | 3.5x (88%) |
| blas3 | gemm | f32 | 512x512x512 | 2.457 ms | 109.24 | 1.28 | 81% | ok | 3.5x (88%) |
| blas3 | gemmt | f64 | 512x512x512 | 3.391 ms | 39.58 | 1.86 | 59% | ok | 3.4x (84%) |
| blas3 | symm | f64 | 512x512x512 | 5.033 ms | 53.34 | 1.25 | 79% | ok | 3.6x (89%) |
| blas3 | syrk | f64 | 512x512x512 | 3.260 ms | 41.17 | 1.29 | 61% | ok | 3.0x (75%) |
| blas3 | syr2k | f64 | 512x512x512 | 5.769 ms | 46.53 | 1.09 | 69% | ok | 3.3x (83%) |
| blas3 | trmm | f64 | 512x512x512 | 2.520 ms | 53.26 | 2.08 | 79% | ok | 3.6x (90%) |
| blas3 | trsm | f64 | 512x512x512 | 2.682 ms | 50.05 | 1.96 | 74% | ok | 3.6x (90%) |
| blas3 | trsm | f32 | 512x512x512 | 1.460 ms | 91.94 | 1.80 | 68% | ok | 3.4x (86%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 28.585 ms | 0.19 | 1.49 | 6% | ok | 0.9x (23%) |
| factor | lu | f64 | 512x512 | 4.208 ms | 21.26 | 1.00 | 31% | ok | 2.2x (55%) |
| factor | lu | f32 | 512x512 | 3.122 ms | 28.66 | 0.67 | 21% | ok | 1.8x (44%) |
| factor | cholesky | f64 | 512x512 | 2.962 ms | 15.11 | 1.42 | 22% | ok | 1.8x (45%) |
| factor | cholesky | f32 | 512x512 | 1.991 ms | 22.47 | 1.05 | 17% | ok | 1.6x (41%) |
| factor | ldlt | f64 | 512x512 | 19.164 ms | 2.33 | 0.22 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 10.486 ms | 4.27 | 0.40 | 6% | ok | 1.1x (27%) |
| factor | qr | f64 | 512x512 | 9.616 ms | 18.61 | 0.44 | 28% | ok | 2.1x (51%) |
| factor | lq | f64 | 512x512 | 36.907 ms | 4.85 | 0.11 | 7% | ok | 1.7x (41%) |
| factor | hessenberg | f64 | 512x512 | 201.563 ms | 2.22 | 0.02 | 3% | ok | 1.7x (43%) |
| factor | householder | f64 | 512x512 | 0.470 ms | 2.23 | 8.92 | 36% | ok | 3.3x (81%) |
| factor | inv | f64 | 512x512 | 115.377 ms | 2.33 | 0.04 | 3% | ok | 1.2x (29%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.201 ms | 12.43 | 0.58 | 18% | ok | 1.4x (35%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 5.390 ms | 4.15 | 0.19 | 6% | ok | 1.4x (34%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.953 ms | 1.74 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 546.814 ms | 0.28 | 0.00 | 0% | ok | 1.1x (27%) |
| eigen | eigenvalue | f64 | 256x256 | 37.310 ms | 4.50 | 0.03 | 7% | ok | 1.2x (30%) |
| eigen | schur | f64 | 256x256 | 47.930 ms | 8.75 | 0.02 | 13% | ok | 1.3x (33%) |
| eigen | svd | f64 | 256x256 | 19.632 ms | 18.80 | 0.05 | 28% | ok | 1.5x (37%) |
| eigen | expm | f64 | 256x256 | 23.221 ms | 10.84 | 0.05 | 16% | ok | 1.6x (40%) |
| eigen | sylvester | f64 | 256x256 | 136.035 ms | 7.40 | 0.01 | 11% | ok | 1.3x (32%) |
| elementwise | exp | f64 | n=8388608 | 81.893 ms | 0.10 | 1.64 | 7% | ok | 1.0x (26%) |
| elementwise | log | f64 | n=8388608 | 101.014 ms | 0.08 | 1.33 | 5% | ok | 1.1x (26%) |
| elementwise | sin | f64 | n=8388608 | 136.430 ms | 0.06 | 0.98 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 66.471 ms | 0.13 | 2.02 | 8% | ok | 1.1x (27%) |
| elementwise | erf | f64 | n=8388608 | 106.876 ms | 0.08 | 1.26 | 5% | ok | 1.1x (28%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.211 ms | 2.16 | 22.48 | 92% | ok | 1.4x (35%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.914 ms | 1.37 | 14.23 | 58% | ok | 1.1x (28%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.088 ms | 6.88 | 14.46 | 59% | ok | 2.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.588 ms | 0.28 | 0.99 | 4% | ok | 1.1x (27%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 9.169 ms | 0.22 | 1.82 | 7% | ok | 1.0x (26%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 33.678 ms | 0.33 | 0.02 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 59.412 ms | 0.19 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.141 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 93.179 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 112.096 ms | 0.20 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 38.525 ms | 0.58 | 0.02 | 1% | ok | 1.1x (27%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 67.628 ms | 0.03 | 0.00 | 0% | ok | 1.1x (28%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.046 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.365 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 112.088 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.523 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.493 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.033 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 26.075 ms | 0.43 | 0.03 | 1% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.249 ms | 0.34 | 1.34 | 5% | ok | 0.8x (20%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.289 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.660 ms | 1.05 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.118 ms | 0.36 | 1.43 | 6% | ok | 0.8x (20%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.011 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 79.877 ms | 0.28 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.718 ms | 0.13 | 0.53 | 2% | ok | 1.1x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.923 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 95.270 ms | 0.23 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.770 ms | 0.13 | 0.52 | 2% | ok | 1.0x (26%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 14.908 ms | 0.75 | 0.05 | 1% | ok | 1.1x (28%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 3.504 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 10.064 ms | 1.11 | 0.07 | 2% | ok | 1.1x (27%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.101 ms | 0.69 | 2.75 | 11% | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 10.690 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 606.110 ms | 0.51 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.868 ms | 0.43 | 1.71 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 25.056 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 164.658 ms | 1.87 | 0.01 | 3% | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.806 ms | 0.43 | 1.73 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.833 ms | 0.00 | 0.00 | n/a | ok | 1.1x (26%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1569.148 ms | 0.39 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.746 ms | 0.11 | 0.45 | 2% | ok | 1.1x (26%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 15.229 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1110.263 ms | 0.56 | 0.00 | 1% | ok | 1.1x (28%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.736 ms | 0.11 | 0.45 | 2% | ok | 1.1x (27%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 6.719 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 28.072 ms | 10.99 | 0.03 | 16% | ok | 1.6x (39%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 2.283 ms | 1.48 | 5.90 | 24% | ok | 1.1x (26%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 63.641 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 107.531 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.143 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.263 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.516 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.732 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 55.573 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.009 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 30.796 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 35.527 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.575 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 31.669 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 37.649 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.103 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.579 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.350 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.923 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.809 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.073 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.743 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.990 ms | 1.32 | 14.74 | 60% | ok | 1.1x (27%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.293 ms | 1.14 | 12.79 | 52% | ok | 1.1x (27%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.693 ms | 0.97 | 10.89 | 44% | ok | 1.1x (27%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.055 ms | 1.20 | 9.07 | 37% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 211.391 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 214.044 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 130.682 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.878 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.214 ms | 2.33 | 9.30 | 38% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.325 ms | 21.78 | 0.34 | 32% | ok | 3.9x (99%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.638 ms | 6.36 | 6.76 | 28% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.788 ms | 6.02 | 24.07 | n/a | ok | 1.0x (26%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.391 ms | 12.06 | 24.13 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.264 ms | 5.14 | 5.14 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.053 ms | 15.94 | 15.94 | n/a | ok | 1.1x (27%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.777 ms | 22.79 | 0.27 | n/a | ok | 2.1x (52%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.681 ms | 40.18 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.142 ms | 52.21 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.143 ms | 52.20 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 9.738 ms | 27.56 | 0.16 | n/a | ok | 3.9x (99%) |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

