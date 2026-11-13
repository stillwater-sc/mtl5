# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,4, CPUs 0,1,2,3, clock policy governor performance on the A78C policies; performance 902-2112 MHz on cpus 0,1; performance 940-2361 MHz on cpus 2,3, 2026-11-13T17:42:18Z.

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

## arduino-ventuno-q -- native, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bf315e4604cf` |
| FMA peak (ceiling) | f64 67.5 GFLOP/s, f32 134.9 GFLOP/s |
| ... hardware, NEON | f64 67.5, f32 134.9 GFLOP/s |
| ... this build, scalar | f64 33.9, f32 33.8 GFLOP/s |
| bandwidth | read 22.9, update 24.8, triad 22.8 GB/s (ceiling 24.8) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 97% | 114% | 0 | 28% |
| blas2 | 10 | 62% | 88% | 0 | 27% |
| blas3 | 10 | 8% | 17% | 0 | 28% |
| factor | 12 | 4% | 38% | 0 | 35% |
| eigen | 8 | 4% | 10% | 0 | 29% |
| elementwise | 5 | 6% | 9% | 0 | 28% |
| sparse | 5 | 55% | 92% | 0 | 28% |
| direct | 19 | 0% | 6% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 28% |
| direct3d | 12 | 2% | 7% | 0 | 28% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 28% |
| smoother | 3 | 53% | 59% | 0 | 28% |
| mg | 1 | 36% | 36% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 32% | 37% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 56% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.553 ms | 3.02 | 24.17 | 97% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.827 ms | 5.93 | 23.74 | 96% | ok | 1.6x (40%) |
| blas1 | axpy | f64 | n=8388608 | 8.337 ms | 2.01 | 24.15 | 97% | ok | 0.9x (24%) |
| blas1 | axpy | f32 | n=8388608 | 3.985 ms | 4.21 | 25.26 | 102% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 4.739 ms | 1.77 | 28.32 | 114% | ok | 1.1x (28%) |
| blas1 | two_norm | f64 | n=8388608 | 2.632 ms | 6.37 | 25.50 | 103% | ok | 1.2x (30%) |
| blas1 | one_norm | f64 | n=8388608 | 7.230 ms | 1.16 | 9.28 | 37% | ok | 1.2x (29%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.421 ms | 0.54 | 4.35 | 18% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.286 ms | 1.15 | 9.21 | 37% | ok | 1.2x (29%) |
| blas1 | dot_k3 | f64 | n=8388608 | 177.416 ms | 1.04 | 0.76 | 3% | ok | 1.0x (26%) |
| blas2 | gemv | f64 | n=4096 | 6.647 ms | 5.05 | 20.20 | 81% | ok | 1.2x (30%) |
| blas2 | gemv | f32 | n=4096 | 3.090 ms | 10.86 | 21.73 | 88% | ok | 1.8x (46%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.757 ms | 4.33 | 17.31 | 70% | ok | 1.3x (32%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.403 ms | 6.21 | 12.43 | 50% | ok | 0.8x (21%) |
| blas2 | symv | f64 | n=4096 | 15.432 ms | 2.17 | 4.35 | 18% | ok | 1.1x (28%) |
| blas2 | trmv | f64 | n=4096 | 4.398 ms | 3.81 | 15.27 | 62% | ok | 1.0x (25%) |
| blas2 | trmv | f32 | n=4096 | 4.449 ms | 3.77 | 7.55 | 30% | ok | 1.1x (27%) |
| blas2 | trsv | f64 | n=4096 | 4.456 ms | 3.77 | 15.08 | 61% | ok | 1.0x (25%) |
| blas2 | trsv | f32 | n=4096 | 4.357 ms | 3.85 | 7.71 | 31% | ok | 1.1x (27%) |
| blas2 | ger | f64 | n=4096 | 15.498 ms | 2.17 | 17.33 | 70% | ok | 1.0x (25%) |
| blas3 | gemm | f64 | 512x512x512 | 26.604 ms | 10.09 | 0.24 | 15% | ok | 2.0x (49%) |
| blas3 | gemm | f32 | 512x512x512 | 11.483 ms | 23.38 | 0.27 | 17% | ok | 2.2x (54%) |
| blas3 | gemmt | f64 | 512x512x512 | 38.825 ms | 3.46 | 0.16 | 5% | ok | 1.1x (27%) |
| blas3 | symm | f64 | 512x512x512 | 48.170 ms | 5.57 | 0.13 | 8% | ok | 1.1x (28%) |
| blas3 | syrk | f64 | 512x512x512 | 43.782 ms | 3.07 | 0.10 | 5% | ok | 0.9x (23%) |
| blas3 | syr2k | f64 | 512x512x512 | 83.861 ms | 3.20 | 0.08 | 5% | ok | 1.0x (24%) |
| blas3 | trmm | f64 | 512x512x512 | 23.764 ms | 5.65 | 0.22 | 8% | ok | 1.1x (28%) |
| blas3 | trsm | f64 | 512x512x512 | 23.350 ms | 5.75 | 0.22 | 9% | ok | 1.1x (28%) |
| blas3 | trsm | f32 | 512x512x512 | 11.426 ms | 11.75 | 0.23 | 9% | ok | 1.1x (27%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 23.188 ms | 0.23 | 1.83 | 7% | ok | 1.3x (33%) |
| factor | lu | f64 | 512x512 | 17.838 ms | 5.02 | 0.24 | 7% | ok | 1.4x (35%) |
| factor | lu | f32 | 512x512 | 8.895 ms | 10.06 | 0.24 | 7% | ok | 1.5x (37%) |
| factor | cholesky | f64 | 512x512 | 18.616 ms | 2.40 | 0.23 | 4% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 17.576 ms | 2.55 | 0.12 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 20.629 ms | 2.17 | 0.20 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 60.673 ms | 0.74 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 147.244 ms | 1.22 | 0.03 | 2% | ok | 1.7x (43%) |
| factor | lq | f64 | 512x512 | 36.240 ms | 4.94 | 0.12 | 7% | ok | 1.7x (42%) |
| factor | hessenberg | f64 | 512x512 | 202.473 ms | 2.21 | 0.02 | 3% | ok | 1.7x (44%) |
| factor | householder | f64 | 512x512 | 0.447 ms | 2.35 | 9.38 | 38% | ok | 3.5x (88%) |
| factor | inv | f64 | 512x512 | 126.331 ms | 2.12 | 0.03 | 3% | ok | 1.2x (30%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.243 ms | 7.31 | 0.34 | 11% | ok | 1.4x (34%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.564 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.955 ms | 1.74 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 528.038 ms | 0.29 | 0.00 | 0% | ok | 1.1x (29%) |
| eigen | eigenvalue | f64 | 256x256 | 60.003 ms | 2.80 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 133.910 ms | 3.13 | 0.01 | 5% | ok | 1.2x (29%) |
| eigen | svd | f64 | 256x256 | 205.360 ms | 1.80 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 38.270 ms | 6.58 | 0.03 | 10% | ok | 1.8x (44%) |
| eigen | sylvester | f64 | 256x256 | 317.094 ms | 3.17 | 0.00 | 5% | ok | 1.2x (29%) |
| elementwise | exp | f64 | n=8388608 | 75.182 ms | 0.11 | 1.79 | 7% | ok | 1.1x (27%) |
| elementwise | log | f64 | n=8388608 | 96.967 ms | 0.09 | 1.38 | 6% | ok | 1.1x (28%) |
| elementwise | sin | f64 | n=8388608 | 136.418 ms | 0.06 | 0.98 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 62.694 ms | 0.13 | 2.14 | 9% | ok | 1.1x (28%) |
| elementwise | erf | f64 | n=8388608 | 102.073 ms | 0.08 | 1.31 | 5% | ok | 1.2x (29%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.195 ms | 2.19 | 22.78 | 92% | ok | 1.4x (34%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.998 ms | 1.31 | 13.63 | 55% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.106 ms | 6.86 | 14.42 | 58% | ok | 2.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.606 ms | 0.28 | 0.99 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.721 ms | 0.23 | 1.91 | 8% | ok | 1.0x (24%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 163.320 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 58.614 ms | 0.19 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.422 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 92.192 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 124.235 ms | 0.18 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 38.920 ms | 0.57 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 185.159 ms | 0.01 | 0.00 | 0% | ok | 1.1x (27%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.749 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.504 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.631 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.270 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.329 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 155.572 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.266 ms | 0.33 | 1.33 | 5% | ok | 0.8x (20%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.587 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.323 ms | 1.08 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.200 ms | 0.34 | 1.37 | 6% | ok | 0.8x (20%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.913 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 77.786 ms | 0.29 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.567 ms | 0.14 | 0.54 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 8.993 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 106.968 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.687 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 19.413 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 14709.630 ms | 0.02 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.721 ms | 0.44 | 1.75 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 41.547 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 181.104 ms | 1.70 | 0.00 | 3% | ok | 1.1x (29%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.976 ms | 0.42 | 1.69 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.560 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1595.252 ms | 0.39 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.929 ms | 0.12 | 0.48 | 2% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 25.818 ms | 0.00 | 0.00 | n/a | ok | 1.2x (29%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2247.524 ms | 0.27 | 0.00 | 0% | ok | 1.1x (27%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.544 ms | 0.12 | 0.49 | 2% | ok | 1.0x (26%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 63.853 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 97.448 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.632 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.199 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.356 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.981 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 57.058 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.824 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.390 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.938 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.474 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.782 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.281 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.103 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.471 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilut | f64 | 128^2 grid | 27.442 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.873 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ildl | f64 | 128^2 grid | 2.733 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.594 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.005 ms | 1.31 | 14.63 | 59% | ok | 1.1x (26%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.253 ms | 1.16 | 13.01 | 53% | ok | 1.1x (28%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.673 ms | 0.98 | 10.97 | 44% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.299 ms | 1.19 | 9.00 | 36% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 209.029 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 242.032 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 134.843 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.886 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.224 ms | 2.32 | 9.29 | 37% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.334 ms | 21.76 | 0.34 | 32% | ok | 3.9x (99%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.634 ms | 6.37 | 6.77 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.812 ms | 5.97 | 23.86 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.373 ms | 12.22 | 24.43 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.274 ms | 5.12 | 5.12 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.063 ms | 15.78 | 15.78 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.479 ms | 23.38 | 0.27 | n/a | ok | 2.2x (56%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.679 ms | 40.19 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.136 ms | 52.27 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.137 ms | 52.26 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 9.738 ms | 27.57 | 0.16 | n/a | ok | 4.0x (99%) |

## arduino-ventuno-q -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON: -march=armv8.2-a+dotprod+fp16+rcpc -mtune=cortex-a78c]`, SIMD NEON_WITHOUT_AES |
| commit | `bf315e4604cf` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.6 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 16.8, f32 33.6 GFLOP/s |
| bandwidth | read 22.1, update 24.3, triad 22.9 GB/s (ceiling 24.3) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 96% | 105% | 0 |
| blas2 | 10 | 66% | 73% | 0 |
| blas3 | 10 | 45% | 58% | 0 |
| factor | 12 | 12% | 31% | 0 |
| eigen | 8 | 15% | 31% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 30% | 68% | 0 |
| direct | 19 | 2% | 7% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 9% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 56% | 0 |
| mg | 1 | 37% | 37% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 67% | 132% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.741 ms | 2.92 | 23.38 | 96% | ok |
| blas1 | dot | f32 | n=8388608 | 2.888 ms | 5.81 | 23.24 | 96% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.917 ms | 2.12 | 25.43 | 105% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.983 ms | 4.21 | 25.27 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 5.244 ms | 1.60 | 25.60 | 105% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.173 ms | 5.29 | 21.15 | 87% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.661 ms | 0.97 | 7.75 | 32% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.655 ms | 0.48 | 3.80 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.072 ms | 1.04 | 8.31 | 34% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 195.766 ms | 0.94 | 0.69 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 8.355 ms | 4.02 | 16.07 | 66% | ok |
| blas2 | gemv | f32 | n=4096 | 4.017 ms | 8.35 | 16.71 | 69% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.743 ms | 4.33 | 17.34 | 71% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.967 ms | 8.46 | 16.93 | 70% | ok |
| blas2 | symv | f64 | n=4096 | 17.009 ms | 1.97 | 3.95 | 16% | ok |
| blas2 | trmv | f64 | n=4096 | 4.919 ms | 3.41 | 13.66 | 56% | ok |
| blas2 | trmv | f32 | n=4096 | 2.749 ms | 6.10 | 12.22 | 50% | ok |
| blas2 | trsv | f64 | n=4096 | 4.761 ms | 3.52 | 14.11 | 58% | ok |
| blas2 | trsv | f32 | n=4096 | 2.806 ms | 5.98 | 11.97 | 49% | ok |
| blas2 | ger | f64 | n=4096 | 15.222 ms | 2.20 | 17.64 | 73% | ok |
| blas3 | gemm | f64 | 512x512x512 | 28.080 ms | 9.56 | 0.22 | 57% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.795 ms | 19.46 | 0.23 | 58% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 18.064 ms | 7.43 | 0.35 | 44% | ok |
| blas3 | symm | f64 | 512x512x512 | 29.081 ms | 9.23 | 0.22 | 55% | ok |
| blas3 | syrk | f64 | 512x512x512 | 40.520 ms | 3.31 | 0.10 | 20% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 81.501 ms | 3.29 | 0.08 | 20% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.616 ms | 7.62 | 0.30 | 45% | ok |
| blas3 | trsm | f64 | 512x512x512 | 18.102 ms | 7.41 | 0.29 | 44% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.510 ms | 15.77 | 0.31 | 47% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 28.571 ms | 0.19 | 1.49 | 6% | ok |
| factor | lu | f64 | 512x512 | 24.420 ms | 3.66 | 0.17 | 22% | ok |
| factor | lu | f32 | 512x512 | 13.050 ms | 6.86 | 0.16 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 22.404 ms | 2.00 | 0.19 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 21.026 ms | 2.13 | 0.10 | 6% | ok |
| factor | ldlt | f64 | 512x512 | 21.646 ms | 2.07 | 0.19 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 69.563 ms | 0.64 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 245.986 ms | 0.73 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 60.859 ms | 2.94 | 0.07 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 343.146 ms | 1.30 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.527 ms | 0.69 | 2.75 | 11% | ok |
| factor | inv | f64 | 512x512 | 145.791 ms | 1.84 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.951 ms | 5.28 | 0.25 | 31% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.782 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.494 ms | 1.55 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 608.132 ms | 0.25 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.614 ms | 2.48 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 151.337 ms | 2.77 | 0.01 | 17% | ok |
| eigen | svd | f64 | 256x256 | 230.004 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 47.851 ms | 5.26 | 0.02 | 31% | ok |
| eigen | sylvester | f64 | 256x256 | 359.454 ms | 2.80 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 85.810 ms | 0.10 | 1.56 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 108.576 ms | 0.08 | 1.24 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 150.238 ms | 0.06 | 0.89 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 72.595 ms | 0.12 | 1.85 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 108.822 ms | 0.08 | 1.23 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.651 ms | 1.59 | 16.49 | 68% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.179 ms | 1.20 | 12.49 | 51% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.256 ms | 3.42 | 7.18 | 30% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.465 ms | 0.24 | 0.85 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.776 ms | 0.23 | 1.90 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 183.514 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 59.518 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.522 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 102.640 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 138.525 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.602 ms | 0.51 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 201.098 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.472 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.491 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.702 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.325 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.999 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 176.350 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.854 ms | 0.41 | 1.63 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 13.501 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.525 ms | 0.97 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.717 ms | 0.44 | 1.76 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.575 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 87.024 ms | 0.26 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.791 ms | 0.13 | 0.52 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 10.197 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 118.892 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.858 ms | 0.13 | 0.52 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 21.687 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 16855.901 ms | 0.02 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.683 ms | 0.44 | 1.75 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 47.998 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 210.803 ms | 1.46 | 0.00 | 9% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.398 ms | 0.46 | 1.82 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.927 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1761.450 ms | 0.35 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.895 ms | 0.12 | 0.47 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 29.682 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2449.155 ms | 0.25 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.522 ms | 0.12 | 0.47 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.992 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 105.297 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 54.330 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 29.086 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.764 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 34.101 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 63.838 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.901 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.789 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 37.873 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 36.005 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.911 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 46.654 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.110 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.916 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 31.449 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.076 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.956 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.064 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.773 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.153 ms | 1.22 | 13.62 | 56% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.512 ms | 1.04 | 11.68 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.951 ms | 0.89 | 9.94 | 41% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.284 ms | 1.19 | 9.00 | 37% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 233.085 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 269.736 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 147.567 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.251 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.113 ms | 4.08 | 16.32 | 67% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 41.658 ms | 6.44 | 0.10 | 38% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.758 ms | 22.14 | 23.52 | 132% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.834 ms | 5.92 | 23.68 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.572 ms | 10.67 | 21.34 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.736 ms | 22.79 | 22.79 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.521 ms | 11.03 | 11.03 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.374 ms | 15.45 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 16.993 ms | 15.80 | 0.12 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 16.749 ms | 16.03 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.697 ms | 72.61 | 0.43 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.160 ms | 20.40 | 0.12 | n/a | ok |

## arduino-ventuno-q -- native-fast, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON: -march=armv8.2-a+dotprod+fp16+rcpc -mtune=cortex-a78c]`, SIMD NEON_WITHOUT_AES |
| commit | `bf315e4604cf` |
| FMA peak (ceiling) | f64 67.7 GFLOP/s, f32 135.3 GFLOP/s |
| ... hardware, NEON | f64 67.7, f32 135.1 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 67.5, f32 135.3 GFLOP/s |
| bandwidth | read 22.9, update 24.5, triad 22.3 GB/s (ceiling 24.5) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 98% | 116% | 0 | 27% |
| blas2 | 10 | 59% | 88% | 0 | 26% |
| blas3 | 10 | 27% | 55% | 0 | 59% |
| factor | 12 | 3% | 40% | 0 | 36% |
| eigen | 8 | 4% | 10% | 0 | 28% |
| elementwise | 5 | 6% | 8% | 0 | 27% |
| sparse | 5 | 54% | 88% | 0 | 28% |
| direct | 19 | 0% | 6% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 28% |
| direct3d | 12 | 2% | 7% | 0 | 28% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 53% | 62% | 0 | 28% |
| mg | 1 | 37% | 37% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 74% | 98% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 96% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.556 ms | 3.02 | 24.16 | 98% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.799 ms | 5.99 | 23.98 | 98% | ok | 1.0x (26%) |
| blas1 | axpy | f64 | n=8388608 | 7.945 ms | 2.11 | 25.34 | 103% | ok | 1.0x (25%) |
| blas1 | axpy | f32 | n=8388608 | 3.990 ms | 4.20 | 25.23 | 103% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 4.710 ms | 1.78 | 28.50 | 116% | ok | 1.1x (28%) |
| blas1 | two_norm | f64 | n=8388608 | 2.650 ms | 6.33 | 25.33 | 103% | ok | 1.2x (30%) |
| blas1 | one_norm | f64 | n=8388608 | 7.895 ms | 1.06 | 8.50 | 35% | ok | 1.1x (27%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.511 ms | 0.54 | 4.33 | 18% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.924 ms | 1.06 | 8.47 | 35% | ok | 1.0x (25%) |
| blas1 | dot_k3 | f64 | n=8388608 | 176.227 ms | 1.05 | 0.76 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 6.746 ms | 4.97 | 19.91 | 81% | ok | 1.2x (31%) |
| blas2 | gemv | f32 | n=4096 | 3.095 ms | 10.84 | 21.70 | 88% | ok | 1.3x (32%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.792 ms | 4.31 | 17.23 | 70% | ok | 1.0x (25%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.613 ms | 7.27 | 14.55 | 59% | ok | 0.9x (21%) |
| blas2 | symv | f64 | n=4096 | 15.580 ms | 2.15 | 4.31 | 18% | ok | 1.1x (27%) |
| blas2 | trmv | f64 | n=4096 | 4.833 ms | 3.47 | 13.90 | 57% | ok | 1.0x (25%) |
| blas2 | trmv | f32 | n=4096 | 2.533 ms | 6.62 | 13.26 | 54% | ok | 1.1x (27%) |
| blas2 | trsv | f64 | n=4096 | 4.880 ms | 3.44 | 13.76 | 56% | ok | 1.0x (24%) |
| blas2 | trsv | f32 | n=4096 | 2.703 ms | 6.21 | 12.43 | 51% | ok | 1.0x (26%) |
| blas2 | ger | f64 | n=4096 | 14.700 ms | 2.28 | 18.27 | 74% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 7.327 ms | 36.63 | 0.86 | 54% | ok | 3.8x (96%) |
| blas3 | gemm | f32 | 512x512x512 | 3.582 ms | 74.94 | 0.88 | 55% | ok | 3.9x (96%) |
| blas3 | gemmt | f64 | 512x512x512 | 5.899 ms | 22.75 | 1.07 | 34% | ok | 3.1x (77%) |
| blas3 | symm | f64 | 512x512x512 | 8.088 ms | 33.19 | 0.78 | 49% | ok | 3.6x (90%) |
| blas3 | syrk | f64 | 512x512x512 | 36.823 ms | 3.64 | 0.11 | 5% | ok | 1.1x (28%) |
| blas3 | syr2k | f64 | 512x512x512 | 77.421 ms | 3.47 | 0.08 | 5% | ok | 1.1x (26%) |
| blas3 | trmm | f64 | 512x512x512 | 7.674 ms | 17.49 | 0.68 | 26% | ok | 2.3x (57%) |
| blas3 | trsm | f64 | 512x512x512 | 7.935 ms | 16.91 | 0.66 | 25% | ok | 2.3x (57%) |
| blas3 | trsm | f32 | 512x512x512 | 3.616 ms | 37.12 | 0.73 | 27% | ok | 2.4x (59%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 23.543 ms | 0.23 | 1.80 | 7% | ok | 1.2x (30%) |
| factor | lu | f64 | 512x512 | 17.032 ms | 5.25 | 0.25 | 8% | ok | 1.4x (36%) |
| factor | lu | f32 | 512x512 | 8.765 ms | 10.21 | 0.24 | 8% | ok | 1.5x (37%) |
| factor | cholesky | f64 | 512x512 | 19.871 ms | 2.25 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 18.811 ms | 2.38 | 0.11 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 18.888 ms | 2.37 | 0.22 | 3% | ok | 1.1x (29%) |
| factor | ldlt_bk | f64 | 512x512 | 61.213 ms | 0.73 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 145.794 ms | 1.23 | 0.03 | 2% | ok | 1.7x (42%) |
| factor | lq | f64 | 512x512 | 36.676 ms | 4.88 | 0.11 | 7% | ok | 1.7x (41%) |
| factor | hessenberg | f64 | 512x512 | 199.564 ms | 2.24 | 0.02 | 3% | ok | 1.7x (43%) |
| factor | householder | f64 | 512x512 | 0.425 ms | 2.47 | 9.87 | 40% | ok | 3.6x (90%) |
| factor | inv | f64 | 512x512 | 128.344 ms | 2.09 | 0.03 | 3% | ok | 1.1x (28%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.429 ms | 7.20 | 0.34 | 11% | ok | 1.4x (34%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.631 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.817 ms | 1.74 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 528.588 ms | 0.29 | 0.00 | 0% | ok | 1.2x (29%) |
| eigen | eigenvalue | f64 | 256x256 | 60.615 ms | 2.77 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 135.354 ms | 3.10 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.437 ms | 1.80 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 36.019 ms | 6.99 | 0.03 | 10% | ok | 1.3x (33%) |
| eigen | sylvester | f64 | 256x256 | 317.116 ms | 3.17 | 0.00 | 5% | ok | 1.1x (28%) |
| elementwise | exp | f64 | n=8388608 | 81.320 ms | 0.10 | 1.65 | 7% | ok | 1.1x (26%) |
| elementwise | log | f64 | n=8388608 | 99.079 ms | 0.08 | 1.35 | 6% | ok | 1.1x (27%) |
| elementwise | sin | f64 | n=8388608 | 134.042 ms | 0.06 | 1.00 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 65.749 ms | 0.13 | 2.04 | 8% | ok | 1.1x (28%) |
| elementwise | erf | f64 | n=8388608 | 106.840 ms | 0.08 | 1.26 | 5% | ok | 1.0x (25%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.259 ms | 2.08 | 21.62 | 88% | ok | 1.3x (33%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.060 ms | 1.27 | 13.22 | 54% | ok | 1.1x (26%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.110 ms | 6.85 | 14.41 | 59% | ok | 2.0x (50%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.880 ms | 0.27 | 0.97 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.874 ms | 0.23 | 1.88 | 8% | ok | 1.0x (25%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 164.612 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 53.761 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.523 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 92.935 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 126.636 ms | 0.18 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.108 ms | 0.57 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 181.933 ms | 0.01 | 0.00 | 0% | ok | 1.1x (28%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.725 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.252 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.628 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.037 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.399 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 156.535 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.073 ms | 0.36 | 1.46 | 6% | ok | 0.9x (22%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.636 ms | 0.00 | 0.00 | n/a | ok | 1.2x (29%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.387 ms | 1.08 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.999 ms | 0.38 | 1.51 | 6% | ok | 0.9x (21%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.945 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 79.003 ms | 0.28 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.656 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.087 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 108.017 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.677 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 19.338 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 14983.658 ms | 0.02 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.871 ms | 0.43 | 1.71 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 42.122 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 186.431 ms | 1.66 | 0.00 | 2% | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.661 ms | 0.44 | 1.76 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.564 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1627.042 ms | 0.38 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.559 ms | 0.11 | 0.46 | 2% | ok | 1.0x (24%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 26.230 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2286.520 ms | 0.27 | 0.00 | 0% | ok | 1.1x (27%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.652 ms | 0.12 | 0.47 | 2% | ok | 1.0x (25%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.135 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 96.905 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.914 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.190 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.358 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.648 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.315 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.459 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.194 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.955 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.403 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.657 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.103 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.104 ms | 0.00 | 0.00 | n/a | ok | 1.1x (26%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.511 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.951 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.807 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.696 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.064 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.608 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.932 ms | 1.35 | 15.18 | 62% | ok | 1.1x (28%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.252 ms | 1.16 | 13.02 | 53% | ok | 1.1x (28%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.707 ms | 0.97 | 10.83 | 44% | ok | 1.1x (27%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.120 ms | 1.20 | 9.05 | 37% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 210.879 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 244.121 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 133.082 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.717 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 3.693 ms | 4.54 | 18.17 | 74% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 10.564 ms | 25.41 | 0.40 | 38% | ok | 3.9x (99%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.742 ms | 22.61 | 24.02 | 98% | ok | 1.0x (26%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.788 ms | 6.02 | 24.07 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.373 ms | 12.22 | 24.44 | n/a | ok | 1.1x (29%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.693 ms | 24.21 | 24.21 | n/a | ok | 1.1x (27%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.374 ms | 12.21 | 12.21 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 4.495 ms | 59.72 | 0.70 | n/a | ok | 3.9x (97%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.304 ms | 62.37 | 0.49 | n/a | ok | 3.9x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.192 ms | 64.04 | 0.38 | n/a | ok | 4.0x (100%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.961 ms | 279.41 | 1.64 | n/a | ok | 3.8x (96%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.324 ms | 80.75 | 0.47 | n/a | ok | 4.0x (99%) |

## arduino-ventuno-q -- blas, 1 thread, class medium

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
| bandwidth | read 22.2, update 24.2, triad 22.7 GB/s (ceiling 24.2) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 87% | 106% | 0 |
| blas2 | 10 | 59% | 73% | 0 |
| blas3 | 10 | 83% | 91% | 0 |
| factor | 12 | 41% | 58% | 0 |
| eigen | 8 | 34% | 75% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 30% | 68% | 0 |
| direct | 23 | 2% | 11% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 15 | 7% | 41% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 41% | 58% | 0 |
| mg | 1 | 38% | 38% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 34% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.700 ms | 2.94 | 23.55 | 97% | ok |
| blas1 | dot | f32 | n=8388608 | 4.543 ms | 3.69 | 14.77 | 61% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.835 ms | 2.14 | 25.70 | 106% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.975 ms | 4.22 | 25.33 | 105% | ok |
| blas1 | scale | f64 | n=8388608 | 5.478 ms | 1.53 | 24.50 | 101% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.183 ms | 5.27 | 21.08 | 87% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.286 ms | 1.01 | 8.10 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.365 ms | 0.48 | 3.86 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.283 ms | 1.01 | 8.10 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 191.774 ms | 0.96 | 0.70 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 9.105 ms | 3.69 | 14.75 | 61% | ok |
| blas2 | gemv | f32 | n=4096 | 3.790 ms | 8.85 | 17.72 | 73% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 9.432 ms | 3.56 | 14.24 | 59% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.795 ms | 8.84 | 17.69 | 73% | ok |
| blas2 | symv | f64 | n=4096 | 11.727 ms | 2.86 | 5.73 | 24% | ok |
| blas2 | trmv | f64 | n=4096 | 6.318 ms | 2.66 | 10.63 | 44% | ok |
| blas2 | trmv | f32 | n=4096 | 3.318 ms | 5.06 | 10.12 | 42% | ok |
| blas2 | trsv | f64 | n=4096 | 6.420 ms | 2.61 | 10.46 | 43% | ok |
| blas2 | trsv | f32 | n=4096 | 3.272 ms | 5.13 | 10.26 | 42% | ok |
| blas2 | ger | f64 | n=4096 | 15.536 ms | 2.16 | 17.28 | 71% | ok |
| blas3 | gemm | f64 | 512x512x512 | 18.080 ms | 14.85 | 0.35 | 88% | ok |
| blas3 | gemm | f32 | 512x512x512 | 8.794 ms | 30.53 | 0.36 | 91% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 11.346 ms | 11.83 | 0.55 | 70% | ok |
| blas3 | symm | f64 | 512x512x512 | 18.122 ms | 14.81 | 0.35 | 88% | ok |
| blas3 | syrk | f64 | 512x512x512 | 9.803 ms | 13.69 | 0.43 | 81% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 19.229 ms | 13.96 | 0.33 | 83% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.147 ms | 14.67 | 0.57 | 87% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.720 ms | 13.81 | 0.54 | 82% | ok |
| blas3 | trsm | f32 | 512x512x512 | 4.998 ms | 26.85 | 0.52 | 80% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 26.687 ms | 0.20 | 1.59 | 7% | ok |
| factor | lu | f64 | 512x512 | 9.170 ms | 9.76 | 0.46 | 58% | ok |
| factor | lu | f32 | 512x512 | 5.527 ms | 16.19 | 0.38 | 48% | ok |
| factor | cholesky | f64 | 512x512 | 5.399 ms | 8.29 | 0.78 | 49% | ok |
| factor | cholesky | f32 | 512x512 | 3.226 ms | 13.87 | 0.65 | 41% | ok |
| factor | ldlt | f64 | 512x512 | 22.516 ms | 1.99 | 0.19 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 68.245 ms | 0.66 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 19.821 ms | 9.03 | 0.21 | 54% | ok |
| factor | lq | f64 | 512x512 | 60.793 ms | 2.94 | 0.07 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 385.983 ms | 1.16 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 1.561 ms | 0.67 | 2.69 | 11% | ok |
| factor | inv | f64 | 512x512 | 128.408 ms | 2.09 | 0.03 | 12% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 10.269 ms | 8.71 | 0.41 | 52% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 7.343 ms | 3.05 | 0.14 | 18% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.569 ms | 1.55 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 595.448 ms | 0.25 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 45.161 ms | 3.71 | 0.02 | 22% | ok |
| eigen | schur | f64 | 256x256 | 63.532 ms | 6.60 | 0.02 | 39% | ok |
| eigen | svd | f64 | 256x256 | 29.272 ms | 12.61 | 0.04 | 75% | ok |
| eigen | expm | f64 | 256x256 | 37.916 ms | 6.64 | 0.03 | 40% | ok |
| eigen | sylvester | f64 | 256x256 | 176.310 ms | 5.71 | 0.01 | 34% | ok |
| elementwise | exp | f64 | n=8388608 | 84.867 ms | 0.10 | 1.58 | 7% | ok |
| elementwise | log | f64 | n=8388608 | 103.798 ms | 0.08 | 1.29 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 146.609 ms | 0.06 | 0.92 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 68.656 ms | 0.12 | 1.95 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 116.494 ms | 0.07 | 1.15 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.655 ms | 1.58 | 16.45 | 68% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.170 ms | 1.21 | 12.55 | 52% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.057 ms | 3.47 | 7.30 | 30% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 12.814 ms | 0.25 | 0.90 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.808 ms | 0.23 | 1.89 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 182.162 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 57.917 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.307 ms | 0.43 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 102.855 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 140.812 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.823 ms | 0.51 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 201.379 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.490 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.493 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.698 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.505 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.897 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 173.713 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.848 ms | 0.41 | 1.64 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.784 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.507 ms | 0.97 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.755 ms | 0.43 | 1.72 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.441 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 87.099 ms | 0.26 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.977 ms | 0.13 | 0.51 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.966 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 119.957 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.992 ms | 0.13 | 0.50 | 2% | ok |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 16.934 ms | 0.66 | 0.04 | 4% | ok |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 3.895 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 11.352 ms | 0.99 | 0.06 | 6% | ok |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.158 ms | 0.65 | 2.61 | 11% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 21.618 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 16471.873 ms | 0.02 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.755 ms | 0.43 | 1.74 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 47.526 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 201.235 ms | 1.53 | 0.00 | 9% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.557 ms | 0.45 | 1.78 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.728 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1721.167 ms | 0.36 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.987 ms | 0.12 | 0.46 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 29.668 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2463.756 ms | 0.25 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.588 ms | 0.12 | 0.47 | 2% | ok |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 7.489 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 45.179 ms | 6.83 | 0.02 | 41% | ok |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 2.419 ms | 1.39 | 5.57 | 23% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 71.026 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 105.157 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 53.686 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 29.064 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.728 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 34.187 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 64.233 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 64.110 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 35.132 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 37.687 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.893 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.918 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 47.305 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.110 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 5.066 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 31.057 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.131 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 3.035 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.065 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.970 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.074 ms | 1.26 | 14.14 | 58% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 4.790 ms | 0.55 | 6.12 | 25% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.969 ms | 0.88 | 9.88 | 41% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 31.810 ms | 1.21 | 9.14 | 38% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 229.496 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 233.902 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 147.618 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.375 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 8.065 ms | 2.08 | 8.32 | 34% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 48.771 ms | 5.50 | 0.09 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.946 ms | 5.70 | 6.05 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.862 ms | 5.86 | 23.45 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.396 ms | 12.02 | 24.04 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.649 ms | 4.60 | 4.60 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.144 ms | 14.67 | 14.67 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 24.695 ms | 10.87 | 0.13 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 26.377 ms | 10.18 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 20.308 ms | 13.22 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 20.308 ms | 13.22 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 38.457 ms | 6.98 | 0.04 | n/a | ok |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

## arduino-ventuno-q -- blas, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bf315e4604cf` |
| FMA peak (ceiling) | f64 67.6 GFLOP/s, f32 135.0 GFLOP/s |
| ... hardware, NEON | f64 67.6, f32 135.0 GFLOP/s |
| ... this build, scalar | f64 33.8, f32 33.8 GFLOP/s |
| bandwidth | read 22.8, update 24.9, triad 22.5 GB/s (ceiling 24.9) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 97% | 114% | 0 | 28% |
| blas2 | 10 | 63% | 105% | 0 | 36% |
| blas3 | 10 | 74% | 81% | 0 | 89% |
| factor | 12 | 18% | 36% | 0 | 44% |
| eigen | 8 | 11% | 28% | 0 | 33% |
| elementwise | 5 | 6% | 9% | 0 | 28% |
| sparse | 5 | 55% | 91% | 0 | 27% |
| direct | 23 | 1% | 11% | 0 | 27% |
| ordering | 4 | n/a | n/a | 0 | 27% |
| direct3d | 15 | 2% | 24% | 0 | 27% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 44% | 59% | 0 | 28% |
| mg | 1 | 36% | 36% | 0 | 24% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 31% | 37% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 53% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.560 ms | 3.02 | 24.14 | 97% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.808 ms | 5.97 | 23.90 | 96% | ok | 1.6x (40%) |
| blas1 | axpy | f64 | n=8388608 | 8.054 ms | 2.08 | 25.00 | 100% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 3.932 ms | 4.27 | 25.60 | 103% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 4.703 ms | 1.78 | 28.54 | 114% | ok | 1.2x (29%) |
| blas1 | two_norm | f64 | n=8388608 | 2.618 ms | 6.41 | 25.63 | 103% | ok | 1.2x (30%) |
| blas1 | one_norm | f64 | n=8388608 | 7.294 ms | 1.15 | 9.20 | 37% | ok | 1.1x (28%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.472 ms | 0.54 | 4.34 | 17% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.592 ms | 1.10 | 8.84 | 35% | ok | 1.1x (27%) |
| blas1 | dot_k3 | f64 | n=8388608 | 171.935 ms | 1.07 | 0.78 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 5.106 ms | 6.57 | 26.30 | 105% | ok | 1.8x (45%) |
| blas2 | gemv | f32 | n=4096 | 2.600 ms | 12.91 | 25.83 | 104% | ok | 1.5x (36%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 8.796 ms | 3.81 | 15.27 | 61% | ok | 1.1x (27%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.545 ms | 7.38 | 14.77 | 59% | ok | 0.8x (21%) |
| blas2 | symv | f64 | n=4096 | 6.368 ms | 5.27 | 10.55 | 42% | ok | 1.8x (46%) |
| blas2 | trmv | f64 | n=4096 | 3.638 ms | 4.61 | 18.47 | 74% | ok | 1.7x (43%) |
| blas2 | trmv | f32 | n=4096 | 2.153 ms | 7.79 | 15.60 | 63% | ok | 1.5x (39%) |
| blas2 | trsv | f64 | n=4096 | 5.895 ms | 2.85 | 11.40 | 46% | ok | 1.1x (27%) |
| blas2 | trsv | f32 | n=4096 | 3.148 ms | 5.33 | 10.67 | 43% | ok | 1.0x (26%) |
| blas2 | ger | f64 | n=4096 | 14.905 ms | 2.25 | 18.01 | 72% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 5.040 ms | 53.26 | 1.25 | 79% | ok | 3.6x (90%) |
| blas3 | gemm | f32 | 512x512x512 | 2.455 ms | 109.36 | 1.28 | 81% | ok | 3.6x (90%) |
| blas3 | gemmt | f64 | 512x512x512 | 3.467 ms | 38.72 | 1.81 | 57% | ok | 3.3x (82%) |
| blas3 | symm | f64 | 512x512x512 | 5.102 ms | 52.62 | 1.23 | 78% | ok | 3.6x (89%) |
| blas3 | syrk | f64 | 512x512x512 | 3.270 ms | 41.04 | 1.28 | 61% | ok | 3.0x (75%) |
| blas3 | syr2k | f64 | 512x512x512 | 5.742 ms | 46.75 | 1.10 | 69% | ok | 3.3x (84%) |
| blas3 | trmm | f64 | 512x512x512 | 2.519 ms | 53.28 | 2.08 | 79% | ok | 3.6x (91%) |
| blas3 | trsm | f64 | 512x512x512 | 2.670 ms | 50.26 | 1.96 | 74% | ok | 3.6x (91%) |
| blas3 | trsm | f32 | 512x512x512 | 1.453 ms | 92.40 | 1.80 | 68% | ok | 3.4x (86%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 23.299 ms | 0.23 | 1.82 | 7% | ok | 1.1x (29%) |
| factor | lu | f64 | 512x512 | 4.171 ms | 21.45 | 1.01 | 32% | ok | 2.2x (55%) |
| factor | lu | f32 | 512x512 | 3.151 ms | 28.40 | 0.67 | 21% | ok | 1.8x (44%) |
| factor | cholesky | f64 | 512x512 | 3.061 ms | 14.61 | 1.37 | 22% | ok | 1.8x (44%) |
| factor | cholesky | f32 | 512x512 | 2.030 ms | 22.04 | 1.03 | 16% | ok | 1.6x (40%) |
| factor | ldlt | f64 | 512x512 | 20.285 ms | 2.21 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 60.801 ms | 0.74 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 9.793 ms | 18.27 | 0.43 | 27% | ok | 2.0x (51%) |
| factor | lq | f64 | 512x512 | 36.537 ms | 4.90 | 0.11 | 7% | ok | 1.7x (42%) |
| factor | hessenberg | f64 | 512x512 | 200.759 ms | 2.23 | 0.02 | 3% | ok | 1.9x (48%) |
| factor | householder | f64 | 512x512 | 0.463 ms | 2.27 | 9.06 | 36% | ok | 3.4x (84%) |
| factor | inv | f64 | 512x512 | 111.848 ms | 2.40 | 0.04 | 4% | ok | 1.1x (29%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.454 ms | 12.00 | 0.56 | 18% | ok | 1.4x (34%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 5.388 ms | 4.15 | 0.19 | 6% | ok | 1.4x (34%) |
| eigen | eigen_symmetric | f64 | 256x256 | 87.186 ms | 1.73 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 535.913 ms | 0.28 | 0.00 | 0% | ok | 1.1x (28%) |
| eigen | eigenvalue | f64 | 256x256 | 37.284 ms | 4.50 | 0.03 | 7% | ok | 1.2x (30%) |
| eigen | schur | f64 | 256x256 | 48.084 ms | 8.72 | 0.02 | 13% | ok | 1.3x (33%) |
| eigen | svd | f64 | 256x256 | 19.720 ms | 18.72 | 0.05 | 28% | ok | 1.5x (37%) |
| eigen | expm | f64 | 256x256 | 23.770 ms | 10.59 | 0.04 | 16% | ok | 1.6x (40%) |
| eigen | sylvester | f64 | 256x256 | 137.018 ms | 7.35 | 0.01 | 11% | ok | 1.3x (32%) |
| elementwise | exp | f64 | n=8388608 | 73.529 ms | 0.11 | 1.83 | 7% | ok | 1.2x (29%) |
| elementwise | log | f64 | n=8388608 | 94.928 ms | 0.09 | 1.41 | 6% | ok | 1.1x (27%) |
| elementwise | sin | f64 | n=8388608 | 130.939 ms | 0.06 | 1.03 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 60.739 ms | 0.14 | 2.21 | 9% | ok | 1.1x (28%) |
| elementwise | erf | f64 | n=8388608 | 103.454 ms | 0.08 | 1.30 | 5% | ok | 1.1x (28%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.200 ms | 2.18 | 22.70 | 91% | ok | 1.4x (34%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.998 ms | 1.31 | 13.63 | 55% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.199 ms | 6.76 | 14.20 | 57% | ok | 1.9x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.932 ms | 0.27 | 0.96 | 4% | ok | 1.1x (27%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 9.006 ms | 0.22 | 1.85 | 7% | ok | 1.0x (24%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 162.927 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 52.170 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.433 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 94.318 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 129.495 ms | 0.17 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.819 ms | 0.56 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 185.564 ms | 0.01 | 0.00 | 0% | ok | 1.1x (27%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.726 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.415 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.631 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.698 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.396 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 155.142 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.997 ms | 0.38 | 1.51 | 6% | ok | 0.9x (23%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.677 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.394 ms | 1.08 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.181 ms | 0.35 | 1.39 | 6% | ok | 0.8x (20%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.003 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 80.892 ms | 0.28 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 6.236 ms | 0.12 | 0.49 | 2% | ok | 1.0x (24%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.330 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 111.051 ms | 0.20 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.717 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 14.939 ms | 0.75 | 0.05 | 1% | ok | 1.1x (28%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 3.507 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 10.156 ms | 1.10 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.100 ms | 0.69 | 2.75 | 11% | ok | 1.1x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 19.109 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 14674.879 ms | 0.02 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.995 ms | 0.42 | 1.69 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 41.724 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 183.697 ms | 1.68 | 0.00 | 2% | ok | 1.1x (27%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.858 ms | 0.43 | 1.71 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.596 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1600.214 ms | 0.39 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.674 ms | 0.12 | 0.49 | 2% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 25.851 ms | 0.00 | 0.00 | n/a | ok | 1.1x (29%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2379.090 ms | 0.26 | 0.00 | 0% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.330 ms | 0.12 | 0.48 | 2% | ok | 1.0x (25%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 6.694 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 28.466 ms | 10.84 | 0.03 | 16% | ok | 1.6x (40%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 2.272 ms | 1.48 | 5.93 | 24% | ok | 1.1x (27%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 63.911 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 96.228 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.913 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.365 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.407 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.888 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.894 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.663 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.278 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 34.001 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.635 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.927 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.893 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.103 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.649 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.618 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.879 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.779 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.726 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.998 ms | 1.31 | 14.68 | 59% | ok | 1.0x (26%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 4.304 ms | 0.61 | 6.81 | 27% | ok | 1.1x (28%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.667 ms | 0.98 | 10.99 | 44% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.654 ms | 1.18 | 8.90 | 36% | ok | 1.0x (24%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 207.240 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 211.894 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 130.721 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 12.112 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.228 ms | 2.32 | 9.28 | 37% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.767 ms | 21.03 | 0.33 | 31% | ok | 3.8x (96%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.639 ms | 6.36 | 6.76 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.835 ms | 5.92 | 23.67 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.388 ms | 12.09 | 24.17 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.263 ms | 5.14 | 5.14 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.042 ms | 16.10 | 16.10 | n/a | ok | 1.1x (27%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.743 ms | 22.86 | 0.27 | n/a | ok | 2.1x (53%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.679 ms | 40.19 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.147 ms | 52.16 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.151 ms | 52.11 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 9.758 ms | 27.51 | 0.16 | n/a | ok | 3.9x (99%) |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

