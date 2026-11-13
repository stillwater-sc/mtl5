# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,8, CPUs 0,2,4,6,8,10,12,14, clock policy governor performance on the P-cores, turbo on; performance 800-4900 MHz on cpus 0,2,8,10,12,14; performance 800-5000 MHz on cpus 4,6, 2026-11-13T06:08:37Z.

## i7-12700k -- native, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `8867b44b3f52` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.4 GFLOP/s |
| bandwidth | read 33.3, update 38.7, triad 32.6 GB/s (ceiling 38.7) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 44% | 97% | 0 |
| blas2 | 10 | 79% | 96% | 0 |
| blas3 | 10 | 17% | 18% | 0 |
| factor | 12 | 8% | 19% | 0 |
| eigen | 8 | 6% | 12% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 49% | 95% | 0 |
| direct | 19 | 1% | 31% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 24% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 62% | 77% | 0 |
| mg | 1 | 57% | 57% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 35% | 85% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.610 ms | 2.99 | 23.92 | 62% | ok |
| blas1 | dot | f32 | n=8388608 | 4.212 ms | 3.98 | 15.93 | 41% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.300 ms | 2.66 | 31.95 | 83% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.683 ms | 3.58 | 21.49 | 56% | ok |
| blas1 | scale | f64 | n=8388608 | 3.570 ms | 2.35 | 37.60 | 97% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.845 ms | 3.46 | 13.85 | 36% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.007 ms | 2.09 | 16.75 | 43% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.033 ms | 1.19 | 9.54 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 3.979 ms | 2.11 | 16.87 | 44% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 76.337 ms | 2.42 | 1.76 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.625 ms | 9.26 | 37.04 | 96% | ok |
| blas2 | gemv | f32 | n=4096 | 2.756 ms | 12.17 | 24.36 | 63% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 4.703 ms | 7.13 | 28.55 | 74% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.203 ms | 15.23 | 30.48 | 79% | ok |
| blas2 | symv | f64 | n=4096 | 6.600 ms | 5.08 | 10.18 | 26% | ok |
| blas2 | trmv | f64 | n=4096 | 2.017 ms | 8.32 | 33.30 | 86% | ok |
| blas2 | trmv | f32 | n=4096 | 1.423 ms | 11.79 | 23.60 | 61% | ok |
| blas2 | trsv | f64 | n=4096 | 1.970 ms | 8.52 | 34.11 | 88% | ok |
| blas2 | trsv | f32 | n=4096 | 1.387 ms | 12.10 | 24.22 | 63% | ok |
| blas2 | ger | f64 | n=4096 | 7.666 ms | 4.38 | 35.03 | 90% | ok |
| blas3 | gemm | f64 | 512x512x512 | 20.043 ms | 13.39 | 0.31 | 17% | ok |
| blas3 | gemm | f32 | 512x512x512 | 10.617 ms | 25.28 | 0.30 | 16% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 11.040 ms | 12.16 | 0.57 | 16% | ok |
| blas3 | symm | f64 | 512x512x512 | 18.549 ms | 14.47 | 0.34 | 18% | ok |
| blas3 | syrk | f64 | 512x512x512 | 13.518 ms | 9.93 | 0.31 | 13% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 24.546 ms | 10.94 | 0.26 | 14% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.948 ms | 13.49 | 0.53 | 17% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.754 ms | 13.76 | 0.54 | 18% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.137 ms | 26.13 | 0.51 | 17% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.102 ms | 0.41 | 3.24 | 8% | ok |
| factor | lu | f64 | 512x512 | 11.572 ms | 7.73 | 0.36 | 10% | ok |
| factor | lu | f32 | 512x512 | 4.527 ms | 19.77 | 0.46 | 13% | ok |
| factor | cholesky | f64 | 512x512 | 7.332 ms | 6.10 | 0.57 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.605 ms | 6.77 | 0.32 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.466 ms | 5.99 | 0.56 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 47.408 ms | 0.94 | 0.09 | 1% | ok |
| factor | qr | f64 | 512x512 | 229.944 ms | 0.78 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 25.829 ms | 6.93 | 0.16 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 270.702 ms | 1.65 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.622 ms | 0.65 | 2.59 | 7%\* | ok |
| factor | inv | f64 | 512x512 | 54.056 ms | 4.97 | 0.08 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.175 ms | 14.49 | 0.68 | 19% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 18.339 ms | 1.22 | 0.06 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 59.157 ms | 2.55 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 752.217 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 34.236 ms | 4.90 | 0.03 | 6% | ok |
| eigen | schur | f64 | 256x256 | 86.008 ms | 4.88 | 0.01 | 6% | ok |
| eigen | svd | f64 | 256x256 | 112.238 ms | 3.29 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 26.494 ms | 9.50 | 0.04 | 12% | ok |
| eigen | sylvester | f64 | 256x256 | 190.677 ms | 5.28 | 0.01 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 39.358 ms | 0.21 | 3.41 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 45.822 ms | 0.18 | 2.93 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 60.010 ms | 0.14 | 2.24 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 28.988 ms | 0.29 | 4.63 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 201.496 ms | 0.04 | 0.67 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.743 ms | 3.52 | 36.67 | 95% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.929 ms | 2.82 | 29.31 | 76% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.669 ms | 8.97 | 18.86 | 49% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.853 ms | 0.67 | 2.37 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.248 ms | 1.61 | 13.34 | 34%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 91.297 ms | 0.12 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.489 ms | 0.44 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.890 ms | 1.26 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.994 ms | 0.56 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 53.545 ms | 0.42 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 17.422 ms | 1.28 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 99.476 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.389 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.021 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.226 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.546 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.362 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 88.520 ms | 0.13 | 0.01 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.248 ms | 3.04 | 12.17 | 31%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.463 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.183 ms | 2.67 | 0.17 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.248 ms | 3.05 | 12.19 | 31%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.096 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.766 ms | 0.63 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.629 ms | 1.20 | 4.81 | 12%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.908 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.716 ms | 0.47 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.637 ms | 1.19 | 4.75 | 12%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 9.548 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 7435.889 ms | 0.04 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.427 ms | 2.36 | 9.44 | 24%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 18.809 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 67.616 ms | 4.56 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.448 ms | 2.33 | 9.31 | 24%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.864 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 555.493 ms | 1.11 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.524 ms | 0.45 | 1.79 | 5%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.653 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 642.588 ms | 0.96 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.516 ms | 0.45 | 1.79 | 5%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 31.250 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 33.290 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 23.032 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.514 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 15.132 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 14.026 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.817 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.460 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.372 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 17.299 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.531 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.965 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.295 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.035 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.673 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.364 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.072 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.005 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.707 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.980 ms | 2.67 | 29.94 | 77% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.217 ms | 2.15 | 24.10 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.562 ms | 1.68 | 18.78 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.248 ms | 2.90 | 21.94 | 57% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 30.185 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 45.271 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 34.900 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 5.019 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.898 ms | 3.43 | 13.70 | 35% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 38.984 ms | 6.89 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.544 ms | 30.86 | 32.79 | 85%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.301 ms | 7.29 | 29.17 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.992 ms | 16.91 | 33.81 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.720 ms | 9.75 | 9.75 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.720 ms | 9.75 | 9.75 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.428 ms | 15.40 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.822 ms | 12.89 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.239 ms | 26.22 | 0.15 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.240 ms | 26.21 | 0.15 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 26.855 ms | 10.00 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## i7-12700k -- native, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `8867b44b3f52` |
| FMA peak (ceiling) | f64 600.8 GFLOP/s, f32 1201.7 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.8, f32 1201.7 GFLOP/s |
| ... this build, scalar | f64 9.8, f32 10.5 GFLOP/s |
| bandwidth | read 83.9, update 78.8, triad 77.3 GB/s (ceiling 83.9) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 84% | 123% | 0 | 28% |
| blas2 | 10 | 42% | 136% | 0 | 13% |
| blas3 | 10 | 2% | 15% | 0 | 15% |
| factor | 12 | 1% | 12% | 0 | 14% |
| eigen | 8 | 1% | 4% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 288% | 0 | 13% |
| direct | 19 | 0% | 14% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 11% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 36% | 0 | 13% |
| mg | 1 | 25% | 25% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 40% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 87% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.490 ms | 11.26 | 90.08 | 107% | ok | 3.8x (47%) |
| blas1 | dot | f32 | n=8388608 | 0.650 ms | 25.83 | 103.31 | 123% | ok | 6.5x (81%) |
| blas1 | axpy | f64 | n=8388608 | 2.832 ms | 5.92 | 71.10 | 85% | ok | 2.2x (28%) |
| blas1 | axpy | f32 | n=8388608 | 1.444 ms | 11.62 | 69.70 | 83% | ok | 3.2x (41%) |
| blas1 | scale | f64 | n=8388608 | 1.915 ms | 4.38 | 70.10 | 84% | ok | 1.9x (23%) |
| blas1 | two_norm | f64 | n=8388608 | 0.742 ms | 22.61 | 90.45 | 108% | ok | 6.5x (82%) |
| blas1 | one_norm | f64 | n=8388608 | 3.943 ms | 2.13 | 17.02 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.916 ms | 1.21 | 9.70 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.918 ms | 2.14 | 17.13 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 75.132 ms | 2.46 | 1.79 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.445 ms | 23.22 | 92.92 | 111% | ok | 2.5x (31%) |
| blas2 | gemv | f32 | n=4096 | 0.588 ms | 57.02 | 114.09 | 136% | ok | 4.7x (59%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.573 ms | 21.33 | 85.37 | 102% | ok | 3.0x (37%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.866 ms | 38.75 | 77.53 | 92% | ok | 2.5x (32%) |
| blas2 | symv | f64 | n=4096 | 6.562 ms | 5.11 | 10.24 | 12% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 1.998 ms | 8.40 | 33.62 | 40% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 1.430 ms | 11.73 | 23.48 | 28% | ok | 1.0x (12%) |
| blas2 | trsv | f64 | n=4096 | 1.958 ms | 8.57 | 34.31 | 41% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.398 ms | 12.00 | 24.03 | 29% | ok | 1.0x (12%) |
| blas2 | ger | f64 | n=4096 | 7.715 ms | 4.35 | 34.80 | 42% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.894 ms | 92.77 | 2.17 | 15% | ok | 6.9x (87%) |
| blas3 | gemm | f32 | 512x512x512 | 1.525 ms | 176.03 | 2.06 | 15% | ok | 7.0x (87%) |
| blas3 | gemmt | f64 | 512x512x512 | 8.940 ms | 15.01 | 0.70 | 2% | ok | 1.2x (15%) |
| blas3 | symm | f64 | 512x512x512 | 16.659 ms | 16.11 | 0.38 | 3% | ok | 1.1x (14%) |
| blas3 | syrk | f64 | 512x512x512 | 11.566 ms | 11.60 | 0.36 | 2% | ok | 1.2x (15%) |
| blas3 | syr2k | f64 | 512x512x512 | 19.514 ms | 13.76 | 0.32 | 2% | ok | 1.3x (16%) |
| blas3 | trmm | f64 | 512x512x512 | 9.126 ms | 14.71 | 0.57 | 2% | ok | 1.1x (14%) |
| blas3 | trsm | f64 | 512x512x512 | 9.279 ms | 14.47 | 0.57 | 2% | ok | 1.1x (13%) |
| blas3 | trsm | f32 | 512x512x512 | 5.041 ms | 26.63 | 0.52 | 2% | ok | 1.0x (13%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.865 ms | 0.41 | 3.30 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 7.439 ms | 12.03 | 0.56 | 2% | ok | 1.6x (19%) |
| factor | lu | f32 | 512x512 | 4.011 ms | 22.31 | 0.52 | 2% | ok | 1.1x (14%) |
| factor | cholesky | f64 | 512x512 | 7.096 ms | 6.30 | 0.59 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 6.454 ms | 6.93 | 0.32 | 1% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 7.189 ms | 6.22 | 0.58 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 45.606 ms | 0.98 | 0.09 | 0% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 129.421 ms | 1.38 | 0.03 | 0% | ok | 1.8x (22%) |
| factor | lq | f64 | 512x512 | 17.287 ms | 10.35 | 0.24 | 2% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 169.442 ms | 2.64 | 0.02 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 0.415 ms | 2.53 | 10.11 | 12%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 50.850 ms | 5.28 | 0.08 | 1% | ok | 1.1x (13%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.274 ms | 14.26 | 0.67 | 2% | ok | 1.0x (12%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 17.460 ms | 1.28 | 0.06 | 0% | ok | 1.1x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 57.630 ms | 2.62 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 735.479 ms | 0.21 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 31.847 ms | 5.27 | 0.03 | 1% | ok | 1.1x (13%) |
| eigen | schur | f64 | 256x256 | 82.477 ms | 5.09 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 109.077 ms | 3.38 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 11.690 ms | 21.53 | 0.09 | 4% | ok | 2.3x (28%) |
| eigen | sylvester | f64 | 256x256 | 177.212 ms | 5.68 | 0.01 | 1% | ok | 1.1x (13%) |
| elementwise | exp | f64 | n=8388608 | 38.541 ms | 0.22 | 3.48 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 45.037 ms | 0.19 | 2.98 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 58.773 ms | 0.14 | 2.28 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.424 ms | 0.30 | 4.72 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 199.195 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.113 ms | 23.21 | 241.51 | 288% ! | ok | 6.6x (82%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.908 ms | 2.88 | 30.00 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.409 ms | 29.73 | 62.51 | 75% | ok | 3.3x (41%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.802 ms | 0.68 | 2.39 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.367 ms | 1.47 | 12.17 | 15%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 89.593 ms | 0.12 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.009 ms | 0.45 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 9.234 ms | 1.21 | 0.08 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.446 ms | 0.55 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 53.086 ms | 0.42 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.757 ms | 1.34 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 85.311 ms | 0.02 | 0.00 | 0% | ok | 1.2x (15%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.249 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.848 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.220 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.723 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.270 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 87.246 ms | 0.13 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.271 ms | 2.79 | 11.15 | 13%\* | ok | 0.9x (11%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.350 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.144 ms | 2.70 | 0.18 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.262 ms | 2.88 | 11.53 | 14%\* | ok | 0.9x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.022 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 36.698 ms | 0.61 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.598 ms | 1.27 | 5.06 | 6%\* | ok | 1.1x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.772 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.324 ms | 0.47 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.606 ms | 1.25 | 4.99 | 6%\* | ok | 1.1x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 9.275 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 7286.320 ms | 0.04 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.525 ms | 2.21 | 8.84 | 11%\* | ok | 0.9x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 18.256 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.516 ms | 4.44 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.536 ms | 2.19 | 8.78 | 10%\* | ok | 0.9x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.743 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 537.356 ms | 1.15 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 8.059 ms | 0.42 | 1.67 | 2%\* | ok | 0.9x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.307 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 626.857 ms | 0.98 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 8.056 ms | 0.42 | 1.67 | 2%\* | ok | 0.9x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.483 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 32.118 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.443 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.213 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.626 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.698 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.160 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.682 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.946 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 16.830 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.080 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.472 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 20.776 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.032 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.575 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | ilut | f64 | 128^2 grid | 12.131 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.044 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.981 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.020 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.703 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.968 ms | 2.70 | 30.30 | 36% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.197 ms | 2.19 | 24.49 | 29% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.533 ms | 1.71 | 19.13 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.714 ms | 2.80 | 21.19 | 25% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.139 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 43.953 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 34.207 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.891 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 4.809 ms | 3.49 | 13.96 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 5.171 ms | 51.92 | 0.81 | 9% | ok | 7.5x (94%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.532 ms | 31.55 | 33.52 | 40%\* | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.307 ms | 7.27 | 29.09 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.978 ms | 17.15 | 34.31 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.684 ms | 9.97 | 9.97 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.684 ms | 9.96 | 9.96 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.503 ms | 107.25 | 1.26 | n/a | ok | 7.0x (87%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.748 ms | 97.70 | 0.76 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.356 ms | 197.97 | 1.16 | n/a | ok | 7.6x (94%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.357 ms | 197.88 | 1.16 | n/a | ok | 7.5x (94%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.607 ms | 74.42 | 0.44 | n/a | ok | 7.4x (93%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

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

## i7-12700k -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `8867b44b3f52` |
| FMA peak (ceiling) | f64 600.9 GFLOP/s, f32 1201.9 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.9, f32 1201.9 GFLOP/s |
| ... this build, AVX2 | f64 600.9, f32 1201.7 GFLOP/s |
| bandwidth | read 81.0, update 77.9, triad 76.2 GB/s (ceiling 81.0) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 90% | 149% | 0 | 22% |
| blas2 | 10 | 49% | 151% | 0 | 13% |
| blas3 | 10 | 12% | 67% | 0 | 19% |
| factor | 12 | 1% | 12% | 0 | 13% |
| eigen | 8 | 1% | 4% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 37% | 302% | 0 | 13% |
| direct | 19 | 0% | 14% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 11% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 28% | 37% | 0 | 13% |
| mg | 1 | 24% | 24% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 49% | 73% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 78% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.408 ms | 11.91 | 95.32 | 118% | ok | 2.6x (33%) |
| blas1 | dot | f32 | n=8388608 | 0.554 ms | 30.27 | 121.06 | 149% | ok | 3.1x (38%) |
| blas1 | axpy | f64 | n=8388608 | 2.773 ms | 6.05 | 72.60 | 90% | ok | 1.7x (21%) |
| blas1 | axpy | f32 | n=8388608 | 1.358 ms | 12.35 | 74.12 | 91% | ok | 1.7x (22%) |
| blas1 | scale | f64 | n=8388608 | 1.899 ms | 4.42 | 70.67 | 87% | ok | 1.8x (23%) |
| blas1 | two_norm | f64 | n=8388608 | 0.563 ms | 29.79 | 119.18 | 147% | ok | 3.9x (49%) |
| blas1 | one_norm | f64 | n=8388608 | 3.950 ms | 2.12 | 16.99 | 21% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.908 ms | 1.21 | 9.71 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.897 ms | 2.15 | 17.22 | 21% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 69.444 ms | 2.66 | 1.93 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.444 ms | 23.24 | 93.00 | 115% | ok | 2.4x (29%) |
| blas2 | gemv | f32 | n=4096 | 0.547 ms | 61.30 | 122.66 | 151% ! | ok | 3.1x (38%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.436 ms | 23.36 | 93.49 | 115% | ok | 2.4x (30%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.678 ms | 49.50 | 99.04 | 122% | ok | 2.5x (31%) |
| blas2 | symv | f64 | n=4096 | 6.117 ms | 5.49 | 10.98 | 14% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 1.755 ms | 9.56 | 38.27 | 47% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 0.839 ms | 19.99 | 40.02 | 49% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 1.790 ms | 9.37 | 37.52 | 46% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.848 ms | 19.79 | 39.62 | 49% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.109 ms | 4.72 | 37.77 | 47% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 0.670 ms | 400.67 | 9.39 | 67% | ok | 5.7x (71%) |
| blas3 | gemm | f32 | 512x512x512 | 0.356 ms | 753.02 | 8.82 | 63% | ok | 5.3x (67%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.250 ms | 107.36 | 5.03 | 18% | ok | 2.1x (26%) |
| blas3 | symm | f64 | 512x512x512 | 1.121 ms | 239.55 | 5.61 | 40% | ok | 3.6x (45%) |
| blas3 | syrk | f64 | 512x512x512 | 10.781 ms | 12.45 | 0.39 | 2% | ok | 0.9x (11%) |
| blas3 | syr2k | f64 | 512x512x512 | 16.423 ms | 16.34 | 0.38 | 3% | ok | 1.2x (15%) |
| blas3 | trmm | f64 | 512x512x512 | 1.856 ms | 72.30 | 2.82 | 12% | ok | 1.5x (19%) |
| blas3 | trsm | f64 | 512x512x512 | 2.183 ms | 61.50 | 2.40 | 10% | ok | 1.3x (17%) |
| blas3 | trsm | f32 | 512x512x512 | 1.108 ms | 121.14 | 2.37 | 10% | ok | 1.2x (16%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.843 ms | 0.41 | 3.31 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 6.458 ms | 13.86 | 0.65 | 2% | ok | 1.7x (21%) |
| factor | lu | f32 | 512x512 | 3.483 ms | 25.69 | 0.60 | 2% | ok | 1.0x (12%) |
| factor | cholesky | f64 | 512x512 | 6.961 ms | 6.43 | 0.60 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 6.403 ms | 6.99 | 0.33 | 1% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 7.076 ms | 6.32 | 0.59 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 45.628 ms | 0.98 | 0.09 | 0% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 129.577 ms | 1.38 | 0.03 | 0% | ok | 1.7x (22%) |
| factor | lq | f64 | 512x512 | 15.067 ms | 11.88 | 0.28 | 2% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 166.367 ms | 2.69 | 0.03 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 0.429 ms | 2.45 | 9.78 | 12%\* | ok | 3.8x (47%) |
| factor | inv | f64 | 512x512 | 51.556 ms | 5.21 | 0.08 | 1% | ok | 1.1x (13%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 4.864 ms | 18.40 | 0.86 | 3% | ok | 1.0x (12%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 16.931 ms | 1.32 | 0.06 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 56.146 ms | 2.69 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 744.017 ms | 0.20 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 30.527 ms | 5.50 | 0.03 | 1% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 77.856 ms | 5.39 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 91.533 ms | 4.03 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 10.101 ms | 24.92 | 0.10 | 4% | ok | 1.2x (15%) |
| eigen | sylvester | f64 | 256x256 | 168.846 ms | 5.96 | 0.01 | 1% | ok | 1.2x (15%) |
| elementwise | exp | f64 | n=8388608 | 38.461 ms | 0.22 | 3.49 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 45.664 ms | 0.18 | 2.94 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 58.403 ms | 0.14 | 2.30 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.264 ms | 0.30 | 4.75 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 201.206 ms | 0.04 | 0.67 | 1% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.111 ms | 23.50 | 244.52 | 302% ! | ok | 6.6x (82%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.900 ms | 2.91 | 30.25 | 37% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.366 ms | 30.65 | 64.44 | 80% | ok | 3.0x (38%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.770 ms | 0.68 | 2.41 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.379 ms | 1.46 | 12.08 | 15%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 71.892 ms | 0.16 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 27.695 ms | 0.40 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 9.136 ms | 1.22 | 0.08 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.352 ms | 0.57 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 55.079 ms | 0.41 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.654 ms | 1.34 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 85.710 ms | 0.02 | 0.00 | 0% | ok | 1.2x (14%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.378 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.902 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.225 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.846 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.300 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 68.729 ms | 0.16 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.272 ms | 2.78 | 11.12 | 14%\* | ok | 0.9x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.388 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.291 ms | 2.61 | 0.17 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.300 ms | 2.52 | 10.08 | 12%\* | ok | 0.9x (11%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.073 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.447 ms | 0.63 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.625 ms | 1.21 | 4.84 | 6%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.808 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 49.248 ms | 0.45 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.622 ms | 1.22 | 4.86 | 6%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 9.284 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 5226.596 ms | 0.06 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.533 ms | 2.20 | 8.79 | 11%\* | ok | 0.9x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 18.265 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 64.898 ms | 4.76 | 0.01 | 1% | ok | 1.1x (14%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.763 ms | 1.91 | 7.65 | 9%\* | ok | 0.9x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.849 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 513.234 ms | 1.20 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 8.057 ms | 0.42 | 1.67 | 2%\* | ok | 0.9x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.372 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 640.538 ms | 0.96 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 8.081 ms | 0.42 | 1.67 | 2%\* | ok | 0.9x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.128 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 28.120 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 20.655 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.393 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.665 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.501 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 20.135 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.575 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.400 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.510 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 14.761 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 15.812 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 17.945 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.032 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.529 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 11.780 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.996 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.722 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.969 ms | 2.70 | 30.27 | 37% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.285 ms | 2.04 | 22.83 | 28% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.483 ms | 1.77 | 19.78 | 24% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.857 ms | 2.59 | 19.56 | 24% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 27.731 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 38.516 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.976 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.493 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 1.913 ms | 8.77 | 35.08 | 43% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 0.912 ms | 294.18 | 4.60 | 49% | ok | 6.6x (82%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.302 ms | 55.56 | 59.04 | 73%\* | ok | 1.1x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.898 ms | 8.84 | 35.35 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.739 ms | 22.70 | 45.39 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.263 ms | 63.74 | 63.74 | n/a | ok | 1.1x (14%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.246 ms | 68.16 | 68.16 | n/a | ok | 1.1x (14%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 0.773 ms | 347.37 | 4.07 | n/a | ok | 6.7x (83%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 0.818 ms | 328.03 | 2.56 | n/a | ok | 6.9x (86%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 0.801 ms | 335.00 | 1.96 | n/a | ok | 7.0x (88%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.332 ms | 808.89 | 4.74 | n/a | ok | 6.3x (79%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.303 ms | 884.67 | 5.18 | n/a | ok | 6.3x (78%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

## i7-12700k -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `8867b44b3f52` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.3 GFLOP/s |
| bandwidth | read 33.1, update 38.5, triad 32.5 GB/s (ceiling 38.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 44% | 98% | 0 |
| blas2 | 10 | 97% | 107% | 0 |
| blas3 | 10 | 81% | 94% | 0 |
| factor | 12 | 24% | 40% | 0 |
| eigen | 8 | 24% | 71% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 48% | 96% | 0 |
| direct | 31 | 3% | 32% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 21 | 24% | 69% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 62% | 77% | 0 |
| mg | 1 | 57% | 57% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 36% | 84% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.725 ms | 2.93 | 23.45 | 61% | ok |
| blas1 | dot | f32 | n=8388608 | 4.273 ms | 3.93 | 15.71 | 41% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.407 ms | 2.62 | 31.42 | 82% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.929 ms | 3.40 | 20.42 | 53% | ok |
| blas1 | scale | f64 | n=8388608 | 3.571 ms | 2.35 | 37.58 | 98% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.874 ms | 3.44 | 13.77 | 36% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.010 ms | 2.09 | 16.74 | 43% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.031 ms | 1.19 | 9.54 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 3.974 ms | 2.11 | 16.89 | 44% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 76.344 ms | 2.42 | 1.76 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.492 ms | 9.61 | 38.46 | 100% | ok |
| blas2 | gemv | f32 | n=4096 | 1.647 ms | 20.37 | 40.77 | 106% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.622 ms | 9.26 | 37.08 | 96% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.628 ms | 20.61 | 41.25 | 107% | ok |
| blas2 | symv | f64 | n=4096 | 1.915 ms | 17.52 | 35.08 | 91% | ok |
| blas2 | trmv | f64 | n=4096 | 1.851 ms | 9.06 | 36.29 | 94% | ok |
| blas2 | trmv | f32 | n=4096 | 0.940 ms | 17.86 | 35.75 | 93% | ok |
| blas2 | trsv | f64 | n=4096 | 1.789 ms | 9.38 | 37.55 | 97% | ok |
| blas2 | trsv | f32 | n=4096 | 0.870 ms | 19.28 | 38.60 | 100% | ok |
| blas2 | ger | f64 | n=4096 | 7.682 ms | 4.37 | 34.95 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.700 ms | 72.55 | 1.70 | 93% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.831 ms | 146.64 | 1.72 | 94% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.580 ms | 52.02 | 2.44 | 66% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.709 ms | 72.38 | 1.70 | 92% | ok |
| blas3 | syrk | f64 | 512x512x512 | 2.545 ms | 52.74 | 1.65 | 67% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 4.732 ms | 56.73 | 1.33 | 72% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.109 ms | 63.65 | 2.49 | 81% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.131 ms | 63.00 | 2.46 | 80% | ok |
| blas3 | trsm | f32 | 512x512x512 | 0.963 ms | 139.44 | 2.72 | 89% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.099 ms | 0.41 | 3.24 | 8% | ok |
| factor | lu | f64 | 512x512 | 3.304 ms | 27.08 | 1.27 | 35% | ok |
| factor | lu | f32 | 512x512 | 1.863 ms | 48.04 | 1.13 | 31% | ok |
| factor | cholesky | f64 | 512x512 | 2.039 ms | 21.94 | 2.06 | 28% | ok |
| factor | cholesky | f32 | 512x512 | 1.188 ms | 37.66 | 1.77 | 24% | ok |
| factor | ldlt | f64 | 512x512 | 7.531 ms | 5.94 | 0.56 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 46.453 ms | 0.96 | 0.09 | 1% | ok |
| factor | qr | f64 | 512x512 | 5.767 ms | 31.03 | 0.73 | 40% | ok |
| factor | lq | f64 | 512x512 | 25.575 ms | 7.00 | 0.16 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 268.326 ms | 1.67 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.640 ms | 0.64 | 2.56 | 7%\* | ok |
| factor | inv | f64 | 512x512 | 45.689 ms | 5.88 | 0.09 | 8% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 3.826 ms | 23.39 | 1.10 | 30% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.699 ms | 13.16 | 0.62 | 17% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 58.911 ms | 2.56 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 745.632 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 16.480 ms | 10.18 | 0.06 | 13% | ok |
| eigen | schur | f64 | 256x256 | 21.027 ms | 19.95 | 0.05 | 25% | ok |
| eigen | svd | f64 | 256x256 | 6.652 ms | 55.48 | 0.16 | 71% | ok |
| eigen | expm | f64 | 256x256 | 11.209 ms | 22.45 | 0.09 | 29% | ok |
| eigen | sylvester | f64 | 256x256 | 53.349 ms | 18.87 | 0.03 | 24% | ok |
| elementwise | exp | f64 | n=8388608 | 39.232 ms | 0.21 | 3.42 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 45.474 ms | 0.18 | 2.95 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 59.953 ms | 0.14 | 2.24 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 28.816 ms | 0.29 | 4.66 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 201.453 ms | 0.04 | 0.67 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.740 ms | 3.54 | 36.82 | 96% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.928 ms | 2.82 | 29.35 | 76% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.751 ms | 8.81 | 18.53 | 48% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.865 ms | 0.67 | 2.36 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.251 ms | 1.60 | 13.31 | 35%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 91.124 ms | 0.12 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.477 ms | 0.44 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.940 ms | 1.25 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.414 ms | 0.57 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 53.590 ms | 0.42 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.821 ms | 1.33 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 99.895 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.340 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.988 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.225 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.096 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.354 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 88.464 ms | 0.13 | 0.01 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.246 ms | 3.07 | 12.28 | 32%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.513 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.203 ms | 2.66 | 0.17 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.248 ms | 3.05 | 12.19 | 32%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.110 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.234 ms | 0.63 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.634 ms | 1.19 | 4.77 | 12%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.926 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.738 ms | 0.47 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.628 ms | 1.20 | 4.82 | 13%\* | ok |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 5.649 ms | 1.98 | 0.13 | 3% | ok |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 1.894 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 3.403 ms | 3.29 | 0.21 | 4% | ok |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 0.327 ms | 2.31 | 9.24 | 24%\* | ok |
| direct | ref_pardiso_spd | f64 | 96^2 grid, nnz 45696 | 4.527 ms | 2.47 | 0.16 | 3% | ok |
| direct | ref_pardiso_spd.analyze | f64 | 96^2 grid, nnz 45696 | 2.109 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_pardiso_spd.factor | f64 | 96^2 grid, nnz 45696 | 1.909 ms | 5.86 | 0.38 | 7% | ok |
| direct | ref_pardiso_spd.solve | f64 | 96^2 grid, nnz 45696 | 0.458 ms | 1.65 | 6.61 | 17%\* | ok |
| direct | ref_pardiso_lu | f64 | 96^2 grid, nnz 45696 | 7.444 ms | 3.01 | 0.10 | 4% | ok |
| direct | ref_pardiso_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.562 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_pardiso_lu.factor | f64 | 96^2 grid, nnz 45696 | 3.244 ms | 6.90 | 0.23 | 9% | ok |
| direct | ref_pardiso_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.577 ms | 1.31 | 5.24 | 14%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 9.482 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 7435.072 ms | 0.04 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.438 ms | 2.34 | 9.37 | 24%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 18.667 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.422 ms | 4.45 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.445 ms | 2.33 | 9.33 | 24%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.891 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 551.473 ms | 1.12 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.511 ms | 0.45 | 1.79 | 5%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.729 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 665.388 ms | 0.93 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.513 ms | 0.45 | 1.79 | 5%\* | ok |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 3.890 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 11.428 ms | 27.00 | 0.08 | 34% | ok |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 0.507 ms | 6.64 | 26.57 | 69%\* | ok |
| direct3d | ref_pardiso_spd.analyze | f64 | 20^3 grid, nnz 53600 | 7.969 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_pardiso_spd.factor | f64 | 20^3 grid, nnz 53600 | 9.684 ms | 31.87 | 0.09 | 41% | ok |
| direct3d | ref_pardiso_spd.solve | f64 | 20^3 grid, nnz 53600 | 0.868 ms | 3.88 | 15.53 | 40%\* | ok |
| direct3d | ref_pardiso_lu.analyze | f64 | 20^3 grid, nnz 53600 | 9.054 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_pardiso_lu.factor | f64 | 20^3 grid, nnz 53600 | 19.083 ms | 32.34 | 0.04 | 41% | ok |
| direct3d | ref_pardiso_lu.solve | f64 | 20^3 grid, nnz 53600 | 1.038 ms | 3.24 | 12.98 | 34%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 31.154 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 32.942 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.933 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.499 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.997 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 14.013 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.979 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.242 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.254 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 17.228 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.513 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.919 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.252 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.034 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.604 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.306 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.077 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.016 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.023 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.371 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.982 ms | 2.66 | 29.85 | 77% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.219 ms | 2.15 | 24.06 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.563 ms | 1.67 | 18.76 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.216 ms | 2.91 | 21.99 | 57% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 30.538 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.806 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.477 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 5.028 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.859 ms | 3.45 | 13.81 | 36% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 38.978 ms | 6.89 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.554 ms | 30.30 | 32.20 | 84%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.308 ms | 7.27 | 29.08 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.991 ms | 16.92 | 33.85 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.721 ms | 9.75 | 9.75 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.720 ms | 9.76 | 9.76 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.419 ms | 15.41 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.803 ms | 12.90 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.224 ms | 26.26 | 0.15 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.222 ms | 26.26 | 0.15 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 26.791 ms | 10.02 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## i7-12700k -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `8867b44b3f52` |
| FMA peak (ceiling) | f64 600.9 GFLOP/s, f32 1201.8 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.9, f32 1201.8 GFLOP/s |
| ... this build, scalar | f64 10.0, f32 10.0 GFLOP/s |
| bandwidth | read 82.5, update 77.7, triad 75.9 GB/s (ceiling 82.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 85% | 125% | 0 | 28% |
| blas2 | 10 | 120% | 198% | 0 | 32% |
| blas3 | 10 | 66% | 85% | 0 | 77% |
| factor | 12 | 5% | 13% | 0 | 18% |
| eigen | 8 | 3% | 10% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 292% | 0 | 13% |
| direct | 31 | 1% | 13% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| direct3d | 21 | 10% | 32% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 30% | 37% | 0 | 13% |
| mg | 1 | 25% | 25% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 40% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 90% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.496 ms | 11.22 | 89.73 | 109% | ok | 3.8x (48%) |
| blas1 | dot | f32 | n=8388608 | 0.650 ms | 25.82 | 103.27 | 125% | ok | 6.6x (82%) |
| blas1 | axpy | f64 | n=8388608 | 2.842 ms | 5.90 | 70.83 | 86% | ok | 2.3x (28%) |
| blas1 | axpy | f32 | n=8388608 | 1.438 ms | 11.66 | 69.98 | 85% | ok | 3.4x (43%) |
| blas1 | scale | f64 | n=8388608 | 1.915 ms | 4.38 | 70.08 | 85% | ok | 1.9x (23%) |
| blas1 | two_norm | f64 | n=8388608 | 0.739 ms | 22.70 | 90.79 | 110% | ok | 6.6x (82%) |
| blas1 | one_norm | f64 | n=8388608 | 3.942 ms | 2.13 | 17.03 | 21% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.907 ms | 1.21 | 9.72 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.912 ms | 2.14 | 17.15 | 21% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 75.066 ms | 2.46 | 1.79 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.441 ms | 23.28 | 93.17 | 113% | ok | 2.4x (30%) |
| blas2 | gemv | f32 | n=4096 | 0.547 ms | 61.35 | 122.76 | 149% | ok | 3.0x (38%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.421 ms | 23.61 | 94.50 | 114% | ok | 2.5x (32%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.678 ms | 49.49 | 99.03 | 120% | ok | 2.4x (30%) |
| blas2 | symv | f64 | n=4096 | 0.630 ms | 53.29 | 106.69 | 129% | ok | 3.0x (38%) |
| blas2 | trmv | f64 | n=4096 | 0.601 ms | 27.93 | 111.83 | 136% | ok | 3.1x (39%) |
| blas2 | trmv | f32 | n=4096 | 0.205 ms | 81.72 | 163.60 | 198% ! | ok | 4.6x (57%) |
| blas2 | trsv | f64 | n=4096 | 1.754 ms | 9.57 | 38.30 | 46% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.851 ms | 19.72 | 39.49 | 48% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.637 ms | 4.39 | 35.16 | 43% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 0.537 ms | 499.49 | 11.71 | 83% | ok | 6.9x (86%) |
| blas3 | gemm | f32 | 512x512x512 | 0.263 ms | 1020.04 | 11.95 | 85% | ok | 7.0x (87%) |
| blas3 | gemmt | f64 | 512x512x512 | 0.699 ms | 192.09 | 9.00 | 32% | ok | 3.7x (46%) |
| blas3 | symm | f64 | 512x512x512 | 0.536 ms | 501.22 | 11.75 | 83% | ok | 6.9x (87%) |
| blas3 | syrk | f64 | 512x512x512 | 0.748 ms | 179.51 | 5.61 | 30% | ok | 3.4x (43%) |
| blas3 | syr2k | f64 | 512x512x512 | 1.122 ms | 239.22 | 5.61 | 40% | ok | 4.2x (53%) |
| blas3 | trmm | f64 | 512x512x512 | 0.341 ms | 394.11 | 15.39 | 66% | ok | 6.2x (77%) |
| blas3 | trsm | f64 | 512x512x512 | 0.340 ms | 394.50 | 15.41 | 66% | ok | 6.3x (78%) |
| blas3 | trsm | f32 | 512x512x512 | 0.159 ms | 843.44 | 16.47 | 70% | ok | 6.0x (76%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.856 ms | 0.41 | 3.30 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 2.101 ms | 42.59 | 2.00 | 7% | ok | 1.6x (20%) |
| factor | lu | f32 | 512x512 | 1.313 ms | 68.17 | 1.60 | 6% | ok | 1.4x (18%) |
| factor | cholesky | f64 | 512x512 | 1.515 ms | 29.54 | 2.77 | 5% | ok | 1.3x (17%) |
| factor | cholesky | f32 | 512x512 | 0.961 ms | 46.54 | 2.18 | 4% | ok | 1.2x (15%) |
| factor | ldlt | f64 | 512x512 | 7.213 ms | 6.20 | 0.58 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 44.642 ms | 1.00 | 0.09 | 0% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 4.067 ms | 44.01 | 1.03 | 7% | ok | 1.4x (18%) |
| factor | lq | f64 | 512x512 | 17.382 ms | 10.30 | 0.24 | 2% | ok | 1.5x (18%) |
| factor | hessenberg | f64 | 512x512 | 166.121 ms | 2.69 | 0.03 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 0.406 ms | 2.59 | 10.33 | 13%\* | ok | 4.0x (50%) |
| factor | inv | f64 | 512x512 | 46.123 ms | 5.82 | 0.09 | 1% | ok | 1.0x (12%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 2.796 ms | 32.01 | 1.50 | 5% | ok | 1.4x (17%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.639 ms | 13.65 | 0.64 | 2% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 58.238 ms | 2.59 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 729.505 ms | 0.21 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 17.438 ms | 9.62 | 0.06 | 2% | ok | 0.9x (12%) |
| eigen | schur | f64 | 256x256 | 21.677 ms | 19.35 | 0.05 | 3% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 6.143 ms | 60.09 | 0.17 | 10% | ok | 1.1x (14%) |
| eigen | expm | f64 | 256x256 | 8.724 ms | 28.85 | 0.12 | 5% | ok | 1.3x (16%) |
| eigen | sylvester | f64 | 256x256 | 55.135 ms | 18.26 | 0.03 | 3% | ok | 1.0x (12%) |
| elementwise | exp | f64 | n=8388608 | 38.553 ms | 0.22 | 3.48 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 44.724 ms | 0.19 | 3.00 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 58.872 ms | 0.14 | 2.28 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.404 ms | 0.30 | 4.73 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 199.780 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.113 ms | 23.15 | 240.84 | 292% ! | ok | 6.5x (82%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.911 ms | 2.87 | 29.88 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.319 ms | 31.75 | 66.75 | 81% | ok | 3.6x (45%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.803 ms | 0.68 | 2.39 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.369 ms | 1.47 | 12.16 | 15%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 89.216 ms | 0.13 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.839 ms | 0.43 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.921 ms | 1.25 | 0.08 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.172 ms | 0.57 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 53.092 ms | 0.42 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.720 ms | 1.34 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 85.929 ms | 0.02 | 0.00 | 0% | ok | 1.2x (15%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.268 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.877 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.216 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.839 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.287 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 86.587 ms | 0.13 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.276 ms | 2.74 | 10.95 | 13%\* | ok | 0.9x (11%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.385 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.078 ms | 2.74 | 0.18 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.275 ms | 2.75 | 11.01 | 13%\* | ok | 0.9x (11%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.025 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 34.987 ms | 0.64 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.609 ms | 1.24 | 4.97 | 6%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.788 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.068 ms | 0.48 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.594 ms | 1.27 | 5.10 | 6%\* | ok | 1.1x (13%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 5.991 ms | 1.87 | 0.12 | 0% | ok | 0.9x (12%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 1.824 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 3.670 ms | 3.05 | 0.20 | 1% | ok | 0.9x (12%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 0.328 ms | 2.31 | 9.23 | 11%\* | ok | 1.0x (12%) |
| direct | ref_pardiso_spd | f64 | 96^2 grid, nnz 45696 | 3.210 ms | 3.48 | 0.23 | 1% | ok | 1.4x (18%) |
| direct | ref_pardiso_spd.analyze | f64 | 96^2 grid, nnz 45696 | 1.992 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct | ref_pardiso_spd.factor | f64 | 96^2 grid, nnz 45696 | 0.618 ms | 18.09 | 1.18 | 3% | ok | 3.1x (39%) |
| direct | ref_pardiso_spd.solve | f64 | 96^2 grid, nnz 45696 | 0.500 ms | 1.51 | 6.05 | 7%\* | ok | 0.9x (11%) |
| direct | ref_pardiso_lu | f64 | 96^2 grid, nnz 45696 | 5.608 ms | 3.99 | 0.13 | 1% | ok | 1.3x (17%) |
| direct | ref_pardiso_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.512 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | ref_pardiso_lu.factor | f64 | 96^2 grid, nnz 45696 | 1.190 ms | 18.80 | 0.61 | 3% | ok | 2.7x (34%) |
| direct | ref_pardiso_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.629 ms | 1.20 | 4.81 | 6%\* | ok | 0.9x (11%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 9.278 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 7286.650 ms | 0.04 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.549 ms | 2.18 | 8.70 | 11%\* | ok | 0.9x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 18.241 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 66.197 ms | 4.66 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.569 ms | 2.15 | 8.59 | 10%\* | ok | 0.9x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.756 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 541.169 ms | 1.14 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 8.057 ms | 0.42 | 1.67 | 2%\* | ok | 0.9x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.373 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 644.934 ms | 0.96 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.791 ms | 0.43 | 1.73 | 2%\* | ok | 1.0x (12%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 3.804 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 8.272 ms | 37.31 | 0.10 | 6% | ok | 1.4x (17%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 0.510 ms | 6.61 | 26.44 | 32%\* | ok | 1.0x (12%) |
| direct3d | ref_pardiso_spd.analyze | f64 | 20^3 grid, nnz 53600 | 7.117 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | ref_pardiso_spd.factor | f64 | 20^3 grid, nnz 53600 | 2.347 ms | 131.48 | 0.37 | 22% | ok | 4.1x (52%) |
| direct3d | ref_pardiso_spd.solve | f64 | 20^3 grid, nnz 53600 | 0.943 ms | 3.57 | 14.29 | 17%\* | ok | 0.9x (12%) |
| direct3d | ref_pardiso_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.140 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | ref_pardiso_lu.factor | f64 | 20^3 grid, nnz 53600 | 5.529 ms | 111.63 | 0.16 | 19% | ok | 3.5x (43%) |
| direct3d | ref_pardiso_lu.solve | f64 | 20^3 grid, nnz 53600 | 1.154 ms | 2.92 | 11.68 | 14%\* | ok | 0.9x (11%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.471 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 32.130 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.355 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.217 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.618 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.706 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.123 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.653 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.926 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 16.787 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.054 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.454 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 20.686 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.035 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.527 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | ilut | f64 | 128^2 grid | 12.003 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.047 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.981 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.020 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.155 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.965 ms | 2.71 | 30.39 | 37% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.198 ms | 2.19 | 24.49 | 30% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.534 ms | 1.71 | 19.11 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.863 ms | 2.77 | 20.97 | 25% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.661 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.086 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.766 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.926 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 4.796 ms | 3.50 | 13.99 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 5.187 ms | 51.75 | 0.81 | 9% | ok | 7.5x (94%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.543 ms | 30.91 | 32.84 | 40%\* | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.267 ms | 7.40 | 29.60 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.981 ms | 17.10 | 34.20 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.683 ms | 9.97 | 9.97 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.683 ms | 9.97 | 9.97 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.410 ms | 111.37 | 1.31 | n/a | ok | 7.2x (90%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.827 ms | 94.96 | 0.74 | n/a | ok | 7.4x (92%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.357 ms | 197.86 | 1.16 | n/a | ok | 7.5x (94%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.356 ms | 197.93 | 1.16 | n/a | ok | 7.5x (94%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.547 ms | 75.67 | 0.44 | n/a | ok | 7.6x (94%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

