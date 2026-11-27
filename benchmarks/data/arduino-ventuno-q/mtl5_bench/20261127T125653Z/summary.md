# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,4, CPUs 0,1,2,3, clock policy governor performance on the A78C policies; performance 902-2112 MHz on cpus 0,1; performance 940-2361 MHz on cpus 2,3, 2026-11-27T13:17:16Z.

## arduino-ventuno-q -- native, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.6 GFLOP/s |
| ... this build, scalar | f64 8.4, f32 8.4 GFLOP/s |
| bandwidth | read 22.6, update 24.9, triad 23.1 GB/s (ceiling 24.9) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 84% | 104% | 0 |
| blas2 | 10 | 60% | 71% | 0 |
| blas3 | 10 | 30% | 33% | 0 |
| factor | 12 | 12% | 32% | 0 |
| eigen | 8 | 15% | 22% | 0 |
| elementwise | 5 | 5% | 7% | 0 |
| sparse | 5 | 30% | 66% | 0 |
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
| blas1 | dot | f64 | n=8388608 | 5.668 ms | 2.96 | 23.68 | 95% | ok |
| blas1 | dot | f32 | n=8388608 | 4.551 ms | 3.69 | 14.75 | 59% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.799 ms | 2.15 | 25.81 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.948 ms | 4.25 | 25.49 | 102% | ok |
| blas1 | scale | f64 | n=8388608 | 5.323 ms | 1.58 | 25.22 | 101% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.200 ms | 5.24 | 20.97 | 84% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.562 ms | 0.98 | 7.84 | 31% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.425 ms | 0.48 | 3.85 | 15% | ok |
| blas1 | sum | f64 | n=8388608 | 8.398 ms | 1.00 | 7.99 | 32% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 200.180 ms | 0.92 | 0.67 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 7.984 ms | 4.20 | 16.82 | 68% | ok |
| blas2 | gemv | f32 | n=4096 | 5.757 ms | 5.83 | 11.66 | 47% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 9.908 ms | 3.39 | 13.55 | 54% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.512 ms | 7.44 | 14.88 | 60% | ok |
| blas2 | symv | f64 | n=4096 | 12.230 ms | 2.74 | 5.49 | 22% | ok |
| blas2 | trmv | f64 | n=4096 | 4.449 ms | 3.77 | 15.10 | 61% | ok |
| blas2 | trmv | f32 | n=4096 | 3.326 ms | 5.04 | 10.10 | 41% | ok |
| blas2 | trsv | f64 | n=4096 | 4.484 ms | 3.74 | 14.98 | 60% | ok |
| blas2 | trsv | f32 | n=4096 | 3.035 ms | 5.53 | 11.07 | 44% | ok |
| blas2 | ger | f64 | n=4096 | 15.249 ms | 2.20 | 17.61 | 71% | ok |
| blas3 | gemm | f64 | 512x512x512 | 51.829 ms | 5.18 | 0.12 | 31% | ok |
| blas3 | gemm | f32 | 512x512x512 | 24.067 ms | 11.15 | 0.13 | 33% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 40.457 ms | 3.32 | 0.16 | 20% | ok |
| blas3 | symm | f64 | 512x512x512 | 52.422 ms | 5.12 | 0.12 | 30% | ok |
| blas3 | syrk | f64 | 512x512x512 | 39.831 ms | 3.37 | 0.11 | 20% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 81.504 ms | 3.29 | 0.08 | 20% | ok |
| blas3 | trmm | f64 | 512x512x512 | 25.821 ms | 5.20 | 0.20 | 31% | ok |
| blas3 | trsm | f64 | 512x512x512 | 26.373 ms | 5.09 | 0.20 | 30% | ok |
| blas3 | trsm | f32 | 512x512x512 | 11.934 ms | 11.25 | 0.22 | 33% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 31.250 ms | 0.17 | 1.36 | 5% | ok |
| factor | lu | f64 | 512x512 | 24.106 ms | 3.71 | 0.17 | 22% | ok |
| factor | lu | f32 | 512x512 | 13.001 ms | 6.88 | 0.16 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 22.280 ms | 2.01 | 0.19 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 19.632 ms | 2.28 | 0.11 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 21.509 ms | 2.08 | 0.20 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 11.079 ms | 4.04 | 0.38 | 24% | ok |
| factor | qr | f64 | 512x512 | 250.582 ms | 0.71 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 60.680 ms | 2.95 | 0.07 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 348.934 ms | 1.28 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.560 ms | 0.67 | 2.69 | 11% | ok |
| factor | inv | f64 | 512x512 | 145.533 ms | 1.84 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.525 ms | 5.41 | 0.25 | 32% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 40.000 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.252 ms | 1.55 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 590.249 ms | 0.26 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.291 ms | 2.49 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 149.835 ms | 2.80 | 0.01 | 17% | ok |
| eigen | svd | f64 | 256x256 | 230.110 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 66.905 ms | 3.76 | 0.02 | 22% | ok |
| eigen | sylvester | f64 | 256x256 | 368.935 ms | 2.73 | 0.00 | 16% | ok |
| elementwise | exp | f64 | n=8388608 | 89.478 ms | 0.09 | 1.50 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 111.117 ms | 0.08 | 1.21 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 150.579 ms | 0.06 | 0.89 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 73.496 ms | 0.11 | 1.83 | 7% | ok |
| elementwise | erf | f64 | n=8388608 | 118.923 ms | 0.07 | 1.13 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.666 ms | 1.57 | 16.34 | 66% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.136 ms | 1.23 | 12.75 | 51% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.978 ms | 3.50 | 7.35 | 30% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.015 ms | 0.25 | 0.88 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 9.070 ms | 0.22 | 1.84 | 7% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 36.850 ms | 0.30 | 0.02 | 2% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 58.758 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.310 ms | 0.43 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 104.823 ms | 0.21 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 121.608 ms | 0.18 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.592 ms | 0.51 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 74.056 ms | 0.03 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.688 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.327 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 124.896 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.583 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 43.666 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.512 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 29.341 ms | 0.38 | 0.02 | 2% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.828 ms | 0.41 | 1.65 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.240 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.937 ms | 0.94 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.782 ms | 0.42 | 1.70 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.405 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 89.851 ms | 0.25 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.933 ms | 0.13 | 0.51 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 8.656 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 101.856 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.944 ms | 0.13 | 0.51 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 11.817 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 666.065 ms | 0.46 | 0.00 | 3% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.704 ms | 0.44 | 1.75 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 27.711 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 185.373 ms | 1.66 | 0.00 | 10% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.643 ms | 0.44 | 1.76 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.693 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1683.838 ms | 0.37 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 30.170 ms | 0.11 | 0.45 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.572 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1231.413 ms | 0.50 | 0.00 | 3% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 30.417 ms | 0.11 | 0.44 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 71.431 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 161.640 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 54.677 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 29.010 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.876 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 34.341 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 62.898 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 64.151 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 35.106 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 39.523 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.959 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.011 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 41.165 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.112 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 5.015 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 30.690 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.141 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 3.019 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.066 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.787 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.121 ms | 1.23 | 13.83 | 56% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.482 ms | 1.05 | 11.81 | 47% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 5.025 ms | 0.52 | 5.84 | 23% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.301 ms | 1.19 | 9.00 | 36% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 231.898 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 235.815 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 144.754 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.309 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 8.033 ms | 2.09 | 8.35 | 34% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 48.585 ms | 5.53 | 0.09 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.942 ms | 5.70 | 6.06 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.851 ms | 5.88 | 23.54 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.408 ms | 11.92 | 23.83 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.653 ms | 4.59 | 4.59 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.167 ms | 14.38 | 14.38 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 24.749 ms | 10.85 | 0.13 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 26.376 ms | 10.18 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 20.312 ms | 13.22 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 20.317 ms | 13.21 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 38.469 ms | 6.98 | 0.04 | n/a | ok |

## arduino-ventuno-q -- native, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 67.5 GFLOP/s, f32 134.9 GFLOP/s |
| ... hardware, NEON | f64 67.5, f32 134.9 GFLOP/s |
| ... this build, scalar | f64 33.8, f32 33.8 GFLOP/s |
| bandwidth | read 23.0, update 24.7, triad 22.4 GB/s (ceiling 24.7) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 98% | 115% | 0 | 28% |
| blas2 | 10 | 62% | 88% | 0 | 27% |
| blas3 | 10 | 9% | 17% | 0 | 28% |
| factor | 12 | 6% | 37% | 0 | 35% |
| eigen | 8 | 4% | 10% | 0 | 28% |
| elementwise | 5 | 5% | 8% | 0 | 28% |
| sparse | 5 | 57% | 92% | 0 | 28% |
| direct | 19 | 1% | 6% | 0 | 27% |
| ordering | 5 | n/a | n/a | 0 | 28% |
| direct3d | 12 | 2% | 7% | 0 | 27% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 52% | 60% | 0 | 27% |
| mg | 1 | 37% | 37% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 32% | 38% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 53% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.544 ms | 3.03 | 24.21 | 98% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.807 ms | 5.98 | 23.91 | 97% | ok | 1.6x (41%) |
| blas1 | axpy | f64 | n=8388608 | 8.188 ms | 2.05 | 24.59 | 100% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 3.948 ms | 4.25 | 25.50 | 103% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 4.742 ms | 1.77 | 28.30 | 115% | ok | 1.1x (28%) |
| blas1 | two_norm | f64 | n=8388608 | 2.603 ms | 6.45 | 25.78 | 104% | ok | 1.2x (31%) |
| blas1 | one_norm | f64 | n=8388608 | 7.322 ms | 1.15 | 9.17 | 37% | ok | 1.2x (29%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.606 ms | 0.54 | 4.30 | 17% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.779 ms | 1.08 | 8.63 | 35% | ok | 1.1x (27%) |
| blas1 | dot_k3 | f64 | n=8388608 | 182.569 ms | 1.01 | 0.74 | 3% | ok | 1.1x (27%) |
| blas2 | gemv | f64 | n=4096 | 6.707 ms | 5.00 | 20.02 | 81% | ok | 1.2x (30%) |
| blas2 | gemv | f32 | n=4096 | 3.089 ms | 10.86 | 21.73 | 88% | ok | 1.9x (47%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.940 ms | 4.23 | 16.91 | 68% | ok | 1.2x (31%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.478 ms | 6.13 | 12.26 | 50% | ok | 0.8x (21%) |
| blas2 | symv | f64 | n=4096 | 11.398 ms | 2.94 | 5.89 | 24% | ok | 1.1x (27%) |
| blas2 | trmv | f64 | n=4096 | 4.380 ms | 3.83 | 15.34 | 62% | ok | 1.0x (25%) |
| blas2 | trmv | f32 | n=4096 | 3.113 ms | 5.39 | 10.79 | 44% | ok | 1.1x (27%) |
| blas2 | trsv | f64 | n=4096 | 4.416 ms | 3.80 | 15.21 | 62% | ok | 1.0x (25%) |
| blas2 | trsv | f32 | n=4096 | 2.871 ms | 5.84 | 11.70 | 47% | ok | 1.1x (26%) |
| blas2 | ger | f64 | n=4096 | 15.456 ms | 2.17 | 17.37 | 70% | ok | 1.0x (25%) |
| blas3 | gemm | f64 | 512x512x512 | 27.021 ms | 9.93 | 0.23 | 15% | ok | 1.9x (48%) |
| blas3 | gemm | f32 | 512x512x512 | 11.703 ms | 22.94 | 0.27 | 17% | ok | 2.1x (51%) |
| blas3 | gemmt | f64 | 512x512x512 | 37.905 ms | 3.54 | 0.17 | 5% | ok | 1.1x (27%) |
| blas3 | symm | f64 | 512x512x512 | 47.667 ms | 5.63 | 0.13 | 8% | ok | 1.1x (27%) |
| blas3 | syrk | f64 | 512x512x512 | 39.794 ms | 3.37 | 0.11 | 5% | ok | 1.0x (25%) |
| blas3 | syr2k | f64 | 512x512x512 | 82.696 ms | 3.25 | 0.08 | 5% | ok | 1.0x (25%) |
| blas3 | trmm | f64 | 512x512x512 | 22.868 ms | 5.87 | 0.23 | 9% | ok | 1.1x (28%) |
| blas3 | trsm | f64 | 512x512x512 | 23.135 ms | 5.80 | 0.23 | 9% | ok | 1.1x (28%) |
| blas3 | trsm | f32 | 512x512x512 | 10.656 ms | 12.60 | 0.25 | 9% | ok | 1.1x (28%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 28.477 ms | 0.19 | 1.49 | 6% | ok | 1.1x (27%) |
| factor | lu | f64 | 512x512 | 17.251 ms | 5.19 | 0.24 | 8% | ok | 1.4x (35%) |
| factor | lu | f32 | 512x512 | 8.843 ms | 10.12 | 0.24 | 8% | ok | 1.5x (37%) |
| factor | cholesky | f64 | 512x512 | 20.002 ms | 2.24 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 17.588 ms | 2.54 | 0.12 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 19.070 ms | 2.35 | 0.22 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 10.213 ms | 4.38 | 0.41 | 6% | ok | 1.1x (27%) |
| factor | qr | f64 | 512x512 | 146.283 ms | 1.22 | 0.03 | 2% | ok | 1.7x (43%) |
| factor | lq | f64 | 512x512 | 36.398 ms | 4.92 | 0.12 | 7% | ok | 1.7x (42%) |
| factor | hessenberg | f64 | 512x512 | 199.401 ms | 2.24 | 0.02 | 3% | ok | 1.7x (44%) |
| factor | householder | f64 | 512x512 | 0.463 ms | 2.27 | 9.07 | 37% | ok | 3.4x (84%) |
| factor | inv | f64 | 512x512 | 125.644 ms | 2.14 | 0.03 | 3% | ok | 1.2x (29%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.039 ms | 7.43 | 0.35 | 11% | ok | 1.4x (34%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.928 ms | 0.62 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 87.102 ms | 1.73 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 526.518 ms | 0.29 | 0.00 | 0% | ok | 1.1x (28%) |
| eigen | eigenvalue | f64 | 256x256 | 60.503 ms | 2.77 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 134.135 ms | 3.13 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.466 ms | 1.80 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 37.876 ms | 6.64 | 0.03 | 10% | ok | 1.8x (44%) |
| eigen | sylvester | f64 | 256x256 | 319.782 ms | 3.15 | 0.00 | 5% | ok | 1.2x (29%) |
| elementwise | exp | f64 | n=8388608 | 81.588 ms | 0.10 | 1.65 | 7% | ok | 1.1x (27%) |
| elementwise | log | f64 | n=8388608 | 100.193 ms | 0.08 | 1.34 | 5% | ok | 1.1x (28%) |
| elementwise | sin | f64 | n=8388608 | 137.115 ms | 0.06 | 0.98 | 4% | ok | 1.1x (27%) |
| elementwise | sqrt | f64 | n=8388608 | 65.329 ms | 0.13 | 2.05 | 8% | ok | 1.1x (28%) |
| elementwise | erf | f64 | n=8388608 | 107.460 ms | 0.08 | 1.25 | 5% | ok | 1.1x (28%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.201 ms | 2.18 | 22.67 | 92% | ok | 1.4x (35%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.931 ms | 1.36 | 14.10 | 57% | ok | 1.1x (28%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.282 ms | 6.67 | 14.02 | 57% | ok | 1.9x (48%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.821 ms | 0.28 | 0.97 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 9.234 ms | 0.22 | 1.80 | 7% | ok | 1.0x (25%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 33.838 ms | 0.33 | 0.02 | 0% | ok | 1.1x (27%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 53.122 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.182 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 94.465 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 110.589 ms | 0.20 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.125 ms | 0.57 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 66.351 ms | 0.03 | 0.00 | 0% | ok | 1.1x (28%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.694 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.200 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 112.539 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.525 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.851 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.044 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 26.280 ms | 0.43 | 0.03 | 1% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.230 ms | 0.34 | 1.36 | 5% | ok | 0.8x (20%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.143 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.711 ms | 1.04 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.172 ms | 0.35 | 1.39 | 6% | ok | 0.8x (21%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.130 ms | 0.00 | 0.00 | n/a | ok | 1.1x (26%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 81.209 ms | 0.28 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.729 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.873 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 93.322 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.689 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 10.763 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 610.937 ms | 0.51 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.926 ms | 0.43 | 1.70 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 24.383 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 166.921 ms | 1.85 | 0.01 | 3% | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.954 ms | 0.42 | 1.69 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.853 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1560.277 ms | 0.40 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.877 ms | 0.12 | 0.47 | 2% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 14.988 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1107.195 ms | 0.56 | 0.00 | 1% | ok | 1.1x (28%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.607 ms | 0.12 | 0.47 | 2% | ok | 1.1x (27%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.454 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 147.763 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.207 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.274 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.519 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.521 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 55.531 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.129 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.389 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 35.466 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.601 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 31.768 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 37.665 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.102 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.621 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.569 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.876 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.778 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.064 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.602 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.978 ms | 1.32 | 14.83 | 60% | ok | 1.1x (27%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.275 ms | 1.15 | 12.89 | 52% | ok | 1.1x (27%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.491 ms | 0.58 | 6.53 | 26% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.100 ms | 1.20 | 9.05 | 37% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 211.039 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 212.404 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 132.784 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.900 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.208 ms | 2.33 | 9.31 | 38% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.390 ms | 21.67 | 0.34 | 32% | ok | 3.9x (98%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.669 ms | 6.29 | 6.68 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.801 ms | 5.99 | 23.96 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.374 ms | 12.21 | 24.42 | n/a | ok | 1.0x (26%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.262 ms | 5.14 | 5.14 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.054 ms | 15.92 | 15.92 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.655 ms | 23.03 | 0.27 | n/a | ok | 2.1x (53%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.680 ms | 40.18 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.142 ms | 52.20 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.143 ms | 52.19 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 9.739 ms | 27.56 | 0.16 | n/a | ok | 4.0x (99%) |

## arduino-ventuno-q -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON: -march=armv8.2-a+dotprod+fp16+rcpc -mtune=cortex-a78c]`, SIMD NEON_WITHOUT_AES |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.6 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 16.8, f32 33.6 GFLOP/s |
| bandwidth | read 22.3, update 24.7, triad 23.1 GB/s (ceiling 24.7) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 94% | 104% | 0 |
| blas2 | 10 | 65% | 72% | 0 |
| blas3 | 10 | 48% | 58% | 0 |
| factor | 12 | 12% | 31% | 0 |
| eigen | 8 | 15% | 32% | 0 |
| elementwise | 5 | 5% | 7% | 0 |
| sparse | 5 | 30% | 67% | 0 |
| direct | 19 | 2% | 7% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 3% | 10% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 58% | 0 |
| mg | 1 | 36% | 36% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 67% | 133% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.729 ms | 2.93 | 23.43 | 95% | ok |
| blas1 | dot | f32 | n=8388608 | 2.884 ms | 5.82 | 23.27 | 94% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.832 ms | 2.14 | 25.71 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.960 ms | 4.24 | 25.42 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 5.315 ms | 1.58 | 25.25 | 102% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.156 ms | 5.32 | 21.27 | 86% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.693 ms | 0.96 | 7.72 | 31% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.642 ms | 0.48 | 3.80 | 15% | ok |
| blas1 | sum | f64 | n=8388608 | 8.520 ms | 0.98 | 7.88 | 32% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 200.676 ms | 0.92 | 0.67 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 8.336 ms | 4.03 | 16.11 | 65% | ok |
| blas2 | gemv | f32 | n=4096 | 4.081 ms | 8.22 | 16.45 | 67% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.820 ms | 4.29 | 17.17 | 70% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.023 ms | 8.34 | 16.69 | 68% | ok |
| blas2 | symv | f64 | n=4096 | 11.651 ms | 2.88 | 5.77 | 23% | ok |
| blas2 | trmv | f64 | n=4096 | 4.947 ms | 3.39 | 13.58 | 55% | ok |
| blas2 | trmv | f32 | n=4096 | 2.701 ms | 6.21 | 12.44 | 50% | ok |
| blas2 | trsv | f64 | n=4096 | 4.767 ms | 3.52 | 14.09 | 57% | ok |
| blas2 | trsv | f32 | n=4096 | 2.741 ms | 6.12 | 12.25 | 50% | ok |
| blas2 | ger | f64 | n=4096 | 15.173 ms | 2.21 | 17.70 | 72% | ok |
| blas3 | gemm | f64 | 512x512x512 | 28.428 ms | 9.44 | 0.22 | 56% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.787 ms | 19.47 | 0.23 | 58% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 18.050 ms | 7.44 | 0.35 | 44% | ok |
| blas3 | symm | f64 | 512x512x512 | 29.199 ms | 9.19 | 0.22 | 55% | ok |
| blas3 | syrk | f64 | 512x512x512 | 16.754 ms | 8.01 | 0.25 | 48% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 32.987 ms | 8.14 | 0.19 | 48% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.623 ms | 7.62 | 0.30 | 45% | ok |
| blas3 | trsm | f64 | 512x512x512 | 17.978 ms | 7.47 | 0.29 | 44% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.506 ms | 15.78 | 0.31 | 47% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 32.400 ms | 0.16 | 1.31 | 5% | ok |
| factor | lu | f64 | 512x512 | 26.429 ms | 3.39 | 0.16 | 20% | ok |
| factor | lu | f32 | 512x512 | 13.440 ms | 6.66 | 0.16 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 22.472 ms | 1.99 | 0.19 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 21.010 ms | 2.13 | 0.10 | 6% | ok |
| factor | ldlt | f64 | 512x512 | 21.646 ms | 2.07 | 0.19 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 11.212 ms | 3.99 | 0.37 | 24% | ok |
| factor | qr | f64 | 512x512 | 246.672 ms | 0.73 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 61.103 ms | 2.93 | 0.07 | 17% | ok |
| factor | hessenberg | f64 | 512x512 | 342.069 ms | 1.31 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.534 ms | 0.68 | 2.73 | 11% | ok |
| factor | inv | f64 | 512x512 | 145.011 ms | 1.85 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.974 ms | 5.27 | 0.25 | 31% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.918 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.261 ms | 1.55 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 605.460 ms | 0.25 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.977 ms | 2.47 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 151.401 ms | 2.77 | 0.01 | 16% | ok |
| eigen | svd | f64 | 256x256 | 230.204 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 47.480 ms | 5.30 | 0.02 | 32% | ok |
| eigen | sylvester | f64 | 256x256 | 358.914 ms | 2.80 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 90.251 ms | 0.09 | 1.49 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 111.208 ms | 0.08 | 1.21 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 150.918 ms | 0.06 | 0.89 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 73.338 ms | 0.11 | 1.83 | 7% | ok |
| elementwise | erf | f64 | n=8388608 | 119.184 ms | 0.07 | 1.13 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.637 ms | 1.60 | 16.63 | 67% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.111 ms | 1.24 | 12.90 | 52% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.975 ms | 3.50 | 7.35 | 30% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.209 ms | 0.25 | 0.87 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.746 ms | 0.23 | 1.90 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 37.280 ms | 0.30 | 0.02 | 2% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 66.207 ms | 0.17 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.388 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 103.420 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 122.294 ms | 0.18 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 42.682 ms | 0.52 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 79.393 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.404 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.424 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 125.841 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.568 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.129 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.622 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 29.737 ms | 0.38 | 0.02 | 2% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.840 ms | 0.41 | 1.64 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.301 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.998 ms | 0.93 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.771 ms | 0.43 | 1.71 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.487 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 88.051 ms | 0.25 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.736 ms | 0.13 | 0.53 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 8.848 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 103.739 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.851 ms | 0.13 | 0.52 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 12.232 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 666.020 ms | 0.46 | 0.00 | 3% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.415 ms | 0.45 | 1.82 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.212 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 187.096 ms | 1.65 | 0.00 | 10% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.353 ms | 0.46 | 1.83 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.705 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1768.966 ms | 0.35 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.165 ms | 0.12 | 0.46 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.900 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1235.380 ms | 0.50 | 0.00 | 3% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 30.112 ms | 0.11 | 0.45 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.920 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 162.378 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 54.174 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 29.224 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.696 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 34.211 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 62.440 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.724 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.705 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 39.409 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.823 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 34.713 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 40.814 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.110 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.961 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 31.213 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.114 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 3.013 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.066 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.742 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.064 ms | 1.27 | 14.21 | 58% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.481 ms | 1.05 | 11.82 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.933 ms | 0.89 | 10.00 | 41% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.334 ms | 1.19 | 8.99 | 36% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 235.461 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 230.707 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 146.331 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.158 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.057 ms | 4.14 | 16.54 | 67% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 41.753 ms | 6.43 | 0.10 | 38% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.749 ms | 22.40 | 23.80 | 133% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.857 ms | 5.87 | 23.49 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.416 ms | 11.85 | 23.70 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.699 ms | 24.01 | 24.01 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.543 ms | 10.87 | 10.87 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.420 ms | 15.41 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 16.688 ms | 16.09 | 0.13 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 16.576 ms | 16.19 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.678 ms | 72.99 | 0.43 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.139 ms | 20.43 | 0.12 | n/a | ok |

## arduino-ventuno-q -- native-fast, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON: -march=armv8.2-a+dotprod+fp16+rcpc -mtune=cortex-a78c]`, SIMD NEON_WITHOUT_AES |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 67.5 GFLOP/s, f32 135.7 GFLOP/s |
| ... hardware, NEON | f64 67.5, f32 134.9 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 67.5, f32 135.7 GFLOP/s |
| bandwidth | read 23.1, update 24.7, triad 22.7 GB/s (ceiling 24.7) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 98% | 115% | 0 | 28% |
| blas2 | 10 | 56% | 88% | 0 | 26% |
| blas3 | 10 | 37% | 55% | 0 | 79% |
| factor | 12 | 6% | 37% | 0 | 37% |
| eigen | 8 | 4% | 13% | 0 | 28% |
| elementwise | 5 | 5% | 8% | 0 | 28% |
| sparse | 5 | 55% | 92% | 0 | 27% |
| direct | 19 | 1% | 7% | 0 | 27% |
| ordering | 5 | n/a | n/a | 0 | 27% |
| direct3d | 12 | 2% | 7% | 0 | 27% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 52% | 60% | 0 | 27% |
| mg | 1 | 37% | 37% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 74% | 99% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 55% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.535 ms | 3.03 | 24.25 | 98% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.794 ms | 6.00 | 24.01 | 97% | ok | 1.0x (26%) |
| blas1 | axpy | f64 | n=8388608 | 8.149 ms | 2.06 | 24.71 | 100% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 3.957 ms | 4.24 | 25.44 | 103% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 4.740 ms | 1.77 | 28.31 | 115% | ok | 1.1x (28%) |
| blas1 | two_norm | f64 | n=8388608 | 2.619 ms | 6.41 | 25.63 | 104% | ok | 1.2x (30%) |
| blas1 | one_norm | f64 | n=8388608 | 7.460 ms | 1.12 | 9.00 | 36% | ok | 1.2x (29%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.559 ms | 0.54 | 4.31 | 17% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.470 ms | 1.12 | 8.98 | 36% | ok | 1.1x (29%) |
| blas1 | dot_k3 | f64 | n=8388608 | 179.851 ms | 1.03 | 0.75 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 6.728 ms | 4.99 | 19.96 | 81% | ok | 1.2x (31%) |
| blas2 | gemv | f32 | n=4096 | 3.070 ms | 10.93 | 21.87 | 88% | ok | 1.3x (33%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.730 ms | 4.34 | 17.37 | 70% | ok | 1.0x (25%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.844 ms | 6.93 | 13.86 | 56% | ok | 0.8x (21%) |
| blas2 | symv | f64 | n=4096 | 11.287 ms | 2.97 | 5.95 | 24% | ok | 1.0x (26%) |
| blas2 | trmv | f64 | n=4096 | 4.878 ms | 3.44 | 13.77 | 56% | ok | 1.0x (25%) |
| blas2 | trmv | f32 | n=4096 | 2.636 ms | 6.37 | 12.74 | 52% | ok | 1.0x (26%) |
| blas2 | trsv | f64 | n=4096 | 4.822 ms | 3.48 | 13.93 | 56% | ok | 1.0x (25%) |
| blas2 | trsv | f32 | n=4096 | 2.680 ms | 6.26 | 12.53 | 51% | ok | 1.0x (26%) |
| blas2 | ger | f64 | n=4096 | 14.715 ms | 2.28 | 18.25 | 74% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 7.415 ms | 36.20 | 0.85 | 54% | ok | 3.8x (96%) |
| blas3 | gemm | f32 | 512x512x512 | 3.577 ms | 75.04 | 0.88 | 55% | ok | 3.9x (96%) |
| blas3 | gemmt | f64 | 512x512x512 | 5.907 ms | 22.72 | 1.07 | 34% | ok | 3.1x (76%) |
| blas3 | symm | f64 | 512x512x512 | 8.209 ms | 32.70 | 0.77 | 48% | ok | 3.6x (89%) |
| blas3 | syrk | f64 | 512x512x512 | 5.307 ms | 25.29 | 0.79 | 37% | ok | 3.2x (79%) |
| blas3 | syr2k | f64 | 512x512x512 | 10.454 ms | 25.68 | 0.60 | 38% | ok | 3.2x (79%) |
| blas3 | trmm | f64 | 512x512x512 | 7.605 ms | 17.65 | 0.69 | 26% | ok | 2.3x (58%) |
| blas3 | trsm | f64 | 512x512x512 | 7.876 ms | 17.04 | 0.67 | 25% | ok | 2.3x (57%) |
| blas3 | trsm | f32 | 512x512x512 | 3.577 ms | 37.53 | 0.73 | 28% | ok | 2.4x (59%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 29.870 ms | 0.18 | 1.42 | 6% | ok | 1.1x (27%) |
| factor | lu | f64 | 512x512 | 17.027 ms | 5.26 | 0.25 | 8% | ok | 1.6x (39%) |
| factor | lu | f32 | 512x512 | 9.004 ms | 9.94 | 0.23 | 7% | ok | 1.5x (37%) |
| factor | cholesky | f64 | 512x512 | 19.896 ms | 2.25 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 18.812 ms | 2.38 | 0.11 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 18.989 ms | 2.36 | 0.22 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 10.302 ms | 4.34 | 0.41 | 6% | ok | 1.1x (27%) |
| factor | qr | f64 | 512x512 | 146.736 ms | 1.22 | 0.03 | 2% | ok | 1.7x (42%) |
| factor | lq | f64 | 512x512 | 36.356 ms | 4.92 | 0.12 | 7% | ok | 1.7x (42%) |
| factor | hessenberg | f64 | 512x512 | 199.533 ms | 2.24 | 0.02 | 3% | ok | 1.7x (43%) |
| factor | householder | f64 | 512x512 | 0.454 ms | 2.31 | 9.24 | 37% | ok | 3.4x (84%) |
| factor | inv | f64 | 512x512 | 130.176 ms | 2.06 | 0.03 | 3% | ok | 1.1x (28%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.221 ms | 7.32 | 0.34 | 11% | ok | 1.4x (35%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.643 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 87.667 ms | 1.72 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 533.780 ms | 0.28 | 0.00 | 0% | ok | 1.1x (28%) |
| eigen | eigenvalue | f64 | 256x256 | 60.897 ms | 2.76 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 135.622 ms | 3.09 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.968 ms | 1.79 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 28.804 ms | 8.74 | 0.04 | 13% | ok | 1.6x (41%) |
| eigen | sylvester | f64 | 256x256 | 316.511 ms | 3.18 | 0.00 | 5% | ok | 1.1x (28%) |
| elementwise | exp | f64 | n=8388608 | 81.857 ms | 0.10 | 1.64 | 7% | ok | 1.1x (28%) |
| elementwise | log | f64 | n=8388608 | 101.623 ms | 0.08 | 1.32 | 5% | ok | 1.1x (27%) |
| elementwise | sin | f64 | n=8388608 | 136.139 ms | 0.06 | 0.99 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 67.482 ms | 0.12 | 1.99 | 8% | ok | 1.1x (27%) |
| elementwise | erf | f64 | n=8388608 | 108.211 ms | 0.08 | 1.24 | 5% | ok | 1.1x (28%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.197 ms | 2.19 | 22.75 | 92% | ok | 1.4x (34%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.996 ms | 1.31 | 13.64 | 55% | ok | 1.1x (26%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.104 ms | 6.86 | 14.42 | 58% | ok | 2.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 12.107 ms | 0.27 | 0.95 | 4% | ok | 1.1x (27%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.830 ms | 0.23 | 1.89 | 8% | ok | 1.0x (25%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 33.882 ms | 0.33 | 0.02 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 59.432 ms | 0.19 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.780 ms | 0.45 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 93.520 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 112.144 ms | 0.20 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 38.741 ms | 0.58 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 70.546 ms | 0.03 | 0.00 | 0% | ok | 1.1x (28%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.707 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.386 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 112.429 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.513 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 39.190 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.087 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 26.553 ms | 0.42 | 0.03 | 1% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.924 ms | 0.39 | 1.57 | 6% | ok | 1.0x (24%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.563 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.962 ms | 1.02 | 0.07 | 2% | ok | 1.1x (27%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.815 ms | 0.42 | 1.67 | 7% | ok | 1.0x (24%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.995 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 80.218 ms | 0.28 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.544 ms | 0.14 | 0.55 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 8.000 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 94.736 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.598 ms | 0.14 | 0.54 | 2% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 11.058 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 612.967 ms | 0.50 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.685 ms | 0.44 | 1.75 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 25.456 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 168.674 ms | 1.83 | 0.01 | 3% | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.590 ms | 0.44 | 1.78 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.955 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1590.540 ms | 0.39 | 0.00 | 1% | ok | 1.1x (28%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.097 ms | 0.12 | 0.48 | 2% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 15.157 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1133.938 ms | 0.54 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.262 ms | 0.12 | 0.46 | 2% | ok | 1.0x (26%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.236 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 148.136 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 49.045 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.508 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.574 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.653 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 55.205 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.657 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.399 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 35.518 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.581 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 31.672 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 37.510 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.104 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.533 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.696 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.841 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.745 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.570 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.966 ms | 1.33 | 14.92 | 60% | ok | 1.1x (26%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.280 ms | 1.15 | 12.86 | 52% | ok | 1.1x (27%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.665 ms | 0.98 | 11.00 | 45% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.059 ms | 1.20 | 9.07 | 37% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 208.665 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 208.767 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 129.223 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.762 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 3.667 ms | 4.58 | 18.30 | 74% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 10.494 ms | 25.58 | 0.40 | 38% | ok | 4.0x (99%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.729 ms | 23.00 | 24.44 | 99% | ok | 1.0x (26%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.787 ms | 6.02 | 24.08 | n/a | ok | 1.0x (26%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.378 ms | 12.17 | 24.35 | n/a | ok | 1.0x (26%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.683 ms | 24.55 | 24.55 | n/a | ok | 1.0x (26%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.362 ms | 12.32 | 12.32 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 4.470 ms | 60.05 | 0.70 | n/a | ok | 3.9x (97%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.250 ms | 63.16 | 0.49 | n/a | ok | 3.9x (98%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.191 ms | 64.06 | 0.38 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.961 ms | 279.30 | 1.64 | n/a | ok | 3.8x (96%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 6.004 ms | 44.71 | 0.26 | n/a | ok | 2.2x (55%) |

## arduino-ventuno-q -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.6 GFLOP/s |
| ... this build, scalar | f64 8.4, f32 8.4 GFLOP/s |
| bandwidth | read 22.2, update 24.6, triad 23.0 GB/s (ceiling 24.6) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 86% | 104% | 0 |
| blas2 | 10 | 59% | 97% | 0 |
| blas3 | 10 | 84% | 93% | 0 |
| factor | 12 | 41% | 58% | 0 |
| eigen | 8 | 34% | 75% | 0 |
| elementwise | 5 | 5% | 7% | 0 |
| sparse | 5 | 30% | 68% | 0 |
| direct | 23 | 2% | 11% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 15 | 7% | 41% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 58% | 0 |
| mg | 1 | 36% | 36% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 34% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.693 ms | 2.95 | 23.57 | 96% | ok |
| blas1 | dot | f32 | n=8388608 | 4.369 ms | 3.84 | 15.36 | 63% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.853 ms | 2.14 | 25.64 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.003 ms | 4.19 | 25.14 | 102% | ok |
| blas1 | scale | f64 | n=8388608 | 5.448 ms | 1.54 | 24.64 | 100% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.180 ms | 5.28 | 21.10 | 86% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.030 ms | 1.04 | 8.36 | 34% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.346 ms | 0.48 | 3.87 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.234 ms | 1.02 | 8.15 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 199.669 ms | 0.92 | 0.67 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 8.909 ms | 3.77 | 15.07 | 61% | ok |
| blas2 | gemv | f32 | n=4096 | 3.722 ms | 9.01 | 18.04 | 73% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 9.331 ms | 3.60 | 14.39 | 59% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.689 ms | 9.10 | 18.20 | 74% | ok |
| blas2 | symv | f64 | n=4096 | 11.678 ms | 2.87 | 5.75 | 23% | ok |
| blas2 | trmv | f64 | n=4096 | 6.299 ms | 2.66 | 10.66 | 43% | ok |
| blas2 | trmv | f32 | n=4096 | 3.256 ms | 5.15 | 10.32 | 42% | ok |
| blas2 | trsv | f64 | n=4096 | 6.417 ms | 2.61 | 10.47 | 43% | ok |
| blas2 | trsv | f32 | n=4096 | 3.224 ms | 5.20 | 10.42 | 42% | ok |
| blas2 | ger | f64 | n=4096 | 11.249 ms | 2.98 | 23.87 | 97% | ok |
| blas3 | gemm | f64 | 512x512x512 | 17.588 ms | 15.26 | 0.36 | 91% | ok |
| blas3 | gemm | f32 | 512x512x512 | 8.631 ms | 31.10 | 0.36 | 93% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 11.269 ms | 11.91 | 0.56 | 71% | ok |
| blas3 | symm | f64 | 512x512x512 | 18.003 ms | 14.91 | 0.35 | 89% | ok |
| blas3 | syrk | f64 | 512x512x512 | 9.763 ms | 13.75 | 0.43 | 82% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 19.122 ms | 14.04 | 0.33 | 84% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.038 ms | 14.85 | 0.58 | 88% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.630 ms | 13.94 | 0.54 | 83% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.001 ms | 26.84 | 0.52 | 80% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 31.842 ms | 0.17 | 1.33 | 5% | ok |
| factor | lu | f64 | 512x512 | 9.184 ms | 9.74 | 0.46 | 58% | ok |
| factor | lu | f32 | 512x512 | 5.533 ms | 16.17 | 0.38 | 48% | ok |
| factor | cholesky | f64 | 512x512 | 5.466 ms | 8.18 | 0.77 | 49% | ok |
| factor | cholesky | f32 | 512x512 | 3.224 ms | 13.88 | 0.65 | 41% | ok |
| factor | ldlt | f64 | 512x512 | 20.928 ms | 2.14 | 0.20 | 13% | ok |
| factor | ldlt_bk | f64 | 512x512 | 11.057 ms | 4.05 | 0.38 | 24% | ok |
| factor | qr | f64 | 512x512 | 19.704 ms | 9.08 | 0.21 | 54% | ok |
| factor | lq | f64 | 512x512 | 60.566 ms | 2.95 | 0.07 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 347.616 ms | 1.29 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.580 ms | 0.66 | 2.66 | 11% | ok |
| factor | inv | f64 | 512x512 | 130.386 ms | 2.06 | 0.03 | 12% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 10.050 ms | 8.90 | 0.42 | 53% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 7.382 ms | 3.03 | 0.14 | 18% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.324 ms | 1.55 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 590.340 ms | 0.26 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 45.048 ms | 3.72 | 0.02 | 22% | ok |
| eigen | schur | f64 | 256x256 | 63.047 ms | 6.65 | 0.02 | 40% | ok |
| eigen | svd | f64 | 256x256 | 29.242 ms | 12.62 | 0.04 | 75% | ok |
| eigen | expm | f64 | 256x256 | 37.512 ms | 6.71 | 0.03 | 40% | ok |
| eigen | sylvester | f64 | 256x256 | 178.590 ms | 5.64 | 0.01 | 34% | ok |
| elementwise | exp | f64 | n=8388608 | 89.409 ms | 0.09 | 1.50 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 111.913 ms | 0.07 | 1.20 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 148.917 ms | 0.06 | 0.90 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 73.432 ms | 0.11 | 1.83 | 7% | ok |
| elementwise | erf | f64 | n=8388608 | 118.055 ms | 0.07 | 1.14 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.639 ms | 1.60 | 16.61 | 68% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.125 ms | 1.23 | 12.82 | 52% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.085 ms | 3.47 | 7.29 | 30% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.051 ms | 0.25 | 0.88 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.995 ms | 0.22 | 1.85 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 37.277 ms | 0.30 | 0.02 | 2% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 59.299 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.351 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 103.441 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 122.175 ms | 0.18 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 42.439 ms | 0.53 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 75.617 ms | 0.03 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.675 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 14.873 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 126.292 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.573 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.071 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.602 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 29.724 ms | 0.38 | 0.02 | 2% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.849 ms | 0.41 | 1.64 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.202 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.827 ms | 0.95 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.755 ms | 0.43 | 1.72 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.642 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 87.327 ms | 0.26 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 6.022 ms | 0.13 | 0.50 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 8.758 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 103.188 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.966 ms | 0.13 | 0.51 | 2% | ok |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 16.746 ms | 0.67 | 0.04 | 4% | ok |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 3.885 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 11.268 ms | 0.99 | 0.06 | 6% | ok |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.165 ms | 0.65 | 2.60 | 11% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 11.771 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 671.285 ms | 0.46 | 0.00 | 3% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.658 ms | 0.44 | 1.76 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 27.340 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 181.866 ms | 1.70 | 0.00 | 10% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.410 ms | 0.45 | 1.82 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.718 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1643.713 ms | 0.38 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 30.105 ms | 0.11 | 0.45 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.891 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1230.569 ms | 0.50 | 0.00 | 3% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 30.932 ms | 0.11 | 0.44 | 2% | ok |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 7.428 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 44.334 ms | 6.96 | 0.02 | 41% | ok |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 2.436 ms | 1.38 | 5.53 | 23% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 71.430 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 161.867 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 55.311 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 29.101 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.970 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 34.431 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 63.351 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.981 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.885 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 39.472 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 36.271 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.038 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 41.132 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.112 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 5.082 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 30.667 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.155 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 3.064 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.065 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 3.005 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.061 ms | 1.27 | 14.23 | 58% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.484 ms | 1.05 | 11.81 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.953 ms | 0.89 | 9.93 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.633 ms | 1.18 | 8.91 | 36% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 229.754 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 234.699 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 146.085 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.328 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 8.029 ms | 2.09 | 8.36 | 34% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 48.547 ms | 5.53 | 0.09 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.940 ms | 5.71 | 6.06 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.844 ms | 5.90 | 23.60 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.402 ms | 11.96 | 23.93 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.649 ms | 4.60 | 4.60 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.178 ms | 14.25 | 14.25 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 24.695 ms | 10.87 | 0.13 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 26.335 ms | 10.19 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 20.304 ms | 13.22 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 20.299 ms | 13.22 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 38.488 ms | 6.97 | 0.04 | n/a | ok |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

## arduino-ventuno-q -- blas, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 67.7 GFLOP/s, f32 134.9 GFLOP/s |
| ... hardware, NEON | f64 67.7, f32 134.9 GFLOP/s |
| ... this build, scalar | f64 33.9, f32 33.9 GFLOP/s |
| bandwidth | read 23.0, update 24.6, triad 22.5 GB/s (ceiling 24.6) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 98% | 115% | 0 | 27% |
| blas2 | 10 | 65% | 118% | 0 | 36% |
| blas3 | 10 | 73% | 81% | 0 | 87% |
| factor | 12 | 18% | 39% | 0 | 43% |
| eigen | 8 | 11% | 28% | 0 | 33% |
| elementwise | 5 | 5% | 8% | 0 | 27% |
| sparse | 5 | 57% | 92% | 0 | 28% |
| direct | 23 | 1% | 11% | 0 | 28% |
| ordering | 5 | n/a | n/a | 0 | 28% |
| direct3d | 15 | 3% | 24% | 0 | 28% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 28% |
| smoother | 3 | 53% | 61% | 0 | 28% |
| mg | 1 | 36% | 36% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 31% | 38% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 54% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.542 ms | 3.03 | 24.22 | 98% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.797 ms | 6.00 | 24.00 | 97% | ok | 1.6x (39%) |
| blas1 | axpy | f64 | n=8388608 | 8.099 ms | 2.07 | 24.86 | 101% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 3.993 ms | 4.20 | 25.21 | 102% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 4.729 ms | 1.77 | 28.38 | 115% | ok | 1.2x (29%) |
| blas1 | two_norm | f64 | n=8388608 | 2.612 ms | 6.42 | 25.69 | 104% | ok | 1.2x (30%) |
| blas1 | one_norm | f64 | n=8388608 | 7.337 ms | 1.14 | 9.15 | 37% | ok | 1.1x (27%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.564 ms | 0.54 | 4.31 | 18% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 8.038 ms | 1.04 | 8.35 | 34% | ok | 1.0x (26%) |
| blas1 | dot_k3 | f64 | n=8388608 | 182.462 ms | 1.01 | 0.74 | 3% | ok | 1.1x (27%) |
| blas2 | gemv | f64 | n=4096 | 5.078 ms | 6.61 | 26.44 | 107% | ok | 1.8x (44%) |
| blas2 | gemv | f32 | n=4096 | 2.599 ms | 12.91 | 25.84 | 105% | ok | 1.4x (36%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 8.420 ms | 3.99 | 15.95 | 65% | ok | 1.1x (28%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.598 ms | 7.30 | 14.60 | 59% | ok | 0.8x (20%) |
| blas2 | symv | f64 | n=4096 | 6.396 ms | 5.25 | 10.50 | 43% | ok | 1.8x (46%) |
| blas2 | trmv | f64 | n=4096 | 3.630 ms | 4.62 | 18.51 | 75% | ok | 1.7x (43%) |
| blas2 | trmv | f32 | n=4096 | 2.154 ms | 7.79 | 15.59 | 63% | ok | 1.5x (38%) |
| blas2 | trsv | f64 | n=4096 | 6.019 ms | 2.79 | 11.16 | 45% | ok | 1.1x (27%) |
| blas2 | trsv | f32 | n=4096 | 3.074 ms | 5.46 | 10.93 | 44% | ok | 1.0x (26%) |
| blas2 | ger | f64 | n=4096 | 9.235 ms | 3.63 | 29.08 | 118% | ok | 1.2x (30%) |
| blas3 | gemm | f64 | 512x512x512 | 5.027 ms | 53.40 | 1.25 | 79% | ok | 3.5x (87%) |
| blas3 | gemm | f32 | 512x512x512 | 2.450 ms | 109.56 | 1.28 | 81% | ok | 3.5x (88%) |
| blas3 | gemmt | f64 | 512x512x512 | 3.358 ms | 39.97 | 1.87 | 59% | ok | 3.4x (84%) |
| blas3 | symm | f64 | 512x512x512 | 5.053 ms | 53.13 | 1.25 | 78% | ok | 3.6x (89%) |
| blas3 | syrk | f64 | 512x512x512 | 3.233 ms | 41.52 | 1.30 | 61% | ok | 3.0x (75%) |
| blas3 | syr2k | f64 | 512x512x512 | 5.723 ms | 46.90 | 1.10 | 69% | ok | 3.3x (84%) |
| blas3 | trmm | f64 | 512x512x512 | 2.537 ms | 52.91 | 2.07 | 78% | ok | 3.6x (89%) |
| blas3 | trsm | f64 | 512x512x512 | 2.708 ms | 49.57 | 1.94 | 73% | ok | 3.6x (89%) |
| blas3 | trsm | f32 | 512x512x512 | 1.444 ms | 92.95 | 1.82 | 69% | ok | 3.5x (87%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 29.595 ms | 0.18 | 1.43 | 6% | ok | 1.1x (27%) |
| factor | lu | f64 | 512x512 | 4.170 ms | 21.46 | 1.01 | 32% | ok | 2.2x (55%) |
| factor | lu | f32 | 512x512 | 3.138 ms | 28.52 | 0.67 | 21% | ok | 1.8x (44%) |
| factor | cholesky | f64 | 512x512 | 2.902 ms | 15.41 | 1.45 | 23% | ok | 1.9x (47%) |
| factor | cholesky | f32 | 512x512 | 1.998 ms | 22.40 | 1.05 | 17% | ok | 1.6x (40%) |
| factor | ldlt | f64 | 512x512 | 18.937 ms | 2.36 | 0.22 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 10.353 ms | 4.32 | 0.41 | 6% | ok | 1.1x (27%) |
| factor | qr | f64 | 512x512 | 9.587 ms | 18.67 | 0.44 | 28% | ok | 2.1x (51%) |
| factor | lq | f64 | 512x512 | 36.585 ms | 4.89 | 0.11 | 7% | ok | 1.7x (41%) |
| factor | hessenberg | f64 | 512x512 | 201.928 ms | 2.22 | 0.02 | 3% | ok | 1.7x (43%) |
| factor | householder | f64 | 512x512 | 0.442 ms | 2.38 | 9.49 | 39% | ok | 3.6x (89%) |
| factor | inv | f64 | 512x512 | 117.675 ms | 2.28 | 0.04 | 3% | ok | 1.1x (28%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.176 ms | 12.47 | 0.58 | 18% | ok | 1.4x (35%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 5.388 ms | 4.15 | 0.19 | 6% | ok | 1.4x (34%) |
| eigen | eigen_symmetric | f64 | 256x256 | 87.239 ms | 1.73 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 545.782 ms | 0.28 | 0.00 | 0% | ok | 1.1x (27%) |
| eigen | eigenvalue | f64 | 256x256 | 37.277 ms | 4.50 | 0.03 | 7% | ok | 1.2x (30%) |
| eigen | schur | f64 | 256x256 | 48.016 ms | 8.74 | 0.02 | 13% | ok | 1.3x (33%) |
| eigen | svd | f64 | 256x256 | 19.590 ms | 18.84 | 0.05 | 28% | ok | 1.5x (37%) |
| eigen | expm | f64 | 256x256 | 23.495 ms | 10.71 | 0.04 | 16% | ok | 1.6x (40%) |
| eigen | sylvester | f64 | 256x256 | 135.895 ms | 7.41 | 0.01 | 11% | ok | 1.3x (33%) |
| elementwise | exp | f64 | n=8388608 | 81.972 ms | 0.10 | 1.64 | 7% | ok | 1.1x (27%) |
| elementwise | log | f64 | n=8388608 | 100.855 ms | 0.08 | 1.33 | 5% | ok | 1.1x (28%) |
| elementwise | sin | f64 | n=8388608 | 136.160 ms | 0.06 | 0.99 | 4% | ok | 1.1x (27%) |
| elementwise | sqrt | f64 | n=8388608 | 67.665 ms | 0.12 | 1.98 | 8% | ok | 1.1x (27%) |
| elementwise | erf | f64 | n=8388608 | 108.280 ms | 0.08 | 1.24 | 5% | ok | 1.1x (27%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.202 ms | 2.18 | 22.65 | 92% | ok | 1.4x (34%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.934 ms | 1.35 | 14.08 | 57% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.137 ms | 6.82 | 14.35 | 58% | ok | 2.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.846 ms | 0.28 | 0.97 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 9.093 ms | 0.22 | 1.83 | 7% | ok | 1.0x (25%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 33.960 ms | 0.33 | 0.02 | 0% | ok | 1.1x (27%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 53.492 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.108 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 93.787 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 111.469 ms | 0.20 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 38.409 ms | 0.58 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 67.503 ms | 0.03 | 0.00 | 0% | ok | 1.1x (28%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.667 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.418 ms | 0.00 | 0.00 | n/a | ok | 1.2x (30%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 111.472 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.525 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.827 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 4.996 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 26.636 ms | 0.42 | 0.03 | 1% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.926 ms | 0.39 | 1.57 | 6% | ok | 1.0x (24%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.009 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.642 ms | 1.05 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.798 ms | 0.42 | 1.68 | 7% | ok | 1.0x (24%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.999 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 79.411 ms | 0.28 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.782 ms | 0.13 | 0.52 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.924 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 93.737 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.771 ms | 0.13 | 0.52 | 2% | ok | 1.0x (26%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 14.585 ms | 0.77 | 0.05 | 1% | ok | 1.1x (29%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 3.493 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 9.852 ms | 1.14 | 0.07 | 2% | ok | 1.1x (29%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.107 ms | 0.68 | 2.73 | 11% | ok | 1.1x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 10.543 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 616.463 ms | 0.50 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.834 ms | 0.43 | 1.72 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 24.510 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 165.101 ms | 1.87 | 0.01 | 3% | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.793 ms | 0.43 | 1.73 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.601 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1567.274 ms | 0.39 | 0.00 | 1% | ok | 1.0x (26%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.254 ms | 0.12 | 0.46 | 2% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 14.902 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1107.794 ms | 0.56 | 0.00 | 1% | ok | 1.1x (28%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.694 ms | 0.12 | 0.47 | 2% | ok | 1.1x (27%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 6.677 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 27.854 ms | 11.08 | 0.03 | 16% | ok | 1.6x (40%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 2.272 ms | 1.48 | 5.93 | 24% | ok | 1.1x (27%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.131 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 147.596 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.452 ms | 0.00 | 0.00 | n/a | ok | 1.1x (29%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.474 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.527 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.958 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 55.756 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.329 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.465 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 35.653 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.616 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 31.759 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 37.653 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.102 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.569 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilut | f64 | 128^2 grid | 27.137 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.861 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ildl | f64 | 128^2 grid | 2.775 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ssor | f64 | 128^2 grid | 0.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.762 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.957 ms | 1.34 | 14.99 | 61% | ok | 1.1x (26%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.257 ms | 1.16 | 12.99 | 53% | ok | 1.1x (28%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.672 ms | 0.98 | 10.98 | 45% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.401 ms | 1.19 | 8.97 | 36% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 207.774 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 210.753 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 136.925 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.954 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.201 ms | 2.33 | 9.32 | 38% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.653 ms | 21.21 | 0.33 | 31% | ok | 3.8x (96%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.664 ms | 6.30 | 6.69 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.809 ms | 5.97 | 23.89 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.367 ms | 12.28 | 24.55 | n/a | ok | 1.0x (26%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.269 ms | 5.13 | 5.13 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.051 ms | 15.96 | 15.96 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.525 ms | 23.29 | 0.27 | n/a | ok | 2.1x (54%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.686 ms | 40.15 | 0.31 | n/a | ok | 3.9x (98%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.143 ms | 52.20 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.141 ms | 52.21 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 9.756 ms | 27.51 | 0.16 | n/a | ok | 3.9x (99%) |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

