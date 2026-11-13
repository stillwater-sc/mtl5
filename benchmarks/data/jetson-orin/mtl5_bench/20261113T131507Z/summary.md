# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,6, CPUs 0,1,2,3,4,5, clock policy nvpmodel 25W (mode 1), jetson_clocks; schedutil 1344-1344 MHz on cpus 0,1,2,3,4,5, 2026-11-13T14:05:56Z.

## jetson-orin -- native, 1 thread, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 10.7 GFLOP/s, f32 21.5 GFLOP/s |
| ... hardware, NEON | f64 10.7, f32 21.5 GFLOP/s |
| ... this build, scalar | f64 5.4, f32 5.4 GFLOP/s |
| bandwidth | read 11.5, update 16.3, triad 14.9 GB/s (ceiling 16.3) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 71% | 129% | 0 |
| blas2 | 10 | 50% | 123% | 0 |
| blas3 | 10 | 33% | 34% | 0 |
| factor | 12 | 13% | 33% | 0 |
| eigen | 8 | 16% | 23% | 0 |
| elementwise | 5 | 8% | 15% | 0 |
| sparse | 5 | 32% | 73% | 0 |
| direct | 19 | 2% | 9% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 12 | 3% | 10% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 60% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 33% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.574 ms | 1.45 | 11.60 | 71% | ok |
| blas1 | dot | f32 | n=8388608 | 6.595 ms | 2.54 | 10.18 | 62% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.245 ms | 1.37 | 16.44 | 101% | ok |
| blas1 | axpy | f32 | n=8388608 | 5.976 ms | 2.81 | 16.84 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 6.359 ms | 1.32 | 21.11 | 129% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.603 ms | 2.99 | 11.98 | 73% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.501 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.753 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.502 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 192.271 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 13.055 ms | 2.57 | 10.29 | 63% | ok |
| blas2 | gemv | f32 | n=4096 | 7.736 ms | 4.34 | 8.68 | 53% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 16.432 ms | 2.04 | 8.17 | 50% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.359 ms | 5.28 | 10.56 | 65% | ok |
| blas2 | symv | f64 | n=4096 | 21.273 ms | 1.58 | 3.16 | 19% | ok |
| blas2 | trmv | f64 | n=4096 | 8.793 ms | 1.91 | 7.64 | 47% | ok |
| blas2 | trmv | f32 | n=4096 | 6.956 ms | 2.41 | 4.83 | 30% | ok |
| blas2 | trsv | f64 | n=4096 | 8.762 ms | 1.91 | 7.67 | 47% | ok |
| blas2 | trsv | f32 | n=4096 | 6.800 ms | 2.47 | 4.94 | 30% | ok |
| blas2 | ger | f64 | n=4096 | 13.421 ms | 2.50 | 20.01 | 123% | ok |
| blas3 | gemm | f64 | 512x512x512 | 75.284 ms | 3.57 | 0.08 | 33% | ok |
| blas3 | gemm | f32 | 512x512x512 | 38.205 ms | 7.03 | 0.08 | 33% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 48.301 ms | 2.78 | 0.13 | 26% | ok |
| blas3 | symm | f64 | 512x512x512 | 76.703 ms | 3.50 | 0.08 | 33% | ok |
| blas3 | syrk | f64 | 512x512x512 | 49.717 ms | 2.70 | 0.08 | 25% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 92.923 ms | 2.89 | 0.07 | 27% | ok |
| blas3 | trmm | f64 | 512x512x512 | 37.237 ms | 3.60 | 0.14 | 34% | ok |
| blas3 | trsm | f64 | 512x512x512 | 37.435 ms | 3.59 | 0.14 | 33% | ok |
| blas3 | trsm | f32 | 512x512x512 | 18.547 ms | 7.24 | 0.14 | 34% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.751 ms | 0.54 | 4.36 | 27% | ok |
| factor | lu | f64 | 512x512 | 35.886 ms | 2.49 | 0.12 | 23% | ok |
| factor | lu | f32 | 512x512 | 20.072 ms | 4.46 | 0.10 | 21% | ok |
| factor | cholesky | f64 | 512x512 | 32.019 ms | 1.40 | 0.13 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 30.031 ms | 1.49 | 0.07 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 34.723 ms | 1.29 | 0.12 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 86.787 ms | 0.52 | 0.05 | 5% | ok |
| factor | qr | f64 | 512x512 | 338.507 ms | 0.53 | 0.01 | 5% | ok |
| factor | lq | f64 | 512x512 | 93.287 ms | 1.92 | 0.04 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 477.346 ms | 0.94 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 2.052 ms | 0.51 | 2.04 | 13% | ok |
| factor | inv | f64 | 512x512 | 192.632 ms | 1.39 | 0.02 | 13% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 25.225 ms | 3.55 | 0.17 | 33% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.297 ms | 0.36 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 145.287 ms | 1.04 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 901.373 ms | 0.17 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 98.825 ms | 1.70 | 0.01 | 16% | ok |
| eigen | schur | f64 | 256x256 | 214.322 ms | 1.96 | 0.00 | 18% | ok |
| eigen | svd | f64 | 256x256 | 359.492 ms | 1.03 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 102.619 ms | 2.45 | 0.01 | 23% | ok |
| eigen | sylvester | f64 | 256x256 | 535.719 ms | 1.88 | 0.00 | 18% | ok |
| elementwise | exp | f64 | n=8388608 | 75.806 ms | 0.11 | 1.77 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.044 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 178.395 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.225 ms | 0.15 | 2.43 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 126.811 ms | 0.07 | 1.06 | 6% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.296 ms | 1.14 | 11.86 | 73% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.061 ms | 0.85 | 8.89 | 55% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.131 ms | 2.44 | 5.14 | 32% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.767 ms | 0.16 | 0.55 | 3% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.672 ms | 0.17 | 1.43 | 9% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 278.731 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 102.885 ms | 0.11 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.190 ms | 0.30 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 131.399 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 187.137 ms | 0.12 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 63.372 ms | 0.35 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 288.606 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.189 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.393 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.138 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 66.913 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 8.758 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 267.141 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.202 ms | 0.34 | 1.37 | 8% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 18.066 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.518 ms | 0.68 | 0.04 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.110 ms | 0.36 | 1.43 | 9% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.628 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 112.740 ms | 0.20 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.552 ms | 0.10 | 0.40 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 14.674 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 162.621 ms | 0.14 | 0.00 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.568 ms | 0.10 | 0.40 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 31.456 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 25405.035 ms | 0.01 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 10.173 ms | 0.33 | 1.32 | 8% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 65.313 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 277.662 ms | 1.11 | 0.00 | 10% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.117 ms | 0.33 | 1.33 | 8% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.371 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1942.313 ms | 0.32 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.898 ms | 0.09 | 0.37 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 42.827 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 3191.773 ms | 0.19 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.943 ms | 0.09 | 0.36 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.050 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 148.465 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 77.972 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.635 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.911 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.589 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.710 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.041 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.861 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.871 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.612 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.508 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 68.050 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.168 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.693 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 45.612 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.571 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.240 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.090 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.314 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.015 ms | 0.87 | 9.73 | 60% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.738 ms | 0.70 | 7.85 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.515 ms | 0.58 | 6.49 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 45.035 ms | 0.85 | 6.45 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 354.792 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 412.376 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 193.769 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 21.146 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 12.502 ms | 1.34 | 5.37 | 33% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 75.465 ms | 3.56 | 0.06 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 4.571 ms | 3.67 | 3.90 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.648 ms | 2.97 | 11.88 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.708 ms | 6.20 | 12.39 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.674 ms | 2.96 | 2.96 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.757 ms | 9.55 | 9.55 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 38.019 ms | 7.06 | 0.08 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 41.090 ms | 6.53 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 31.872 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 31.870 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 60.128 ms | 4.46 | 0.03 | n/a | ok |

## jetson-orin -- native, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 64.3 GFLOP/s, f32 128.6 GFLOP/s |
| ... hardware, NEON | f64 64.3, f32 128.6 GFLOP/s |
| ... this build, scalar | f64 32.1, f32 32.1 GFLOP/s |
| bandwidth | read 27.7, update 35.8, triad 33.3 GB/s (ceiling 35.8) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 85% | 123% | 0 | 35% |
| blas2 | 10 | 56% | 76% | 0 | 17% |
| blas3 | 10 | 5% | 23% | 0 | 17% |
| factor | 12 | 2% | 22% | 0 | 23% |
| eigen | 8 | 3% | 8% | 0 | 17% |
| elementwise | 5 | 3% | 7% | 0 | 17% |
| sparse | 5 | 25% | 91% | 0 | 18% |
| direct | 19 | 0% | 4% | 0 | 17% |
| ordering | 4 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 1% | 4% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 22% | 27% | 0 | 17% |
| mg | 1 | 19% | 19% | 0 | 18% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 15% | 32% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 71% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.547 ms | 3.69 | 29.52 | 82% | ok | 2.5x (42%) |
| blas1 | dot | f32 | n=8388608 | 2.200 ms | 7.62 | 30.50 | 85% | ok | 3.0x (50%) |
| blas1 | axpy | f64 | n=8388608 | 5.834 ms | 2.88 | 34.51 | 96% | ok | 2.1x (35%) |
| blas1 | axpy | f32 | n=8388608 | 2.637 ms | 6.36 | 38.17 | 107% | ok | 2.3x (38%) |
| blas1 | scale | f64 | n=8388608 | 3.046 ms | 2.75 | 44.06 | 123% | ok | 2.1x (35%) |
| blas1 | two_norm | f64 | n=8388608 | 2.174 ms | 7.72 | 30.86 | 86% | ok | 2.6x (43%) |
| blas1 | one_norm | f64 | n=8388608 | 12.501 ms | 0.67 | 5.37 | 15% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.752 ms | 0.45 | 3.58 | 10% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.507 ms | 0.67 | 5.37 | 15% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 192.411 ms | 0.96 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 5.533 ms | 6.06 | 24.27 | 68% | ok | 2.4x (39%) |
| blas2 | gemv | f32 | n=4096 | 2.468 ms | 13.60 | 27.20 | 76% | ok | 3.1x (52%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 6.170 ms | 5.44 | 21.76 | 61% | ok | 2.7x (44%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.318 ms | 10.11 | 20.24 | 57% | ok | 1.9x (32%) |
| blas2 | symv | f64 | n=4096 | 21.242 ms | 1.58 | 3.16 | 9% | ok | 1.0x (17%) |
| blas2 | trmv | f64 | n=4096 | 9.483 ms | 1.77 | 7.08 | 20% | ok | 0.9x (15%) |
| blas2 | trmv | f32 | n=4096 | 6.943 ms | 2.42 | 4.84 | 14% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 8.755 ms | 1.92 | 7.67 | 21% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 6.770 ms | 2.48 | 4.96 | 14% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 13.381 ms | 2.51 | 20.07 | 56% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 21.328 ms | 12.59 | 0.29 | 20% | ok | 3.5x (59%) |
| blas3 | gemm | f32 | 512x512x512 | 8.975 ms | 29.91 | 0.35 | 23% | ok | 4.3x (71%) |
| blas3 | gemmt | f64 | 512x512x512 | 48.170 ms | 2.79 | 0.13 | 4% | ok | 1.0x (17%) |
| blas3 | symm | f64 | 512x512x512 | 75.838 ms | 3.54 | 0.08 | 6% | ok | 1.0x (17%) |
| blas3 | syrk | f64 | 512x512x512 | 49.302 ms | 2.72 | 0.09 | 4% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 512x512x512 | 92.813 ms | 2.89 | 0.07 | 5% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 512x512x512 | 39.442 ms | 3.40 | 0.13 | 5% | ok | 0.9x (16%) |
| blas3 | trsm | f64 | 512x512x512 | 38.993 ms | 3.44 | 0.13 | 5% | ok | 1.0x (16%) |
| blas3 | trsm | f32 | 512x512x512 | 19.019 ms | 7.06 | 0.14 | 5% | ok | 1.0x (16%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.283 ms | 0.57 | 4.57 | 13% | ok | 1.1x (18%) |
| factor | lu | f64 | 512x512 | 24.592 ms | 3.64 | 0.17 | 6% | ok | 1.5x (24%) |
| factor | lu | f32 | 512x512 | 14.304 ms | 6.26 | 0.15 | 5% | ok | 1.4x (23%) |
| factor | cholesky | f64 | 512x512 | 31.747 ms | 1.41 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 30.177 ms | 1.48 | 0.07 | 1% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 34.449 ms | 1.30 | 0.12 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 84.688 ms | 0.53 | 0.05 | 1% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 203.221 ms | 0.88 | 0.02 | 1% | ok | 1.7x (28%) |
| factor | lq | f64 | 512x512 | 66.508 ms | 2.69 | 0.06 | 4% | ok | 1.4x (23%) |
| factor | hessenberg | f64 | 512x512 | 291.198 ms | 1.54 | 0.01 | 2% | ok | 1.6x (27%) |
| factor | householder | f64 | 512x512 | 0.531 ms | 1.98 | 7.90 | 22% | ok | 3.9x (64%) |
| factor | inv | f64 | 512x512 | 184.827 ms | 1.45 | 0.02 | 2% | ok | 1.0x (17%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 19.343 ms | 4.63 | 0.22 | 7% | ok | 1.3x (22%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 62.339 ms | 0.36 | 0.02 | 1% | ok | 1.0x (16%) |
| eigen | eigen_symmetric | f64 | 256x256 | 145.915 ms | 1.03 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 902.360 ms | 0.17 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 98.787 ms | 1.70 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 213.522 ms | 1.96 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 358.160 ms | 1.03 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 49.324 ms | 5.10 | 0.02 | 8% | ok | 2.1x (35%) |
| eigen | sylvester | f64 | 256x256 | 507.490 ms | 1.98 | 0.00 | 3% | ok | 1.1x (18%) |
| elementwise | exp | f64 | n=8388608 | 75.708 ms | 0.11 | 1.77 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 107.862 ms | 0.08 | 1.24 | 3% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 178.030 ms | 0.05 | 0.75 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 54.998 ms | 0.15 | 2.44 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 126.300 ms | 0.07 | 1.06 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.834 ms | 3.14 | 32.66 | 91% | ok | 2.8x (46%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.074 ms | 0.85 | 8.86 | 25% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 5.814 ms | 7.20 | 15.14 | 42% | ok | 2.9x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.756 ms | 0.16 | 0.55 | 2% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.019 ms | 0.18 | 1.51 | 4% | ok | 1.1x (18%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 276.578 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 102.874 ms | 0.11 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.196 ms | 0.30 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 132.335 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 187.893 ms | 0.12 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 62.048 ms | 0.36 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 286.443 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.097 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.172 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.150 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 65.965 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 8.789 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 265.641 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.173 ms | 0.35 | 1.39 | 4% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 18.054 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.529 ms | 0.68 | 0.04 | 1% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.144 ms | 0.35 | 1.41 | 4% | ok | 1.0x (16%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.612 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 113.374 ms | 0.20 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.544 ms | 0.10 | 0.40 | 1% | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 14.650 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 163.214 ms | 0.14 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.608 ms | 0.10 | 0.40 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 31.035 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 25358.418 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 10.006 ms | 0.34 | 1.35 | 4% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 64.802 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 277.310 ms | 1.11 | 0.00 | 2% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.019 ms | 0.34 | 1.35 | 4% | ok | 1.0x (17%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.165 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1901.116 ms | 0.32 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.802 ms | 0.09 | 0.37 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 42.033 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 3156.822 ms | 0.20 | 0.00 | 0% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.831 ms | 0.09 | 0.37 | 1% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 148.484 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.031 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.661 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.908 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.638 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.720 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.033 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.926 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.803 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.679 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.523 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 68.373 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.167 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.700 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 45.347 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.608 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.242 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.089 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.321 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.013 ms | 0.87 | 9.73 | 27% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.739 ms | 0.70 | 7.84 | 22% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.521 ms | 0.58 | 6.49 | 18% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 42.281 ms | 0.91 | 6.87 | 19% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 353.283 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 413.710 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 197.399 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 21.178 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 12.505 ms | 1.34 | 5.37 | 15% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.927 ms | 20.77 | 0.32 | 32% | ok | 5.8x (97%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 5.605 ms | 2.99 | 3.18 | 9% | ok | 0.8x (14%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.584 ms | 3.00 | 12.02 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.700 ms | 6.21 | 12.43 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.674 ms | 2.96 | 2.96 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.758 ms | 9.55 | 9.55 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 8.975 ms | 29.91 | 0.35 | n/a | ok | 4.2x (71%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 7.067 ms | 37.98 | 0.30 | n/a | ok | 5.8x (97%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.471 ms | 49.06 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.484 ms | 48.95 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 10.299 ms | 26.06 | 0.15 | n/a | ok | 5.8x (97%) |

## jetson-orin -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 10.7 GFLOP/s, f32 21.5 GFLOP/s |
| ... hardware, NEON | f64 10.7, f32 21.5 GFLOP/s |
| ... this build, NEON | f64 10.7, f32 21.5 GFLOP/s |
| bandwidth | read 11.5, update 16.2, triad 14.8 GB/s (ceiling 16.2) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 73% | 131% | 0 |
| blas2 | 10 | 57% | 123% | 0 |
| blas3 | 10 | 50% | 61% | 0 |
| factor | 12 | 13% | 33% | 0 |
| eigen | 8 | 16% | 32% | 0 |
| elementwise | 5 | 8% | 15% | 0 |
| sparse | 5 | 31% | 73% | 0 |
| direct | 19 | 2% | 9% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 12 | 3% | 10% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 40% | 48% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 66% | 119% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.586 ms | 1.45 | 11.58 | 71% | ok |
| blas1 | dot | f32 | n=8388608 | 5.642 ms | 2.97 | 11.89 | 73% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.240 ms | 1.37 | 16.45 | 101% | ok |
| blas1 | axpy | f32 | n=8388608 | 5.974 ms | 2.81 | 16.85 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 6.329 ms | 1.33 | 21.21 | 131% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.590 ms | 3.00 | 12.01 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.502 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.751 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.501 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 192.103 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 14.478 ms | 2.32 | 9.27 | 57% | ok |
| blas2 | gemv | f32 | n=4096 | 6.275 ms | 5.35 | 10.70 | 66% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 14.015 ms | 2.39 | 9.58 | 59% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.115 ms | 5.49 | 10.98 | 68% | ok |
| blas2 | symv | f64 | n=4096 | 21.348 ms | 1.57 | 3.15 | 19% | ok |
| blas2 | trmv | f64 | n=4096 | 9.112 ms | 1.84 | 7.37 | 45% | ok |
| blas2 | trmv | f32 | n=4096 | 4.265 ms | 3.93 | 7.88 | 49% | ok |
| blas2 | trsv | f64 | n=4096 | 9.139 ms | 1.84 | 7.35 | 45% | ok |
| blas2 | trsv | f32 | n=4096 | 4.219 ms | 3.98 | 7.96 | 49% | ok |
| blas2 | ger | f64 | n=4096 | 13.403 ms | 2.50 | 20.03 | 123% | ok |
| blas3 | gemm | f64 | 512x512x512 | 41.437 ms | 6.48 | 0.15 | 60% | ok |
| blas3 | gemm | f32 | 512x512x512 | 20.480 ms | 13.11 | 0.15 | 61% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 25.787 ms | 5.20 | 0.24 | 48% | ok |
| blas3 | symm | f64 | 512x512x512 | 42.556 ms | 6.31 | 0.15 | 59% | ok |
| blas3 | syrk | f64 | 512x512x512 | 49.517 ms | 2.71 | 0.08 | 25% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 92.068 ms | 2.92 | 0.07 | 27% | ok |
| blas3 | trmm | f64 | 512x512x512 | 24.897 ms | 5.39 | 0.21 | 50% | ok |
| blas3 | trsm | f64 | 512x512x512 | 25.765 ms | 5.21 | 0.20 | 49% | ok |
| blas3 | trsm | f32 | 512x512x512 | 12.534 ms | 10.71 | 0.21 | 50% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.471 ms | 0.56 | 4.48 | 28% | ok |
| factor | lu | f64 | 512x512 | 35.774 ms | 2.50 | 0.12 | 23% | ok |
| factor | lu | f32 | 512x512 | 20.324 ms | 4.40 | 0.10 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 34.234 ms | 1.31 | 0.12 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 30.156 ms | 1.48 | 0.07 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 34.631 ms | 1.29 | 0.12 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 84.417 ms | 0.53 | 0.05 | 5% | ok |
| factor | qr | f64 | 512x512 | 333.546 ms | 0.54 | 0.01 | 5% | ok |
| factor | lq | f64 | 512x512 | 93.457 ms | 1.91 | 0.04 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 470.722 ms | 0.95 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 1.987 ms | 0.53 | 2.11 | 13% | ok |
| factor | inv | f64 | 512x512 | 193.277 ms | 1.39 | 0.02 | 13% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 25.348 ms | 3.53 | 0.17 | 33% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.584 ms | 0.36 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 145.543 ms | 1.04 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 959.519 ms | 0.16 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 98.704 ms | 1.70 | 0.01 | 16% | ok |
| eigen | schur | f64 | 256x256 | 214.856 ms | 1.95 | 0.00 | 18% | ok |
| eigen | svd | f64 | 256x256 | 359.355 ms | 1.03 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 72.312 ms | 3.48 | 0.01 | 32% | ok |
| eigen | sylvester | f64 | 256x256 | 514.319 ms | 1.96 | 0.00 | 18% | ok |
| elementwise | exp | f64 | n=8388608 | 76.097 ms | 0.11 | 1.76 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.284 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 178.826 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.223 ms | 0.15 | 2.43 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 126.726 ms | 0.07 | 1.06 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.302 ms | 1.14 | 11.83 | 73% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.056 ms | 0.86 | 8.91 | 55% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.492 ms | 2.39 | 5.03 | 31% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 22.503 ms | 0.14 | 0.51 | 3% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.634 ms | 0.17 | 1.43 | 9% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 281.322 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 90.360 ms | 0.12 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.298 ms | 0.30 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 135.444 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 191.855 ms | 0.12 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 61.964 ms | 0.36 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 291.542 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.051 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.046 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.125 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 65.968 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 8.808 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 270.330 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.187 ms | 0.35 | 1.38 | 9% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 18.227 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.586 ms | 0.67 | 0.04 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.110 ms | 0.36 | 1.43 | 9% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.663 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 115.466 ms | 0.19 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.607 ms | 0.10 | 0.40 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 14.810 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 165.970 ms | 0.13 | 0.00 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.720 ms | 0.10 | 0.39 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 31.364 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 25398.565 ms | 0.01 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 10.165 ms | 0.33 | 1.33 | 8% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 65.422 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 278.297 ms | 1.11 | 0.00 | 10% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.341 ms | 0.33 | 1.30 | 8% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.083 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1898.672 ms | 0.33 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 37.383 ms | 0.09 | 0.36 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 41.818 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 3180.967 ms | 0.19 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 37.497 ms | 0.09 | 0.36 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.245 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 148.413 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.176 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.614 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.752 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.640 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.227 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.127 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 50.113 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.689 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.627 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.574 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 68.169 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.166 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.993 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 45.377 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.491 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.245 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.090 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.370 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 7.633 ms | 0.34 | 3.84 | 24% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.744 ms | 0.70 | 7.83 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.519 ms | 0.58 | 6.49 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 44.919 ms | 0.86 | 6.47 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 353.131 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 414.149 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 198.071 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.863 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 6.263 ms | 2.68 | 10.72 | 66% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 64.160 ms | 4.18 | 0.07 | 39% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.310 ms | 12.80 | 13.60 | 119% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.620 ms | 2.99 | 11.94 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.713 ms | 6.18 | 12.37 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.227 ms | 13.67 | 13.67 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.348 ms | 7.15 | 7.15 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 26.048 ms | 10.31 | 0.12 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 27.843 ms | 9.64 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 33.419 ms | 8.03 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 8.584 ms | 31.27 | 0.18 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 23.388 ms | 11.48 | 0.07 | n/a | ok |

## jetson-orin -- native-fast, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 64.3 GFLOP/s, f32 128.7 GFLOP/s |
| ... hardware, NEON | f64 64.3, f32 128.7 GFLOP/s |
| ... this build, NEON | f64 64.2, f32 128.2 GFLOP/s |
| bandwidth | read 27.3, update 34.8, triad 32.4 GB/s (ceiling 34.8) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 87% | 136% | 0 | 40% |
| blas2 | 10 | 50% | 82% | 0 | 17% |
| blas3 | 10 | 19% | 51% | 0 | 38% |
| factor | 12 | 2% | 22% | 0 | 23% |
| eigen | 8 | 3% | 8% | 0 | 17% |
| elementwise | 5 | 4% | 7% | 0 | 17% |
| sparse | 5 | 25% | 93% | 0 | 18% |
| direct | 19 | 0% | 4% | 0 | 17% |
| ordering | 4 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 1% | 4% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 19% | 23% | 0 | 17% |
| mg | 1 | 20% | 20% | 0 | 18% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 34% | 39% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 81% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.565 ms | 3.68 | 29.40 | 85% | ok | 2.5x (42%) |
| blas1 | dot | f32 | n=8388608 | 2.230 ms | 7.52 | 30.09 | 87% | ok | 2.5x (42%) |
| blas1 | axpy | f64 | n=8388608 | 5.038 ms | 3.33 | 39.96 | 115% | ok | 2.4x (40%) |
| blas1 | axpy | f32 | n=8388608 | 2.467 ms | 6.80 | 40.80 | 117% | ok | 2.4x (40%) |
| blas1 | scale | f64 | n=8388608 | 2.838 ms | 2.96 | 47.29 | 136% | ok | 2.2x (37%) |
| blas1 | two_norm | f64 | n=8388608 | 2.182 ms | 7.69 | 30.75 | 88% | ok | 2.6x (43%) |
| blas1 | one_norm | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 15% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.749 ms | 0.45 | 3.58 | 10% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 15% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 190.873 ms | 0.97 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 5.340 ms | 6.28 | 25.14 | 72% | ok | 2.7x (45%) |
| blas2 | gemv | f32 | n=4096 | 2.346 ms | 14.30 | 28.62 | 82% | ok | 2.7x (45%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.667 ms | 4.38 | 17.51 | 50% | ok | 1.8x (30%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.443 ms | 9.75 | 19.50 | 56% | ok | 1.8x (30%) |
| blas2 | symv | f64 | n=4096 | 21.343 ms | 1.57 | 3.15 | 9% | ok | 1.0x (17%) |
| blas2 | trmv | f64 | n=4096 | 9.142 ms | 1.84 | 7.35 | 21% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 4.257 ms | 3.94 | 7.89 | 23% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 9.246 ms | 1.81 | 7.27 | 21% | ok | 1.0x (16%) |
| blas2 | trsv | f32 | n=4096 | 4.250 ms | 3.95 | 7.90 | 23% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 13.403 ms | 2.50 | 20.03 | 58% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 8.182 ms | 32.81 | 0.77 | 51% | ok | 5.1x (84%) |
| blas3 | gemm | f32 | 512x512x512 | 4.092 ms | 65.61 | 0.77 | 51% | ok | 5.0x (83%) |
| blas3 | gemmt | f64 | 512x512x512 | 7.401 ms | 18.14 | 0.85 | 28% | ok | 3.5x (58%) |
| blas3 | symm | f64 | 512x512x512 | 9.194 ms | 29.20 | 0.68 | 45% | ok | 4.6x (77%) |
| blas3 | syrk | f64 | 512x512x512 | 49.054 ms | 2.74 | 0.09 | 4% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 512x512x512 | 93.353 ms | 2.88 | 0.07 | 4% | ok | 1.0x (16%) |
| blas3 | trmm | f64 | 512x512x512 | 10.861 ms | 12.36 | 0.48 | 19% | ok | 2.3x (38%) |
| blas3 | trsm | f64 | 512x512x512 | 11.602 ms | 11.57 | 0.45 | 18% | ok | 2.2x (37%) |
| blas3 | trsm | f32 | 512x512x512 | 5.642 ms | 23.79 | 0.46 | 18% | ok | 2.2x (37%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.705 ms | 0.55 | 4.38 | 13% | ok | 1.0x (16%) |
| factor | lu | f64 | 512x512 | 24.906 ms | 3.59 | 0.17 | 6% | ok | 1.4x (24%) |
| factor | lu | f32 | 512x512 | 14.418 ms | 6.21 | 0.15 | 5% | ok | 1.4x (23%) |
| factor | cholesky | f64 | 512x512 | 33.889 ms | 1.32 | 0.12 | 2% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 30.092 ms | 1.49 | 0.07 | 1% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 34.425 ms | 1.30 | 0.12 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 83.917 ms | 0.53 | 0.05 | 1% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 203.731 ms | 0.88 | 0.02 | 1% | ok | 1.6x (27%) |
| factor | lq | f64 | 512x512 | 59.739 ms | 3.00 | 0.07 | 5% | ok | 1.6x (26%) |
| factor | hessenberg | f64 | 512x512 | 299.925 ms | 1.49 | 0.01 | 2% | ok | 1.6x (26%) |
| factor | householder | f64 | 512x512 | 0.542 ms | 1.94 | 7.74 | 22% | ok | 3.7x (61%) |
| factor | inv | f64 | 512x512 | 181.126 ms | 1.48 | 0.02 | 2% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 19.555 ms | 4.58 | 0.21 | 7% | ok | 1.3x (22%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.446 ms | 0.36 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 145.085 ms | 1.04 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 942.397 ms | 0.16 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 98.833 ms | 1.70 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 214.587 ms | 1.95 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 359.105 ms | 1.03 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 49.596 ms | 5.07 | 0.02 | 8% | ok | 1.5x (24%) |
| eigen | sylvester | f64 | 256x256 | 501.888 ms | 2.01 | 0.00 | 3% | ok | 1.0x (17%) |
| elementwise | exp | f64 | n=8388608 | 75.460 ms | 0.11 | 1.78 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 107.344 ms | 0.08 | 1.25 | 4% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 178.525 ms | 0.05 | 0.75 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 54.986 ms | 0.15 | 2.44 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 126.727 ms | 0.07 | 1.06 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.841 ms | 3.11 | 32.36 | 93% | ok | 2.7x (46%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.093 ms | 0.85 | 8.80 | 25% | ok | 1.0x (16%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.726 ms | 8.86 | 18.63 | 54% | ok | 3.7x (62%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.611 ms | 0.16 | 0.56 | 2% | ok | 1.1x (18%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.008 ms | 0.18 | 1.51 | 4% | ok | 1.1x (18%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 282.218 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 89.435 ms | 0.13 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.293 ms | 0.30 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 137.821 ms | 0.16 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 194.543 ms | 0.12 | 0.00 | 0% | ok | 1.0x (16%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 74.319 ms | 0.30 | 0.01 | 0% | ok | 0.8x (14%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 287.032 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.007 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.095 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.128 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 65.447 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 8.835 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 270.205 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.142 ms | 0.35 | 1.41 | 4% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 18.175 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.531 ms | 0.68 | 0.04 | 1% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.087 ms | 0.36 | 1.45 | 4% | ok | 1.0x (17%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.720 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 117.440 ms | 0.19 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.625 ms | 0.10 | 0.40 | 1% | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 14.740 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 167.649 ms | 0.13 | 0.00 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.632 ms | 0.10 | 0.40 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 31.490 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 25346.402 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 9.867 ms | 0.34 | 1.37 | 4% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 65.027 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 280.349 ms | 1.10 | 0.00 | 2% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.180 ms | 0.33 | 1.32 | 4% | ok | 1.0x (17%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.261 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1897.301 ms | 0.33 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.891 ms | 0.09 | 0.37 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 42.494 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 3097.120 ms | 0.20 | 0.00 | 0% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.976 ms | 0.09 | 0.36 | 1% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.192 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 147.449 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 77.681 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.433 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.588 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.555 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.318 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.919 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.759 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.527 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.453 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.410 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 68.655 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.166 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.993 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 45.202 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.474 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.263 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.090 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.397 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 7.514 ms | 0.35 | 3.90 | 11% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.744 ms | 0.70 | 7.83 | 23% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.518 ms | 0.58 | 6.49 | 19% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 42.124 ms | 0.91 | 6.90 | 20% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 358.587 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 419.394 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 194.431 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.906 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 6.261 ms | 2.68 | 10.72 | 31% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.238 ms | 21.93 | 0.34 | 34% | ok | 5.2x (87%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.322 ms | 12.69 | 13.49 | 39% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.601 ms | 3.00 | 11.98 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.680 ms | 6.26 | 12.52 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.198 ms | 14.01 | 14.01 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.349 ms | 7.14 | 7.14 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 5.141 ms | 52.22 | 0.61 | n/a | ok | 5.1x (84%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.320 ms | 50.45 | 0.39 | n/a | ok | 5.2x (87%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 6.467 ms | 41.51 | 0.24 | n/a | ok | 5.2x (86%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.767 ms | 151.89 | 0.89 | n/a | ok | 4.9x (81%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.533 ms | 59.22 | 0.35 | n/a | ok | 5.2x (86%) |

## jetson-orin -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 10.7 GFLOP/s, f32 21.4 GFLOP/s |
| ... hardware, NEON | f64 10.7, f32 21.4 GFLOP/s |
| ... this build, scalar | f64 5.4, f32 5.4 GFLOP/s |
| bandwidth | read 11.5, update 16.1, triad 14.7 GB/s (ceiling 16.1) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 72% | 131% | 0 |
| blas2 | 10 | 64% | 124% | 0 |
| blas3 | 10 | 89% | 96% | 0 |
| factor | 12 | 42% | 61% | 0 |
| eigen | 8 | 37% | 80% | 0 |
| elementwise | 5 | 8% | 15% | 0 |
| sparse | 5 | 32% | 74% | 0 |
| direct | 23 | 2% | 14% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 15 | 8% | 36% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 49% | 60% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 33% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.593 ms | 1.45 | 11.58 | 72% | ok |
| blas1 | dot | f32 | n=8388608 | 6.594 ms | 2.54 | 10.18 | 63% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.247 ms | 1.37 | 16.44 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 5.953 ms | 2.82 | 16.91 | 105% | ok |
| blas1 | scale | f64 | n=8388608 | 6.355 ms | 1.32 | 21.12 | 131% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.597 ms | 3.00 | 11.99 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.501 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.752 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.502 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 192.761 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 12.602 ms | 2.66 | 10.66 | 66% | ok |
| blas2 | gemv | f32 | n=4096 | 5.865 ms | 5.72 | 11.45 | 71% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 12.993 ms | 2.58 | 10.34 | 64% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.785 ms | 5.80 | 11.61 | 72% | ok |
| blas2 | symv | f64 | n=4096 | 16.025 ms | 2.09 | 4.19 | 26% | ok |
| blas2 | trmv | f64 | n=4096 | 8.571 ms | 1.96 | 7.84 | 49% | ok |
| blas2 | trmv | f32 | n=4096 | 4.803 ms | 3.49 | 6.99 | 43% | ok |
| blas2 | trsv | f64 | n=4096 | 8.538 ms | 1.97 | 7.87 | 49% | ok |
| blas2 | trsv | f32 | n=4096 | 4.825 ms | 3.48 | 6.96 | 43% | ok |
| blas2 | ger | f64 | n=4096 | 13.411 ms | 2.50 | 20.02 | 124% | ok |
| blas3 | gemm | f64 | 512x512x512 | 26.644 ms | 10.07 | 0.24 | 94% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.114 ms | 20.47 | 0.24 | 96% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 16.562 ms | 8.10 | 0.38 | 76% | ok |
| blas3 | symm | f64 | 512x512x512 | 26.628 ms | 10.08 | 0.24 | 94% | ok |
| blas3 | syrk | f64 | 512x512x512 | 14.424 ms | 9.31 | 0.29 | 87% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 28.228 ms | 9.51 | 0.22 | 89% | ok |
| blas3 | trmm | f64 | 512x512x512 | 13.629 ms | 9.85 | 0.38 | 92% | ok |
| blas3 | trsm | f64 | 512x512x512 | 14.943 ms | 8.98 | 0.35 | 84% | ok |
| blas3 | trsm | f32 | 512x512x512 | 7.561 ms | 17.75 | 0.35 | 83% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.865 ms | 0.54 | 4.30 | 27% | ok |
| factor | lu | f64 | 512x512 | 13.648 ms | 6.56 | 0.31 | 61% | ok |
| factor | lu | f32 | 512x512 | 8.394 ms | 10.66 | 0.25 | 50% | ok |
| factor | cholesky | f64 | 512x512 | 7.906 ms | 5.66 | 0.53 | 53% | ok |
| factor | cholesky | f32 | 512x512 | 4.955 ms | 9.03 | 0.42 | 42% | ok |
| factor | ldlt | f64 | 512x512 | 34.621 ms | 1.29 | 0.12 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 84.741 ms | 0.53 | 0.05 | 5% | ok |
| factor | qr | f64 | 512x512 | 28.458 ms | 6.29 | 0.15 | 59% | ok |
| factor | lq | f64 | 512x512 | 93.234 ms | 1.92 | 0.04 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 476.741 ms | 0.94 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 2.067 ms | 0.51 | 2.03 | 13% | ok |
| factor | inv | f64 | 512x512 | 175.707 ms | 1.53 | 0.02 | 14% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 14.989 ms | 5.97 | 0.28 | 56% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 10.310 ms | 2.17 | 0.10 | 20% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 145.246 ms | 1.04 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 937.397 ms | 0.16 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 62.822 ms | 2.67 | 0.02 | 25% | ok |
| eigen | schur | f64 | 256x256 | 91.016 ms | 4.61 | 0.01 | 43% | ok |
| eigen | svd | f64 | 256x256 | 43.178 ms | 8.55 | 0.02 | 80% | ok |
| eigen | expm | f64 | 256x256 | 56.879 ms | 4.42 | 0.02 | 41% | ok |
| eigen | sylvester | f64 | 256x256 | 255.649 ms | 3.94 | 0.01 | 37% | ok |
| elementwise | exp | f64 | n=8388608 | 76.149 ms | 0.11 | 1.76 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.395 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 179.495 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.124 ms | 0.15 | 2.43 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 127.732 ms | 0.07 | 1.05 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.288 ms | 1.14 | 11.90 | 74% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.059 ms | 0.86 | 8.90 | 55% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.127 ms | 2.45 | 5.14 | 32% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.513 ms | 0.16 | 0.56 | 3% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.850 ms | 0.17 | 1.40 | 9% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 285.328 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 101.240 ms | 0.11 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.481 ms | 0.30 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 133.237 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 190.539 ms | 0.12 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 61.945 ms | 0.36 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 298.119 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.129 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.222 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.128 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 67.838 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 8.847 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 273.309 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.160 ms | 0.35 | 1.40 | 9% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 18.250 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.646 ms | 0.67 | 0.04 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.133 ms | 0.35 | 1.42 | 9% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.607 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 114.334 ms | 0.20 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.728 ms | 0.10 | 0.39 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 14.696 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 166.279 ms | 0.13 | 0.00 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.713 ms | 0.10 | 0.39 | 2% | ok |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 28.248 ms | 0.40 | 0.03 | 4% | ok |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 5.951 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 20.586 ms | 0.54 | 0.04 | 5% | ok |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.372 ms | 0.55 | 2.20 | 14% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 31.372 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 25745.989 ms | 0.01 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 10.409 ms | 0.32 | 1.29 | 8% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 65.303 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 279.918 ms | 1.10 | 0.00 | 10% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.121 ms | 0.33 | 1.33 | 8% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.179 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1919.402 ms | 0.32 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 37.727 ms | 0.09 | 0.36 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 42.510 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 3241.004 ms | 0.19 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 37.338 ms | 0.09 | 0.36 | 2% | ok |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 11.413 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 79.275 ms | 3.89 | 0.01 | 36% | ok |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 3.161 ms | 1.07 | 4.26 | 26% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.113 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 148.372 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.068 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.592 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.999 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.630 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.617 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.172 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 50.010 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.788 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.625 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.472 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 68.559 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.165 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.845 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 45.289 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.460 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.269 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.090 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 3.972 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.043 ms | 0.86 | 9.64 | 60% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.740 ms | 0.70 | 7.84 | 49% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 8.029 ms | 0.33 | 3.65 | 23% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 45.032 ms | 0.85 | 6.45 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 352.753 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 358.517 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 190.624 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.918 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 12.504 ms | 1.34 | 5.37 | 33% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 75.778 ms | 3.54 | 0.06 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 4.583 ms | 3.66 | 3.89 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.625 ms | 2.98 | 11.93 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.714 ms | 6.18 | 12.37 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.675 ms | 2.96 | 2.96 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.761 ms | 9.53 | 9.53 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 38.023 ms | 7.06 | 0.08 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 41.103 ms | 6.53 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 31.895 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 31.867 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 60.127 ms | 4.46 | 0.03 | n/a | ok |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

## jetson-orin -- blas, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 64.0 GFLOP/s, f32 127.9 GFLOP/s |
| ... hardware, NEON | f64 64.0, f32 127.9 GFLOP/s |
| ... this build, scalar | f64 31.9, f32 31.9 GFLOP/s |
| bandwidth | read 30.1, update 33.4, triad 30.8 GB/s (ceiling 33.4) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 91% | 141% | 0 | 40% |
| blas2 | 10 | 61% | 89% | 0 | 33% |
| blas3 | 10 | 72% | 88% | 0 | 86% |
| factor | 12 | 13% | 24% | 0 | 28% |
| eigen | 8 | 7% | 19% | 0 | 20% |
| elementwise | 5 | 4% | 7% | 0 | 17% |
| sparse | 5 | 27% | 97% | 0 | 18% |
| direct | 23 | 0% | 6% | 0 | 17% |
| ordering | 4 | n/a | n/a | 0 | 17% |
| direct3d | 15 | 1% | 13% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 23% | 29% | 0 | 17% |
| mg | 1 | 21% | 21% | 0 | 18% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 16% | 32% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 72% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.547 ms | 3.69 | 29.52 | 88% | ok | 2.5x (42%) |
| blas1 | dot | f32 | n=8388608 | 2.197 ms | 7.64 | 30.54 | 91% | ok | 3.0x (50%) |
| blas1 | axpy | f64 | n=8388608 | 5.049 ms | 3.32 | 39.88 | 119% | ok | 2.4x (40%) |
| blas1 | axpy | f32 | n=8388608 | 2.469 ms | 6.80 | 40.77 | 122% | ok | 2.4x (40%) |
| blas1 | scale | f64 | n=8388608 | 2.853 ms | 2.94 | 47.04 | 141% | ok | 2.2x (37%) |
| blas1 | two_norm | f64 | n=8388608 | 2.179 ms | 7.70 | 30.80 | 92% | ok | 2.6x (43%) |
| blas1 | one_norm | f64 | n=8388608 | 12.501 ms | 0.67 | 5.37 | 16% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.751 ms | 0.45 | 3.58 | 11% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.502 ms | 0.67 | 5.37 | 16% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 191.724 ms | 0.96 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 4.609 ms | 7.28 | 29.13 | 87% | ok | 2.7x (46%) |
| blas2 | gemv | f32 | n=4096 | 2.266 ms | 14.81 | 29.63 | 89% | ok | 2.6x (43%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 6.547 ms | 5.13 | 20.51 | 61% | ok | 2.0x (33%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.790 ms | 8.85 | 17.71 | 53% | ok | 1.5x (25%) |
| blas2 | symv | f64 | n=4096 | 9.847 ms | 3.41 | 6.82 | 20% | ok | 1.6x (27%) |
| blas2 | trmv | f64 | n=4096 | 2.834 ms | 5.92 | 23.70 | 71% | ok | 3.0x (50%) |
| blas2 | trmv | f32 | n=4096 | 1.438 ms | 11.67 | 23.36 | 70% | ok | 3.3x (56%) |
| blas2 | trsv | f64 | n=4096 | 8.566 ms | 1.96 | 7.84 | 23% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 4.764 ms | 3.52 | 7.05 | 21% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 13.427 ms | 2.50 | 20.00 | 60% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 4.826 ms | 55.63 | 1.30 | 87% | ok | 5.5x (92%) |
| blas3 | gemm | f32 | 512x512x512 | 2.475 ms | 108.48 | 1.27 | 85% | ok | 5.3x (88%) |
| blas3 | gemmt | f64 | 512x512x512 | 4.117 ms | 32.60 | 1.53 | 51% | ok | 4.0x (67%) |
| blas3 | symm | f64 | 512x512x512 | 4.788 ms | 56.06 | 1.31 | 88% | ok | 5.6x (93%) |
| blas3 | syrk | f64 | 512x512x512 | 3.574 ms | 37.55 | 1.17 | 59% | ok | 4.0x (67%) |
| blas3 | syr2k | f64 | 512x512x512 | 6.009 ms | 44.68 | 1.05 | 70% | ok | 4.7x (78%) |
| blas3 | trmm | f64 | 512x512x512 | 2.576 ms | 52.11 | 2.04 | 81% | ok | 5.3x (88%) |
| blas3 | trsm | f64 | 512x512x512 | 2.899 ms | 46.30 | 1.81 | 72% | ok | 5.2x (86%) |
| blas3 | trsm | f32 | 512x512x512 | 1.570 ms | 85.49 | 1.67 | 67% | ok | 4.8x (80%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.922 ms | 0.54 | 4.28 | 13% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 5.709 ms | 15.67 | 0.73 | 24% | ok | 2.4x (40%) |
| factor | lu | f32 | 512x512 | 4.588 ms | 19.50 | 0.46 | 15% | ok | 1.8x (30%) |
| factor | cholesky | f64 | 512x512 | 4.154 ms | 10.77 | 1.01 | 17% | ok | 1.9x (32%) |
| factor | cholesky | f32 | 512x512 | 3.208 ms | 13.95 | 0.65 | 11% | ok | 1.5x (26%) |
| factor | ldlt | f64 | 512x512 | 34.497 ms | 1.30 | 0.12 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 84.073 ms | 0.53 | 0.05 | 1% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 13.784 ms | 12.98 | 0.30 | 20% | ok | 2.1x (34%) |
| factor | lq | f64 | 512x512 | 59.451 ms | 3.01 | 0.07 | 5% | ok | 1.6x (26%) |
| factor | hessenberg | f64 | 512x512 | 288.803 ms | 1.55 | 0.01 | 2% | ok | 1.7x (28%) |
| factor | householder | f64 | 512x512 | 0.568 ms | 1.85 | 7.38 | 22% | ok | 3.6x (61%) |
| factor | inv | f64 | 512x512 | 162.339 ms | 1.65 | 0.03 | 3% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 11.154 ms | 8.02 | 0.38 | 13% | ok | 1.3x (22%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 7.017 ms | 3.19 | 0.15 | 5% | ok | 1.5x (24%) |
| eigen | eigen_symmetric | f64 | 256x256 | 144.917 ms | 1.04 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 953.501 ms | 0.16 | 0.00 | 0% | ok | 1.0x (16%) |
| eigen | eigenvalue | f64 | 256x256 | 56.452 ms | 2.97 | 0.02 | 5% | ok | 1.1x (19%) |
| eigen | schur | f64 | 256x256 | 74.081 ms | 5.66 | 0.01 | 9% | ok | 1.2x (20%) |
| eigen | svd | f64 | 256x256 | 29.753 ms | 12.41 | 0.04 | 19% | ok | 1.5x (24%) |
| eigen | expm | f64 | 256x256 | 37.330 ms | 6.74 | 0.03 | 11% | ok | 1.5x (25%) |
| eigen | sylvester | f64 | 256x256 | 210.356 ms | 4.79 | 0.01 | 7% | ok | 1.2x (20%) |
| elementwise | exp | f64 | n=8388608 | 75.728 ms | 0.11 | 1.77 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 107.760 ms | 0.08 | 1.25 | 4% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 177.866 ms | 0.05 | 0.75 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 55.014 ms | 0.15 | 2.44 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 126.910 ms | 0.07 | 1.06 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.837 ms | 3.13 | 32.52 | 97% | ok | 2.7x (46%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.060 ms | 0.86 | 8.90 | 27% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.713 ms | 8.89 | 18.68 | 56% | ok | 3.6x (61%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.428 ms | 0.16 | 0.56 | 2% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.066 ms | 0.18 | 1.50 | 5% | ok | 1.1x (18%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 284.308 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 101.154 ms | 0.11 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.434 ms | 0.30 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 132.432 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 192.681 ms | 0.12 | 0.00 | 0% | ok | 1.0x (16%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 61.718 ms | 0.36 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 295.053 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.111 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.140 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.150 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 66.045 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 8.801 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 274.958 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.145 ms | 0.35 | 1.41 | 4% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 18.176 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.589 ms | 0.67 | 0.04 | 1% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.066 ms | 0.37 | 1.46 | 4% | ok | 1.0x (17%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.594 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 115.103 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.672 ms | 0.10 | 0.39 | 1% | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 14.634 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 167.040 ms | 0.13 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.721 ms | 0.10 | 0.39 | 1% | ok | 1.0x (17%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 64.421 ms | 0.17 | 0.01 | 0% | ok | 0.4x (7%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 6.004 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 92.405 ms | 0.12 | 0.01 | 0% | ok | 0.2x (4%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.447 ms | 0.52 | 2.09 | 6% | ok | 0.9x (16%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 31.529 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 25710.258 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 10.238 ms | 0.33 | 1.32 | 4% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 65.760 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 276.992 ms | 1.11 | 0.00 | 2% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.006 ms | 0.34 | 1.35 | 4% | ok | 1.0x (17%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.210 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1910.214 ms | 0.32 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 37.268 ms | 0.09 | 0.36 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 42.148 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 3164.961 ms | 0.20 | 0.00 | 0% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 37.461 ms | 0.09 | 0.36 | 1% | ok | 1.0x (17%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 11.419 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 452.756 ms | 0.68 | 0.00 | 1% | ok | 0.2x (3%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 3.053 ms | 1.10 | 4.41 | 13% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.052 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 148.069 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 77.892 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.445 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.756 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.480 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.503 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.945 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.854 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.707 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.559 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.492 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 68.520 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.166 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.856 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 45.308 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.449 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.264 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.090 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 4.134 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.041 ms | 0.86 | 9.64 | 29% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.746 ms | 0.70 | 7.83 | 23% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 8.026 ms | 0.33 | 3.65 | 11% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 42.186 ms | 0.91 | 6.89 | 21% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 350.933 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 357.065 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 192.759 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.805 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 12.510 ms | 1.34 | 5.36 | 16% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.949 ms | 20.73 | 0.32 | 32% | ok | 5.9x (98%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 4.571 ms | 3.67 | 3.90 | 12% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.621 ms | 2.98 | 11.94 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.694 ms | 6.23 | 12.45 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.677 ms | 2.96 | 2.96 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.758 ms | 9.54 | 9.54 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 8.857 ms | 30.31 | 0.36 | n/a | ok | 4.3x (72%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 7.064 ms | 38.00 | 0.30 | n/a | ok | 5.8x (97%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.475 ms | 49.03 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.473 ms | 49.05 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 10.298 ms | 26.07 | 0.15 | n/a | ok | 5.8x (97%) |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

