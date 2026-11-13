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

