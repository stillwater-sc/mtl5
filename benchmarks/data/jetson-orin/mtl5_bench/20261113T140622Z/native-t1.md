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
| bandwidth | read 11.5, update 16.2, triad 14.9 GB/s (ceiling 16.2) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 72% | 131% | 0 |
| blas2 | 10 | 51% | 123% | 0 |
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
| mixed | 3 | 33% | 33% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.588 ms | 1.45 | 11.58 | 72% | ok |
| blas1 | dot | f32 | n=8388608 | 6.591 ms | 2.55 | 10.18 | 63% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.245 ms | 1.37 | 16.44 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 5.957 ms | 2.82 | 16.90 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 6.350 ms | 1.32 | 21.14 | 131% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.632 ms | 2.98 | 11.92 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.495 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.747 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.498 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 192.519 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 13.071 ms | 2.57 | 10.27 | 63% | ok |
| blas2 | gemv | f32 | n=4096 | 7.718 ms | 4.35 | 8.70 | 54% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 16.327 ms | 2.06 | 8.22 | 51% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.401 ms | 5.24 | 10.49 | 65% | ok |
| blas2 | symv | f64 | n=4096 | 21.348 ms | 1.57 | 3.15 | 19% | ok |
| blas2 | trmv | f64 | n=4096 | 8.888 ms | 1.89 | 7.56 | 47% | ok |
| blas2 | trmv | f32 | n=4096 | 6.963 ms | 2.41 | 4.82 | 30% | ok |
| blas2 | trsv | f64 | n=4096 | 8.854 ms | 1.89 | 7.59 | 47% | ok |
| blas2 | trsv | f32 | n=4096 | 6.792 ms | 2.47 | 4.95 | 31% | ok |
| blas2 | ger | f64 | n=4096 | 13.434 ms | 2.50 | 19.99 | 123% | ok |
| blas3 | gemm | f64 | 512x512x512 | 75.321 ms | 3.56 | 0.08 | 33% | ok |
| blas3 | gemm | f32 | 512x512x512 | 38.152 ms | 7.04 | 0.08 | 33% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 48.436 ms | 2.77 | 0.13 | 26% | ok |
| blas3 | symm | f64 | 512x512x512 | 74.231 ms | 3.62 | 0.08 | 34% | ok |
| blas3 | syrk | f64 | 512x512x512 | 49.577 ms | 2.71 | 0.08 | 25% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 93.062 ms | 2.88 | 0.07 | 27% | ok |
| blas3 | trmm | f64 | 512x512x512 | 37.226 ms | 3.61 | 0.14 | 34% | ok |
| blas3 | trsm | f64 | 512x512x512 | 37.318 ms | 3.60 | 0.14 | 34% | ok |
| blas3 | trsm | f32 | 512x512x512 | 18.568 ms | 7.23 | 0.14 | 34% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.814 ms | 0.54 | 4.33 | 27% | ok |
| factor | lu | f64 | 512x512 | 35.983 ms | 2.49 | 0.12 | 23% | ok |
| factor | lu | f32 | 512x512 | 20.099 ms | 4.45 | 0.10 | 21% | ok |
| factor | cholesky | f64 | 512x512 | 31.977 ms | 1.40 | 0.13 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 30.042 ms | 1.49 | 0.07 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 34.594 ms | 1.29 | 0.12 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 85.064 ms | 0.53 | 0.05 | 5% | ok |
| factor | qr | f64 | 512x512 | 339.778 ms | 0.53 | 0.01 | 5% | ok |
| factor | lq | f64 | 512x512 | 93.093 ms | 1.92 | 0.05 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 475.890 ms | 0.94 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 1.946 ms | 0.54 | 2.16 | 13% | ok |
| factor | inv | f64 | 512x512 | 193.362 ms | 1.39 | 0.02 | 13% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 25.256 ms | 3.54 | 0.17 | 33% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.528 ms | 0.36 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 145.103 ms | 1.04 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 901.604 ms | 0.17 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 98.654 ms | 1.70 | 0.01 | 16% | ok |
| eigen | schur | f64 | 256x256 | 214.658 ms | 1.95 | 0.00 | 18% | ok |
| eigen | svd | f64 | 256x256 | 358.648 ms | 1.03 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 102.417 ms | 2.46 | 0.01 | 23% | ok |
| eigen | sylvester | f64 | 256x256 | 535.447 ms | 1.88 | 0.00 | 18% | ok |
| elementwise | exp | f64 | n=8388608 | 75.823 ms | 0.11 | 1.77 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.176 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 179.018 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.238 ms | 0.15 | 2.43 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 127.618 ms | 0.07 | 1.05 | 6% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.304 ms | 1.14 | 11.82 | 73% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.061 ms | 0.86 | 8.90 | 55% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.136 ms | 2.44 | 5.14 | 32% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.502 ms | 0.16 | 0.56 | 3% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.710 ms | 0.17 | 1.42 | 9% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 277.597 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 102.732 ms | 0.11 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.473 ms | 0.30 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 131.659 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 187.801 ms | 0.12 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 64.906 ms | 0.34 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 290.356 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.212 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.342 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.144 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 66.398 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 8.830 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 266.959 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.238 ms | 0.34 | 1.35 | 8% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 18.084 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.583 ms | 0.67 | 0.04 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.141 ms | 0.35 | 1.41 | 9% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.704 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 114.003 ms | 0.20 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.559 ms | 0.10 | 0.40 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 14.719 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 162.189 ms | 0.14 | 0.00 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.612 ms | 0.10 | 0.40 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 31.402 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 25417.714 ms | 0.01 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 10.172 ms | 0.33 | 1.32 | 8% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 65.495 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 280.815 ms | 1.10 | 0.00 | 10% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.158 ms | 0.33 | 1.33 | 8% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.303 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1926.882 ms | 0.32 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 37.010 ms | 0.09 | 0.36 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 43.250 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 3195.146 ms | 0.19 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 37.034 ms | 0.09 | 0.36 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.077 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 148.111 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.028 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.677 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.935 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.697 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.901 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.874 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 50.191 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 55.060 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.883 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.706 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 69.059 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.168 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.746 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 45.761 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.563 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.229 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.090 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.318 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.005 ms | 0.87 | 9.76 | 60% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.737 ms | 0.70 | 7.85 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.515 ms | 0.58 | 6.50 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 45.005 ms | 0.85 | 6.46 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 353.448 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 412.346 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 193.239 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 21.008 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 12.499 ms | 1.34 | 5.37 | 33% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 75.355 ms | 3.56 | 0.06 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 5.604 ms | 2.99 | 3.18 | 28% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.601 ms | 3.00 | 11.98 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.702 ms | 6.21 | 12.42 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.675 ms | 2.96 | 2.96 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.757 ms | 9.55 | 9.55 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 38.117 ms | 7.04 | 0.08 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 41.072 ms | 6.54 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 31.865 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 31.891 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 60.123 ms | 4.46 | 0.03 | n/a | ok |

