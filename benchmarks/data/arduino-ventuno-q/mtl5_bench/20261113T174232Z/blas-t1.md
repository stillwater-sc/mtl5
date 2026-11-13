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
| bandwidth | read 22.3, update 24.6, triad 23.0 GB/s (ceiling 24.6) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 89% | 104% | 0 |
| blas2 | 10 | 58% | 73% | 0 |
| blas3 | 10 | 83% | 93% | 0 |
| factor | 12 | 41% | 58% | 0 |
| eigen | 8 | 34% | 75% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 30% | 68% | 0 |
| direct | 23 | 2% | 11% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 15 | 7% | 42% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 40% | 56% | 0 |
| mg | 1 | 37% | 37% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 34% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.697 ms | 2.95 | 23.56 | 96% | ok |
| blas1 | dot | f32 | n=8388608 | 4.506 ms | 3.72 | 14.89 | 60% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.820 ms | 2.15 | 25.74 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.954 ms | 4.24 | 25.46 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 5.402 ms | 1.55 | 24.85 | 101% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.049 ms | 5.50 | 22.01 | 89% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.279 ms | 1.01 | 8.11 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.437 ms | 0.48 | 3.85 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.453 ms | 0.99 | 7.94 | 32% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 191.304 ms | 0.96 | 0.70 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 8.985 ms | 3.73 | 14.95 | 61% | ok |
| blas2 | gemv | f32 | n=4096 | 3.819 ms | 8.79 | 17.58 | 71% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 9.325 ms | 3.60 | 14.40 | 58% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.717 ms | 9.03 | 18.06 | 73% | ok |
| blas2 | symv | f64 | n=4096 | 11.531 ms | 2.91 | 5.83 | 24% | ok |
| blas2 | trmv | f64 | n=4096 | 6.296 ms | 2.66 | 10.67 | 43% | ok |
| blas2 | trmv | f32 | n=4096 | 3.262 ms | 5.14 | 10.30 | 42% | ok |
| blas2 | trsv | f64 | n=4096 | 6.375 ms | 2.63 | 10.54 | 43% | ok |
| blas2 | trsv | f32 | n=4096 | 3.155 ms | 5.32 | 10.65 | 43% | ok |
| blas2 | ger | f64 | n=4096 | 15.409 ms | 2.18 | 17.42 | 71% | ok |
| blas3 | gemm | f64 | 512x512x512 | 17.720 ms | 15.15 | 0.36 | 90% | ok |
| blas3 | gemm | f32 | 512x512x512 | 8.641 ms | 31.07 | 0.36 | 93% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 11.387 ms | 11.79 | 0.55 | 70% | ok |
| blas3 | symm | f64 | 512x512x512 | 17.933 ms | 14.97 | 0.35 | 89% | ok |
| blas3 | syrk | f64 | 512x512x512 | 9.857 ms | 13.62 | 0.43 | 81% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 19.194 ms | 13.99 | 0.33 | 83% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.059 ms | 14.82 | 0.58 | 88% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.710 ms | 13.82 | 0.54 | 82% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.006 ms | 26.81 | 0.52 | 80% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 26.407 ms | 0.20 | 1.61 | 7% | ok |
| factor | lu | f64 | 512x512 | 9.159 ms | 9.77 | 0.46 | 58% | ok |
| factor | lu | f32 | 512x512 | 5.506 ms | 16.25 | 0.38 | 48% | ok |
| factor | cholesky | f64 | 512x512 | 5.387 ms | 8.30 | 0.78 | 50% | ok |
| factor | cholesky | f32 | 512x512 | 3.216 ms | 13.91 | 0.65 | 41% | ok |
| factor | ldlt | f64 | 512x512 | 22.599 ms | 1.98 | 0.19 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 68.372 ms | 0.65 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 19.763 ms | 9.06 | 0.21 | 54% | ok |
| factor | lq | f64 | 512x512 | 60.637 ms | 2.95 | 0.07 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 349.518 ms | 1.28 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.550 ms | 0.68 | 2.71 | 11% | ok |
| factor | inv | f64 | 512x512 | 131.166 ms | 2.05 | 0.03 | 12% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 10.242 ms | 8.74 | 0.41 | 52% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 7.361 ms | 3.04 | 0.14 | 18% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.025 ms | 1.56 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 626.441 ms | 0.24 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 45.139 ms | 3.72 | 0.02 | 22% | ok |
| eigen | schur | f64 | 256x256 | 63.383 ms | 6.62 | 0.02 | 39% | ok |
| eigen | svd | f64 | 256x256 | 29.428 ms | 12.54 | 0.04 | 75% | ok |
| eigen | expm | f64 | 256x256 | 38.007 ms | 6.62 | 0.03 | 39% | ok |
| eigen | sylvester | f64 | 256x256 | 175.947 ms | 5.72 | 0.01 | 34% | ok |
| elementwise | exp | f64 | n=8388608 | 83.479 ms | 0.10 | 1.61 | 7% | ok |
| elementwise | log | f64 | n=8388608 | 107.659 ms | 0.08 | 1.25 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 150.444 ms | 0.06 | 0.89 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 68.978 ms | 0.12 | 1.95 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 112.370 ms | 0.07 | 1.19 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.637 ms | 1.60 | 16.63 | 68% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.161 ms | 1.21 | 12.60 | 51% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.998 ms | 3.49 | 7.34 | 30% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 12.756 ms | 0.26 | 0.90 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.806 ms | 0.23 | 1.89 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 181.317 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 57.470 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.357 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 104.893 ms | 0.21 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 142.881 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.859 ms | 0.51 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 201.371 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.793 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.690 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.700 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.531 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.909 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 174.088 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.847 ms | 0.41 | 1.64 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.747 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.453 ms | 0.98 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.728 ms | 0.44 | 1.75 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.431 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 89.806 ms | 0.25 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.928 ms | 0.13 | 0.51 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 10.010 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 120.746 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.968 ms | 0.13 | 0.51 | 2% | ok |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 16.746 ms | 0.67 | 0.04 | 4% | ok |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 3.892 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 11.253 ms | 0.99 | 0.06 | 6% | ok |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.161 ms | 0.65 | 2.60 | 11% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 21.428 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 16474.028 ms | 0.02 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.697 ms | 0.44 | 1.75 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 46.632 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 205.652 ms | 1.50 | 0.00 | 9% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.546 ms | 0.45 | 1.79 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.870 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1705.322 ms | 0.36 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.087 ms | 0.12 | 0.46 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 29.688 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2446.518 ms | 0.25 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.648 ms | 0.12 | 0.47 | 2% | ok |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 7.423 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 44.133 ms | 6.99 | 0.02 | 42% | ok |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 2.405 ms | 1.40 | 5.60 | 23% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.716 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 104.150 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 53.896 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 28.639 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.562 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 33.777 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 64.284 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.996 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.891 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 37.568 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.758 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.885 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 47.638 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.112 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 5.050 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 31.124 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.126 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 3.046 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.066 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.972 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.117 ms | 1.24 | 13.85 | 56% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 4.621 ms | 0.57 | 6.35 | 26% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.948 ms | 0.89 | 9.95 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 31.966 ms | 1.20 | 9.09 | 37% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 229.321 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 234.081 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 143.721 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.402 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 8.024 ms | 2.09 | 8.36 | 34% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 48.613 ms | 5.52 | 0.09 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.936 ms | 5.71 | 6.07 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.848 ms | 5.89 | 23.56 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.392 ms | 12.05 | 24.10 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.643 ms | 4.61 | 4.61 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.169 ms | 14.35 | 14.35 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 24.595 ms | 10.91 | 0.13 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 26.363 ms | 10.18 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 20.303 ms | 13.22 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 20.299 ms | 13.22 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 38.478 ms | 6.98 | 0.04 | n/a | ok |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

