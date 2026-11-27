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

