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

