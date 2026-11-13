## arduino-ventuno-q -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.6 GFLOP/s |
| ... this build, scalar | f64 8.4, f32 8.4 GFLOP/s |
| bandwidth | read 21.9, update 24.5, triad 22.9 GB/s (ceiling 24.5) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 86% | 105% | 0 |
| blas2 | 10 | 58% | 73% | 0 |
| blas3 | 10 | 83% | 93% | 0 |
| factor | 12 | 41% | 58% | 0 |
| eigen | 8 | 34% | 75% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 30% | 66% | 0 |
| direct | 23 | 2% | 11% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 15 | 7% | 41% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 40% | 57% | 0 |
| mg | 1 | 37% | 37% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 34% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.657 ms | 2.97 | 23.72 | 97% | ok |
| blas1 | dot | f32 | n=8388608 | 4.325 ms | 3.88 | 15.52 | 63% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.843 ms | 2.14 | 25.67 | 105% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.084 ms | 4.11 | 24.65 | 101% | ok |
| blas1 | scale | f64 | n=8388608 | 5.505 ms | 1.52 | 24.38 | 100% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.178 ms | 5.28 | 21.11 | 86% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.036 ms | 1.04 | 8.35 | 34% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.256 ms | 0.49 | 3.89 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.402 ms | 1.00 | 7.99 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 196.714 ms | 0.94 | 0.68 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 9.077 ms | 3.70 | 14.79 | 60% | ok |
| blas2 | gemv | f32 | n=4096 | 3.796 ms | 8.84 | 17.69 | 72% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 9.420 ms | 3.56 | 14.25 | 58% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.731 ms | 8.99 | 17.99 | 73% | ok |
| blas2 | symv | f64 | n=4096 | 11.813 ms | 2.84 | 5.69 | 23% | ok |
| blas2 | trmv | f64 | n=4096 | 6.309 ms | 2.66 | 10.65 | 43% | ok |
| blas2 | trmv | f32 | n=4096 | 3.316 ms | 5.06 | 10.13 | 41% | ok |
| blas2 | trsv | f64 | n=4096 | 6.415 ms | 2.62 | 10.47 | 43% | ok |
| blas2 | trsv | f32 | n=4096 | 3.275 ms | 5.12 | 10.26 | 42% | ok |
| blas2 | ger | f64 | n=4096 | 15.434 ms | 2.17 | 17.40 | 71% | ok |
| blas3 | gemm | f64 | 512x512x512 | 17.947 ms | 14.96 | 0.35 | 89% | ok |
| blas3 | gemm | f32 | 512x512x512 | 8.634 ms | 31.09 | 0.36 | 93% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 11.317 ms | 11.86 | 0.56 | 71% | ok |
| blas3 | symm | f64 | 512x512x512 | 18.073 ms | 14.85 | 0.35 | 88% | ok |
| blas3 | syrk | f64 | 512x512x512 | 9.887 ms | 13.58 | 0.42 | 81% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 19.243 ms | 13.95 | 0.33 | 83% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.065 ms | 14.81 | 0.58 | 88% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.652 ms | 13.91 | 0.54 | 83% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.009 ms | 26.80 | 0.52 | 80% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 29.549 ms | 0.18 | 1.44 | 6% | ok |
| factor | lu | f64 | 512x512 | 9.162 ms | 9.77 | 0.46 | 58% | ok |
| factor | lu | f32 | 512x512 | 5.530 ms | 16.18 | 0.38 | 48% | ok |
| factor | cholesky | f64 | 512x512 | 5.379 ms | 8.32 | 0.78 | 49% | ok |
| factor | cholesky | f32 | 512x512 | 3.262 ms | 13.71 | 0.64 | 41% | ok |
| factor | ldlt | f64 | 512x512 | 22.905 ms | 1.95 | 0.18 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 68.922 ms | 0.65 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 19.749 ms | 9.06 | 0.21 | 54% | ok |
| factor | lq | f64 | 512x512 | 60.914 ms | 2.94 | 0.07 | 17% | ok |
| factor | hessenberg | f64 | 512x512 | 350.914 ms | 1.27 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.558 ms | 0.67 | 2.69 | 11% | ok |
| factor | inv | f64 | 512x512 | 128.510 ms | 2.09 | 0.03 | 12% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 10.259 ms | 8.72 | 0.41 | 52% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 7.374 ms | 3.03 | 0.14 | 18% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.456 ms | 1.55 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 608.274 ms | 0.25 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 45.001 ms | 3.73 | 0.02 | 22% | ok |
| eigen | schur | f64 | 256x256 | 63.188 ms | 6.64 | 0.02 | 39% | ok |
| eigen | svd | f64 | 256x256 | 29.429 ms | 12.54 | 0.04 | 75% | ok |
| eigen | expm | f64 | 256x256 | 37.806 ms | 6.66 | 0.03 | 40% | ok |
| eigen | sylvester | f64 | 256x256 | 175.665 ms | 5.73 | 0.01 | 34% | ok |
| elementwise | exp | f64 | n=8388608 | 88.445 ms | 0.09 | 1.52 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 110.218 ms | 0.08 | 1.22 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 147.300 ms | 0.06 | 0.91 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 71.481 ms | 0.12 | 1.88 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 116.986 ms | 0.07 | 1.15 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.690 ms | 1.55 | 16.12 | 66% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.134 ms | 1.23 | 12.76 | 52% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.060 ms | 3.47 | 7.30 | 30% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 12.860 ms | 0.25 | 0.89 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.793 ms | 0.23 | 1.89 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 182.329 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 57.662 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.374 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 104.421 ms | 0.21 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 142.047 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.899 ms | 0.51 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 200.078 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.807 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 14.308 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.698 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.176 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.872 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 173.843 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.793 ms | 0.42 | 1.69 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.706 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.412 ms | 0.98 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.714 ms | 0.44 | 1.76 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.610 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 89.173 ms | 0.25 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.886 ms | 0.13 | 0.51 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.974 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 121.692 ms | 0.18 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.865 ms | 0.13 | 0.52 | 2% | ok |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 16.844 ms | 0.66 | 0.04 | 4% | ok |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 3.891 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 11.274 ms | 0.99 | 0.06 | 6% | ok |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.142 ms | 0.66 | 2.65 | 11% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 21.444 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 16477.650 ms | 0.02 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.770 ms | 0.43 | 1.73 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 46.516 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 211.890 ms | 1.46 | 0.00 | 9% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.523 ms | 0.45 | 1.79 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.553 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1710.962 ms | 0.36 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.114 ms | 0.12 | 0.46 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 29.255 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2391.842 ms | 0.26 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.561 ms | 0.12 | 0.47 | 2% | ok |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 7.543 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 44.487 ms | 6.94 | 0.02 | 41% | ok |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 2.410 ms | 1.40 | 5.59 | 23% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.704 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 104.334 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 53.482 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 29.105 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.521 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 33.941 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 64.219 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.866 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.857 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 37.390 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.818 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 40.332 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 47.554 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.112 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 5.154 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 30.996 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.138 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 3.042 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.067 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.960 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.106 ms | 1.24 | 13.93 | 57% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 4.712 ms | 0.56 | 6.22 | 25% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.966 ms | 0.88 | 9.89 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.011 ms | 1.20 | 9.08 | 37% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 232.761 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 237.950 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 146.022 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.320 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 8.119 ms | 2.07 | 8.27 | 34% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 48.685 ms | 5.51 | 0.09 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.949 ms | 5.69 | 6.05 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.836 ms | 5.92 | 23.67 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.388 ms | 12.09 | 24.17 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.649 ms | 4.60 | 4.60 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.162 ms | 14.44 | 14.44 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 24.713 ms | 10.86 | 0.13 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 26.421 ms | 10.16 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 20.319 ms | 13.21 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 20.326 ms | 13.21 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 38.424 ms | 6.99 | 0.04 | n/a | ok |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

