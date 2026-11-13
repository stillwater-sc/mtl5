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

