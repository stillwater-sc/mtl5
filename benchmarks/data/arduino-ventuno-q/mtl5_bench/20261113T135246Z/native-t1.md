## arduino-ventuno-q -- native, 1 thread, class medium

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
| bandwidth | read 22.5, update 24.5, triad 23.0 GB/s (ceiling 24.5) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 88% | 104% | 0 |
| blas2 | 10 | 60% | 71% | 0 |
| blas3 | 10 | 30% | 32% | 0 |
| factor | 12 | 12% | 32% | 0 |
| eigen | 8 | 15% | 22% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 29% | 67% | 0 |
| direct | 19 | 2% | 7% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 9% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 57% | 0 |
| mg | 1 | 36% | 36% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 34% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.666 ms | 2.96 | 23.69 | 97% | ok |
| blas1 | dot | f32 | n=8388608 | 4.310 ms | 3.89 | 15.57 | 64% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.903 ms | 2.12 | 25.47 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.969 ms | 4.23 | 25.36 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 5.401 ms | 1.55 | 24.85 | 101% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.106 ms | 5.40 | 21.61 | 88% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.574 ms | 0.98 | 7.83 | 32% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.520 ms | 0.48 | 3.83 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.167 ms | 1.03 | 8.22 | 34% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 196.604 ms | 0.94 | 0.68 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 7.902 ms | 4.25 | 16.99 | 69% | ok |
| blas2 | gemv | f32 | n=4096 | 5.745 ms | 5.84 | 11.69 | 48% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 10.032 ms | 3.34 | 13.39 | 55% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.492 ms | 7.47 | 14.95 | 61% | ok |
| blas2 | symv | f64 | n=4096 | 16.688 ms | 2.01 | 4.03 | 16% | ok |
| blas2 | trmv | f64 | n=4096 | 4.388 ms | 3.82 | 15.31 | 62% | ok |
| blas2 | trmv | f32 | n=4096 | 4.841 ms | 3.47 | 6.94 | 28% | ok |
| blas2 | trsv | f64 | n=4096 | 4.559 ms | 3.68 | 14.74 | 60% | ok |
| blas2 | trsv | f32 | n=4096 | 4.715 ms | 3.56 | 7.12 | 29% | ok |
| blas2 | ger | f64 | n=4096 | 15.333 ms | 2.19 | 17.51 | 71% | ok |
| blas3 | gemm | f64 | 512x512x512 | 52.859 ms | 5.08 | 0.12 | 30% | ok |
| blas3 | gemm | f32 | 512x512x512 | 26.181 ms | 10.25 | 0.12 | 31% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 43.506 ms | 3.09 | 0.14 | 18% | ok |
| blas3 | symm | f64 | 512x512x512 | 53.896 ms | 4.98 | 0.12 | 30% | ok |
| blas3 | syrk | f64 | 512x512x512 | 42.506 ms | 3.16 | 0.10 | 19% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 83.191 ms | 3.23 | 0.08 | 19% | ok |
| blas3 | trmm | f64 | 512x512x512 | 26.838 ms | 5.00 | 0.20 | 30% | ok |
| blas3 | trsm | f64 | 512x512x512 | 26.279 ms | 5.11 | 0.20 | 30% | ok |
| blas3 | trsm | f32 | 512x512x512 | 12.398 ms | 10.83 | 0.21 | 32% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 29.081 ms | 0.18 | 1.46 | 6% | ok |
| factor | lu | f64 | 512x512 | 25.143 ms | 3.56 | 0.17 | 21% | ok |
| factor | lu | f32 | 512x512 | 13.111 ms | 6.82 | 0.16 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 21.013 ms | 2.13 | 0.20 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 19.638 ms | 2.28 | 0.11 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 22.965 ms | 1.95 | 0.18 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 68.553 ms | 0.65 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 252.533 ms | 0.71 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 61.371 ms | 2.92 | 0.07 | 17% | ok |
| factor | hessenberg | f64 | 512x512 | 349.685 ms | 1.28 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.570 ms | 0.67 | 2.67 | 11% | ok |
| factor | inv | f64 | 512x512 | 148.144 ms | 1.81 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.867 ms | 5.30 | 0.25 | 32% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.872 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.172 ms | 1.55 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 591.344 ms | 0.26 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.350 ms | 2.49 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 150.146 ms | 2.79 | 0.01 | 17% | ok |
| eigen | svd | f64 | 256x256 | 230.174 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 67.415 ms | 3.73 | 0.02 | 22% | ok |
| eigen | sylvester | f64 | 256x256 | 367.716 ms | 2.74 | 0.00 | 16% | ok |
| elementwise | exp | f64 | n=8388608 | 88.534 ms | 0.09 | 1.52 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 111.132 ms | 0.08 | 1.21 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 150.815 ms | 0.06 | 0.89 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 71.566 ms | 0.12 | 1.88 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 117.890 ms | 0.07 | 1.14 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.668 ms | 1.57 | 16.32 | 67% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.143 ms | 1.22 | 12.71 | 52% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.279 ms | 3.41 | 7.17 | 29% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 12.789 ms | 0.25 | 0.90 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.421 ms | 0.24 | 1.98 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 186.080 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 65.414 ms | 0.17 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.462 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 103.116 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 139.212 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.131 ms | 0.52 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 205.238 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.496 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.575 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.701 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.956 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.886 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 173.979 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.844 ms | 0.41 | 1.64 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.752 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.384 ms | 0.98 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.777 ms | 0.43 | 1.70 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.449 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 87.775 ms | 0.25 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.738 ms | 0.13 | 0.53 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.957 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 117.776 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.747 ms | 0.13 | 0.53 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 21.510 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 16497.701 ms | 0.02 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.464 ms | 0.45 | 1.81 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 46.835 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 204.790 ms | 1.51 | 0.00 | 9% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.663 ms | 0.44 | 1.76 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.623 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1679.652 ms | 0.37 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.586 ms | 0.12 | 0.49 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 29.217 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2446.085 ms | 0.25 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.297 ms | 0.12 | 0.48 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.678 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 104.037 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 53.842 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 28.860 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.496 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 34.077 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 63.839 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 64.597 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.403 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 37.434 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.817 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.651 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 46.848 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.112 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.901 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 30.864 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.153 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 3.008 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.066 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.767 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.116 ms | 1.24 | 13.86 | 57% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.499 ms | 1.05 | 11.73 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.956 ms | 0.89 | 9.92 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.588 ms | 1.18 | 8.92 | 36% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 233.105 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 270.050 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 150.771 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.356 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 8.051 ms | 2.08 | 8.34 | 34% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 48.586 ms | 5.52 | 0.09 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.942 ms | 5.70 | 6.06 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.829 ms | 5.93 | 23.72 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.403 ms | 11.95 | 23.91 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.653 ms | 4.59 | 4.59 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.150 ms | 14.59 | 14.59 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 27.231 ms | 9.86 | 0.12 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 26.363 ms | 10.18 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 20.307 ms | 13.22 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 20.320 ms | 13.21 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 38.538 ms | 6.97 | 0.04 | n/a | ok |

