# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,4, CPUs 0,1,2,3, clock policy governor performance on the A78C policies; performance 902-2112 MHz on cpus 0,1; performance 940-2361 MHz on cpus 2,3, 2026-11-13T14:30:00Z.

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

## arduino-ventuno-q -- native, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 67.7 GFLOP/s, f32 135.6 GFLOP/s |
| ... hardware, NEON | f64 67.7, f32 135.6 GFLOP/s |
| ... this build, scalar | f64 33.7, f32 33.8 GFLOP/s |
| bandwidth | read 23.1, update 24.9, triad 22.6 GB/s (ceiling 24.9) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 97% | 107% | 0 | 27% |
| blas2 | 10 | 61% | 84% | 0 | 27% |
| blas3 | 10 | 8% | 17% | 0 | 28% |
| factor | 12 | 4% | 37% | 0 | 36% |
| eigen | 8 | 4% | 10% | 0 | 28% |
| elementwise | 5 | 5% | 8% | 0 | 28% |
| sparse | 5 | 55% | 92% | 0 | 27% |
| direct | 19 | 0% | 6% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 27% |
| direct3d | 12 | 2% | 7% | 0 | 28% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 28% |
| smoother | 3 | 52% | 60% | 0 | 28% |
| mg | 1 | 36% | 36% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 31% | 37% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 59% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.565 ms | 3.01 | 24.12 | 97% | ok | 1.0x (25%) |
| blas1 | dot | f32 | n=8388608 | 2.809 ms | 5.97 | 23.89 | 96% | ok | 1.5x (38%) |
| blas1 | axpy | f64 | n=8388608 | 8.271 ms | 2.03 | 24.34 | 98% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 4.080 ms | 4.11 | 24.67 | 99% | ok | 1.0x (24%) |
| blas1 | scale | f64 | n=8388608 | 5.030 ms | 1.67 | 26.69 | 107% | ok | 1.1x (27%) |
| blas1 | two_norm | f64 | n=8388608 | 2.630 ms | 6.38 | 25.52 | 103% | ok | 1.2x (30%) |
| blas1 | one_norm | f64 | n=8388608 | 7.223 ms | 1.16 | 9.29 | 37% | ok | 1.2x (30%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.369 ms | 0.55 | 4.37 | 18% | ok | 1.1x (29%) |
| blas1 | sum | f64 | n=8388608 | 7.533 ms | 1.11 | 8.91 | 36% | ok | 1.1x (27%) |
| blas1 | dot_k3 | f64 | n=8388608 | 181.040 ms | 1.02 | 0.74 | 3% | ok | 1.1x (27%) |
| blas2 | gemv | f64 | n=4096 | 6.787 ms | 4.94 | 19.79 | 80% | ok | 1.2x (29%) |
| blas2 | gemv | f32 | n=4096 | 3.210 ms | 10.45 | 20.92 | 84% | ok | 1.8x (45%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.770 ms | 4.32 | 17.28 | 70% | ok | 1.3x (32%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.855 ms | 5.73 | 11.47 | 46% | ok | 0.8x (19%) |
| blas2 | symv | f64 | n=4096 | 15.892 ms | 2.11 | 4.23 | 17% | ok | 1.1x (26%) |
| blas2 | trmv | f64 | n=4096 | 4.445 ms | 3.77 | 15.11 | 61% | ok | 1.0x (25%) |
| blas2 | trmv | f32 | n=4096 | 4.416 ms | 3.80 | 7.61 | 31% | ok | 1.1x (27%) |
| blas2 | trsv | f64 | n=4096 | 4.411 ms | 3.80 | 15.23 | 61% | ok | 1.0x (26%) |
| blas2 | trsv | f32 | n=4096 | 4.335 ms | 3.87 | 7.75 | 31% | ok | 1.1x (27%) |
| blas2 | ger | f64 | n=4096 | 15.145 ms | 2.22 | 17.73 | 71% | ok | 1.0x (25%) |
| blas3 | gemm | f64 | 512x512x512 | 27.012 ms | 9.94 | 0.23 | 15% | ok | 2.0x (49%) |
| blas3 | gemm | f32 | 512x512x512 | 11.520 ms | 23.30 | 0.27 | 17% | ok | 2.3x (57%) |
| blas3 | gemmt | f64 | 512x512x512 | 40.679 ms | 3.30 | 0.15 | 5% | ok | 1.1x (27%) |
| blas3 | symm | f64 | 512x512x512 | 49.202 ms | 5.46 | 0.13 | 8% | ok | 1.1x (27%) |
| blas3 | syrk | f64 | 512x512x512 | 42.684 ms | 3.14 | 0.10 | 5% | ok | 1.0x (25%) |
| blas3 | syr2k | f64 | 512x512x512 | 81.236 ms | 3.30 | 0.08 | 5% | ok | 1.0x (26%) |
| blas3 | trmm | f64 | 512x512x512 | 23.478 ms | 5.72 | 0.22 | 8% | ok | 1.1x (29%) |
| blas3 | trsm | f64 | 512x512x512 | 23.521 ms | 5.71 | 0.22 | 8% | ok | 1.1x (28%) |
| blas3 | trsm | f32 | 512x512x512 | 11.136 ms | 12.05 | 0.24 | 9% | ok | 1.1x (28%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 28.457 ms | 0.19 | 1.49 | 6% | ok | 1.0x (26%) |
| factor | lu | f64 | 512x512 | 17.276 ms | 5.18 | 0.24 | 8% | ok | 1.5x (36%) |
| factor | lu | f32 | 512x512 | 8.912 ms | 10.04 | 0.24 | 7% | ok | 1.5x (37%) |
| factor | cholesky | f64 | 512x512 | 18.702 ms | 2.39 | 0.22 | 4% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 17.589 ms | 2.54 | 0.12 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 20.304 ms | 2.20 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 61.480 ms | 0.73 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 148.163 ms | 1.21 | 0.03 | 2% | ok | 1.7x (43%) |
| factor | lq | f64 | 512x512 | 36.519 ms | 4.90 | 0.11 | 7% | ok | 1.7x (42%) |
| factor | hessenberg | f64 | 512x512 | 201.396 ms | 2.22 | 0.02 | 3% | ok | 1.7x (43%) |
| factor | householder | f64 | 512x512 | 0.457 ms | 2.30 | 9.18 | 37% | ok | 3.4x (86%) |
| factor | inv | f64 | 512x512 | 131.397 ms | 2.04 | 0.03 | 3% | ok | 1.1x (28%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.299 ms | 7.28 | 0.34 | 11% | ok | 1.4x (34%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.539 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.777 ms | 1.74 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 528.275 ms | 0.29 | 0.00 | 0% | ok | 1.1x (28%) |
| eigen | eigenvalue | f64 | 256x256 | 60.137 ms | 2.79 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 134.231 ms | 3.12 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.588 ms | 1.80 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 38.531 ms | 6.53 | 0.03 | 10% | ok | 1.7x (44%) |
| eigen | sylvester | f64 | 256x256 | 316.791 ms | 3.18 | 0.00 | 5% | ok | 1.2x (29%) |
| elementwise | exp | f64 | n=8388608 | 79.669 ms | 0.11 | 1.68 | 7% | ok | 1.1x (28%) |
| elementwise | log | f64 | n=8388608 | 98.335 ms | 0.09 | 1.36 | 5% | ok | 1.1x (28%) |
| elementwise | sin | f64 | n=8388608 | 134.940 ms | 0.06 | 0.99 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 63.682 ms | 0.13 | 2.11 | 8% | ok | 1.1x (28%) |
| elementwise | erf | f64 | n=8388608 | 106.245 ms | 0.08 | 1.26 | 5% | ok | 1.1x (28%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.197 ms | 2.19 | 22.75 | 92% | ok | 1.4x (35%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.990 ms | 1.32 | 13.68 | 55% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.098 ms | 6.87 | 14.44 | 58% | ok | 2.0x (50%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.882 ms | 0.27 | 0.97 | 4% | ok | 1.1x (27%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.657 ms | 0.23 | 1.92 | 8% | ok | 1.0x (24%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 163.169 ms | 0.07 | 0.00 | 0% | ok | 1.1x (29%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 58.413 ms | 0.19 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.165 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 92.778 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 124.585 ms | 0.18 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 38.850 ms | 0.58 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 183.898 ms | 0.01 | 0.00 | 0% | ok | 1.1x (28%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.878 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.476 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.631 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 39.799 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.334 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 155.664 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.206 ms | 0.34 | 1.37 | 6% | ok | 0.8x (21%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.559 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.310 ms | 1.08 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.165 ms | 0.35 | 1.40 | 6% | ok | 0.8x (21%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.960 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 78.249 ms | 0.29 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.555 ms | 0.14 | 0.54 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 8.992 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 107.309 ms | 0.21 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.542 ms | 0.14 | 0.55 | 2% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 18.980 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 14715.704 ms | 0.02 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.653 ms | 0.44 | 1.76 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 41.574 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 184.711 ms | 1.67 | 0.00 | 2% | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.945 ms | 0.42 | 1.70 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.404 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1583.313 ms | 0.39 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 26.629 ms | 0.13 | 0.51 | 2% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 26.002 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2229.607 ms | 0.28 | 0.00 | 0% | ok | 1.1x (27%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.340 ms | 0.12 | 0.49 | 2% | ok | 1.0x (26%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.151 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 96.538 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.466 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 25.928 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.227 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.501 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.783 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.906 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.549 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.822 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.377 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.662 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.501 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.101 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.497 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.603 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.886 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.730 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ssor | f64 | 128^2 grid | 0.065 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.584 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.965 ms | 1.33 | 14.93 | 60% | ok | 1.1x (27%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.264 ms | 1.16 | 12.95 | 52% | ok | 1.1x (28%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.661 ms | 0.98 | 11.02 | 44% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.416 ms | 1.19 | 8.97 | 36% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 208.498 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 241.457 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 135.467 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.854 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.223 ms | 2.32 | 9.29 | 37% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.702 ms | 21.13 | 0.33 | 31% | ok | 3.8x (96%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.636 ms | 6.37 | 6.76 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.818 ms | 5.95 | 23.82 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.375 ms | 12.20 | 24.40 | n/a | ok | 1.0x (26%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.262 ms | 5.14 | 5.14 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.044 ms | 16.08 | 16.08 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.491 ms | 23.36 | 0.27 | n/a | ok | 2.4x (59%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.682 ms | 40.17 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.141 ms | 52.21 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.140 ms | 52.22 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 9.773 ms | 27.47 | 0.16 | n/a | ok | 3.9x (99%) |

## arduino-ventuno-q -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON_WITHOUT_AES |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.6 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 16.8, f32 33.6 GFLOP/s |
| bandwidth | read 22.5, update 24.5, triad 23.0 GB/s (ceiling 24.5) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 95% | 104% | 0 |
| blas2 | 10 | 68% | 71% | 0 |
| blas3 | 10 | 45% | 58% | 0 |
| factor | 12 | 12% | 32% | 0 |
| eigen | 8 | 15% | 31% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 30% | 67% | 0 |
| direct | 19 | 2% | 7% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 9% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 40% | 57% | 0 |
| mg | 1 | 37% | 37% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 39% | 67% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.745 ms | 2.92 | 23.36 | 95% | ok |
| blas1 | dot | f32 | n=8388608 | 2.886 ms | 5.81 | 23.25 | 95% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.854 ms | 2.14 | 25.63 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.969 ms | 4.23 | 25.36 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 5.352 ms | 1.57 | 25.08 | 102% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.194 ms | 5.25 | 21.01 | 86% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.346 ms | 1.01 | 8.04 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.247 ms | 0.49 | 3.89 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.402 ms | 1.00 | 7.99 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 201.741 ms | 0.91 | 0.67 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 7.981 ms | 4.20 | 16.83 | 69% | ok |
| blas2 | gemv | f32 | n=4096 | 3.995 ms | 8.40 | 16.81 | 68% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.727 ms | 4.34 | 17.38 | 71% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.009 ms | 8.37 | 16.75 | 68% | ok |
| blas2 | symv | f64 | n=4096 | 17.149 ms | 1.96 | 3.92 | 16% | ok |
| blas2 | trmv | f64 | n=4096 | 5.001 ms | 3.35 | 13.43 | 55% | ok |
| blas2 | trmv | f32 | n=4096 | 2.626 ms | 6.39 | 12.79 | 52% | ok |
| blas2 | trsv | f64 | n=4096 | 4.785 ms | 3.51 | 14.04 | 57% | ok |
| blas2 | trsv | f32 | n=4096 | 2.640 ms | 6.35 | 12.72 | 52% | ok |
| blas2 | ger | f64 | n=4096 | 15.711 ms | 2.14 | 17.09 | 70% | ok |
| blas3 | gemm | f64 | 512x512x512 | 28.097 ms | 9.55 | 0.22 | 57% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.830 ms | 19.41 | 0.23 | 58% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 17.902 ms | 7.50 | 0.35 | 45% | ok |
| blas3 | symm | f64 | 512x512x512 | 29.071 ms | 9.23 | 0.22 | 55% | ok |
| blas3 | syrk | f64 | 512x512x512 | 40.878 ms | 3.28 | 0.10 | 20% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 82.410 ms | 3.26 | 0.08 | 19% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.909 ms | 7.49 | 0.29 | 45% | ok |
| blas3 | trsm | f64 | 512x512x512 | 18.002 ms | 7.46 | 0.29 | 44% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.678 ms | 15.47 | 0.30 | 46% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 30.579 ms | 0.17 | 1.39 | 6% | ok |
| factor | lu | f64 | 512x512 | 24.651 ms | 3.63 | 0.17 | 22% | ok |
| factor | lu | f32 | 512x512 | 14.369 ms | 6.23 | 0.15 | 19% | ok |
| factor | cholesky | f64 | 512x512 | 22.357 ms | 2.00 | 0.19 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 19.661 ms | 2.28 | 0.11 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 22.783 ms | 1.96 | 0.18 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 68.602 ms | 0.65 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 251.011 ms | 0.71 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 60.809 ms | 2.94 | 0.07 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 350.209 ms | 1.28 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.559 ms | 0.67 | 2.69 | 11% | ok |
| factor | inv | f64 | 512x512 | 146.684 ms | 1.83 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.711 ms | 5.35 | 0.25 | 32% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.947 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.062 ms | 1.56 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 601.567 ms | 0.25 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.094 ms | 2.50 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 150.044 ms | 2.80 | 0.01 | 17% | ok |
| eigen | svd | f64 | 256x256 | 230.816 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 47.768 ms | 5.27 | 0.02 | 31% | ok |
| eigen | sylvester | f64 | 256x256 | 356.972 ms | 2.82 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 88.870 ms | 0.09 | 1.51 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 107.038 ms | 0.08 | 1.25 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 147.995 ms | 0.06 | 0.91 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 71.034 ms | 0.12 | 1.89 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 117.656 ms | 0.07 | 1.14 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.652 ms | 1.58 | 16.48 | 67% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.187 ms | 1.20 | 12.45 | 51% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.088 ms | 3.46 | 7.28 | 30% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.181 ms | 0.25 | 0.87 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.498 ms | 0.24 | 1.96 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 185.667 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 61.311 ms | 0.18 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.458 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 102.692 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 139.760 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 42.975 ms | 0.52 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 208.197 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.748 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.253 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.696 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.656 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.884 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 178.049 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.880 ms | 0.40 | 1.61 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.960 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.627 ms | 0.96 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.744 ms | 0.43 | 1.73 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.517 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 86.350 ms | 0.26 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.726 ms | 0.13 | 0.53 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.990 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 121.169 ms | 0.18 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.756 ms | 0.13 | 0.53 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 21.793 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 16923.537 ms | 0.02 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.490 ms | 0.45 | 1.80 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 48.297 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 211.238 ms | 1.46 | 0.00 | 9% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.647 ms | 0.44 | 1.76 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.603 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1716.112 ms | 0.36 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.766 ms | 0.12 | 0.47 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 29.791 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2515.802 ms | 0.25 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.923 ms | 0.12 | 0.48 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.458 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 103.320 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 53.681 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 28.898 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.413 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 34.067 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 63.230 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.953 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.663 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 37.458 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.533 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.284 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 46.109 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.112 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.969 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 30.945 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.130 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.984 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.063 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.723 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.085 ms | 1.26 | 14.06 | 57% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 4.673 ms | 0.56 | 6.28 | 26% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.963 ms | 0.88 | 9.90 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.081 ms | 1.20 | 9.06 | 37% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 234.038 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 269.083 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 145.266 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.324 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.103 ms | 4.09 | 16.35 | 67% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 41.467 ms | 6.47 | 0.10 | 39% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.650 ms | 6.33 | 6.73 | 38% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.824 ms | 5.94 | 23.76 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.398 ms | 12.00 | 24.01 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.330 ms | 7.20 | 7.20 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.257 ms | 3.94 | 3.94 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.566 ms | 15.28 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 16.744 ms | 16.03 | 0.13 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 16.597 ms | 16.17 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 23.666 ms | 11.34 | 0.07 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 50.930 ms | 5.27 | 0.03 | n/a | ok |

## arduino-ventuno-q -- native-fast, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON_WITHOUT_AES |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 67.8 GFLOP/s, f32 135.0 GFLOP/s |
| ... hardware, NEON | f64 67.7, f32 135.0 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 67.8, f32 134.9 GFLOP/s |
| bandwidth | read 23.3, update 25.1, triad 23.0 GB/s (ceiling 25.1) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 96% | 104% | 0 | 27% |
| blas2 | 10 | 55% | 88% | 0 | 26% |
| blas3 | 10 | 27% | 56% | 0 | 60% |
| factor | 12 | 3% | 37% | 0 | 36% |
| eigen | 8 | 4% | 13% | 0 | 28% |
| elementwise | 5 | 5% | 8% | 0 | 27% |
| sparse | 5 | 56% | 90% | 0 | 28% |
| direct | 19 | 0% | 6% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 28% |
| direct3d | 12 | 2% | 7% | 0 | 27% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 44% | 59% | 0 | 28% |
| mg | 1 | 36% | 36% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 38% | 73% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 98% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.554 ms | 3.02 | 24.16 | 96% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.792 ms | 6.01 | 24.04 | 96% | ok | 1.0x (26%) |
| blas1 | axpy | f64 | n=8388608 | 8.038 ms | 2.09 | 25.05 | 100% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 3.990 ms | 4.20 | 25.23 | 100% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 5.151 ms | 1.63 | 26.06 | 104% | ok | 1.0x (26%) |
| blas1 | two_norm | f64 | n=8388608 | 2.627 ms | 6.39 | 25.55 | 102% | ok | 1.2x (30%) |
| blas1 | one_norm | f64 | n=8388608 | 7.401 ms | 1.13 | 9.07 | 36% | ok | 1.1x (28%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.974 ms | 0.53 | 4.20 | 17% | ok | 1.1x (27%) |
| blas1 | sum | f64 | n=8388608 | 7.465 ms | 1.12 | 8.99 | 36% | ok | 1.1x (28%) |
| blas1 | dot_k3 | f64 | n=8388608 | 177.963 ms | 1.04 | 0.75 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 6.738 ms | 4.98 | 19.93 | 79% | ok | 1.2x (30%) |
| blas2 | gemv | f32 | n=4096 | 3.039 ms | 11.04 | 22.09 | 88% | ok | 1.3x (33%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.681 ms | 4.37 | 17.48 | 70% | ok | 1.0x (25%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.932 ms | 6.80 | 13.61 | 54% | ok | 0.8x (20%) |
| blas2 | symv | f64 | n=4096 | 15.838 ms | 2.12 | 4.24 | 17% | ok | 1.1x (27%) |
| blas2 | trmv | f64 | n=4096 | 4.840 ms | 3.47 | 13.88 | 55% | ok | 1.0x (26%) |
| blas2 | trmv | f32 | n=4096 | 2.579 ms | 6.51 | 13.02 | 52% | ok | 1.0x (25%) |
| blas2 | trsv | f64 | n=4096 | 4.845 ms | 3.46 | 13.86 | 55% | ok | 1.0x (25%) |
| blas2 | trsv | f32 | n=4096 | 2.551 ms | 6.58 | 13.17 | 52% | ok | 1.0x (26%) |
| blas2 | ger | f64 | n=4096 | 15.026 ms | 2.23 | 17.87 | 71% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 7.429 ms | 36.14 | 0.85 | 53% | ok | 3.8x (95%) |
| blas3 | gemm | f32 | 512x512x512 | 3.575 ms | 75.09 | 0.88 | 56% | ok | 3.9x (97%) |
| blas3 | gemmt | f64 | 512x512x512 | 5.869 ms | 22.87 | 1.07 | 34% | ok | 3.1x (76%) |
| blas3 | symm | f64 | 512x512x512 | 8.116 ms | 33.08 | 0.78 | 49% | ok | 3.6x (90%) |
| blas3 | syrk | f64 | 512x512x512 | 36.927 ms | 3.63 | 0.11 | 5% | ok | 1.1x (28%) |
| blas3 | syr2k | f64 | 512x512x512 | 77.699 ms | 3.45 | 0.08 | 5% | ok | 1.1x (27%) |
| blas3 | trmm | f64 | 512x512x512 | 7.731 ms | 17.36 | 0.68 | 26% | ok | 2.3x (58%) |
| blas3 | trsm | f64 | 512x512x512 | 7.934 ms | 16.92 | 0.66 | 25% | ok | 2.3x (57%) |
| blas3 | trsm | f32 | 512x512x512 | 3.631 ms | 36.97 | 0.72 | 27% | ok | 2.4x (60%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 29.301 ms | 0.18 | 1.45 | 6% | ok | 1.0x (26%) |
| factor | lu | f64 | 512x512 | 17.247 ms | 5.19 | 0.24 | 8% | ok | 1.4x (36%) |
| factor | lu | f32 | 512x512 | 8.933 ms | 10.02 | 0.23 | 7% | ok | 1.6x (40%) |
| factor | cholesky | f64 | 512x512 | 19.926 ms | 2.25 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 17.611 ms | 2.54 | 0.12 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 20.281 ms | 2.21 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 60.877 ms | 0.73 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 147.322 ms | 1.21 | 0.03 | 2% | ok | 1.7x (43%) |
| factor | lq | f64 | 512x512 | 36.695 ms | 4.88 | 0.11 | 7% | ok | 1.7x (41%) |
| factor | hessenberg | f64 | 512x512 | 199.764 ms | 2.24 | 0.02 | 3% | ok | 1.8x (44%) |
| factor | householder | f64 | 512x512 | 0.450 ms | 2.33 | 9.32 | 37% | ok | 3.5x (87%) |
| factor | inv | f64 | 512x512 | 126.765 ms | 2.12 | 0.03 | 3% | ok | 1.2x (29%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.187 ms | 7.34 | 0.34 | 11% | ok | 1.4x (34%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.778 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.901 ms | 1.74 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 526.387 ms | 0.29 | 0.00 | 0% | ok | 1.1x (29%) |
| eigen | eigenvalue | f64 | 256x256 | 60.205 ms | 2.79 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 134.676 ms | 3.11 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.826 ms | 1.79 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 29.163 ms | 8.63 | 0.04 | 13% | ok | 1.6x (41%) |
| eigen | sylvester | f64 | 256x256 | 312.718 ms | 3.22 | 0.01 | 5% | ok | 1.1x (29%) |
| elementwise | exp | f64 | n=8388608 | 80.963 ms | 0.10 | 1.66 | 7% | ok | 1.1x (27%) |
| elementwise | log | f64 | n=8388608 | 99.473 ms | 0.08 | 1.35 | 5% | ok | 1.1x (27%) |
| elementwise | sin | f64 | n=8388608 | 135.627 ms | 0.06 | 0.99 | 4% | ok | 1.1x (27%) |
| elementwise | sqrt | f64 | n=8388608 | 65.678 ms | 0.13 | 2.04 | 8% | ok | 1.1x (27%) |
| elementwise | erf | f64 | n=8388608 | 106.526 ms | 0.08 | 1.26 | 5% | ok | 1.1x (28%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.200 ms | 2.18 | 22.70 | 90% | ok | 1.4x (34%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.920 ms | 1.36 | 14.18 | 56% | ok | 1.1x (28%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.090 ms | 6.88 | 14.46 | 58% | ok | 2.0x (50%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 12.100 ms | 0.27 | 0.95 | 4% | ok | 1.1x (27%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.640 ms | 0.23 | 1.93 | 8% | ok | 1.0x (25%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 168.079 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 55.332 ms | 0.20 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.449 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 94.008 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 129.625 ms | 0.17 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 38.952 ms | 0.57 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 189.308 ms | 0.01 | 0.00 | 0% | ok | 1.1x (27%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.698 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.354 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.629 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.602 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.330 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 160.509 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.999 ms | 0.38 | 1.51 | 6% | ok | 0.9x (24%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.683 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.358 ms | 1.08 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.947 ms | 0.39 | 1.55 | 6% | ok | 0.9x (22%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.977 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 79.556 ms | 0.28 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.594 ms | 0.14 | 0.54 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.020 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 108.201 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.533 ms | 0.14 | 0.55 | 2% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 19.286 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 15064.515 ms | 0.02 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.676 ms | 0.44 | 1.76 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 42.308 ms | 0.00 | 0.00 | n/a | ok | 1.1x (29%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 185.502 ms | 1.66 | 0.00 | 2% | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.971 ms | 0.42 | 1.69 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.557 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1589.092 ms | 0.39 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.003 ms | 0.12 | 0.46 | 2% | ok | 1.0x (25%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 26.319 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2346.841 ms | 0.26 | 0.00 | 0% | ok | 1.1x (27%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.998 ms | 0.12 | 0.48 | 2% | ok | 1.0x (25%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 63.813 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 95.793 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.077 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 25.980 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.281 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.604 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.069 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.335 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.248 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.611 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.156 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.336 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 41.897 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.103 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.559 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.436 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.874 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.742 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.600 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.975 ms | 1.33 | 14.85 | 59% | ok | 1.1x (26%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 4.067 ms | 0.64 | 7.21 | 29% | ok | 1.1x (29%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.681 ms | 0.98 | 10.94 | 44% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 31.839 ms | 1.21 | 9.13 | 36% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 208.066 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 240.396 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 134.712 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.946 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 3.676 ms | 4.56 | 18.26 | 73% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 10.528 ms | 25.50 | 0.40 | 38% | ok | 3.9x (98%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.368 ms | 7.08 | 7.53 | 30% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.786 ms | 6.02 | 24.09 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.374 ms | 12.21 | 24.41 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.122 ms | 7.91 | 7.91 | n/a | ok | 1.1x (27%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 3.810 ms | 4.40 | 4.40 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 4.482 ms | 59.89 | 0.70 | n/a | ok | 3.9x (98%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.251 ms | 63.14 | 0.49 | n/a | ok | 3.9x (98%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.194 ms | 64.01 | 0.38 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.959 ms | 45.04 | 0.26 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 12.721 ms | 21.10 | 0.12 | n/a | ok | 4.0x (100%) |

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

## arduino-ventuno-q -- blas, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 67.4 GFLOP/s, f32 135.0 GFLOP/s |
| ... hardware, NEON | f64 67.4, f32 135.0 GFLOP/s |
| ... this build, scalar | f64 33.7, f32 33.7 GFLOP/s |
| bandwidth | read 22.9, update 24.7, triad 22.7 GB/s (ceiling 24.7) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 98% | 113% | 0 | 28% |
| blas2 | 10 | 63% | 106% | 0 | 36% |
| blas3 | 10 | 75% | 81% | 0 | 88% |
| factor | 12 | 18% | 39% | 0 | 44% |
| eigen | 8 | 11% | 28% | 0 | 33% |
| elementwise | 5 | 5% | 8% | 0 | 28% |
| sparse | 5 | 56% | 91% | 0 | 28% |
| direct | 23 | 1% | 11% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 28% |
| direct3d | 15 | 2% | 24% | 0 | 28% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 44% | 59% | 0 | 28% |
| mg | 1 | 36% | 36% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 31% | 38% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 53% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.541 ms | 3.03 | 24.22 | 98% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.802 ms | 5.99 | 23.95 | 97% | ok | 1.5x (39%) |
| blas1 | axpy | f64 | n=8388608 | 7.953 ms | 2.11 | 25.31 | 102% | ok | 1.0x (25%) |
| blas1 | axpy | f32 | n=8388608 | 3.985 ms | 4.21 | 25.26 | 102% | ok | 1.0x (26%) |
| blas1 | scale | f64 | n=8388608 | 4.798 ms | 1.75 | 27.97 | 113% | ok | 1.1x (29%) |
| blas1 | two_norm | f64 | n=8388608 | 2.624 ms | 6.39 | 25.58 | 103% | ok | 1.2x (30%) |
| blas1 | one_norm | f64 | n=8388608 | 7.777 ms | 1.08 | 8.63 | 35% | ok | 1.0x (26%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.485 ms | 0.54 | 4.33 | 18% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.287 ms | 1.15 | 9.21 | 37% | ok | 1.2x (29%) |
| blas1 | dot_k3 | f64 | n=8388608 | 175.199 ms | 1.05 | 0.77 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 5.136 ms | 6.53 | 26.15 | 106% | ok | 1.8x (44%) |
| blas2 | gemv | f32 | n=4096 | 2.603 ms | 12.89 | 25.79 | 104% | ok | 1.5x (36%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 8.596 ms | 3.90 | 15.62 | 63% | ok | 1.1x (27%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.825 ms | 6.95 | 13.91 | 56% | ok | 0.8x (19%) |
| blas2 | symv | f64 | n=4096 | 6.401 ms | 5.24 | 10.49 | 42% | ok | 1.8x (46%) |
| blas2 | trmv | f64 | n=4096 | 3.656 ms | 4.59 | 18.37 | 74% | ok | 1.7x (43%) |
| blas2 | trmv | f32 | n=4096 | 2.164 ms | 7.75 | 15.52 | 63% | ok | 1.5x (38%) |
| blas2 | trsv | f64 | n=4096 | 5.997 ms | 2.80 | 11.20 | 45% | ok | 1.1x (27%) |
| blas2 | trsv | f32 | n=4096 | 3.052 ms | 5.50 | 11.00 | 45% | ok | 1.1x (27%) |
| blas2 | ger | f64 | n=4096 | 14.909 ms | 2.25 | 18.01 | 73% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 4.986 ms | 53.84 | 1.26 | 80% | ok | 3.6x (90%) |
| blas3 | gemm | f32 | 512x512x512 | 2.444 ms | 109.85 | 1.29 | 81% | ok | 3.5x (88%) |
| blas3 | gemmt | f64 | 512x512x512 | 3.352 ms | 40.04 | 1.88 | 59% | ok | 3.4x (84%) |
| blas3 | symm | f64 | 512x512x512 | 5.031 ms | 53.36 | 1.25 | 79% | ok | 3.6x (90%) |
| blas3 | syrk | f64 | 512x512x512 | 3.245 ms | 41.36 | 1.29 | 61% | ok | 3.0x (76%) |
| blas3 | syr2k | f64 | 512x512x512 | 5.685 ms | 47.22 | 1.11 | 70% | ok | 3.4x (85%) |
| blas3 | trmm | f64 | 512x512x512 | 2.501 ms | 53.67 | 2.10 | 80% | ok | 3.6x (91%) |
| blas3 | trsm | f64 | 512x512x512 | 2.655 ms | 50.55 | 1.97 | 75% | ok | 3.6x (91%) |
| blas3 | trsm | f32 | 512x512x512 | 2.812 ms | 47.73 | 0.93 | 35% | ok | 1.8x (45%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 28.344 ms | 0.19 | 1.50 | 6% | ok | 1.0x (26%) |
| factor | lu | f64 | 512x512 | 4.095 ms | 21.85 | 1.02 | 32% | ok | 2.2x (56%) |
| factor | lu | f32 | 512x512 | 3.119 ms | 28.69 | 0.67 | 21% | ok | 1.8x (44%) |
| factor | cholesky | f64 | 512x512 | 2.970 ms | 15.06 | 1.41 | 22% | ok | 1.8x (45%) |
| factor | cholesky | f32 | 512x512 | 1.996 ms | 22.42 | 1.05 | 17% | ok | 1.6x (41%) |
| factor | ldlt | f64 | 512x512 | 20.291 ms | 2.20 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 61.396 ms | 0.73 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 9.650 ms | 18.55 | 0.43 | 28% | ok | 2.0x (51%) |
| factor | lq | f64 | 512x512 | 36.887 ms | 4.85 | 0.11 | 7% | ok | 1.7x (41%) |
| factor | hessenberg | f64 | 512x512 | 201.530 ms | 2.22 | 0.02 | 3% | ok | 1.7x (44%) |
| factor | householder | f64 | 512x512 | 0.434 ms | 2.42 | 9.67 | 39% | ok | 3.6x (90%) |
| factor | inv | f64 | 512x512 | 113.836 ms | 2.36 | 0.04 | 3% | ok | 1.1x (28%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.342 ms | 12.19 | 0.57 | 18% | ok | 1.4x (35%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 5.389 ms | 4.15 | 0.19 | 6% | ok | 1.4x (34%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.805 ms | 1.74 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 528.715 ms | 0.29 | 0.00 | 0% | ok | 1.2x (29%) |
| eigen | eigenvalue | f64 | 256x256 | 37.466 ms | 4.48 | 0.03 | 7% | ok | 1.2x (30%) |
| eigen | schur | f64 | 256x256 | 47.821 ms | 8.77 | 0.02 | 13% | ok | 1.3x (33%) |
| eigen | svd | f64 | 256x256 | 19.382 ms | 19.04 | 0.05 | 28% | ok | 1.5x (38%) |
| eigen | expm | f64 | 256x256 | 23.579 ms | 10.67 | 0.04 | 16% | ok | 1.6x (40%) |
| eigen | sylvester | f64 | 256x256 | 136.028 ms | 7.40 | 0.01 | 11% | ok | 1.3x (32%) |
| elementwise | exp | f64 | n=8388608 | 80.671 ms | 0.10 | 1.66 | 7% | ok | 1.1x (27%) |
| elementwise | log | f64 | n=8388608 | 99.998 ms | 0.08 | 1.34 | 5% | ok | 1.1x (28%) |
| elementwise | sin | f64 | n=8388608 | 132.528 ms | 0.06 | 1.01 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 64.929 ms | 0.13 | 2.07 | 8% | ok | 1.1x (28%) |
| elementwise | erf | f64 | n=8388608 | 106.420 ms | 0.08 | 1.26 | 5% | ok | 1.1x (27%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.204 ms | 2.17 | 22.61 | 91% | ok | 1.4x (35%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.961 ms | 1.33 | 13.89 | 56% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.155 ms | 6.80 | 14.31 | 58% | ok | 2.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.663 ms | 0.28 | 0.99 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.969 ms | 0.22 | 1.86 | 8% | ok | 1.0x (25%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 162.886 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 51.975 ms | 0.22 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.354 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 94.418 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 129.128 ms | 0.17 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.923 ms | 0.56 | 0.02 | 1% | ok | 1.1x (27%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 184.129 ms | 0.01 | 0.00 | 0% | ok | 1.1x (27%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.744 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.500 ms | 0.00 | 0.00 | n/a | ok | 1.1x (29%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.627 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.195 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.311 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 155.379 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.017 ms | 0.37 | 1.50 | 6% | ok | 0.9x (22%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.821 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.350 ms | 1.08 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.996 ms | 0.38 | 1.52 | 6% | ok | 0.9x (21%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.965 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 79.637 ms | 0.28 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.685 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.053 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 110.179 ms | 0.20 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.711 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 14.798 ms | 0.76 | 0.05 | 1% | ok | 1.1x (28%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 3.495 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 9.970 ms | 1.12 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.096 ms | 0.69 | 2.76 | 11% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 18.950 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 14664.925 ms | 0.02 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 8.023 ms | 0.42 | 1.68 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 41.749 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 184.717 ms | 1.67 | 0.00 | 2% | ok | 1.1x (29%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.825 ms | 0.43 | 1.72 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.632 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1477.609 ms | 0.42 | 0.00 | 1% | ok | 1.2x (29%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.922 ms | 0.12 | 0.48 | 2% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 26.100 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2257.167 ms | 0.27 | 0.00 | 0% | ok | 1.1x (26%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.119 ms | 0.12 | 0.48 | 2% | ok | 1.0x (25%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 6.655 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 27.722 ms | 11.13 | 0.03 | 17% | ok | 1.6x (40%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 2.274 ms | 1.48 | 5.93 | 24% | ok | 1.1x (27%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.185 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 96.293 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.518 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.185 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.308 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.602 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.845 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.936 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.387 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.796 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.618 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.747 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.214 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.102 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.641 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilut | f64 | 128^2 grid | 27.727 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.867 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.781 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.065 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.732 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.018 ms | 1.30 | 14.53 | 59% | ok | 1.0x (26%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 4.253 ms | 0.62 | 6.90 | 28% | ok | 1.1x (28%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.667 ms | 0.98 | 11.00 | 44% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.373 ms | 1.19 | 8.98 | 36% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 207.642 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 212.611 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 137.968 ms | 0.00 | 0.00 | n/a | ok | 1.1x (26%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.928 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.196 ms | 2.33 | 9.33 | 38% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.850 ms | 20.89 | 0.33 | 31% | ok | 3.8x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.637 ms | 6.36 | 6.76 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.804 ms | 5.98 | 23.93 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.378 ms | 12.17 | 24.35 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.274 ms | 5.12 | 5.12 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.039 ms | 16.15 | 16.15 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.622 ms | 23.10 | 0.27 | n/a | ok | 2.1x (53%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.687 ms | 40.14 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.136 ms | 52.26 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.136 ms | 52.27 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 9.736 ms | 27.57 | 0.16 | n/a | ok | 3.9x (99%) |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

