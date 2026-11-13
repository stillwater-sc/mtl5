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
| bandwidth | read 22.7, update 24.7, triad 23.2 GB/s (ceiling 24.7) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 83% | 104% | 0 |
| blas2 | 10 | 58% | 72% | 0 |
| blas3 | 10 | 29% | 32% | 0 |
| factor | 12 | 12% | 32% | 0 |
| eigen | 8 | 15% | 22% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 30% | 66% | 0 |
| direct | 19 | 2% | 7% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 8% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 57% | 0 |
| mg | 1 | 37% | 37% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 34% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.642 ms | 2.97 | 23.79 | 96% | ok |
| blas1 | dot | f32 | n=8388608 | 4.507 ms | 3.72 | 14.89 | 60% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.860 ms | 2.13 | 25.61 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.972 ms | 4.22 | 25.34 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 5.258 ms | 1.60 | 25.53 | 103% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.292 ms | 5.10 | 20.39 | 83% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.414 ms | 1.00 | 7.98 | 32% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.436 ms | 0.48 | 3.85 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.383 ms | 1.00 | 8.01 | 32% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 199.639 ms | 0.92 | 0.67 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 7.937 ms | 4.23 | 16.92 | 69% | ok |
| blas2 | gemv | f32 | n=4096 | 5.734 ms | 5.85 | 11.71 | 47% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 10.301 ms | 3.26 | 13.04 | 53% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.724 ms | 7.10 | 14.21 | 58% | ok |
| blas2 | symv | f64 | n=4096 | 16.971 ms | 1.98 | 3.96 | 16% | ok |
| blas2 | trmv | f64 | n=4096 | 4.536 ms | 3.70 | 14.81 | 60% | ok |
| blas2 | trmv | f32 | n=4096 | 4.770 ms | 3.52 | 7.04 | 29% | ok |
| blas2 | trsv | f64 | n=4096 | 4.445 ms | 3.77 | 15.11 | 61% | ok |
| blas2 | trsv | f32 | n=4096 | 4.778 ms | 3.51 | 7.03 | 28% | ok |
| blas2 | ger | f64 | n=4096 | 15.202 ms | 2.21 | 17.66 | 72% | ok |
| blas3 | gemm | f64 | 512x512x512 | 51.639 ms | 5.20 | 0.12 | 31% | ok |
| blas3 | gemm | f32 | 512x512x512 | 24.620 ms | 10.90 | 0.13 | 32% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 39.689 ms | 3.38 | 0.16 | 20% | ok |
| blas3 | symm | f64 | 512x512x512 | 55.004 ms | 4.88 | 0.11 | 29% | ok |
| blas3 | syrk | f64 | 512x512x512 | 41.668 ms | 3.22 | 0.10 | 19% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 80.014 ms | 3.35 | 0.08 | 20% | ok |
| blas3 | trmm | f64 | 512x512x512 | 26.720 ms | 5.02 | 0.20 | 30% | ok |
| blas3 | trsm | f64 | 512x512x512 | 26.791 ms | 5.01 | 0.20 | 30% | ok |
| blas3 | trsm | f32 | 512x512x512 | 14.229 ms | 9.43 | 0.18 | 28% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 27.421 ms | 0.19 | 1.55 | 6% | ok |
| factor | lu | f64 | 512x512 | 24.576 ms | 3.64 | 0.17 | 22% | ok |
| factor | lu | f32 | 512x512 | 12.967 ms | 6.90 | 0.16 | 21% | ok |
| factor | cholesky | f64 | 512x512 | 20.869 ms | 2.14 | 0.20 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 19.654 ms | 2.28 | 0.11 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 22.962 ms | 1.95 | 0.18 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 68.363 ms | 0.65 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 253.736 ms | 0.71 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 61.172 ms | 2.93 | 0.07 | 17% | ok |
| factor | hessenberg | f64 | 512x512 | 352.764 ms | 1.27 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.557 ms | 0.67 | 2.69 | 11% | ok |
| factor | inv | f64 | 512x512 | 145.325 ms | 1.85 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.850 ms | 5.31 | 0.25 | 32% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.797 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.167 ms | 1.55 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 593.830 ms | 0.25 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.324 ms | 2.49 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 149.873 ms | 2.80 | 0.01 | 17% | ok |
| eigen | svd | f64 | 256x256 | 230.848 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 67.386 ms | 3.73 | 0.02 | 22% | ok |
| eigen | sylvester | f64 | 256x256 | 368.562 ms | 2.73 | 0.00 | 16% | ok |
| elementwise | exp | f64 | n=8388608 | 89.881 ms | 0.09 | 1.49 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 106.227 ms | 0.08 | 1.26 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 144.138 ms | 0.06 | 0.93 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 70.317 ms | 0.12 | 1.91 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 115.918 ms | 0.07 | 1.16 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.675 ms | 1.56 | 16.26 | 66% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.177 ms | 1.20 | 12.51 | 51% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.997 ms | 3.49 | 7.34 | 30% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 12.802 ms | 0.25 | 0.90 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.427 ms | 0.24 | 1.98 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 182.605 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 65.159 ms | 0.17 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.313 ms | 0.43 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 102.173 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 138.209 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 42.919 ms | 0.52 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 201.001 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.788 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.625 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.708 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.774 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.958 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 174.323 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.854 ms | 0.41 | 1.63 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.836 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.410 ms | 0.98 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.753 ms | 0.43 | 1.73 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.395 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 87.516 ms | 0.26 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.730 ms | 0.13 | 0.53 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 10.015 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 118.572 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.810 ms | 0.13 | 0.52 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 21.611 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 16507.441 ms | 0.02 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.418 ms | 0.45 | 1.82 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 46.999 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 216.704 ms | 1.42 | 0.00 | 8% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.690 ms | 0.44 | 1.75 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.860 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1783.975 ms | 0.35 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.519 ms | 0.12 | 0.49 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 30.194 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2438.882 ms | 0.25 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.828 ms | 0.12 | 0.48 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.592 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 104.190 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 54.364 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 28.949 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.524 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 33.873 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 63.720 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.930 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.770 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 37.472 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.702 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.908 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 47.330 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.119 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.912 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 30.802 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.148 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 3.004 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.065 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.736 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.077 ms | 1.26 | 14.12 | 57% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.475 ms | 1.06 | 11.85 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.965 ms | 0.88 | 9.89 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.155 ms | 1.20 | 9.04 | 37% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 234.152 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 270.772 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 148.423 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.274 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 8.102 ms | 2.07 | 8.28 | 34% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 48.907 ms | 5.49 | 0.09 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.941 ms | 5.70 | 6.06 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.854 ms | 5.88 | 23.51 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.393 ms | 12.04 | 24.09 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.651 ms | 4.60 | 4.60 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.153 ms | 14.55 | 14.55 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 28.039 ms | 9.57 | 0.11 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 26.365 ms | 10.18 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 20.317 ms | 13.21 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 20.300 ms | 13.22 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 38.516 ms | 6.97 | 0.04 | n/a | ok |

