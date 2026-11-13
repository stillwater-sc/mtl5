# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,4, CPUs 0,1,2,3, clock policy governor performance on the A78C policies; performance 902-2112 MHz on cpus 0,1; performance 940-2361 MHz on cpus 2,3, 2026-11-13T13:52:41Z.

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

## arduino-ventuno-q -- native, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 67.4 GFLOP/s, f32 135.5 GFLOP/s |
| ... hardware, NEON | f64 67.4, f32 135.5 GFLOP/s |
| ... this build, scalar | f64 25.9, f32 33.9 GFLOP/s |
| bandwidth | read 23.0, update 24.8, triad 22.5 GB/s (ceiling 24.8) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 97% | 114% | 0 | 28% |
| blas2 | 10 | 61% | 90% | 0 | 27% |
| blas3 | 10 | 8% | 17% | 0 | 28% |
| factor | 12 | 4% | 38% | 0 | 36% |
| eigen | 8 | 4% | 10% | 0 | 28% |
| elementwise | 5 | 5% | 8% | 0 | 28% |
| sparse | 5 | 55% | 92% | 0 | 27% |
| direct | 19 | 0% | 6% | 0 | 27% |
| ordering | 4 | n/a | n/a | 0 | 28% |
| direct3d | 12 | 2% | 7% | 0 | 28% |
| krylov | 13 | n/a | n/a | 0 | 27% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 52% | 60% | 0 | 27% |
| mg | 1 | 36% | 36% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 32% | 38% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 60% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.579 ms | 3.01 | 24.06 | 97% | ok | 1.0x (25%) |
| blas1 | dot | f32 | n=8388608 | 2.806 ms | 5.98 | 23.92 | 96% | ok | 1.6x (40%) |
| blas1 | axpy | f64 | n=8388608 | 8.023 ms | 2.09 | 25.09 | 101% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 3.944 ms | 4.25 | 25.52 | 103% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 4.744 ms | 1.77 | 28.29 | 114% | ok | 1.1x (28%) |
| blas1 | two_norm | f64 | n=8388608 | 2.627 ms | 6.39 | 25.54 | 103% | ok | 1.3x (31%) |
| blas1 | one_norm | f64 | n=8388608 | 7.299 ms | 1.15 | 9.19 | 37% | ok | 1.2x (29%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.473 ms | 0.54 | 4.34 | 17% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.804 ms | 1.07 | 8.60 | 35% | ok | 1.1x (27%) |
| blas1 | dot_k3 | f64 | n=8388608 | 180.731 ms | 1.02 | 0.74 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 6.744 ms | 4.98 | 19.91 | 80% | ok | 1.2x (29%) |
| blas2 | gemv | f32 | n=4096 | 3.000 ms | 11.19 | 22.38 | 90% | ok | 1.9x (48%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.733 ms | 4.34 | 17.37 | 70% | ok | 1.3x (33%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.652 ms | 5.94 | 11.88 | 48% | ok | 0.8x (21%) |
| blas2 | symv | f64 | n=4096 | 15.999 ms | 2.10 | 4.20 | 17% | ok | 1.1x (27%) |
| blas2 | trmv | f64 | n=4096 | 4.418 ms | 3.80 | 15.20 | 61% | ok | 1.0x (26%) |
| blas2 | trmv | f32 | n=4096 | 4.447 ms | 3.77 | 7.55 | 30% | ok | 1.1x (27%) |
| blas2 | trsv | f64 | n=4096 | 4.420 ms | 3.80 | 15.20 | 61% | ok | 1.0x (25%) |
| blas2 | trsv | f32 | n=4096 | 4.333 ms | 3.87 | 7.75 | 31% | ok | 1.1x (28%) |
| blas2 | ger | f64 | n=4096 | 14.995 ms | 2.24 | 17.91 | 72% | ok | 1.0x (25%) |
| blas3 | gemm | f64 | 512x512x512 | 26.781 ms | 10.02 | 0.23 | 15% | ok | 1.9x (48%) |
| blas3 | gemm | f32 | 512x512x512 | 11.715 ms | 22.91 | 0.27 | 17% | ok | 2.1x (53%) |
| blas3 | gemmt | f64 | 512x512x512 | 39.569 ms | 3.39 | 0.16 | 5% | ok | 1.0x (25%) |
| blas3 | symm | f64 | 512x512x512 | 48.281 ms | 5.56 | 0.13 | 8% | ok | 1.1x (28%) |
| blas3 | syrk | f64 | 512x512x512 | 39.616 ms | 3.39 | 0.11 | 5% | ok | 1.1x (26%) |
| blas3 | syr2k | f64 | 512x512x512 | 82.559 ms | 3.25 | 0.08 | 5% | ok | 1.0x (24%) |
| blas3 | trmm | f64 | 512x512x512 | 23.591 ms | 5.69 | 0.22 | 8% | ok | 1.1x (28%) |
| blas3 | trsm | f64 | 512x512x512 | 23.256 ms | 5.77 | 0.23 | 9% | ok | 1.2x (29%) |
| blas3 | trsm | f32 | 512x512x512 | 11.113 ms | 12.08 | 0.24 | 9% | ok | 1.3x (32%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 28.305 ms | 0.19 | 1.50 | 6% | ok | 1.0x (24%) |
| factor | lu | f64 | 512x512 | 17.167 ms | 5.21 | 0.24 | 8% | ok | 1.4x (36%) |
| factor | lu | f32 | 512x512 | 8.864 ms | 10.09 | 0.24 | 7% | ok | 1.5x (37%) |
| factor | cholesky | f64 | 512x512 | 18.759 ms | 2.38 | 0.22 | 4% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 17.588 ms | 2.54 | 0.12 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 20.176 ms | 2.22 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 60.924 ms | 0.73 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 146.976 ms | 1.22 | 0.03 | 2% | ok | 1.7x (43%) |
| factor | lq | f64 | 512x512 | 36.524 ms | 4.90 | 0.11 | 7% | ok | 1.7x (42%) |
| factor | hessenberg | f64 | 512x512 | 201.107 ms | 2.22 | 0.02 | 3% | ok | 1.8x (44%) |
| factor | householder | f64 | 512x512 | 0.439 ms | 2.39 | 9.55 | 38% | ok | 3.5x (89%) |
| factor | inv | f64 | 512x512 | 126.031 ms | 2.13 | 0.03 | 3% | ok | 1.2x (29%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.009 ms | 7.45 | 0.35 | 11% | ok | 1.4x (35%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.514 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.740 ms | 1.74 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 526.984 ms | 0.29 | 0.00 | 0% | ok | 1.1x (28%) |
| eigen | eigenvalue | f64 | 256x256 | 59.946 ms | 2.80 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 134.099 ms | 3.13 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.482 ms | 1.80 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 38.413 ms | 6.55 | 0.03 | 10% | ok | 1.8x (44%) |
| eigen | sylvester | f64 | 256x256 | 317.446 ms | 3.17 | 0.00 | 5% | ok | 1.2x (29%) |
| elementwise | exp | f64 | n=8388608 | 76.502 ms | 0.11 | 1.75 | 7% | ok | 1.2x (29%) |
| elementwise | log | f64 | n=8388608 | 101.108 ms | 0.08 | 1.33 | 5% | ok | 1.1x (26%) |
| elementwise | sin | f64 | n=8388608 | 130.148 ms | 0.06 | 1.03 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 65.947 ms | 0.13 | 2.04 | 8% | ok | 1.1x (27%) |
| elementwise | erf | f64 | n=8388608 | 104.495 ms | 0.08 | 1.28 | 5% | ok | 1.1x (28%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.194 ms | 2.19 | 22.80 | 92% | ok | 1.4x (35%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.987 ms | 1.32 | 13.70 | 55% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.115 ms | 6.85 | 14.40 | 58% | ok | 2.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.861 ms | 0.27 | 0.97 | 4% | ok | 1.1x (27%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.697 ms | 0.23 | 1.91 | 8% | ok | 1.0x (24%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 163.728 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 58.682 ms | 0.19 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.470 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 93.495 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 125.966 ms | 0.18 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.207 ms | 0.57 | 0.02 | 1% | ok | 1.1x (27%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 187.084 ms | 0.01 | 0.00 | 0% | ok | 1.1x (27%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.704 ms | 0.00 | 0.00 | n/a | ok | 1.1x (29%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.212 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.634 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.373 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.412 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 155.293 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.014 ms | 0.38 | 1.50 | 6% | ok | 0.9x (23%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.673 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.348 ms | 1.08 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.996 ms | 0.38 | 1.52 | 6% | ok | 0.9x (22%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.960 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 79.261 ms | 0.28 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.568 ms | 0.14 | 0.54 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.017 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 107.678 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.574 ms | 0.14 | 0.54 | 2% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 19.308 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 14705.103 ms | 0.02 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.659 ms | 0.44 | 1.76 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 42.084 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 187.753 ms | 1.64 | 0.00 | 2% | ok | 1.2x (29%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.995 ms | 0.42 | 1.69 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.635 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1589.976 ms | 0.39 | 0.00 | 1% | ok | 1.1x (28%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.617 ms | 0.12 | 0.49 | 2% | ok | 1.0x (25%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 26.561 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2194.938 ms | 0.28 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.332 ms | 0.12 | 0.49 | 2% | ok | 1.0x (25%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.152 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 97.049 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 51.535 ms | 0.00 | 0.00 | n/a | ok | 1.1x (26%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 27.309 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 32.543 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 31.863 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 58.379 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 60.465 ms | 0.00 | 0.00 | n/a | ok | 1.1x (26%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 32.467 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 34.785 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 33.360 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 36.552 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 43.130 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.103 ms | 0.00 | 0.00 | n/a | ok | 1.2x (29%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.489 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.539 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.879 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.727 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ssor | f64 | 128^2 grid | 0.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.591 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.971 ms | 1.33 | 14.88 | 60% | ok | 1.1x (26%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.262 ms | 1.16 | 12.97 | 52% | ok | 1.1x (27%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.677 ms | 0.98 | 10.95 | 44% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.660 ms | 1.18 | 8.90 | 36% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 209.041 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 241.219 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 136.173 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.858 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.191 ms | 2.33 | 9.33 | 38% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.308 ms | 21.81 | 0.34 | 32% | ok | 4.0x (99%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.638 ms | 6.36 | 6.76 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.804 ms | 5.98 | 23.94 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.378 ms | 12.17 | 24.34 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.274 ms | 5.12 | 5.12 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.048 ms | 16.00 | 16.00 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.600 ms | 23.14 | 0.27 | n/a | ok | 2.4x (60%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.682 ms | 40.18 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.138 ms | 52.24 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.137 ms | 52.26 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 9.738 ms | 27.56 | 0.16 | n/a | ok | 4.0x (99%) |

## arduino-ventuno-q -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON_WITHOUT_AES |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.5 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 16.8, f32 33.6 GFLOP/s |
| bandwidth | read 22.2, update 24.6, triad 23.2 GB/s (ceiling 24.6) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 95% | 104% | 0 |
| blas2 | 10 | 67% | 70% | 0 |
| blas3 | 10 | 45% | 58% | 0 |
| factor | 12 | 12% | 32% | 0 |
| eigen | 8 | 15% | 31% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 30% | 66% | 0 |
| direct | 19 | 2% | 7% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 9% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 40% | 57% | 0 |
| mg | 1 | 37% | 37% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 38% | 67% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.754 ms | 2.92 | 23.33 | 95% | ok |
| blas1 | dot | f32 | n=8388608 | 2.880 ms | 5.83 | 23.30 | 95% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.880 ms | 2.13 | 25.55 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.990 ms | 4.20 | 25.23 | 102% | ok |
| blas1 | scale | f64 | n=8388608 | 5.334 ms | 1.57 | 25.16 | 102% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.083 ms | 5.44 | 21.77 | 88% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.722 ms | 0.96 | 7.69 | 31% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.781 ms | 0.47 | 3.77 | 15% | ok |
| blas1 | sum | f64 | n=8388608 | 8.803 ms | 0.95 | 7.62 | 31% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 199.420 ms | 0.93 | 0.67 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 8.071 ms | 4.16 | 16.64 | 68% | ok |
| blas2 | gemv | f32 | n=4096 | 4.091 ms | 8.20 | 16.41 | 67% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.802 ms | 4.30 | 17.21 | 70% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.981 ms | 8.43 | 16.87 | 68% | ok |
| blas2 | symv | f64 | n=4096 | 17.261 ms | 1.94 | 3.89 | 16% | ok |
| blas2 | trmv | f64 | n=4096 | 4.981 ms | 3.37 | 13.49 | 55% | ok |
| blas2 | trmv | f32 | n=4096 | 2.622 ms | 6.40 | 12.81 | 52% | ok |
| blas2 | trsv | f64 | n=4096 | 4.783 ms | 3.51 | 14.04 | 57% | ok |
| blas2 | trsv | f32 | n=4096 | 2.658 ms | 6.31 | 12.63 | 51% | ok |
| blas2 | ger | f64 | n=4096 | 15.473 ms | 2.17 | 17.35 | 70% | ok |
| blas3 | gemm | f64 | 512x512x512 | 28.205 ms | 9.52 | 0.22 | 57% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.815 ms | 19.43 | 0.23 | 58% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 17.912 ms | 7.49 | 0.35 | 45% | ok |
| blas3 | symm | f64 | 512x512x512 | 29.084 ms | 9.23 | 0.22 | 55% | ok |
| blas3 | syrk | f64 | 512x512x512 | 41.928 ms | 3.20 | 0.10 | 19% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 81.040 ms | 3.31 | 0.08 | 20% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.798 ms | 7.54 | 0.29 | 45% | ok |
| blas3 | trsm | f64 | 512x512x512 | 17.831 ms | 7.53 | 0.29 | 45% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.532 ms | 15.73 | 0.31 | 47% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 26.268 ms | 0.20 | 1.62 | 7% | ok |
| factor | lu | f64 | 512x512 | 24.320 ms | 3.68 | 0.17 | 22% | ok |
| factor | lu | f32 | 512x512 | 12.810 ms | 6.99 | 0.16 | 21% | ok |
| factor | cholesky | f64 | 512x512 | 22.376 ms | 2.00 | 0.19 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 19.660 ms | 2.28 | 0.11 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 22.946 ms | 1.95 | 0.18 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 68.380 ms | 0.65 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 251.855 ms | 0.71 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 61.018 ms | 2.93 | 0.07 | 17% | ok |
| factor | hessenberg | f64 | 512x512 | 349.998 ms | 1.28 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.537 ms | 0.68 | 2.73 | 11% | ok |
| factor | inv | f64 | 512x512 | 144.373 ms | 1.86 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.665 ms | 5.37 | 0.25 | 32% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.982 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.279 ms | 1.55 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 615.256 ms | 0.25 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.250 ms | 2.49 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 150.625 ms | 2.78 | 0.01 | 17% | ok |
| eigen | svd | f64 | 256x256 | 230.727 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 47.810 ms | 5.26 | 0.02 | 31% | ok |
| eigen | sylvester | f64 | 256x256 | 357.299 ms | 2.82 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 84.549 ms | 0.10 | 1.59 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 103.500 ms | 0.08 | 1.30 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 148.552 ms | 0.06 | 0.90 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 65.731 ms | 0.13 | 2.04 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 110.356 ms | 0.08 | 1.22 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.687 ms | 1.55 | 16.14 | 66% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.135 ms | 1.23 | 12.76 | 52% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.991 ms | 3.49 | 7.34 | 30% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.490 ms | 0.24 | 0.85 | 3% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.511 ms | 0.24 | 1.96 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 184.091 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 60.897 ms | 0.18 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.435 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 101.989 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 139.046 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.130 ms | 0.52 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 207.623 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.829 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.334 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.701 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 43.515 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.890 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 175.000 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.794 ms | 0.42 | 1.69 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.801 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.458 ms | 0.98 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.753 ms | 0.43 | 1.73 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.416 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 87.228 ms | 0.26 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.806 ms | 0.13 | 0.52 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 10.015 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 118.398 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.829 ms | 0.13 | 0.52 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 21.543 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 16892.197 ms | 0.02 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.348 ms | 0.46 | 1.83 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 47.087 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 210.469 ms | 1.47 | 0.00 | 9% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.589 ms | 0.44 | 1.78 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.529 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1697.655 ms | 0.36 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.334 ms | 0.12 | 0.49 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 29.087 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2536.966 ms | 0.24 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.744 ms | 0.12 | 0.49 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.786 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 104.352 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 53.872 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 28.879 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.469 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 33.859 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 63.330 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.760 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.721 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 37.534 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 36.169 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.367 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 46.708 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.113 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.995 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 30.963 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.138 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.982 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.065 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.744 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.087 ms | 1.25 | 14.05 | 57% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 4.752 ms | 0.55 | 6.17 | 25% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.965 ms | 0.88 | 9.89 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.191 ms | 1.19 | 9.03 | 37% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 234.177 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 271.098 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 148.139 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.172 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.084 ms | 4.11 | 16.43 | 67% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 41.611 ms | 6.45 | 0.10 | 38% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.654 ms | 6.32 | 6.72 | 38% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.822 ms | 5.94 | 23.78 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.394 ms | 12.04 | 24.07 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.314 ms | 7.25 | 7.25 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.254 ms | 3.94 | 3.94 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.397 ms | 15.43 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 16.952 ms | 15.83 | 0.12 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 16.616 ms | 16.16 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 23.646 ms | 11.35 | 0.07 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 50.977 ms | 5.27 | 0.03 | n/a | ok |

## arduino-ventuno-q -- native-fast, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON_WITHOUT_AES |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 67.8 GFLOP/s, f32 135.2 GFLOP/s |
| ... hardware, NEON | f64 67.5, f32 135.2 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 67.8, f32 135.1 GFLOP/s |
| bandwidth | read 23.1, update 25.0, triad 22.8 GB/s (ceiling 25.0) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 97% | 113% | 0 | 28% |
| blas2 | 10 | 56% | 88% | 0 | 26% |
| blas3 | 10 | 25% | 55% | 0 | 57% |
| factor | 12 | 3% | 39% | 0 | 35% |
| eigen | 8 | 4% | 13% | 0 | 28% |
| elementwise | 5 | 5% | 9% | 0 | 26% |
| sparse | 5 | 55% | 91% | 0 | 28% |
| direct | 19 | 0% | 6% | 0 | 27% |
| ordering | 4 | n/a | n/a | 0 | 28% |
| direct3d | 12 | 2% | 7% | 0 | 27% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 45% | 60% | 0 | 28% |
| mg | 1 | 36% | 36% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 38% | 74% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 97% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.555 ms | 3.02 | 24.16 | 97% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.805 ms | 5.98 | 23.92 | 96% | ok | 1.0x (26%) |
| blas1 | axpy | f64 | n=8388608 | 8.157 ms | 2.06 | 24.68 | 99% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 3.979 ms | 4.22 | 25.30 | 101% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 4.759 ms | 1.76 | 28.21 | 113% | ok | 1.1x (28%) |
| blas1 | two_norm | f64 | n=8388608 | 2.634 ms | 6.37 | 25.48 | 102% | ok | 1.2x (29%) |
| blas1 | one_norm | f64 | n=8388608 | 7.882 ms | 1.06 | 8.51 | 34% | ok | 1.1x (28%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.624 ms | 0.54 | 4.30 | 17% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 8.041 ms | 1.04 | 8.35 | 33% | ok | 1.1x (27%) |
| blas1 | dot_k3 | f64 | n=8388608 | 180.048 ms | 1.03 | 0.75 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 6.835 ms | 4.91 | 19.65 | 79% | ok | 1.2x (30%) |
| blas2 | gemv | f32 | n=4096 | 3.048 ms | 11.01 | 22.03 | 88% | ok | 1.3x (34%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.715 ms | 4.35 | 17.41 | 70% | ok | 1.0x (25%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.925 ms | 6.81 | 13.63 | 55% | ok | 0.8x (20%) |
| blas2 | symv | f64 | n=4096 | 15.467 ms | 2.17 | 4.34 | 17% | ok | 1.1x (28%) |
| blas2 | trmv | f64 | n=4096 | 4.837 ms | 3.47 | 13.89 | 56% | ok | 1.0x (26%) |
| blas2 | trmv | f32 | n=4096 | 2.616 ms | 6.41 | 12.84 | 51% | ok | 1.0x (25%) |
| blas2 | trsv | f64 | n=4096 | 4.849 ms | 3.46 | 13.85 | 55% | ok | 1.0x (25%) |
| blas2 | trsv | f32 | n=4096 | 2.558 ms | 6.56 | 13.13 | 53% | ok | 1.0x (26%) |
| blas2 | ger | f64 | n=4096 | 14.982 ms | 2.24 | 17.92 | 72% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 7.404 ms | 36.26 | 0.85 | 53% | ok | 3.8x (95%) |
| blas3 | gemm | f32 | 512x512x512 | 3.586 ms | 74.86 | 0.88 | 55% | ok | 3.9x (96%) |
| blas3 | gemmt | f64 | 512x512x512 | 5.923 ms | 22.66 | 1.06 | 33% | ok | 3.0x (76%) |
| blas3 | symm | f64 | 512x512x512 | 8.127 ms | 33.03 | 0.77 | 49% | ok | 3.6x (89%) |
| blas3 | syrk | f64 | 512x512x512 | 39.616 ms | 3.39 | 0.11 | 5% | ok | 1.1x (26%) |
| blas3 | syr2k | f64 | 512x512x512 | 85.766 ms | 3.13 | 0.07 | 5% | ok | 0.9x (24%) |
| blas3 | trmm | f64 | 512x512x512 | 7.799 ms | 17.21 | 0.67 | 25% | ok | 2.3x (57%) |
| blas3 | trsm | f64 | 512x512x512 | 8.073 ms | 16.63 | 0.65 | 25% | ok | 2.2x (55%) |
| blas3 | trsm | f32 | 512x512x512 | 4.984 ms | 26.93 | 0.53 | 20% | ok | 1.7x (43%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 28.373 ms | 0.19 | 1.50 | 6% | ok | 0.9x (23%) |
| factor | lu | f64 | 512x512 | 17.217 ms | 5.20 | 0.24 | 8% | ok | 1.4x (35%) |
| factor | lu | f32 | 512x512 | 8.926 ms | 10.02 | 0.23 | 7% | ok | 1.4x (36%) |
| factor | cholesky | f64 | 512x512 | 19.979 ms | 2.24 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 17.569 ms | 2.55 | 0.12 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 20.323 ms | 2.20 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 61.355 ms | 0.73 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 146.966 ms | 1.22 | 0.03 | 2% | ok | 1.7x (43%) |
| factor | lq | f64 | 512x512 | 36.790 ms | 4.86 | 0.11 | 7% | ok | 1.7x (41%) |
| factor | hessenberg | f64 | 512x512 | 199.962 ms | 2.24 | 0.02 | 3% | ok | 1.8x (44%) |
| factor | householder | f64 | 512x512 | 0.427 ms | 2.46 | 9.82 | 39% | ok | 3.6x (90%) |
| factor | inv | f64 | 512x512 | 126.047 ms | 2.13 | 0.03 | 3% | ok | 1.1x (29%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.228 ms | 7.32 | 0.34 | 11% | ok | 1.4x (34%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.813 ms | 0.62 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 87.016 ms | 1.74 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 527.104 ms | 0.29 | 0.00 | 0% | ok | 1.2x (29%) |
| eigen | eigenvalue | f64 | 256x256 | 60.291 ms | 2.78 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 134.671 ms | 3.11 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.773 ms | 1.79 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 29.124 ms | 8.64 | 0.04 | 13% | ok | 1.6x (41%) |
| eigen | sylvester | f64 | 256x256 | 314.992 ms | 3.20 | 0.00 | 5% | ok | 1.1x (28%) |
| elementwise | exp | f64 | n=8388608 | 80.767 ms | 0.10 | 1.66 | 7% | ok | 1.0x (26%) |
| elementwise | log | f64 | n=8388608 | 100.787 ms | 0.08 | 1.33 | 5% | ok | 1.0x (26%) |
| elementwise | sin | f64 | n=8388608 | 134.424 ms | 0.06 | 1.00 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 62.521 ms | 0.13 | 2.15 | 9% | ok | 1.1x (26%) |
| elementwise | erf | f64 | n=8388608 | 101.749 ms | 0.08 | 1.32 | 5% | ok | 1.1x (27%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.202 ms | 2.18 | 22.66 | 91% | ok | 1.4x (35%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.998 ms | 1.31 | 13.63 | 55% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.107 ms | 6.86 | 14.42 | 58% | ok | 2.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 12.140 ms | 0.27 | 0.95 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.694 ms | 0.23 | 1.91 | 8% | ok | 1.0x (24%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 164.721 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 55.331 ms | 0.20 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.518 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 94.575 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 127.652 ms | 0.18 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 38.940 ms | 0.57 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 189.474 ms | 0.01 | 0.00 | 0% | ok | 1.1x (27%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.694 ms | 0.00 | 0.00 | n/a | ok | 1.1x (29%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.153 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.631 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.827 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.420 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 156.392 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.016 ms | 0.38 | 1.50 | 6% | ok | 0.9x (22%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.743 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.387 ms | 1.08 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.995 ms | 0.38 | 1.52 | 6% | ok | 0.9x (22%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.997 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 81.417 ms | 0.27 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.574 ms | 0.14 | 0.54 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.027 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 109.094 ms | 0.21 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.566 ms | 0.14 | 0.54 | 2% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 19.153 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 15065.494 ms | 0.02 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.679 ms | 0.44 | 1.75 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 42.413 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 185.491 ms | 1.66 | 0.00 | 2% | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.976 ms | 0.42 | 1.69 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.721 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1587.984 ms | 0.39 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.301 ms | 0.11 | 0.46 | 2% | ok | 0.9x (23%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 26.273 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2354.770 ms | 0.26 | 0.00 | 0% | ok | 1.1x (27%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.791 ms | 0.12 | 0.48 | 2% | ok | 1.0x (25%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.252 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 96.839 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.857 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.094 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.334 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.549 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.174 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.790 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.506 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.726 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.454 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.586 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.179 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.102 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.582 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.480 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.866 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.739 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.592 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.971 ms | 1.33 | 14.88 | 60% | ok | 1.1x (26%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 4.161 ms | 0.63 | 7.05 | 28% | ok | 1.1x (29%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.638 ms | 0.99 | 11.12 | 45% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.485 ms | 1.18 | 8.95 | 36% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 211.902 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 244.706 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 135.230 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.798 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 3.637 ms | 4.61 | 18.45 | 74% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 10.514 ms | 25.53 | 0.40 | 38% | ok | 4.0x (99%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.380 ms | 7.05 | 7.49 | 30% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.795 ms | 6.00 | 24.01 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.372 ms | 12.23 | 24.46 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.102 ms | 7.98 | 7.98 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 3.810 ms | 4.40 | 4.40 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 4.463 ms | 60.14 | 0.70 | n/a | ok | 3.9x (97%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.255 ms | 63.09 | 0.49 | n/a | ok | 4.0x (100%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.193 ms | 64.02 | 0.38 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.958 ms | 45.06 | 0.26 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 12.734 ms | 21.08 | 0.12 | n/a | ok | 4.0x (100%) |

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
| bandwidth | read 22.2, update 24.2, triad 22.7 GB/s (ceiling 24.2) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 85% | 106% | 0 |
| blas2 | 10 | 60% | 74% | 0 |
| blas3 | 10 | 83% | 92% | 0 |
| factor | 12 | 41% | 58% | 0 |
| eigen | 8 | 34% | 74% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 30% | 67% | 0 |
| direct | 23 | 2% | 11% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 15 | 7% | 41% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 41% | 57% | 0 |
| mg | 1 | 37% | 37% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 34% | 35% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.683 ms | 2.95 | 23.62 | 98% | ok |
| blas1 | dot | f32 | n=8388608 | 4.511 ms | 3.72 | 14.88 | 61% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.862 ms | 2.13 | 25.61 | 106% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.987 ms | 4.21 | 25.25 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 5.480 ms | 1.53 | 24.49 | 101% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.253 ms | 5.16 | 20.63 | 85% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.662 ms | 0.97 | 7.75 | 32% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.609 ms | 0.48 | 3.81 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.117 ms | 1.03 | 8.27 | 34% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 187.560 ms | 0.98 | 0.72 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 9.037 ms | 3.71 | 14.86 | 61% | ok |
| blas2 | gemv | f32 | n=4096 | 3.856 ms | 8.70 | 17.41 | 72% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 9.275 ms | 3.62 | 14.48 | 60% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.733 ms | 8.99 | 17.99 | 74% | ok |
| blas2 | symv | f64 | n=4096 | 11.580 ms | 2.90 | 5.80 | 24% | ok |
| blas2 | trmv | f64 | n=4096 | 6.276 ms | 2.67 | 10.70 | 44% | ok |
| blas2 | trmv | f32 | n=4096 | 3.282 ms | 5.11 | 10.23 | 42% | ok |
| blas2 | trsv | f64 | n=4096 | 6.391 ms | 2.63 | 10.51 | 43% | ok |
| blas2 | trsv | f32 | n=4096 | 3.230 ms | 5.19 | 10.40 | 43% | ok |
| blas2 | ger | f64 | n=4096 | 15.349 ms | 2.19 | 17.49 | 72% | ok |
| blas3 | gemm | f64 | 512x512x512 | 17.798 ms | 15.08 | 0.35 | 90% | ok |
| blas3 | gemm | f32 | 512x512x512 | 8.654 ms | 31.02 | 0.36 | 92% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 11.401 ms | 11.77 | 0.55 | 70% | ok |
| blas3 | symm | f64 | 512x512x512 | 18.111 ms | 14.82 | 0.35 | 88% | ok |
| blas3 | syrk | f64 | 512x512x512 | 9.842 ms | 13.64 | 0.43 | 81% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 19.251 ms | 13.94 | 0.33 | 83% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.089 ms | 14.77 | 0.58 | 88% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.747 ms | 13.77 | 0.54 | 82% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.019 ms | 26.74 | 0.52 | 80% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 25.886 ms | 0.21 | 1.64 | 7% | ok |
| factor | lu | f64 | 512x512 | 9.162 ms | 9.77 | 0.46 | 58% | ok |
| factor | lu | f32 | 512x512 | 5.538 ms | 16.16 | 0.38 | 48% | ok |
| factor | cholesky | f64 | 512x512 | 5.409 ms | 8.27 | 0.78 | 49% | ok |
| factor | cholesky | f32 | 512x512 | 3.244 ms | 13.79 | 0.65 | 41% | ok |
| factor | ldlt | f64 | 512x512 | 22.645 ms | 1.98 | 0.19 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 69.422 ms | 0.64 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 19.899 ms | 8.99 | 0.21 | 54% | ok |
| factor | lq | f64 | 512x512 | 60.503 ms | 2.96 | 0.07 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 350.045 ms | 1.28 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.563 ms | 0.67 | 2.68 | 11% | ok |
| factor | inv | f64 | 512x512 | 131.041 ms | 2.05 | 0.03 | 12% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 10.300 ms | 8.69 | 0.41 | 52% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 7.370 ms | 3.04 | 0.14 | 18% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 96.991 ms | 1.56 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 593.306 ms | 0.25 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 45.104 ms | 3.72 | 0.02 | 22% | ok |
| eigen | schur | f64 | 256x256 | 63.626 ms | 6.59 | 0.02 | 39% | ok |
| eigen | svd | f64 | 256x256 | 29.496 ms | 12.51 | 0.04 | 74% | ok |
| eigen | expm | f64 | 256x256 | 38.029 ms | 6.62 | 0.03 | 39% | ok |
| eigen | sylvester | f64 | 256x256 | 176.078 ms | 5.72 | 0.01 | 34% | ok |
| elementwise | exp | f64 | n=8388608 | 89.263 ms | 0.09 | 1.50 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 103.467 ms | 0.08 | 1.30 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 148.006 ms | 0.06 | 0.91 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 69.891 ms | 0.12 | 1.92 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 113.324 ms | 0.07 | 1.18 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.675 ms | 1.56 | 16.26 | 67% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.159 ms | 1.21 | 12.61 | 52% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.035 ms | 3.48 | 7.32 | 30% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 12.862 ms | 0.25 | 0.89 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.812 ms | 0.23 | 1.89 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 181.825 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 57.716 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.437 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 105.182 ms | 0.21 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 143.052 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 44.241 ms | 0.51 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 200.412 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.781 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.321 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.695 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.529 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.871 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 173.801 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.787 ms | 0.42 | 1.69 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.696 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.420 ms | 0.98 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.707 ms | 0.44 | 1.77 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.434 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 89.319 ms | 0.25 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.949 ms | 0.13 | 0.51 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.997 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 121.561 ms | 0.18 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.818 ms | 0.13 | 0.52 | 2% | ok |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 16.663 ms | 0.67 | 0.04 | 4% | ok |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 3.888 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 11.162 ms | 1.00 | 0.07 | 6% | ok |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.163 ms | 0.65 | 2.60 | 11% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 21.448 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 16466.380 ms | 0.02 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.721 ms | 0.44 | 1.75 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 46.209 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 204.997 ms | 1.51 | 0.00 | 9% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.545 ms | 0.45 | 1.79 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.820 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1779.822 ms | 0.35 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.306 ms | 0.11 | 0.46 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 29.055 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2457.092 ms | 0.25 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.600 ms | 0.12 | 0.47 | 2% | ok |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 7.481 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 45.079 ms | 6.85 | 0.02 | 41% | ok |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 2.411 ms | 1.40 | 5.59 | 23% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 71.014 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 104.175 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 53.866 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 28.932 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.542 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 33.970 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 63.892 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.649 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.541 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 37.520 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.956 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 40.111 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 47.485 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.113 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 5.063 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 31.314 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.131 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 3.026 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.065 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.968 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.133 ms | 1.23 | 13.75 | 57% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 4.727 ms | 0.55 | 6.20 | 26% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.969 ms | 0.88 | 9.88 | 41% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.703 ms | 1.18 | 8.89 | 37% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 229.827 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 235.554 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 144.709 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.342 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 8.027 ms | 2.09 | 8.36 | 35% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 48.624 ms | 5.52 | 0.09 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.938 ms | 5.71 | 6.07 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.845 ms | 5.90 | 23.59 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.392 ms | 12.06 | 24.11 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.652 ms | 4.59 | 4.59 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.166 ms | 14.39 | 14.39 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 24.641 ms | 10.89 | 0.13 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 26.386 ms | 10.17 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 20.300 ms | 13.22 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 20.302 ms | 13.22 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 38.385 ms | 6.99 | 0.04 | n/a | ok |

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
| FMA peak (ceiling) | f64 67.8 GFLOP/s, f32 135.0 GFLOP/s |
| ... hardware, NEON | f64 67.8, f32 135.0 GFLOP/s |
| ... this build, scalar | f64 33.9, f32 33.7 GFLOP/s |
| bandwidth | read 22.6, update 24.6, triad 22.4 GB/s (ceiling 24.6) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 98% | 116% | 0 | 28% |
| blas2 | 10 | 63% | 108% | 0 | 37% |
| blas3 | 10 | 74% | 81% | 0 | 88% |
| factor | 12 | 18% | 37% | 0 | 44% |
| eigen | 8 | 11% | 28% | 0 | 33% |
| elementwise | 5 | 6% | 9% | 0 | 27% |
| sparse | 5 | 56% | 91% | 0 | 28% |
| direct | 23 | 1% | 11% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 27% |
| direct3d | 15 | 2% | 24% | 0 | 27% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 45% | 61% | 0 | 28% |
| mg | 1 | 36% | 36% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 32% | 38% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 53% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.550 ms | 3.02 | 24.18 | 98% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.809 ms | 5.97 | 23.89 | 97% | ok | 1.6x (40%) |
| blas1 | axpy | f64 | n=8388608 | 7.923 ms | 2.12 | 25.41 | 103% | ok | 1.0x (25%) |
| blas1 | axpy | f32 | n=8388608 | 3.966 ms | 4.23 | 25.38 | 103% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 4.714 ms | 1.78 | 28.47 | 116% | ok | 1.2x (29%) |
| blas1 | two_norm | f64 | n=8388608 | 2.627 ms | 6.39 | 25.55 | 104% | ok | 1.2x (31%) |
| blas1 | one_norm | f64 | n=8388608 | 7.408 ms | 1.13 | 9.06 | 37% | ok | 1.2x (29%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.518 ms | 0.54 | 4.32 | 18% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.457 ms | 1.12 | 9.00 | 37% | ok | 1.1x (27%) |
| blas1 | dot_k3 | f64 | n=8388608 | 165.155 ms | 1.12 | 0.81 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 5.060 ms | 6.63 | 26.54 | 108% | ok | 1.8x (45%) |
| blas2 | gemv | f32 | n=4096 | 2.610 ms | 12.85 | 25.72 | 105% | ok | 1.5x (37%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 8.708 ms | 3.85 | 15.42 | 63% | ok | 1.1x (27%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.608 ms | 7.28 | 14.57 | 59% | ok | 0.8x (20%) |
| blas2 | symv | f64 | n=4096 | 6.390 ms | 5.25 | 10.51 | 43% | ok | 1.8x (45%) |
| blas2 | trmv | f64 | n=4096 | 3.641 ms | 4.61 | 18.45 | 75% | ok | 1.7x (43%) |
| blas2 | trmv | f32 | n=4096 | 2.171 ms | 7.73 | 15.47 | 63% | ok | 1.5x (38%) |
| blas2 | trsv | f64 | n=4096 | 6.044 ms | 2.78 | 11.11 | 45% | ok | 1.1x (26%) |
| blas2 | trsv | f32 | n=4096 | 3.078 ms | 5.45 | 10.91 | 44% | ok | 1.0x (26%) |
| blas2 | ger | f64 | n=4096 | 14.823 ms | 2.26 | 18.11 | 74% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 5.072 ms | 52.93 | 1.24 | 78% | ok | 3.5x (88%) |
| blas3 | gemm | f32 | 512x512x512 | 2.469 ms | 108.72 | 1.27 | 81% | ok | 3.5x (88%) |
| blas3 | gemmt | f64 | 512x512x512 | 6.506 ms | 20.63 | 0.97 | 30% | ok | 1.8x (44%) |
| blas3 | symm | f64 | 512x512x512 | 5.109 ms | 52.54 | 1.23 | 77% | ok | 3.5x (89%) |
| blas3 | syrk | f64 | 512x512x512 | 3.293 ms | 40.76 | 1.27 | 60% | ok | 3.0x (75%) |
| blas3 | syr2k | f64 | 512x512x512 | 5.813 ms | 46.18 | 1.08 | 68% | ok | 3.3x (83%) |
| blas3 | trmm | f64 | 512x512x512 | 2.546 ms | 52.73 | 2.06 | 78% | ok | 3.6x (89%) |
| blas3 | trsm | f64 | 512x512x512 | 2.690 ms | 49.89 | 1.95 | 74% | ok | 3.6x (91%) |
| blas3 | trsm | f32 | 512x512x512 | 1.464 ms | 91.68 | 1.79 | 68% | ok | 3.4x (86%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 24.733 ms | 0.21 | 1.72 | 7% | ok | 1.0x (26%) |
| factor | lu | f64 | 512x512 | 4.111 ms | 21.76 | 1.02 | 32% | ok | 2.2x (56%) |
| factor | lu | f32 | 512x512 | 3.132 ms | 28.56 | 0.67 | 21% | ok | 1.8x (44%) |
| factor | cholesky | f64 | 512x512 | 3.006 ms | 14.88 | 1.40 | 22% | ok | 1.8x (45%) |
| factor | cholesky | f32 | 512x512 | 2.016 ms | 22.20 | 1.04 | 16% | ok | 1.6x (40%) |
| factor | ldlt | f64 | 512x512 | 20.319 ms | 2.20 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 60.997 ms | 0.73 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 9.678 ms | 18.49 | 0.43 | 27% | ok | 2.1x (51%) |
| factor | lq | f64 | 512x512 | 36.543 ms | 4.90 | 0.11 | 7% | ok | 1.7x (41%) |
| factor | hessenberg | f64 | 512x512 | 199.979 ms | 2.24 | 0.02 | 3% | ok | 1.8x (44%) |
| factor | householder | f64 | 512x512 | 0.462 ms | 2.27 | 9.07 | 37% | ok | 3.4x (85%) |
| factor | inv | f64 | 512x512 | 123.948 ms | 2.17 | 0.03 | 3% | ok | 1.1x (26%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.393 ms | 12.10 | 0.57 | 18% | ok | 1.4x (35%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 5.382 ms | 4.16 | 0.19 | 6% | ok | 1.4x (34%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.889 ms | 1.74 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 526.810 ms | 0.29 | 0.00 | 0% | ok | 1.1x (28%) |
| eigen | eigenvalue | f64 | 256x256 | 37.219 ms | 4.51 | 0.03 | 7% | ok | 1.2x (30%) |
| eigen | schur | f64 | 256x256 | 47.958 ms | 8.75 | 0.02 | 13% | ok | 1.3x (33%) |
| eigen | svd | f64 | 256x256 | 19.597 ms | 18.83 | 0.05 | 28% | ok | 1.5x (38%) |
| eigen | expm | f64 | 256x256 | 23.598 ms | 10.66 | 0.04 | 16% | ok | 1.6x (40%) |
| eigen | sylvester | f64 | 256x256 | 136.346 ms | 7.38 | 0.01 | 11% | ok | 1.3x (32%) |
| elementwise | exp | f64 | n=8388608 | 75.329 ms | 0.11 | 1.78 | 7% | ok | 1.2x (30%) |
| elementwise | log | f64 | n=8388608 | 97.613 ms | 0.09 | 1.38 | 6% | ok | 1.1x (26%) |
| elementwise | sin | f64 | n=8388608 | 135.814 ms | 0.06 | 0.99 | 4% | ok | 1.1x (27%) |
| elementwise | sqrt | f64 | n=8388608 | 58.898 ms | 0.14 | 2.28 | 9% | ok | 1.2x (30%) |
| elementwise | erf | f64 | n=8388608 | 105.347 ms | 0.08 | 1.27 | 5% | ok | 1.1x (27%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.216 ms | 2.15 | 22.38 | 91% | ok | 1.4x (34%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.960 ms | 1.34 | 13.89 | 56% | ok | 1.1x (28%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.320 ms | 6.63 | 13.93 | 57% | ok | 1.9x (48%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.659 ms | 0.28 | 0.99 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.995 ms | 0.22 | 1.85 | 8% | ok | 1.0x (24%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 163.070 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 52.079 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.463 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 95.332 ms | 0.23 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 128.899 ms | 0.17 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.751 ms | 0.56 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 184.731 ms | 0.01 | 0.00 | 0% | ok | 1.1x (27%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.121 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.116 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.634 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.592 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.366 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 154.844 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.979 ms | 0.38 | 1.53 | 6% | ok | 0.9x (23%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.790 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.395 ms | 1.08 | 0.07 | 2% | ok | 1.1x (27%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.983 ms | 0.38 | 1.53 | 6% | ok | 0.9x (22%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.063 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 80.118 ms | 0.28 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.689 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.024 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 110.241 ms | 0.20 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.663 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 14.880 ms | 0.75 | 0.05 | 1% | ok | 1.1x (28%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 3.511 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 10.082 ms | 1.11 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.096 ms | 0.69 | 2.76 | 11% | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 19.264 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 14669.533 ms | 0.02 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 8.022 ms | 0.42 | 1.68 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 42.498 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 183.808 ms | 1.68 | 0.00 | 2% | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.842 ms | 0.43 | 1.72 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.791 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1613.493 ms | 0.38 | 0.00 | 1% | ok | 1.1x (28%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.566 ms | 0.12 | 0.49 | 2% | ok | 1.1x (27%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 26.745 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2394.236 ms | 0.26 | 0.00 | 0% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.947 ms | 0.12 | 0.48 | 2% | ok | 1.0x (26%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 6.670 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 28.119 ms | 10.97 | 0.03 | 16% | ok | 1.6x (40%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 2.276 ms | 1.48 | 5.92 | 24% | ok | 1.1x (26%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.006 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 96.122 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.398 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.238 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.243 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.568 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.917 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.718 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.407 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.910 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.513 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.713 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.314 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.103 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.653 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.814 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.872 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.800 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.064 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.684 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.945 ms | 1.35 | 15.08 | 61% | ok | 1.1x (27%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 4.227 ms | 0.62 | 6.94 | 28% | ok | 1.1x (28%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.677 ms | 0.98 | 10.96 | 45% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.632 ms | 1.18 | 8.91 | 36% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 207.067 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 211.502 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 134.226 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.952 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.204 ms | 2.33 | 9.32 | 38% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.307 ms | 21.81 | 0.34 | 32% | ok | 4.0x (99%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.640 ms | 6.36 | 6.75 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.022 ms | 5.55 | 22.21 | n/a | ok | 0.9x (24%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.426 ms | 11.76 | 23.52 | n/a | ok | 1.0x (24%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.271 ms | 5.13 | 5.13 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.046 ms | 16.04 | 16.04 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.690 ms | 22.96 | 0.27 | n/a | ok | 2.1x (53%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.679 ms | 40.19 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.137 ms | 52.26 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.142 ms | 52.20 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 9.722 ms | 27.61 | 0.16 | n/a | ok | 3.9x (99%) |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

