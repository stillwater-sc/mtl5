## i7-12700k -- native, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.3 GFLOP/s |
| bandwidth | read 33.1, update 38.5, triad 32.6 GB/s (ceiling 38.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 44% | 97% | 0 |
| blas2 | 10 | 79% | 95% | 0 |
| blas3 | 10 | 17% | 18% | 0 |
| factor | 12 | 8% | 18% | 0 |
| eigen | 8 | 6% | 12% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 49% | 94% | 0 |
| direct | 19 | 1% | 32% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 24% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 62% | 77% | 0 |
| mg | 1 | 56% | 56% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 35% | 83% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.882 ms | 2.85 | 22.82 | 59% | ok |
| blas1 | dot | f32 | n=8388608 | 4.219 ms | 3.98 | 15.91 | 41% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.420 ms | 2.61 | 31.36 | 81% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.719 ms | 3.56 | 21.33 | 55% | ok |
| blas1 | scale | f64 | n=8388608 | 3.578 ms | 2.34 | 37.52 | 97% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.908 ms | 3.42 | 13.67 | 35% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.025 ms | 2.08 | 16.67 | 43% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.039 ms | 1.19 | 9.53 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 3.983 ms | 2.11 | 16.85 | 44% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 77.039 ms | 2.40 | 1.74 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.658 ms | 9.17 | 36.71 | 95% | ok |
| blas2 | gemv | f32 | n=4096 | 2.737 ms | 12.26 | 24.53 | 64% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 4.775 ms | 7.03 | 28.12 | 73% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.204 ms | 15.22 | 30.46 | 79% | ok |
| blas2 | symv | f64 | n=4096 | 3.200 ms | 10.49 | 20.99 | 54% | ok |
| blas2 | trmv | f64 | n=4096 | 2.018 ms | 8.31 | 33.28 | 86% | ok |
| blas2 | trmv | f32 | n=4096 | 1.429 ms | 11.74 | 23.51 | 61% | ok |
| blas2 | trsv | f64 | n=4096 | 1.970 ms | 8.52 | 34.11 | 89% | ok |
| blas2 | trsv | f32 | n=4096 | 1.395 ms | 12.03 | 24.08 | 62% | ok |
| blas2 | ger | f64 | n=4096 | 7.722 ms | 4.35 | 34.77 | 90% | ok |
| blas3 | gemm | f64 | 512x512x512 | 19.993 ms | 13.43 | 0.31 | 17% | ok |
| blas3 | gemm | f32 | 512x512x512 | 10.364 ms | 25.90 | 0.30 | 17% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 12.524 ms | 10.72 | 0.50 | 14% | ok |
| blas3 | symm | f64 | 512x512x512 | 20.101 ms | 13.35 | 0.31 | 17% | ok |
| blas3 | syrk | f64 | 512x512x512 | 13.818 ms | 9.71 | 0.30 | 12% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 24.373 ms | 11.01 | 0.26 | 14% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.602 ms | 13.98 | 0.55 | 18% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.691 ms | 13.85 | 0.54 | 18% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.272 ms | 25.46 | 0.50 | 16% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.178 ms | 0.40 | 3.22 | 8% | ok |
| factor | lu | f64 | 512x512 | 11.665 ms | 7.67 | 0.36 | 10% | ok |
| factor | lu | f32 | 512x512 | 4.725 ms | 18.94 | 0.44 | 12% | ok |
| factor | cholesky | f64 | 512x512 | 7.368 ms | 6.07 | 0.57 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.633 ms | 6.74 | 0.32 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.477 ms | 5.98 | 0.56 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 4.844 ms | 9.24 | 0.87 | 12% | ok |
| factor | qr | f64 | 512x512 | 231.100 ms | 0.77 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 25.936 ms | 6.90 | 0.16 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 272.021 ms | 1.64 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.662 ms | 0.63 | 2.52 | 7%\* | ok |
| factor | inv | f64 | 512x512 | 54.167 ms | 4.96 | 0.08 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.304 ms | 14.19 | 0.67 | 18% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 18.333 ms | 1.22 | 0.06 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 59.030 ms | 2.56 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 745.861 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 32.564 ms | 5.15 | 0.03 | 7% | ok |
| eigen | schur | f64 | 256x256 | 83.533 ms | 5.02 | 0.01 | 6% | ok |
| eigen | svd | f64 | 256x256 | 109.952 ms | 3.36 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 25.857 ms | 9.73 | 0.04 | 12% | ok |
| eigen | sylvester | f64 | 256x256 | 190.209 ms | 5.29 | 0.01 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 39.426 ms | 0.21 | 3.40 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 46.030 ms | 0.18 | 2.92 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 60.156 ms | 0.14 | 2.23 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 29.081 ms | 0.29 | 4.62 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 203.149 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.751 ms | 3.48 | 36.23 | 94% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.914 ms | 2.86 | 29.78 | 77% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.661 ms | 8.99 | 18.89 | 49% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 5.289 ms | 0.62 | 2.17 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.307 ms | 1.54 | 12.73 | 33%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 14.494 ms | 0.77 | 0.05 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 24.983 ms | 0.45 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.401 ms | 1.33 | 0.09 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.825 ms | 0.56 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.526 ms | 0.50 | 0.02 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 17.187 ms | 1.30 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 24.865 ms | 0.08 | 0.01 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.337 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.992 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 58.043 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.186 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.270 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.118 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 12.731 ms | 0.88 | 0.06 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.248 ms | 3.05 | 12.21 | 32%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.069 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.068 ms | 2.75 | 0.18 | 4% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.251 ms | 3.01 | 12.04 | 31%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.076 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 36.010 ms | 0.62 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.635 ms | 1.19 | 4.77 | 12%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.427 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.815 ms | 0.58 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.640 ms | 1.18 | 4.72 | 12%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.954 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 204.374 ms | 1.51 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.441 ms | 2.34 | 9.35 | 24%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.622 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 60.307 ms | 5.12 | 0.01 | 7% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.463 ms | 2.30 | 9.21 | 24%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.865 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 552.964 ms | 1.12 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.532 ms | 0.45 | 1.79 | 5%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.275 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 455.080 ms | 1.36 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.558 ms | 0.45 | 1.78 | 5%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 31.388 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 35.193 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 23.184 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.653 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 15.202 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 14.186 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.954 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.483 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.411 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 18.299 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.730 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 16.217 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 19.230 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.042 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.554 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 11.455 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.089 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.019 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.028 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.728 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.992 ms | 2.64 | 29.57 | 77% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.221 ms | 2.14 | 24.02 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.564 ms | 1.67 | 18.75 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.361 ms | 2.88 | 21.75 | 56% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 30.788 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 31.604 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.604 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.979 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.914 ms | 3.41 | 13.66 | 35% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 38.992 ms | 6.88 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.556 ms | 30.18 | 32.07 | 83%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.304 ms | 7.28 | 29.13 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.995 ms | 16.86 | 33.72 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.723 ms | 9.74 | 9.74 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.725 ms | 9.73 | 9.73 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.485 ms | 15.35 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.826 ms | 12.89 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.230 ms | 26.24 | 0.15 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.229 ms | 26.24 | 0.15 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 26.845 ms | 10.00 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

