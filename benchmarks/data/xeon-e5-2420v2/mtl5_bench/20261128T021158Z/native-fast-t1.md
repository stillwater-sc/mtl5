## xeon-e5-2420v2 -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 21.0 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 21.0 GFLOP/s |
| ... this build, SSE4 | f64 10.5, f32 21.0 GFLOP/s |
| bandwidth | read 10.4, update 10.2, triad 8.2 GB/s (ceiling 10.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 104% | 110% | 0 |
| blas2 | 10 | 91% | 99% | 0 |
| blas3 | 10 | 76% | 95% | 0 |
| factor | 12 | 17% | 41% | 0 |
| eigen | 8 | 15% | 48% | 0 |
| elementwise | 5 | 9% | 14% | 0 |
| sparse | 5 | 56% | 105% | 0 |
| direct | 19 | 3% | 35% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 19% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 78% | 84% | 0 |
| mg | 1 | 77% | 77% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 74% | 108% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 12.025 ms | 1.40 | 11.16 | 107% | ok |
| blas1 | dot | f32 | n=8388608 | 5.951 ms | 2.82 | 11.28 | 108% | ok |
| blas1 | axpy | f64 | n=8388608 | 18.823 ms | 0.89 | 10.70 | 103% | ok |
| blas1 | axpy | f32 | n=8388608 | 8.960 ms | 1.87 | 11.23 | 108% | ok |
| blas1 | scale | f64 | n=8388608 | 12.376 ms | 0.68 | 10.85 | 104% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.884 ms | 2.85 | 11.40 | 110% | ok |
| blas1 | one_norm | f64 | n=8388608 | 9.780 ms | 0.86 | 6.86 | 66% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 13.188 ms | 0.64 | 5.09 | 49% | ok |
| blas1 | sum | f64 | n=8388608 | 9.713 ms | 0.86 | 6.91 | 66% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 271.773 ms | 0.68 | 0.49 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 13.478 ms | 2.49 | 9.96 | 96% | ok |
| blas2 | gemv | f32 | n=4096 | 6.705 ms | 5.00 | 10.01 | 96% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 13.027 ms | 2.58 | 10.31 | 99% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.549 ms | 5.12 | 10.25 | 98% | ok |
| blas2 | symv | f64 | n=4096 | 12.349 ms | 2.72 | 5.44 | 52% | ok |
| blas2 | trmv | f64 | n=4096 | 7.160 ms | 2.34 | 9.38 | 90% | ok |
| blas2 | trmv | f32 | n=4096 | 3.705 ms | 4.53 | 9.07 | 87% | ok |
| blas2 | trsv | f64 | n=4096 | 7.111 ms | 2.36 | 9.45 | 91% | ok |
| blas2 | trsv | f32 | n=4096 | 3.756 ms | 4.47 | 8.94 | 86% | ok |
| blas2 | ger | f64 | n=4096 | 28.218 ms | 1.19 | 9.52 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 28.115 ms | 9.55 | 0.22 | 91% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.481 ms | 19.91 | 0.23 | 95% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 17.072 ms | 7.86 | 0.37 | 75% | ok |
| blas3 | symm | f64 | 512x512x512 | 29.262 ms | 9.17 | 0.22 | 87% | ok |
| blas3 | syrk | f64 | 512x512x512 | 16.869 ms | 7.96 | 0.25 | 76% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 33.716 ms | 7.96 | 0.19 | 76% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.758 ms | 7.56 | 0.30 | 72% | ok |
| blas3 | trsm | f64 | 512x512x512 | 18.289 ms | 7.34 | 0.29 | 70% | ok |
| blas3 | trsm | f32 | 512x512x512 | 9.378 ms | 14.31 | 0.28 | 68% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.412 ms | 0.14 | 1.11 | 11% | ok |
| factor | lu | f64 | 512x512 | 29.734 ms | 3.01 | 0.14 | 29% | ok |
| factor | lu | f32 | 512x512 | 15.633 ms | 5.72 | 0.13 | 27% | ok |
| factor | cholesky | f64 | 512x512 | 25.609 ms | 1.75 | 0.16 | 17% | ok |
| factor | cholesky | f32 | 512x512 | 24.128 ms | 1.85 | 0.09 | 9% | ok |
| factor | ldlt | f64 | 512x512 | 25.945 ms | 1.72 | 0.16 | 16% | ok |
| factor | ldlt_bk | f64 | 512x512 | 13.104 ms | 3.41 | 0.32 | 32% | ok |
| factor | qr | f64 | 512x512 | 500.732 ms | 0.36 | 0.01 | 3% | ok |
| factor | lq | f64 | 512x512 | 73.623 ms | 2.43 | 0.06 | 23% | ok |
| factor | hessenberg | f64 | 512x512 | 622.280 ms | 0.72 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.301 ms | 0.32 | 1.27 | 12%\* | ok |
| factor | inv | f64 | 512x512 | 155.344 ms | 1.73 | 0.03 | 16% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 21.003 ms | 4.26 | 0.20 | 41% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 67.177 ms | 0.33 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 171.196 ms | 0.88 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 967.696 ms | 0.16 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 105.778 ms | 1.59 | 0.01 | 15% | ok |
| eigen | schur | f64 | 256x256 | 270.197 ms | 1.55 | 0.00 | 15% | ok |
| eigen | svd | f64 | 256x256 | 262.334 ms | 1.41 | 0.00 | 13% | ok |
| eigen | expm | f64 | 256x256 | 50.388 ms | 4.99 | 0.02 | 48% | ok |
| eigen | sylvester | f64 | 256x256 | 600.040 ms | 1.68 | 0.00 | 16% | ok |
| elementwise | exp | f64 | n=8388608 | 150.818 ms | 0.06 | 0.89 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 140.470 ms | 0.06 | 0.96 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 170.953 ms | 0.05 | 0.79 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 91.558 ms | 0.09 | 1.47 | 14% | ok |
| elementwise | erf | f64 | n=8388608 | 2477.217 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.484 ms | 1.05 | 10.96 | 105% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.097 ms | 0.85 | 8.79 | 84% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.201 ms | 2.75 | 5.79 | 56% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.177 ms | 0.20 | 0.71 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.798 ms | 0.35 | 2.87 | 28% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 42.331 ms | 0.26 | 0.02 | 3% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 69.930 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.380 ms | 0.42 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 117.919 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 131.275 ms | 0.17 | 0.01 | 2% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 47.885 ms | 0.47 | 0.02 | 4% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 81.608 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.287 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.537 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 130.186 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.562 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 45.309 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.438 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.939 ms | 0.33 | 0.02 | 3% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.843 ms | 0.90 | 3.59 | 34%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.981 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 13.039 ms | 0.86 | 0.06 | 8% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.833 ms | 0.91 | 3.63 | 35%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.581 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 104.755 ms | 0.21 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.988 ms | 0.19 | 0.76 | 7%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 10.400 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 113.176 ms | 0.20 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.802 ms | 0.20 | 0.80 | 8%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 13.375 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 635.201 ms | 0.49 | 0.00 | 5% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 6.781 ms | 0.50 | 1.99 | 19%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 30.614 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 206.136 ms | 1.50 | 0.00 | 14% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 6.821 ms | 0.49 | 1.98 | 19%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 13.064 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1729.335 ms | 0.36 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.704 ms | 0.10 | 0.41 | 4%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 19.160 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1991.166 ms | 0.31 | 0.00 | 3% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.827 ms | 0.10 | 0.41 | 4%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 80.806 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 134.486 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 59.731 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 32.150 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 38.493 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 37.671 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 69.398 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 69.740 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 39.416 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 44.079 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 40.718 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.769 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 47.722 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.129 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.660 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 33.824 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.958 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.756 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.093 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.072 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.602 ms | 0.73 | 8.14 | 78% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.342 ms | 0.78 | 8.78 | 84% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.922 ms | 0.67 | 7.48 | 72% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 36.181 ms | 1.06 | 8.03 | 77% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 140.800 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 146.962 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 133.044 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 16.135 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 5.953 ms | 2.82 | 11.27 | 108% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 61.749 ms | 4.35 | 0.07 | 41% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.320 ms | 7.23 | 7.68 | 74% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.050 ms | 2.77 | 11.09 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.964 ms | 5.66 | 11.32 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.079 ms | 8.07 | 8.07 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.640 ms | 10.23 | 10.23 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 18.838 ms | 14.25 | 0.17 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 22.138 ms | 12.13 | 0.09 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 22.100 ms | 12.15 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 16.138 ms | 16.63 | 0.10 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 12.377 ms | 21.69 | 0.13 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

