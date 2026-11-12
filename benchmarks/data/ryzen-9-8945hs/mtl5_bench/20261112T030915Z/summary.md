# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,8, CPUs 0,2,4,6,8,10,12,14, clock policy power scheme copy of Balanced with processor state 100-100% and EPP 0, boost aggressive; AC, 2026-11-12T03:09:15Z.

## ryzen-9-8945hs -- native, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 77.3 GFLOP/s, f32 126.1 GFLOP/s |
| ... hardware, AVX-512F | f64 77.3, f32 126.1 GFLOP/s |
| ... this build, scalar | f64 1.9, f32 1.9 GFLOP/s |
| bandwidth | read 36.6, update 52.2, triad 37.9 GB/s (ceiling 52.2) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 20% | 60% | 0 |
| blas2 | 10 | 44% | 73% | 0 |
| blas3 | 10 | 12% | 21% | 0 |
| factor | 12 | 3% | 4% | 0 |
| eigen | 8 | 3% | 9% | 0 |
| elementwise | 5 | 6% | 10% | 0 |
| sparse | 5 | 22% | 56% | 0 |
| direct | 7 | 0% | 1% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 27% | 32% | 0 |
| mg | 1 | 29% | 29% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 11% | 20% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 12.284 ms | 1.37 | 10.93 | 21% | ok |
| blas1 | dot | f32 | n=8388608 | 12.249 ms | 1.37 | 5.48 | 11% | ok |
| blas1 | axpy | f64 | n=8388608 | 15.043 ms | 1.12 | 13.38 | 26% | ok |
| blas1 | axpy | f32 | n=8388608 | 14.925 ms | 1.12 | 6.74 | 13% | ok |
| blas1 | scale | f64 | n=8388608 | 4.298 ms | 1.95 | 31.23 | 60% | ok |
| blas1 | two_norm | f64 | n=8388608 | 11.718 ms | 1.43 | 5.73 | 11% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.401 ms | 1.31 | 10.48 | 20% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.405 ms | 1.31 | 10.48 | 20% | ok |
| blas1 | sum | f64 | n=8388608 | 6.399 ms | 1.31 | 10.49 | 20% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 109.536 ms | 1.68 | 1.23 | 2% | ok |
| blas2 | gemv | f64 | n=4096 | 3.539 ms | 9.48 | 37.94 | 73% | ok |
| blas2 | gemv | f32 | n=4096 | 2.926 ms | 11.47 | 22.95 | 44% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.608 ms | 4.41 | 17.65 | 34% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.473 ms | 13.57 | 27.15 | 52% | ok |
| blas2 | symv | f64 | n=4096 | 10.190 ms | 3.29 | 6.59 | 13% | ok |
| blas2 | trmv | f64 | n=4096 | 2.029 ms | 8.27 | 33.11 | 63% | ok |
| blas2 | trmv | f32 | n=4096 | 1.827 ms | 9.18 | 18.38 | 35% | ok |
| blas2 | trsv | f64 | n=4096 | 1.990 ms | 8.43 | 33.76 | 65% | ok |
| blas2 | trsv | f32 | n=4096 | 1.892 ms | 8.87 | 17.76 | 34% | ok |
| blas2 | ger | f64 | n=4096 | 16.533 ms | 2.03 | 16.24 | 31% | ok |
| blas3 | gemm | f64 | 512x512x512 | 19.555 ms | 13.73 | 0.32 | 18% | ok |
| blas3 | gemm | f32 | 512x512x512 | 10.224 ms | 26.25 | 0.31 | 21% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 13.860 ms | 9.68 | 0.45 | 13% | ok |
| blas3 | symm | f64 | 512x512x512 | 19.454 ms | 13.80 | 0.32 | 18% | ok |
| blas3 | syrk | f64 | 512x512x512 | 16.791 ms | 7.99 | 0.25 | 10% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 29.202 ms | 9.19 | 0.22 | 12% | ok |
| blas3 | trmm | f64 | 512x512x512 | 30.782 ms | 4.36 | 0.17 | 6% | ok |
| blas3 | trsm | f64 | 512x512x512 | 35.275 ms | 3.80 | 0.15 | 5% | ok |
| blas3 | trsm | f32 | 512x512x512 | 32.356 ms | 4.15 | 0.08 | 3% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 11.237 ms | 0.47 | 3.78 | 7% | ok |
| factor | lu | f64 | 512x512 | 39.884 ms | 2.24 | 0.11 | 3% | ok |
| factor | lu | f32 | 512x512 | 38.400 ms | 2.33 | 0.05 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 14.366 ms | 3.11 | 0.29 | 4% | ok |
| factor | cholesky | f32 | 512x512 | 13.745 ms | 3.25 | 0.15 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 14.465 ms | 3.09 | 0.29 | 4% | ok |
| factor | ldlt_bk | f64 | 512x512 | 206.801 ms | 0.22 | 0.02 | 0% | ok |
| factor | qr | f64 | 512x512 | 819.927 ms | 0.22 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 81.044 ms | 2.21 | 0.05 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 945.372 ms | 0.47 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.901 ms | 0.21 | 0.86 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 110.128 ms | 2.44 | 0.04 | 3% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 41.418 ms | 2.16 | 0.10 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 49.191 ms | 0.45 | 0.02 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 163.312 ms | 0.92 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 653.539 ms | 0.23 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 74.320 ms | 2.26 | 0.01 | 3% | ok |
| eigen | schur | f64 | 256x256 | 144.622 ms | 2.90 | 0.01 | 4% | ok |
| eigen | svd | f64 | 256x256 | 169.403 ms | 2.18 | 0.01 | 3% | ok |
| eigen | expm | f64 | 256x256 | 37.447 ms | 6.72 | 0.03 | 9% | ok |
| eigen | sylvester | f64 | 256x256 | 329.128 ms | 3.06 | 0.00 | 4% | ok |
| elementwise | exp | f64 | n=8388608 | 39.798 ms | 0.21 | 3.37 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 51.722 ms | 0.16 | 2.59 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 48.050 ms | 0.17 | 2.79 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 26.551 ms | 0.32 | 5.06 | 10% | ok |
| elementwise | erf | f64 | n=8388608 | 39.008 ms | 0.22 | 3.44 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.935 ms | 2.80 | 29.12 | 56% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.520 ms | 1.72 | 17.92 | 34% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 7.700 ms | 5.44 | 11.43 | 22% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.885 ms | 0.16 | 0.55 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.934 ms | 1.04 | 8.61 | 17%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 134.126 ms | 0.08 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 39.610 ms | 0.28 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 22.004 ms | 0.51 | 0.03 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 67.082 ms | 0.33 | 0.01 | 0% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 90.768 ms | 0.25 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 32.312 ms | 0.69 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 96.883 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.579 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.954 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.499 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 18.176 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 49.721 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 55.499 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 38.906 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.178 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 27.562 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.362 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 64.562 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 42.888 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.077 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 29.883 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 28.777 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 32.153 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 39.232 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.048 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.927 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 28.300 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.267 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.153 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.027 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.813 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.744 ms | 1.50 | 16.82 | 32% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.085 ms | 1.26 | 14.06 | 27% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.233 ms | 1.17 | 13.14 | 25% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 19.503 ms | 1.97 | 14.90 | 29% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.913 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 58.797 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 69.704 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.739 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 11.811 ms | 1.42 | 5.68 | 11% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 36.887 ms | 7.28 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.726 ms | 9.72 | 10.33 | 20% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.317 ms | 3.89 | 15.54 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.222 ms | 5.21 | 10.41 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.728 ms | 9.71 | 9.71 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.729 ms | 9.71 | 9.71 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 9.959 ms | 26.95 | 0.32 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 38.480 ms | 6.98 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 38.775 ms | 6.92 | 0.04 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 38.877 ms | 6.90 | 0.04 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 34.577 ms | 7.76 | 0.05 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## ryzen-9-8945hs -- native, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 522.9 GFLOP/s, f32 1009.8 GFLOP/s |
| ... hardware, AVX-512F | f64 522.9, f32 1009.8 GFLOP/s |
| ... this build, scalar | f64 14.4, f32 14.3 GFLOP/s |
| bandwidth | read 52.4, update 53.1, triad 40.9 GB/s (ceiling 53.1) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 84% | 117% | 0 | 52% |
| blas2 | 10 | 60% | 113% | 0 | 13% |
| blas3 | 10 | 2% | 21% | 0 | 13% |
| factor | 12 | 1% | 6% | 0 | 14% |
| eigen | 8 | 0% | 2% | 0 | 13% |
| elementwise | 5 | 6% | 10% | 0 | 13% |
| sparse | 5 | 31% | 235% | 0 | 12% |
| direct | 7 | 0% | 0% | 0 | 12% |
| ordering | 4 | n/a | n/a | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 26% | 31% | 0 | 12% |
| mg | 1 | 25% | 25% | 0 | 11% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 11% | 20% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 95% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.326 ms | 7.21 | 57.71 | 109% | ok | 5.3x (66%) |
| blas1 | dot | f32 | n=8388608 | 1.565 ms | 10.72 | 42.88 | 81% | ok | 7.8x (98%) |
| blas1 | axpy | f64 | n=8388608 | 3.592 ms | 4.67 | 56.05 | 106% | ok | 4.2x (52%) |
| blas1 | axpy | f32 | n=8388608 | 1.973 ms | 8.50 | 51.02 | 96% | ok | 7.6x (95%) |
| blas1 | scale | f64 | n=8388608 | 2.152 ms | 3.90 | 62.36 | 117% | ok | 2.0x (25%) |
| blas1 | two_norm | f64 | n=8388608 | 1.501 ms | 11.18 | 44.70 | 84% | ok | 7.8x (98%) |
| blas1 | one_norm | f64 | n=8388608 | 6.398 ms | 1.31 | 10.49 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.432 ms | 1.30 | 10.43 | 20% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 6.395 ms | 1.31 | 10.49 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 108.900 ms | 1.69 | 1.23 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 2.403 ms | 13.96 | 55.88 | 105% | ok | 1.5x (18%) |
| blas2 | gemv | f32 | n=4096 | 1.117 ms | 30.05 | 60.12 | 113% | ok | 2.6x (33%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.388 ms | 14.05 | 56.23 | 106% | ok | 3.2x (40%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.184 ms | 28.35 | 56.73 | 107% | ok | 2.1x (26%) |
| blas2 | symv | f64 | n=4096 | 10.639 ms | 3.15 | 6.31 | 12% | ok | 1.0x (12%) |
| blas2 | trmv | f64 | n=4096 | 2.342 ms | 7.16 | 28.68 | 54% | ok | 0.9x (11%) |
| blas2 | trmv | f32 | n=4096 | 1.815 ms | 9.24 | 18.50 | 35% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 2.107 ms | 7.96 | 31.88 | 60% | ok | 0.9x (12%) |
| blas2 | trsv | f32 | n=4096 | 1.875 ms | 8.95 | 17.91 | 34% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 16.364 ms | 2.05 | 16.41 | 31% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 2.558 ms | 104.93 | 2.46 | 20% | ok | 7.6x (96%) |
| blas3 | gemm | f32 | 512x512x512 | 1.283 ms | 209.21 | 2.45 | 21% | ok | 8.0x (100%) |
| blas3 | gemmt | f64 | 512x512x512 | 13.970 ms | 9.61 | 0.45 | 2% | ok | 1.0x (12%) |
| blas3 | symm | f64 | 512x512x512 | 19.373 ms | 13.86 | 0.32 | 3% | ok | 1.0x (13%) |
| blas3 | syrk | f64 | 512x512x512 | 17.014 ms | 7.89 | 0.25 | 2% | ok | 1.0x (12%) |
| blas3 | syr2k | f64 | 512x512x512 | 28.611 ms | 9.38 | 0.22 | 2% | ok | 1.0x (13%) |
| blas3 | trmm | f64 | 512x512x512 | 31.096 ms | 4.32 | 0.17 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f64 | 512x512x512 | 35.600 ms | 3.77 | 0.15 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f32 | 512x512x512 | 32.247 ms | 4.16 | 0.08 | 0% | ok | 1.0x (13%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 15.310 ms | 0.35 | 2.77 | 5% | ok | 0.7x (9%) |
| factor | lu | f64 | 512x512 | 45.340 ms | 1.97 | 0.09 | 0% | ok | 0.9x (11%) |
| factor | lu | f32 | 512x512 | 33.624 ms | 2.66 | 0.06 | 0% | ok | 1.1x (14%) |
| factor | cholesky | f64 | 512x512 | 15.827 ms | 2.83 | 0.27 | 1% | ok | 0.9x (11%) |
| factor | cholesky | f32 | 512x512 | 13.734 ms | 3.26 | 0.15 | 0% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 14.440 ms | 3.10 | 0.29 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 211.406 ms | 0.21 | 0.02 | 0% | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 502.665 ms | 0.36 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 54.143 ms | 3.31 | 0.08 | 1% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 586.325 ms | 0.76 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.256 ms | 0.84 | 3.34 | 6%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 98.287 ms | 2.73 | 0.04 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 29.924 ms | 2.99 | 0.14 | 1% | ok | 1.4x (17%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 48.385 ms | 0.46 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 163.059 ms | 0.93 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 640.406 ms | 0.24 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 75.302 ms | 2.23 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | schur | f64 | 256x256 | 146.036 ms | 2.87 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 155.047 ms | 2.38 | 0.01 | 0% | ok | 1.1x (14%) |
| eigen | expm | f64 | 256x256 | 23.886 ms | 10.54 | 0.04 | 2% | ok | 1.6x (20%) |
| eigen | sylvester | f64 | 256x256 | 324.524 ms | 3.10 | 0.00 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 35.514 ms | 0.24 | 3.78 | 7% | ok | 1.1x (14%) |
| elementwise | log | f64 | n=8388608 | 51.819 ms | 0.16 | 2.59 | 5% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 48.027 ms | 0.17 | 2.79 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 26.513 ms | 0.32 | 5.06 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 38.917 ms | 0.22 | 3.45 | 6% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.219 ms | 11.97 | 124.57 | 235% ! | ok | 4.3x (53%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.655 ms | 1.58 | 16.46 | 31% | ok | 0.9x (11%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.321 ms | 18.04 | 37.94 | 71% | ok | 3.3x (41%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 22.910 ms | 0.14 | 0.50 | 1%\* | ok | 0.9x (11%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.947 ms | 1.03 | 8.55 | 16%\* | ok | 1.0x (12%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 136.020 ms | 0.08 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 39.936 ms | 0.28 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 23.371 ms | 0.48 | 0.03 | 0% | ok | 0.9x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 68.700 ms | 0.33 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 90.819 ms | 0.25 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 32.920 ms | 0.68 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 97.042 ms | 0.02 | 0.00 | 0% | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.631 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.968 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.515 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.530 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 49.888 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 55.465 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 38.640 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.076 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 27.442 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.309 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 64.181 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 42.959 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.008 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 29.892 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 28.761 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 31.984 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 39.212 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.048 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.981 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 28.981 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 2.316 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 128^2 grid | 2.176 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.027 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.810 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.776 ms | 1.47 | 16.52 | 31% | ok | 1.0x (12%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.102 ms | 1.25 | 13.95 | 26% | ok | 1.0x (12%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.225 ms | 1.18 | 13.18 | 25% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 21.688 ms | 1.77 | 13.40 | 25% | ok | 0.9x (11%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 42.422 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 58.826 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 72.076 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.724 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 11.727 ms | 1.43 | 5.72 | 11% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.832 ms | 55.55 | 0.87 | 11% | ok | 7.6x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.717 ms | 9.77 | 10.38 | 20% | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.276 ms | 3.92 | 15.69 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.218 ms | 5.21 | 10.43 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.726 ms | 9.72 | 9.72 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.726 ms | 9.72 | 9.72 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.286 ms | 208.72 | 2.45 | n/a | ok | 7.7x (97%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.024 ms | 53.43 | 0.42 | n/a | ok | 7.7x (96%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.061 ms | 53.05 | 0.31 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.058 ms | 53.08 | 0.31 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.530 ms | 59.26 | 0.35 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

## ryzen-9-8945hs -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 63.2 GFLOP/s, f32 126.6 GFLOP/s |
| ... hardware, AVX-512F | f64 63.2, f32 126.6 GFLOP/s |
| ... this build, SSE2 | f64 40.7, f32 81.3 GFLOP/s |
| bandwidth | read 45.9, update 51.7, triad 37.9 GB/s (ceiling 51.7) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 105% | 128% | 0 |
| blas2 | 10 | 64% | 77% | 0 |
| blas3 | 10 | 17% | 31% | 0 |
| factor | 12 | 3% | 5% | 0 |
| eigen | 8 | 4% | 12% | 0 |
| elementwise | 5 | 7% | 10% | 0 |
| sparse | 5 | 16% | 53% | 0 |
| direct | 7 | 0% | 1% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 30% | 31% | 0 |
| mg | 1 | 29% | 29% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 60% | 60% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 2.506 ms | 6.70 | 53.57 | 104% | ok |
| blas1 | dot | f32 | n=8388608 | 1.238 ms | 13.55 | 54.20 | 105% | ok |
| blas1 | axpy | f64 | n=8388608 | 3.691 ms | 4.55 | 54.54 | 105% | ok |
| blas1 | axpy | f32 | n=8388608 | 1.734 ms | 9.68 | 58.06 | 112% | ok |
| blas1 | scale | f64 | n=8388608 | 2.025 ms | 4.14 | 66.28 | 128% | ok |
| blas1 | two_norm | f64 | n=8388608 | 1.182 ms | 14.20 | 56.79 | 110% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.444 ms | 1.30 | 10.41 | 20% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.456 ms | 1.30 | 10.40 | 20% | ok |
| blas1 | sum | f64 | n=8388608 | 6.441 ms | 1.30 | 10.42 | 20% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 106.259 ms | 1.74 | 1.26 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.401 ms | 9.87 | 39.48 | 76% | ok |
| blas2 | gemv | f32 | n=4096 | 1.681 ms | 19.96 | 39.94 | 77% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 22.715 ms | 1.48 | 5.91 | 11% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 7.382 ms | 4.55 | 9.10 | 18% | ok |
| blas2 | symv | f64 | n=4096 | 10.190 ms | 3.29 | 6.59 | 13% | ok |
| blas2 | trmv | f64 | n=4096 | 2.028 ms | 8.27 | 33.12 | 64% | ok |
| blas2 | trmv | f32 | n=4096 | 1.012 ms | 16.57 | 33.17 | 64% | ok |
| blas2 | trsv | f64 | n=4096 | 1.971 ms | 8.51 | 34.08 | 66% | ok |
| blas2 | trsv | f32 | n=4096 | 1.013 ms | 16.56 | 33.15 | 64% | ok |
| blas2 | ger | f64 | n=4096 | 16.385 ms | 2.05 | 16.39 | 32% | ok |
| blas3 | gemm | f64 | 512x512x512 | 13.883 ms | 19.34 | 0.45 | 31% | ok |
| blas3 | gemm | f32 | 512x512x512 | 6.958 ms | 38.58 | 0.45 | 30% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 9.319 ms | 14.40 | 0.68 | 23% | ok |
| blas3 | symm | f64 | 512x512x512 | 15.136 ms | 17.73 | 0.42 | 28% | ok |
| blas3 | syrk | f64 | 512x512x512 | 16.781 ms | 8.00 | 0.25 | 13% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 29.373 ms | 9.14 | 0.21 | 14% | ok |
| blas3 | trmm | f64 | 512x512x512 | 12.214 ms | 10.99 | 0.43 | 17% | ok |
| blas3 | trsm | f64 | 512x512x512 | 12.407 ms | 10.82 | 0.42 | 17% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.238 ms | 16.29 | 0.32 | 13% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 11.342 ms | 0.47 | 3.74 | 7% | ok |
| factor | lu | f64 | 512x512 | 40.283 ms | 2.22 | 0.10 | 4% | ok |
| factor | lu | f32 | 512x512 | 39.171 ms | 2.28 | 0.05 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 14.374 ms | 3.11 | 0.29 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 13.806 ms | 3.24 | 0.15 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 14.509 ms | 3.08 | 0.29 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 208.838 ms | 0.21 | 0.02 | 0% | ok |
| factor | qr | f64 | 512x512 | 818.881 ms | 0.22 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 82.060 ms | 2.18 | 0.05 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 942.390 ms | 0.47 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.962 ms | 0.21 | 0.85 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 111.863 ms | 2.40 | 0.04 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 42.353 ms | 2.11 | 0.10 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 48.246 ms | 0.46 | 0.02 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 167.565 ms | 0.90 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 650.720 ms | 0.23 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 75.665 ms | 2.22 | 0.01 | 4% | ok |
| eigen | schur | f64 | 256x256 | 146.790 ms | 2.86 | 0.01 | 5% | ok |
| eigen | svd | f64 | 256x256 | 152.497 ms | 2.42 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 33.081 ms | 7.61 | 0.03 | 12% | ok |
| eigen | sylvester | f64 | 256x256 | 332.851 ms | 3.02 | 0.00 | 5% | ok |
| elementwise | exp | f64 | n=8388608 | 33.815 ms | 0.25 | 3.97 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 60.647 ms | 0.14 | 2.21 | 4% | ok |
| elementwise | sin | f64 | n=8388608 | 51.013 ms | 0.16 | 2.63 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 26.721 ms | 0.31 | 5.02 | 10% | ok |
| elementwise | erf | f64 | n=8388608 | 39.385 ms | 0.21 | 3.41 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.998 ms | 2.62 | 27.28 | 53% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 4.006 ms | 0.65 | 6.80 | 13% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 7.782 ms | 5.38 | 11.31 | 22% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 21.242 ms | 0.15 | 0.54 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.956 ms | 1.03 | 8.51 | 16%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 125.450 ms | 0.09 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 37.056 ms | 0.30 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 19.893 ms | 0.56 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 66.158 ms | 0.34 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 91.351 ms | 0.24 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 31.941 ms | 0.70 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 105.233 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.598 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 7.009 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.498 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.995 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 41.366 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 39.183 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 34.905 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.124 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.753 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 19.896 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 39.593 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 36.804 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.005 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 24.632 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 23.311 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 24.859 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 28.186 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.048 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.926 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 28.771 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.326 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.227 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.027 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.806 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.871 ms | 1.40 | 15.68 | 30% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.837 ms | 1.43 | 15.97 | 31% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.226 ms | 1.18 | 13.17 | 25% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 19.519 ms | 1.97 | 14.89 | 29% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.853 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 58.734 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 72.987 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.121 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 2.174 ms | 7.72 | 30.87 | 60% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 19.282 ms | 13.92 | 0.22 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.578 ms | 29.01 | 30.82 | 60% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.428 ms | 11.75 | 47.00 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.494 ms | 33.99 | 67.98 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.430 ms | 38.99 | 38.99 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.389 ms | 43.18 | 43.18 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 18.942 ms | 14.17 | 0.17 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 22.051 ms | 12.17 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 22.725 ms | 11.81 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.586 ms | 58.53 | 0.34 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.558 ms | 58.89 | 0.35 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## ryzen-9-8945hs -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 505.1 GFLOP/s, f32 1008.7 GFLOP/s |
| ... hardware, AVX-512F | f64 505.1, f32 1008.7 GFLOP/s |
| ... this build, SSE2 | f64 316.4, f32 631.0 GFLOP/s |
| bandwidth | read 52.6, update 53.2, triad 40.8 GB/s (ceiling 53.2) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 110% | 125% | 0 | 13% |
| blas2 | 10 | 63% | 119% | 0 | 13% |
| blas3 | 10 | 5% | 19% | 0 | 20% |
| factor | 12 | 1% | 6% | 0 | 18% |
| eigen | 8 | 0% | 2% | 0 | 13% |
| elementwise | 5 | 6% | 9% | 0 | 13% |
| sparse | 5 | 33% | 237% | 0 | 33% |
| direct | 7 | 0% | 0% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 30% | 30% | 0 | 13% |
| mg | 1 | 29% | 29% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 58% | 58% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 55% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.292 ms | 7.32 | 58.57 | 110% | ok | 1.1x (14%) |
| blas1 | dot | f32 | n=8388608 | 1.007 ms | 16.66 | 66.64 | 125% | ok | 1.2x (15%) |
| blas1 | axpy | f64 | n=8388608 | 3.539 ms | 4.74 | 56.89 | 107% | ok | 1.0x (13%) |
| blas1 | axpy | f32 | n=8388608 | 1.672 ms | 10.03 | 60.20 | 113% | ok | 1.0x (13%) |
| blas1 | scale | f64 | n=8388608 | 2.152 ms | 3.90 | 62.37 | 117% | ok | 0.9x (12%) |
| blas1 | two_norm | f64 | n=8388608 | 1.049 ms | 16.00 | 63.98 | 120% | ok | 1.1x (14%) |
| blas1 | one_norm | f64 | n=8388608 | 6.397 ms | 1.31 | 10.49 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.410 ms | 1.31 | 10.47 | 20% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 6.393 ms | 1.31 | 10.50 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 112.158 ms | 1.65 | 1.20 | 2% | ok | 0.9x (12%) |
| blas2 | gemv | f64 | n=4096 | 2.360 ms | 14.22 | 56.91 | 107% | ok | 1.4x (18%) |
| blas2 | gemv | f32 | n=4096 | 1.064 ms | 31.55 | 63.13 | 119% | ok | 1.6x (20%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 9.973 ms | 3.36 | 13.46 | 25% | ok | 2.3x (28%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.980 ms | 6.74 | 13.48 | 25% | ok | 1.5x (19%) |
| blas2 | symv | f64 | n=4096 | 10.196 ms | 3.29 | 6.59 | 12% | ok | 1.0x (12%) |
| blas2 | trmv | f64 | n=4096 | 1.974 ms | 8.50 | 34.02 | 64% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 1.080 ms | 15.53 | 31.09 | 58% | ok | 0.9x (12%) |
| blas2 | trsv | f64 | n=4096 | 1.968 ms | 8.53 | 34.14 | 64% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.008 ms | 16.65 | 33.33 | 63% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 16.261 ms | 2.06 | 16.51 | 31% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 2.763 ms | 97.14 | 2.28 | 19% | ok | 5.0x (63%) |
| blas3 | gemm | f32 | 512x512x512 | 1.563 ms | 171.77 | 2.01 | 17% | ok | 4.5x (56%) |
| blas3 | gemmt | f64 | 512x512x512 | 5.348 ms | 25.09 | 1.18 | 5% | ok | 1.7x (22%) |
| blas3 | symm | f64 | 512x512x512 | 3.882 ms | 69.15 | 1.62 | 14% | ok | 3.9x (49%) |
| blas3 | syrk | f64 | 512x512x512 | 16.845 ms | 7.97 | 0.25 | 2% | ok | 1.0x (12%) |
| blas3 | syr2k | f64 | 512x512x512 | 29.076 ms | 9.23 | 0.22 | 2% | ok | 1.0x (13%) |
| blas3 | trmm | f64 | 512x512x512 | 7.953 ms | 16.88 | 0.66 | 3% | ok | 1.5x (19%) |
| blas3 | trsm | f64 | 512x512x512 | 7.567 ms | 17.74 | 0.69 | 4% | ok | 1.6x (20%) |
| blas3 | trsm | f32 | 512x512x512 | 5.508 ms | 24.37 | 0.48 | 2% | ok | 1.5x (19%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 11.078 ms | 0.48 | 3.83 | 7% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 27.348 ms | 3.27 | 0.15 | 1% | ok | 1.5x (18%) |
| factor | lu | f32 | 512x512 | 25.824 ms | 3.46 | 0.08 | 0% | ok | 1.5x (19%) |
| factor | cholesky | f64 | 512x512 | 14.351 ms | 3.12 | 0.29 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 13.727 ms | 3.26 | 0.15 | 0% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 14.440 ms | 3.10 | 0.29 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 206.855 ms | 0.22 | 0.02 | 0% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 501.383 ms | 0.36 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 54.087 ms | 3.31 | 0.08 | 1% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 583.027 ms | 0.77 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.262 ms | 0.83 | 3.32 | 6%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 97.412 ms | 2.76 | 0.04 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 29.709 ms | 3.01 | 0.14 | 1% | ok | 1.4x (18%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 45.656 ms | 0.49 | 0.02 | 0% | ok | 1.1x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 162.296 ms | 0.93 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 638.299 ms | 0.24 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 71.125 ms | 2.36 | 0.01 | 0% | ok | 1.1x (13%) |
| eigen | schur | f64 | 256x256 | 135.039 ms | 3.11 | 0.01 | 1% | ok | 1.1x (14%) |
| eigen | svd | f64 | 256x256 | 154.998 ms | 2.38 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | expm | f64 | 256x256 | 23.261 ms | 10.82 | 0.05 | 2% | ok | 1.4x (18%) |
| eigen | sylvester | f64 | 256x256 | 324.229 ms | 3.10 | 0.00 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 33.419 ms | 0.25 | 4.02 | 8% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 52.367 ms | 0.16 | 2.56 | 5% | ok | 1.2x (14%) |
| elementwise | sin | f64 | n=8388608 | 48.972 ms | 0.17 | 2.74 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 26.750 ms | 0.31 | 5.02 | 9% | ok | 1.0x (12%) |
| elementwise | erf | f64 | n=8388608 | 38.889 ms | 0.22 | 3.45 | 6% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.216 ms | 12.11 | 126.01 | 237% ! | ok | 4.6x (58%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.531 ms | 1.71 | 17.78 | 33% | ok | 2.6x (33%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.343 ms | 17.87 | 37.58 | 71% | ok | 3.3x (42%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 22.692 ms | 0.14 | 0.51 | 1%\* | ok | 0.9x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.910 ms | 1.05 | 8.72 | 16%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 125.362 ms | 0.09 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 36.587 ms | 0.31 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 19.427 ms | 0.58 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 67.837 ms | 0.33 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 90.814 ms | 0.25 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 33.320 ms | 0.67 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 107.242 ms | 0.02 | 0.00 | 0% | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.612 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 7.080 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.508 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 18.024 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 41.469 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 39.535 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 34.920 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.049 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 26.744 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 27.245 ms | 0.00 | 0.00 | n/a | ok | 0.7x (9%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 39.051 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 36.831 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 21.999 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 24.865 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 23.622 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 24.888 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 27.904 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.048 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 3.070 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 28.186 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 2.261 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 2.155 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.027 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.811 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.863 ms | 1.40 | 15.74 | 30% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.834 ms | 1.43 | 15.99 | 30% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.228 ms | 1.17 | 13.16 | 25% | ok | 1.0x (12%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.925 ms | 2.03 | 15.36 | 29% | ok | 1.0x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.776 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 58.737 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 69.558 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.120 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 2.173 ms | 7.72 | 30.88 | 58% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 3.173 ms | 84.60 | 1.32 | 17% | ok | 6.1x (76%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.583 ms | 28.80 | 30.60 | 58% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.461 ms | 11.49 | 45.94 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.496 ms | 33.85 | 67.70 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.430 ms | 38.98 | 38.98 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.399 ms | 42.03 | 42.03 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.962 ms | 90.63 | 1.06 | n/a | ok | 6.4x (80%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 3.082 ms | 87.09 | 0.68 | n/a | ok | 7.2x (89%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 3.073 ms | 87.37 | 0.51 | n/a | ok | 7.4x (92%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.798 ms | 336.51 | 1.97 | n/a | ok | 5.7x (72%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 1.043 ms | 257.44 | 1.51 | n/a | ok | 4.4x (55%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

## ryzen-9-8945hs -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 65.3 GFLOP/s, f32 125.9 GFLOP/s |
| ... hardware, AVX-512F | f64 65.3, f32 125.9 GFLOP/s |
| ... this build, scalar | f64 1.6, f32 1.9 GFLOP/s |
| bandwidth | read 36.7, update 52.1, triad 37.9 GB/s (ceiling 52.1) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 20% | 98% | 0 |
| blas2 | 10 | 69% | 88% | 0 |
| blas3 | 10 | 67% | 90% | 0 |
| factor | 12 | 12% | 23% | 0 |
| eigen | 8 | 17% | 52% | 0 |
| elementwise | 5 | 6% | 10% | 0 |
| sparse | 5 | 22% | 56% | 0 |
| direct | 7 | 0% | 1% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 27% | 32% | 0 |
| mg | 1 | 29% | 29% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 11% | 19% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 14.421 ms | 1.16 | 9.31 | 18% | ok |
| blas1 | dot | f32 | n=8388608 | 12.500 ms | 1.34 | 5.37 | 10% | ok |
| blas1 | axpy | f64 | n=8388608 | 15.001 ms | 1.12 | 13.42 | 26% | ok |
| blas1 | axpy | f32 | n=8388608 | 12.800 ms | 1.31 | 7.86 | 15% | ok |
| blas1 | scale | f64 | n=8388608 | 2.643 ms | 3.17 | 50.77 | 98% | ok |
| blas1 | two_norm | f64 | n=8388608 | 13.389 ms | 1.25 | 5.01 | 10% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.393 ms | 1.31 | 10.50 | 20% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.419 ms | 1.31 | 10.46 | 20% | ok |
| blas1 | sum | f64 | n=8388608 | 6.394 ms | 1.31 | 10.50 | 20% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 111.621 ms | 1.65 | 1.20 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.712 ms | 9.04 | 36.18 | 69% | ok |
| blas2 | gemv | f32 | n=4096 | 1.474 ms | 22.77 | 45.56 | 88% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.564 ms | 9.41 | 37.67 | 72% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.560 ms | 21.51 | 43.04 | 83% | ok |
| blas2 | symv | f64 | n=4096 | 2.079 ms | 16.14 | 32.31 | 62% | ok |
| blas2 | trmv | f64 | n=4096 | 1.869 ms | 8.98 | 35.95 | 69% | ok |
| blas2 | trmv | f32 | n=4096 | 0.950 ms | 17.65 | 35.34 | 68% | ok |
| blas2 | trsv | f64 | n=4096 | 1.910 ms | 8.78 | 35.17 | 68% | ok |
| blas2 | trsv | f32 | n=4096 | 0.942 ms | 17.81 | 35.65 | 68% | ok |
| blas2 | ger | f64 | n=4096 | 16.284 ms | 2.06 | 16.49 | 32% | ok |
| blas3 | gemm | f64 | 512x512x512 | 4.577 ms | 58.65 | 1.37 | 90% | ok |
| blas3 | gemm | f32 | 512x512x512 | 2.377 ms | 112.93 | 1.32 | 90% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 3.421 ms | 39.24 | 1.84 | 60% | ok |
| blas3 | symm | f64 | 512x512x512 | 4.759 ms | 56.41 | 1.32 | 86% | ok |
| blas3 | syrk | f64 | 512x512x512 | 3.713 ms | 36.15 | 1.13 | 55% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 6.137 ms | 43.74 | 1.03 | 67% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.363 ms | 56.80 | 2.22 | 87% | ok |
| blas3 | trsm | f64 | 512x512x512 | 3.503 ms | 38.32 | 1.50 | 59% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.638 ms | 81.94 | 1.60 | 65% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 11.075 ms | 0.48 | 3.83 | 7% | ok |
| factor | lu | f64 | 512x512 | 6.379 ms | 14.03 | 0.66 | 21% | ok |
| factor | lu | f32 | 512x512 | 3.067 ms | 29.18 | 0.68 | 23% | ok |
| factor | cholesky | f64 | 512x512 | 5.494 ms | 8.14 | 0.76 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 2.107 ms | 21.24 | 1.00 | 17% | ok |
| factor | ldlt | f64 | 512x512 | 14.838 ms | 3.02 | 0.28 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 204.407 ms | 0.22 | 0.02 | 0% | ok |
| factor | qr | f64 | 512x512 | 12.976 ms | 13.79 | 0.32 | 21% | ok |
| factor | lq | f64 | 512x512 | 82.283 ms | 2.17 | 0.05 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 942.426 ms | 0.47 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.943 ms | 0.21 | 0.85 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 76.845 ms | 3.49 | 0.05 | 5% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.013 ms | 12.76 | 0.60 | 20% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 2.110 ms | 10.60 | 0.50 | 16% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 164.794 ms | 0.92 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 644.865 ms | 0.23 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 21.049 ms | 7.97 | 0.05 | 12% | ok |
| eigen | schur | f64 | 256x256 | 27.429 ms | 15.29 | 0.04 | 23% | ok |
| eigen | svd | f64 | 256x256 | 10.890 ms | 33.89 | 0.10 | 52% | ok |
| eigen | expm | f64 | 256x256 | 20.222 ms | 12.44 | 0.05 | 19% | ok |
| eigen | sylvester | f64 | 256x256 | 91.736 ms | 10.97 | 0.02 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 33.658 ms | 0.25 | 3.99 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 54.591 ms | 0.15 | 2.46 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 48.203 ms | 0.17 | 2.78 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 26.540 ms | 0.32 | 5.06 | 10% | ok |
| elementwise | erf | f64 | n=8388608 | 42.861 ms | 0.20 | 3.13 | 6% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.938 ms | 2.79 | 29.02 | 56% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.930 ms | 1.36 | 14.11 | 27% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 7.784 ms | 5.38 | 11.31 | 22% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 21.528 ms | 0.15 | 0.53 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.953 ms | 1.03 | 8.52 | 16%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 130.309 ms | 0.09 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 38.935 ms | 0.29 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 22.716 ms | 0.49 | 0.03 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 66.643 ms | 0.34 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 91.394 ms | 0.24 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 30.953 ms | 0.72 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 102.434 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.602 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.959 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.501 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.630 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 51.336 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 58.327 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 40.844 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.042 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 30.578 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.951 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 70.681 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 44.039 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.640 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 31.292 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 30.084 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 33.481 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.022 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.045 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.930 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 28.703 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.276 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.171 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.027 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.745 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.772 ms | 1.48 | 16.55 | 32% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.058 ms | 1.27 | 14.25 | 27% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.228 ms | 1.17 | 13.16 | 25% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 19.582 ms | 1.96 | 14.84 | 29% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.726 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.567 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 69.577 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.961 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 14.386 ms | 1.17 | 4.66 | 9% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 37.456 ms | 7.17 | 0.11 | 11% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.835 ms | 9.14 | 9.71 | 19% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.281 ms | 3.92 | 15.67 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.235 ms | 5.19 | 10.37 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.727 ms | 9.72 | 9.72 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.728 ms | 9.71 | 9.71 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 10.216 ms | 26.28 | 0.31 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 38.424 ms | 6.99 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 38.719 ms | 6.93 | 0.04 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 38.830 ms | 6.91 | 0.04 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 34.534 ms | 7.77 | 0.05 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## ryzen-9-8945hs -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 504.9 GFLOP/s, f32 1000.8 GFLOP/s |
| ... hardware, AVX-512F | f64 504.9, f32 1000.8 GFLOP/s |
| ... this build, scalar | f64 12.3, f32 14.4 GFLOP/s |
| bandwidth | read 52.5, update 53.1, triad 40.7 GB/s (ceiling 53.1) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 81% | 118% | 0 | 53% |
| blas2 | 10 | 100% | 116% | 0 | 16% |
| blas3 | 10 | 45% | 78% | 0 | 68% |
| factor | 12 | 2% | 6% | 0 | 12% |
| eigen | 8 | 2% | 5% | 0 | 12% |
| elementwise | 5 | 6% | 10% | 0 | 13% |
| sparse | 5 | 28% | 98% | 0 | 13% |
| direct | 7 | 0% | 0% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 27% | 30% | 0 | 12% |
| mg | 1 | 26% | 26% | 0 | 11% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 11% | 18% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 95% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.352 ms | 7.13 | 57.06 | 107% | ok | 6.1x (77%) |
| blas1 | dot | f32 | n=8388608 | 1.566 ms | 10.71 | 42.86 | 81% | ok | 8.0x (100%) |
| blas1 | axpy | f64 | n=8388608 | 3.571 ms | 4.70 | 56.37 | 106% | ok | 4.2x (53%) |
| blas1 | axpy | f32 | n=8388608 | 1.813 ms | 9.25 | 55.52 | 104% | ok | 7.1x (88%) |
| blas1 | scale | f64 | n=8388608 | 2.146 ms | 3.91 | 62.54 | 118% | ok | 1.2x (15%) |
| blas1 | two_norm | f64 | n=8388608 | 1.701 ms | 9.86 | 39.46 | 74% | ok | 7.9x (98%) |
| blas1 | one_norm | f64 | n=8388608 | 6.397 ms | 1.31 | 10.49 | 20% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.423 ms | 1.31 | 10.45 | 20% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 6.394 ms | 1.31 | 10.50 | 20% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 111.408 ms | 1.66 | 1.20 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 2.538 ms | 13.22 | 52.92 | 100% | ok | 1.5x (18%) |
| blas2 | gemv | f32 | n=4096 | 1.177 ms | 28.50 | 57.02 | 107% | ok | 1.3x (16%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.709 ms | 12.39 | 49.57 | 93% | ok | 1.3x (16%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.319 ms | 25.43 | 50.88 | 96% | ok | 1.2x (15%) |
| blas2 | symv | f64 | n=4096 | 1.255 ms | 26.73 | 53.52 | 101% | ok | 1.7x (21%) |
| blas2 | trmv | f64 | n=4096 | 1.256 ms | 13.36 | 53.48 | 101% | ok | 1.5x (19%) |
| blas2 | trmv | f32 | n=4096 | 0.543 ms | 30.91 | 61.89 | 116% | ok | 1.8x (22%) |
| blas2 | trsv | f64 | n=4096 | 1.738 ms | 9.65 | 38.65 | 73% | ok | 1.1x (14%) |
| blas2 | trsv | f32 | n=4096 | 0.944 ms | 17.78 | 35.59 | 67% | ok | 1.0x (12%) |
| blas2 | ger | f64 | n=4096 | 16.304 ms | 2.06 | 16.47 | 31% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 0.679 ms | 395.11 | 9.26 | 78% | ok | 6.7x (84%) |
| blas3 | gemm | f32 | 512x512x512 | 0.386 ms | 695.25 | 8.15 | 69% | ok | 6.2x (77%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.665 ms | 80.59 | 3.78 | 16% | ok | 2.1x (26%) |
| blas3 | symm | f64 | 512x512x512 | 0.680 ms | 394.87 | 9.25 | 78% | ok | 7.0x (88%) |
| blas3 | syrk | f64 | 512x512x512 | 2.002 ms | 67.03 | 2.09 | 13% | ok | 1.9x (23%) |
| blas3 | syr2k | f64 | 512x512x512 | 2.704 ms | 99.29 | 2.33 | 20% | ok | 2.3x (28%) |
| blas3 | trmm | f64 | 512x512x512 | 0.455 ms | 295.05 | 11.53 | 58% | ok | 5.2x (65%) |
| blas3 | trsm | f64 | 512x512x512 | 0.619 ms | 216.86 | 8.47 | 43% | ok | 5.7x (71%) |
| blas3 | trsm | f32 | 512x512x512 | 0.301 ms | 445.91 | 8.71 | 45% | ok | 5.4x (68%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.934 ms | 0.49 | 3.88 | 7% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 11.301 ms | 7.92 | 0.37 | 2% | ok | 0.6x (7%) |
| factor | lu | f32 | 512x512 | 5.086 ms | 17.59 | 0.41 | 2% | ok | 0.6x (8%) |
| factor | cholesky | f64 | 512x512 | 6.892 ms | 6.49 | 0.61 | 1% | ok | 0.8x (10%) |
| factor | cholesky | f32 | 512x512 | 2.448 ms | 18.28 | 0.86 | 2% | ok | 0.9x (11%) |
| factor | ldlt | f64 | 512x512 | 14.861 ms | 3.01 | 0.28 | 1% | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 208.001 ms | 0.22 | 0.02 | 0% | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 14.186 ms | 12.62 | 0.30 | 2% | ok | 0.9x (11%) |
| factor | lq | f64 | 512x512 | 54.589 ms | 3.28 | 0.08 | 1% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 581.926 ms | 0.77 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.262 ms | 0.83 | 3.32 | 6%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 82.227 ms | 3.26 | 0.05 | 1% | ok | 0.9x (12%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 8.890 ms | 10.06 | 0.47 | 2% | ok | 0.8x (10%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 4.354 ms | 5.14 | 0.24 | 1% | ok | 0.5x (6%) |
| eigen | eigen_symmetric | f64 | 256x256 | 164.264 ms | 0.92 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 632.834 ms | 0.24 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 25.176 ms | 6.66 | 0.04 | 1% | ok | 0.8x (10%) |
| eigen | schur | f64 | 256x256 | 31.292 ms | 13.40 | 0.03 | 3% | ok | 0.9x (11%) |
| eigen | svd | f64 | 256x256 | 13.562 ms | 27.22 | 0.08 | 5% | ok | 0.8x (10%) |
| eigen | expm | f64 | 256x256 | 21.416 ms | 11.75 | 0.05 | 2% | ok | 0.9x (12%) |
| eigen | sylvester | f64 | 256x256 | 99.564 ms | 10.11 | 0.02 | 2% | ok | 0.9x (12%) |
| elementwise | exp | f64 | n=8388608 | 33.478 ms | 0.25 | 4.01 | 8% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 53.352 ms | 0.16 | 2.52 | 5% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 47.937 ms | 0.17 | 2.80 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 26.436 ms | 0.32 | 5.08 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 42.412 ms | 0.20 | 3.16 | 6% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.523 ms | 5.00 | 52.07 | 98% | ok | 1.8x (22%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.817 ms | 1.44 | 14.98 | 28% | ok | 1.1x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.344 ms | 17.86 | 37.56 | 71% | ok | 3.3x (42%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 21.551 ms | 0.15 | 0.53 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.926 ms | 1.04 | 8.65 | 16%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 126.269 ms | 0.09 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 38.507 ms | 0.29 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 23.558 ms | 0.47 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 67.254 ms | 0.33 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 90.984 ms | 0.25 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 31.943 ms | 0.70 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 101.647 ms | 0.02 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.604 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.924 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.502 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.587 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 50.965 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 58.304 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 39.520 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.854 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 28.312 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.720 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 69.200 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 44.074 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.624 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 30.761 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 29.804 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 33.434 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 41.410 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.048 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.974 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 28.810 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 2.288 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 128^2 grid | 2.148 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.028 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.698 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.839 ms | 1.42 | 15.95 | 30% | ok | 1.0x (12%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.072 ms | 1.26 | 14.16 | 27% | ok | 1.0x (12%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.593 ms | 1.01 | 11.31 | 21% | ok | 0.9x (11%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 21.346 ms | 1.80 | 13.62 | 26% | ok | 0.9x (11%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 42.203 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.724 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 69.550 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.971 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 14.381 ms | 1.17 | 4.67 | 9% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.886 ms | 54.94 | 0.86 | 11% | ok | 7.7x (96%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.835 ms | 9.14 | 9.71 | 18% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.405 ms | 3.81 | 15.24 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.234 ms | 5.19 | 10.38 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.726 ms | 9.72 | 9.72 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.727 ms | 9.72 | 9.72 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.303 ms | 206.08 | 2.41 | n/a | ok | 7.8x (98%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.034 ms | 53.32 | 0.42 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.056 ms | 53.10 | 0.31 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.064 ms | 53.01 | 0.31 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.545 ms | 59.06 | 0.35 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

