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

