## ryzen-9-8945hs -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 63.2 GFLOP/s, f32 126.3 GFLOP/s |
| ... hardware, AVX-512F | f64 63.2, f32 126.3 GFLOP/s |
| ... this build, SSE2 | f64 40.7, f32 81.6 GFLOP/s |
| bandwidth | read 45.7, update 52.3, triad 37.9 GB/s (ceiling 52.3) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 103% | 128% | 0 |
| blas2 | 10 | 54% | 73% | 0 |
| blas3 | 10 | 14% | 20% | 0 |
| factor | 12 | 3% | 5% | 0 |
| eigen | 8 | 4% | 12% | 0 |
| elementwise | 5 | 7% | 10% | 0 |
| sparse | 5 | 21% | 57% | 0 |
| direct | 7 | 0% | 1% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 30% | 32% | 0 |
| mg | 1 | 29% | 29% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 59% | 59% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 2.500 ms | 6.71 | 53.68 | 103% | ok |
| blas1 | dot | f32 | n=8388608 | 1.255 ms | 13.37 | 53.48 | 102% | ok |
| blas1 | axpy | f64 | n=8388608 | 3.595 ms | 4.67 | 56.00 | 107% | ok |
| blas1 | axpy | f32 | n=8388608 | 1.749 ms | 9.59 | 57.54 | 110% | ok |
| blas1 | scale | f64 | n=8388608 | 2.000 ms | 4.19 | 67.12 | 128% | ok |
| blas1 | two_norm | f64 | n=8388608 | 1.181 ms | 14.20 | 56.82 | 109% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.395 ms | 1.31 | 10.49 | 20% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.400 ms | 1.31 | 10.49 | 20% | ok |
| blas1 | sum | f64 | n=8388608 | 6.390 ms | 1.31 | 10.50 | 20% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 105.029 ms | 1.76 | 1.28 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.498 ms | 9.59 | 38.38 | 73% | ok |
| blas2 | gemv | f32 | n=4096 | 1.840 ms | 18.24 | 36.49 | 70% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 22.253 ms | 1.51 | 6.03 | 12% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 8.261 ms | 4.06 | 8.13 | 16% | ok |
| blas2 | symv | f64 | n=4096 | 10.434 ms | 3.22 | 6.44 | 12% | ok |
| blas2 | trmv | f64 | n=4096 | 1.956 ms | 8.58 | 34.34 | 66% | ok |
| blas2 | trmv | f32 | n=4096 | 1.243 ms | 13.50 | 27.02 | 52% | ok |
| blas2 | trsv | f64 | n=4096 | 2.337 ms | 7.18 | 28.75 | 55% | ok |
| blas2 | trsv | f32 | n=4096 | 1.194 ms | 14.05 | 28.12 | 54% | ok |
| blas2 | ger | f64 | n=4096 | 20.381 ms | 1.65 | 13.17 | 25% | ok |
| blas3 | gemm | f64 | 512x512x512 | 21.614 ms | 12.42 | 0.29 | 20% | ok |
| blas3 | gemm | f32 | 512x512x512 | 10.825 ms | 24.80 | 0.29 | 20% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 13.861 ms | 9.68 | 0.45 | 15% | ok |
| blas3 | symm | f64 | 512x512x512 | 22.713 ms | 11.82 | 0.28 | 19% | ok |
| blas3 | syrk | f64 | 512x512x512 | 21.422 ms | 6.27 | 0.20 | 10% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 34.823 ms | 7.71 | 0.18 | 12% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.339 ms | 7.74 | 0.30 | 12% | ok |
| blas3 | trsm | f64 | 512x512x512 | 14.950 ms | 8.98 | 0.35 | 14% | ok |
| blas3 | trsm | f32 | 512x512x512 | 11.683 ms | 11.49 | 0.22 | 9% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 14.767 ms | 0.36 | 2.88 | 5% | ok |
| factor | lu | f64 | 512x512 | 62.529 ms | 1.43 | 0.07 | 2% | ok |
| factor | lu | f32 | 512x512 | 60.922 ms | 1.47 | 0.03 | 1% | ok |
| factor | cholesky | f64 | 512x512 | 14.793 ms | 3.02 | 0.28 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 13.714 ms | 3.26 | 0.15 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 14.410 ms | 3.10 | 0.29 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 205.022 ms | 0.22 | 0.02 | 0% | ok |
| factor | qr | f64 | 512x512 | 815.861 ms | 0.22 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 81.457 ms | 2.20 | 0.05 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 932.770 ms | 0.48 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.911 ms | 0.21 | 0.85 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 111.077 ms | 2.42 | 0.04 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 41.410 ms | 2.16 | 0.10 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 48.094 ms | 0.47 | 0.02 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 166.196 ms | 0.91 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 649.868 ms | 0.23 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 75.800 ms | 2.21 | 0.01 | 3% | ok |
| eigen | schur | f64 | 256x256 | 141.079 ms | 2.97 | 0.01 | 5% | ok |
| eigen | svd | f64 | 256x256 | 149.845 ms | 2.46 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 32.603 ms | 7.72 | 0.03 | 12% | ok |
| eigen | sylvester | f64 | 256x256 | 332.324 ms | 3.03 | 0.00 | 5% | ok |
| elementwise | exp | f64 | n=8388608 | 33.426 ms | 0.25 | 4.02 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 51.729 ms | 0.16 | 2.59 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 48.967 ms | 0.17 | 2.74 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 26.439 ms | 0.32 | 5.08 | 10% | ok |
| elementwise | erf | f64 | n=8388608 | 38.976 ms | 0.22 | 3.44 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.919 ms | 2.85 | 29.62 | 57% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.456 ms | 1.80 | 18.70 | 36% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 7.954 ms | 5.26 | 11.07 | 21% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 21.535 ms | 0.15 | 0.53 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.932 ms | 1.04 | 8.62 | 16%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 127.373 ms | 0.09 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 37.139 ms | 0.30 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 20.192 ms | 0.55 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 78.922 ms | 0.28 | 0.01 | 0% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 91.607 ms | 0.24 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 32.577 ms | 0.69 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 104.595 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.589 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.967 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.500 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.939 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 41.333 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 39.078 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 34.910 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.053 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.667 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 19.877 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 39.245 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 36.837 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 21.978 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 24.450 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 23.283 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 24.774 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 27.857 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.048 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.923 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 28.356 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.284 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.183 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.027 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.804 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.769 ms | 1.48 | 16.58 | 32% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.846 ms | 1.42 | 15.89 | 30% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.915 ms | 0.90 | 10.06 | 19% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 19.383 ms | 1.98 | 15.00 | 29% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.571 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 58.743 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 72.118 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.099 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 2.174 ms | 7.72 | 30.87 | 59% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 19.251 ms | 13.94 | 0.22 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.577 ms | 29.06 | 30.88 | 59% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.445 ms | 11.61 | 46.46 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.497 ms | 33.73 | 67.46 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.430 ms | 38.99 | 38.99 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.386 ms | 43.50 | 43.50 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 18.877 ms | 14.22 | 0.17 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 21.941 ms | 12.23 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 22.451 ms | 11.96 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.603 ms | 58.32 | 0.34 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.559 ms | 58.88 | 0.35 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

