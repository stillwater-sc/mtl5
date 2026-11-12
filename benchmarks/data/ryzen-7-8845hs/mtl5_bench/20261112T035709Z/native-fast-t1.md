## ryzen-7-8845hs -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 80.1 GFLOP/s, f32 160.1 GFLOP/s |
| ... hardware, AVX-512F | f64 80.1, f32 160.1 GFLOP/s |
| ... this build, SSE2 | f64 40.0, f32 80.1 GFLOP/s |
| bandwidth | read 47.2, update 52.8, triad 39.0 GB/s (ceiling 52.8) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 107% | 133% | 0 |
| blas2 | 10 | 69% | 76% | 0 |
| blas3 | 10 | 17% | 31% | 0 |
| factor | 12 | 3% | 5% | 0 |
| eigen | 8 | 4% | 12% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 27% | 61% | 0 |
| direct | 7 | 0% | 1% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 32% | 37% | 0 |
| mg | 1 | 34% | 34% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 69% | 71% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 2.382 ms | 7.04 | 56.34 | 107% | ok |
| blas1 | dot | f32 | n=8388608 | 1.182 ms | 14.20 | 56.79 | 107% | ok |
| blas1 | axpy | f64 | n=8388608 | 3.439 ms | 4.88 | 58.55 | 111% | ok |
| blas1 | axpy | f32 | n=8388608 | 1.646 ms | 10.19 | 61.17 | 116% | ok |
| blas1 | scale | f64 | n=8388608 | 1.903 ms | 4.41 | 70.53 | 133% | ok |
| blas1 | two_norm | f64 | n=8388608 | 1.165 ms | 14.40 | 57.59 | 109% | ok |
| blas1 | one_norm | f64 | n=8388608 | 5.065 ms | 1.66 | 13.25 | 25% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 5.113 ms | 1.64 | 13.12 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 5.075 ms | 1.65 | 13.22 | 25% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 91.203 ms | 2.02 | 1.47 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.346 ms | 10.03 | 40.13 | 76% | ok |
| blas2 | gemv | f32 | n=4096 | 1.706 ms | 19.67 | 39.37 | 74% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 21.609 ms | 1.55 | 6.21 | 12% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 7.868 ms | 4.26 | 8.53 | 16% | ok |
| blas2 | symv | f64 | n=4096 | 8.705 ms | 3.85 | 7.72 | 15% | ok |
| blas2 | trmv | f64 | n=4096 | 1.824 ms | 9.20 | 36.82 | 70% | ok |
| blas2 | trmv | f32 | n=4096 | 0.944 ms | 17.76 | 35.56 | 67% | ok |
| blas2 | trsv | f64 | n=4096 | 1.845 ms | 9.09 | 36.40 | 69% | ok |
| blas2 | trsv | f32 | n=4096 | 0.901 ms | 18.62 | 37.27 | 71% | ok |
| blas2 | ger | f64 | n=4096 | 13.538 ms | 2.48 | 19.83 | 38% | ok |
| blas3 | gemm | f64 | 512x512x512 | 10.978 ms | 24.45 | 0.57 | 31% | ok |
| blas3 | gemm | f32 | 512x512x512 | 5.565 ms | 48.24 | 0.57 | 30% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 7.415 ms | 18.10 | 0.85 | 23% | ok |
| blas3 | symm | f64 | 512x512x512 | 11.973 ms | 22.42 | 0.53 | 28% | ok |
| blas3 | syrk | f64 | 512x512x512 | 13.566 ms | 9.89 | 0.31 | 12% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 23.003 ms | 11.67 | 0.27 | 15% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.725 ms | 13.80 | 0.54 | 17% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.845 ms | 13.63 | 0.53 | 17% | ok |
| blas3 | trsm | f32 | 512x512x512 | 6.574 ms | 20.42 | 0.40 | 13% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.083 ms | 0.58 | 4.68 | 9% | ok |
| factor | lu | f64 | 512x512 | 32.813 ms | 2.73 | 0.13 | 3% | ok |
| factor | lu | f32 | 512x512 | 31.412 ms | 2.85 | 0.07 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 11.382 ms | 3.93 | 0.37 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 10.904 ms | 4.10 | 0.19 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 11.433 ms | 3.91 | 0.37 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 166.832 ms | 0.27 | 0.03 | 0% | ok |
| factor | qr | f64 | 512x512 | 645.997 ms | 0.28 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 68.522 ms | 2.61 | 0.06 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 752.540 ms | 0.59 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.000 ms | 0.26 | 1.05 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 93.824 ms | 2.86 | 0.04 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 34.158 ms | 2.62 | 0.12 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 37.521 ms | 0.60 | 0.03 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 127.218 ms | 1.19 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 512.542 ms | 0.29 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 54.014 ms | 3.11 | 0.02 | 4% | ok |
| eigen | schur | f64 | 256x256 | 109.018 ms | 3.85 | 0.01 | 5% | ok |
| eigen | svd | f64 | 256x256 | 118.388 ms | 3.12 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 26.294 ms | 9.57 | 0.04 | 12% | ok |
| eigen | sylvester | f64 | 256x256 | 259.941 ms | 3.87 | 0.01 | 5% | ok |
| elementwise | exp | f64 | n=8388608 | 27.353 ms | 0.31 | 4.91 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 42.478 ms | 0.20 | 3.16 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 39.362 ms | 0.21 | 3.41 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 21.114 ms | 0.40 | 6.36 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 31.732 ms | 0.26 | 4.23 | 8% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.851 ms | 3.08 | 32.00 | 61% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.324 ms | 1.98 | 20.57 | 39% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.177 ms | 6.78 | 14.25 | 27% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 18.036 ms | 0.18 | 0.64 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.815 ms | 1.11 | 9.17 | 17%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 99.470 ms | 0.11 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 29.388 ms | 0.38 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 16.270 ms | 0.69 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 54.953 ms | 0.41 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 74.370 ms | 0.30 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 26.666 ms | 0.84 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 84.782 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.675 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.691 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.445 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.474 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 33.757 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 31.894 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 28.291 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 13.784 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 18.055 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 15.877 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 32.118 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 29.495 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 17.583 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 19.592 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 19.357 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 19.720 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 22.163 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.036 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.403 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 23.132 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.833 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.761 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.665 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.736 ms | 1.51 | 16.89 | 32% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.501 ms | 1.74 | 19.54 | 37% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.818 ms | 1.44 | 16.13 | 31% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 16.074 ms | 2.39 | 18.08 | 34% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.399 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 47.014 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 55.463 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 6.453 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 1.840 ms | 9.12 | 36.46 | 69% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 15.320 ms | 17.52 | 0.27 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.476 ms | 35.22 | 37.42 | 71% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.202 ms | 13.96 | 55.84 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.463 ms | 36.24 | 72.47 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.347 ms | 48.35 | 48.35 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.318 ms | 52.79 | 52.79 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 14.928 ms | 17.98 | 0.21 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 17.341 ms | 15.48 | 0.12 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 17.723 ms | 15.15 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.711 ms | 72.34 | 0.42 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.688 ms | 72.78 | 0.43 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

