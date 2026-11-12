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

