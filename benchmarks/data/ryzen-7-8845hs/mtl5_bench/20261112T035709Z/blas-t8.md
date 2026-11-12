## ryzen-7-8845hs -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 625.3 GFLOP/s, f32 1251.4 GFLOP/s |
| ... hardware, AVX-512F | f64 625.3, f32 1251.4 GFLOP/s |
| ... this build, scalar | f64 12.2, f32 14.4 GFLOP/s |
| bandwidth | read 53.1, update 53.3, triad 41.2 GB/s (ceiling 53.3) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 93% | 121% | 0 | 42% |
| blas2 | 10 | 101% | 119% | 0 | 16% |
| blas3 | 10 | 40% | 71% | 0 | 63% |
| factor | 12 | 2% | 8% | 0 | 11% |
| eigen | 8 | 2% | 5% | 0 | 11% |
| elementwise | 5 | 7% | 12% | 0 | 12% |
| sparse | 5 | 34% | 257% | 0 | 13% |
| direct | 7 | 0% | 0% | 0 | 12% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 37% | 38% | 0 | 13% |
| mg | 1 | 35% | 35% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 11% | 22% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 82% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.309 ms | 7.27 | 58.12 | 109% | ok | 5.0x (62%) |
| blas1 | dot | f32 | n=8388608 | 1.359 ms | 12.35 | 49.38 | 93% | ok | 7.6x (95%) |
| blas1 | axpy | f64 | n=8388608 | 3.519 ms | 4.77 | 57.20 | 107% | ok | 3.4x (42%) |
| blas1 | axpy | f32 | n=8388608 | 1.776 ms | 9.44 | 56.67 | 106% | ok | 5.8x (72%) |
| blas1 | scale | f64 | n=8388608 | 2.084 ms | 4.02 | 64.40 | 121% | ok | 1.1x (14%) |
| blas1 | two_norm | f64 | n=8388608 | 1.518 ms | 11.05 | 44.20 | 83% | ok | 7.3x (91%) |
| blas1 | one_norm | f64 | n=8388608 | 5.069 ms | 1.65 | 13.24 | 25% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.100 ms | 1.64 | 13.16 | 25% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 5.079 ms | 1.65 | 13.21 | 25% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 86.046 ms | 2.14 | 1.56 | 3% | ok | 1.0x (12%) |
| blas2 | gemv | f64 | n=4096 | 2.550 ms | 13.16 | 52.67 | 99% | ok | 1.3x (16%) |
| blas2 | gemv | f32 | n=4096 | 1.156 ms | 29.02 | 58.07 | 109% | ok | 1.4x (17%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.695 ms | 12.45 | 49.82 | 93% | ok | 1.3x (16%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.251 ms | 26.82 | 53.67 | 101% | ok | 1.2x (15%) |
| blas2 | symv | f64 | n=4096 | 1.215 ms | 27.61 | 55.28 | 104% | ok | 1.5x (19%) |
| blas2 | trmv | f64 | n=4096 | 1.218 ms | 13.77 | 55.15 | 103% | ok | 1.3x (17%) |
| blas2 | trmv | f32 | n=4096 | 0.530 ms | 31.67 | 63.41 | 119% | ok | 1.6x (20%) |
| blas2 | trsv | f64 | n=4096 | 1.589 ms | 10.56 | 42.28 | 79% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.946 ms | 17.74 | 35.52 | 67% | ok | 1.0x (12%) |
| blas2 | ger | f64 | n=4096 | 13.670 ms | 2.45 | 19.64 | 37% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 0.605 ms | 443.84 | 10.40 | 71% | ok | 6.0x (75%) |
| blas3 | gemm | f32 | 512x512x512 | 0.340 ms | 790.68 | 9.27 | 63% | ok | 5.6x (70%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.356 ms | 98.97 | 4.64 | 16% | ok | 2.0x (25%) |
| blas3 | symm | f64 | 512x512x512 | 0.607 ms | 442.38 | 10.37 | 71% | ok | 6.3x (78%) |
| blas3 | syrk | f64 | 512x512x512 | 1.720 ms | 78.03 | 2.44 | 12% | ok | 1.8x (22%) |
| blas3 | syr2k | f64 | 512x512x512 | 2.387 ms | 112.48 | 2.64 | 18% | ok | 2.1x (26%) |
| blas3 | trmm | f64 | 512x512x512 | 0.398 ms | 337.23 | 13.17 | 54% | ok | 4.7x (59%) |
| blas3 | trsm | f64 | 512x512x512 | 0.557 ms | 241.05 | 9.42 | 39% | ok | 5.1x (64%) |
| blas3 | trsm | f32 | 512x512x512 | 0.266 ms | 503.82 | 9.84 | 40% | ok | 5.1x (63%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.245 ms | 0.57 | 4.59 | 9% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 9.104 ms | 9.83 | 0.46 | 2% | ok | 0.5x (7%) |
| factor | lu | f32 | 512x512 | 4.345 ms | 20.59 | 0.48 | 2% | ok | 0.6x (7%) |
| factor | cholesky | f64 | 512x512 | 5.960 ms | 7.51 | 0.70 | 1%\* | ok | 0.7x (9%) |
| factor | cholesky | f32 | 512x512 | 2.308 ms | 19.39 | 0.91 | 2%\* | ok | 0.7x (9%) |
| factor | ldlt | f64 | 512x512 | 11.867 ms | 3.77 | 0.35 | 1%\* | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 167.173 ms | 0.27 | 0.03 | 0%\* | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 11.893 ms | 15.05 | 0.35 | 2% | ok | 0.9x (11%) |
| factor | lq | f64 | 512x512 | 45.484 ms | 3.93 | 0.09 | 1% | ok | 1.4x (18%) |
| factor | hessenberg | f64 | 512x512 | 477.079 ms | 0.94 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.004 ms | 1.05 | 4.18 | 8%\* | ok | 3.9x (48%) |
| factor | inv | f64 | 512x512 | 65.954 ms | 4.07 | 0.06 | 1% | ok | 0.9x (11%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.245 ms | 12.35 | 0.58 | 2% | ok | 0.8x (10%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 3.588 ms | 6.23 | 0.29 | 1% | ok | 0.5x (6%) |
| eigen | eigen_symmetric | f64 | 256x256 | 133.014 ms | 1.14 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 519.779 ms | 0.29 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 256x256 | 21.078 ms | 7.96 | 0.05 | 1% | ok | 0.8x (10%) |
| eigen | schur | f64 | 256x256 | 26.127 ms | 16.05 | 0.04 | 3% | ok | 0.8x (10%) |
| eigen | svd | f64 | 256x256 | 12.133 ms | 30.42 | 0.09 | 5% | ok | 0.7x (9%) |
| eigen | expm | f64 | 256x256 | 18.256 ms | 13.78 | 0.06 | 2% | ok | 0.9x (11%) |
| eigen | sylvester | f64 | 256x256 | 80.883 ms | 12.45 | 0.02 | 2% | ok | 0.9x (11%) |
| elementwise | exp | f64 | n=8388608 | 28.560 ms | 0.29 | 4.70 | 9% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 43.937 ms | 0.19 | 3.05 | 6% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 39.470 ms | 0.21 | 3.40 | 6% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 21.387 ms | 0.39 | 6.28 | 12% | ok | 1.0x (12%) |
| elementwise | erf | f64 | n=8388608 | 34.953 ms | 0.24 | 3.84 | 7% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.198 ms | 13.19 | 137.25 | 257% ! | ok | 12.2x (153%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.516 ms | 1.73 | 17.96 | 34% | ok | 0.9x (11%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.320 ms | 18.05 | 37.96 | 71% | ok | 2.8x (36%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 18.291 ms | 0.18 | 0.63 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.705 ms | 1.18 | 9.77 | 18%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 101.545 ms | 0.11 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 31.941 ms | 0.35 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 18.977 ms | 0.59 | 0.04 | 0% | ok | 0.9x (11%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 54.848 ms | 0.41 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 74.254 ms | 0.30 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 26.133 ms | 0.86 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 80.579 ms | 0.02 | 0.00 | 0%\* | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.723 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.692 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.422 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.111 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 40.303 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 46.739 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 31.310 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.299 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.467 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.965 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 54.976 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.869 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 20.462 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 24.452 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 23.691 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 26.532 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 33.729 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.036 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.535 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| pc | ilut | f64 | 128^2 grid | 22.782 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 1.870 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 128^2 grid | 1.773 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.022 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.329 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.461 ms | 1.79 | 20.07 | 38% | ok | 1.1x (14%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.499 ms | 1.75 | 19.57 | 37% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.764 ms | 1.48 | 16.63 | 31% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.486 ms | 2.48 | 18.77 | 35% | ok | 1.0x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.459 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 33.106 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 58.023 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.046 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 11.440 ms | 1.47 | 5.87 | 11% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 5.023 ms | 53.44 | 0.84 | 9% | ok | 5.9x (73%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.506 ms | 11.14 | 11.83 | 22% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.136 ms | 5.35 | 21.40 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.893 ms | 5.80 | 11.60 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.408 ms | 11.91 | 11.91 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.409 ms | 11.90 | 11.90 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.281 ms | 209.50 | 2.46 | n/a | ok | 6.6x (82%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.212 ms | 63.73 | 0.50 | n/a | ok | 7.2x (90%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.241 ms | 63.29 | 0.37 | n/a | ok | 7.2x (90%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.240 ms | 63.32 | 0.37 | n/a | ok | 7.2x (90%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.859 ms | 69.55 | 0.41 | n/a | ok | 7.1x (88%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

