## ryzen-7-8845hs -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 626.9 GFLOP/s, f32 1253.3 GFLOP/s |
| ... hardware, AVX-512F | f64 626.9, f32 1253.3 GFLOP/s |
| ... this build, scalar | f64 12.2, f32 14.4 GFLOP/s |
| bandwidth | read 53.3, update 53.3, triad 41.3 GB/s (ceiling 53.3) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 93% | 114% | 0 | 43% |
| blas2 | 10 | 100% | 107% | 0 | 16% |
| blas3 | 10 | 40% | 71% | 0 | 63% |
| factor | 12 | 2% | 8% | 0 | 11% |
| eigen | 8 | 2% | 5% | 0 | 11% |
| elementwise | 5 | 7% | 11% | 0 | 12% |
| sparse | 5 | 40% | 252% | 0 | 13% |
| direct | 7 | 0% | 0% | 0 | 12% |
| ordering | 4 | n/a | n/a | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 34% | 37% | 0 | 12% |
| mg | 1 | 35% | 35% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 11% | 22% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 87% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.317 ms | 7.24 | 57.91 | 109% | ok | 4.9x (62%) |
| blas1 | dot | f32 | n=8388608 | 1.357 ms | 12.36 | 49.44 | 93% | ok | 7.2x (90%) |
| blas1 | axpy | f64 | n=8388608 | 3.534 ms | 4.75 | 56.97 | 107% | ok | 3.4x (43%) |
| blas1 | axpy | f32 | n=8388608 | 1.997 ms | 8.40 | 50.42 | 95% | ok | 5.1x (64%) |
| blas1 | scale | f64 | n=8388608 | 2.203 ms | 3.81 | 60.92 | 114% | ok | 1.0x (13%) |
| blas1 | two_norm | f64 | n=8388608 | 1.461 ms | 11.48 | 45.92 | 86% | ok | 7.2x (91%) |
| blas1 | one_norm | f64 | n=8388608 | 5.061 ms | 1.66 | 13.26 | 25% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.076 ms | 1.65 | 13.22 | 25% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 5.061 ms | 1.66 | 13.26 | 25% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 87.462 ms | 2.11 | 1.53 | 3% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 2.576 ms | 13.03 | 52.13 | 98% | ok | 1.2x (16%) |
| blas2 | gemv | f32 | n=4096 | 1.174 ms | 28.58 | 57.18 | 107% | ok | 1.3x (17%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.667 ms | 12.58 | 50.36 | 94% | ok | 1.2x (15%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.257 ms | 26.70 | 53.42 | 100% | ok | 1.1x (14%) |
| blas2 | symv | f64 | n=4096 | 1.243 ms | 27.00 | 54.06 | 101% | ok | 1.5x (18%) |
| blas2 | trmv | f64 | n=4096 | 1.228 ms | 13.66 | 54.68 | 103% | ok | 1.3x (17%) |
| blas2 | trmv | f32 | n=4096 | 0.590 ms | 28.42 | 56.90 | 107% | ok | 1.5x (18%) |
| blas2 | trsv | f64 | n=4096 | 1.773 ms | 9.46 | 37.88 | 71% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.880 ms | 19.06 | 38.16 | 72% | ok | 1.1x (13%) |
| blas2 | ger | f64 | n=4096 | 13.716 ms | 2.45 | 19.58 | 37% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 0.606 ms | 442.82 | 10.38 | 71% | ok | 6.0x (75%) |
| blas3 | gemm | f32 | 512x512x512 | 0.337 ms | 797.02 | 9.34 | 64% | ok | 5.6x (70%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.358 ms | 98.83 | 4.63 | 16% | ok | 2.0x (25%) |
| blas3 | symm | f64 | 512x512x512 | 0.609 ms | 440.56 | 10.33 | 70% | ok | 6.2x (77%) |
| blas3 | syrk | f64 | 512x512x512 | 1.694 ms | 79.25 | 2.48 | 13% | ok | 1.7x (22%) |
| blas3 | syr2k | f64 | 512x512x512 | 2.385 ms | 112.53 | 2.64 | 18% | ok | 2.0x (25%) |
| blas3 | trmm | f64 | 512x512x512 | 0.389 ms | 345.30 | 13.49 | 55% | ok | 4.9x (61%) |
| blas3 | trsm | f64 | 512x512x512 | 0.534 ms | 251.20 | 9.81 | 40% | ok | 5.3x (66%) |
| blas3 | trsm | f32 | 512x512x512 | 0.265 ms | 507.25 | 9.91 | 40% | ok | 5.0x (63%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.413 ms | 0.56 | 4.51 | 8% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 9.206 ms | 9.72 | 0.46 | 2% | ok | 0.5x (7%) |
| factor | lu | f32 | 512x512 | 4.327 ms | 20.68 | 0.48 | 2% | ok | 0.6x (7%) |
| factor | cholesky | f64 | 512x512 | 6.006 ms | 7.45 | 0.70 | 1%\* | ok | 0.7x (9%) |
| factor | cholesky | f32 | 512x512 | 2.333 ms | 19.18 | 0.90 | 2%\* | ok | 0.7x (9%) |
| factor | ldlt | f64 | 512x512 | 11.812 ms | 3.79 | 0.36 | 1%\* | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 165.676 ms | 0.27 | 0.03 | 0%\* | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 11.927 ms | 15.00 | 0.35 | 2% | ok | 0.9x (11%) |
| factor | lq | f64 | 512x512 | 45.876 ms | 3.90 | 0.09 | 1% | ok | 1.4x (18%) |
| factor | hessenberg | f64 | 512x512 | 474.331 ms | 0.94 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.018 ms | 1.03 | 4.12 | 8%\* | ok | 3.8x (48%) |
| factor | inv | f64 | 512x512 | 66.451 ms | 4.04 | 0.06 | 1% | ok | 0.9x (11%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.328 ms | 12.21 | 0.57 | 2% | ok | 0.8x (9%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 3.598 ms | 6.22 | 0.29 | 1% | ok | 0.5x (6%) |
| eigen | eigen_symmetric | f64 | 256x256 | 128.680 ms | 1.17 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 503.438 ms | 0.30 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 21.106 ms | 7.95 | 0.05 | 1% | ok | 0.8x (10%) |
| eigen | schur | f64 | 256x256 | 26.400 ms | 15.89 | 0.04 | 3% | ok | 0.8x (10%) |
| eigen | svd | f64 | 256x256 | 12.154 ms | 30.37 | 0.09 | 5% | ok | 0.7x (9%) |
| eigen | expm | f64 | 256x256 | 18.149 ms | 13.87 | 0.06 | 2% | ok | 0.9x (11%) |
| eigen | sylvester | f64 | 256x256 | 81.732 ms | 12.32 | 0.02 | 2% | ok | 0.9x (11%) |
| elementwise | exp | f64 | n=8388608 | 28.661 ms | 0.29 | 4.68 | 9% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 44.373 ms | 0.19 | 3.02 | 6% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 40.050 ms | 0.21 | 3.35 | 6% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 21.928 ms | 0.38 | 6.12 | 11% | ok | 1.0x (12%) |
| elementwise | erf | f64 | n=8388608 | 35.602 ms | 0.24 | 3.77 | 7% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.202 ms | 12.93 | 134.47 | 252% ! | ok | 4.1x (52%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.283 ms | 2.04 | 21.22 | 40% | ok | 1.0x (12%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.281 ms | 18.36 | 38.60 | 72% | ok | 2.9x (36%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.841 ms | 0.19 | 0.68 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.706 ms | 1.18 | 9.76 | 18%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 101.796 ms | 0.11 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 30.910 ms | 0.36 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 18.414 ms | 0.61 | 0.04 | 0% | ok | 0.9x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 54.705 ms | 0.41 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 73.417 ms | 0.30 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 26.165 ms | 0.86 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 80.932 ms | 0.02 | 0.00 | 0%\* | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.731 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.635 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.428 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.197 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 40.191 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 45.928 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 31.257 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.233 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.332 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.938 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 55.251 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.581 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 20.216 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 24.296 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 23.598 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 26.977 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 32.847 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.037 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.415 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 25.076 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 1.841 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 128^2 grid | 1.748 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.022 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.786 ms | 0.00 | 0.00 | n/a | ok | 0.7x (9%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.607 ms | 1.63 | 18.25 | 34% | ok | 1.0x (12%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.491 ms | 1.76 | 19.67 | 37% | ok | 1.0x (12%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.763 ms | 1.48 | 16.64 | 31% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.463 ms | 2.49 | 18.80 | 35% | ok | 1.0x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 33.321 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 33.852 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 54.581 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.975 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 11.439 ms | 1.47 | 5.87 | 11% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.097 ms | 65.52 | 1.02 | 10% | ok | 7.2x (90%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.491 ms | 11.25 | 11.95 | 22% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.175 ms | 5.28 | 21.13 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.891 ms | 5.80 | 11.60 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.402 ms | 11.97 | 11.97 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.402 ms | 11.96 | 11.96 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.200 ms | 223.73 | 2.62 | n/a | ok | 6.9x (87%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.205 ms | 63.84 | 0.50 | n/a | ok | 7.2x (90%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.234 ms | 63.40 | 0.37 | n/a | ok | 7.2x (90%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.240 ms | 63.30 | 0.37 | n/a | ok | 7.2x (90%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.841 ms | 69.88 | 0.41 | n/a | ok | 7.1x (89%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

