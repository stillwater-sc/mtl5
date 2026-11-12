## ryzen-9-8945hs -- native, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 544.4 GFLOP/s, f32 1011.9 GFLOP/s |
| ... hardware, AVX-512F | f64 544.4, f32 1011.9 GFLOP/s |
| ... this build, scalar | f64 14.5, f32 14.4 GFLOP/s |
| bandwidth | read 52.3, update 53.2, triad 40.9 GB/s (ceiling 53.2) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 84% | 119% | 0 | 53% |
| blas2 | 10 | 61% | 107% | 0 | 13% |
| blas3 | 10 | 2% | 21% | 0 | 13% |
| factor | 12 | 1% | 6% | 0 | 18% |
| eigen | 8 | 0% | 2% | 0 | 13% |
| elementwise | 5 | 5% | 10% | 0 | 13% |
| sparse | 5 | 32% | 185% | 0 | 17% |
| direct | 7 | 0% | 0% | 0 | 12% |
| ordering | 4 | n/a | n/a | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 11% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 28% | 31% | 0 | 12% |
| mg | 1 | 29% | 29% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 11% | 19% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 83% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.315 ms | 7.25 | 57.99 | 109% | ok | 5.3x (66%) |
| blas1 | dot | f32 | n=8388608 | 1.565 ms | 10.72 | 42.87 | 81% | ok | 7.8x (98%) |
| blas1 | axpy | f64 | n=8388608 | 3.541 ms | 4.74 | 56.86 | 107% | ok | 4.2x (53%) |
| blas1 | axpy | f32 | n=8388608 | 1.955 ms | 8.58 | 51.50 | 97% | ok | 7.6x (95%) |
| blas1 | scale | f64 | n=8388608 | 2.115 ms | 3.97 | 63.47 | 119% | ok | 2.0x (25%) |
| blas1 | two_norm | f64 | n=8388608 | 1.501 ms | 11.18 | 44.72 | 84% | ok | 7.8x (98%) |
| blas1 | one_norm | f64 | n=8388608 | 6.389 ms | 1.31 | 10.50 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.406 ms | 1.31 | 10.48 | 20% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 6.393 ms | 1.31 | 10.50 | 20% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 109.711 ms | 1.68 | 1.22 | 2% | ok | 1.0x (12%) |
| blas2 | gemv | f64 | n=4096 | 2.385 ms | 14.07 | 56.31 | 106% | ok | 1.4x (18%) |
| blas2 | gemv | f32 | n=4096 | 1.181 ms | 28.40 | 56.83 | 107% | ok | 2.5x (31%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.362 ms | 14.21 | 56.85 | 107% | ok | 3.3x (41%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.182 ms | 28.39 | 56.80 | 107% | ok | 2.1x (27%) |
| blas2 | symv | f64 | n=4096 | 10.366 ms | 3.24 | 6.48 | 12% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 2.114 ms | 7.94 | 31.78 | 60% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=4096 | 1.823 ms | 9.20 | 18.42 | 35% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 2.076 ms | 8.08 | 32.35 | 61% | ok | 1.0x (12%) |
| blas2 | trsv | f32 | n=4096 | 1.869 ms | 8.98 | 17.97 | 34% | ok | 1.0x (12%) |
| blas2 | ger | f64 | n=4096 | 16.210 ms | 2.07 | 16.56 | 31% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 2.533 ms | 105.99 | 2.48 | 19% | ok | 7.9x (99%) |
| blas3 | gemm | f32 | 512x512x512 | 1.266 ms | 212.00 | 2.48 | 21% | ok | 7.9x (98%) |
| blas3 | gemmt | f64 | 512x512x512 | 13.912 ms | 9.65 | 0.45 | 2% | ok | 1.0x (12%) |
| blas3 | symm | f64 | 512x512x512 | 19.756 ms | 13.59 | 0.32 | 2% | ok | 1.1x (13%) |
| blas3 | syrk | f64 | 512x512x512 | 16.886 ms | 7.95 | 0.25 | 1% | ok | 1.0x (12%) |
| blas3 | syr2k | f64 | 512x512x512 | 28.757 ms | 9.33 | 0.22 | 2% | ok | 1.0x (13%) |
| blas3 | trmm | f64 | 512x512x512 | 31.264 ms | 4.29 | 0.17 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f64 | 512x512x512 | 37.072 ms | 3.62 | 0.14 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f32 | 512x512x512 | 33.483 ms | 4.01 | 0.08 | 0% | ok | 1.0x (12%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.884 ms | 0.49 | 3.90 | 7% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 27.621 ms | 3.24 | 0.15 | 1% | ok | 1.5x (18%) |
| factor | lu | f32 | 512x512 | 30.019 ms | 2.98 | 0.07 | 0% | ok | 1.3x (16%) |
| factor | cholesky | f64 | 512x512 | 14.367 ms | 3.11 | 0.29 | 1% | ok | 1.0x (12%) |
| factor | cholesky | f32 | 512x512 | 13.720 ms | 3.26 | 0.15 | 0% | ok | 1.0x (12%) |
| factor | ldlt | f64 | 512x512 | 14.416 ms | 3.10 | 0.29 | 1% | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 205.959 ms | 0.22 | 0.02 | 0% | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 500.573 ms | 0.36 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 54.157 ms | 3.30 | 0.08 | 1% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 576.677 ms | 0.78 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.267 ms | 0.83 | 3.31 | 6%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 98.837 ms | 2.72 | 0.04 | 0% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 29.059 ms | 3.08 | 0.14 | 1% | ok | 1.4x (18%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 49.481 ms | 0.45 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 163.409 ms | 0.92 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 641.051 ms | 0.24 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 256x256 | 75.625 ms | 2.22 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | schur | f64 | 256x256 | 156.657 ms | 2.68 | 0.01 | 0% | ok | 0.9x (12%) |
| eigen | svd | f64 | 256x256 | 176.201 ms | 2.09 | 0.01 | 0% | ok | 0.8x (11%) |
| eigen | expm | f64 | 256x256 | 23.813 ms | 10.57 | 0.04 | 2% | ok | 1.6x (20%) |
| eigen | sylvester | f64 | 256x256 | 324.020 ms | 3.11 | 0.00 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 50.395 ms | 0.17 | 2.66 | 5% | ok | 0.8x (10%) |
| elementwise | log | f64 | n=8388608 | 61.334 ms | 0.14 | 2.19 | 4% | ok | 0.9x (11%) |
| elementwise | sin | f64 | n=8388608 | 48.002 ms | 0.17 | 2.80 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 26.497 ms | 0.32 | 5.07 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 38.790 ms | 0.22 | 3.46 | 7% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.277 ms | 9.45 | 98.27 | 185% ! | ok | 3.4x (42%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.618 ms | 1.62 | 16.83 | 32% | ok | 1.3x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.355 ms | 17.78 | 37.38 | 70% | ok | 3.4x (42%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 30.168 ms | 0.11 | 0.38 | 1%\* | ok | 0.7x (9%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.946 ms | 1.03 | 8.55 | 16%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 135.235 ms | 0.08 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 40.139 ms | 0.28 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 23.098 ms | 0.48 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 67.811 ms | 0.33 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 92.133 ms | 0.24 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 33.135 ms | 0.68 | 0.02 | 0% | ok | 0.9x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 96.896 ms | 0.02 | 0.00 | 0% | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.619 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.994 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.500 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.897 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 49.792 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 55.816 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 38.615 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 23.293 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.942 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 26.083 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 75.592 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.956 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 29.295 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 34.724 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 33.654 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 32.076 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 39.219 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.045 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.980 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 28.990 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 2.243 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 2.194 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.026 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.811 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.793 ms | 1.46 | 16.36 | 31% | ok | 1.0x (12%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.942 ms | 1.35 | 15.10 | 28% | ok | 0.9x (12%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.270 ms | 1.15 | 12.92 | 24% | ok | 1.0x (12%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.979 ms | 2.03 | 15.31 | 29% | ok | 1.0x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.609 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 58.685 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 66.880 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.719 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 11.742 ms | 1.43 | 5.72 | 11% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.817 ms | 55.72 | 0.87 | 10% | ok | 7.6x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.721 ms | 9.75 | 10.36 | 19% | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.016 ms | 3.34 | 13.38 | n/a | ok | 0.9x (11%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.219 ms | 5.21 | 10.42 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.727 ms | 9.71 | 9.71 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.728 ms | 9.71 | 9.71 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.278 ms | 210.03 | 2.46 | n/a | ok | 7.7x (97%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.022 ms | 53.46 | 0.42 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.043 ms | 53.23 | 0.31 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.053 ms | 53.12 | 0.31 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 5.173 ms | 51.89 | 0.30 | n/a | ok | 6.7x (83%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

