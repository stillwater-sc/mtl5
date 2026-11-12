## ryzen-9-8945hs -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 506.3 GFLOP/s, f32 1011.9 GFLOP/s |
| ... hardware, AVX-512F | f64 506.3, f32 1011.9 GFLOP/s |
| ... this build, scalar | f64 12.3, f32 14.5 GFLOP/s |
| bandwidth | read 52.7, update 52.9, triad 41.1 GB/s (ceiling 52.9) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 81% | 119% | 0 | 53% |
| blas2 | 10 | 100% | 117% | 0 | 17% |
| blas3 | 10 | 44% | 78% | 0 | 68% |
| factor | 12 | 2% | 6% | 0 | 12% |
| eigen | 8 | 2% | 5% | 0 | 11% |
| elementwise | 5 | 6% | 10% | 0 | 13% |
| sparse | 5 | 21% | 254% | 0 | 13% |
| direct | 7 | 0% | 0% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 30% | 32% | 0 | 13% |
| mg | 1 | 29% | 29% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 11% | 18% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 95% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.351 ms | 7.14 | 57.09 | 108% | ok | 6.1x (77%) |
| blas1 | dot | f32 | n=8388608 | 1.562 ms | 10.74 | 42.96 | 81% | ok | 7.9x (98%) |
| blas1 | axpy | f64 | n=8388608 | 3.521 ms | 4.76 | 57.18 | 108% | ok | 4.2x (53%) |
| blas1 | axpy | f32 | n=8388608 | 1.798 ms | 9.33 | 55.97 | 106% | ok | 7.1x (89%) |
| blas1 | scale | f64 | n=8388608 | 2.134 ms | 3.93 | 62.88 | 119% | ok | 1.2x (15%) |
| blas1 | two_norm | f64 | n=8388608 | 1.698 ms | 9.88 | 39.53 | 75% | ok | 9.1x (114%) |
| blas1 | one_norm | f64 | n=8388608 | 6.392 ms | 1.31 | 10.50 | 20% | ok | 1.2x (14%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.405 ms | 1.31 | 10.48 | 20% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 6.394 ms | 1.31 | 10.50 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 111.710 ms | 1.65 | 1.20 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 2.570 ms | 13.06 | 52.25 | 99% | ok | 1.3x (16%) |
| blas2 | gemv | f32 | n=4096 | 1.148 ms | 29.23 | 58.50 | 111% | ok | 1.4x (18%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.678 ms | 12.53 | 50.15 | 95% | ok | 1.3x (17%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.244 ms | 26.97 | 53.96 | 102% | ok | 1.3x (16%) |
| blas2 | symv | f64 | n=4096 | 1.273 ms | 26.36 | 52.77 | 100% | ok | 1.6x (20%) |
| blas2 | trmv | f64 | n=4096 | 1.237 ms | 13.56 | 54.29 | 103% | ok | 1.4x (17%) |
| blas2 | trmv | f32 | n=4096 | 0.540 ms | 31.04 | 62.14 | 117% | ok | 1.8x (22%) |
| blas2 | trsv | f64 | n=4096 | 1.820 ms | 9.22 | 36.91 | 70% | ok | 0.9x (12%) |
| blas2 | trsv | f32 | n=4096 | 0.932 ms | 18.00 | 36.03 | 68% | ok | 1.0x (12%) |
| blas2 | ger | f64 | n=4096 | 16.217 ms | 2.07 | 16.56 | 31% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 0.682 ms | 393.83 | 9.23 | 78% | ok | 6.7x (84%) |
| blas3 | gemm | f32 | 512x512x512 | 0.379 ms | 709.02 | 8.31 | 70% | ok | 6.3x (78%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.672 ms | 80.25 | 3.76 | 16% | ok | 2.0x (26%) |
| blas3 | symm | f64 | 512x512x512 | 0.684 ms | 392.62 | 9.20 | 78% | ok | 6.9x (87%) |
| blas3 | syrk | f64 | 512x512x512 | 2.007 ms | 66.88 | 2.09 | 13% | ok | 1.8x (23%) |
| blas3 | syr2k | f64 | 512x512x512 | 2.725 ms | 98.51 | 2.31 | 19% | ok | 2.2x (28%) |
| blas3 | trmm | f64 | 512x512x512 | 0.451 ms | 297.47 | 11.62 | 59% | ok | 5.2x (65%) |
| blas3 | trsm | f64 | 512x512x512 | 0.613 ms | 218.81 | 8.55 | 43% | ok | 5.7x (71%) |
| blas3 | trsm | f32 | 512x512x512 | 0.301 ms | 445.91 | 8.71 | 44% | ok | 5.4x (68%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 11.133 ms | 0.48 | 3.81 | 7% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 11.145 ms | 8.03 | 0.38 | 2% | ok | 0.6x (7%) |
| factor | lu | f32 | 512x512 | 5.082 ms | 17.61 | 0.41 | 2% | ok | 0.6x (8%) |
| factor | cholesky | f64 | 512x512 | 6.993 ms | 6.40 | 0.60 | 1% | ok | 0.8x (10%) |
| factor | cholesky | f32 | 512x512 | 2.442 ms | 18.32 | 0.86 | 2% | ok | 0.9x (11%) |
| factor | ldlt | f64 | 512x512 | 14.818 ms | 3.02 | 0.28 | 1% | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 211.419 ms | 0.21 | 0.02 | 0% | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 14.125 ms | 12.67 | 0.30 | 3% | ok | 0.9x (11%) |
| factor | lq | f64 | 512x512 | 54.165 ms | 3.30 | 0.08 | 1% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 573.414 ms | 0.78 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.269 ms | 0.83 | 3.30 | 6%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 82.207 ms | 3.27 | 0.05 | 1% | ok | 0.9x (12%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 8.818 ms | 10.15 | 0.48 | 2% | ok | 0.8x (10%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 4.261 ms | 5.25 | 0.25 | 1% | ok | 0.5x (6%) |
| eigen | eigen_symmetric | f64 | 256x256 | 164.039 ms | 0.92 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 619.276 ms | 0.24 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 25.142 ms | 6.67 | 0.04 | 1% | ok | 0.8x (10%) |
| eigen | schur | f64 | 256x256 | 31.143 ms | 13.47 | 0.03 | 3% | ok | 0.9x (11%) |
| eigen | svd | f64 | 256x256 | 13.775 ms | 26.79 | 0.08 | 5% | ok | 0.8x (10%) |
| eigen | expm | f64 | 256x256 | 21.439 ms | 11.74 | 0.05 | 2% | ok | 1.0x (12%) |
| eigen | sylvester | f64 | 256x256 | 101.372 ms | 9.93 | 0.02 | 2% | ok | 0.9x (11%) |
| elementwise | exp | f64 | n=8388608 | 33.422 ms | 0.25 | 4.02 | 8% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 53.304 ms | 0.16 | 2.52 | 5% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 47.965 ms | 0.17 | 2.80 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 26.549 ms | 0.32 | 5.06 | 10% | ok | 1.0x (12%) |
| elementwise | erf | f64 | n=8388608 | 42.414 ms | 0.20 | 3.16 | 6% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.202 ms | 12.94 | 134.60 | 254% ! | ok | 4.5x (56%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.480 ms | 1.06 | 10.98 | 21% | ok | 0.6x (7%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.351 ms | 17.81 | 37.45 | 71% | ok | 3.4x (42%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 24.035 ms | 0.14 | 0.48 | 1%\* | ok | 0.9x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.878 ms | 1.07 | 8.86 | 17%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 124.491 ms | 0.09 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 38.809 ms | 0.29 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 22.820 ms | 0.49 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 66.970 ms | 0.33 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 92.117 ms | 0.24 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 31.735 ms | 0.70 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 100.413 ms | 0.02 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.632 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 7.013 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.509 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.462 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 51.032 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 57.630 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 39.467 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.792 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 28.238 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.678 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 68.981 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 43.744 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.561 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 30.797 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 29.839 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 33.278 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 41.389 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.048 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.994 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 28.485 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 2.259 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 2.160 ms | 0.00 | 0.00 | n/a | ok | 1.2x (14%) |
| pc | ssor | f64 | 128^2 grid | 0.027 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.638 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.751 ms | 1.49 | 16.75 | 32% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.833 ms | 1.43 | 16.00 | 30% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.219 ms | 1.18 | 13.21 | 25% | ok | 1.0x (12%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.703 ms | 2.06 | 15.54 | 29% | ok | 1.0x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.472 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.255 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 70.565 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.892 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 14.380 ms | 1.17 | 4.67 | 9% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.877 ms | 55.04 | 0.86 | 11% | ok | 7.7x (96%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.836 ms | 9.14 | 9.71 | 18% | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.322 ms | 3.88 | 15.53 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.236 ms | 5.19 | 10.37 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.726 ms | 9.72 | 9.72 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.726 ms | 9.72 | 9.72 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.304 ms | 205.92 | 2.41 | n/a | ok | 7.9x (98%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.034 ms | 53.33 | 0.42 | n/a | ok | 7.7x (96%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.041 ms | 53.25 | 0.31 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.046 ms | 53.20 | 0.31 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.533 ms | 59.22 | 0.35 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

