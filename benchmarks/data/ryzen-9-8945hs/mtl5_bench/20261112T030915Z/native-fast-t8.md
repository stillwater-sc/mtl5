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

