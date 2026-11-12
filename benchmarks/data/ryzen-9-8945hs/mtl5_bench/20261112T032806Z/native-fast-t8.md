## ryzen-9-8945hs -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 505.9 GFLOP/s, f32 1011.7 GFLOP/s |
| ... hardware, AVX-512F | f64 505.9, f32 1011.7 GFLOP/s |
| ... this build, SSE2 | f64 317.4, f32 633.8 GFLOP/s |
| bandwidth | read 52.7, update 53.0, triad 40.9 GB/s (ceiling 53.0) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 110% | 125% | 0 | 13% |
| blas2 | 10 | 62% | 118% | 0 | 16% |
| blas3 | 10 | 5% | 19% | 0 | 31% |
| factor | 12 | 1% | 6% | 0 | 19% |
| eigen | 8 | 0% | 2% | 0 | 13% |
| elementwise | 5 | 6% | 10% | 0 | 12% |
| sparse | 5 | 36% | 241% | 0 | 13% |
| direct | 7 | 0% | 0% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 29% | 30% | 0 | 12% |
| mg | 1 | 29% | 29% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 58% | 58% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 73% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.292 ms | 7.32 | 58.56 | 110% | ok | 1.1x (14%) |
| blas1 | dot | f32 | n=8388608 | 1.009 ms | 16.63 | 66.51 | 125% | ok | 1.2x (16%) |
| blas1 | axpy | f64 | n=8388608 | 3.519 ms | 4.77 | 57.21 | 108% | ok | 1.0x (13%) |
| blas1 | axpy | f32 | n=8388608 | 1.671 ms | 10.04 | 60.25 | 114% | ok | 1.0x (13%) |
| blas1 | scale | f64 | n=8388608 | 2.111 ms | 3.97 | 63.60 | 120% | ok | 0.9x (12%) |
| blas1 | two_norm | f64 | n=8388608 | 1.061 ms | 15.81 | 63.24 | 119% | ok | 1.1x (14%) |
| blas1 | one_norm | f64 | n=8388608 | 6.417 ms | 1.31 | 10.46 | 20% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.401 ms | 1.31 | 10.48 | 20% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 6.394 ms | 1.31 | 10.50 | 20% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 105.179 ms | 1.75 | 1.28 | 2% | ok | 1.0x (12%) |
| blas2 | gemv | f64 | n=4096 | 2.335 ms | 14.37 | 57.50 | 108% | ok | 1.5x (19%) |
| blas2 | gemv | f32 | n=4096 | 1.071 ms | 31.33 | 62.70 | 118% | ok | 1.7x (21%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 10.106 ms | 3.32 | 13.29 | 25% | ok | 2.2x (28%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.084 ms | 6.60 | 13.21 | 25% | ok | 1.6x (20%) |
| blas2 | symv | f64 | n=4096 | 10.181 ms | 3.30 | 6.60 | 12% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 1.977 ms | 8.48 | 33.97 | 64% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=4096 | 1.167 ms | 14.37 | 28.78 | 54% | ok | 1.1x (13%) |
| blas2 | trsv | f64 | n=4096 | 1.960 ms | 8.56 | 34.26 | 65% | ok | 1.2x (15%) |
| blas2 | trsv | f32 | n=4096 | 1.024 ms | 16.39 | 32.81 | 62% | ok | 1.2x (15%) |
| blas2 | ger | f64 | n=4096 | 16.364 ms | 2.05 | 16.41 | 31% | ok | 1.2x (16%) |
| blas3 | gemm | f64 | 512x512x512 | 2.732 ms | 98.27 | 2.30 | 19% | ok | 7.9x (99%) |
| blas3 | gemm | f32 | 512x512x512 | 1.553 ms | 172.86 | 2.03 | 17% | ok | 7.0x (87%) |
| blas3 | gemmt | f64 | 512x512x512 | 5.625 ms | 23.86 | 1.12 | 5% | ok | 2.5x (31%) |
| blas3 | symm | f64 | 512x512x512 | 3.886 ms | 69.07 | 1.62 | 14% | ok | 5.8x (73%) |
| blas3 | syrk | f64 | 512x512x512 | 17.038 ms | 7.88 | 0.25 | 2% | ok | 1.3x (16%) |
| blas3 | syr2k | f64 | 512x512x512 | 30.512 ms | 8.80 | 0.21 | 2% | ok | 1.1x (14%) |
| blas3 | trmm | f64 | 512x512x512 | 6.934 ms | 19.36 | 0.76 | 4% | ok | 2.5x (31%) |
| blas3 | trsm | f64 | 512x512x512 | 7.357 ms | 18.24 | 0.71 | 4% | ok | 2.0x (25%) |
| blas3 | trsm | f32 | 512x512x512 | 5.496 ms | 24.42 | 0.48 | 2% | ok | 2.1x (27%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 11.229 ms | 0.47 | 3.78 | 7% | ok | 1.3x (16%) |
| factor | lu | f64 | 512x512 | 27.520 ms | 3.25 | 0.15 | 1% | ok | 2.3x (28%) |
| factor | lu | f32 | 512x512 | 26.059 ms | 3.43 | 0.08 | 0% | ok | 2.3x (29%) |
| factor | cholesky | f64 | 512x512 | 14.292 ms | 3.13 | 0.29 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 13.720 ms | 3.26 | 0.15 | 0% | ok | 1.0x (12%) |
| factor | ldlt | f64 | 512x512 | 14.397 ms | 3.11 | 0.29 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 211.274 ms | 0.21 | 0.02 | 0% | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 498.148 ms | 0.36 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 52.735 ms | 3.39 | 0.08 | 1% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 576.798 ms | 0.78 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.262 ms | 0.83 | 3.32 | 6%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 98.454 ms | 2.73 | 0.04 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 29.304 ms | 3.05 | 0.14 | 1% | ok | 1.4x (18%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 47.850 ms | 0.47 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 159.964 ms | 0.94 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 634.102 ms | 0.24 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 74.892 ms | 2.24 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 143.368 ms | 2.93 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 149.733 ms | 2.47 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 23.283 ms | 10.81 | 0.05 | 2% | ok | 1.4x (18%) |
| eigen | sylvester | f64 | 256x256 | 326.885 ms | 3.08 | 0.00 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 33.551 ms | 0.25 | 4.00 | 8% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 51.693 ms | 0.16 | 2.60 | 5% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 48.948 ms | 0.17 | 2.74 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 26.453 ms | 0.32 | 5.07 | 10% | ok | 1.0x (12%) |
| elementwise | erf | f64 | n=8388608 | 39.016 ms | 0.22 | 3.44 | 6% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.213 ms | 12.29 | 127.84 | 241% ! | ok | 4.3x (54%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.441 ms | 1.82 | 18.90 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.293 ms | 18.27 | 38.41 | 72% | ok | 3.5x (43%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 24.368 ms | 0.13 | 0.47 | 1%\* | ok | 0.9x (11%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.950 ms | 1.03 | 8.54 | 16%\* | ok | 1.0x (12%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 126.374 ms | 0.09 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 35.577 ms | 0.31 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 18.848 ms | 0.59 | 0.04 | 0% | ok | 1.1x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 66.367 ms | 0.34 | 0.01 | 0% | ok | 1.2x (15%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 89.705 ms | 0.25 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 31.939 ms | 0.70 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 104.711 ms | 0.02 | 0.00 | 0% | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.627 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.960 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.506 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 18.409 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 41.242 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 39.075 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 34.667 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.010 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.618 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 19.880 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 39.030 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 36.698 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 21.926 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 24.490 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 23.294 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 24.740 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 28.077 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.048 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.974 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 29.144 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 2.275 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 2.189 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.027 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.813 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.828 ms | 1.43 | 16.04 | 30% | ok | 1.0x (12%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.895 ms | 1.38 | 15.48 | 29% | ok | 1.0x (12%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.219 ms | 1.18 | 13.21 | 25% | ok | 1.3x (16%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.781 ms | 2.05 | 15.48 | 29% | ok | 1.0x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.667 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 58.633 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 71.690 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.076 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 2.173 ms | 7.72 | 30.89 | 58% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 3.154 ms | 85.10 | 1.33 | 17% | ok | 6.1x (76%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.576 ms | 29.14 | 30.96 | 58% | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.453 ms | 11.55 | 46.19 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.496 ms | 33.84 | 67.68 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.429 ms | 39.14 | 39.14 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.384 ms | 43.69 | 43.69 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.880 ms | 93.20 | 1.09 | n/a | ok | 6.6x (82%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 3.039 ms | 88.33 | 0.69 | n/a | ok | 7.2x (90%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 3.063 ms | 87.63 | 0.51 | n/a | ok | 7.3x (92%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.787 ms | 340.87 | 2.00 | n/a | ok | 5.8x (73%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.783 ms | 342.65 | 2.01 | n/a | ok | 5.8x (73%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

