## AMD Ryzen 7 2700X Eight-Core Processor -- native-fast, 16 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 2700X Eight-Core Processor (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.51.36256 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `e096dbbc6788` |
| FMA peak (ceiling) | f64 246.6 GFLOP/s, f32 494.3 GFLOP/s |
| ... hardware, AVX2+FMA | f64 243.3, f32 488.7 GFLOP/s |
| ... this build, SSE2 | f64 246.6, f32 494.3 GFLOP/s |
| bandwidth | read 25.4, update 26.2, triad 20.9 GB/s (ceiling 26.2) |
| last-level cache | 8 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 112% | 118% | 0 | 8% |
| blas2 | 6 | 81% | 116% | 0 | 6% |
| blas3 | 10 | 5% | 23% | 0 | 10% |
| factor | 12 | 1% | 24% | 0 | 10% |
| eigen | 8 | 1% | 3% | 0 | 7% |
| elementwise | 5 | 9% | 19% | 0 | 6% |
| sparse | 5 | 33% | 107% | 0 | 7% |
| direct | 7 | 0% | 0% | 0 | 6% |
| ordering | 4 | n/a | n/a | 0 | 6% |
| krylov | 13 | n/a | n/a | 0 | 6% |
| pc | 7 | n/a | n/a | 0 | 7% |
| smoother | 3 | 49% | 55% | 0 | 6% |
| mg | 1 | 46% | 46% | 0 | 6% |
| sparse_eig | 4 | n/a | n/a | 0 | 6% |
| mixed | 3 | 76% | 77% | 0 | 6% |
| integer | 9 | n/a | n/a | 0 | 18% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.357 ms | 3.85 | 30.81 | 117% | ok | 1.2x (8%) |
| blas1 | dot | f32 | n=8388608 | 2.250 ms | 7.46 | 29.83 | 114% | ok | 1.1x (7%) |
| blas1 | axpy | f64 | n=8388608 | 6.836 ms | 2.45 | 29.45 | 112% | ok | 1.3x (8%) |
| blas1 | axpy | f32 | n=8388608 | 3.264 ms | 5.14 | 30.84 | 118% | ok | 1.4x (9%) |
| blas1 | scale | f64 | n=8388608 | 4.458 ms | 1.88 | 30.11 | 115% | ok | 1.5x (10%) |
| blas1 | two_norm | f64 | n=8388608 | 2.295 ms | 7.31 | 29.24 | 111% | ok | 1.2x (8%) |
| blas1 | one_norm | f64 | n=8388608 | 6.608 ms | 1.27 | 10.16 | 39% | ok | 1.0x (6%) |
| blas1 | infinity_norm | f64 | n=8388608 | 14.559 ms | 0.58 | 4.61 | 18% | ok | 1.5x (9%) |
| blas1 | sum | f64 | n=8388608 | 6.648 ms | 1.26 | 10.09 | 38% | ok | 1.0x (6%) |
| blas1 | dot_k3 | f64 | n=8388608 | 117.447 ms | 1.57 | 1.14 | 4% | ok | 1.0x (6%) |
| blas2 | gemv | f64 | n=4096 | 4.405 ms | 7.62 | 30.48 | 116% | ok | 1.2x (8%) |
| blas2 | gemv | f32 | n=4096 | 2.266 ms | 14.81 | 29.63 | 113% | ok | 1.2x (8%) |
| blas2 | symv | f64 | n=4096 | 10.024 ms | 3.35 | 6.70 | 26% | ok | 1.0x (6%) |
| blas2 | trmv | f64 | n=4096 | 3.159 ms | 5.31 | 21.26 | 81% | ok | 0.9x (6%) |
| blas2 | trsv | f64 | n=4096 | 3.365 ms | 4.99 | 19.96 | 76% | ok | 0.9x (6%) |
| blas2 | ger | f64 | n=4096 | 24.890 ms | 1.35 | 10.79 | 41% | ok | 1.0x (6%) |
| blas3 | gemm | f64 | 512x512x512 | 4.760 ms | 56.40 | 1.32 | 23% | ok | 4.5x (28%) |
| blas3 | gemm | f32 | 512x512x512 | 3.837 ms | 69.97 | 0.82 | 14% | ok | 2.6x (16%) |
| blas3 | gemmt | f64 | 512x512x512 | 14.098 ms | 9.52 | 0.45 | 4% | ok | 1.0x (6%) |
| blas3 | symm | f64 | 512x512x512 | 5.832 ms | 46.02 | 1.08 | 19% | ok | 3.8x (24%) |
| blas3 | syrk | f64 | 512x512x512 | 18.837 ms | 7.13 | 0.22 | 3% | ok | 1.0x (6%) |
| blas3 | syr2k | f64 | 512x512x512 | 28.438 ms | 9.44 | 0.22 | 4% | ok | 1.1x (7%) |
| blas3 | trmm | f64 | 512x512x512 | 11.427 ms | 11.75 | 0.46 | 5% | ok | 1.6x (10%) |
| blas3 | trsm | f64 | 512x512x512 | 11.385 ms | 11.79 | 0.46 | 5% | ok | 1.9x (12%) |
| blas3 | trsm | f32 | 512x512x512 | 8.993 ms | 14.93 | 0.29 | 3% | ok | 1.3x (8%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 16.964 ms | 0.31 | 2.50 | 10% | ok | 1.0x (6%) |
| factor | lu | f64 | 512x512 | 49.409 ms | 1.81 | 0.08 | 1% | ok | 3.0x (19%) |
| factor | lu | f32 | 512x512 | 82.944 ms | 1.08 | 0.03 | 0% | ok | 0.6x (4%) |
| factor | cholesky | f64 | 512x512 | 16.451 ms | 2.72 | 0.25 | 1% | ok | 1.0x (6%) |
| factor | cholesky | f32 | 512x512 | 16.409 ms | 2.73 | 0.13 | 1% | ok | 1.0x (6%) |
| factor | ldlt | f64 | 512x512 | 15.829 ms | 2.83 | 0.26 | 1% | ok | 1.0x (6%) |
| factor | ldlt_bk | f64 | 512x512 | 160.108 ms | 0.28 | 0.03 | 0% | ok | 1.0x (6%) |
| factor | qr | f64 | 512x512 | 312.080 ms | 0.57 | 0.01 | 0% | ok | 1.6x (10%) |
| factor | lq | f64 | 512x512 | 109.561 ms | 1.63 | 0.04 | 1% | ok | 1.6x (10%) |
| factor | hessenberg | f64 | 512x512 | 434.708 ms | 1.03 | 0.01 | 0% | ok | 1.6x (10%) |
| factor | householder | f64 | 512x512 | 0.673 ms | 1.56 | 6.24 | 24%\* | ok | 4.9x (31%) |
| factor | inv | f64 | 512x512 | 159.895 ms | 1.68 | 0.03 | 1% | ok | 1.3x (8%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 53.593 ms | 1.67 | 0.08 | 1% | ok | 2.9x (18%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 43.571 ms | 0.51 | 0.02 | 0% | ok | 0.9x (6%) |
| eigen | eigen_symmetric | f64 | 256x256 | 119.587 ms | 1.26 | 0.01 | 1% | ok | 1.0x (6%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 584.781 ms | 0.26 | 0.00 | 0% | ok | 1.0x (7%) |
| eigen | eigenvalue | f64 | 256x256 | 68.048 ms | 2.47 | 0.02 | 1% | ok | 1.0x (6%) |
| eigen | schur | f64 | 256x256 | 185.943 ms | 2.26 | 0.01 | 1% | ok | 1.2x (7%) |
| eigen | svd | f64 | 256x256 | 174.965 ms | 2.11 | 0.01 | 1% | ok | 1.0x (7%) |
| eigen | expm | f64 | 256x256 | 37.893 ms | 6.64 | 0.03 | 3% | ok | 1.3x (8%) |
| eigen | sylvester | f64 | 256x256 | 390.383 ms | 2.58 | 0.00 | 1% | ok | 1.0x (6%) |
| elementwise | exp | f64 | n=8388608 | 58.003 ms | 0.14 | 2.31 | 9% | ok | 1.1x (7%) |
| elementwise | log | f64 | n=8388608 | 74.233 ms | 0.11 | 1.81 | 7% | ok | 0.9x (6%) |
| elementwise | sin | f64 | n=8388608 | 58.646 ms | 0.14 | 2.29 | 9% | ok | 1.0x (6%) |
| elementwise | sqrt | f64 | n=8388608 | 27.444 ms | 0.31 | 4.89 | 19% | ok | 1.0x (7%) |
| elementwise | erf | f64 | n=8388608 | 62.164 ms | 0.13 | 2.16 | 8% | ok | 1.0x (6%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.967 ms | 2.71 | 28.17 | 107% | ok | 1.3x (8%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.138 ms | 0.83 | 8.68 | 33% | ok | 1.0x (6%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.519 ms | 9.27 | 19.48 | 74% | ok | 1.7x (10%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 19.535 ms | 0.17 | 0.59 | 2% | ok | 1.1x (7%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 3.252 ms | 0.62 | 5.12 | 20% | ok | 0.9x (6%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 166.490 ms | 0.07 | 0.00 | 0% | ok | 1.0x (6%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 49.649 ms | 0.23 | 0.01 | 0% | ok | 1.1x (7%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 28.916 ms | 0.39 | 0.03 | 0% | ok | 1.1x (7%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 107.379 ms | 0.21 | 0.01 | 0% | ok | 0.9x (6%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 131.631 ms | 0.17 | 0.01 | 0% | ok | 1.0x (6%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 51.293 ms | 0.44 | 0.01 | 0% | ok | 1.0x (6%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 388.801 ms | 0.00 | 0.00 | 0% | ok | 1.0x (6%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 6.058 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 9.506 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.856 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 26.683 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 43.101 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 51.286 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 36.871 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.931 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 25.154 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 21.456 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 47.519 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 43.657 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 26.641 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 34.188 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 37.655 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 53.020 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 47.587 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.061 ms | 0.00 | 0.00 | n/a | ok | 1.2x (8%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.770 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| pc | ilut | f64 | 128^2 grid | 43.149 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| pc | ic_0 | f64 | 128^2 grid | 3.839 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| pc | ildl | f64 | 128^2 grid | 3.596 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| pc | ssor | f64 | 128^2 grid | 0.029 ms | 0.00 | 0.00 | n/a | ok | 1.3x (8%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.118 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.286 ms | 1.15 | 12.83 | 49% | ok | 1.0x (6%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.034 ms | 1.29 | 14.42 | 55% | ok | 1.0x (6%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.442 ms | 1.07 | 12.01 | 46% | ok | 1.0x (6%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 24.290 ms | 1.58 | 11.97 | 46% | ok | 1.0x (6%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 48.740 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 74.842 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 127.283 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.789 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| mixed | dot | f32>f64 | n=8388608 | 3.302 ms | 5.08 | 20.32 | 77% | ok | 1.0x (6%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.258 ms | 63.04 | 0.99 | 26% | ok | 6.0x (38%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.893 ms | 18.79 | 19.97 | 76% | ok | 1.0x (6%) |
| integer | dot | i32 | n=8388608, |v| <= 15 | 3.023 ms | 5.55 | 22.20 | n/a | ok | 0.9x (6%) |
| integer | dot | i16>i32 | n=8388608, |v| <= 15 | 1.229 ms | 13.65 | 27.31 | n/a | ok | 1.0x (6%) |
| integer | dot | i8>i32 | n=8388608, |v| <= 15 | 0.885 ms | 18.95 | 18.95 | n/a | ok | 1.0x (6%) |
| integer | dot | u8i8>i32 | n=8388608, |v| <= 15 | 0.757 ms | 22.18 | 22.18 | n/a | ok | 0.9x (6%) |
| integer | gemm | i32 | 512^3, |v| <= 2047 | 4.165 ms | 64.44 | 0.76 | n/a | ok | 6.6x (41%) |
| integer | gemm | i16>i32 | 512^3, |v| <= 2047 | 6.555 ms | 40.95 | 0.32 | n/a | ok | 4.2x (26%) |
| integer | gemm | i8>i32 | 512^3, |v| <= 127 | 4.432 ms | 60.56 | 0.35 | n/a | ok | 6.6x (41%) |
| integer | gemm_quad | i8>i32 | 512^3, |v| <= 127 | 2.109 ms | 127.31 | 0.75 | n/a | ok | 3.6x (23%) |
| integer | gemm_quad | u8i8>i32 | 512^3, |v| <= 127 | 2.294 ms | 117.03 | 0.69 | n/a | ok | 2.9x (18%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

