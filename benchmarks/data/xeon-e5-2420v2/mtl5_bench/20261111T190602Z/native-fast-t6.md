## xeon-e5-2420v2 -- native-fast, 6 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 59.1 GFLOP/s, f32 118.2 GFLOP/s |
| ... hardware, SSE2 | f64 59.1, f32 118.2 GFLOP/s |
| ... this build, SSE4 | f64 59.0, f32 118.1 GFLOP/s |
| bandwidth | read 11.4, update 10.7, triad 8.5 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 98% | 108% | 0 | 17% |
| blas2 | 10 | 83% | 108% | 0 | 17% |
| blas3 | 10 | 28% | 78% | 0 | 36% |
| factor | 12 | 3% | 38% | 0 | 20% |
| eigen | 8 | 3% | 13% | 0 | 17% |
| elementwise | 5 | 8% | 12% | 0 | 17% |
| sparse | 5 | 55% | 103% | 0 | 17% |
| direct | 7 | 0% | 1% | 0 | 17% |
| ordering | 4 | n/a | n/a | 0 | 16% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 73% | 82% | 0 | 16% |
| mg | 1 | 73% | 73% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 69% | 81% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 79% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 10.917 ms | 1.54 | 12.29 | 108% | ok | 1.1x (18%) |
| blas1 | dot | f32 | n=8388608 | 5.469 ms | 3.07 | 12.27 | 108% | ok | 1.1x (18%) |
| blas1 | axpy | f64 | n=8388608 | 17.918 ms | 0.94 | 11.24 | 98% | ok | 1.0x (17%) |
| blas1 | axpy | f32 | n=8388608 | 8.505 ms | 1.97 | 11.84 | 104% | ok | 1.1x (18%) |
| blas1 | scale | f64 | n=8388608 | 12.184 ms | 0.69 | 11.02 | 97% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 5.455 ms | 3.08 | 12.30 | 108% | ok | 1.1x (18%) |
| blas1 | one_norm | f64 | n=8388608 | 9.769 ms | 0.86 | 6.87 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.020 ms | 0.64 | 5.15 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.759 ms | 0.86 | 6.88 | 60% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 273.764 ms | 0.67 | 0.49 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 10.936 ms | 3.07 | 12.28 | 108% | ok | 1.2x (20%) |
| blas2 | gemv | f32 | n=4096 | 5.594 ms | 6.00 | 12.00 | 105% | ok | 1.1x (19%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 24.788 ms | 1.35 | 5.42 | 47% | ok | 1.5x (25%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 12.992 ms | 2.58 | 5.17 | 45% | ok | 1.3x (22%) |
| blas2 | symv | f64 | n=4096 | 16.360 ms | 2.05 | 4.11 | 36% | ok | 1.0x (17%) |
| blas2 | trmv | f64 | n=4096 | 6.877 ms | 2.44 | 9.77 | 86% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 3.530 ms | 4.75 | 9.52 | 83% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 7.056 ms | 2.38 | 9.52 | 83% | ok | 1.0x (16%) |
| blas2 | trsv | f32 | n=4096 | 3.527 ms | 4.76 | 9.52 | 83% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 27.883 ms | 1.20 | 9.63 | 84% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 5.948 ms | 45.13 | 1.06 | 76% | ok | 4.7x (78%) |
| blas3 | gemm | f32 | 512x512x512 | 2.910 ms | 92.26 | 1.08 | 78% | ok | 4.6x (77%) |
| blas3 | gemmt | f64 | 512x512x512 | 5.336 ms | 25.15 | 1.18 | 43% | ok | 3.2x (53%) |
| blas3 | symm | f64 | 512x512x512 | 6.920 ms | 38.79 | 0.91 | 66% | ok | 4.2x (70%) |
| blas3 | syrk | f64 | 512x512x512 | 37.350 ms | 3.59 | 0.11 | 6% | ok | 1.0x (16%) |
| blas3 | syr2k | f64 | 512x512x512 | 63.719 ms | 4.21 | 0.10 | 7% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 512x512x512 | 8.165 ms | 16.44 | 0.64 | 28% | ok | 2.2x (36%) |
| blas3 | trsm | f64 | 512x512x512 | 8.657 ms | 15.50 | 0.61 | 26% | ok | 2.1x (35%) |
| blas3 | trsm | f32 | 512x512x512 | 4.754 ms | 28.23 | 0.55 | 24% | ok | 1.9x (31%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.432 ms | 0.14 | 1.11 | 10% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 23.340 ms | 3.83 | 0.18 | 6% | ok | 1.3x (21%) |
| factor | lu | f32 | 512x512 | 13.493 ms | 6.63 | 0.16 | 6% | ok | 1.2x (20%) |
| factor | cholesky | f64 | 512x512 | 25.400 ms | 1.76 | 0.17 | 3% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 23.796 ms | 1.88 | 0.09 | 2% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 25.796 ms | 1.73 | 0.16 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 138.345 ms | 0.32 | 0.03 | 1% | ok | 1.1x (18%) |
| factor | qr | f64 | 512x512 | 322.890 ms | 0.55 | 0.01 | 1% | ok | 1.6x (26%) |
| factor | lq | f64 | 512x512 | 51.607 ms | 3.47 | 0.08 | 6% | ok | 1.4x (24%) |
| factor | hessenberg | f64 | 512x512 | 405.770 ms | 1.10 | 0.01 | 2% | ok | 1.5x (25%) |
| factor | householder | f64 | 512x512 | 0.963 ms | 1.09 | 4.36 | 38%\* | ok | 3.3x (55%) |
| factor | inv | f64 | 512x512 | 145.466 ms | 1.85 | 0.03 | 3% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 18.902 ms | 4.73 | 0.22 | 8% | ok | 1.2x (19%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 60.887 ms | 0.37 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 172.094 ms | 0.88 | 0.01 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1004.881 ms | 0.15 | 0.00 | 0% | ok | 1.0x (16%) |
| eigen | eigenvalue | f64 | 256x256 | 104.506 ms | 1.61 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 262.364 ms | 1.60 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 259.741 ms | 1.42 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 32.331 ms | 7.78 | 0.03 | 13% | ok | 1.6x (26%) |
| eigen | sylvester | f64 | 256x256 | 570.426 ms | 1.76 | 0.00 | 3% | ok | 1.0x (17%) |
| elementwise | exp | f64 | n=8388608 | 151.316 ms | 0.06 | 0.89 | 8% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 141.084 ms | 0.06 | 0.95 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 175.435 ms | 0.05 | 0.77 | 7% | ok | 1.0x (16%) |
| elementwise | sqrt | f64 | n=8388608 | 96.207 ms | 0.09 | 1.40 | 12% | ok | 0.9x (16%) |
| elementwise | erf | f64 | n=8388608 | 2479.881 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.315 ms | 1.13 | 11.76 | 103% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 4.301 ms | 0.61 | 6.33 | 55% | ok | 1.0x (16%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.399 ms | 3.67 | 7.72 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 17.231 ms | 0.19 | 0.67 | 6%\* | ok | 0.9x (16%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.839 ms | 0.34 | 2.85 | 25% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 238.483 ms | 0.05 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 70.372 ms | 0.16 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.543 ms | 0.42 | 0.03 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 116.327 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 155.351 ms | 0.14 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 49.233 ms | 0.45 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 235.217 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.276 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 27.547 ms | 0.00 | 0.00 | n/a | ok | 0.4x (7%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.656 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.661 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 80.335 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 112.174 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 64.527 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 32.157 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 38.401 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 37.037 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 66.444 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 74.151 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 37.531 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 43.810 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 42.069 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 46.548 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 54.852 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.130 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.459 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 33.438 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 2.869 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 2.765 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.076 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.200 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.526 ms | 0.74 | 8.32 | 73% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.130 ms | 0.84 | 9.37 | 82% | ok | 1.0x (16%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.904 ms | 0.67 | 7.51 | 66% | ok | 1.0x (16%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 34.987 ms | 1.10 | 8.31 | 73% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 140.147 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 255.962 ms | 0.00 | 0.00 | n/a | ok | 0.7x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 153.168 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 16.273 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 7.249 ms | 2.31 | 9.26 | 81% | ok | 1.0x (16%) |
| mixed | gemm | f32>f64 | 512x512x512 | 11.796 ms | 22.76 | 0.36 | 39% | ok | 5.3x (88%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.259 ms | 7.43 | 7.89 | 69% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.328 ms | 2.65 | 10.61 | n/a | ok | 0.9x (16%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.263 ms | 5.14 | 10.28 | n/a | ok | 0.9x (15%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.080 ms | 8.06 | 8.06 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.652 ms | 10.16 | 10.16 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 3.970 ms | 67.62 | 0.79 | n/a | ok | 4.7x (79%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.555 ms | 58.93 | 0.46 | n/a | ok | 4.8x (81%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.516 ms | 59.44 | 0.35 | n/a | ok | 4.9x (81%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.342 ms | 80.32 | 0.47 | n/a | ok | 4.8x (80%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 2.590 ms | 103.63 | 0.61 | n/a | ok | 4.8x (80%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

