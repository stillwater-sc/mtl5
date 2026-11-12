## xeon-e5-2420v2 -- native, 6 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 59.0 GFLOP/s, f32 118.1 GFLOP/s |
| ... hardware, SSE2 | f64 59.0, f32 118.1 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.5 GFLOP/s |
| bandwidth | read 11.4, update 10.6, triad 8.4 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 54% | 107% | 0 | 81% |
| blas2 | 10 | 85% | 107% | 0 | 17% |
| blas3 | 10 | 7% | 39% | 0 | 17% |
| factor | 12 | 3% | 38% | 0 | 19% |
| eigen | 8 | 3% | 11% | 0 | 18% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 68% | 103% | 0 | 17% |
| direct | 7 | 0% | 1% | 0 | 17% |
| ordering | 4 | n/a | n/a | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 70% | 77% | 0 | 16% |
| mg | 1 | 73% | 73% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 18% |
| mixed | 3 | 22% | 67% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 91% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 33.035 ms | 0.51 | 4.06 | 36% | ok | 5.6x (93%) |
| blas1 | dot | f32 | n=8388608 | 5.515 ms | 3.04 | 12.17 | 107% | ok | 4.9x (81%) |
| blas1 | axpy | f64 | n=8388608 | 38.865 ms | 0.43 | 5.18 | 45% | ok | 5.5x (92%) |
| blas1 | axpy | f32 | n=8388608 | 16.368 ms | 1.03 | 6.15 | 54% | ok | 5.6x (93%) |
| blas1 | scale | f64 | n=8388608 | 12.190 ms | 0.69 | 11.01 | 97% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 33.124 ms | 0.51 | 2.03 | 18% | ok | 5.8x (97%) |
| blas1 | one_norm | f64 | n=8388608 | 9.771 ms | 0.86 | 6.87 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.048 ms | 0.64 | 5.14 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.738 ms | 0.86 | 6.89 | 60% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 280.165 ms | 0.66 | 0.48 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 10.976 ms | 3.06 | 12.23 | 107% | ok | 1.2x (20%) |
| blas2 | gemv | f32 | n=4096 | 5.524 ms | 6.07 | 12.15 | 107% | ok | 1.5x (25%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 10.959 ms | 3.06 | 12.25 | 107% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.921 ms | 4.85 | 9.70 | 85% | ok | 0.9x (16%) |
| blas2 | symv | f64 | n=4096 | 16.161 ms | 2.08 | 4.16 | 36% | ok | 1.0x (17%) |
| blas2 | trmv | f64 | n=4096 | 7.239 ms | 2.32 | 9.28 | 81% | ok | 1.0x (16%) |
| blas2 | trmv | f32 | n=4096 | 4.855 ms | 3.46 | 6.92 | 61% | ok | 1.0x (16%) |
| blas2 | trsv | f64 | n=4096 | 7.004 ms | 2.40 | 9.59 | 84% | ok | 1.0x (16%) |
| blas2 | trsv | f32 | n=4096 | 4.751 ms | 3.53 | 7.07 | 62% | ok | 1.0x (16%) |
| blas2 | ger | f64 | n=4096 | 27.851 ms | 1.20 | 9.64 | 85% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 11.598 ms | 23.14 | 0.54 | 39% | ok | 5.7x (94%) |
| blas3 | gemm | f32 | 512x512x512 | 6.435 ms | 41.71 | 0.49 | 35% | ok | 5.4x (90%) |
| blas3 | gemmt | f64 | 512x512x512 | 29.345 ms | 4.57 | 0.21 | 8% | ok | 1.0x (16%) |
| blas3 | symm | f64 | 512x512x512 | 61.011 ms | 4.40 | 0.10 | 7% | ok | 1.0x (17%) |
| blas3 | syrk | f64 | 512x512x512 | 38.337 ms | 3.50 | 0.11 | 6% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 512x512x512 | 65.877 ms | 4.07 | 0.10 | 7% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 512x512x512 | 33.168 ms | 4.05 | 0.16 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f64 | 512x512x512 | 32.426 ms | 4.14 | 0.16 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f32 | 512x512x512 | 16.974 ms | 7.91 | 0.15 | 7% | ok | 1.0x (16%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.535 ms | 0.14 | 1.10 | 10% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 24.017 ms | 3.73 | 0.17 | 6% | ok | 1.3x (21%) |
| factor | lu | f32 | 512x512 | 14.022 ms | 6.38 | 0.15 | 5% | ok | 1.2x (19%) |
| factor | cholesky | f64 | 512x512 | 25.323 ms | 1.77 | 0.17 | 3% | ok | 1.0x (16%) |
| factor | cholesky | f32 | 512x512 | 23.902 ms | 1.87 | 0.09 | 2% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 25.385 ms | 1.76 | 0.17 | 3% | ok | 1.0x (16%) |
| factor | ldlt_bk | f64 | 512x512 | 135.416 ms | 0.33 | 0.03 | 1% | ok | 1.1x (18%) |
| factor | qr | f64 | 512x512 | 321.919 ms | 0.56 | 0.01 | 1% | ok | 1.6x (27%) |
| factor | lq | f64 | 512x512 | 51.133 ms | 3.50 | 0.08 | 6% | ok | 1.4x (24%) |
| factor | hessenberg | f64 | 512x512 | 404.048 ms | 1.11 | 0.01 | 2% | ok | 1.6x (26%) |
| factor | householder | f64 | 512x512 | 0.962 ms | 1.09 | 4.36 | 38%\* | ok | 3.3x (56%) |
| factor | inv | f64 | 512x512 | 148.026 ms | 1.81 | 0.03 | 3% | ok | 1.0x (17%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 19.129 ms | 4.68 | 0.22 | 8% | ok | 1.1x (19%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 60.437 ms | 0.37 | 0.02 | 1% | ok | 1.1x (18%) |
| eigen | eigen_symmetric | f64 | 256x256 | 169.090 ms | 0.89 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 990.463 ms | 0.15 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 106.743 ms | 1.57 | 0.01 | 3% | ok | 1.1x (18%) |
| eigen | schur | f64 | 256x256 | 263.592 ms | 1.59 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 304.044 ms | 1.21 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 37.434 ms | 6.72 | 0.03 | 11% | ok | 2.3x (38%) |
| eigen | sylvester | f64 | 256x256 | 585.827 ms | 1.72 | 0.00 | 3% | ok | 1.1x (18%) |
| elementwise | exp | f64 | n=8388608 | 152.559 ms | 0.05 | 0.88 | 8% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 142.274 ms | 0.06 | 0.94 | 8% | ok | 1.0x (16%) |
| elementwise | sin | f64 | n=8388608 | 172.044 ms | 0.05 | 0.78 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 90.762 ms | 0.09 | 1.48 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 2471.498 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.313 ms | 1.13 | 11.77 | 103% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.093 ms | 0.85 | 8.80 | 77% | ok | 1.0x (16%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.354 ms | 3.69 | 7.76 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.154 ms | 0.20 | 0.71 | 6%\* | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.845 ms | 0.34 | 2.85 | 25% | ok | 1.0x (16%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 274.391 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 69.378 ms | 0.16 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.931 ms | 0.42 | 0.03 | 1% | ok | 1.0x (16%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 116.980 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 154.391 ms | 0.14 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 48.442 ms | 0.46 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 231.361 ms | 0.01 | 0.00 | 0% | ok | 1.0x (16%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.908 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.268 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.610 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.874 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 231.783 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 393.968 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 126.557 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 102.049 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 121.256 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 79.084 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 518.805 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 174.265 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 90.778 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 136.612 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 137.614 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 172.218 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 250.033 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.141 ms | 0.00 | 0.00 | n/a | ok | 0.9x (15%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.414 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 33.126 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 2.897 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 2.707 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.076 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.143 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.695 ms | 0.71 | 7.94 | 70% | ok | 1.0x (16%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.342 ms | 0.78 | 8.77 | 77% | ok | 0.9x (16%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.832 ms | 0.68 | 7.65 | 67% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 34.936 ms | 1.10 | 8.32 | 73% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 141.782 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 189.310 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 145.148 ms | 0.00 | 0.00 | n/a | ok | 1.1x (19%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.702 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 187.734 ms | 0.09 | 0.36 | 3% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 20.564 ms | 13.05 | 0.20 | 22% | ok | 5.5x (92%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.337 ms | 7.18 | 7.63 | 67% | ok | 1.0x (16%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.447 ms | 2.60 | 10.41 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.145 ms | 5.33 | 10.67 | n/a | ok | 1.0x (16%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.984 ms | 3.37 | 3.37 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 5.013 ms | 3.35 | 3.35 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 10.348 ms | 25.94 | 0.30 | n/a | ok | 5.6x (93%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 8.238 ms | 32.58 | 0.25 | n/a | ok | 5.5x (92%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.336 ms | 61.90 | 0.36 | n/a | ok | 5.5x (91%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.335 ms | 61.93 | 0.36 | n/a | ok | 5.5x (92%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.096 ms | 20.50 | 0.12 | n/a | ok | 5.5x (92%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

