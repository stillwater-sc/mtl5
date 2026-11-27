## xeon-e5-2420v2 -- native, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 21.1 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 21.1 GFLOP/s |
| ... this build, scalar | f64 0.1, f32 0.7 GFLOP/s |
| bandwidth | read 10.2, update 10.3, triad 8.2 GB/s (ceiling 10.3) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 24% | 105% | 0 |
| blas2 | 10 | 92% | 102% | 0 |
| blas3 | 10 | 38% | 40% | 0 |
| factor | 12 | 17% | 39% | 0 |
| eigen | 8 | 14% | 28% | 0 |
| elementwise | 5 | 9% | 14% | 0 |
| sparse | 5 | 57% | 110% | 0 |
| direct | 19 | 3% | 36% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 19% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 79% | 92% | 0 |
| mg | 1 | 77% | 77% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 23% | 76% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 185.155 ms | 0.09 | 0.72 | 7% | ok |
| blas1 | dot | f32 | n=8388608 | 26.911 ms | 0.62 | 2.49 | 24% | ok |
| blas1 | axpy | f64 | n=8388608 | 211.873 ms | 0.08 | 0.95 | 9% | ok |
| blas1 | axpy | f32 | n=8388608 | 91.315 ms | 0.18 | 1.10 | 11% | ok |
| blas1 | scale | f64 | n=8388608 | 12.375 ms | 0.68 | 10.85 | 105% | ok |
| blas1 | two_norm | f64 | n=8388608 | 147.086 ms | 0.11 | 0.46 | 4% | ok |
| blas1 | one_norm | f64 | n=8388608 | 9.724 ms | 0.86 | 6.90 | 67% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 13.077 ms | 0.64 | 5.13 | 50% | ok |
| blas1 | sum | f64 | n=8388608 | 9.684 ms | 0.87 | 6.93 | 67% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 275.030 ms | 0.67 | 0.49 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 13.817 ms | 2.43 | 9.72 | 94% | ok |
| blas2 | gemv | f32 | n=4096 | 8.343 ms | 4.02 | 8.05 | 78% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 13.238 ms | 2.53 | 10.14 | 98% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.410 ms | 5.23 | 10.47 | 102% | ok |
| blas2 | symv | f64 | n=4096 | 12.372 ms | 2.71 | 5.43 | 53% | ok |
| blas2 | trmv | f64 | n=4096 | 7.229 ms | 2.32 | 9.29 | 90% | ok |
| blas2 | trmv | f32 | n=4096 | 4.653 ms | 3.61 | 7.22 | 70% | ok |
| blas2 | trsv | f64 | n=4096 | 7.075 ms | 2.37 | 9.49 | 92% | ok |
| blas2 | trsv | f32 | n=4096 | 4.383 ms | 3.83 | 7.66 | 74% | ok |
| blas2 | ger | f64 | n=4096 | 28.232 ms | 1.19 | 9.51 | 92% | ok |
| blas3 | gemm | f64 | 512x512x512 | 64.508 ms | 4.16 | 0.10 | 40% | ok |
| blas3 | gemm | f32 | 512x512x512 | 35.761 ms | 7.51 | 0.09 | 36% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 35.876 ms | 3.74 | 0.18 | 36% | ok |
| blas3 | symm | f64 | 512x512x512 | 67.242 ms | 3.99 | 0.09 | 38% | ok |
| blas3 | syrk | f64 | 512x512x512 | 37.944 ms | 3.54 | 0.11 | 34% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 65.483 ms | 4.10 | 0.10 | 39% | ok |
| blas3 | trmm | f64 | 512x512x512 | 33.065 ms | 4.06 | 0.16 | 39% | ok |
| blas3 | trsm | f64 | 512x512x512 | 33.552 ms | 4.00 | 0.16 | 38% | ok |
| blas3 | trsm | f32 | 512x512x512 | 16.479 ms | 8.14 | 0.16 | 39% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.126 ms | 0.14 | 1.11 | 11% | ok |
| factor | lu | f64 | 512x512 | 30.910 ms | 2.89 | 0.14 | 28% | ok |
| factor | lu | f32 | 512x512 | 16.295 ms | 5.49 | 0.13 | 26% | ok |
| factor | cholesky | f64 | 512x512 | 24.790 ms | 1.80 | 0.17 | 17% | ok |
| factor | cholesky | f32 | 512x512 | 23.484 ms | 1.91 | 0.09 | 9% | ok |
| factor | ldlt | f64 | 512x512 | 25.389 ms | 1.76 | 0.17 | 17% | ok |
| factor | ldlt_bk | f64 | 512x512 | 13.824 ms | 3.24 | 0.30 | 31% | ok |
| factor | qr | f64 | 512x512 | 512.654 ms | 0.35 | 0.01 | 3% | ok |
| factor | lq | f64 | 512x512 | 73.903 ms | 2.42 | 0.06 | 23% | ok |
| factor | hessenberg | f64 | 512x512 | 625.437 ms | 0.72 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.231 ms | 0.32 | 1.30 | 13%\* | ok |
| factor | inv | f64 | 512x512 | 151.211 ms | 1.78 | 0.03 | 17% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 21.653 ms | 4.13 | 0.19 | 39% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 60.254 ms | 0.37 | 0.02 | 4% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 175.098 ms | 0.86 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 995.315 ms | 0.15 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 110.516 ms | 1.52 | 0.01 | 14% | ok |
| eigen | schur | f64 | 256x256 | 265.967 ms | 1.58 | 0.00 | 15% | ok |
| eigen | svd | f64 | 256x256 | 301.374 ms | 1.22 | 0.00 | 12% | ok |
| eigen | expm | f64 | 256x256 | 84.976 ms | 2.96 | 0.01 | 28% | ok |
| eigen | sylvester | f64 | 256x256 | 618.878 ms | 1.63 | 0.00 | 16% | ok |
| elementwise | exp | f64 | n=8388608 | 151.721 ms | 0.06 | 0.88 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 140.541 ms | 0.06 | 0.96 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 169.823 ms | 0.05 | 0.79 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 90.759 ms | 0.09 | 1.48 | 14% | ok |
| elementwise | erf | f64 | n=8388608 | 2473.537 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.390 ms | 1.10 | 11.39 | 110% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.101 ms | 0.84 | 8.78 | 85% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.066 ms | 2.78 | 5.84 | 57% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.593 ms | 0.20 | 0.69 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.878 ms | 0.34 | 2.83 | 27% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 40.937 ms | 0.27 | 0.02 | 3% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 73.129 ms | 0.15 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.986 ms | 0.41 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 116.003 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 129.622 ms | 0.17 | 0.01 | 2% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 47.909 ms | 0.47 | 0.02 | 4% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 82.196 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.884 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.111 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 125.389 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.534 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 41.089 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.268 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.214 ms | 0.34 | 0.02 | 3% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.805 ms | 0.94 | 3.76 | 36%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.443 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.519 ms | 0.89 | 0.06 | 9% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.842 ms | 0.90 | 3.59 | 35%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.310 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 103.879 ms | 0.22 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.798 ms | 0.20 | 0.80 | 8%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.825 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 112.590 ms | 0.20 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.817 ms | 0.20 | 0.79 | 8%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 12.856 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 638.893 ms | 0.48 | 0.00 | 5% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 6.785 ms | 0.50 | 1.99 | 19%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.708 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 206.874 ms | 1.49 | 0.00 | 14% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 6.776 ms | 0.50 | 1.99 | 19%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.527 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1747.428 ms | 0.35 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.885 ms | 0.10 | 0.41 | 4%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 17.962 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1537.333 ms | 0.40 | 0.00 | 4% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.917 ms | 0.10 | 0.41 | 4%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 221.850 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 561.747 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 134.293 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 104.711 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 125.477 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 81.484 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 533.565 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 165.713 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 86.554 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 187.534 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 179.094 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 196.890 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 291.407 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.172 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.379 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 33.369 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.900 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.743 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.086 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.079 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.615 ms | 0.72 | 8.11 | 79% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.102 ms | 0.84 | 9.45 | 92% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.808 ms | 0.69 | 7.70 | 75% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 36.402 ms | 1.06 | 7.98 | 77% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 141.696 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 146.503 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 132.434 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 36.795 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 188.933 ms | 0.09 | 0.36 | 3% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 113.134 ms | 2.37 | 0.04 | 23% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.267 ms | 7.40 | 7.86 | 76% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.435 ms | 2.61 | 10.43 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.136 ms | 5.35 | 10.70 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.978 ms | 3.37 | 3.37 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.981 ms | 3.37 | 3.37 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 57.755 ms | 4.65 | 0.05 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 45.175 ms | 5.94 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 23.735 ms | 11.31 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 23.670 ms | 11.34 | 0.07 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 70.450 ms | 3.81 | 0.02 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

