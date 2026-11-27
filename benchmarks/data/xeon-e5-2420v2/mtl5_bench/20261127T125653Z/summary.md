# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,6, CPUs 0,1,2,3,4,5, clock policy governor performance on every CPU, turbo on; performance 1200-2700 MHz on cpus 0,1,2,3,4,5, 2026-11-27T13:21:42Z.

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

## xeon-e5-2420v2 -- native, 6 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 58.9 GFLOP/s, f32 117.9 GFLOP/s |
| ... hardware, SSE2 | f64 58.9, f32 117.9 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.7 GFLOP/s |
| bandwidth | read 11.1, update 10.4, triad 8.3 GB/s (ceiling 11.1) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 54% | 110% | 0 | 77% |
| blas2 | 10 | 89% | 111% | 0 | 17% |
| blas3 | 10 | 7% | 40% | 0 | 17% |
| factor | 12 | 6% | 41% | 0 | 23% |
| eigen | 8 | 3% | 11% | 0 | 17% |
| elementwise | 5 | 8% | 14% | 0 | 17% |
| sparse | 5 | 70% | 107% | 0 | 17% |
| direct | 19 | 1% | 32% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 4% | 17% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 76% | 85% | 0 | 17% |
| mg | 1 | 74% | 74% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 22% | 70% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 54% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 39.859 ms | 0.42 | 3.37 | 30% | ok | 4.6x (77%) |
| blas1 | dot | f32 | n=8388608 | 5.497 ms | 3.05 | 12.21 | 110% | ok | 4.9x (82%) |
| blas1 | axpy | f64 | n=8388608 | 38.135 ms | 0.44 | 5.28 | 48% | ok | 5.6x (93%) |
| blas1 | axpy | f32 | n=8388608 | 16.755 ms | 1.00 | 6.01 | 54% | ok | 5.4x (91%) |
| blas1 | scale | f64 | n=8388608 | 12.151 ms | 0.69 | 11.05 | 100% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 26.063 ms | 0.64 | 2.57 | 23% | ok | 5.6x (94%) |
| blas1 | one_norm | f64 | n=8388608 | 9.733 ms | 0.86 | 6.89 | 62% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 12.973 ms | 0.65 | 5.17 | 47% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.824 ms | 0.85 | 6.83 | 62% | ok | 1.0x (16%) |
| blas1 | dot_k3 | f64 | n=8388608 | 274.360 ms | 0.67 | 0.49 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 10.933 ms | 3.07 | 12.28 | 111% | ok | 1.3x (21%) |
| blas2 | gemv | f32 | n=4096 | 5.481 ms | 6.12 | 12.25 | 111% | ok | 1.5x (25%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 10.920 ms | 3.07 | 12.30 | 111% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.561 ms | 6.03 | 12.07 | 109% | ok | 1.2x (19%) |
| blas2 | symv | f64 | n=4096 | 12.258 ms | 2.74 | 5.48 | 50% | ok | 1.0x (17%) |
| blas2 | trmv | f64 | n=4096 | 7.055 ms | 2.38 | 9.52 | 86% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 4.471 ms | 3.75 | 7.51 | 68% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 6.858 ms | 2.45 | 9.79 | 89% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 4.327 ms | 3.88 | 7.76 | 70% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 27.949 ms | 1.20 | 9.61 | 87% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 11.520 ms | 23.30 | 0.55 | 40% | ok | 5.6x (93%) |
| blas3 | gemm | f32 | 512x512x512 | 6.237 ms | 43.04 | 0.50 | 37% | ok | 5.7x (96%) |
| blas3 | gemmt | f64 | 512x512x512 | 36.087 ms | 3.72 | 0.17 | 6% | ok | 1.0x (17%) |
| blas3 | symm | f64 | 512x512x512 | 67.906 ms | 3.95 | 0.09 | 7% | ok | 1.0x (17%) |
| blas3 | syrk | f64 | 512x512x512 | 37.716 ms | 3.56 | 0.11 | 6% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 512x512x512 | 65.518 ms | 4.10 | 0.10 | 7% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 512x512x512 | 33.322 ms | 4.03 | 0.16 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f64 | 512x512x512 | 33.768 ms | 3.97 | 0.16 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f32 | 512x512x512 | 16.409 ms | 8.18 | 0.16 | 7% | ok | 1.0x (17%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.298 ms | 0.14 | 1.11 | 10% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 21.254 ms | 4.21 | 0.20 | 7% | ok | 1.5x (24%) |
| factor | lu | f32 | 512x512 | 11.917 ms | 7.51 | 0.18 | 6% | ok | 1.4x (23%) |
| factor | cholesky | f64 | 512x512 | 24.823 ms | 1.80 | 0.17 | 3% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 23.690 ms | 1.89 | 0.09 | 2% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 25.044 ms | 1.79 | 0.17 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 13.673 ms | 3.27 | 0.31 | 6% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 318.415 ms | 0.56 | 0.01 | 1% | ok | 1.6x (27%) |
| factor | lq | f64 | 512x512 | 48.266 ms | 3.71 | 0.09 | 6% | ok | 1.5x (26%) |
| factor | hessenberg | f64 | 512x512 | 398.316 ms | 1.12 | 0.01 | 2% | ok | 1.6x (26%) |
| factor | householder | f64 | 512x512 | 0.922 ms | 1.14 | 4.55 | 41%\* | ok | 3.5x (58%) |
| factor | inv | f64 | 512x512 | 142.223 ms | 1.89 | 0.03 | 3% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 18.160 ms | 4.93 | 0.23 | 8% | ok | 1.2x (20%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 63.803 ms | 0.35 | 0.02 | 1% | ok | 0.9x (16%) |
| eigen | eigen_symmetric | f64 | 256x256 | 172.214 ms | 0.88 | 0.01 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1001.239 ms | 0.15 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 108.059 ms | 1.55 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 270.657 ms | 1.55 | 0.00 | 3% | ok | 1.0x (16%) |
| eigen | svd | f64 | 256x256 | 301.002 ms | 1.23 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 37.299 ms | 6.75 | 0.03 | 11% | ok | 2.3x (38%) |
| eigen | sylvester | f64 | 256x256 | 582.573 ms | 1.73 | 0.00 | 3% | ok | 1.1x (18%) |
| elementwise | exp | f64 | n=8388608 | 150.768 ms | 0.06 | 0.89 | 8% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 139.494 ms | 0.06 | 0.96 | 9% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 172.038 ms | 0.05 | 0.78 | 7% | ok | 1.0x (16%) |
| elementwise | sqrt | f64 | n=8388608 | 89.459 ms | 0.09 | 1.50 | 14% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 2477.686 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.293 ms | 1.14 | 11.87 | 107% | ok | 1.0x (17%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.125 ms | 0.84 | 8.71 | 79% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.327 ms | 3.70 | 7.77 | 70% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 17.020 ms | 0.19 | 0.68 | 6%\* | ok | 1.0x (16%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.870 ms | 0.34 | 2.84 | 26% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 40.767 ms | 0.27 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 69.054 ms | 0.16 | 0.01 | 0% | ok | 1.1x (18%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.982 ms | 0.45 | 0.03 | 1% | ok | 1.1x (18%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 115.092 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 132.744 ms | 0.17 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 57.658 ms | 0.39 | 0.01 | 1% | ok | 0.8x (14%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 82.419 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 19.937 ms | 0.00 | 0.00 | n/a | ok | 0.4x (7%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 25.110 ms | 0.00 | 0.00 | n/a | ok | 0.5x (8%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 125.893 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.527 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.972 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.289 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.255 ms | 0.34 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.849 ms | 0.89 | 3.56 | 32%\* | ok | 0.9x (16%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.459 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.546 ms | 0.89 | 0.06 | 2% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.872 ms | 0.87 | 3.47 | 31%\* | ok | 1.0x (16%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.326 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 103.771 ms | 0.22 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.780 ms | 0.20 | 0.80 | 7%\* | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.745 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 115.383 ms | 0.19 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.784 ms | 0.20 | 0.80 | 7%\* | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 12.963 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 634.986 ms | 0.49 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 6.981 ms | 0.48 | 1.93 | 17%\* | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.617 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 207.450 ms | 1.49 | 0.00 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.015 ms | 0.48 | 1.92 | 17%\* | ok | 1.0x (16%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.851 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1711.261 ms | 0.36 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.861 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 18.099 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1530.013 ms | 0.40 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.749 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 221.985 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 571.639 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 123.333 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 99.087 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 117.861 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 77.193 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 526.995 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 162.017 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 86.430 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 179.229 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 174.037 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 195.881 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 291.203 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.171 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.410 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 33.474 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 2.893 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 2.715 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.079 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.155 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.480 ms | 0.75 | 8.43 | 76% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.105 ms | 0.84 | 9.45 | 85% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.788 ms | 0.69 | 7.74 | 70% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.407 ms | 1.09 | 8.21 | 74% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 141.642 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 148.011 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 128.089 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 37.020 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 189.515 ms | 0.09 | 0.35 | 3% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 20.481 ms | 13.11 | 0.20 | 22% | ok | 5.5x (92%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.318 ms | 7.24 | 7.69 | 70% | ok | 1.0x (16%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.461 ms | 2.60 | 10.39 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.125 ms | 5.37 | 10.74 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.979 ms | 3.37 | 3.37 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.997 ms | 3.36 | 3.36 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.555 ms | 23.23 | 0.27 | n/a | ok | 5.0x (83%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 8.225 ms | 32.64 | 0.25 | n/a | ok | 5.5x (92%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 7.382 ms | 36.37 | 0.21 | n/a | ok | 3.2x (54%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.322 ms | 62.11 | 0.36 | n/a | ok | 5.5x (91%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.060 ms | 20.55 | 0.12 | n/a | ok | 5.4x (90%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## xeon-e5-2420v2 -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 21.0 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 21.0 GFLOP/s |
| ... this build, SSE4 | f64 10.5, f32 21.0 GFLOP/s |
| bandwidth | read 10.5, update 10.4, triad 8.2 GB/s (ceiling 10.5) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 104% | 109% | 0 |
| blas2 | 10 | 90% | 100% | 0 |
| blas3 | 10 | 76% | 90% | 0 |
| factor | 12 | 17% | 41% | 0 |
| eigen | 8 | 15% | 48% | 0 |
| elementwise | 5 | 8% | 13% | 0 |
| sparse | 5 | 56% | 106% | 0 |
| direct | 19 | 3% | 36% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 19% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 78% | 89% | 0 |
| mg | 1 | 78% | 78% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 74% | 107% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 12.003 ms | 1.40 | 11.18 | 107% | ok |
| blas1 | dot | f32 | n=8388608 | 5.934 ms | 2.83 | 11.31 | 108% | ok |
| blas1 | axpy | f64 | n=8388608 | 18.806 ms | 0.89 | 10.71 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 8.972 ms | 1.87 | 11.22 | 107% | ok |
| blas1 | scale | f64 | n=8388608 | 12.353 ms | 0.68 | 10.87 | 104% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.894 ms | 2.85 | 11.39 | 109% | ok |
| blas1 | one_norm | f64 | n=8388608 | 9.746 ms | 0.86 | 6.89 | 66% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 13.368 ms | 0.63 | 5.02 | 48% | ok |
| blas1 | sum | f64 | n=8388608 | 9.765 ms | 0.86 | 6.87 | 66% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 271.158 ms | 0.68 | 0.49 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 13.358 ms | 2.51 | 10.05 | 96% | ok |
| blas2 | gemv | f32 | n=4096 | 6.684 ms | 5.02 | 10.05 | 96% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 13.080 ms | 2.57 | 10.27 | 98% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.420 ms | 5.23 | 10.46 | 100% | ok |
| blas2 | symv | f64 | n=4096 | 12.280 ms | 2.73 | 5.47 | 52% | ok |
| blas2 | trmv | f64 | n=4096 | 7.078 ms | 2.37 | 9.49 | 90% | ok |
| blas2 | trmv | f32 | n=4096 | 3.650 ms | 4.60 | 9.20 | 88% | ok |
| blas2 | trsv | f64 | n=4096 | 7.157 ms | 2.34 | 9.39 | 89% | ok |
| blas2 | trsv | f32 | n=4096 | 3.931 ms | 4.27 | 8.54 | 81% | ok |
| blas2 | ger | f64 | n=4096 | 29.793 ms | 1.13 | 9.01 | 86% | ok |
| blas3 | gemm | f64 | 512x512x512 | 29.679 ms | 9.04 | 0.21 | 86% | ok |
| blas3 | gemm | f32 | 512x512x512 | 14.233 ms | 18.86 | 0.22 | 90% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 16.833 ms | 7.97 | 0.37 | 76% | ok |
| blas3 | symm | f64 | 512x512x512 | 28.699 ms | 9.35 | 0.22 | 89% | ok |
| blas3 | syrk | f64 | 512x512x512 | 16.749 ms | 8.01 | 0.25 | 76% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 33.427 ms | 8.03 | 0.19 | 77% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.431 ms | 7.70 | 0.30 | 73% | ok |
| blas3 | trsm | f64 | 512x512x512 | 18.058 ms | 7.43 | 0.29 | 71% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.828 ms | 15.20 | 0.30 | 72% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.244 ms | 0.14 | 1.11 | 11% | ok |
| factor | lu | f64 | 512x512 | 29.307 ms | 3.05 | 0.14 | 29% | ok |
| factor | lu | f32 | 512x512 | 15.340 ms | 5.83 | 0.14 | 28% | ok |
| factor | cholesky | f64 | 512x512 | 25.360 ms | 1.76 | 0.17 | 17% | ok |
| factor | cholesky | f32 | 512x512 | 23.795 ms | 1.88 | 0.09 | 9% | ok |
| factor | ldlt | f64 | 512x512 | 25.706 ms | 1.74 | 0.16 | 17% | ok |
| factor | ldlt_bk | f64 | 512x512 | 13.356 ms | 3.35 | 0.31 | 32% | ok |
| factor | qr | f64 | 512x512 | 514.986 ms | 0.35 | 0.01 | 3% | ok |
| factor | lq | f64 | 512x512 | 73.535 ms | 2.43 | 0.06 | 23% | ok |
| factor | hessenberg | f64 | 512x512 | 624.495 ms | 0.72 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.187 ms | 0.33 | 1.32 | 13%\* | ok |
| factor | inv | f64 | 512x512 | 153.076 ms | 1.75 | 0.03 | 17% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 20.927 ms | 4.28 | 0.20 | 41% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.332 ms | 0.36 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 173.769 ms | 0.87 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1000.933 ms | 0.15 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 102.697 ms | 1.63 | 0.01 | 16% | ok |
| eigen | schur | f64 | 256x256 | 258.472 ms | 1.62 | 0.00 | 15% | ok |
| eigen | svd | f64 | 256x256 | 261.597 ms | 1.41 | 0.00 | 13% | ok |
| eigen | expm | f64 | 256x256 | 49.894 ms | 5.04 | 0.02 | 48% | ok |
| eigen | sylvester | f64 | 256x256 | 577.818 ms | 1.74 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 150.921 ms | 0.06 | 0.89 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 140.938 ms | 0.06 | 0.95 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 169.215 ms | 0.05 | 0.79 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 95.191 ms | 0.09 | 1.41 | 13% | ok |
| elementwise | erf | f64 | n=8388608 | 2477.572 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.450 ms | 1.07 | 11.11 | 106% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.146 ms | 0.83 | 8.66 | 83% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.085 ms | 2.78 | 5.84 | 56% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.061 ms | 0.20 | 0.72 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.854 ms | 0.34 | 2.84 | 27% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 40.721 ms | 0.27 | 0.02 | 3% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 69.317 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.939 ms | 0.45 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 117.640 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 129.151 ms | 0.17 | 0.01 | 2% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 47.761 ms | 0.47 | 0.02 | 4% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 78.575 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.076 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.316 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 127.964 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.555 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 39.889 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.338 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.336 ms | 0.34 | 0.02 | 3% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.799 ms | 0.95 | 3.79 | 36%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.487 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.588 ms | 0.89 | 0.06 | 8% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.817 ms | 0.92 | 3.70 | 35%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.558 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 110.504 ms | 0.20 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 4.592 ms | 0.16 | 0.66 | 6%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 10.578 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 118.474 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.874 ms | 0.20 | 0.78 | 7%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 13.253 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 638.640 ms | 0.48 | 0.00 | 5% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 6.836 ms | 0.49 | 1.97 | 19%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 29.126 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 206.029 ms | 1.50 | 0.00 | 14% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 6.897 ms | 0.49 | 1.95 | 19%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.940 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1727.139 ms | 0.36 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.837 ms | 0.10 | 0.41 | 4%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 18.466 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1958.880 ms | 0.32 | 0.00 | 3% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.855 ms | 0.10 | 0.41 | 4%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 80.886 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 171.315 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 60.126 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 32.039 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 38.106 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 37.203 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 64.883 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 69.066 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 37.583 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 43.429 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 39.702 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 38.320 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 44.863 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.127 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.661 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 32.962 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.880 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.756 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.081 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.073 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.585 ms | 0.73 | 8.18 | 78% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.126 ms | 0.84 | 9.38 | 89% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.896 ms | 0.67 | 7.53 | 72% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.671 ms | 1.08 | 8.15 | 78% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 140.512 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 144.666 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 131.666 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 16.239 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 5.961 ms | 2.81 | 11.26 | 107% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 62.013 ms | 4.33 | 0.07 | 41% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.283 ms | 7.35 | 7.81 | 74% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.048 ms | 2.77 | 11.10 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.010 ms | 5.57 | 11.15 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.098 ms | 8.00 | 8.00 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.650 ms | 10.17 | 10.17 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 18.780 ms | 14.29 | 0.17 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 22.092 ms | 12.15 | 0.09 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 22.185 ms | 12.10 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 16.286 ms | 16.48 | 0.10 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 12.439 ms | 21.58 | 0.13 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## xeon-e5-2420v2 -- native-fast, 6 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 59.1 GFLOP/s, f32 118.2 GFLOP/s |
| ... hardware, SSE2 | f64 59.1, f32 118.2 GFLOP/s |
| ... this build, SSE4 | f64 59.1, f32 118.1 GFLOP/s |
| bandwidth | read 11.4, update 10.7, triad 8.5 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 99% | 108% | 0 | 18% |
| blas2 | 10 | 85% | 108% | 0 | 18% |
| blas3 | 10 | 46% | 79% | 0 | 57% |
| factor | 12 | 6% | 42% | 0 | 23% |
| eigen | 8 | 3% | 13% | 0 | 17% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 68% | 103% | 0 | 17% |
| direct | 19 | 1% | 31% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 4% | 17% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 70% | 82% | 0 | 17% |
| mg | 1 | 65% | 65% | 0 | 15% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 68% | 91% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 79% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 10.867 ms | 1.54 | 12.35 | 108% | ok | 1.1x (18%) |
| blas1 | dot | f32 | n=8388608 | 5.439 ms | 3.08 | 12.34 | 108% | ok | 1.1x (18%) |
| blas1 | axpy | f64 | n=8388608 | 17.846 ms | 0.94 | 11.28 | 99% | ok | 1.1x (18%) |
| blas1 | axpy | f32 | n=8388608 | 8.444 ms | 1.99 | 11.92 | 104% | ok | 1.1x (18%) |
| blas1 | scale | f64 | n=8388608 | 12.141 ms | 0.69 | 11.05 | 97% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 5.424 ms | 3.09 | 12.37 | 108% | ok | 1.1x (18%) |
| blas1 | one_norm | f64 | n=8388608 | 9.815 ms | 0.85 | 6.84 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.059 ms | 0.64 | 5.14 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.765 ms | 0.86 | 6.87 | 60% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 272.067 ms | 0.68 | 0.49 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 10.909 ms | 3.08 | 12.31 | 108% | ok | 1.2x (20%) |
| blas2 | gemv | f32 | n=4096 | 5.452 ms | 6.15 | 12.31 | 108% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 11.058 ms | 3.03 | 12.14 | 106% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.820 ms | 5.77 | 11.54 | 101% | ok | 1.1x (18%) |
| blas2 | symv | f64 | n=4096 | 12.669 ms | 2.65 | 5.30 | 46% | ok | 1.0x (16%) |
| blas2 | trmv | f64 | n=4096 | 7.059 ms | 2.38 | 9.52 | 83% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 3.577 ms | 4.69 | 9.39 | 82% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 6.914 ms | 2.43 | 9.72 | 85% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 3.809 ms | 4.40 | 8.82 | 77% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 28.222 ms | 1.19 | 9.51 | 83% | ok | 1.1x (18%) |
| blas3 | gemm | f64 | 512x512x512 | 5.860 ms | 45.81 | 1.07 | 77% | ok | 5.1x (84%) |
| blas3 | gemm | f32 | 512x512x512 | 2.883 ms | 93.10 | 1.09 | 79% | ok | 4.9x (82%) |
| blas3 | gemmt | f64 | 512x512x512 | 4.988 ms | 26.91 | 1.26 | 45% | ok | 3.4x (56%) |
| blas3 | symm | f64 | 512x512x512 | 6.903 ms | 38.89 | 0.91 | 66% | ok | 4.2x (69%) |
| blas3 | syrk | f64 | 512x512x512 | 4.925 ms | 27.25 | 0.85 | 46% | ok | 3.4x (57%) |
| blas3 | syr2k | f64 | 512x512x512 | 9.540 ms | 28.14 | 0.66 | 48% | ok | 3.5x (58%) |
| blas3 | trmm | f64 | 512x512x512 | 7.880 ms | 17.03 | 0.67 | 29% | ok | 2.2x (37%) |
| blas3 | trsm | f64 | 512x512x512 | 8.623 ms | 15.56 | 0.61 | 26% | ok | 2.1x (35%) |
| blas3 | trsm | f32 | 512x512x512 | 4.174 ms | 32.15 | 0.63 | 27% | ok | 2.1x (35%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.721 ms | 0.14 | 1.10 | 10% | ok | 1.0x (16%) |
| factor | lu | f64 | 512x512 | 20.435 ms | 4.38 | 0.21 | 7% | ok | 1.4x (24%) |
| factor | lu | f32 | 512x512 | 11.244 ms | 7.96 | 0.19 | 7% | ok | 1.4x (23%) |
| factor | cholesky | f64 | 512x512 | 25.358 ms | 1.76 | 0.17 | 3% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 23.878 ms | 1.87 | 0.09 | 2% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 25.561 ms | 1.75 | 0.16 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 13.057 ms | 3.43 | 0.32 | 6% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 322.805 ms | 0.55 | 0.01 | 1% | ok | 1.6x (27%) |
| factor | lq | f64 | 512x512 | 47.859 ms | 3.74 | 0.09 | 6% | ok | 1.5x (26%) |
| factor | hessenberg | f64 | 512x512 | 400.919 ms | 1.12 | 0.01 | 2% | ok | 1.6x (26%) |
| factor | householder | f64 | 512x512 | 0.882 ms | 1.19 | 4.76 | 42%\* | ok | 3.6x (60%) |
| factor | inv | f64 | 512x512 | 143.920 ms | 1.87 | 0.03 | 3% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.784 ms | 5.33 | 0.25 | 9% | ok | 1.2x (21%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 60.919 ms | 0.37 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 171.578 ms | 0.88 | 0.01 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 991.950 ms | 0.15 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 106.141 ms | 1.58 | 0.01 | 3% | ok | 1.0x (16%) |
| eigen | schur | f64 | 256x256 | 260.247 ms | 1.61 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 260.226 ms | 1.42 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 32.626 ms | 7.71 | 0.03 | 13% | ok | 1.5x (25%) |
| eigen | sylvester | f64 | 256x256 | 582.115 ms | 1.73 | 0.00 | 3% | ok | 1.0x (17%) |
| elementwise | exp | f64 | n=8388608 | 152.144 ms | 0.06 | 0.88 | 8% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 139.537 ms | 0.06 | 0.96 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 171.860 ms | 0.05 | 0.78 | 7% | ok | 1.0x (16%) |
| elementwise | sqrt | f64 | n=8388608 | 90.077 ms | 0.09 | 1.49 | 13% | ok | 1.1x (18%) |
| elementwise | erf | f64 | n=8388608 | 2499.840 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.300 ms | 1.14 | 11.84 | 103% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.123 ms | 0.84 | 8.72 | 76% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.323 ms | 3.70 | 7.78 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.463 ms | 0.20 | 0.70 | 6%\* | ok | 1.0x (16%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.873 ms | 0.34 | 2.83 | 25% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 40.864 ms | 0.27 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 70.705 ms | 0.16 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 25.151 ms | 0.44 | 0.03 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 116.767 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 130.195 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 48.086 ms | 0.47 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 79.138 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.272 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.299 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 127.925 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.543 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.643 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.772 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.276 ms | 0.34 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.854 ms | 0.89 | 3.54 | 31%\* | ok | 0.9x (16%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.579 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.524 ms | 0.89 | 0.06 | 2% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.884 ms | 0.86 | 3.42 | 30%\* | ok | 0.9x (15%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.776 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 103.937 ms | 0.22 | 0.01 | 0% | ok | 1.1x (18%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.761 ms | 0.20 | 0.80 | 7%\* | ok | 1.2x (20%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 10.229 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 112.938 ms | 0.20 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.881 ms | 0.19 | 0.78 | 7%\* | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 13.219 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 635.964 ms | 0.49 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.028 ms | 0.48 | 1.92 | 17%\* | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.935 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 206.045 ms | 1.50 | 0.00 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.076 ms | 0.48 | 1.90 | 17%\* | ok | 1.0x (16%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.874 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1720.063 ms | 0.36 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.667 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 18.296 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1938.059 ms | 0.32 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.597 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 80.244 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 164.476 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 59.958 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 31.782 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 38.223 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 36.992 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 65.666 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 69.337 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 37.402 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 43.472 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 39.396 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 38.321 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 44.707 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.128 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.598 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 33.310 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | ic_0 | f64 | 128^2 grid | 2.857 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 2.764 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.080 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.219 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.644 ms | 0.72 | 8.05 | 70% | ok | 1.0x (16%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.122 ms | 0.84 | 9.39 | 82% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.805 ms | 0.69 | 7.71 | 67% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 39.232 ms | 0.98 | 7.41 | 65% | ok | 0.9x (15%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 139.864 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 144.545 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 129.901 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 16.047 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 6.458 ms | 2.60 | 10.39 | 91% | ok | 0.9x (15%) |
| mixed | gemm | f32>f64 | 512x512x512 | 11.733 ms | 22.88 | 0.36 | 39% | ok | 5.3x (88%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.282 ms | 7.35 | 7.81 | 68% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.014 ms | 2.79 | 11.16 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.978 ms | 5.63 | 11.27 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.082 ms | 8.06 | 8.06 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.641 ms | 10.22 | 10.22 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 3.948 ms | 67.99 | 0.80 | n/a | ok | 4.8x (79%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.541 ms | 59.11 | 0.46 | n/a | ok | 4.9x (81%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.492 ms | 59.76 | 0.35 | n/a | ok | 4.9x (82%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.315 ms | 80.98 | 0.47 | n/a | ok | 4.9x (82%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 2.565 ms | 104.66 | 0.61 | n/a | ok | 4.8x (81%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## xeon-e5-2420v2 -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 20.9 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 20.9 GFLOP/s |
| ... this build, scalar | f64 0.1, f32 0.7 GFLOP/s |
| bandwidth | read 10.2, update 10.3, triad 8.1 GB/s (ceiling 10.3) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 24% | 105% | 0 |
| blas2 | 10 | 92% | 104% | 0 |
| blas3 | 10 | 163% | 182% | 0 |
| factor | 12 | 55% | 92% | 0 |
| eigen | 8 | 50% | 119% | 0 |
| elementwise | 5 | 8% | 14% | 0 |
| sparse | 5 | 56% | 110% | 0 |
| direct | 19 | 3% | 35% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 19% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 79% | 91% | 0 |
| mg | 1 | 73% | 73% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 22% | 77% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 198.936 ms | 0.08 | 0.67 | 7% | ok |
| blas1 | dot | f32 | n=8388608 | 27.457 ms | 0.61 | 2.44 | 24% | ok |
| blas1 | axpy | f64 | n=8388608 | 212.927 ms | 0.08 | 0.95 | 9% | ok |
| blas1 | axpy | f32 | n=8388608 | 93.529 ms | 0.18 | 1.08 | 10% | ok |
| blas1 | scale | f64 | n=8388608 | 12.445 ms | 0.67 | 10.79 | 105% | ok |
| blas1 | two_norm | f64 | n=8388608 | 194.668 ms | 0.09 | 0.34 | 3% | ok |
| blas1 | one_norm | f64 | n=8388608 | 9.793 ms | 0.86 | 6.85 | 66% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 13.049 ms | 0.64 | 5.14 | 50% | ok |
| blas1 | sum | f64 | n=8388608 | 9.784 ms | 0.86 | 6.86 | 67% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 279.348 ms | 0.66 | 0.48 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 14.367 ms | 2.34 | 9.35 | 91% | ok |
| blas2 | gemv | f32 | n=4096 | 6.242 ms | 5.38 | 10.76 | 104% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 12.948 ms | 2.59 | 10.37 | 101% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.561 ms | 5.11 | 10.23 | 99% | ok |
| blas2 | symv | f64 | n=4096 | 7.131 ms | 4.71 | 9.42 | 91% | ok |
| blas2 | trmv | f64 | n=4096 | 7.594 ms | 2.21 | 8.85 | 86% | ok |
| blas2 | trmv | f32 | n=4096 | 3.751 ms | 4.47 | 8.96 | 87% | ok |
| blas2 | trsv | f64 | n=4096 | 7.489 ms | 2.24 | 8.97 | 87% | ok |
| blas2 | trsv | f32 | n=4096 | 3.549 ms | 4.73 | 9.46 | 92% | ok |
| blas2 | ger | f64 | n=4096 | 28.428 ms | 1.18 | 9.45 | 92% | ok |
| blas3 | gemm | f64 | 512x512x512 | 14.146 ms | 18.98 | 0.44 | 181% | ok |
| blas3 | gemm | f32 | 512x512x512 | 8.046 ms | 33.36 | 0.39 | 160% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 8.794 ms | 15.26 | 0.72 | 146% | ok |
| blas3 | symm | f64 | 512x512x512 | 14.106 ms | 19.03 | 0.45 | 182% | ok |
| blas3 | syrk | f64 | 512x512x512 | 7.857 ms | 17.08 | 0.53 | 163% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 14.957 ms | 17.95 | 0.42 | 171% | ok |
| blas3 | trmm | f64 | 512x512x512 | 7.521 ms | 17.85 | 0.70 | 170% | ok |
| blas3 | trsm | f64 | 512x512x512 | 8.137 ms | 16.50 | 0.64 | 157% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.269 ms | 25.47 | 0.50 | 122% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.265 ms | 0.14 | 1.11 | 11% | ok |
| factor | lu | f64 | 512x512 | 9.254 ms | 9.67 | 0.45 | 92% | ok |
| factor | lu | f32 | 512x512 | 6.321 ms | 14.16 | 0.33 | 68% | ok |
| factor | cholesky | f64 | 512x512 | 5.820 ms | 7.69 | 0.72 | 73% | ok |
| factor | cholesky | f32 | 512x512 | 3.896 ms | 11.48 | 0.54 | 55% | ok |
| factor | ldlt | f64 | 512x512 | 25.500 ms | 1.75 | 0.16 | 17% | ok |
| factor | ldlt_bk | f64 | 512x512 | 13.359 ms | 3.35 | 0.31 | 32% | ok |
| factor | qr | f64 | 512x512 | 19.958 ms | 8.97 | 0.21 | 86% | ok |
| factor | lq | f64 | 512x512 | 74.984 ms | 2.39 | 0.06 | 23% | ok |
| factor | hessenberg | f64 | 512x512 | 614.689 ms | 0.73 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.258 ms | 0.32 | 1.29 | 12%\* | ok |
| factor | inv | f64 | 512x512 | 129.693 ms | 2.07 | 0.03 | 20% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.887 ms | 6.94 | 0.33 | 66% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 6.256 ms | 3.58 | 0.17 | 34% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 172.461 ms | 0.88 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1002.267 ms | 0.15 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 53.769 ms | 3.12 | 0.02 | 30% | ok |
| eigen | schur | f64 | 256x256 | 72.185 ms | 5.81 | 0.01 | 55% | ok |
| eigen | svd | f64 | 256x256 | 29.665 ms | 12.44 | 0.04 | 119% | ok |
| eigen | expm | f64 | 256x256 | 35.852 ms | 7.02 | 0.03 | 67% | ok |
| eigen | sylvester | f64 | 256x256 | 192.807 ms | 5.22 | 0.01 | 50% | ok |
| elementwise | exp | f64 | n=8388608 | 159.244 ms | 0.05 | 0.84 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 139.871 ms | 0.06 | 0.96 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 169.619 ms | 0.05 | 0.79 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 90.266 ms | 0.09 | 1.49 | 14% | ok |
| elementwise | erf | f64 | n=8388608 | 2470.324 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.404 ms | 1.09 | 11.33 | 110% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.099 ms | 0.84 | 8.79 | 85% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.172 ms | 2.76 | 5.80 | 56% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.328 ms | 0.20 | 0.70 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.916 ms | 0.34 | 2.81 | 27% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 40.704 ms | 0.27 | 0.02 | 3% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 68.634 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 25.040 ms | 0.45 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 116.065 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 129.846 ms | 0.17 | 0.01 | 2% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 47.343 ms | 0.47 | 0.02 | 5% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 82.696 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.988 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.366 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 126.136 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.544 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 44.271 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.419 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.888 ms | 0.33 | 0.02 | 3% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.843 ms | 0.90 | 3.59 | 35%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.559 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.892 ms | 0.87 | 0.06 | 8% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.836 ms | 0.90 | 3.62 | 35%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.620 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 104.238 ms | 0.21 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.830 ms | 0.20 | 0.79 | 8%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.903 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 112.362 ms | 0.20 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.705 ms | 0.20 | 0.82 | 8%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 13.104 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 635.925 ms | 0.49 | 0.00 | 5% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 6.849 ms | 0.49 | 1.97 | 19%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 29.952 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 207.496 ms | 1.49 | 0.00 | 14% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 6.817 ms | 0.49 | 1.98 | 19%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.915 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1722.964 ms | 0.36 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.644 ms | 0.10 | 0.41 | 4%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 18.490 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1537.634 ms | 0.40 | 0.00 | 4% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.903 ms | 0.10 | 0.41 | 4%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 233.067 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 587.067 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 128.018 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 103.846 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 122.591 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 80.448 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 536.049 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 174.816 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.941 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 202.284 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 192.984 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 211.189 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 305.954 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.172 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.374 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 33.712 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.904 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.736 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.078 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 3.520 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.597 ms | 0.73 | 8.15 | 79% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.113 ms | 0.84 | 9.42 | 91% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.024 ms | 0.65 | 7.29 | 71% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 38.671 ms | 0.99 | 7.52 | 73% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 146.332 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 153.998 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 130.006 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.801 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 187.480 ms | 0.09 | 0.36 | 3% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 115.851 ms | 2.32 | 0.04 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.255 ms | 7.44 | 7.90 | 77% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.513 ms | 2.58 | 10.30 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.151 ms | 5.33 | 10.65 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.992 ms | 3.36 | 3.36 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.985 ms | 3.37 | 3.37 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 57.418 ms | 4.68 | 0.05 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 45.086 ms | 5.95 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 23.603 ms | 11.37 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 23.706 ms | 11.32 | 0.07 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 73.734 ms | 3.64 | 0.02 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

## xeon-e5-2420v2 -- blas, 6 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 59.0 GFLOP/s, f32 118.2 GFLOP/s |
| ... hardware, SSE2 | f64 59.0, f32 118.2 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 2.5 GFLOP/s |
| bandwidth | read 11.4, update 10.6, triad 8.5 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 53% | 107% | 0 | 83% |
| blas2 | 10 | 101% | 108% | 0 | 20% |
| blas3 | 10 | 129% | 158% | 0 | 77% |
| factor | 12 | 18% | 40% | 0 | 25% |
| eigen | 8 | 9% | 23% | 0 | 17% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 68% | 105% | 0 | 17% |
| direct | 19 | 1% | 31% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 4% | 17% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 74% | 83% | 0 | 17% |
| mg | 1 | 72% | 72% | 0 | 18% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 22% | 68% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 83% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 32.950 ms | 0.51 | 4.07 | 36% | ok | 6.0x (101%) |
| blas1 | dot | f32 | n=8388608 | 5.491 ms | 3.06 | 12.22 | 107% | ok | 5.0x (83%) |
| blas1 | axpy | f64 | n=8388608 | 38.410 ms | 0.44 | 5.24 | 46% | ok | 5.5x (92%) |
| blas1 | axpy | f32 | n=8388608 | 16.739 ms | 1.00 | 6.01 | 53% | ok | 5.6x (93%) |
| blas1 | scale | f64 | n=8388608 | 12.036 ms | 0.70 | 11.15 | 98% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 32.980 ms | 0.51 | 2.03 | 18% | ok | 5.9x (98%) |
| blas1 | one_norm | f64 | n=8388608 | 9.734 ms | 0.86 | 6.89 | 61% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.095 ms | 0.64 | 5.12 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.761 ms | 0.86 | 6.88 | 60% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 278.191 ms | 0.66 | 0.48 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 11.313 ms | 2.97 | 11.87 | 104% | ok | 1.3x (21%) |
| blas2 | gemv | f32 | n=4096 | 5.473 ms | 6.13 | 12.27 | 108% | ok | 1.1x (19%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 10.924 ms | 3.07 | 12.29 | 108% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.690 ms | 5.90 | 11.80 | 104% | ok | 1.2x (19%) |
| blas2 | symv | f64 | n=4096 | 5.829 ms | 5.76 | 11.52 | 101% | ok | 1.2x (20%) |
| blas2 | trmv | f64 | n=4096 | 5.927 ms | 2.83 | 11.33 | 100% | ok | 1.3x (21%) |
| blas2 | trmv | f32 | n=4096 | 3.019 ms | 5.56 | 11.12 | 98% | ok | 1.2x (21%) |
| blas2 | trsv | f64 | n=4096 | 7.785 ms | 2.16 | 8.63 | 76% | ok | 1.0x (16%) |
| blas2 | trsv | f32 | n=4096 | 3.376 ms | 4.97 | 9.95 | 87% | ok | 1.1x (18%) |
| blas2 | ger | f64 | n=4096 | 26.058 ms | 1.29 | 10.30 | 91% | ok | 1.1x (18%) |
| blas3 | gemm | f64 | 512x512x512 | 2.876 ms | 93.33 | 2.19 | 158% | ok | 4.9x (82%) |
| blas3 | gemm | f32 | 512x512x512 | 1.738 ms | 154.48 | 1.81 | 131% | ok | 4.6x (77%) |
| blas3 | gemmt | f64 | 512x512x512 | 2.532 ms | 53.01 | 2.48 | 90% | ok | 3.5x (58%) |
| blas3 | symm | f64 | 512x512x512 | 2.877 ms | 93.31 | 2.19 | 158% | ok | 4.9x (82%) |
| blas3 | syrk | f64 | 512x512x512 | 2.508 ms | 53.52 | 1.67 | 91% | ok | 3.1x (52%) |
| blas3 | syr2k | f64 | 512x512x512 | 3.731 ms | 71.95 | 1.69 | 122% | ok | 4.0x (67%) |
| blas3 | trmm | f64 | 512x512x512 | 1.617 ms | 83.03 | 3.24 | 141% | ok | 4.7x (78%) |
| blas3 | trsm | f64 | 512x512x512 | 1.761 ms | 76.21 | 2.98 | 129% | ok | 4.6x (77%) |
| blas3 | trsm | f32 | 512x512x512 | 1.238 ms | 108.44 | 2.12 | 92% | ok | 4.3x (71%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.230 ms | 0.14 | 1.11 | 10% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 5.524 ms | 16.20 | 0.76 | 27% | ok | 1.7x (28%) |
| factor | lu | f32 | 512x512 | 3.810 ms | 23.49 | 0.55 | 20% | ok | 1.7x (28%) |
| factor | cholesky | f64 | 512x512 | 4.101 ms | 10.91 | 1.02 | 18% | ok | 1.4x (24%) |
| factor | cholesky | f32 | 512x512 | 3.022 ms | 14.81 | 0.69 | 13% | ok | 1.3x (21%) |
| factor | ldlt | f64 | 512x512 | 25.511 ms | 1.75 | 0.16 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 13.476 ms | 3.32 | 0.31 | 6% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 13.387 ms | 13.37 | 0.31 | 23% | ok | 1.5x (25%) |
| factor | lq | f64 | 512x512 | 49.665 ms | 3.60 | 0.08 | 6% | ok | 1.5x (25%) |
| factor | hessenberg | f64 | 512x512 | 406.258 ms | 1.10 | 0.01 | 2% | ok | 1.5x (25%) |
| factor | householder | f64 | 512x512 | 0.915 ms | 1.15 | 4.59 | 40%\* | ok | 3.6x (59%) |
| factor | inv | f64 | 512x512 | 134.334 ms | 2.00 | 0.03 | 3% | ok | 1.0x (16%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 8.425 ms | 10.62 | 0.50 | 18% | ok | 1.5x (25%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 5.807 ms | 3.85 | 0.18 | 7% | ok | 1.1x (18%) |
| eigen | eigen_symmetric | f64 | 256x256 | 173.146 ms | 0.87 | 0.01 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1016.530 ms | 0.15 | 0.00 | 0% | ok | 1.0x (16%) |
| eigen | eigenvalue | f64 | 256x256 | 57.597 ms | 2.91 | 0.02 | 5% | ok | 0.9x (16%) |
| eigen | schur | f64 | 256x256 | 78.037 ms | 5.37 | 0.01 | 9% | ok | 0.9x (15%) |
| eigen | svd | f64 | 256x256 | 26.866 ms | 13.74 | 0.04 | 23% | ok | 1.1x (18%) |
| eigen | expm | f64 | 256x256 | 27.085 ms | 9.29 | 0.04 | 16% | ok | 1.3x (22%) |
| eigen | sylvester | f64 | 256x256 | 196.842 ms | 5.11 | 0.01 | 9% | ok | 1.0x (16%) |
| elementwise | exp | f64 | n=8388608 | 151.615 ms | 0.06 | 0.89 | 8% | ok | 1.1x (18%) |
| elementwise | log | f64 | n=8388608 | 139.619 ms | 0.06 | 0.96 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 174.750 ms | 0.05 | 0.77 | 7% | ok | 1.0x (16%) |
| elementwise | sqrt | f64 | n=8388608 | 89.523 ms | 0.09 | 1.50 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 2476.815 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.281 ms | 1.15 | 11.94 | 105% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.139 ms | 0.83 | 8.68 | 76% | ok | 1.0x (16%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.332 ms | 3.70 | 7.77 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.307 ms | 0.20 | 0.71 | 6%\* | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.943 ms | 0.34 | 2.80 | 25% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 40.583 ms | 0.28 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 68.805 ms | 0.16 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.406 ms | 0.42 | 0.03 | 1% | ok | 0.9x (16%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 117.664 ms | 0.19 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 131.719 ms | 0.17 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 57.437 ms | 0.39 | 0.01 | 1% | ok | 0.8x (14%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 82.619 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.942 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.152 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 124.832 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.538 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.807 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.206 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.335 ms | 0.34 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.856 ms | 0.88 | 3.53 | 31%\* | ok | 1.0x (16%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.618 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.542 ms | 0.89 | 0.06 | 2% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.875 ms | 0.86 | 3.46 | 30%\* | ok | 1.0x (16%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.354 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 105.132 ms | 0.21 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.873 ms | 0.20 | 0.78 | 7%\* | ok | 1.0x (16%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.812 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 116.375 ms | 0.19 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.840 ms | 0.20 | 0.79 | 7%\* | ok | 1.0x (16%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 12.898 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 635.465 ms | 0.49 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.030 ms | 0.48 | 1.92 | 17%\* | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.747 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 208.593 ms | 1.48 | 0.00 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.031 ms | 0.48 | 1.92 | 17%\* | ok | 1.0x (16%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.517 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1739.581 ms | 0.35 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.796 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 17.882 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1533.475 ms | 0.40 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.777 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 231.958 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 599.331 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 127.833 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 105.366 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 127.466 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 79.960 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 528.316 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 175.244 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.363 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 205.008 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 193.285 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 213.188 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 306.791 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.172 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.340 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 33.249 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 2.895 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 2.760 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.081 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 3.395 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.494 ms | 0.75 | 8.39 | 74% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.097 ms | 0.85 | 9.47 | 83% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.793 ms | 0.69 | 7.73 | 68% | ok | 1.1x (18%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.387 ms | 1.09 | 8.21 | 72% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 140.984 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 148.388 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 129.715 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.373 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 188.239 ms | 0.09 | 0.36 | 3% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 20.507 ms | 13.09 | 0.20 | 22% | ok | 5.6x (94%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.321 ms | 7.23 | 7.68 | 68% | ok | 1.0x (16%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.480 ms | 2.59 | 10.36 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.126 ms | 5.37 | 10.73 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.983 ms | 3.37 | 3.37 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.981 ms | 3.37 | 3.37 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.523 ms | 23.30 | 0.27 | n/a | ok | 5.0x (83%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 8.208 ms | 32.70 | 0.26 | n/a | ok | 5.5x (92%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.317 ms | 62.18 | 0.36 | n/a | ok | 5.5x (91%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.320 ms | 62.14 | 0.36 | n/a | ok | 5.5x (91%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.070 ms | 20.54 | 0.12 | n/a | ok | 5.6x (94%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

