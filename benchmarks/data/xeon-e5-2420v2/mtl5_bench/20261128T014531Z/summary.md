# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,6, CPUs 0,1,2,3,4,5, clock policy governor performance on every CPU, turbo on; performance 1200-2700 MHz on cpus 0,1,2,3,4,5, 2026-11-28T02:10:06Z.

## xeon-e5-2420v2 -- native, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 10.6 GFLOP/s, f32 21.0 GFLOP/s |
| ... hardware, SSE2 | f64 10.6, f32 21.0 GFLOP/s |
| ... this build, scalar | f64 0.1, f32 0.7 GFLOP/s |
| bandwidth | read 10.2, update 10.2, triad 8.1 GB/s (ceiling 10.2) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 24% | 106% | 0 |
| blas2 | 10 | 92% | 101% | 0 |
| blas3 | 10 | 38% | 39% | 0 |
| factor | 12 | 17% | 39% | 0 |
| eigen | 8 | 15% | 28% | 0 |
| elementwise | 5 | 9% | 15% | 0 |
| sparse | 5 | 57% | 105% | 0 |
| direct | 19 | 3% | 36% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 19% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 80% | 87% | 0 |
| mg | 1 | 79% | 79% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 22% | 75% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 183.083 ms | 0.09 | 0.73 | 7% | ok |
| blas1 | dot | f32 | n=8388608 | 27.800 ms | 0.60 | 2.41 | 24% | ok |
| blas1 | axpy | f64 | n=8388608 | 212.486 ms | 0.08 | 0.95 | 9% | ok |
| blas1 | axpy | f32 | n=8388608 | 91.388 ms | 0.18 | 1.10 | 11% | ok |
| blas1 | scale | f64 | n=8388608 | 12.402 ms | 0.68 | 10.82 | 106% | ok |
| blas1 | two_norm | f64 | n=8388608 | 184.651 ms | 0.09 | 0.36 | 4% | ok |
| blas1 | one_norm | f64 | n=8388608 | 9.703 ms | 0.86 | 6.92 | 68% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 13.040 ms | 0.64 | 5.15 | 50% | ok |
| blas1 | sum | f64 | n=8388608 | 9.744 ms | 0.86 | 6.89 | 67% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 276.407 ms | 0.67 | 0.49 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 13.546 ms | 2.48 | 9.91 | 97% | ok |
| blas2 | gemv | f32 | n=4096 | 8.528 ms | 3.93 | 7.87 | 77% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 13.248 ms | 2.53 | 10.14 | 99% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.505 ms | 5.16 | 10.32 | 101% | ok |
| blas2 | symv | f64 | n=4096 | 12.117 ms | 2.77 | 5.54 | 54% | ok |
| blas2 | trmv | f64 | n=4096 | 7.177 ms | 2.34 | 9.36 | 92% | ok |
| blas2 | trmv | f32 | n=4096 | 4.623 ms | 3.63 | 7.27 | 71% | ok |
| blas2 | trsv | f64 | n=4096 | 6.955 ms | 2.41 | 9.66 | 94% | ok |
| blas2 | trsv | f32 | n=4096 | 4.414 ms | 3.80 | 7.61 | 74% | ok |
| blas2 | ger | f64 | n=4096 | 28.484 ms | 1.18 | 9.43 | 92% | ok |
| blas3 | gemm | f64 | 512x512x512 | 65.157 ms | 4.12 | 0.10 | 39% | ok |
| blas3 | gemm | f32 | 512x512x512 | 34.578 ms | 7.76 | 0.09 | 37% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 36.231 ms | 3.70 | 0.17 | 35% | ok |
| blas3 | symm | f64 | 512x512x512 | 66.903 ms | 4.01 | 0.09 | 38% | ok |
| blas3 | syrk | f64 | 512x512x512 | 38.166 ms | 3.52 | 0.11 | 33% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 66.084 ms | 4.06 | 0.10 | 38% | ok |
| blas3 | trmm | f64 | 512x512x512 | 33.020 ms | 4.06 | 0.16 | 39% | ok |
| blas3 | trsm | f64 | 512x512x512 | 33.867 ms | 3.96 | 0.15 | 38% | ok |
| blas3 | trsm | f32 | 512x512x512 | 16.359 ms | 8.20 | 0.16 | 39% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 40.518 ms | 0.13 | 1.05 | 10% | ok |
| factor | lu | f64 | 512x512 | 30.984 ms | 2.89 | 0.14 | 27% | ok |
| factor | lu | f32 | 512x512 | 16.232 ms | 5.51 | 0.13 | 26% | ok |
| factor | cholesky | f64 | 512x512 | 25.150 ms | 1.78 | 0.17 | 17% | ok |
| factor | cholesky | f32 | 512x512 | 23.524 ms | 1.90 | 0.09 | 9% | ok |
| factor | ldlt | f64 | 512x512 | 25.880 ms | 1.73 | 0.16 | 16% | ok |
| factor | ldlt_bk | f64 | 512x512 | 13.324 ms | 3.36 | 0.31 | 32% | ok |
| factor | qr | f64 | 512x512 | 503.185 ms | 0.36 | 0.01 | 3% | ok |
| factor | lq | f64 | 512x512 | 73.786 ms | 2.43 | 0.06 | 23% | ok |
| factor | hessenberg | f64 | 512x512 | 610.949 ms | 0.73 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.304 ms | 0.32 | 1.27 | 12%\* | ok |
| factor | inv | f64 | 512x512 | 153.051 ms | 1.75 | 0.03 | 17% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 21.591 ms | 4.14 | 0.19 | 39% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 62.575 ms | 0.36 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 170.164 ms | 0.89 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 963.095 ms | 0.16 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 107.434 ms | 1.56 | 0.01 | 15% | ok |
| eigen | schur | f64 | 256x256 | 267.341 ms | 1.57 | 0.00 | 15% | ok |
| eigen | svd | f64 | 256x256 | 304.776 ms | 1.21 | 0.00 | 11% | ok |
| eigen | expm | f64 | 256x256 | 84.617 ms | 2.97 | 0.01 | 28% | ok |
| eigen | sylvester | f64 | 256x256 | 624.945 ms | 1.61 | 0.00 | 15% | ok |
| elementwise | exp | f64 | n=8388608 | 151.144 ms | 0.06 | 0.89 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 140.475 ms | 0.06 | 0.96 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 172.044 ms | 0.05 | 0.78 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 89.931 ms | 0.09 | 1.49 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 2470.273 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.544 ms | 1.03 | 10.70 | 105% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.097 ms | 0.85 | 8.79 | 86% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.067 ms | 2.78 | 5.84 | 57% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.633 ms | 0.20 | 0.69 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.861 ms | 0.34 | 2.84 | 28% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 40.547 ms | 0.28 | 0.02 | 3% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 69.023 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.743 ms | 0.45 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 115.273 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 131.357 ms | 0.17 | 0.01 | 2% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 47.941 ms | 0.47 | 0.02 | 4% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 81.691 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.898 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.398 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 125.382 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.527 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 41.598 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.176 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.540 ms | 0.33 | 0.02 | 3% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.814 ms | 0.93 | 3.71 | 36%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.279 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.663 ms | 0.88 | 0.06 | 8% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.821 ms | 0.92 | 3.68 | 36%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.384 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 110.366 ms | 0.20 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.879 ms | 0.19 | 0.78 | 8%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.771 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 112.525 ms | 0.20 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.888 ms | 0.19 | 0.78 | 8%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 13.009 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 633.912 ms | 0.49 | 0.00 | 5% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 6.806 ms | 0.49 | 1.98 | 19%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.932 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 205.685 ms | 1.50 | 0.00 | 14% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 6.796 ms | 0.50 | 1.98 | 19%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.481 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1729.147 ms | 0.36 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.661 ms | 0.10 | 0.41 | 4%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 18.061 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1519.532 ms | 0.41 | 0.00 | 4% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.459 ms | 0.10 | 0.42 | 4%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 231.567 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 406.632 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 135.926 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 109.289 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 129.005 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 81.666 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 533.424 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 175.409 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.799 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 202.377 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 192.997 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 212.933 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 308.674 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.170 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.442 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 33.292 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.930 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.801 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.098 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.144 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.585 ms | 0.73 | 8.18 | 80% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.293 ms | 0.79 | 8.91 | 87% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.904 ms | 0.67 | 7.51 | 73% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.939 ms | 1.07 | 8.09 | 79% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 140.637 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 145.145 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 132.673 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.668 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 186.719 ms | 0.09 | 0.36 | 4% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 113.890 ms | 2.36 | 0.04 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.325 ms | 7.22 | 7.67 | 75% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.458 ms | 2.60 | 10.39 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.145 ms | 5.33 | 10.67 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.986 ms | 3.36 | 3.36 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.974 ms | 3.37 | 3.37 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 57.734 ms | 4.65 | 0.05 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 45.169 ms | 5.94 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 23.659 ms | 11.35 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 23.606 ms | 11.37 | 0.07 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 70.389 ms | 3.81 | 0.02 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## xeon-e5-2420v2 -- native, 6 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 58.9 GFLOP/s, f32 118.2 GFLOP/s |
| ... hardware, SSE2 | f64 58.9, f32 118.2 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.7 GFLOP/s |
| bandwidth | read 11.4, update 10.7, triad 8.5 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 53% | 107% | 0 | 84% |
| blas2 | 10 | 85% | 107% | 0 | 17% |
| blas3 | 10 | 7% | 39% | 0 | 17% |
| factor | 12 | 6% | 40% | 0 | 22% |
| eigen | 8 | 3% | 11% | 0 | 17% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 68% | 105% | 0 | 17% |
| direct | 19 | 1% | 30% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 4% | 17% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 74% | 79% | 0 | 17% |
| mg | 1 | 72% | 72% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 22% | 69% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 84% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 32.745 ms | 0.51 | 4.10 | 36% | ok | 5.6x (93%) |
| blas1 | dot | f32 | n=8388608 | 5.492 ms | 3.05 | 12.22 | 107% | ok | 5.1x (84%) |
| blas1 | axpy | f64 | n=8388608 | 38.030 ms | 0.44 | 5.29 | 46% | ok | 5.6x (93%) |
| blas1 | axpy | f32 | n=8388608 | 16.746 ms | 1.00 | 6.01 | 53% | ok | 5.5x (91%) |
| blas1 | scale | f64 | n=8388608 | 12.144 ms | 0.69 | 11.05 | 97% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 33.137 ms | 0.51 | 2.03 | 18% | ok | 5.6x (93%) |
| blas1 | one_norm | f64 | n=8388608 | 9.719 ms | 0.86 | 6.91 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.065 ms | 0.64 | 5.14 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.712 ms | 0.86 | 6.91 | 61% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 277.277 ms | 0.67 | 0.48 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 10.946 ms | 3.07 | 12.27 | 107% | ok | 1.2x (21%) |
| blas2 | gemv | f32 | n=4096 | 5.498 ms | 6.10 | 12.21 | 107% | ok | 1.6x (26%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 11.235 ms | 2.99 | 11.95 | 105% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.480 ms | 6.12 | 12.25 | 107% | ok | 1.2x (20%) |
| blas2 | symv | f64 | n=4096 | 12.461 ms | 2.69 | 5.39 | 47% | ok | 1.0x (16%) |
| blas2 | trmv | f64 | n=4096 | 7.221 ms | 2.32 | 9.30 | 81% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 4.920 ms | 3.41 | 6.83 | 60% | ok | 0.9x (16%) |
| blas2 | trsv | f64 | n=4096 | 6.954 ms | 2.41 | 9.66 | 85% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 4.704 ms | 3.57 | 7.14 | 63% | ok | 0.9x (16%) |
| blas2 | ger | f64 | n=4096 | 28.553 ms | 1.18 | 9.40 | 82% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 11.546 ms | 23.25 | 0.54 | 39% | ok | 5.6x (94%) |
| blas3 | gemm | f32 | 512x512x512 | 6.252 ms | 42.94 | 0.50 | 36% | ok | 5.5x (92%) |
| blas3 | gemmt | f64 | 512x512x512 | 36.165 ms | 3.71 | 0.17 | 6% | ok | 1.0x (17%) |
| blas3 | symm | f64 | 512x512x512 | 65.870 ms | 4.08 | 0.10 | 7% | ok | 1.0x (17%) |
| blas3 | syrk | f64 | 512x512x512 | 38.103 ms | 3.52 | 0.11 | 6% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 512x512x512 | 65.772 ms | 4.08 | 0.10 | 7% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 512x512x512 | 32.449 ms | 4.14 | 0.16 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f64 | 512x512x512 | 33.187 ms | 4.04 | 0.16 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f32 | 512x512x512 | 17.096 ms | 7.85 | 0.15 | 7% | ok | 1.0x (16%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.329 ms | 0.14 | 1.11 | 10% | ok | 1.1x (18%) |
| factor | lu | f64 | 512x512 | 21.911 ms | 4.08 | 0.19 | 7% | ok | 1.4x (24%) |
| factor | lu | f32 | 512x512 | 12.106 ms | 7.39 | 0.17 | 6% | ok | 1.3x (22%) |
| factor | cholesky | f64 | 512x512 | 25.039 ms | 1.79 | 0.17 | 3% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 23.897 ms | 1.87 | 0.09 | 2% | ok | 1.0x (16%) |
| factor | ldlt | f64 | 512x512 | 25.403 ms | 1.76 | 0.17 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 13.460 ms | 3.32 | 0.31 | 6% | ok | 1.0x (16%) |
| factor | qr | f64 | 512x512 | 323.232 ms | 0.55 | 0.01 | 1% | ok | 1.6x (26%) |
| factor | lq | f64 | 512x512 | 49.135 ms | 3.64 | 0.09 | 6% | ok | 1.5x (25%) |
| factor | hessenberg | f64 | 512x512 | 401.253 ms | 1.11 | 0.01 | 2% | ok | 1.5x (25%) |
| factor | householder | f64 | 512x512 | 0.910 ms | 1.15 | 4.61 | 40%\* | ok | 3.6x (61%) |
| factor | inv | f64 | 512x512 | 143.211 ms | 1.87 | 0.03 | 3% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.975 ms | 5.27 | 0.25 | 9% | ok | 1.3x (21%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.981 ms | 0.36 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 172.488 ms | 0.88 | 0.01 | 1% | ok | 1.0x (16%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 989.790 ms | 0.15 | 0.00 | 0% | ok | 1.0x (16%) |
| eigen | eigenvalue | f64 | 256x256 | 107.657 ms | 1.56 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 264.611 ms | 1.59 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 302.166 ms | 1.22 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 37.624 ms | 6.69 | 0.03 | 11% | ok | 2.2x (37%) |
| eigen | sylvester | f64 | 256x256 | 595.989 ms | 1.69 | 0.00 | 3% | ok | 1.0x (17%) |
| elementwise | exp | f64 | n=8388608 | 152.448 ms | 0.06 | 0.88 | 8% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 144.359 ms | 0.06 | 0.93 | 8% | ok | 1.0x (16%) |
| elementwise | sin | f64 | n=8388608 | 170.535 ms | 0.05 | 0.79 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 90.137 ms | 0.09 | 1.49 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 2479.223 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.271 ms | 1.15 | 11.99 | 105% | ok | 1.1x (19%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.121 ms | 0.84 | 8.73 | 76% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.343 ms | 3.69 | 7.76 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.590 ms | 0.20 | 0.69 | 6%\* | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.953 ms | 0.34 | 2.80 | 24% | ok | 1.0x (16%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 40.981 ms | 0.27 | 0.02 | 0% | ok | 1.0x (16%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 68.613 ms | 0.16 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.964 ms | 0.45 | 0.03 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 124.298 ms | 0.18 | 0.01 | 0% | ok | 0.9x (15%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 131.575 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 48.252 ms | 0.46 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 82.463 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.924 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.188 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 125.979 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.524 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 44.278 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.238 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.247 ms | 0.34 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.872 ms | 0.87 | 3.47 | 30%\* | ok | 0.9x (16%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.363 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.483 ms | 0.90 | 0.06 | 2% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.908 ms | 0.83 | 3.33 | 29%\* | ok | 0.9x (15%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.283 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 103.981 ms | 0.22 | 0.01 | 0% | ok | 1.1x (18%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.767 ms | 0.20 | 0.80 | 7%\* | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.789 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 114.934 ms | 0.19 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.773 ms | 0.20 | 0.80 | 7%\* | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 12.788 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 633.652 ms | 0.49 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 6.968 ms | 0.48 | 1.93 | 17%\* | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.870 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 204.500 ms | 1.51 | 0.00 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.042 ms | 0.48 | 1.91 | 17%\* | ok | 1.0x (16%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.410 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1712.214 ms | 0.36 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.554 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 18.042 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1527.563 ms | 0.40 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.544 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 230.933 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 401.282 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 127.825 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 103.319 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 121.844 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 80.536 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 533.111 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 174.165 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 90.975 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 201.302 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 194.830 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 211.890 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 310.407 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.184 ms | 0.00 | 0.00 | n/a | ok | 0.9x (15%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.380 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 32.697 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 2.874 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 2.723 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.093 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.197 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.469 ms | 0.75 | 8.45 | 74% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.239 ms | 0.81 | 9.05 | 79% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.850 ms | 0.68 | 7.62 | 67% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.589 ms | 1.08 | 8.17 | 72% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 141.479 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 146.627 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 127.291 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.450 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 187.857 ms | 0.09 | 0.36 | 3% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 20.522 ms | 13.08 | 0.20 | 22% | ok | 5.5x (92%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.277 ms | 7.37 | 7.83 | 69% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.400 ms | 2.62 | 10.49 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.114 ms | 5.39 | 10.77 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.977 ms | 3.37 | 3.37 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.980 ms | 3.37 | 3.37 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.514 ms | 23.31 | 0.27 | n/a | ok | 5.0x (84%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 8.247 ms | 32.55 | 0.25 | n/a | ok | 5.5x (91%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.324 ms | 62.07 | 0.36 | n/a | ok | 5.5x (91%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.328 ms | 62.03 | 0.36 | n/a | ok | 5.5x (91%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.067 ms | 20.54 | 0.12 | n/a | ok | 5.4x (90%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## xeon-e5-2420v2 -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 21.1 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 21.1 GFLOP/s |
| ... this build, SSE4 | f64 10.5, f32 21.0 GFLOP/s |
| bandwidth | read 10.5, update 10.2, triad 8.1 GB/s (ceiling 10.5) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 102% | 108% | 0 |
| blas2 | 10 | 93% | 99% | 0 |
| blas3 | 10 | 76% | 95% | 0 |
| factor | 12 | 17% | 40% | 0 |
| eigen | 8 | 15% | 47% | 0 |
| elementwise | 5 | 9% | 14% | 0 |
| sparse | 5 | 56% | 106% | 0 |
| direct | 19 | 3% | 36% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 19% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 79% | 85% | 0 |
| mg | 1 | 78% | 78% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 73% | 108% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.881 ms | 1.41 | 11.30 | 108% | ok |
| blas1 | dot | f32 | n=8388608 | 5.954 ms | 2.82 | 11.27 | 108% | ok |
| blas1 | axpy | f64 | n=8388608 | 18.748 ms | 0.89 | 10.74 | 103% | ok |
| blas1 | axpy | f32 | n=8388608 | 9.724 ms | 1.73 | 10.35 | 99% | ok |
| blas1 | scale | f64 | n=8388608 | 12.593 ms | 0.67 | 10.66 | 102% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.934 ms | 2.83 | 11.31 | 108% | ok |
| blas1 | one_norm | f64 | n=8388608 | 9.734 ms | 0.86 | 6.89 | 66% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 12.998 ms | 0.65 | 5.16 | 49% | ok |
| blas1 | sum | f64 | n=8388608 | 9.732 ms | 0.86 | 6.90 | 66% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 271.039 ms | 0.68 | 0.50 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 12.926 ms | 2.60 | 10.39 | 99% | ok |
| blas2 | gemv | f32 | n=4096 | 6.460 ms | 5.19 | 10.39 | 99% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 13.062 ms | 2.57 | 10.28 | 98% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.456 ms | 5.20 | 10.40 | 99% | ok |
| blas2 | symv | f64 | n=4096 | 12.141 ms | 2.76 | 5.53 | 53% | ok |
| blas2 | trmv | f64 | n=4096 | 6.877 ms | 2.44 | 9.77 | 93% | ok |
| blas2 | trmv | f32 | n=4096 | 3.561 ms | 4.71 | 9.43 | 90% | ok |
| blas2 | trsv | f64 | n=4096 | 6.883 ms | 2.44 | 9.76 | 93% | ok |
| blas2 | trsv | f32 | n=4096 | 3.624 ms | 4.63 | 9.27 | 88% | ok |
| blas2 | ger | f64 | n=4096 | 28.113 ms | 1.19 | 9.55 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 27.647 ms | 9.71 | 0.23 | 92% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.443 ms | 19.97 | 0.23 | 95% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 16.984 ms | 7.90 | 0.37 | 75% | ok |
| blas3 | symm | f64 | 512x512x512 | 28.876 ms | 9.30 | 0.22 | 88% | ok |
| blas3 | syrk | f64 | 512x512x512 | 16.758 ms | 8.01 | 0.25 | 76% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 33.484 ms | 8.02 | 0.19 | 76% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.618 ms | 7.62 | 0.30 | 72% | ok |
| blas3 | trsm | f64 | 512x512x512 | 18.328 ms | 7.32 | 0.29 | 70% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.825 ms | 15.21 | 0.30 | 72% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.406 ms | 0.14 | 1.11 | 11% | ok |
| factor | lu | f64 | 512x512 | 29.208 ms | 3.06 | 0.14 | 29% | ok |
| factor | lu | f32 | 512x512 | 15.522 ms | 5.76 | 0.14 | 27% | ok |
| factor | cholesky | f64 | 512x512 | 25.500 ms | 1.75 | 0.16 | 17% | ok |
| factor | cholesky | f32 | 512x512 | 23.941 ms | 1.87 | 0.09 | 9% | ok |
| factor | ldlt | f64 | 512x512 | 25.968 ms | 1.72 | 0.16 | 16% | ok |
| factor | ldlt_bk | f64 | 512x512 | 12.846 ms | 3.48 | 0.33 | 33% | ok |
| factor | qr | f64 | 512x512 | 497.670 ms | 0.36 | 0.01 | 3% | ok |
| factor | lq | f64 | 512x512 | 73.912 ms | 2.42 | 0.06 | 23% | ok |
| factor | hessenberg | f64 | 512x512 | 614.568 ms | 0.73 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.225 ms | 0.33 | 1.30 | 12%\* | ok |
| factor | inv | f64 | 512x512 | 155.320 ms | 1.73 | 0.03 | 16% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 21.135 ms | 4.23 | 0.20 | 40% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 62.956 ms | 0.36 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 172.746 ms | 0.87 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 990.613 ms | 0.15 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 106.089 ms | 1.58 | 0.01 | 15% | ok |
| eigen | schur | f64 | 256x256 | 261.304 ms | 1.61 | 0.00 | 15% | ok |
| eigen | svd | f64 | 256x256 | 261.081 ms | 1.41 | 0.00 | 13% | ok |
| eigen | expm | f64 | 256x256 | 50.650 ms | 4.97 | 0.02 | 47% | ok |
| eigen | sylvester | f64 | 256x256 | 590.504 ms | 1.70 | 0.00 | 16% | ok |
| elementwise | exp | f64 | n=8388608 | 150.444 ms | 0.06 | 0.89 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 139.358 ms | 0.06 | 0.96 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 170.085 ms | 0.05 | 0.79 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 89.720 ms | 0.09 | 1.50 | 14% | ok |
| elementwise | erf | f64 | n=8388608 | 2523.353 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.453 ms | 1.07 | 11.10 | 106% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.134 ms | 0.84 | 8.69 | 83% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.113 ms | 2.77 | 5.83 | 56% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.080 ms | 0.20 | 0.72 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.796 ms | 0.35 | 2.87 | 27% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 41.341 ms | 0.27 | 0.02 | 3% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 69.586 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 25.124 ms | 0.45 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 119.016 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 132.841 ms | 0.17 | 0.01 | 2% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 51.103 ms | 0.44 | 0.01 | 4% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 84.045 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.476 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.813 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 128.670 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.546 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.884 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.552 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.992 ms | 0.33 | 0.02 | 3% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.809 ms | 0.93 | 3.74 | 36%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.657 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.879 ms | 0.87 | 0.06 | 8% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.828 ms | 0.91 | 3.65 | 35%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.487 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 106.131 ms | 0.21 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.875 ms | 0.20 | 0.78 | 7%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.967 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 114.425 ms | 0.20 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.843 ms | 0.20 | 0.79 | 8%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 13.449 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 636.992 ms | 0.48 | 0.00 | 5% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 6.928 ms | 0.49 | 1.95 | 19%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 29.378 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 212.236 ms | 1.45 | 0.00 | 14% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 6.799 ms | 0.50 | 1.98 | 19%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.816 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1730.439 ms | 0.36 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 33.864 ms | 0.10 | 0.40 | 4%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 18.254 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1952.851 ms | 0.32 | 0.00 | 3% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.931 ms | 0.10 | 0.41 | 4%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 81.320 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 133.167 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 60.276 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 32.009 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 38.510 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 37.290 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 65.713 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 69.394 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 37.587 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 43.796 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 40.194 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 38.623 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 45.599 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.129 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.613 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 33.303 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.912 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.774 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.092 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.132 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.556 ms | 0.74 | 8.25 | 79% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.285 ms | 0.80 | 8.93 | 85% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.932 ms | 0.67 | 7.46 | 71% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.736 ms | 1.08 | 8.13 | 78% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 138.937 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 145.349 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 128.362 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 15.918 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 5.933 ms | 2.83 | 11.31 | 108% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 61.740 ms | 4.35 | 0.07 | 41% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.323 ms | 7.22 | 7.67 | 73% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.049 ms | 2.77 | 11.10 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.968 ms | 5.65 | 11.30 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.082 ms | 8.06 | 8.06 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.646 ms | 10.19 | 10.19 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 18.731 ms | 14.33 | 0.17 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 22.068 ms | 12.16 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 22.072 ms | 12.16 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 16.150 ms | 16.62 | 0.10 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 12.335 ms | 21.76 | 0.13 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## xeon-e5-2420v2 -- native-fast, 6 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 59.0 GFLOP/s, f32 118.1 GFLOP/s |
| ... hardware, SSE2 | f64 59.0, f32 118.1 GFLOP/s |
| ... this build, SSE4 | f64 58.9, f32 118.0 GFLOP/s |
| bandwidth | read 11.4, update 10.7, triad 8.5 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 99% | 108% | 0 | 18% |
| blas2 | 10 | 86% | 108% | 0 | 17% |
| blas3 | 10 | 34% | 51% | 0 | 42% |
| factor | 12 | 6% | 40% | 0 | 22% |
| eigen | 8 | 3% | 13% | 0 | 17% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 68% | 105% | 0 | 17% |
| direct | 19 | 1% | 31% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 4% | 17% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 72% | 77% | 0 | 17% |
| mg | 1 | 72% | 72% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 68% | 93% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 79% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 10.873 ms | 1.54 | 12.34 | 108% | ok | 1.1x (18%) |
| blas1 | dot | f32 | n=8388608 | 5.439 ms | 3.08 | 12.34 | 108% | ok | 1.1x (18%) |
| blas1 | axpy | f64 | n=8388608 | 17.833 ms | 0.94 | 11.29 | 99% | ok | 1.1x (18%) |
| blas1 | axpy | f32 | n=8388608 | 8.430 ms | 1.99 | 11.94 | 105% | ok | 1.2x (19%) |
| blas1 | scale | f64 | n=8388608 | 12.091 ms | 0.69 | 11.10 | 97% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 5.423 ms | 3.09 | 12.37 | 108% | ok | 1.1x (18%) |
| blas1 | one_norm | f64 | n=8388608 | 9.715 ms | 0.86 | 6.91 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.050 ms | 0.64 | 5.14 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.727 ms | 0.86 | 6.90 | 60% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 269.923 ms | 0.68 | 0.50 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 10.902 ms | 3.08 | 12.32 | 108% | ok | 1.2x (20%) |
| blas2 | gemv | f32 | n=4096 | 5.453 ms | 6.15 | 12.31 | 108% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 11.056 ms | 3.03 | 12.15 | 106% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.812 ms | 5.77 | 11.55 | 101% | ok | 1.1x (19%) |
| blas2 | symv | f64 | n=4096 | 12.188 ms | 2.75 | 5.51 | 48% | ok | 1.0x (17%) |
| blas2 | trmv | f64 | n=4096 | 6.844 ms | 2.45 | 9.81 | 86% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 3.584 ms | 4.68 | 9.37 | 82% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 6.898 ms | 2.43 | 9.74 | 85% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 3.601 ms | 4.66 | 9.33 | 82% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 28.566 ms | 1.17 | 9.40 | 82% | ok | 1.0x (16%) |
| blas3 | gemm | f64 | 512x512x512 | 8.835 ms | 30.38 | 0.71 | 51% | ok | 3.1x (52%) |
| blas3 | gemm | f32 | 512x512x512 | 4.482 ms | 59.89 | 0.70 | 51% | ok | 3.0x (50%) |
| blas3 | gemmt | f64 | 512x512x512 | 6.849 ms | 19.60 | 0.92 | 33% | ok | 2.5x (41%) |
| blas3 | symm | f64 | 512x512x512 | 9.929 ms | 27.03 | 0.63 | 46% | ok | 2.9x (48%) |
| blas3 | syrk | f64 | 512x512x512 | 6.696 ms | 20.04 | 0.63 | 34% | ok | 2.5x (42%) |
| blas3 | syr2k | f64 | 512x512x512 | 13.173 ms | 20.38 | 0.48 | 35% | ok | 2.5x (42%) |
| blas3 | trmm | f64 | 512x512x512 | 9.296 ms | 14.44 | 0.56 | 24% | ok | 1.9x (32%) |
| blas3 | trsm | f64 | 512x512x512 | 8.361 ms | 16.05 | 0.63 | 27% | ok | 2.2x (37%) |
| blas3 | trsm | f32 | 512x512x512 | 4.165 ms | 32.23 | 0.63 | 27% | ok | 2.1x (35%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 37.931 ms | 0.14 | 1.12 | 10% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 21.023 ms | 4.26 | 0.20 | 7% | ok | 1.4x (23%) |
| factor | lu | f32 | 512x512 | 11.714 ms | 7.64 | 0.18 | 6% | ok | 1.3x (22%) |
| factor | cholesky | f64 | 512x512 | 25.783 ms | 1.74 | 0.16 | 3% | ok | 1.0x (16%) |
| factor | cholesky | f32 | 512x512 | 24.366 ms | 1.84 | 0.09 | 2% | ok | 1.0x (16%) |
| factor | ldlt | f64 | 512x512 | 26.044 ms | 1.72 | 0.16 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 12.769 ms | 3.50 | 0.33 | 6% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 328.509 ms | 0.54 | 0.01 | 1% | ok | 1.5x (25%) |
| factor | lq | f64 | 512x512 | 49.131 ms | 3.64 | 0.09 | 6% | ok | 1.5x (25%) |
| factor | hessenberg | f64 | 512x512 | 400.302 ms | 1.12 | 0.01 | 2% | ok | 1.5x (26%) |
| factor | householder | f64 | 512x512 | 0.916 ms | 1.15 | 4.58 | 40%\* | ok | 3.5x (59%) |
| factor | inv | f64 | 512x512 | 147.875 ms | 1.82 | 0.03 | 3% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.899 ms | 5.29 | 0.25 | 9% | ok | 1.3x (21%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 60.387 ms | 0.37 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 172.515 ms | 0.88 | 0.01 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 999.956 ms | 0.15 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 105.214 ms | 1.59 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 260.339 ms | 1.61 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 258.015 ms | 1.43 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 31.919 ms | 7.88 | 0.03 | 13% | ok | 1.6x (26%) |
| eigen | sylvester | f64 | 256x256 | 570.675 ms | 1.76 | 0.00 | 3% | ok | 1.0x (17%) |
| elementwise | exp | f64 | n=8388608 | 152.139 ms | 0.06 | 0.88 | 8% | ok | 1.0x (16%) |
| elementwise | log | f64 | n=8388608 | 140.983 ms | 0.06 | 0.95 | 8% | ok | 1.0x (16%) |
| elementwise | sin | f64 | n=8388608 | 170.768 ms | 0.05 | 0.79 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 89.415 ms | 0.09 | 1.50 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 2472.308 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.272 ms | 1.15 | 11.98 | 105% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.110 ms | 0.84 | 8.75 | 77% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.317 ms | 3.70 | 7.78 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.087 ms | 0.20 | 0.71 | 6%\* | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.904 ms | 0.34 | 2.82 | 25% | ok | 1.0x (16%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 41.856 ms | 0.27 | 0.02 | 0% | ok | 1.0x (16%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 70.224 ms | 0.16 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 25.218 ms | 0.44 | 0.03 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 117.340 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 130.631 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 47.876 ms | 0.47 | 0.02 | 1% | ok | 1.1x (18%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 79.303 ms | 0.02 | 0.00 | 0% | ok | 1.1x (18%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.184 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.280 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 128.261 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.544 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.921 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.563 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.316 ms | 0.34 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.859 ms | 0.88 | 3.52 | 31%\* | ok | 0.9x (16%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.685 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.743 ms | 0.88 | 0.06 | 1% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.881 ms | 0.86 | 3.43 | 30%\* | ok | 0.9x (16%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.473 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 104.640 ms | 0.21 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.750 ms | 0.20 | 0.81 | 7%\* | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.902 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 113.608 ms | 0.20 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.771 ms | 0.20 | 0.80 | 7%\* | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 13.308 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 654.943 ms | 0.47 | 0.00 | 1% | ok | 1.0x (16%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.140 ms | 0.47 | 1.89 | 17%\* | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.991 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 205.799 ms | 1.50 | 0.00 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.077 ms | 0.48 | 1.90 | 17%\* | ok | 1.0x (16%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.796 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1751.501 ms | 0.35 | 0.00 | 1% | ok | 1.0x (16%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.669 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 18.149 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1964.629 ms | 0.31 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.500 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 80.317 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 122.997 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 59.380 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 31.881 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 38.279 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 37.055 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 65.311 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 69.632 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 37.368 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 43.460 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 39.876 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 38.635 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 44.963 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.129 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.644 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 33.284 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 2.876 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 2.779 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.093 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.189 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.589 ms | 0.73 | 8.17 | 72% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.315 ms | 0.79 | 8.85 | 77% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.921 ms | 0.67 | 7.48 | 65% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.565 ms | 1.08 | 8.17 | 72% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 139.140 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 144.651 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 130.203 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 16.766 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| mixed | dot | f32>f64 | n=8388608 | 6.318 ms | 2.66 | 10.62 | 93% | ok | 0.9x (16%) |
| mixed | gemm | f32>f64 | 512x512x512 | 11.758 ms | 22.83 | 0.36 | 39% | ok | 5.3x (88%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.284 ms | 7.34 | 7.80 | 68% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.037 ms | 2.78 | 11.12 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.972 ms | 5.64 | 11.29 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.083 ms | 8.06 | 8.06 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.637 ms | 10.25 | 10.25 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 3.946 ms | 68.03 | 0.80 | n/a | ok | 4.7x (79%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.528 ms | 59.28 | 0.46 | n/a | ok | 4.9x (81%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.491 ms | 59.78 | 0.35 | n/a | ok | 4.9x (82%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.315 ms | 80.98 | 0.47 | n/a | ok | 4.9x (81%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 2.565 ms | 104.67 | 0.61 | n/a | ok | 4.8x (80%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## xeon-e5-2420v2 -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 20.9 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 20.9 GFLOP/s |
| ... this build, scalar | f64 0.1, f32 0.7 GFLOP/s |
| bandwidth | read 10.2, update 10.3, triad 8.2 GB/s (ceiling 10.3) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 23% | 104% | 0 |
| blas2 | 10 | 90% | 103% | 0 |
| blas3 | 10 | 164% | 182% | 0 |
| factor | 12 | 55% | 93% | 0 |
| eigen | 8 | 51% | 120% | 0 |
| elementwise | 5 | 9% | 14% | 0 |
| sparse | 5 | 56% | 108% | 0 |
| direct | 19 | 3% | 37% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 19% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 80% | 86% | 0 |
| mg | 1 | 77% | 77% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 21% | 73% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 187.237 ms | 0.09 | 0.72 | 7% | ok |
| blas1 | dot | f32 | n=8388608 | 27.846 ms | 0.60 | 2.41 | 23% | ok |
| blas1 | axpy | f64 | n=8388608 | 212.495 ms | 0.08 | 0.95 | 9% | ok |
| blas1 | axpy | f32 | n=8388608 | 91.608 ms | 0.18 | 1.10 | 11% | ok |
| blas1 | scale | f64 | n=8388608 | 12.532 ms | 0.67 | 10.71 | 104% | ok |
| blas1 | two_norm | f64 | n=8388608 | 185.889 ms | 0.09 | 0.36 | 4% | ok |
| blas1 | one_norm | f64 | n=8388608 | 9.927 ms | 0.85 | 6.76 | 66% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 13.130 ms | 0.64 | 5.11 | 50% | ok |
| blas1 | sum | f64 | n=8388608 | 9.760 ms | 0.86 | 6.88 | 67% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 278.968 ms | 0.66 | 0.48 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 14.513 ms | 2.31 | 9.25 | 90% | ok |
| blas2 | gemv | f32 | n=4096 | 6.323 ms | 5.31 | 10.62 | 103% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 13.311 ms | 2.52 | 10.09 | 98% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.777 ms | 4.95 | 9.91 | 96% | ok |
| blas2 | symv | f64 | n=4096 | 7.234 ms | 4.64 | 9.29 | 90% | ok |
| blas2 | trmv | f64 | n=4096 | 7.632 ms | 2.20 | 8.80 | 85% | ok |
| blas2 | trmv | f32 | n=4096 | 3.755 ms | 4.47 | 8.94 | 87% | ok |
| blas2 | trsv | f64 | n=4096 | 7.535 ms | 2.23 | 8.91 | 87% | ok |
| blas2 | trsv | f32 | n=4096 | 3.694 ms | 4.54 | 9.09 | 88% | ok |
| blas2 | ger | f64 | n=4096 | 28.796 ms | 1.17 | 9.32 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 14.058 ms | 19.09 | 0.45 | 182% | ok |
| blas3 | gemm | f32 | 512x512x512 | 9.007 ms | 29.80 | 0.35 | 142% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 8.995 ms | 14.92 | 0.70 | 143% | ok |
| blas3 | symm | f64 | 512x512x512 | 14.145 ms | 18.98 | 0.44 | 181% | ok |
| blas3 | syrk | f64 | 512x512x512 | 7.798 ms | 17.21 | 0.54 | 164% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 15.214 ms | 17.64 | 0.41 | 169% | ok |
| blas3 | trmm | f64 | 512x512x512 | 7.569 ms | 17.73 | 0.69 | 169% | ok |
| blas3 | trsm | f64 | 512x512x512 | 8.162 ms | 16.44 | 0.64 | 157% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.293 ms | 25.36 | 0.50 | 121% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.502 ms | 0.14 | 1.10 | 11% | ok |
| factor | lu | f64 | 512x512 | 9.234 ms | 9.69 | 0.45 | 93% | ok |
| factor | lu | f32 | 512x512 | 6.375 ms | 14.04 | 0.33 | 67% | ok |
| factor | cholesky | f64 | 512x512 | 5.758 ms | 7.77 | 0.73 | 74% | ok |
| factor | cholesky | f32 | 512x512 | 3.896 ms | 11.48 | 0.54 | 55% | ok |
| factor | ldlt | f64 | 512x512 | 25.018 ms | 1.79 | 0.17 | 17% | ok |
| factor | ldlt_bk | f64 | 512x512 | 13.592 ms | 3.29 | 0.31 | 31% | ok |
| factor | qr | f64 | 512x512 | 19.600 ms | 9.13 | 0.21 | 87% | ok |
| factor | lq | f64 | 512x512 | 73.982 ms | 2.42 | 0.06 | 23% | ok |
| factor | hessenberg | f64 | 512x512 | 621.571 ms | 0.72 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.247 ms | 0.32 | 1.29 | 13%\* | ok |
| factor | inv | f64 | 512x512 | 129.634 ms | 2.07 | 0.03 | 20% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.862 ms | 6.96 | 0.33 | 66% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 6.475 ms | 3.46 | 0.16 | 33% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 174.060 ms | 0.87 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1007.853 ms | 0.15 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 53.747 ms | 3.12 | 0.02 | 30% | ok |
| eigen | schur | f64 | 256x256 | 72.464 ms | 5.79 | 0.01 | 55% | ok |
| eigen | svd | f64 | 256x256 | 29.448 ms | 12.53 | 0.04 | 120% | ok |
| eigen | expm | f64 | 256x256 | 35.852 ms | 7.02 | 0.03 | 67% | ok |
| eigen | sylvester | f64 | 256x256 | 190.475 ms | 5.28 | 0.01 | 51% | ok |
| elementwise | exp | f64 | n=8388608 | 152.711 ms | 0.05 | 0.88 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 139.360 ms | 0.06 | 0.96 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 170.365 ms | 0.05 | 0.79 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 92.280 ms | 0.09 | 1.45 | 14% | ok |
| elementwise | erf | f64 | n=8388608 | 2504.795 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.451 ms | 1.07 | 11.11 | 108% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.126 ms | 0.84 | 8.71 | 85% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.219 ms | 2.75 | 5.79 | 56% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.250 ms | 0.20 | 0.71 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.825 ms | 0.34 | 2.86 | 28% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 40.706 ms | 0.27 | 0.02 | 3% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 68.958 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.665 ms | 0.45 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 113.940 ms | 0.20 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 128.442 ms | 0.17 | 0.01 | 2% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 47.549 ms | 0.47 | 0.02 | 4% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 81.632 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.843 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.143 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 125.303 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.563 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 41.492 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.225 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.274 ms | 0.34 | 0.02 | 3% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.802 ms | 0.94 | 3.77 | 37%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.318 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.452 ms | 0.90 | 0.06 | 9% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.818 ms | 0.92 | 3.70 | 36%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.350 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 102.271 ms | 0.22 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.817 ms | 0.20 | 0.79 | 8%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.821 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 110.685 ms | 0.20 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.847 ms | 0.20 | 0.79 | 8%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 12.972 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 644.048 ms | 0.48 | 0.00 | 5% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 6.842 ms | 0.49 | 1.97 | 19%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.865 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 208.371 ms | 1.48 | 0.00 | 14% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 6.892 ms | 0.49 | 1.96 | 19%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.456 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1769.780 ms | 0.35 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.871 ms | 0.10 | 0.41 | 4%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 18.253 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1574.807 ms | 0.39 | 0.00 | 4% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.562 ms | 0.10 | 0.41 | 4%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 242.856 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 407.369 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 128.964 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 104.488 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 123.363 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 80.615 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 536.398 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 174.944 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.426 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 203.412 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 195.741 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 214.663 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 307.864 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.172 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.371 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 32.719 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.870 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.736 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.092 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 3.395 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.566 ms | 0.73 | 8.22 | 80% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.294 ms | 0.79 | 8.90 | 86% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.937 ms | 0.66 | 7.45 | 72% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 36.728 ms | 1.05 | 7.91 | 77% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 137.803 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 145.067 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 129.419 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.391 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 188.540 ms | 0.09 | 0.36 | 3% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 119.769 ms | 2.24 | 0.04 | 21% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.372 ms | 7.07 | 7.52 | 73% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.467 ms | 2.59 | 10.38 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.125 ms | 5.37 | 10.74 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.978 ms | 3.37 | 3.37 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.983 ms | 3.37 | 3.37 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 59.039 ms | 4.55 | 0.05 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 45.196 ms | 5.94 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 23.667 ms | 11.34 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 23.804 ms | 11.28 | 0.07 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 70.096 ms | 3.83 | 0.02 | n/a | ok |

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
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 59.0 GFLOP/s, f32 118.1 GFLOP/s |
| ... hardware, SSE2 | f64 59.0, f32 118.1 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.7 GFLOP/s |
| bandwidth | read 11.4, update 10.6, triad 8.4 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 53% | 107% | 0 | 84% |
| blas2 | 10 | 103% | 108% | 0 | 20% |
| blas3 | 10 | 126% | 158% | 0 | 77% |
| factor | 12 | 18% | 40% | 0 | 25% |
| eigen | 8 | 9% | 23% | 0 | 17% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 69% | 106% | 0 | 17% |
| direct | 19 | 1% | 31% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 4% | 17% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 74% | 81% | 0 | 17% |
| mg | 1 | 70% | 70% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 16% |
| mixed | 3 | 22% | 68% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 85% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 33.511 ms | 0.50 | 4.01 | 35% | ok | 5.6x (93%) |
| blas1 | dot | f32 | n=8388608 | 5.513 ms | 3.04 | 12.17 | 107% | ok | 5.1x (84%) |
| blas1 | axpy | f64 | n=8388608 | 38.978 ms | 0.43 | 5.17 | 45% | ok | 5.5x (91%) |
| blas1 | axpy | f32 | n=8388608 | 16.765 ms | 1.00 | 6.00 | 53% | ok | 5.5x (91%) |
| blas1 | scale | f64 | n=8388608 | 12.100 ms | 0.69 | 11.09 | 98% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 33.078 ms | 0.51 | 2.03 | 18% | ok | 5.6x (94%) |
| blas1 | one_norm | f64 | n=8388608 | 9.783 ms | 0.86 | 6.86 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.026 ms | 0.64 | 5.15 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.704 ms | 0.86 | 6.92 | 61% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 279.009 ms | 0.66 | 0.48 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 11.290 ms | 2.97 | 11.89 | 105% | ok | 1.3x (21%) |
| blas2 | gemv | f32 | n=4096 | 5.462 ms | 6.14 | 12.29 | 108% | ok | 1.2x (19%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 10.931 ms | 3.07 | 12.28 | 108% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.691 ms | 5.90 | 11.80 | 104% | ok | 1.2x (20%) |
| blas2 | symv | f64 | n=4096 | 5.791 ms | 5.79 | 11.60 | 102% | ok | 1.2x (21%) |
| blas2 | trmv | f64 | n=4096 | 5.723 ms | 2.93 | 11.74 | 103% | ok | 1.3x (22%) |
| blas2 | trmv | f32 | n=4096 | 2.971 ms | 5.65 | 11.30 | 99% | ok | 1.3x (21%) |
| blas2 | trsv | f64 | n=4096 | 7.450 ms | 2.25 | 9.02 | 79% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 3.537 ms | 4.74 | 9.50 | 84% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 25.844 ms | 1.30 | 10.39 | 91% | ok | 1.1x (19%) |
| blas3 | gemm | f64 | 512x512x512 | 2.884 ms | 93.06 | 2.18 | 158% | ok | 4.9x (81%) |
| blas3 | gemm | f32 | 512x512x512 | 1.811 ms | 148.25 | 1.74 | 126% | ok | 5.0x (83%) |
| blas3 | gemmt | f64 | 512x512x512 | 2.580 ms | 52.03 | 2.44 | 88% | ok | 3.5x (58%) |
| blas3 | symm | f64 | 512x512x512 | 2.930 ms | 91.61 | 2.15 | 155% | ok | 4.8x (80%) |
| blas3 | syrk | f64 | 512x512x512 | 2.534 ms | 52.96 | 1.66 | 90% | ok | 3.1x (51%) |
| blas3 | syr2k | f64 | 512x512x512 | 3.718 ms | 72.19 | 1.69 | 122% | ok | 4.1x (68%) |
| blas3 | trmm | f64 | 512x512x512 | 1.623 ms | 82.71 | 3.23 | 140% | ok | 4.7x (78%) |
| blas3 | trsm | f64 | 512x512x512 | 1.767 ms | 75.96 | 2.97 | 129% | ok | 4.6x (77%) |
| blas3 | trsm | f32 | 512x512x512 | 1.255 ms | 106.96 | 2.09 | 91% | ok | 4.2x (70%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.690 ms | 0.14 | 1.10 | 10% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 5.615 ms | 15.93 | 0.75 | 27% | ok | 1.6x (27%) |
| factor | lu | f32 | 512x512 | 3.907 ms | 22.90 | 0.54 | 19% | ok | 1.6x (27%) |
| factor | cholesky | f64 | 512x512 | 4.152 ms | 10.78 | 1.01 | 18% | ok | 1.4x (23%) |
| factor | cholesky | f32 | 512x512 | 3.077 ms | 14.54 | 0.68 | 12% | ok | 1.3x (21%) |
| factor | ldlt | f64 | 512x512 | 25.070 ms | 1.78 | 0.17 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 13.367 ms | 3.35 | 0.31 | 6% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 13.017 ms | 13.75 | 0.32 | 23% | ok | 1.5x (25%) |
| factor | lq | f64 | 512x512 | 48.458 ms | 3.69 | 0.09 | 6% | ok | 1.5x (25%) |
| factor | hessenberg | f64 | 512x512 | 399.180 ms | 1.12 | 0.01 | 2% | ok | 1.6x (26%) |
| factor | householder | f64 | 512x512 | 0.918 ms | 1.14 | 4.57 | 40%\* | ok | 3.5x (59%) |
| factor | inv | f64 | 512x512 | 132.179 ms | 2.03 | 0.03 | 3% | ok | 1.0x (16%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 8.437 ms | 10.61 | 0.50 | 18% | ok | 1.5x (25%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 5.898 ms | 3.79 | 0.18 | 6% | ok | 1.1x (18%) |
| eigen | eigen_symmetric | f64 | 256x256 | 173.868 ms | 0.87 | 0.01 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 996.770 ms | 0.15 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 56.997 ms | 2.94 | 0.02 | 5% | ok | 0.9x (16%) |
| eigen | schur | f64 | 256x256 | 75.576 ms | 5.55 | 0.01 | 9% | ok | 1.0x (16%) |
| eigen | svd | f64 | 256x256 | 26.868 ms | 13.74 | 0.04 | 23% | ok | 1.1x (18%) |
| eigen | expm | f64 | 256x256 | 27.388 ms | 9.19 | 0.04 | 16% | ok | 1.3x (22%) |
| eigen | sylvester | f64 | 256x256 | 192.234 ms | 5.24 | 0.01 | 9% | ok | 1.0x (17%) |
| elementwise | exp | f64 | n=8388608 | 151.270 ms | 0.06 | 0.89 | 8% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 145.734 ms | 0.06 | 0.92 | 8% | ok | 1.0x (16%) |
| elementwise | sin | f64 | n=8388608 | 170.749 ms | 0.05 | 0.79 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 90.431 ms | 0.09 | 1.48 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 2469.917 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.268 ms | 1.15 | 12.01 | 106% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.120 ms | 0.84 | 8.73 | 77% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.304 ms | 3.70 | 7.79 | 69% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.125 ms | 0.20 | 0.71 | 6%\* | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.973 ms | 0.34 | 2.79 | 25% | ok | 1.0x (16%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 40.711 ms | 0.27 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 69.378 ms | 0.16 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 25.083 ms | 0.45 | 0.03 | 1% | ok | 1.0x (16%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 117.087 ms | 0.19 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 130.899 ms | 0.17 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 47.560 ms | 0.47 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 81.336 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.921 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.154 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 126.360 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.522 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.149 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.285 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.316 ms | 0.34 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.852 ms | 0.89 | 3.55 | 31%\* | ok | 0.9x (16%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.496 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.568 ms | 0.89 | 0.06 | 2% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.874 ms | 0.86 | 3.46 | 30%\* | ok | 0.9x (16%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.414 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 103.793 ms | 0.22 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.841 ms | 0.20 | 0.79 | 7%\* | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.829 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 113.763 ms | 0.20 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.797 ms | 0.20 | 0.80 | 7%\* | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 12.921 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 634.055 ms | 0.49 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 6.995 ms | 0.48 | 1.93 | 17%\* | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.670 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 203.009 ms | 1.52 | 0.00 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.025 ms | 0.48 | 1.92 | 17%\* | ok | 1.0x (16%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.424 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1696.103 ms | 0.36 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.920 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 17.920 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1497.284 ms | 0.41 | 0.00 | 1% | ok | 1.1x (18%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.473 ms | 0.10 | 0.42 | 4%\* | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 230.935 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 404.912 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 127.757 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 103.748 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 122.127 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 79.902 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 533.889 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 174.583 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.496 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 202.192 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 192.837 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 211.428 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 307.914 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.172 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.352 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 32.517 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 2.906 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | ildl | f64 | 128^2 grid | 2.737 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.092 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 3.533 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.482 ms | 0.75 | 8.42 | 74% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.179 ms | 0.82 | 9.22 | 81% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.801 ms | 0.69 | 7.71 | 68% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 36.545 ms | 1.05 | 7.95 | 70% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 140.890 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 147.607 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 130.809 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.542 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 187.515 ms | 0.09 | 0.36 | 3% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 20.487 ms | 13.10 | 0.20 | 22% | ok | 5.8x (97%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.308 ms | 7.27 | 7.73 | 68% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.510 ms | 2.58 | 10.31 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.137 ms | 5.35 | 10.70 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.985 ms | 3.37 | 3.37 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.977 ms | 3.37 | 3.37 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.565 ms | 23.21 | 0.27 | n/a | ok | 5.1x (85%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 8.229 ms | 32.62 | 0.25 | n/a | ok | 5.5x (92%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.314 ms | 62.22 | 0.36 | n/a | ok | 5.5x (91%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.312 ms | 62.25 | 0.36 | n/a | ok | 5.5x (92%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.003 ms | 20.64 | 0.12 | n/a | ok | 5.4x (90%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

