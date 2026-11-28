# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,8, CPUs 0,2,4,6,8,10,12,14, clock policy governor performance on the P-cores, turbo on; performance 800-4900 MHz on cpus 0,2,8,10,12,14; performance 800-5000 MHz on cpus 4,6, 2026-11-28T01:57:31Z.

## i7-12700k -- native, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.5 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.5 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.3 GFLOP/s |
| bandwidth | read 33.0, update 38.4, triad 32.4 GB/s (ceiling 38.4) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 44% | 98% | 0 |
| blas2 | 10 | 79% | 95% | 0 |
| blas3 | 10 | 17% | 18% | 0 |
| factor | 12 | 8% | 18% | 0 |
| eigen | 8 | 6% | 12% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 49% | 95% | 0 |
| direct | 19 | 1% | 32% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 24% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 62% | 77% | 0 |
| mg | 1 | 56% | 56% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 35% | 83% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.629 ms | 2.98 | 23.84 | 62% | ok |
| blas1 | dot | f32 | n=8388608 | 4.243 ms | 3.95 | 15.82 | 41% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.422 ms | 2.61 | 31.35 | 82% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.719 ms | 3.55 | 21.33 | 56% | ok |
| blas1 | scale | f64 | n=8388608 | 3.577 ms | 2.35 | 37.52 | 98% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.880 ms | 3.44 | 13.75 | 36% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.018 ms | 2.09 | 16.70 | 43% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.048 ms | 1.19 | 9.52 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 4.005 ms | 2.09 | 16.76 | 44% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 77.381 ms | 2.38 | 1.73 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.698 ms | 9.07 | 36.31 | 95% | ok |
| blas2 | gemv | f32 | n=4096 | 2.743 ms | 12.23 | 24.48 | 64% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 4.767 ms | 7.04 | 28.17 | 73% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.221 ms | 15.11 | 30.23 | 79% | ok |
| blas2 | symv | f64 | n=4096 | 3.206 ms | 10.47 | 20.96 | 55% | ok |
| blas2 | trmv | f64 | n=4096 | 2.012 ms | 8.34 | 33.38 | 87% | ok |
| blas2 | trmv | f32 | n=4096 | 1.426 ms | 11.77 | 23.56 | 61% | ok |
| blas2 | trsv | f64 | n=4096 | 1.980 ms | 8.47 | 33.93 | 88% | ok |
| blas2 | trsv | f32 | n=4096 | 1.393 ms | 12.04 | 24.11 | 63% | ok |
| blas2 | ger | f64 | n=4096 | 7.732 ms | 4.34 | 34.73 | 90% | ok |
| blas3 | gemm | f64 | 512x512x512 | 20.079 ms | 13.37 | 0.31 | 17% | ok |
| blas3 | gemm | f32 | 512x512x512 | 10.366 ms | 25.90 | 0.30 | 17% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 12.559 ms | 10.69 | 0.50 | 14% | ok |
| blas3 | symm | f64 | 512x512x512 | 19.782 ms | 13.57 | 0.32 | 17% | ok |
| blas3 | syrk | f64 | 512x512x512 | 13.629 ms | 9.85 | 0.31 | 13% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 24.378 ms | 11.01 | 0.26 | 14% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.562 ms | 14.04 | 0.55 | 18% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.633 ms | 13.93 | 0.54 | 18% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.193 ms | 25.85 | 0.50 | 17% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.245 ms | 0.40 | 3.21 | 8% | ok |
| factor | lu | f64 | 512x512 | 11.646 ms | 7.68 | 0.36 | 10% | ok |
| factor | lu | f32 | 512x512 | 4.299 ms | 20.81 | 0.49 | 13% | ok |
| factor | cholesky | f64 | 512x512 | 7.331 ms | 6.10 | 0.57 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.633 ms | 6.75 | 0.32 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.438 ms | 6.02 | 0.56 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 4.766 ms | 9.39 | 0.88 | 12% | ok |
| factor | qr | f64 | 512x512 | 226.778 ms | 0.79 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 25.845 ms | 6.92 | 0.16 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 267.932 ms | 1.67 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.623 ms | 0.65 | 2.58 | 7%\* | ok |
| factor | inv | f64 | 512x512 | 54.149 ms | 4.96 | 0.08 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.192 ms | 14.45 | 0.68 | 18% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 18.133 ms | 1.23 | 0.06 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 58.921 ms | 2.56 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 760.796 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 33.007 ms | 5.08 | 0.03 | 6% | ok |
| eigen | schur | f64 | 256x256 | 82.922 ms | 5.06 | 0.01 | 6% | ok |
| eigen | svd | f64 | 256x256 | 111.897 ms | 3.30 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 25.816 ms | 9.75 | 0.04 | 12% | ok |
| eigen | sylvester | f64 | 256x256 | 185.805 ms | 5.42 | 0.01 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 39.713 ms | 0.21 | 3.38 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 46.125 ms | 0.18 | 2.91 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 60.698 ms | 0.14 | 2.21 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 29.268 ms | 0.29 | 4.59 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 203.750 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.749 ms | 3.50 | 36.36 | 95% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.922 ms | 2.84 | 29.53 | 77% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.668 ms | 8.97 | 18.86 | 49% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 5.105 ms | 0.64 | 2.25 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.345 ms | 1.49 | 12.38 | 32%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 15.190 ms | 0.74 | 0.05 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 24.993 ms | 0.45 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.382 ms | 1.33 | 0.09 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.098 ms | 0.56 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.619 ms | 0.50 | 0.02 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 17.878 ms | 1.25 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 25.331 ms | 0.07 | 0.01 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.350 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.976 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 58.048 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.185 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.318 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.100 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 12.742 ms | 0.88 | 0.06 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.247 ms | 3.06 | 12.25 | 32%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.074 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.068 ms | 2.75 | 0.18 | 4% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.251 ms | 3.01 | 12.04 | 31%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.137 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.484 ms | 0.58 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.779 ms | 0.97 | 3.88 | 10%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.598 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.795 ms | 0.58 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.645 ms | 1.17 | 4.69 | 12%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.949 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 204.523 ms | 1.51 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.485 ms | 2.27 | 9.08 | 24%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.696 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 60.140 ms | 5.13 | 0.01 | 7% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.516 ms | 2.22 | 8.89 | 23%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.886 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 558.692 ms | 1.10 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.596 ms | 0.44 | 1.77 | 5%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.289 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 456.249 ms | 1.35 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.575 ms | 0.44 | 1.78 | 5%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 31.303 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 35.026 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 23.113 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.624 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 15.174 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 14.149 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.853 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.531 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.371 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 18.215 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.759 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 16.222 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 19.198 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.042 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.562 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 11.639 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.089 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.023 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.028 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.722 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.992 ms | 2.64 | 29.58 | 77% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.222 ms | 2.14 | 23.99 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.565 ms | 1.67 | 18.74 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.497 ms | 2.85 | 21.53 | 56% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 30.670 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 31.428 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.944 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 5.023 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.937 ms | 3.40 | 13.59 | 35% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 39.031 ms | 6.88 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.559 ms | 30.00 | 31.88 | 83%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.311 ms | 7.26 | 29.04 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.997 ms | 16.83 | 33.65 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.726 ms | 9.72 | 9.72 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.725 ms | 9.73 | 9.73 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.608 ms | 15.25 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.837 ms | 12.88 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.252 ms | 26.18 | 0.15 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.236 ms | 26.22 | 0.15 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 26.867 ms | 9.99 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## i7-12700k -- native, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 600.0 GFLOP/s, f32 1200.3 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.0, f32 1200.3 GFLOP/s |
| ... this build, scalar | f64 9.9, f32 10.0 GFLOP/s |
| bandwidth | read 83.1, update 78.0, triad 76.8 GB/s (ceiling 83.1) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 89% | 125% | 0 | 29% |
| blas2 | 10 | 42% | 134% | 0 | 13% |
| blas3 | 10 | 2% | 17% | 0 | 14% |
| factor | 12 | 2% | 12% | 0 | 14% |
| eigen | 8 | 1% | 4% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 290% | 0 | 13% |
| direct | 19 | 0% | 14% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 10% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 36% | 0 | 13% |
| mg | 1 | 26% | 26% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 39% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 94% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.401 ms | 11.97 | 95.78 | 115% | ok | 4.0x (50%) |
| blas1 | dot | f32 | n=8388608 | 0.648 ms | 25.89 | 103.57 | 125% | ok | 6.5x (82%) |
| blas1 | axpy | f64 | n=8388608 | 2.765 ms | 6.07 | 72.82 | 88% | ok | 2.3x (29%) |
| blas1 | axpy | f32 | n=8388608 | 1.364 ms | 12.30 | 73.78 | 89% | ok | 3.5x (43%) |
| blas1 | scale | f64 | n=8388608 | 1.776 ms | 4.72 | 75.57 | 91% | ok | 2.0x (25%) |
| blas1 | two_norm | f64 | n=8388608 | 0.748 ms | 22.42 | 89.67 | 108% | ok | 6.5x (82%) |
| blas1 | one_norm | f64 | n=8388608 | 4.011 ms | 2.09 | 16.73 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.953 ms | 1.21 | 9.65 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 4.008 ms | 2.09 | 16.74 | 20% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 76.351 ms | 2.42 | 1.76 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.506 ms | 22.29 | 89.19 | 107% | ok | 2.5x (31%) |
| blas2 | gemv | f32 | n=4096 | 0.602 ms | 55.72 | 111.49 | 134% | ok | 4.6x (57%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.571 ms | 21.36 | 85.48 | 103% | ok | 3.0x (38%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.856 ms | 39.19 | 78.41 | 94% | ok | 2.6x (32%) |
| blas2 | symv | f64 | n=4096 | 3.148 ms | 10.66 | 21.34 | 26% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 2.015 ms | 8.33 | 33.34 | 40% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=4096 | 1.417 ms | 11.84 | 23.70 | 29% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 1.969 ms | 8.52 | 34.12 | 41% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.375 ms | 12.20 | 24.42 | 29% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.677 ms | 4.37 | 34.98 | 42% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 2.659 ms | 100.95 | 2.37 | 17% | ok | 7.6x (94%) |
| blas3 | gemm | f32 | 512x512x512 | 1.359 ms | 197.55 | 2.32 | 16% | ok | 7.6x (95%) |
| blas3 | gemmt | f64 | 512x512x512 | 10.952 ms | 12.26 | 0.57 | 2% | ok | 1.1x (14%) |
| blas3 | symm | f64 | 512x512x512 | 19.083 ms | 14.07 | 0.33 | 2% | ok | 1.0x (13%) |
| blas3 | syrk | f64 | 512x512x512 | 12.000 ms | 11.18 | 0.35 | 2% | ok | 1.1x (14%) |
| blas3 | syr2k | f64 | 512x512x512 | 20.171 ms | 13.31 | 0.31 | 2% | ok | 1.2x (15%) |
| blas3 | trmm | f64 | 512x512x512 | 9.288 ms | 14.45 | 0.56 | 2% | ok | 1.0x (13%) |
| blas3 | trsm | f64 | 512x512x512 | 9.417 ms | 14.25 | 0.56 | 2% | ok | 1.0x (13%) |
| blas3 | trsm | f32 | 512x512x512 | 5.135 ms | 26.14 | 0.51 | 2% | ok | 1.0x (13%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.010 ms | 0.41 | 3.26 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 6.881 ms | 13.00 | 0.61 | 2% | ok | 1.7x (21%) |
| factor | lu | f32 | 512x512 | 3.729 ms | 24.00 | 0.56 | 2% | ok | 1.2x (14%) |
| factor | cholesky | f64 | 512x512 | 7.142 ms | 6.26 | 0.59 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 6.473 ms | 6.91 | 0.32 | 1% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 7.230 ms | 6.19 | 0.58 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 4.678 ms | 9.56 | 0.90 | 2% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 129.316 ms | 1.38 | 0.03 | 0% | ok | 1.8x (22%) |
| factor | lq | f64 | 512x512 | 16.675 ms | 10.73 | 0.25 | 2% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 157.873 ms | 2.83 | 0.03 | 0% | ok | 1.7x (21%) |
| factor | householder | f64 | 512x512 | 0.407 ms | 2.58 | 10.31 | 12%\* | ok | 4.0x (50%) |
| factor | inv | f64 | 512x512 | 48.301 ms | 5.56 | 0.09 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 5.566 ms | 16.08 | 0.75 | 3% | ok | 1.1x (14%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 18.045 ms | 1.24 | 0.06 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 57.941 ms | 2.61 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 740.111 ms | 0.20 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 31.585 ms | 5.31 | 0.03 | 1% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 79.611 ms | 5.27 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 108.406 ms | 3.40 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 10.770 ms | 23.37 | 0.10 | 4% | ok | 2.4x (30%) |
| eigen | sylvester | f64 | 256x256 | 176.456 ms | 5.70 | 0.01 | 1% | ok | 1.1x (13%) |
| elementwise | exp | f64 | n=8388608 | 38.750 ms | 0.22 | 3.46 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 45.137 ms | 0.19 | 2.97 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 60.566 ms | 0.14 | 2.22 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.500 ms | 0.29 | 4.71 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 203.041 ms | 0.04 | 0.66 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.113 ms | 23.17 | 241.07 | 290% ! | ok | 6.6x (83%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.904 ms | 2.89 | 30.11 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.310 ms | 31.96 | 67.21 | 81% | ok | 3.6x (45%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 5.026 ms | 0.65 | 2.29 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.528 ms | 1.31 | 10.89 | 13%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 14.372 ms | 0.78 | 0.05 | 0% | ok | 1.1x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 24.602 ms | 0.45 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.407 ms | 1.33 | 0.09 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.286 ms | 0.56 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 43.820 ms | 0.51 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.346 ms | 1.37 | 0.04 | 0% | ok | 1.1x (14%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 24.300 ms | 0.08 | 0.01 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.292 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.852 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 57.027 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.181 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.949 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.059 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 12.100 ms | 0.92 | 0.06 | 0% | ok | 1.1x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.254 ms | 2.97 | 11.89 | 14%\* | ok | 1.0x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 3.980 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 3.974 ms | 2.82 | 0.18 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.258 ms | 2.93 | 11.73 | 14%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.026 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 36.913 ms | 0.61 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.616 ms | 1.23 | 4.91 | 6%\* | ok | 1.3x (16%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.356 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.263 ms | 0.58 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.616 ms | 1.23 | 4.91 | 6%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.789 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 196.270 ms | 1.57 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.628 ms | 2.07 | 8.28 | 10%\* | ok | 0.9x (11%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.416 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 58.814 ms | 5.25 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.651 ms | 2.04 | 8.16 | 10%\* | ok | 0.9x (11%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.778 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 546.760 ms | 1.13 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.710 ms | 0.44 | 1.75 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.092 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 462.749 ms | 1.33 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.813 ms | 0.43 | 1.72 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.590 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 33.915 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.608 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.860 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.780 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.843 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.157 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.005 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.992 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 17.856 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.346 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 15.899 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 18.655 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.039 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.494 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 11.219 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.064 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.996 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.026 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.711 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.977 ms | 2.68 | 30.02 | 36% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.217 ms | 2.15 | 24.11 | 29% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.537 ms | 1.70 | 19.09 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.568 ms | 2.83 | 21.42 | 26% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.797 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 30.624 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 32.836 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.910 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 4.813 ms | 3.49 | 13.94 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 5.126 ms | 52.37 | 0.82 | 9% | ok | 7.6x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.554 ms | 30.30 | 32.19 | 39%\* | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.283 ms | 7.35 | 29.40 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.973 ms | 17.25 | 34.49 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.688 ms | 9.94 | 9.94 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.687 ms | 9.95 | 9.95 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.287 ms | 117.38 | 1.38 | n/a | ok | 7.7x (96%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.746 ms | 97.77 | 0.76 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.355 ms | 198.05 | 1.16 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.355 ms | 198.18 | 1.16 | n/a | ok | 7.6x (94%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.541 ms | 75.81 | 0.44 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

## i7-12700k -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.5 GFLOP/s |
| ... this build, AVX2 | f64 78.2, f32 156.6 GFLOP/s |
| bandwidth | read 37.6, update 41.5, triad 31.4 GB/s (ceiling 41.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 87% | 103% | 0 |
| blas2 | 10 | 93% | 96% | 0 |
| blas3 | 10 | 67% | 91% | 0 |
| factor | 12 | 8% | 25% | 0 |
| eigen | 8 | 7% | 28% | 0 |
| elementwise | 5 | 7% | 11% | 0 |
| sparse | 5 | 52% | 89% | 0 |
| direct | 19 | 1% | 27% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 4% | 22% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 58% | 72% | 0 |
| mg | 1 | 52% | 52% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 84% | 131% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 3.728 ms | 4.50 | 36.00 | 87% | ok |
| blas1 | dot | f32 | n=8388608 | 1.710 ms | 9.81 | 39.24 | 95% | ok |
| blas1 | axpy | f64 | n=8388608 | 4.780 ms | 3.51 | 42.12 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 2.348 ms | 7.15 | 42.87 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 3.459 ms | 2.43 | 38.80 | 94% | ok |
| blas1 | two_norm | f64 | n=8388608 | 2.224 ms | 7.54 | 30.17 | 73% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.040 ms | 2.08 | 16.61 | 40% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.043 ms | 1.19 | 9.53 | 23% | ok |
| blas1 | sum | f64 | n=8388608 | 3.971 ms | 2.11 | 16.90 | 41% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 70.909 ms | 2.60 | 1.89 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.420 ms | 9.81 | 39.27 | 95% | ok |
| blas2 | gemv | f32 | n=4096 | 1.684 ms | 19.92 | 39.87 | 96% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.470 ms | 9.67 | 38.70 | 93% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.683 ms | 19.93 | 39.89 | 96% | ok |
| blas2 | symv | f64 | n=4096 | 2.924 ms | 11.48 | 22.97 | 55% | ok |
| blas2 | trmv | f64 | n=4096 | 1.787 ms | 9.39 | 37.59 | 91% | ok |
| blas2 | trmv | f32 | n=4096 | 0.863 ms | 19.45 | 38.94 | 94% | ok |
| blas2 | trsv | f64 | n=4096 | 1.818 ms | 9.23 | 36.96 | 89% | ok |
| blas2 | trsv | f32 | n=4096 | 0.870 ms | 19.28 | 38.60 | 93% | ok |
| blas2 | ger | f64 | n=4096 | 7.142 ms | 4.70 | 37.60 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.800 ms | 70.64 | 1.66 | 90% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.893 ms | 141.79 | 1.66 | 91% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.629 ms | 51.06 | 2.39 | 65% | ok |
| blas3 | symm | f64 | 512x512x512 | 4.057 ms | 66.17 | 1.55 | 85% | ok |
| blas3 | syrk | f64 | 512x512x512 | 2.570 ms | 52.23 | 1.63 | 67% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 5.000 ms | 53.68 | 1.26 | 69% | ok |
| blas3 | trmm | f64 | 512x512x512 | 3.045 ms | 44.08 | 1.72 | 56% | ok |
| blas3 | trsm | f64 | 512x512x512 | 3.094 ms | 43.39 | 1.69 | 55% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.446 ms | 92.81 | 1.81 | 59% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.152 ms | 0.40 | 3.23 | 8% | ok |
| factor | lu | f64 | 512x512 | 11.114 ms | 8.05 | 0.38 | 10% | ok |
| factor | lu | f32 | 512x512 | 3.107 ms | 28.80 | 0.68 | 18% | ok |
| factor | cholesky | f64 | 512x512 | 7.118 ms | 6.29 | 0.59 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.541 ms | 6.84 | 0.32 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.249 ms | 6.17 | 0.58 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 4.537 ms | 9.86 | 0.92 | 13% | ok |
| factor | qr | f64 | 512x512 | 228.328 ms | 0.78 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 23.397 ms | 7.65 | 0.18 | 10% | ok |
| factor | hessenberg | f64 | 512x512 | 262.934 ms | 1.70 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.614 ms | 0.65 | 2.60 | 6%\* | ok |
| factor | inv | f64 | 512x512 | 54.960 ms | 4.88 | 0.08 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 4.592 ms | 19.48 | 0.91 | 25% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 16.593 ms | 1.35 | 0.06 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 56.729 ms | 2.66 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 760.008 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 30.671 ms | 5.47 | 0.03 | 7% | ok |
| eigen | schur | f64 | 256x256 | 80.032 ms | 5.24 | 0.01 | 7% | ok |
| eigen | svd | f64 | 256x256 | 93.209 ms | 3.96 | 0.01 | 5% | ok |
| eigen | expm | f64 | 256x256 | 11.643 ms | 21.61 | 0.09 | 28% | ok |
| eigen | sylvester | f64 | 256x256 | 176.091 ms | 5.72 | 0.01 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 39.491 ms | 0.21 | 3.40 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 45.903 ms | 0.18 | 2.92 | 7% | ok |
| elementwise | sin | f64 | n=8388608 | 59.926 ms | 0.14 | 2.24 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 29.042 ms | 0.29 | 4.62 | 11% | ok |
| elementwise | erf | f64 | n=8388608 | 202.365 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.736 ms | 3.55 | 36.98 | 89% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.919 ms | 2.85 | 29.64 | 71% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.050 ms | 10.34 | 21.74 | 52% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.803 ms | 0.68 | 2.39 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.323 ms | 1.52 | 12.58 | 30%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 15.104 ms | 0.74 | 0.05 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 27.026 ms | 0.41 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.894 ms | 1.26 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.845 ms | 0.56 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.143 ms | 0.51 | 0.02 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.809 ms | 1.33 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 25.297 ms | 0.07 | 0.01 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.452 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.013 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 58.522 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.194 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.514 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.135 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 12.746 ms | 0.88 | 0.06 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.273 ms | 2.77 | 11.09 | 27%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.140 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.494 ms | 2.49 | 0.16 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.271 ms | 2.79 | 11.15 | 27%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.113 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.732 ms | 0.63 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.649 ms | 1.17 | 4.66 | 11%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.462 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.084 ms | 0.59 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.647 ms | 1.17 | 4.68 | 11%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.946 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 205.839 ms | 1.50 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.471 ms | 2.29 | 9.16 | 22%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.665 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 67.222 ms | 4.59 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.681 ms | 2.00 | 8.02 | 19%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.935 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 527.749 ms | 1.17 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.543 ms | 0.45 | 1.79 | 4%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.323 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 693.928 ms | 0.89 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.548 ms | 0.45 | 1.79 | 4%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.199 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 29.999 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 20.985 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.450 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.742 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.483 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 19.816 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.746 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.431 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.471 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 14.025 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 13.178 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 14.499 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.036 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.588 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 11.930 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.092 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.030 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.028 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.714 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.986 ms | 2.65 | 29.74 | 72% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.221 ms | 2.14 | 24.02 | 58% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.418 ms | 1.85 | 20.69 | 50% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.440 ms | 2.86 | 21.63 | 52% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 28.609 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 28.936 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.112 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.589 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 1.934 ms | 8.68 | 34.70 | 84% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 5.998 ms | 44.75 | 0.70 | 57% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.327 ms | 51.27 | 54.48 | 131%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.903 ms | 8.82 | 35.26 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.767 ms | 21.86 | 43.72 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.289 ms | 58.05 | 58.05 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.277 ms | 60.55 | 60.55 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 3.978 ms | 67.48 | 0.79 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.584 ms | 48.08 | 0.38 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.570 ms | 48.19 | 0.28 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 2.098 ms | 127.93 | 0.75 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 1.870 ms | 143.55 | 0.84 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## i7-12700k -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 600.8 GFLOP/s, f32 1201.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.8, f32 1201.6 GFLOP/s |
| ... this build, AVX2 | f64 600.6, f32 1201.1 GFLOP/s |
| bandwidth | read 82.0, update 79.0, triad 77.0 GB/s (ceiling 82.0) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 93% | 149% | 0 | 23% |
| blas2 | 10 | 49% | 147% | 0 | 13% |
| blas3 | 10 | 18% | 66% | 0 | 27% |
| factor | 12 | 2% | 12% | 0 | 14% |
| eigen | 8 | 1% | 4% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 300% | 0 | 13% |
| direct | 19 | 0% | 14% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 10% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 30% | 37% | 0 | 13% |
| mg | 1 | 26% | 26% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 49% | 71% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 76% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.403 ms | 11.96 | 95.66 | 117% | ok | 2.7x (33%) |
| blas1 | dot | f32 | n=8388608 | 0.549 ms | 30.58 | 122.30 | 149% | ok | 3.1x (39%) |
| blas1 | axpy | f64 | n=8388608 | 2.662 ms | 6.30 | 75.63 | 92% | ok | 1.8x (22%) |
| blas1 | axpy | f32 | n=8388608 | 1.252 ms | 13.40 | 80.38 | 98% | ok | 1.9x (23%) |
| blas1 | scale | f64 | n=8388608 | 1.760 ms | 4.77 | 76.25 | 93% | ok | 2.0x (25%) |
| blas1 | two_norm | f64 | n=8388608 | 0.568 ms | 29.56 | 118.22 | 144% | ok | 3.9x (49%) |
| blas1 | one_norm | f64 | n=8388608 | 3.956 ms | 2.12 | 16.96 | 21% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.917 ms | 1.21 | 9.70 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.909 ms | 2.15 | 17.17 | 21% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 70.023 ms | 2.64 | 1.92 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.406 ms | 23.87 | 95.53 | 117% | ok | 2.4x (30%) |
| blas2 | gemv | f32 | n=4096 | 0.556 ms | 60.38 | 120.82 | 147% | ok | 3.0x (38%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.425 ms | 23.54 | 94.22 | 115% | ok | 2.4x (30%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.676 ms | 49.67 | 99.39 | 121% | ok | 2.5x (31%) |
| blas2 | symv | f64 | n=4096 | 2.862 ms | 11.72 | 23.47 | 29% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 1.757 ms | 9.55 | 38.22 | 47% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 0.840 ms | 19.96 | 39.96 | 49% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 1.795 ms | 9.35 | 37.43 | 46% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.848 ms | 19.79 | 39.61 | 48% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.121 ms | 4.71 | 37.71 | 46% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 0.674 ms | 398.10 | 9.33 | 66% | ok | 5.6x (70%) |
| blas3 | gemm | f32 | 512x512x512 | 0.354 ms | 758.85 | 8.89 | 63% | ok | 5.4x (67%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.220 ms | 110.01 | 5.16 | 18% | ok | 2.2x (27%) |
| blas3 | symm | f64 | 512x512x512 | 0.914 ms | 293.55 | 6.88 | 49% | ok | 4.4x (55%) |
| blas3 | syrk | f64 | 512x512x512 | 1.269 ms | 105.74 | 3.30 | 18% | ok | 2.0x (25%) |
| blas3 | syr2k | f64 | 512x512x512 | 2.304 ms | 116.53 | 2.73 | 19% | ok | 2.2x (27%) |
| blas3 | trmm | f64 | 512x512x512 | 1.670 ms | 80.36 | 3.14 | 13% | ok | 1.8x (23%) |
| blas3 | trsm | f64 | 512x512x512 | 1.728 ms | 77.68 | 3.03 | 13% | ok | 1.8x (22%) |
| blas3 | trsm | f32 | 512x512x512 | 0.947 ms | 141.79 | 2.77 | 12% | ok | 1.5x (19%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.925 ms | 0.41 | 3.29 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 6.185 ms | 14.47 | 0.68 | 2% | ok | 1.8x (22%) |
| factor | lu | f32 | 512x512 | 2.808 ms | 31.87 | 0.75 | 3% | ok | 1.1x (14%) |
| factor | cholesky | f64 | 512x512 | 6.899 ms | 6.49 | 0.61 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 6.392 ms | 7.00 | 0.33 | 1% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 7.057 ms | 6.34 | 0.59 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 4.481 ms | 9.99 | 0.94 | 2% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 127.944 ms | 1.40 | 0.03 | 0% | ok | 1.8x (22%) |
| factor | lq | f64 | 512x512 | 14.376 ms | 12.45 | 0.29 | 2% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 149.733 ms | 2.99 | 0.03 | 0% | ok | 1.8x (22%) |
| factor | householder | f64 | 512x512 | 0.433 ms | 2.43 | 9.69 | 12%\* | ok | 3.7x (47%) |
| factor | inv | f64 | 512x512 | 49.272 ms | 5.45 | 0.09 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 4.293 ms | 20.84 | 0.98 | 3% | ok | 1.1x (13%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 16.330 ms | 1.37 | 0.06 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 56.193 ms | 2.69 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 742.754 ms | 0.20 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 30.267 ms | 5.54 | 0.03 | 1% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 79.057 ms | 5.31 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 88.584 ms | 4.17 | 0.01 | 1% | ok | 1.1x (13%) |
| eigen | expm | f64 | 256x256 | 9.398 ms | 26.78 | 0.11 | 4% | ok | 1.2x (15%) |
| eigen | sylvester | f64 | 256x256 | 173.001 ms | 5.82 | 0.01 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 38.744 ms | 0.22 | 3.46 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 44.672 ms | 0.19 | 3.00 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 58.933 ms | 0.14 | 2.28 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.486 ms | 0.29 | 4.71 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 201.451 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.111 ms | 23.62 | 245.78 | 300% ! | ok | 6.6x (83%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.911 ms | 2.87 | 29.88 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.304 ms | 32.12 | 67.54 | 82% | ok | 3.1x (39%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.839 ms | 0.67 | 2.38 | 3%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.521 ms | 1.32 | 10.95 | 13%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 14.977 ms | 0.75 | 0.05 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 26.614 ms | 0.42 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.987 ms | 1.24 | 0.08 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.112 ms | 0.56 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.507 ms | 0.50 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 20.579 ms | 1.09 | 0.04 | 0% | ok | 0.8x (10%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 24.951 ms | 0.08 | 0.01 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.388 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.886 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 57.312 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.188 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.156 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.093 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 12.448 ms | 0.90 | 0.06 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.258 ms | 2.94 | 11.74 | 14%\* | ok | 1.1x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.044 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.476 ms | 2.50 | 0.16 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.279 ms | 2.71 | 10.84 | 13%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.075 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.946 ms | 0.62 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.622 ms | 1.22 | 4.86 | 6%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.399 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.737 ms | 0.58 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.624 ms | 1.21 | 4.85 | 6%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.848 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 197.584 ms | 1.56 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.638 ms | 2.06 | 8.23 | 10%\* | ok | 0.9x (11%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.392 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 67.601 ms | 4.56 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.854 ms | 1.82 | 7.27 | 9%\* | ok | 0.9x (11%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.863 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 516.051 ms | 1.20 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.740 ms | 0.44 | 1.74 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.176 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 690.476 ms | 0.89 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.725 ms | 0.44 | 1.74 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.089 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 29.902 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 21.162 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.453 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.697 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.484 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 19.821 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.560 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.386 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.088 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 13.678 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 12.992 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 14.353 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.034 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.559 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 11.749 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.065 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 1.011 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.027 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.707 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.964 ms | 2.72 | 30.43 | 37% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.198 ms | 2.19 | 24.49 | 30% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.396 ms | 1.88 | 21.01 | 26% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.515 ms | 2.85 | 21.51 | 26% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 27.705 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 28.215 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 32.780 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.518 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 1.852 ms | 9.06 | 36.24 | 44% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 0.905 ms | 296.65 | 4.64 | 49% | ok | 6.6x (83%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.307 ms | 54.63 | 58.05 | 71%\* | ok | 1.1x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.889 ms | 8.88 | 35.53 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.747 ms | 22.45 | 44.90 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.269 ms | 62.26 | 62.26 | n/a | ok | 1.1x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.253 ms | 66.27 | 66.27 | n/a | ok | 1.1x (14%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 0.622 ms | 431.44 | 5.06 | n/a | ok | 6.4x (80%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 0.824 ms | 325.71 | 2.54 | n/a | ok | 6.8x (85%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 0.810 ms | 331.28 | 1.94 | n/a | ok | 6.9x (86%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.330 ms | 814.51 | 4.77 | n/a | ok | 6.4x (80%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.306 ms | 878.15 | 5.15 | n/a | ok | 6.1x (76%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

## i7-12700k -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.3 GFLOP/s |
| bandwidth | read 33.1, update 38.5, triad 32.5 GB/s (ceiling 38.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 44% | 97% | 0 |
| blas2 | 10 | 100% | 106% | 0 |
| blas3 | 10 | 81% | 94% | 0 |
| factor | 12 | 25% | 40% | 0 |
| eigen | 8 | 24% | 70% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 49% | 94% | 0 |
| direct | 31 | 3% | 32% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 21 | 24% | 69% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 62% | 77% | 0 |
| mg | 1 | 56% | 56% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 36% | 84% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.625 ms | 2.98 | 23.86 | 62% | ok |
| blas1 | dot | f32 | n=8388608 | 4.229 ms | 3.97 | 15.87 | 41% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.438 ms | 2.61 | 31.27 | 81% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.727 ms | 3.55 | 21.30 | 55% | ok |
| blas1 | scale | f64 | n=8388608 | 3.578 ms | 2.34 | 37.51 | 97% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.881 ms | 3.44 | 13.75 | 36% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.013 ms | 2.09 | 16.72 | 43% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.061 ms | 1.19 | 9.50 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 4.001 ms | 2.10 | 16.77 | 44% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 76.779 ms | 2.40 | 1.75 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.506 ms | 9.57 | 38.30 | 100% | ok |
| blas2 | gemv | f32 | n=4096 | 1.649 ms | 20.35 | 40.72 | 106% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.631 ms | 9.24 | 36.99 | 96% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.639 ms | 20.48 | 40.97 | 106% | ok |
| blas2 | symv | f64 | n=4096 | 1.924 ms | 17.44 | 34.91 | 91% | ok |
| blas2 | trmv | f64 | n=4096 | 1.858 ms | 9.03 | 36.15 | 94% | ok |
| blas2 | trmv | f32 | n=4096 | 0.945 ms | 17.75 | 35.53 | 92% | ok |
| blas2 | trsv | f64 | n=4096 | 1.797 ms | 9.33 | 37.37 | 97% | ok |
| blas2 | trsv | f32 | n=4096 | 0.871 ms | 19.26 | 38.56 | 100% | ok |
| blas2 | ger | f64 | n=4096 | 6.828 ms | 4.91 | 39.32 | 102% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.708 ms | 72.40 | 1.70 | 92% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.833 ms | 146.46 | 1.72 | 94% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.573 ms | 52.16 | 2.44 | 67% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.715 ms | 72.25 | 1.69 | 92% | ok |
| blas3 | syrk | f64 | 512x512x512 | 2.544 ms | 52.75 | 1.65 | 67% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 4.706 ms | 57.04 | 1.34 | 73% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.109 ms | 63.64 | 2.49 | 81% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.135 ms | 62.86 | 2.46 | 80% | ok |
| blas3 | trsm | f32 | 512x512x512 | 0.961 ms | 139.62 | 2.73 | 89% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.223 ms | 0.40 | 3.21 | 8% | ok |
| factor | lu | f64 | 512x512 | 3.293 ms | 27.17 | 1.27 | 35% | ok |
| factor | lu | f32 | 512x512 | 1.880 ms | 47.60 | 1.12 | 30% | ok |
| factor | cholesky | f64 | 512x512 | 2.059 ms | 21.73 | 2.04 | 28% | ok |
| factor | cholesky | f32 | 512x512 | 1.153 ms | 38.80 | 1.82 | 25% | ok |
| factor | ldlt | f64 | 512x512 | 7.507 ms | 5.96 | 0.56 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 5.002 ms | 8.94 | 0.84 | 11% | ok |
| factor | qr | f64 | 512x512 | 5.744 ms | 31.15 | 0.73 | 40% | ok |
| factor | lq | f64 | 512x512 | 25.994 ms | 6.88 | 0.16 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 266.121 ms | 1.68 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.638 ms | 0.64 | 2.56 | 7%\* | ok |
| factor | inv | f64 | 512x512 | 46.074 ms | 5.83 | 0.09 | 7% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 3.820 ms | 23.42 | 1.10 | 30% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.734 ms | 12.90 | 0.60 | 16% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 58.920 ms | 2.56 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 756.369 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 15.991 ms | 10.49 | 0.07 | 13% | ok |
| eigen | schur | f64 | 256x256 | 21.077 ms | 19.90 | 0.05 | 25% | ok |
| eigen | svd | f64 | 256x256 | 6.709 ms | 55.02 | 0.16 | 70% | ok |
| eigen | expm | f64 | 256x256 | 11.181 ms | 22.51 | 0.09 | 29% | ok |
| eigen | sylvester | f64 | 256x256 | 53.131 ms | 18.95 | 0.03 | 24% | ok |
| elementwise | exp | f64 | n=8388608 | 39.475 ms | 0.21 | 3.40 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 46.358 ms | 0.18 | 2.90 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 60.298 ms | 0.14 | 2.23 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 29.128 ms | 0.29 | 4.61 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 202.211 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.750 ms | 3.49 | 36.29 | 94% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.925 ms | 2.83 | 29.44 | 77% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.676 ms | 8.96 | 18.83 | 49% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.853 ms | 0.67 | 2.37 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.323 ms | 1.52 | 12.59 | 33%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 14.441 ms | 0.77 | 0.05 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 24.975 ms | 0.45 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.432 ms | 1.33 | 0.09 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.599 ms | 0.55 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.500 ms | 0.50 | 0.02 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 17.022 ms | 1.31 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 24.772 ms | 0.08 | 0.01 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.374 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.028 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 58.114 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.186 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.186 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.105 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 12.104 ms | 0.92 | 0.06 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.247 ms | 3.06 | 12.23 | 32%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.071 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.085 ms | 2.74 | 0.18 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.249 ms | 3.03 | 12.14 | 32%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.086 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.712 ms | 0.63 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.631 ms | 1.20 | 4.79 | 12%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.437 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.371 ms | 0.58 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.641 ms | 1.18 | 4.72 | 12%\* | ok |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 5.738 ms | 1.95 | 0.13 | 2% | ok |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 1.891 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 3.469 ms | 3.22 | 0.21 | 4% | ok |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 0.329 ms | 2.30 | 9.19 | 24%\* | ok |
| direct | ref_pardiso_spd | f64 | 96^2 grid, nnz 45696 | 4.561 ms | 2.45 | 0.16 | 3% | ok |
| direct | ref_pardiso_spd.analyze | f64 | 96^2 grid, nnz 45696 | 2.121 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_pardiso_spd.factor | f64 | 96^2 grid, nnz 45696 | 1.912 ms | 5.85 | 0.38 | 7% | ok |
| direct | ref_pardiso_spd.solve | f64 | 96^2 grid, nnz 45696 | 0.461 ms | 1.64 | 6.56 | 17%\* | ok |
| direct | ref_pardiso_lu | f64 | 96^2 grid, nnz 45696 | 7.585 ms | 2.95 | 0.10 | 4% | ok |
| direct | ref_pardiso_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.596 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_pardiso_lu.factor | f64 | 96^2 grid, nnz 45696 | 3.248 ms | 6.89 | 0.23 | 9% | ok |
| direct | ref_pardiso_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.577 ms | 1.31 | 5.24 | 14%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.929 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 206.424 ms | 1.49 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.483 ms | 2.27 | 9.09 | 24%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.671 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 59.930 ms | 5.15 | 0.01 | 7% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.478 ms | 2.28 | 9.12 | 24%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.870 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 557.055 ms | 1.11 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.547 ms | 0.45 | 1.79 | 5%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.253 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 453.313 ms | 1.36 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.525 ms | 0.45 | 1.79 | 5%\* | ok |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 3.898 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 11.452 ms | 26.95 | 0.07 | 34% | ok |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 0.508 ms | 6.63 | 26.52 | 69%\* | ok |
| direct3d | ref_pardiso_spd.analyze | f64 | 20^3 grid, nnz 53600 | 7.993 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_pardiso_spd.factor | f64 | 20^3 grid, nnz 53600 | 9.684 ms | 31.87 | 0.09 | 41% | ok |
| direct3d | ref_pardiso_spd.solve | f64 | 20^3 grid, nnz 53600 | 0.874 ms | 3.85 | 15.42 | 40%\* | ok |
| direct3d | ref_pardiso_lu.analyze | f64 | 20^3 grid, nnz 53600 | 9.048 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_pardiso_lu.factor | f64 | 20^3 grid, nnz 53600 | 19.144 ms | 32.24 | 0.04 | 41% | ok |
| direct3d | ref_pardiso_lu.solve | f64 | 20^3 grid, nnz 53600 | 1.036 ms | 3.25 | 13.00 | 34%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 31.411 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 35.168 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 23.181 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.696 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 15.192 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 14.183 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.683 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.296 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.397 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 18.341 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.806 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 16.276 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 19.224 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.041 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.559 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 11.652 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.090 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.034 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.027 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.157 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.987 ms | 2.65 | 29.72 | 77% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.221 ms | 2.14 | 24.03 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.563 ms | 1.67 | 18.77 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.391 ms | 2.87 | 21.71 | 56% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 30.619 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 31.920 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.494 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.999 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.866 ms | 3.45 | 13.79 | 36% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 39.000 ms | 6.88 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.554 ms | 30.27 | 32.16 | 84%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.309 ms | 7.27 | 29.06 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.996 ms | 16.84 | 33.69 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.724 ms | 9.73 | 9.73 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.725 ms | 9.73 | 9.73 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.446 ms | 15.39 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.837 ms | 12.88 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.234 ms | 26.23 | 0.15 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.232 ms | 26.23 | 0.15 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 26.866 ms | 9.99 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## i7-12700k -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 600.3 GFLOP/s, f32 1201.1 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.3, f32 1201.1 GFLOP/s |
| ... this build, scalar | f64 10.0, f32 9.9 GFLOP/s |
| bandwidth | read 82.1, update 77.2, triad 75.7 GB/s (ceiling 82.1) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 91% | 125% | 0 | 29% |
| blas2 | 10 | 119% | 199% | 0 | 32% |
| blas3 | 10 | 66% | 84% | 0 | 77% |
| factor | 12 | 5% | 12% | 0 | 17% |
| eigen | 8 | 3% | 10% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 37% | 296% | 0 | 13% |
| direct | 31 | 1% | 15% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 21 | 10% | 32% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 30% | 37% | 0 | 13% |
| mg | 1 | 27% | 27% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 40% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 92% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.393 ms | 12.04 | 96.34 | 117% | ok | 4.0x (50%) |
| blas1 | dot | f32 | n=8388608 | 0.652 ms | 25.71 | 102.86 | 125% | ok | 6.5x (81%) |
| blas1 | axpy | f64 | n=8388608 | 2.734 ms | 6.14 | 73.64 | 90% | ok | 2.4x (29%) |
| blas1 | axpy | f32 | n=8388608 | 1.348 ms | 12.45 | 74.70 | 91% | ok | 3.5x (44%) |
| blas1 | scale | f64 | n=8388608 | 1.775 ms | 4.73 | 75.62 | 92% | ok | 2.0x (25%) |
| blas1 | two_norm | f64 | n=8388608 | 0.749 ms | 22.41 | 89.63 | 109% | ok | 6.5x (81%) |
| blas1 | one_norm | f64 | n=8388608 | 3.954 ms | 2.12 | 16.97 | 21% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.970 ms | 1.20 | 9.63 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.951 ms | 2.12 | 16.99 | 21% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 75.961 ms | 2.43 | 1.77 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.419 ms | 23.65 | 94.65 | 115% | ok | 2.5x (31%) |
| blas2 | gemv | f32 | n=4096 | 0.550 ms | 60.95 | 121.97 | 149% | ok | 3.0x (37%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.426 ms | 23.53 | 94.15 | 115% | ok | 2.5x (32%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.685 ms | 48.98 | 98.00 | 119% | ok | 2.4x (30%) |
| blas2 | symv | f64 | n=4096 | 0.621 ms | 54.06 | 108.22 | 132% | ok | 3.1x (39%) |
| blas2 | trmv | f64 | n=4096 | 0.597 ms | 28.08 | 112.44 | 137% | ok | 3.1x (39%) |
| blas2 | trmv | f32 | n=4096 | 0.206 ms | 81.64 | 163.43 | 199% ! | ok | 4.6x (57%) |
| blas2 | trsv | f64 | n=4096 | 1.766 ms | 9.50 | 38.04 | 46% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.856 ms | 19.59 | 39.22 | 48% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 3.638 ms | 9.22 | 73.81 | 90% | ok | 1.9x (23%) |
| blas3 | gemm | f64 | 512x512x512 | 0.537 ms | 499.89 | 11.72 | 83% | ok | 6.9x (86%) |
| blas3 | gemm | f32 | 512x512x512 | 0.267 ms | 1007.14 | 11.80 | 84% | ok | 6.9x (86%) |
| blas3 | gemmt | f64 | 512x512x512 | 0.694 ms | 193.37 | 9.06 | 32% | ok | 3.7x (46%) |
| blas3 | symm | f64 | 512x512x512 | 0.539 ms | 497.91 | 11.67 | 83% | ok | 6.9x (86%) |
| blas3 | syrk | f64 | 512x512x512 | 0.753 ms | 178.21 | 5.57 | 30% | ok | 3.4x (42%) |
| blas3 | syr2k | f64 | 512x512x512 | 1.124 ms | 238.72 | 5.59 | 40% | ok | 4.2x (52%) |
| blas3 | trmm | f64 | 512x512x512 | 0.341 ms | 394.03 | 15.39 | 66% | ok | 6.2x (77%) |
| blas3 | trsm | f64 | 512x512x512 | 0.337 ms | 398.35 | 15.56 | 66% | ok | 6.3x (79%) |
| blas3 | trsm | f32 | 512x512x512 | 0.161 ms | 835.66 | 16.32 | 70% | ok | 6.0x (75%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.979 ms | 0.41 | 3.27 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 2.085 ms | 42.92 | 2.01 | 7% | ok | 1.6x (20%) |
| factor | lu | f32 | 512x512 | 1.363 ms | 65.66 | 1.54 | 5% | ok | 1.4x (17%) |
| factor | cholesky | f64 | 512x512 | 1.569 ms | 28.51 | 2.67 | 5% | ok | 1.3x (16%) |
| factor | cholesky | f32 | 512x512 | 1.021 ms | 43.80 | 2.05 | 4% | ok | 1.1x (14%) |
| factor | ldlt | f64 | 512x512 | 7.204 ms | 6.21 | 0.58 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 4.557 ms | 9.82 | 0.92 | 2% | ok | 1.1x (14%) |
| factor | qr | f64 | 512x512 | 4.160 ms | 43.02 | 1.01 | 7% | ok | 1.4x (17%) |
| factor | lq | f64 | 512x512 | 16.487 ms | 10.85 | 0.25 | 2% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 154.548 ms | 2.89 | 0.03 | 0% | ok | 1.7x (22%) |
| factor | householder | f64 | 512x512 | 0.421 ms | 2.50 | 9.97 | 12%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 45.975 ms | 5.84 | 0.09 | 1% | ok | 1.0x (13%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 2.828 ms | 31.64 | 1.48 | 5% | ok | 1.4x (17%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.680 ms | 13.32 | 0.62 | 2% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 57.447 ms | 2.63 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 743.200 ms | 0.20 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 17.340 ms | 9.68 | 0.06 | 2% | ok | 0.9x (12%) |
| eigen | schur | f64 | 256x256 | 21.725 ms | 19.31 | 0.05 | 3% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 6.203 ms | 59.51 | 0.17 | 10% | ok | 1.1x (14%) |
| eigen | expm | f64 | 256x256 | 8.679 ms | 29.00 | 0.12 | 5% | ok | 1.3x (16%) |
| eigen | sylvester | f64 | 256x256 | 55.011 ms | 18.30 | 0.03 | 3% | ok | 1.0x (12%) |
| elementwise | exp | f64 | n=8388608 | 38.734 ms | 0.22 | 3.47 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 45.520 ms | 0.18 | 2.95 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 59.245 ms | 0.14 | 2.27 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.580 ms | 0.29 | 4.70 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 200.847 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.112 ms | 23.33 | 242.73 | 296% ! | ok | 6.7x (84%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.896 ms | 2.92 | 30.40 | 37% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.310 ms | 31.96 | 67.20 | 82% | ok | 3.6x (45%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.717 ms | 0.69 | 2.44 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.536 ms | 1.31 | 10.84 | 13%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 14.340 ms | 0.78 | 0.05 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 24.641 ms | 0.45 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.554 ms | 1.31 | 0.09 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.876 ms | 0.55 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 45.308 ms | 0.49 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 21.204 ms | 1.06 | 0.03 | 0% | ok | 0.8x (10%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 24.724 ms | 0.08 | 0.01 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.274 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.914 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 57.101 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.182 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.919 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.058 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 11.877 ms | 0.94 | 0.06 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.253 ms | 2.98 | 11.94 | 15%\* | ok | 1.0x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 3.978 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.202 ms | 2.66 | 0.17 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.256 ms | 2.96 | 11.83 | 14%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.039 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 37.457 ms | 0.60 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.614 ms | 1.23 | 4.93 | 6%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.458 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 39.424 ms | 0.57 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.615 ms | 1.23 | 4.92 | 6%\* | ok | 1.0x (13%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 5.973 ms | 1.87 | 0.12 | 0% | ok | 1.0x (12%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 1.815 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 3.656 ms | 3.06 | 0.20 | 1% | ok | 0.9x (12%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 0.328 ms | 2.31 | 9.23 | 11%\* | ok | 1.0x (13%) |
| direct | ref_pardiso_spd | f64 | 96^2 grid, nnz 45696 | 3.214 ms | 3.48 | 0.23 | 1% | ok | 1.4x (18%) |
| direct | ref_pardiso_spd.analyze | f64 | 96^2 grid, nnz 45696 | 1.994 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct | ref_pardiso_spd.factor | f64 | 96^2 grid, nnz 45696 | 0.611 ms | 18.31 | 1.20 | 3% | ok | 3.1x (39%) |
| direct | ref_pardiso_spd.solve | f64 | 96^2 grid, nnz 45696 | 0.498 ms | 1.52 | 6.07 | 7%\* | ok | 0.9x (12%) |
| direct | ref_pardiso_lu | f64 | 96^2 grid, nnz 45696 | 5.549 ms | 4.03 | 0.13 | 1% | ok | 1.4x (17%) |
| direct | ref_pardiso_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.510 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | ref_pardiso_lu.factor | f64 | 96^2 grid, nnz 45696 | 1.133 ms | 19.75 | 0.65 | 3% | ok | 2.9x (36%) |
| direct | ref_pardiso_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.636 ms | 1.19 | 4.75 | 6%\* | ok | 0.9x (11%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.825 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 206.696 ms | 1.49 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.557 ms | 2.16 | 8.66 | 11%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.452 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 58.289 ms | 5.29 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.648 ms | 2.04 | 8.18 | 10%\* | ok | 0.9x (11%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.780 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 542.570 ms | 1.14 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.717 ms | 0.44 | 1.75 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.116 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 439.433 ms | 1.40 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.739 ms | 0.44 | 1.74 | 2%\* | ok | 1.0x (12%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 3.808 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 8.428 ms | 36.62 | 0.10 | 6% | ok | 1.4x (17%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 0.511 ms | 6.59 | 26.36 | 32%\* | ok | 1.0x (12%) |
| direct3d | ref_pardiso_spd.analyze | f64 | 20^3 grid, nnz 53600 | 7.068 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | ref_pardiso_spd.factor | f64 | 20^3 grid, nnz 53600 | 2.366 ms | 130.44 | 0.36 | 22% | ok | 4.1x (51%) |
| direct3d | ref_pardiso_spd.solve | f64 | 20^3 grid, nnz 53600 | 0.947 ms | 3.56 | 14.24 | 17%\* | ok | 0.9x (12%) |
| direct3d | ref_pardiso_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.120 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | ref_pardiso_lu.factor | f64 | 20^3 grid, nnz 53600 | 5.545 ms | 111.31 | 0.15 | 19% | ok | 3.5x (43%) |
| direct3d | ref_pardiso_lu.solve | f64 | 20^3 grid, nnz 53600 | 1.164 ms | 2.89 | 11.58 | 14%\* | ok | 0.9x (11%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.521 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 33.708 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.457 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.315 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.695 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.796 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.071 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.818 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.971 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 17.772 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.360 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 15.821 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 18.620 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.041 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.492 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 11.339 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.062 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 1.006 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.026 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.147 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.971 ms | 2.70 | 30.22 | 37% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.201 ms | 2.18 | 24.41 | 30% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.538 ms | 1.70 | 19.07 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.297 ms | 2.89 | 21.86 | 27% | ok | 1.0x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.272 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 31.059 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 32.654 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.908 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 4.809 ms | 3.49 | 13.96 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 5.162 ms | 52.01 | 0.81 | 9% | ok | 7.6x (94%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.543 ms | 30.90 | 32.84 | 40%\* | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.325 ms | 7.22 | 28.87 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.984 ms | 17.05 | 34.10 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.685 ms | 9.96 | 9.96 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.685 ms | 9.96 | 9.96 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.373 ms | 113.12 | 1.33 | n/a | ok | 7.4x (92%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.786 ms | 96.35 | 0.75 | n/a | ok | 7.5x (93%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.354 ms | 198.26 | 1.16 | n/a | ok | 7.6x (94%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.354 ms | 198.25 | 1.16 | n/a | ok | 7.6x (94%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.552 ms | 75.58 | 0.44 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

