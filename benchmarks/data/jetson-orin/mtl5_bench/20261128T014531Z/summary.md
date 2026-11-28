# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,6, CPUs 0,1,2,3,4,5, clock policy nvpmodel 25W (mode 1), jetson_clocks; schedutil 1344-1344 MHz on cpus 0,1,2,3,4,5, 2026-11-28T02:10:34Z.

## jetson-orin -- native, 1 thread, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 10.7 GFLOP/s, f32 21.5 GFLOP/s |
| ... hardware, NEON | f64 10.7, f32 21.5 GFLOP/s |
| ... this build, scalar | f64 5.4, f32 5.4 GFLOP/s |
| bandwidth | read 11.5, update 16.1, triad 14.7 GB/s (ceiling 16.1) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 72% | 131% | 0 |
| blas2 | 10 | 49% | 124% | 0 |
| blas3 | 10 | 32% | 33% | 0 |
| factor | 12 | 13% | 33% | 0 |
| eigen | 8 | 16% | 23% | 0 |
| elementwise | 5 | 8% | 15% | 0 |
| sparse | 5 | 32% | 73% | 0 |
| direct | 19 | 2% | 9% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 4% | 11% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 49% | 62% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 33% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.623 ms | 1.44 | 11.55 | 72% | ok |
| blas1 | dot | f32 | n=8388608 | 6.594 ms | 2.54 | 10.18 | 63% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.280 ms | 1.37 | 16.39 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 5.961 ms | 2.81 | 16.89 | 105% | ok |
| blas1 | scale | f64 | n=8388608 | 6.360 ms | 1.32 | 21.11 | 131% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.620 ms | 2.99 | 11.94 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.499 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.750 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 192.548 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 12.975 ms | 2.59 | 10.35 | 64% | ok |
| blas2 | gemv | f32 | n=4096 | 7.729 ms | 4.34 | 8.69 | 54% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 17.097 ms | 1.96 | 7.85 | 49% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.439 ms | 5.21 | 10.43 | 65% | ok |
| blas2 | symv | f64 | n=4096 | 26.244 ms | 1.28 | 2.56 | 16% | ok |
| blas2 | trmv | f64 | n=4096 | 9.030 ms | 1.86 | 7.44 | 46% | ok |
| blas2 | trmv | f32 | n=4096 | 5.255 ms | 3.19 | 6.39 | 40% | ok |
| blas2 | trsv | f64 | n=4096 | 8.902 ms | 1.88 | 7.55 | 47% | ok |
| blas2 | trsv | f32 | n=4096 | 4.826 ms | 3.48 | 6.96 | 43% | ok |
| blas2 | ger | f64 | n=4096 | 13.488 ms | 2.49 | 19.91 | 124% | ok |
| blas3 | gemm | f64 | 512x512x512 | 75.156 ms | 3.57 | 0.08 | 33% | ok |
| blas3 | gemm | f32 | 512x512x512 | 38.223 ms | 7.02 | 0.08 | 33% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 50.010 ms | 2.68 | 0.13 | 25% | ok |
| blas3 | symm | f64 | 512x512x512 | 75.488 ms | 3.56 | 0.08 | 33% | ok |
| blas3 | syrk | f64 | 512x512x512 | 51.856 ms | 2.59 | 0.08 | 24% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 95.709 ms | 2.80 | 0.07 | 26% | ok |
| blas3 | trmm | f64 | 512x512x512 | 39.367 ms | 3.41 | 0.13 | 32% | ok |
| blas3 | trsm | f64 | 512x512x512 | 39.289 ms | 3.42 | 0.13 | 32% | ok |
| blas3 | trsm | f32 | 512x512x512 | 19.155 ms | 7.01 | 0.14 | 33% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.831 ms | 0.54 | 4.32 | 27% | ok |
| factor | lu | f64 | 512x512 | 35.944 ms | 2.49 | 0.12 | 23% | ok |
| factor | lu | f32 | 512x512 | 20.367 ms | 4.39 | 0.10 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 31.841 ms | 1.41 | 0.13 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 30.170 ms | 1.48 | 0.07 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 32.334 ms | 1.38 | 0.13 | 13% | ok |
| factor | ldlt_bk | f64 | 512x512 | 16.249 ms | 2.75 | 0.26 | 26% | ok |
| factor | qr | f64 | 512x512 | 330.384 ms | 0.54 | 0.01 | 5% | ok |
| factor | lq | f64 | 512x512 | 93.503 ms | 1.91 | 0.04 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 471.181 ms | 0.95 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 2.079 ms | 0.51 | 2.02 | 13% | ok |
| factor | inv | f64 | 512x512 | 193.574 ms | 1.39 | 0.02 | 13% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 25.583 ms | 3.50 | 0.16 | 33% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.550 ms | 0.36 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 145.511 ms | 1.04 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 900.244 ms | 0.17 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 99.555 ms | 1.69 | 0.01 | 16% | ok |
| eigen | schur | f64 | 256x256 | 216.386 ms | 1.94 | 0.00 | 18% | ok |
| eigen | svd | f64 | 256x256 | 358.935 ms | 1.03 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 102.182 ms | 2.46 | 0.01 | 23% | ok |
| eigen | sylvester | f64 | 256x256 | 538.735 ms | 1.87 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 75.932 ms | 0.11 | 1.77 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.051 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 178.785 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.148 ms | 0.15 | 2.43 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 127.257 ms | 0.07 | 1.05 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.306 ms | 1.14 | 11.81 | 73% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.050 ms | 0.86 | 8.93 | 56% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.126 ms | 2.45 | 5.14 | 32% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.408 ms | 0.16 | 0.56 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.719 ms | 0.17 | 1.42 | 9% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 54.300 ms | 0.21 | 0.01 | 2% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 97.739 ms | 0.11 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 36.162 ms | 0.31 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 132.896 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 152.192 ms | 0.15 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 60.701 ms | 0.37 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 113.467 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.068 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.179 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 181.939 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.892 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 66.165 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 7.831 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 44.400 ms | 0.25 | 0.02 | 2% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.173 ms | 0.35 | 1.39 | 9% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 16.580 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.817 ms | 0.67 | 0.04 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.099 ms | 0.36 | 1.44 | 9% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.596 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 113.385 ms | 0.20 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.643 ms | 0.10 | 0.40 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 12.363 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 128.187 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.665 ms | 0.10 | 0.39 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 16.760 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 840.617 ms | 0.37 | 0.00 | 3% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 9.982 ms | 0.34 | 1.35 | 8% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 37.061 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 257.263 ms | 1.20 | 0.00 | 11% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.097 ms | 0.33 | 1.33 | 8% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.184 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1907.894 ms | 0.32 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.420 ms | 0.09 | 0.37 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 23.498 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1542.360 ms | 0.40 | 0.00 | 4% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.506 ms | 0.09 | 0.37 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.370 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 168.377 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.201 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.973 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 51.275 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.973 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.129 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.443 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.876 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 57.722 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.685 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 50.771 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 59.714 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.169 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.806 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 45.161 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.536 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.339 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.112 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.403 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.945 ms | 0.89 | 9.96 | 62% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.759 ms | 0.70 | 7.80 | 49% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.538 ms | 0.58 | 6.46 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 44.966 ms | 0.86 | 6.46 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 352.825 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 354.475 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 189.155 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.990 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 12.503 ms | 1.34 | 5.37 | 33% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 75.619 ms | 3.55 | 0.06 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 4.566 ms | 3.67 | 3.90 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.636 ms | 2.98 | 11.91 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.709 ms | 6.19 | 12.39 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.672 ms | 2.96 | 2.96 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.760 ms | 9.53 | 9.53 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 38.106 ms | 7.04 | 0.08 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 41.118 ms | 6.53 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 31.886 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 31.882 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 60.135 ms | 4.46 | 0.03 | n/a | ok |

## jetson-orin -- native, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 64.3 GFLOP/s, f32 128.5 GFLOP/s |
| ... hardware, NEON | f64 64.3, f32 128.5 GFLOP/s |
| ... this build, scalar | f64 32.1, f32 32.1 GFLOP/s |
| bandwidth | read 27.9, update 34.7, triad 33.2 GB/s (ceiling 34.7) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 88% | 136% | 0 | 37% |
| blas2 | 10 | 58% | 80% | 0 | 18% |
| blas3 | 10 | 6% | 23% | 0 | 18% |
| factor | 12 | 4% | 23% | 0 | 23% |
| eigen | 8 | 3% | 8% | 0 | 17% |
| elementwise | 5 | 4% | 7% | 0 | 17% |
| sparse | 5 | 26% | 91% | 0 | 18% |
| direct | 19 | 0% | 4% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 1% | 4% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 23% | 29% | 0 | 17% |
| mg | 1 | 20% | 20% | 0 | 18% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 15% | 32% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 70% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.568 ms | 3.67 | 29.38 | 85% | ok | 2.5x (42%) |
| blas1 | dot | f32 | n=8388608 | 2.193 ms | 7.65 | 30.60 | 88% | ok | 3.0x (50%) |
| blas1 | axpy | f64 | n=8388608 | 5.507 ms | 3.05 | 36.56 | 105% | ok | 2.2x (37%) |
| blas1 | axpy | f32 | n=8388608 | 2.646 ms | 6.34 | 38.04 | 110% | ok | 2.3x (38%) |
| blas1 | scale | f64 | n=8388608 | 2.851 ms | 2.94 | 47.07 | 136% | ok | 2.2x (37%) |
| blas1 | two_norm | f64 | n=8388608 | 2.179 ms | 7.70 | 30.80 | 89% | ok | 2.6x (43%) |
| blas1 | one_norm | f64 | n=8388608 | 12.499 ms | 0.67 | 5.37 | 15% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.749 ms | 0.45 | 3.58 | 10% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 15% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 192.341 ms | 0.96 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 5.667 ms | 5.92 | 23.70 | 68% | ok | 2.3x (38%) |
| blas2 | gemv | f32 | n=4096 | 2.418 ms | 13.88 | 27.77 | 80% | ok | 3.2x (53%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 6.175 ms | 5.43 | 21.74 | 63% | ok | 2.8x (46%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.237 ms | 10.37 | 20.74 | 60% | ok | 2.0x (33%) |
| blas2 | symv | f64 | n=4096 | 24.024 ms | 1.40 | 2.80 | 8% | ok | 1.1x (18%) |
| blas2 | trmv | f64 | n=4096 | 9.415 ms | 1.78 | 7.13 | 21% | ok | 1.0x (16%) |
| blas2 | trmv | f32 | n=4096 | 5.266 ms | 3.19 | 6.38 | 18% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 9.088 ms | 1.85 | 7.39 | 21% | ok | 1.0x (16%) |
| blas2 | trsv | f32 | n=4096 | 4.827 ms | 3.48 | 6.96 | 20% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 13.418 ms | 2.50 | 20.01 | 58% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 21.409 ms | 12.54 | 0.29 | 20% | ok | 3.5x (59%) |
| blas3 | gemm | f32 | 512x512x512 | 9.065 ms | 29.61 | 0.35 | 23% | ok | 4.2x (70%) |
| blas3 | gemmt | f64 | 512x512x512 | 48.468 ms | 2.77 | 0.13 | 4% | ok | 1.0x (17%) |
| blas3 | symm | f64 | 512x512x512 | 73.668 ms | 3.64 | 0.09 | 6% | ok | 1.0x (17%) |
| blas3 | syrk | f64 | 512x512x512 | 50.969 ms | 2.63 | 0.08 | 4% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 512x512x512 | 90.922 ms | 2.95 | 0.07 | 5% | ok | 1.1x (18%) |
| blas3 | trmm | f64 | 512x512x512 | 37.218 ms | 3.61 | 0.14 | 6% | ok | 1.1x (18%) |
| blas3 | trsm | f64 | 512x512x512 | 37.921 ms | 3.54 | 0.14 | 6% | ok | 1.0x (17%) |
| blas3 | trsm | f32 | 512x512x512 | 19.335 ms | 6.94 | 0.14 | 5% | ok | 1.0x (17%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.259 ms | 0.57 | 4.59 | 13% | ok | 1.1x (18%) |
| factor | lu | f64 | 512x512 | 24.605 ms | 3.64 | 0.17 | 6% | ok | 1.5x (24%) |
| factor | lu | f32 | 512x512 | 14.433 ms | 6.20 | 0.15 | 5% | ok | 1.4x (24%) |
| factor | cholesky | f64 | 512x512 | 31.787 ms | 1.41 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 30.096 ms | 1.49 | 0.07 | 1% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 32.281 ms | 1.39 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 15.966 ms | 2.80 | 0.26 | 4% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 204.035 ms | 0.88 | 0.02 | 1% | ok | 1.6x (27%) |
| factor | lq | f64 | 512x512 | 67.229 ms | 2.66 | 0.06 | 4% | ok | 1.4x (23%) |
| factor | hessenberg | f64 | 512x512 | 300.860 ms | 1.49 | 0.01 | 2% | ok | 1.6x (26%) |
| factor | householder | f64 | 512x512 | 0.535 ms | 1.96 | 7.84 | 23% | ok | 3.9x (65%) |
| factor | inv | f64 | 512x512 | 181.518 ms | 1.48 | 0.02 | 2% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 19.539 ms | 4.58 | 0.21 | 7% | ok | 1.3x (22%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.677 ms | 0.36 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 145.596 ms | 1.04 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 951.796 ms | 0.16 | 0.00 | 0% | ok | 0.9x (16%) |
| eigen | eigenvalue | f64 | 256x256 | 99.257 ms | 1.69 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 214.107 ms | 1.96 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 358.621 ms | 1.03 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 48.730 ms | 5.16 | 0.02 | 8% | ok | 2.1x (35%) |
| eigen | sylvester | f64 | 256x256 | 505.980 ms | 1.99 | 0.00 | 3% | ok | 1.1x (18%) |
| elementwise | exp | f64 | n=8388608 | 75.846 ms | 0.11 | 1.77 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 108.007 ms | 0.08 | 1.24 | 4% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 178.248 ms | 0.05 | 0.75 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 55.131 ms | 0.15 | 2.43 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 127.123 ms | 0.07 | 1.06 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.863 ms | 3.03 | 31.56 | 91% | ok | 2.7x (45%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.025 ms | 0.87 | 9.00 | 26% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.734 ms | 8.85 | 18.60 | 54% | ok | 3.6x (60%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.412 ms | 0.16 | 0.56 | 2% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.065 ms | 0.18 | 1.50 | 4% | ok | 1.1x (18%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 53.026 ms | 0.21 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 97.678 ms | 0.11 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 36.192 ms | 0.31 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 133.921 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 156.177 ms | 0.14 | 0.00 | 0% | ok | 1.0x (16%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 60.535 ms | 0.37 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 113.432 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.046 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 16.990 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 181.073 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.891 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 65.460 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 7.838 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 44.005 ms | 0.25 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.143 ms | 0.35 | 1.41 | 4% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 16.498 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.764 ms | 0.67 | 0.04 | 1% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.096 ms | 0.36 | 1.44 | 4% | ok | 1.0x (17%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.581 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 116.904 ms | 0.19 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.663 ms | 0.10 | 0.39 | 1% | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 12.337 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 131.122 ms | 0.17 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.641 ms | 0.10 | 0.40 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 16.811 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 816.343 ms | 0.38 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 9.822 ms | 0.34 | 1.37 | 4% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 36.954 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 257.006 ms | 1.20 | 0.00 | 2% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.049 ms | 0.34 | 1.34 | 4% | ok | 1.0x (17%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.302 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1941.815 ms | 0.32 | 0.00 | 0% | ok | 1.0x (16%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.395 ms | 0.09 | 0.37 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 23.570 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1524.818 ms | 0.40 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.210 ms | 0.09 | 0.37 | 1% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 105.990 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 167.924 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 77.786 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.707 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.934 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.777 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 88.653 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.839 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.521 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 57.562 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.424 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 50.571 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 59.516 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.170 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.789 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 44.758 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.527 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.326 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.112 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.366 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.927 ms | 0.89 | 10.02 | 29% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.735 ms | 0.70 | 7.85 | 23% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.523 ms | 0.58 | 6.48 | 19% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 41.893 ms | 0.92 | 6.94 | 20% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 350.535 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 353.272 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 190.171 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 21.086 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 12.503 ms | 1.34 | 5.37 | 15% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.962 ms | 20.71 | 0.32 | 32% | ok | 5.8x (97%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 4.566 ms | 3.67 | 3.90 | 11% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.606 ms | 2.99 | 11.97 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.754 ms | 6.09 | 12.18 | n/a | ok | 1.0x (16%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.672 ms | 2.96 | 2.96 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.757 ms | 9.55 | 9.55 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 9.020 ms | 29.76 | 0.35 | n/a | ok | 4.2x (70%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 7.059 ms | 38.03 | 0.30 | n/a | ok | 5.8x (97%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.477 ms | 49.01 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.473 ms | 49.05 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 10.302 ms | 26.06 | 0.15 | n/a | ok | 5.8x (97%) |

## jetson-orin -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 10.7 GFLOP/s, f32 21.5 GFLOP/s |
| ... hardware, NEON | f64 10.7, f32 21.5 GFLOP/s |
| ... this build, NEON | f64 10.7, f32 21.5 GFLOP/s |
| bandwidth | read 11.5, update 16.1, triad 14.8 GB/s (ceiling 16.1) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 74% | 131% | 0 |
| blas2 | 10 | 60% | 124% | 0 |
| blas3 | 10 | 51% | 61% | 0 |
| factor | 12 | 13% | 33% | 0 |
| eigen | 8 | 16% | 33% | 0 |
| elementwise | 5 | 8% | 15% | 0 |
| sparse | 5 | 32% | 74% | 0 |
| direct | 19 | 2% | 9% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 4% | 11% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 63% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 66% | 117% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.608 ms | 1.45 | 11.56 | 72% | ok |
| blas1 | dot | f32 | n=8388608 | 5.656 ms | 2.97 | 11.87 | 74% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.248 ms | 1.37 | 16.44 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 5.989 ms | 2.80 | 16.81 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 6.356 ms | 1.32 | 21.12 | 131% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.607 ms | 2.99 | 11.97 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.499 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.751 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.502 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 192.412 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 13.689 ms | 2.45 | 9.81 | 61% | ok |
| blas2 | gemv | f32 | n=4096 | 6.151 ms | 5.46 | 10.92 | 68% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 13.891 ms | 2.42 | 9.67 | 60% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.218 ms | 5.40 | 10.80 | 67% | ok |
| blas2 | symv | f64 | n=4096 | 24.667 ms | 1.36 | 2.72 | 17% | ok |
| blas2 | trmv | f64 | n=4096 | 9.371 ms | 1.79 | 7.17 | 44% | ok |
| blas2 | trmv | f32 | n=4096 | 4.256 ms | 3.94 | 7.89 | 49% | ok |
| blas2 | trsv | f64 | n=4096 | 9.292 ms | 1.81 | 7.23 | 45% | ok |
| blas2 | trsv | f32 | n=4096 | 4.154 ms | 4.04 | 8.08 | 50% | ok |
| blas2 | ger | f64 | n=4096 | 13.425 ms | 2.50 | 20.00 | 124% | ok |
| blas3 | gemm | f64 | 512x512x512 | 41.367 ms | 6.49 | 0.15 | 60% | ok |
| blas3 | gemm | f32 | 512x512x512 | 20.465 ms | 13.12 | 0.15 | 61% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 25.782 ms | 5.21 | 0.24 | 49% | ok |
| blas3 | symm | f64 | 512x512x512 | 42.254 ms | 6.35 | 0.15 | 59% | ok |
| blas3 | syrk | f64 | 512x512x512 | 24.534 ms | 5.47 | 0.17 | 51% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 49.172 ms | 5.46 | 0.13 | 51% | ok |
| blas3 | trmm | f64 | 512x512x512 | 24.877 ms | 5.40 | 0.21 | 50% | ok |
| blas3 | trsm | f64 | 512x512x512 | 25.848 ms | 5.19 | 0.20 | 48% | ok |
| blas3 | trsm | f32 | 512x512x512 | 12.521 ms | 10.72 | 0.21 | 50% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.673 ms | 0.55 | 4.39 | 27% | ok |
| factor | lu | f64 | 512x512 | 35.479 ms | 2.52 | 0.12 | 23% | ok |
| factor | lu | f32 | 512x512 | 20.593 ms | 4.35 | 0.10 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 31.897 ms | 1.40 | 0.13 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 30.109 ms | 1.49 | 0.07 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 32.568 ms | 1.37 | 0.13 | 13% | ok |
| factor | ldlt_bk | f64 | 512x512 | 16.265 ms | 2.75 | 0.26 | 26% | ok |
| factor | qr | f64 | 512x512 | 336.687 ms | 0.53 | 0.01 | 5% | ok |
| factor | lq | f64 | 512x512 | 93.377 ms | 1.92 | 0.04 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 474.138 ms | 0.94 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 2.039 ms | 0.52 | 2.06 | 13% | ok |
| factor | inv | f64 | 512x512 | 192.099 ms | 1.40 | 0.02 | 13% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 25.620 ms | 3.49 | 0.16 | 33% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.631 ms | 0.36 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 145.491 ms | 1.04 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 900.422 ms | 0.17 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 99.290 ms | 1.69 | 0.01 | 16% | ok |
| eigen | schur | f64 | 256x256 | 215.670 ms | 1.94 | 0.00 | 18% | ok |
| eigen | svd | f64 | 256x256 | 358.949 ms | 1.03 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 71.463 ms | 3.52 | 0.01 | 33% | ok |
| eigen | sylvester | f64 | 256x256 | 520.950 ms | 1.93 | 0.00 | 18% | ok |
| elementwise | exp | f64 | n=8388608 | 75.963 ms | 0.11 | 1.77 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.127 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 179.199 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.182 ms | 0.15 | 2.43 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 127.395 ms | 0.07 | 1.05 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.295 ms | 1.14 | 11.87 | 74% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.064 ms | 0.85 | 8.89 | 55% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.075 ms | 2.45 | 5.16 | 32% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.240 ms | 0.16 | 0.57 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.703 ms | 0.17 | 1.42 | 9% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 54.440 ms | 0.21 | 0.01 | 2% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 97.843 ms | 0.11 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 35.965 ms | 0.31 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 136.383 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 157.206 ms | 0.14 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 72.111 ms | 0.31 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 106.114 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.087 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.110 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 179.520 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.892 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 67.130 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 7.750 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 44.191 ms | 0.25 | 0.02 | 2% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.135 ms | 0.35 | 1.42 | 9% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 16.474 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.763 ms | 0.67 | 0.04 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.086 ms | 0.36 | 1.45 | 9% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.573 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 116.440 ms | 0.19 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.610 ms | 0.10 | 0.40 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 12.296 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 131.081 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.575 ms | 0.10 | 0.40 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 16.828 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 846.907 ms | 0.36 | 0.00 | 3% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 9.876 ms | 0.34 | 1.36 | 8% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 35.899 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 256.229 ms | 1.20 | 0.00 | 11% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.065 ms | 0.33 | 1.34 | 8% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 15.949 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1914.167 ms | 0.32 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.387 ms | 0.09 | 0.37 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 22.761 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1501.403 ms | 0.41 | 0.00 | 4% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.410 ms | 0.09 | 0.37 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.236 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 167.422 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.021 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.465 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.874 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.606 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 87.486 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.386 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.677 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 57.362 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.206 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 50.303 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 58.676 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.167 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.798 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 45.539 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.509 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.310 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.112 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.353 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.878 ms | 0.91 | 10.19 | 63% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.748 ms | 0.70 | 7.82 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.521 ms | 0.58 | 6.49 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 44.677 ms | 0.86 | 6.51 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 359.373 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 363.098 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 189.779 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.547 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 6.261 ms | 2.68 | 10.72 | 66% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 64.078 ms | 4.19 | 0.07 | 39% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.338 ms | 12.54 | 13.33 | 117% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.616 ms | 2.99 | 11.95 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.719 ms | 6.17 | 12.34 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.220 ms | 13.75 | 13.75 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.347 ms | 7.15 | 7.15 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 26.092 ms | 10.29 | 0.12 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 27.143 ms | 9.89 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 33.376 ms | 8.04 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 8.554 ms | 31.38 | 0.18 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 23.374 ms | 11.48 | 0.07 | n/a | ok |

## jetson-orin -- native-fast, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 64.3 GFLOP/s, f32 128.5 GFLOP/s |
| ... hardware, NEON | f64 64.1, f32 128.5 GFLOP/s |
| ... this build, NEON | f64 64.3, f32 128.4 GFLOP/s |
| bandwidth | read 27.1, update 34.5, triad 33.1 GB/s (ceiling 34.5) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 89% | 136% | 0 | 39% |
| blas2 | 10 | 49% | 84% | 0 | 17% |
| blas3 | 10 | 30% | 51% | 0 | 59% |
| factor | 12 | 4% | 23% | 0 | 24% |
| eigen | 8 | 3% | 9% | 0 | 17% |
| elementwise | 5 | 4% | 7% | 0 | 17% |
| sparse | 5 | 26% | 94% | 0 | 17% |
| direct | 19 | 0% | 4% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 1% | 4% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 23% | 30% | 0 | 17% |
| mg | 1 | 20% | 20% | 0 | 18% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 34% | 39% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 81% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.269 ms | 3.93 | 31.44 | 91% | ok | 2.7x (45%) |
| blas1 | dot | f32 | n=8388608 | 2.251 ms | 7.45 | 29.81 | 86% | ok | 2.5x (42%) |
| blas1 | axpy | f64 | n=8388608 | 5.273 ms | 3.18 | 38.18 | 111% | ok | 2.3x (39%) |
| blas1 | axpy | f32 | n=8388608 | 2.461 ms | 6.82 | 40.90 | 119% | ok | 2.4x (41%) |
| blas1 | scale | f64 | n=8388608 | 2.851 ms | 2.94 | 47.07 | 136% | ok | 2.2x (37%) |
| blas1 | two_norm | f64 | n=8388608 | 2.183 ms | 7.69 | 30.74 | 89% | ok | 2.6x (43%) |
| blas1 | one_norm | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 16% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.751 ms | 0.45 | 3.58 | 10% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 16% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 191.410 ms | 0.96 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 5.560 ms | 6.04 | 24.15 | 70% | ok | 2.5x (41%) |
| blas2 | gemv | f32 | n=4096 | 2.315 ms | 14.49 | 29.00 | 84% | ok | 2.7x (44%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.953 ms | 4.22 | 16.88 | 49% | ok | 1.7x (29%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.640 ms | 9.22 | 18.45 | 53% | ok | 1.7x (28%) |
| blas2 | symv | f64 | n=4096 | 23.854 ms | 1.41 | 2.82 | 8% | ok | 1.0x (17%) |
| blas2 | trmv | f64 | n=4096 | 9.388 ms | 1.79 | 7.16 | 21% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 4.234 ms | 3.96 | 7.93 | 23% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 9.293 ms | 1.81 | 7.23 | 21% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 4.148 ms | 4.04 | 8.10 | 23% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 13.422 ms | 2.50 | 20.00 | 58% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 8.184 ms | 32.80 | 0.77 | 51% | ok | 5.1x (84%) |
| blas3 | gemm | f32 | 512x512x512 | 4.106 ms | 65.38 | 0.77 | 51% | ok | 5.0x (83%) |
| blas3 | gemmt | f64 | 512x512x512 | 7.459 ms | 17.99 | 0.84 | 28% | ok | 3.5x (58%) |
| blas3 | symm | f64 | 512x512x512 | 9.124 ms | 29.42 | 0.69 | 46% | ok | 4.6x (77%) |
| blas3 | syrk | f64 | 512x512x512 | 6.987 ms | 19.21 | 0.60 | 30% | ok | 3.5x (59%) |
| blas3 | syr2k | f64 | 512x512x512 | 13.954 ms | 19.24 | 0.45 | 30% | ok | 3.5x (59%) |
| blas3 | trmm | f64 | 512x512x512 | 10.764 ms | 12.47 | 0.49 | 19% | ok | 2.3x (39%) |
| blas3 | trsm | f64 | 512x512x512 | 11.628 ms | 11.54 | 0.45 | 18% | ok | 2.2x (37%) |
| blas3 | trsm | f32 | 512x512x512 | 5.685 ms | 23.61 | 0.46 | 18% | ok | 2.2x (37%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.674 ms | 0.55 | 4.39 | 13% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 24.423 ms | 3.66 | 0.17 | 6% | ok | 1.5x (24%) |
| factor | lu | f32 | 512x512 | 14.396 ms | 6.22 | 0.15 | 5% | ok | 1.4x (24%) |
| factor | cholesky | f64 | 512x512 | 31.745 ms | 1.41 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 30.177 ms | 1.48 | 0.07 | 1% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 32.299 ms | 1.39 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 16.088 ms | 2.78 | 0.26 | 4% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 205.232 ms | 0.87 | 0.02 | 1% | ok | 1.6x (27%) |
| factor | lq | f64 | 512x512 | 59.632 ms | 3.00 | 0.07 | 5% | ok | 1.6x (26%) |
| factor | hessenberg | f64 | 512x512 | 288.147 ms | 1.55 | 0.01 | 2% | ok | 1.6x (27%) |
| factor | householder | f64 | 512x512 | 0.519 ms | 2.02 | 8.08 | 23% | ok | 3.9x (66%) |
| factor | inv | f64 | 512x512 | 180.677 ms | 1.49 | 0.02 | 2% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 19.435 ms | 4.60 | 0.22 | 7% | ok | 1.3x (22%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.953 ms | 0.36 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 145.126 ms | 1.04 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 900.163 ms | 0.17 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 99.553 ms | 1.69 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 214.585 ms | 1.95 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 358.540 ms | 1.03 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 42.607 ms | 5.91 | 0.02 | 9% | ok | 1.7x (28%) |
| eigen | sylvester | f64 | 256x256 | 501.665 ms | 2.01 | 0.00 | 3% | ok | 1.0x (17%) |
| elementwise | exp | f64 | n=8388608 | 75.116 ms | 0.11 | 1.79 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 107.225 ms | 0.08 | 1.25 | 4% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 177.161 ms | 0.05 | 0.76 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 54.847 ms | 0.15 | 2.45 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 126.130 ms | 0.07 | 1.06 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.838 ms | 3.12 | 32.49 | 94% | ok | 2.7x (46%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.088 ms | 0.85 | 8.82 | 26% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.759 ms | 8.80 | 18.50 | 54% | ok | 3.6x (60%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.278 ms | 0.16 | 0.57 | 2% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.224 ms | 0.18 | 1.48 | 4% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 54.191 ms | 0.21 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 97.480 ms | 0.11 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 36.047 ms | 0.31 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 136.564 ms | 0.16 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 155.552 ms | 0.14 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 60.965 ms | 0.37 | 0.01 | 1% | ok | 1.2x (20%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 104.892 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.067 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 16.993 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 178.988 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.879 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 65.968 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 7.797 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 44.155 ms | 0.25 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.127 ms | 0.36 | 1.42 | 4% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 16.721 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.800 ms | 0.67 | 0.04 | 1% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.120 ms | 0.36 | 1.43 | 4% | ok | 1.0x (16%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.610 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 117.154 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.529 ms | 0.10 | 0.40 | 1% | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 12.320 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 130.791 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.524 ms | 0.10 | 0.40 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 16.849 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 843.970 ms | 0.37 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 9.663 ms | 0.35 | 1.39 | 4% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 37.129 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 257.883 ms | 1.20 | 0.00 | 2% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.031 ms | 0.34 | 1.34 | 4% | ok | 1.0x (17%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.144 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1915.723 ms | 0.32 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.356 ms | 0.09 | 0.37 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 23.243 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1521.810 ms | 0.41 | 0.00 | 1% | ok | 1.0x (16%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.292 ms | 0.09 | 0.37 | 1% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.134 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 167.079 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.060 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.605 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 51.081 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.845 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 88.247 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.802 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.859 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 57.491 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.352 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 50.607 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 59.171 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.167 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.772 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 45.790 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.482 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.320 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.112 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.371 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.872 ms | 0.91 | 10.21 | 30% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.732 ms | 0.70 | 7.86 | 23% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.519 ms | 0.58 | 6.49 | 19% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 42.050 ms | 0.91 | 6.91 | 20% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 360.664 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 365.177 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 189.424 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.768 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| mixed | dot | f32>f64 | n=8388608 | 6.261 ms | 2.68 | 10.72 | 31% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.292 ms | 21.84 | 0.34 | 34% | ok | 5.2x (87%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.317 ms | 12.74 | 13.54 | 39% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.675 ms | 2.96 | 11.82 | n/a | ok | 1.0x (16%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.695 ms | 6.23 | 12.45 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.224 ms | 13.71 | 13.71 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.346 ms | 7.15 | 7.15 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 5.163 ms | 51.99 | 0.61 | n/a | ok | 5.1x (84%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.313 ms | 50.53 | 0.39 | n/a | ok | 5.1x (85%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 6.464 ms | 41.53 | 0.24 | n/a | ok | 5.2x (86%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.762 ms | 152.37 | 0.89 | n/a | ok | 4.9x (81%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.533 ms | 59.21 | 0.35 | n/a | ok | 5.2x (86%) |

## jetson-orin -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 10.7 GFLOP/s, f32 21.5 GFLOP/s |
| ... hardware, NEON | f64 10.7, f32 21.5 GFLOP/s |
| ... this build, scalar | f64 5.4, f32 5.4 GFLOP/s |
| bandwidth | read 11.5, update 16.0, triad 14.7 GB/s (ceiling 16.0) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 72% | 132% | 0 |
| blas2 | 10 | 65% | 126% | 0 |
| blas3 | 10 | 89% | 96% | 0 |
| factor | 12 | 42% | 61% | 0 |
| eigen | 8 | 37% | 79% | 0 |
| elementwise | 5 | 8% | 15% | 0 |
| sparse | 5 | 32% | 74% | 0 |
| direct | 23 | 2% | 14% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 15 | 8% | 36% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 49% | 62% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 34% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.621 ms | 1.44 | 11.55 | 72% | ok |
| blas1 | dot | f32 | n=8388608 | 6.594 ms | 2.54 | 10.18 | 64% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.235 ms | 1.37 | 16.45 | 103% | ok |
| blas1 | axpy | f32 | n=8388608 | 5.959 ms | 2.82 | 16.89 | 106% | ok |
| blas1 | scale | f64 | n=8388608 | 6.373 ms | 1.32 | 21.06 | 132% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.616 ms | 2.99 | 11.95 | 75% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.501 ms | 0.67 | 5.37 | 34% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.751 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 34% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 192.451 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 12.655 ms | 2.65 | 10.61 | 66% | ok |
| blas2 | gemv | f32 | n=4096 | 5.882 ms | 5.71 | 11.42 | 71% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 12.980 ms | 2.59 | 10.35 | 65% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.777 ms | 5.81 | 11.62 | 73% | ok |
| blas2 | symv | f64 | n=4096 | 15.932 ms | 2.11 | 4.22 | 26% | ok |
| blas2 | trmv | f64 | n=4096 | 8.609 ms | 1.95 | 7.80 | 49% | ok |
| blas2 | trmv | f32 | n=4096 | 4.836 ms | 3.47 | 6.95 | 43% | ok |
| blas2 | trsv | f64 | n=4096 | 8.587 ms | 1.95 | 7.82 | 49% | ok |
| blas2 | trsv | f32 | n=4096 | 4.810 ms | 3.49 | 6.98 | 44% | ok |
| blas2 | ger | f64 | n=4096 | 13.354 ms | 2.51 | 20.11 | 126% | ok |
| blas3 | gemm | f64 | 512x512x512 | 26.609 ms | 10.09 | 0.24 | 94% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.092 ms | 20.50 | 0.24 | 96% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 16.552 ms | 8.11 | 0.38 | 76% | ok |
| blas3 | symm | f64 | 512x512x512 | 26.663 ms | 10.07 | 0.24 | 94% | ok |
| blas3 | syrk | f64 | 512x512x512 | 14.439 ms | 9.30 | 0.29 | 87% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 28.062 ms | 9.57 | 0.22 | 89% | ok |
| blas3 | trmm | f64 | 512x512x512 | 13.598 ms | 9.87 | 0.39 | 92% | ok |
| blas3 | trsm | f64 | 512x512x512 | 14.953 ms | 8.98 | 0.35 | 84% | ok |
| blas3 | trsm | f32 | 512x512x512 | 7.554 ms | 17.77 | 0.35 | 83% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.855 ms | 0.54 | 4.31 | 27% | ok |
| factor | lu | f64 | 512x512 | 13.578 ms | 6.59 | 0.31 | 61% | ok |
| factor | lu | f32 | 512x512 | 8.407 ms | 10.64 | 0.25 | 50% | ok |
| factor | cholesky | f64 | 512x512 | 7.868 ms | 5.69 | 0.53 | 53% | ok |
| factor | cholesky | f32 | 512x512 | 4.979 ms | 8.98 | 0.42 | 42% | ok |
| factor | ldlt | f64 | 512x512 | 32.302 ms | 1.39 | 0.13 | 13% | ok |
| factor | ldlt_bk | f64 | 512x512 | 16.182 ms | 2.76 | 0.26 | 26% | ok |
| factor | qr | f64 | 512x512 | 28.355 ms | 6.31 | 0.15 | 59% | ok |
| factor | lq | f64 | 512x512 | 92.031 ms | 1.94 | 0.05 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 473.834 ms | 0.94 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 1.997 ms | 0.53 | 2.10 | 13% | ok |
| factor | inv | f64 | 512x512 | 170.586 ms | 1.57 | 0.02 | 15% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 14.814 ms | 6.04 | 0.28 | 56% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 10.383 ms | 2.15 | 0.10 | 20% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 144.411 ms | 1.05 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 897.825 ms | 0.17 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 63.041 ms | 2.66 | 0.02 | 25% | ok |
| eigen | schur | f64 | 256x256 | 90.075 ms | 4.66 | 0.01 | 43% | ok |
| eigen | svd | f64 | 256x256 | 43.365 ms | 8.51 | 0.02 | 79% | ok |
| eigen | expm | f64 | 256x256 | 56.396 ms | 4.46 | 0.02 | 42% | ok |
| eigen | sylvester | f64 | 256x256 | 255.306 ms | 3.94 | 0.01 | 37% | ok |
| elementwise | exp | f64 | n=8388608 | 76.019 ms | 0.11 | 1.77 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.081 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 178.491 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.150 ms | 0.15 | 2.43 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 126.675 ms | 0.07 | 1.06 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.308 ms | 1.13 | 11.80 | 74% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.029 ms | 0.86 | 8.99 | 56% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.403 ms | 2.41 | 5.06 | 32% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.126 ms | 0.16 | 0.57 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 12.200 ms | 0.16 | 1.36 | 9% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 54.872 ms | 0.20 | 0.01 | 2% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 101.192 ms | 0.11 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 36.205 ms | 0.31 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 131.914 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 149.064 ms | 0.15 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 61.132 ms | 0.37 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 105.752 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.111 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.273 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 181.346 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.894 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 66.116 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 7.814 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 44.448 ms | 0.25 | 0.02 | 2% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.194 ms | 0.34 | 1.38 | 9% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 16.548 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.907 ms | 0.66 | 0.04 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.160 ms | 0.35 | 1.40 | 9% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.654 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 114.073 ms | 0.20 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.797 ms | 0.10 | 0.39 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 12.324 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 125.942 ms | 0.18 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.758 ms | 0.10 | 0.39 | 2% | ok |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 28.215 ms | 0.40 | 0.03 | 4% | ok |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 5.973 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 20.499 ms | 0.55 | 0.04 | 5% | ok |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.372 ms | 0.55 | 2.20 | 14% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 16.765 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 830.224 ms | 0.37 | 0.00 | 3% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 10.457 ms | 0.32 | 1.29 | 8% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 37.165 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 256.620 ms | 1.20 | 0.00 | 11% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.303 ms | 0.33 | 1.31 | 8% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.104 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1898.113 ms | 0.33 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 38.244 ms | 0.09 | 0.35 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 23.337 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1522.103 ms | 0.41 | 0.00 | 4% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 38.025 ms | 0.09 | 0.35 | 2% | ok |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 11.460 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 79.946 ms | 3.86 | 0.01 | 36% | ok |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 3.152 ms | 1.07 | 4.28 | 27% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.106 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 168.020 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.290 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.816 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 51.251 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 50.015 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.167 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.160 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.928 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 57.772 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.746 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 50.763 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 59.814 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.169 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.855 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 44.665 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.509 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.428 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.112 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 4.051 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.941 ms | 0.89 | 9.97 | 62% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.743 ms | 0.70 | 7.83 | 49% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.525 ms | 0.58 | 6.48 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 44.933 ms | 0.86 | 6.47 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 351.469 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 357.363 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 190.228 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.959 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 12.501 ms | 1.34 | 5.37 | 34% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 75.502 ms | 3.56 | 0.06 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 4.584 ms | 3.66 | 3.89 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.651 ms | 2.97 | 11.88 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.705 ms | 6.20 | 12.41 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.675 ms | 2.96 | 2.96 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.760 ms | 9.53 | 9.53 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 38.044 ms | 7.06 | 0.08 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 41.096 ms | 6.53 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 31.891 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 31.875 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 60.141 ms | 4.46 | 0.03 | n/a | ok |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

## jetson-orin -- blas, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 64.3 GFLOP/s, f32 128.5 GFLOP/s |
| ... hardware, NEON | f64 64.3, f32 128.5 GFLOP/s |
| ... this build, scalar | f64 32.1, f32 32.1 GFLOP/s |
| bandwidth | read 27.4, update 34.1, triad 32.4 GB/s (ceiling 34.1) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 88% | 138% | 0 | 37% |
| blas2 | 10 | 74% | 123% | 0 | 43% |
| blas3 | 10 | 72% | 88% | 0 | 85% |
| factor | 12 | 13% | 25% | 0 | 27% |
| eigen | 8 | 7% | 19% | 0 | 20% |
| elementwise | 5 | 4% | 7% | 0 | 17% |
| sparse | 5 | 27% | 95% | 0 | 17% |
| direct | 23 | 1% | 6% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 15 | 2% | 13% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 23% | 30% | 0 | 17% |
| mg | 1 | 20% | 20% | 0 | 18% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 16% | 32% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 70% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.573 ms | 3.67 | 29.35 | 86% | ok | 2.5x (42%) |
| blas1 | dot | f32 | n=8388608 | 2.209 ms | 7.60 | 30.38 | 89% | ok | 3.0x (50%) |
| blas1 | axpy | f64 | n=8388608 | 5.492 ms | 3.05 | 36.66 | 107% | ok | 2.2x (37%) |
| blas1 | axpy | f32 | n=8388608 | 2.460 ms | 6.82 | 40.91 | 120% | ok | 2.4x (40%) |
| blas1 | scale | f64 | n=8388608 | 2.848 ms | 2.95 | 47.12 | 138% | ok | 2.2x (37%) |
| blas1 | two_norm | f64 | n=8388608 | 2.236 ms | 7.50 | 30.01 | 88% | ok | 2.5x (42%) |
| blas1 | one_norm | f64 | n=8388608 | 12.499 ms | 0.67 | 5.37 | 16% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.750 ms | 0.45 | 3.58 | 10% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 16% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 192.271 ms | 0.96 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 4.622 ms | 7.26 | 29.05 | 85% | ok | 2.7x (46%) |
| blas2 | gemv | f32 | n=4096 | 2.281 ms | 14.71 | 29.43 | 86% | ok | 2.6x (43%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 6.331 ms | 5.30 | 21.21 | 62% | ok | 2.1x (34%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.736 ms | 8.98 | 17.97 | 53% | ok | 1.5x (26%) |
| blas2 | symv | f64 | n=4096 | 5.102 ms | 6.58 | 13.17 | 39% | ok | 3.1x (52%) |
| blas2 | trmv | f64 | n=4096 | 2.649 ms | 6.33 | 25.36 | 74% | ok | 3.2x (54%) |
| blas2 | trmv | f32 | n=4096 | 1.333 ms | 12.58 | 25.19 | 74% | ok | 3.6x (60%) |
| blas2 | trsv | f64 | n=4096 | 8.535 ms | 1.97 | 7.87 | 23% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 4.833 ms | 3.47 | 6.95 | 20% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 6.381 ms | 5.26 | 42.08 | 123% | ok | 2.1x (35%) |
| blas3 | gemm | f64 | 512x512x512 | 4.830 ms | 55.58 | 1.30 | 86% | ok | 5.5x (92%) |
| blas3 | gemm | f32 | 512x512x512 | 2.480 ms | 108.23 | 1.27 | 84% | ok | 5.3x (88%) |
| blas3 | gemmt | f64 | 512x512x512 | 4.054 ms | 33.11 | 1.55 | 52% | ok | 4.1x (68%) |
| blas3 | symm | f64 | 512x512x512 | 4.768 ms | 56.30 | 1.32 | 88% | ok | 5.6x (93%) |
| blas3 | syrk | f64 | 512x512x512 | 3.554 ms | 37.76 | 1.18 | 59% | ok | 4.1x (68%) |
| blas3 | syr2k | f64 | 512x512x512 | 6.079 ms | 44.16 | 1.03 | 69% | ok | 4.6x (77%) |
| blas3 | trmm | f64 | 512x512x512 | 2.596 ms | 51.70 | 2.02 | 80% | ok | 5.2x (87%) |
| blas3 | trsm | f64 | 512x512x512 | 2.915 ms | 46.04 | 1.80 | 72% | ok | 5.1x (85%) |
| blas3 | trsm | f32 | 512x512x512 | 1.585 ms | 84.65 | 1.65 | 66% | ok | 4.8x (79%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.862 ms | 0.54 | 4.31 | 13% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 5.681 ms | 15.75 | 0.74 | 25% | ok | 2.4x (40%) |
| factor | lu | f32 | 512x512 | 4.537 ms | 19.72 | 0.46 | 15% | ok | 1.9x (31%) |
| factor | cholesky | f64 | 512x512 | 4.152 ms | 10.77 | 1.01 | 17% | ok | 1.9x (32%) |
| factor | cholesky | f32 | 512x512 | 3.168 ms | 14.12 | 0.66 | 11% | ok | 1.6x (26%) |
| factor | ldlt | f64 | 512x512 | 32.145 ms | 1.39 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 15.835 ms | 2.83 | 0.26 | 4% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 13.486 ms | 13.27 | 0.31 | 21% | ok | 2.1x (35%) |
| factor | lq | f64 | 512x512 | 59.209 ms | 3.02 | 0.07 | 5% | ok | 1.6x (26%) |
| factor | hessenberg | f64 | 512x512 | 296.683 ms | 1.51 | 0.01 | 2% | ok | 1.6x (27%) |
| factor | householder | f64 | 512x512 | 0.536 ms | 1.96 | 7.82 | 23% | ok | 3.7x (62%) |
| factor | inv | f64 | 512x512 | 165.178 ms | 1.63 | 0.03 | 3% | ok | 1.0x (17%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 10.970 ms | 8.16 | 0.38 | 13% | ok | 1.4x (23%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 7.098 ms | 3.15 | 0.15 | 5% | ok | 1.5x (24%) |
| eigen | eigen_symmetric | f64 | 256x256 | 144.498 ms | 1.04 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 928.153 ms | 0.16 | 0.00 | 0% | ok | 1.0x (16%) |
| eigen | eigenvalue | f64 | 256x256 | 56.463 ms | 2.97 | 0.02 | 5% | ok | 1.1x (19%) |
| eigen | schur | f64 | 256x256 | 74.108 ms | 5.66 | 0.01 | 9% | ok | 1.2x (20%) |
| eigen | svd | f64 | 256x256 | 29.745 ms | 12.41 | 0.04 | 19% | ok | 1.5x (24%) |
| eigen | expm | f64 | 256x256 | 36.953 ms | 6.81 | 0.03 | 11% | ok | 1.5x (25%) |
| eigen | sylvester | f64 | 256x256 | 211.350 ms | 4.76 | 0.01 | 7% | ok | 1.2x (20%) |
| elementwise | exp | f64 | n=8388608 | 75.889 ms | 0.11 | 1.77 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 107.945 ms | 0.08 | 1.24 | 4% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 179.126 ms | 0.05 | 0.75 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 55.043 ms | 0.15 | 2.44 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 126.964 ms | 0.07 | 1.06 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.839 ms | 3.12 | 32.47 | 95% | ok | 2.8x (46%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.003 ms | 0.87 | 9.07 | 27% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.769 ms | 8.78 | 18.46 | 54% | ok | 3.6x (61%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.029 ms | 0.16 | 0.57 | 2% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.648 ms | 0.17 | 1.43 | 4% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 54.627 ms | 0.20 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 101.083 ms | 0.11 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 35.957 ms | 0.31 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 133.299 ms | 0.17 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 151.679 ms | 0.15 | 0.00 | 0% | ok | 1.0x (16%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 60.456 ms | 0.37 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 105.701 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.115 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 16.782 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 181.578 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.894 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 65.706 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 7.739 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 44.532 ms | 0.25 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.159 ms | 0.35 | 1.40 | 4% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 16.352 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.802 ms | 0.67 | 0.04 | 1% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.106 ms | 0.36 | 1.44 | 4% | ok | 1.0x (17%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.637 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 113.794 ms | 0.20 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.780 ms | 0.10 | 0.39 | 1% | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 12.321 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 127.996 ms | 0.17 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.738 ms | 0.10 | 0.39 | 1% | ok | 1.0x (17%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 30.035 ms | 0.37 | 0.02 | 1% | ok | 0.9x (16%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 5.915 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 22.335 ms | 0.50 | 0.03 | 1% | ok | 0.9x (15%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.454 ms | 0.52 | 2.08 | 6% | ok | 0.9x (16%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 16.761 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 829.953 ms | 0.37 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 10.284 ms | 0.33 | 1.31 | 4% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 36.782 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 256.526 ms | 1.20 | 0.00 | 2% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.184 ms | 0.33 | 1.32 | 4% | ok | 1.0x (17%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.255 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1883.681 ms | 0.33 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 38.042 ms | 0.09 | 0.35 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 23.349 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1534.570 ms | 0.40 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 38.073 ms | 0.09 | 0.35 | 1% | ok | 1.0x (17%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 11.450 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 148.051 ms | 2.08 | 0.01 | 3% | ok | 0.5x (9%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 3.139 ms | 1.07 | 4.29 | 13% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.379 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 168.730 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.186 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.757 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 51.201 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.974 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 88.999 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.916 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.931 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 57.733 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.521 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 50.683 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 59.390 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.169 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.888 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 44.999 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.540 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.385 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.111 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 4.189 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.885 ms | 0.91 | 10.16 | 30% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.742 ms | 0.70 | 7.84 | 23% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.528 ms | 0.58 | 6.48 | 19% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 42.126 ms | 0.91 | 6.90 | 20% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 354.252 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 359.915 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 187.264 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 21.178 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| mixed | dot | f32>f64 | n=8388608 | 12.506 ms | 1.34 | 5.37 | 16% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.933 ms | 20.76 | 0.32 | 32% | ok | 5.8x (97%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 4.587 ms | 3.66 | 3.89 | 11% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.621 ms | 2.98 | 11.94 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.714 ms | 6.18 | 12.37 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.648 ms | 2.97 | 2.97 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.758 ms | 9.55 | 9.55 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 9.004 ms | 29.81 | 0.35 | n/a | ok | 4.2x (70%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 7.067 ms | 37.98 | 0.30 | n/a | ok | 5.8x (97%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.481 ms | 48.98 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.483 ms | 48.96 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 10.303 ms | 26.05 | 0.15 | n/a | ok | 5.8x (97%) |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

