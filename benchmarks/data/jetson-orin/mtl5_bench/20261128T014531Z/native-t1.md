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

