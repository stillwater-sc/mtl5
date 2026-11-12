## jetson-orin -- native, 1 thread, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 10.7 GFLOP/s, f32 21.5 GFLOP/s |
| ... hardware, NEON | f64 10.7, f32 21.5 GFLOP/s |
| ... this build, scalar | f64 5.4, f32 5.4 GFLOP/s |
| bandwidth | read 11.5, update 16.1, triad 14.7 GB/s (ceiling 16.1) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 72% | 131% | 0 |
| blas2 | 10 | 35% | 124% | 0 |
| blas3 | 10 | 33% | 33% | 0 |
| factor | 12 | 13% | 27% | 0 |
| eigen | 8 | 16% | 23% | 0 |
| elementwise | 5 | 8% | 15% | 0 |
| sparse | 5 | 31% | 73% | 0 |
| direct | 7 | 1% | 3% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 49% | 60% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 33% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.564 ms | 1.45 | 11.61 | 72% | ok |
| blas1 | dot | f32 | n=8388608 | 6.594 ms | 2.54 | 10.18 | 63% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.215 ms | 1.37 | 16.48 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 5.963 ms | 2.81 | 16.88 | 105% | ok |
| blas1 | scale | f64 | n=8388608 | 6.361 ms | 1.32 | 21.10 | 131% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.597 ms | 3.00 | 11.99 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.501 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.751 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.501 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 192.567 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 17.870 ms | 1.88 | 7.51 | 47% | ok |
| blas2 | gemv | f32 | n=4096 | 11.767 ms | 2.85 | 5.71 | 35% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 15.557 ms | 2.16 | 8.63 | 54% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.317 ms | 5.31 | 10.63 | 66% | ok |
| blas2 | symv | f64 | n=4096 | 21.300 ms | 1.58 | 3.15 | 20% | ok |
| blas2 | trmv | f64 | n=4096 | 12.036 ms | 1.39 | 5.58 | 35% | ok |
| blas2 | trmv | f32 | n=4096 | 26.543 ms | 0.63 | 1.27 | 8% | ok |
| blas2 | trsv | f64 | n=4096 | 11.981 ms | 1.40 | 5.61 | 35% | ok |
| blas2 | trsv | f32 | n=4096 | 26.979 ms | 0.62 | 1.24 | 8% | ok |
| blas2 | ger | f64 | n=4096 | 13.442 ms | 2.50 | 19.97 | 124% | ok |
| blas3 | gemm | f64 | 512x512x512 | 75.376 ms | 3.56 | 0.08 | 33% | ok |
| blas3 | gemm | f32 | 512x512x512 | 38.225 ms | 7.02 | 0.08 | 33% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 48.829 ms | 2.75 | 0.13 | 26% | ok |
| blas3 | symm | f64 | 512x512x512 | 75.854 ms | 3.54 | 0.08 | 33% | ok |
| blas3 | syrk | f64 | 512x512x512 | 51.735 ms | 2.59 | 0.08 | 24% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 93.758 ms | 2.86 | 0.07 | 27% | ok |
| blas3 | trmm | f64 | 512x512x512 | 37.969 ms | 3.53 | 0.14 | 33% | ok |
| blas3 | trsm | f64 | 512x512x512 | 38.218 ms | 3.51 | 0.14 | 33% | ok |
| blas3 | trsm | f32 | 512x512x512 | 19.272 ms | 6.96 | 0.14 | 32% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.391 ms | 0.57 | 4.52 | 28% | ok |
| factor | lu | f64 | 512x512 | 36.144 ms | 2.48 | 0.12 | 23% | ok |
| factor | lu | f32 | 512x512 | 20.408 ms | 4.38 | 0.10 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 31.813 ms | 1.41 | 0.13 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 30.106 ms | 1.49 | 0.07 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 34.490 ms | 1.30 | 0.12 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 84.799 ms | 0.53 | 0.05 | 5% | ok |
| factor | qr | f64 | 512x512 | 338.912 ms | 0.53 | 0.01 | 5% | ok |
| factor | lq | f64 | 512x512 | 91.843 ms | 1.95 | 0.05 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 477.021 ms | 0.94 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 2.005 ms | 0.52 | 2.09 | 13% | ok |
| factor | inv | f64 | 512x512 | 191.878 ms | 1.40 | 0.02 | 13% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 30.666 ms | 2.92 | 0.14 | 27% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.020 ms | 0.37 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 144.117 ms | 1.05 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 969.420 ms | 0.16 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 98.502 ms | 1.70 | 0.01 | 16% | ok |
| eigen | schur | f64 | 256x256 | 214.100 ms | 1.96 | 0.00 | 18% | ok |
| eigen | svd | f64 | 256x256 | 358.747 ms | 1.03 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 102.755 ms | 2.45 | 0.01 | 23% | ok |
| eigen | sylvester | f64 | 256x256 | 535.690 ms | 1.88 | 0.00 | 18% | ok |
| elementwise | exp | f64 | n=8388608 | 75.825 ms | 0.11 | 1.77 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.097 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 178.753 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.073 ms | 0.15 | 2.44 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 127.080 ms | 0.07 | 1.06 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.314 ms | 1.13 | 11.77 | 73% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.060 ms | 0.86 | 8.90 | 55% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.359 ms | 2.41 | 5.07 | 31% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.633 ms | 0.16 | 0.56 | 3% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.841 ms | 0.17 | 1.41 | 9% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 278.321 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 99.064 ms | 0.11 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.607 ms | 0.30 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 132.334 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 188.134 ms | 0.12 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 62.077 ms | 0.36 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 289.333 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.125 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.382 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.126 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 65.382 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.047 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 148.703 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.159 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.650 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.862 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.716 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.792 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.184 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 50.113 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.831 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.888 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.829 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 68.701 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.166 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.852 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 45.116 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.485 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.254 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.089 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.252 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.021 ms | 0.87 | 9.71 | 60% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.741 ms | 0.70 | 7.84 | 49% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 7.737 ms | 0.34 | 3.79 | 24% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 44.906 ms | 0.86 | 6.47 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 355.689 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 415.615 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 201.376 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.983 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 12.504 ms | 1.34 | 5.37 | 33% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 75.454 ms | 3.56 | 0.06 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 4.576 ms | 3.67 | 3.90 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.630 ms | 2.98 | 11.92 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.733 ms | 6.14 | 12.28 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.656 ms | 2.97 | 2.97 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.758 ms | 9.54 | 9.54 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 38.084 ms | 7.05 | 0.08 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 41.114 ms | 6.53 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 31.885 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 31.879 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 60.119 ms | 4.47 | 0.03 | n/a | ok |

