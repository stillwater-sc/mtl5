## jetson-orin -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 10.7 GFLOP/s, f32 21.5 GFLOP/s |
| ... hardware, NEON | f64 10.7, f32 21.5 GFLOP/s |
| ... this build, NEON | f64 10.7, f32 21.5 GFLOP/s |
| bandwidth | read 11.5, update 16.2, triad 14.8 GB/s (ceiling 16.2) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 74% | 130% | 0 |
| blas2 | 10 | 63% | 124% | 0 |
| blas3 | 10 | 51% | 61% | 0 |
| factor | 12 | 13% | 33% | 0 |
| eigen | 8 | 16% | 33% | 0 |
| elementwise | 5 | 8% | 15% | 0 |
| sparse | 5 | 31% | 72% | 0 |
| direct | 19 | 2% | 9% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 4% | 11% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 62% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 66% | 117% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.597 ms | 1.45 | 11.57 | 71% | ok |
| blas1 | dot | f32 | n=8388608 | 5.628 ms | 2.98 | 11.92 | 74% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.250 ms | 1.37 | 16.43 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 5.992 ms | 2.80 | 16.80 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 6.363 ms | 1.32 | 21.09 | 130% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.600 ms | 3.00 | 11.98 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.503 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.754 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.502 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 192.361 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 13.130 ms | 2.56 | 10.23 | 63% | ok |
| blas2 | gemv | f32 | n=4096 | 6.261 ms | 5.36 | 10.72 | 66% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 13.157 ms | 2.55 | 10.21 | 63% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.115 ms | 5.49 | 10.98 | 68% | ok |
| blas2 | symv | f64 | n=4096 | 25.973 ms | 1.29 | 2.59 | 16% | ok |
| blas2 | trmv | f64 | n=4096 | 9.100 ms | 1.84 | 7.38 | 46% | ok |
| blas2 | trmv | f32 | n=4096 | 4.367 ms | 3.84 | 7.69 | 48% | ok |
| blas2 | trsv | f64 | n=4096 | 9.266 ms | 1.81 | 7.25 | 45% | ok |
| blas2 | trsv | f32 | n=4096 | 4.265 ms | 3.93 | 7.87 | 49% | ok |
| blas2 | ger | f64 | n=4096 | 13.430 ms | 2.50 | 19.99 | 124% | ok |
| blas3 | gemm | f64 | 512x512x512 | 41.264 ms | 6.51 | 0.15 | 61% | ok |
| blas3 | gemm | f32 | 512x512x512 | 20.715 ms | 12.96 | 0.15 | 60% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 25.793 ms | 5.20 | 0.24 | 48% | ok |
| blas3 | symm | f64 | 512x512x512 | 42.183 ms | 6.36 | 0.15 | 59% | ok |
| blas3 | syrk | f64 | 512x512x512 | 24.547 ms | 5.47 | 0.17 | 51% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 49.251 ms | 5.45 | 0.13 | 51% | ok |
| blas3 | trmm | f64 | 512x512x512 | 24.904 ms | 5.39 | 0.21 | 50% | ok |
| blas3 | trsm | f64 | 512x512x512 | 25.626 ms | 5.24 | 0.20 | 49% | ok |
| blas3 | trsm | f32 | 512x512x512 | 12.434 ms | 10.79 | 0.21 | 50% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.908 ms | 0.54 | 4.29 | 26% | ok |
| factor | lu | f64 | 512x512 | 35.288 ms | 2.54 | 0.12 | 24% | ok |
| factor | lu | f32 | 512x512 | 20.613 ms | 4.34 | 0.10 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 34.110 ms | 1.31 | 0.12 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 30.115 ms | 1.49 | 0.07 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 34.677 ms | 1.29 | 0.12 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 16.297 ms | 2.75 | 0.26 | 26% | ok |
| factor | qr | f64 | 512x512 | 350.040 ms | 0.51 | 0.01 | 5% | ok |
| factor | lq | f64 | 512x512 | 93.326 ms | 1.92 | 0.04 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 489.637 ms | 0.91 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 2.011 ms | 0.52 | 2.09 | 13% | ok |
| factor | inv | f64 | 512x512 | 190.589 ms | 1.41 | 0.02 | 13% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 25.227 ms | 3.55 | 0.17 | 33% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.552 ms | 0.36 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 145.446 ms | 1.04 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 962.623 ms | 0.16 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 98.819 ms | 1.70 | 0.01 | 16% | ok |
| eigen | schur | f64 | 256x256 | 214.719 ms | 1.95 | 0.00 | 18% | ok |
| eigen | svd | f64 | 256x256 | 358.942 ms | 1.03 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 71.608 ms | 3.51 | 0.01 | 33% | ok |
| eigen | sylvester | f64 | 256x256 | 518.514 ms | 1.94 | 0.00 | 18% | ok |
| elementwise | exp | f64 | n=8388608 | 75.994 ms | 0.11 | 1.77 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.224 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 179.041 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.257 ms | 0.15 | 2.43 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 127.363 ms | 0.07 | 1.05 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.326 ms | 1.13 | 11.71 | 72% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.056 ms | 0.86 | 8.91 | 55% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.369 ms | 2.41 | 5.07 | 31% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.047 ms | 0.16 | 0.57 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 12.209 ms | 0.16 | 1.36 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 53.240 ms | 0.21 | 0.01 | 2% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 97.573 ms | 0.11 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 36.228 ms | 0.31 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 134.543 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 155.151 ms | 0.14 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 60.717 ms | 0.37 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 112.911 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.113 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 16.967 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 181.287 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.893 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 65.656 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 7.798 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 43.133 ms | 0.26 | 0.02 | 2% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.163 ms | 0.35 | 1.40 | 9% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 16.578 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.799 ms | 0.67 | 0.04 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.104 ms | 0.36 | 1.44 | 9% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.615 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 114.674 ms | 0.20 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.718 ms | 0.10 | 0.39 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 12.356 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 132.866 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.960 ms | 0.09 | 0.38 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 16.910 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 841.530 ms | 0.37 | 0.00 | 3% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 10.460 ms | 0.32 | 1.29 | 8% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 36.976 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 257.588 ms | 1.20 | 0.00 | 11% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.400 ms | 0.32 | 1.30 | 8% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.215 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1887.998 ms | 0.33 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 38.316 ms | 0.09 | 0.35 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 23.477 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1534.386 ms | 0.40 | 0.00 | 4% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 38.182 ms | 0.09 | 0.35 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.152 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 229.851 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.124 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.718 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 51.036 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.917 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 88.517 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.084 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.717 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 57.681 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.415 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 50.649 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 59.218 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.168 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.891 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 45.054 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.522 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.322 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.097 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.388 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.941 ms | 0.89 | 9.97 | 62% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.740 ms | 0.70 | 7.84 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.516 ms | 0.58 | 6.49 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 45.012 ms | 0.85 | 6.46 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 358.295 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 362.065 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 191.835 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.681 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 6.261 ms | 2.68 | 10.72 | 66% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 64.057 ms | 4.19 | 0.07 | 39% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.338 ms | 12.54 | 13.33 | 117% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.648 ms | 2.97 | 11.88 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.719 ms | 6.17 | 12.34 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.242 ms | 13.50 | 13.50 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.347 ms | 7.15 | 7.15 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 26.004 ms | 10.32 | 0.12 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 27.842 ms | 9.64 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 33.358 ms | 8.05 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 8.573 ms | 31.31 | 0.18 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 23.393 ms | 11.47 | 0.07 | n/a | ok |

