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

