## i7-12700k -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, AVX2 | f64 78.3, f32 156.6 GFLOP/s |
| bandwidth | read 38.8, update 41.3, triad 31.4 GB/s (ceiling 41.3) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 89% | 104% | 0 |
| blas2 | 10 | 91% | 99% | 0 |
| blas3 | 10 | 65% | 90% | 0 |
| factor | 12 | 8% | 23% | 0 |
| eigen | 8 | 4% | 25% | 0 |
| elementwise | 5 | 7% | 11% | 0 |
| sparse | 5 | 52% | 89% | 0 |
| direct | 7 | 1% | 2% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 54% | 72% | 0 |
| mg | 1 | 49% | 49% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 85% | 133% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 3.637 ms | 4.61 | 36.91 | 89% | ok |
| blas1 | dot | f32 | n=8388608 | 1.680 ms | 9.99 | 39.96 | 97% | ok |
| blas1 | axpy | f64 | n=8388608 | 4.762 ms | 3.52 | 42.27 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 2.339 ms | 7.17 | 43.04 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 3.468 ms | 2.42 | 38.70 | 94% | ok |
| blas1 | two_norm | f64 | n=8388608 | 2.098 ms | 8.00 | 31.99 | 78% | ok |
| blas1 | one_norm | f64 | n=8388608 | 3.969 ms | 2.11 | 16.91 | 41% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.079 ms | 1.18 | 9.48 | 23% | ok |
| blas1 | sum | f64 | n=8388608 | 3.936 ms | 2.13 | 17.05 | 41% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 70.649 ms | 2.61 | 1.90 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.372 ms | 9.95 | 39.83 | 97% | ok |
| blas2 | gemv | f32 | n=4096 | 1.652 ms | 20.32 | 40.65 | 99% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 10.074 ms | 3.33 | 13.33 | 32% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.424 ms | 7.58 | 15.18 | 37% | ok |
| blas2 | symv | f64 | n=4096 | 6.112 ms | 5.49 | 10.99 | 27% | ok |
| blas2 | trmv | f64 | n=4096 | 1.769 ms | 9.48 | 37.97 | 92% | ok |
| blas2 | trmv | f32 | n=4096 | 0.882 ms | 19.01 | 38.07 | 92% | ok |
| blas2 | trsv | f64 | n=4096 | 1.809 ms | 9.27 | 37.13 | 90% | ok |
| blas2 | trsv | f32 | n=4096 | 0.922 ms | 18.19 | 36.41 | 88% | ok |
| blas2 | ger | f64 | n=4096 | 7.154 ms | 4.69 | 37.53 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.820 ms | 70.28 | 1.65 | 90% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.896 ms | 141.61 | 1.66 | 90% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.647 ms | 50.70 | 2.38 | 65% | ok |
| blas3 | symm | f64 | 512x512x512 | 4.105 ms | 65.39 | 1.53 | 84% | ok |
| blas3 | syrk | f64 | 512x512x512 | 9.929 ms | 13.52 | 0.42 | 17% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 20.086 ms | 13.36 | 0.31 | 17% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.721 ms | 49.32 | 1.93 | 63% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.775 ms | 48.37 | 1.89 | 62% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.304 ms | 102.96 | 2.01 | 66% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.977 ms | 0.41 | 3.27 | 8% | ok |
| factor | lu | f64 | 512x512 | 13.415 ms | 6.67 | 0.31 | 9% | ok |
| factor | lu | f32 | 512x512 | 4.628 ms | 19.33 | 0.45 | 12% | ok |
| factor | cholesky | f64 | 512x512 | 7.059 ms | 6.34 | 0.59 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.612 ms | 6.77 | 0.32 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.159 ms | 6.25 | 0.59 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 46.381 ms | 0.96 | 0.09 | 1% | ok |
| factor | qr | f64 | 512x512 | 216.184 ms | 0.83 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 23.208 ms | 7.71 | 0.18 | 10% | ok |
| factor | hessenberg | f64 | 512x512 | 252.072 ms | 1.77 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.571 ms | 0.67 | 2.67 | 6%\* | ok |
| factor | inv | f64 | 512x512 | 54.947 ms | 4.89 | 0.08 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 4.951 ms | 18.07 | 0.85 | 23% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 16.721 ms | 1.34 | 0.06 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 67.231 ms | 2.25 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1041.155 ms | 0.15 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 30.625 ms | 5.48 | 0.03 | 7% | ok |
| eigen | schur | f64 | 256x256 | 92.448 ms | 4.54 | 0.01 | 6% | ok |
| eigen | svd | f64 | 256x256 | 110.589 ms | 3.34 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 12.859 ms | 19.57 | 0.08 | 25% | ok |
| eigen | sylvester | f64 | 256x256 | 292.629 ms | 3.44 | 0.01 | 4% | ok |
| elementwise | exp | f64 | n=8388608 | 39.088 ms | 0.21 | 3.43 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 46.519 ms | 0.18 | 2.89 | 7% | ok |
| elementwise | sin | f64 | n=8388608 | 59.452 ms | 0.14 | 2.26 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 28.584 ms | 0.29 | 4.70 | 11% | ok |
| elementwise | erf | f64 | n=8388608 | 201.270 ms | 0.04 | 0.67 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.740 ms | 3.53 | 36.77 | 89% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.923 ms | 2.84 | 29.51 | 72% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.075 ms | 10.28 | 21.61 | 52% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.824 ms | 0.68 | 2.38 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.309 ms | 1.53 | 12.71 | 31%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 72.920 ms | 0.15 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 27.631 ms | 0.40 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 9.198 ms | 1.22 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.543 ms | 0.55 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 57.072 ms | 0.39 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 17.117 ms | 1.31 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 98.659 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.603 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.106 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.226 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 16.011 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.203 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 28.928 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 21.184 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.516 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.793 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.630 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 20.630 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 23.333 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.635 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.800 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 15.109 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 16.191 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 18.605 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.035 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.575 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.169 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.089 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.019 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.024 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.705 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.984 ms | 2.66 | 29.81 | 72% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.312 ms | 2.00 | 22.36 | 54% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.511 ms | 1.73 | 19.41 | 47% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.413 ms | 2.67 | 20.17 | 49% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.068 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 40.736 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 43.640 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.572 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 1.904 ms | 8.81 | 35.26 | 85% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 6.056 ms | 44.32 | 0.69 | 57% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.326 ms | 51.50 | 54.72 | 133%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.854 ms | 9.05 | 36.20 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.749 ms | 22.40 | 44.81 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.293 ms | 57.21 | 57.21 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.281 ms | 59.65 | 59.65 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 4.004 ms | 67.04 | 0.79 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.624 ms | 47.73 | 0.37 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.604 ms | 47.90 | 0.28 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 2.099 ms | 127.88 | 0.75 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 1.880 ms | 142.82 | 0.84 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

