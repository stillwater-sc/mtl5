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
| bandwidth | read 37.8, update 41.5, triad 31.5 GB/s (ceiling 41.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 87% | 103% | 0 |
| blas2 | 10 | 91% | 96% | 0 |
| blas3 | 10 | 65% | 91% | 0 |
| factor | 12 | 8% | 21% | 0 |
| eigen | 8 | 7% | 27% | 0 |
| elementwise | 5 | 7% | 11% | 0 |
| sparse | 5 | 52% | 88% | 0 |
| direct | 7 | 1% | 2% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 54% | 72% | 0 |
| mg | 1 | 49% | 49% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 85% | 132% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 3.697 ms | 4.54 | 36.30 | 87% | ok |
| blas1 | dot | f32 | n=8388608 | 1.704 ms | 9.85 | 39.39 | 95% | ok |
| blas1 | axpy | f64 | n=8388608 | 4.786 ms | 3.51 | 42.06 | 101% | ok |
| blas1 | axpy | f32 | n=8388608 | 2.357 ms | 7.12 | 42.71 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 3.459 ms | 2.42 | 38.80 | 93% | ok |
| blas1 | two_norm | f64 | n=8388608 | 2.212 ms | 7.59 | 30.34 | 73% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.022 ms | 2.09 | 16.69 | 40% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.036 ms | 1.19 | 9.54 | 23% | ok |
| blas1 | sum | f64 | n=8388608 | 3.964 ms | 2.12 | 16.93 | 41% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 71.043 ms | 2.60 | 1.89 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.419 ms | 9.81 | 39.27 | 95% | ok |
| blas2 | gemv | f32 | n=4096 | 1.685 ms | 19.92 | 39.85 | 96% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 9.430 ms | 3.56 | 14.24 | 34% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.014 ms | 8.36 | 16.73 | 40% | ok |
| blas2 | symv | f64 | n=4096 | 6.205 ms | 5.41 | 10.83 | 26% | ok |
| blas2 | trmv | f64 | n=4096 | 1.796 ms | 9.34 | 37.41 | 90% | ok |
| blas2 | trmv | f32 | n=4096 | 0.864 ms | 19.43 | 38.89 | 94% | ok |
| blas2 | trsv | f64 | n=4096 | 1.818 ms | 9.23 | 36.95 | 89% | ok |
| blas2 | trsv | f32 | n=4096 | 0.889 ms | 18.88 | 37.79 | 91% | ok |
| blas2 | ger | f64 | n=4096 | 7.139 ms | 4.70 | 37.61 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.787 ms | 70.87 | 1.66 | 91% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.896 ms | 141.60 | 1.66 | 90% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.626 ms | 51.11 | 2.40 | 65% | ok |
| blas3 | symm | f64 | 512x512x512 | 4.040 ms | 66.44 | 1.56 | 85% | ok |
| blas3 | syrk | f64 | 512x512x512 | 9.819 ms | 13.67 | 0.43 | 17% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 20.122 ms | 13.34 | 0.31 | 17% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.705 ms | 49.62 | 1.94 | 63% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.748 ms | 48.84 | 1.91 | 62% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.283 ms | 104.59 | 2.04 | 67% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.320 ms | 0.40 | 3.19 | 8% | ok |
| factor | lu | f64 | 512x512 | 10.475 ms | 8.54 | 0.40 | 11% | ok |
| factor | lu | f32 | 512x512 | 3.391 ms | 26.38 | 0.62 | 17% | ok |
| factor | cholesky | f64 | 512x512 | 7.083 ms | 6.32 | 0.59 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.545 ms | 6.84 | 0.32 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.201 ms | 6.21 | 0.58 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 46.075 ms | 0.97 | 0.09 | 1% | ok |
| factor | qr | f64 | 512x512 | 232.890 ms | 0.77 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 24.261 ms | 7.38 | 0.17 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 267.725 ms | 1.67 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.661 ms | 0.63 | 2.52 | 6%\* | ok |
| factor | inv | f64 | 512x512 | 55.093 ms | 4.87 | 0.08 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 5.458 ms | 16.39 | 0.77 | 21% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 16.564 ms | 1.35 | 0.06 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 56.664 ms | 2.66 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 742.662 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 30.493 ms | 5.50 | 0.03 | 7% | ok |
| eigen | schur | f64 | 256x256 | 79.022 ms | 5.31 | 0.01 | 7% | ok |
| eigen | svd | f64 | 256x256 | 89.919 ms | 4.10 | 0.01 | 5% | ok |
| eigen | expm | f64 | 256x256 | 11.746 ms | 21.42 | 0.09 | 27% | ok |
| eigen | sylvester | f64 | 256x256 | 175.459 ms | 5.74 | 0.01 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 39.502 ms | 0.21 | 3.40 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 47.053 ms | 0.18 | 2.85 | 7% | ok |
| elementwise | sin | f64 | n=8388608 | 60.051 ms | 0.14 | 2.24 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 29.134 ms | 0.29 | 4.61 | 11% | ok |
| elementwise | erf | f64 | n=8388608 | 203.371 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.743 ms | 3.52 | 36.67 | 88% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.926 ms | 2.83 | 29.42 | 71% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.056 ms | 10.32 | 21.71 | 52% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.828 ms | 0.67 | 2.38 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.277 ms | 1.57 | 13.03 | 31%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 72.881 ms | 0.15 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 27.404 ms | 0.41 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 9.064 ms | 1.23 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.598 ms | 0.55 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 56.829 ms | 0.39 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 17.033 ms | 1.31 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 99.783 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.410 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.034 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.227 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.264 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.443 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 28.591 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 21.367 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.491 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.766 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.680 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 20.212 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.882 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.509 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.686 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 14.976 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 16.074 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 18.336 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.035 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.576 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.142 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.090 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.012 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.023 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.698 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.983 ms | 2.66 | 29.83 | 72% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.312 ms | 1.99 | 22.35 | 54% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.511 ms | 1.73 | 19.41 | 47% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.388 ms | 2.67 | 20.20 | 49% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 28.542 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 39.810 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 39.924 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.597 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 1.896 ms | 8.85 | 35.40 | 85% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 5.999 ms | 44.75 | 0.70 | 57% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.324 ms | 51.73 | 54.96 | 132%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.921 ms | 8.73 | 34.94 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.767 ms | 21.88 | 43.75 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.292 ms | 57.54 | 57.54 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.279 ms | 60.03 | 60.03 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 3.958 ms | 67.83 | 0.79 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.623 ms | 47.74 | 0.37 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.682 ms | 47.25 | 0.28 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 2.115 ms | 126.92 | 0.74 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 1.904 ms | 141.01 | 0.83 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

