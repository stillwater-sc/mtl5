## Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz -- native, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-42-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `73d11400a0e3` |
| FMA peak (ceiling) | f64 9.6 GFLOP/s, f32 19.5 GFLOP/s |
| ... hardware, SSE2 | f64 9.6, f32 19.5 GFLOP/s |
| ... this build, scalar | f64 0.1, f32 0.6 GFLOP/s |
| bandwidth | read 9.8, update 9.7, triad 7.8 GB/s (ceiling 9.8) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 24% | 102% | 0 |
| blas2 | 6 | 66% | 93% | 0 |
| blas3 | 10 | 41% | 46% | 0 |
| factor | 12 | 18% | 40% | 0 |
| eigen | 8 | 16% | 27% | 0 |
| elementwise | 5 | 9% | 15% | 0 |
| sparse | 5 | 56% | 108% | 0 |
| direct | 7 | 2% | 5% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 79% | 92% | 0 |
| mg | 1 | 80% | 80% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 24% | 75% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 198.243 ms | 0.08 | 0.68 | 7% | ok |
| blas1 | dot | f32 | n=8388608 | 28.236 ms | 0.59 | 2.38 | 24% | ok |
| blas1 | axpy | f64 | n=8388608 | 224.022 ms | 0.07 | 0.90 | 9% | ok |
| blas1 | axpy | f32 | n=8388608 | 95.701 ms | 0.18 | 1.05 | 11% | ok |
| blas1 | scale | f64 | n=8388608 | 13.455 ms | 0.62 | 9.98 | 102% | ok |
| blas1 | two_norm | f64 | n=8388608 | 190.803 ms | 0.09 | 0.35 | 4% | ok |
| blas1 | one_norm | f64 | n=8388608 | 10.098 ms | 0.83 | 6.65 | 68% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 13.683 ms | 0.61 | 4.90 | 50% | ok |
| blas1 | sum | f64 | n=8388608 | 10.133 ms | 0.83 | 6.62 | 68% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 289.856 ms | 0.64 | 0.46 | 7% | ok |
| blas2 | gemv | f64 | n=4096 | 20.321 ms | 1.65 | 6.61 | 68% | ok |
| blas2 | gemv | f32 | n=4096 | 19.943 ms | 1.68 | 3.37 | 34% | ok |
| blas2 | symv | f64 | n=4096 | 16.725 ms | 2.01 | 4.02 | 41% | ok |
| blas2 | trmv | f64 | n=4096 | 10.726 ms | 1.56 | 6.26 | 64% | ok |
| blas2 | trsv | f64 | n=4096 | 10.480 ms | 1.60 | 6.41 | 66% | ok |
| blas2 | ger | f64 | n=4096 | 29.615 ms | 1.13 | 9.07 | 93% | ok |
| blas3 | gemm | f64 | 512x512x512 | 66.564 ms | 4.03 | 0.09 | 42% | ok |
| blas3 | gemm | f32 | 512x512x512 | 35.988 ms | 7.46 | 0.09 | 38% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 30.421 ms | 4.41 | 0.21 | 46% | ok |
| blas3 | symm | f64 | 512x512x512 | 62.056 ms | 4.33 | 0.10 | 45% | ok |
| blas3 | syrk | f64 | 512x512x512 | 39.830 ms | 3.37 | 0.11 | 35% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 68.388 ms | 3.93 | 0.09 | 41% | ok |
| blas3 | trmm | f64 | 512x512x512 | 35.637 ms | 3.77 | 0.15 | 39% | ok |
| blas3 | trsm | f64 | 512x512x512 | 33.551 ms | 4.00 | 0.16 | 42% | ok |
| blas3 | trsm | f32 | 512x512x512 | 16.855 ms | 7.96 | 0.16 | 41% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 39.116 ms | 0.14 | 1.09 | 11% | ok |
| factor | lu | f64 | 512x512 | 32.285 ms | 2.77 | 0.13 | 29% | ok |
| factor | lu | f32 | 512x512 | 17.280 ms | 5.18 | 0.12 | 27% | ok |
| factor | cholesky | f64 | 512x512 | 25.843 ms | 1.73 | 0.16 | 18% | ok |
| factor | cholesky | f32 | 512x512 | 24.279 ms | 1.84 | 0.09 | 9% | ok |
| factor | ldlt | f64 | 512x512 | 26.204 ms | 1.71 | 0.16 | 18% | ok |
| factor | ldlt_bk | f64 | 512x512 | 142.446 ms | 0.31 | 0.03 | 3% | ok |
| factor | qr | f64 | 512x512 | 531.062 ms | 0.34 | 0.01 | 4% | ok |
| factor | lq | f64 | 512x512 | 77.074 ms | 2.32 | 0.05 | 24% | ok |
| factor | hessenberg | f64 | 512x512 | 646.905 ms | 0.69 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.395 ms | 0.31 | 1.24 | 13%\* | ok |
| factor | inv | f64 | 512x512 | 200.828 ms | 1.34 | 0.02 | 14% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 23.386 ms | 3.83 | 0.18 | 40% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 62.970 ms | 0.36 | 0.02 | 4% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 183.661 ms | 0.82 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1027.840 ms | 0.15 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 110.626 ms | 1.52 | 0.01 | 16% | ok |
| eigen | schur | f64 | 256x256 | 281.036 ms | 1.49 | 0.00 | 16% | ok |
| eigen | svd | f64 | 256x256 | 325.255 ms | 1.13 | 0.00 | 12% | ok |
| eigen | expm | f64 | 256x256 | 95.632 ms | 2.63 | 0.01 | 27% | ok |
| eigen | sylvester | f64 | 256x256 | 642.453 ms | 1.57 | 0.00 | 16% | ok |
| elementwise | exp | f64 | n=8388608 | 156.889 ms | 0.05 | 0.86 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 144.485 ms | 0.06 | 0.93 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 175.782 ms | 0.05 | 0.76 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 92.492 ms | 0.09 | 1.45 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 2665.074 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.583 ms | 1.01 | 10.54 | 108% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.284 ms | 0.80 | 8.29 | 85% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 16.108 ms | 2.60 | 5.47 | 56% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.856 ms | 0.19 | 0.68 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 6.355 ms | 0.32 | 2.62 | 27% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 290.392 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 71.977 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 27.667 ms | 0.40 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 119.953 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 159.967 ms | 0.14 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 50.015 ms | 0.45 | 0.01 | 5% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 241.148 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.241 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.700 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.627 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.300 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 241.432 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 408.723 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 132.261 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 105.410 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 126.490 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 82.109 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 535.829 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 179.758 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 94.660 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 168.989 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 144.262 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 176.494 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 260.328 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.129 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.638 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 34.116 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.976 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.822 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.075 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.145 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.798 ms | 0.69 | 7.72 | 79% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.259 ms | 0.80 | 9.00 | 92% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.955 ms | 0.66 | 7.42 | 76% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 36.974 ms | 1.04 | 7.86 | 80% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 146.977 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 197.334 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 153.608 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 44.124 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 195.240 ms | 0.09 | 0.34 | 4% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 117.350 ms | 2.29 | 0.04 | 24% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.427 ms | 6.91 | 7.34 | 75% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.709 ms | 2.50 | 10.00 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.257 ms | 5.15 | 10.30 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.511 ms | 3.04 | 3.04 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 5.366 ms | 3.13 | 3.13 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 59.607 ms | 4.50 | 0.05 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 46.793 ms | 5.74 | 0.04 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 24.456 ms | 10.98 | 0.06 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 24.497 ms | 10.96 | 0.06 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 74.304 ms | 3.61 | 0.02 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

