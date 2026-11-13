## i7-12700k -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `8867b44b3f52` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, AVX2 | f64 78.3, f32 156.6 GFLOP/s |
| bandwidth | read 37.8, update 41.5, triad 31.5 GB/s (ceiling 41.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 88% | 104% | 0 |
| blas2 | 10 | 94% | 97% | 0 |
| blas3 | 10 | 65% | 91% | 0 |
| factor | 12 | 8% | 24% | 0 |
| eigen | 8 | 7% | 27% | 0 |
| elementwise | 5 | 7% | 11% | 0 |
| sparse | 5 | 53% | 89% | 0 |
| direct | 19 | 1% | 29% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 12 | 4% | 23% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 54% | 72% | 0 |
| mg | 1 | 49% | 49% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 84% | 135% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 3.671 ms | 4.57 | 36.56 | 88% | ok |
| blas1 | dot | f32 | n=8388608 | 1.686 ms | 9.95 | 39.80 | 96% | ok |
| blas1 | axpy | f64 | n=8388608 | 4.756 ms | 3.53 | 42.33 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 2.342 ms | 7.16 | 42.98 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 3.454 ms | 2.43 | 38.85 | 94% | ok |
| blas1 | two_norm | f64 | n=8388608 | 2.186 ms | 7.68 | 30.70 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 3.986 ms | 2.10 | 16.83 | 41% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.029 ms | 1.19 | 9.55 | 23% | ok |
| blas1 | sum | f64 | n=8388608 | 3.935 ms | 2.13 | 17.06 | 41% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 70.596 ms | 2.61 | 1.90 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.393 ms | 9.89 | 39.58 | 95% | ok |
| blas2 | gemv | f32 | n=4096 | 1.689 ms | 19.87 | 39.76 | 96% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.451 ms | 9.72 | 38.91 | 94% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.675 ms | 20.03 | 40.08 | 97% | ok |
| blas2 | symv | f64 | n=4096 | 6.176 ms | 5.43 | 10.88 | 26% | ok |
| blas2 | trmv | f64 | n=4096 | 1.782 ms | 9.42 | 37.70 | 91% | ok |
| blas2 | trmv | f32 | n=4096 | 0.857 ms | 19.57 | 39.19 | 94% | ok |
| blas2 | trsv | f64 | n=4096 | 1.818 ms | 9.23 | 36.96 | 89% | ok |
| blas2 | trsv | f32 | n=4096 | 0.865 ms | 19.39 | 38.82 | 94% | ok |
| blas2 | ger | f64 | n=4096 | 7.128 ms | 4.71 | 37.67 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.795 ms | 70.74 | 1.66 | 90% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.892 ms | 141.89 | 1.66 | 91% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.628 ms | 51.07 | 2.39 | 65% | ok |
| blas3 | symm | f64 | 512x512x512 | 4.044 ms | 66.38 | 1.56 | 85% | ok |
| blas3 | syrk | f64 | 512x512x512 | 9.491 ms | 14.14 | 0.44 | 18% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 20.260 ms | 13.25 | 0.31 | 17% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.699 ms | 49.72 | 1.94 | 63% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.754 ms | 48.73 | 1.90 | 62% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.293 ms | 103.79 | 2.03 | 66% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.101 ms | 0.41 | 3.24 | 8% | ok |
| factor | lu | f64 | 512x512 | 10.840 ms | 8.25 | 0.39 | 11% | ok |
| factor | lu | f32 | 512x512 | 3.643 ms | 24.56 | 0.58 | 16% | ok |
| factor | cholesky | f64 | 512x512 | 7.107 ms | 6.30 | 0.59 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.578 ms | 6.80 | 0.32 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.174 ms | 6.24 | 0.58 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 46.374 ms | 0.96 | 0.09 | 1% | ok |
| factor | qr | f64 | 512x512 | 221.941 ms | 0.81 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 23.179 ms | 7.72 | 0.18 | 10% | ok |
| factor | hessenberg | f64 | 512x512 | 257.501 ms | 1.74 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.619 ms | 0.65 | 2.59 | 6%\* | ok |
| factor | inv | f64 | 512x512 | 55.052 ms | 4.88 | 0.08 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 4.798 ms | 18.65 | 0.87 | 24% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 17.083 ms | 1.31 | 0.06 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 57.581 ms | 2.62 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 752.584 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 30.872 ms | 5.43 | 0.03 | 7% | ok |
| eigen | schur | f64 | 256x256 | 80.182 ms | 5.23 | 0.01 | 7% | ok |
| eigen | svd | f64 | 256x256 | 103.577 ms | 3.56 | 0.01 | 5% | ok |
| eigen | expm | f64 | 256x256 | 11.709 ms | 21.49 | 0.09 | 27% | ok |
| eigen | sylvester | f64 | 256x256 | 176.365 ms | 5.71 | 0.01 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 39.343 ms | 0.21 | 3.41 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 46.811 ms | 0.18 | 2.87 | 7% | ok |
| elementwise | sin | f64 | n=8388608 | 59.864 ms | 0.14 | 2.24 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 28.846 ms | 0.29 | 4.65 | 11% | ok |
| elementwise | erf | f64 | n=8388608 | 201.097 ms | 0.04 | 0.67 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.735 ms | 3.56 | 37.03 | 89% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.924 ms | 2.83 | 29.46 | 71% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.034 ms | 10.38 | 21.83 | 53% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.824 ms | 0.68 | 2.38 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.250 ms | 1.61 | 13.31 | 32%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 72.975 ms | 0.15 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 27.441 ms | 0.41 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 9.035 ms | 1.24 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.502 ms | 0.57 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 55.178 ms | 0.41 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.903 ms | 1.32 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 99.941 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.501 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.028 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.225 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.157 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.343 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 70.124 ms | 0.16 | 0.01 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.253 ms | 2.98 | 11.93 | 29%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.491 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.284 ms | 2.61 | 0.17 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.272 ms | 2.78 | 11.12 | 27%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.134 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.378 ms | 0.63 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.642 ms | 1.18 | 4.71 | 11%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.921 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 49.278 ms | 0.45 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.639 ms | 1.18 | 4.74 | 11%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 9.594 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 5323.295 ms | 0.06 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.440 ms | 2.34 | 9.36 | 23%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 18.739 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 67.439 ms | 4.58 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.657 ms | 2.03 | 8.14 | 20%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.936 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 525.953 ms | 1.17 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.494 ms | 0.45 | 1.80 | 4%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.668 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 657.303 ms | 0.94 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.499 ms | 0.45 | 1.80 | 4%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.369 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 28.621 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 20.869 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.474 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.721 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.613 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 20.247 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.835 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.494 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.625 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 14.926 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 15.984 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 18.205 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.034 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.575 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.019 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.097 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.029 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.023 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.733 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.976 ms | 2.68 | 30.03 | 72% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.308 ms | 2.00 | 22.42 | 54% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.509 ms | 1.73 | 19.44 | 47% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.294 ms | 2.69 | 20.33 | 49% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 28.513 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 39.547 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 34.610 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.589 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 1.928 ms | 8.70 | 34.80 | 84% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 5.996 ms | 44.77 | 0.70 | 57% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.318 ms | 52.68 | 55.97 | 135%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.909 ms | 8.79 | 35.15 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.759 ms | 22.09 | 44.19 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.291 ms | 57.66 | 57.66 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.279 ms | 60.10 | 60.10 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 3.980 ms | 67.45 | 0.79 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.621 ms | 47.75 | 0.37 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.699 ms | 47.11 | 0.28 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 2.126 ms | 126.27 | 0.74 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 1.908 ms | 140.65 | 0.82 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

