## xeon-e5-2420v2 -- native, 6 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 58.9 GFLOP/s, f32 117.9 GFLOP/s |
| ... hardware, SSE2 | f64 58.9, f32 117.9 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.7 GFLOP/s |
| bandwidth | read 11.1, update 10.4, triad 8.3 GB/s (ceiling 11.1) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 54% | 110% | 0 | 77% |
| blas2 | 10 | 89% | 111% | 0 | 17% |
| blas3 | 10 | 7% | 40% | 0 | 17% |
| factor | 12 | 6% | 41% | 0 | 23% |
| eigen | 8 | 3% | 11% | 0 | 17% |
| elementwise | 5 | 8% | 14% | 0 | 17% |
| sparse | 5 | 70% | 107% | 0 | 17% |
| direct | 19 | 1% | 32% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 4% | 17% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 76% | 85% | 0 | 17% |
| mg | 1 | 74% | 74% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 22% | 70% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 54% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 39.859 ms | 0.42 | 3.37 | 30% | ok | 4.6x (77%) |
| blas1 | dot | f32 | n=8388608 | 5.497 ms | 3.05 | 12.21 | 110% | ok | 4.9x (82%) |
| blas1 | axpy | f64 | n=8388608 | 38.135 ms | 0.44 | 5.28 | 48% | ok | 5.6x (93%) |
| blas1 | axpy | f32 | n=8388608 | 16.755 ms | 1.00 | 6.01 | 54% | ok | 5.4x (91%) |
| blas1 | scale | f64 | n=8388608 | 12.151 ms | 0.69 | 11.05 | 100% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 26.063 ms | 0.64 | 2.57 | 23% | ok | 5.6x (94%) |
| blas1 | one_norm | f64 | n=8388608 | 9.733 ms | 0.86 | 6.89 | 62% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 12.973 ms | 0.65 | 5.17 | 47% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.824 ms | 0.85 | 6.83 | 62% | ok | 1.0x (16%) |
| blas1 | dot_k3 | f64 | n=8388608 | 274.360 ms | 0.67 | 0.49 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 10.933 ms | 3.07 | 12.28 | 111% | ok | 1.3x (21%) |
| blas2 | gemv | f32 | n=4096 | 5.481 ms | 6.12 | 12.25 | 111% | ok | 1.5x (25%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 10.920 ms | 3.07 | 12.30 | 111% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.561 ms | 6.03 | 12.07 | 109% | ok | 1.2x (19%) |
| blas2 | symv | f64 | n=4096 | 12.258 ms | 2.74 | 5.48 | 50% | ok | 1.0x (17%) |
| blas2 | trmv | f64 | n=4096 | 7.055 ms | 2.38 | 9.52 | 86% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 4.471 ms | 3.75 | 7.51 | 68% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 6.858 ms | 2.45 | 9.79 | 89% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 4.327 ms | 3.88 | 7.76 | 70% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 27.949 ms | 1.20 | 9.61 | 87% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 11.520 ms | 23.30 | 0.55 | 40% | ok | 5.6x (93%) |
| blas3 | gemm | f32 | 512x512x512 | 6.237 ms | 43.04 | 0.50 | 37% | ok | 5.7x (96%) |
| blas3 | gemmt | f64 | 512x512x512 | 36.087 ms | 3.72 | 0.17 | 6% | ok | 1.0x (17%) |
| blas3 | symm | f64 | 512x512x512 | 67.906 ms | 3.95 | 0.09 | 7% | ok | 1.0x (17%) |
| blas3 | syrk | f64 | 512x512x512 | 37.716 ms | 3.56 | 0.11 | 6% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 512x512x512 | 65.518 ms | 4.10 | 0.10 | 7% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 512x512x512 | 33.322 ms | 4.03 | 0.16 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f64 | 512x512x512 | 33.768 ms | 3.97 | 0.16 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f32 | 512x512x512 | 16.409 ms | 8.18 | 0.16 | 7% | ok | 1.0x (17%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.298 ms | 0.14 | 1.11 | 10% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 21.254 ms | 4.21 | 0.20 | 7% | ok | 1.5x (24%) |
| factor | lu | f32 | 512x512 | 11.917 ms | 7.51 | 0.18 | 6% | ok | 1.4x (23%) |
| factor | cholesky | f64 | 512x512 | 24.823 ms | 1.80 | 0.17 | 3% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 23.690 ms | 1.89 | 0.09 | 2% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 25.044 ms | 1.79 | 0.17 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 13.673 ms | 3.27 | 0.31 | 6% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 318.415 ms | 0.56 | 0.01 | 1% | ok | 1.6x (27%) |
| factor | lq | f64 | 512x512 | 48.266 ms | 3.71 | 0.09 | 6% | ok | 1.5x (26%) |
| factor | hessenberg | f64 | 512x512 | 398.316 ms | 1.12 | 0.01 | 2% | ok | 1.6x (26%) |
| factor | householder | f64 | 512x512 | 0.922 ms | 1.14 | 4.55 | 41%\* | ok | 3.5x (58%) |
| factor | inv | f64 | 512x512 | 142.223 ms | 1.89 | 0.03 | 3% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 18.160 ms | 4.93 | 0.23 | 8% | ok | 1.2x (20%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 63.803 ms | 0.35 | 0.02 | 1% | ok | 0.9x (16%) |
| eigen | eigen_symmetric | f64 | 256x256 | 172.214 ms | 0.88 | 0.01 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1001.239 ms | 0.15 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 108.059 ms | 1.55 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 270.657 ms | 1.55 | 0.00 | 3% | ok | 1.0x (16%) |
| eigen | svd | f64 | 256x256 | 301.002 ms | 1.23 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 37.299 ms | 6.75 | 0.03 | 11% | ok | 2.3x (38%) |
| eigen | sylvester | f64 | 256x256 | 582.573 ms | 1.73 | 0.00 | 3% | ok | 1.1x (18%) |
| elementwise | exp | f64 | n=8388608 | 150.768 ms | 0.06 | 0.89 | 8% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 139.494 ms | 0.06 | 0.96 | 9% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 172.038 ms | 0.05 | 0.78 | 7% | ok | 1.0x (16%) |
| elementwise | sqrt | f64 | n=8388608 | 89.459 ms | 0.09 | 1.50 | 14% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 2477.686 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.293 ms | 1.14 | 11.87 | 107% | ok | 1.0x (17%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.125 ms | 0.84 | 8.71 | 79% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.327 ms | 3.70 | 7.77 | 70% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 17.020 ms | 0.19 | 0.68 | 6%\* | ok | 1.0x (16%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.870 ms | 0.34 | 2.84 | 26% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 40.767 ms | 0.27 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 69.054 ms | 0.16 | 0.01 | 0% | ok | 1.1x (18%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.982 ms | 0.45 | 0.03 | 1% | ok | 1.1x (18%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 115.092 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 132.744 ms | 0.17 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 57.658 ms | 0.39 | 0.01 | 1% | ok | 0.8x (14%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 82.419 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 19.937 ms | 0.00 | 0.00 | n/a | ok | 0.4x (7%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 25.110 ms | 0.00 | 0.00 | n/a | ok | 0.5x (8%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 125.893 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.527 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.972 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.289 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.255 ms | 0.34 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.849 ms | 0.89 | 3.56 | 32%\* | ok | 0.9x (16%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.459 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.546 ms | 0.89 | 0.06 | 2% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.872 ms | 0.87 | 3.47 | 31%\* | ok | 1.0x (16%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.326 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 103.771 ms | 0.22 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.780 ms | 0.20 | 0.80 | 7%\* | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.745 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 115.383 ms | 0.19 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.784 ms | 0.20 | 0.80 | 7%\* | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 12.963 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 634.986 ms | 0.49 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 6.981 ms | 0.48 | 1.93 | 17%\* | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.617 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 207.450 ms | 1.49 | 0.00 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.015 ms | 0.48 | 1.92 | 17%\* | ok | 1.0x (16%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.851 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1711.261 ms | 0.36 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.861 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 18.099 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1530.013 ms | 0.40 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.749 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 221.985 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 571.639 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 123.333 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 99.087 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 117.861 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 77.193 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 526.995 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 162.017 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 86.430 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 179.229 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 174.037 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 195.881 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 291.203 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.171 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.410 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 33.474 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 2.893 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 2.715 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.079 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.155 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.480 ms | 0.75 | 8.43 | 76% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.105 ms | 0.84 | 9.45 | 85% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.788 ms | 0.69 | 7.74 | 70% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.407 ms | 1.09 | 8.21 | 74% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 141.642 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 148.011 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 128.089 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 37.020 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 189.515 ms | 0.09 | 0.35 | 3% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 20.481 ms | 13.11 | 0.20 | 22% | ok | 5.5x (92%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.318 ms | 7.24 | 7.69 | 70% | ok | 1.0x (16%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.461 ms | 2.60 | 10.39 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.125 ms | 5.37 | 10.74 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.979 ms | 3.37 | 3.37 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.997 ms | 3.36 | 3.36 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.555 ms | 23.23 | 0.27 | n/a | ok | 5.0x (83%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 8.225 ms | 32.64 | 0.25 | n/a | ok | 5.5x (92%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 7.382 ms | 36.37 | 0.21 | n/a | ok | 3.2x (54%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.322 ms | 62.11 | 0.36 | n/a | ok | 5.5x (91%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.060 ms | 20.55 | 0.12 | n/a | ok | 5.4x (90%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

