## jetson-orin -- native-fast, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 64.2 GFLOP/s, f32 128.6 GFLOP/s |
| ... hardware, NEON | f64 64.0, f32 128.6 GFLOP/s |
| ... this build, NEON | f64 64.2, f32 128.4 GFLOP/s |
| bandwidth | read 27.8, update 34.7, triad 33.0 GB/s (ceiling 34.7) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 87% | 136% | 0 | 37% |
| blas2 | 10 | 50% | 85% | 0 | 19% |
| blas3 | 10 | 30% | 51% | 0 | 59% |
| factor | 12 | 4% | 23% | 0 | 24% |
| eigen | 8 | 3% | 9% | 0 | 17% |
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
| mixed | 3 | 34% | 39% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 80% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.578 ms | 3.66 | 29.31 | 85% | ok | 2.5x (42%) |
| blas1 | dot | f32 | n=8388608 | 2.224 ms | 7.54 | 30.17 | 87% | ok | 2.5x (42%) |
| blas1 | axpy | f64 | n=8388608 | 5.523 ms | 3.04 | 36.45 | 105% | ok | 2.2x (37%) |
| blas1 | axpy | f32 | n=8388608 | 2.464 ms | 6.81 | 40.86 | 118% | ok | 2.4x (41%) |
| blas1 | scale | f64 | n=8388608 | 2.846 ms | 2.95 | 47.16 | 136% | ok | 2.2x (37%) |
| blas1 | two_norm | f64 | n=8388608 | 2.181 ms | 7.69 | 30.77 | 89% | ok | 2.6x (43%) |
| blas1 | one_norm | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 15% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.750 ms | 0.45 | 3.58 | 10% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 15% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 191.819 ms | 0.96 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 5.202 ms | 6.45 | 25.81 | 74% | ok | 2.5x (42%) |
| blas2 | gemv | f32 | n=4096 | 2.290 ms | 14.65 | 29.32 | 85% | ok | 2.7x (46%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.765 ms | 4.32 | 17.29 | 50% | ok | 1.7x (28%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.566 ms | 9.41 | 18.83 | 54% | ok | 1.7x (29%) |
| blas2 | symv | f64 | n=4096 | 23.305 ms | 1.44 | 2.88 | 8% | ok | 1.1x (19%) |
| blas2 | trmv | f64 | n=4096 | 9.138 ms | 1.84 | 7.35 | 21% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 4.281 ms | 3.92 | 7.85 | 23% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 9.091 ms | 1.85 | 7.39 | 21% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 4.275 ms | 3.92 | 7.86 | 23% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 13.389 ms | 2.51 | 20.05 | 58% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 8.152 ms | 32.93 | 0.77 | 51% | ok | 5.1x (84%) |
| blas3 | gemm | f32 | 512x512x512 | 4.084 ms | 65.73 | 0.77 | 51% | ok | 5.1x (85%) |
| blas3 | gemmt | f64 | 512x512x512 | 7.415 ms | 18.10 | 0.85 | 28% | ok | 3.5x (58%) |
| blas3 | symm | f64 | 512x512x512 | 9.131 ms | 29.40 | 0.69 | 46% | ok | 4.6x (77%) |
| blas3 | syrk | f64 | 512x512x512 | 6.989 ms | 19.20 | 0.60 | 30% | ok | 3.5x (59%) |
| blas3 | syr2k | f64 | 512x512x512 | 13.962 ms | 19.23 | 0.45 | 30% | ok | 3.5x (59%) |
| blas3 | trmm | f64 | 512x512x512 | 10.844 ms | 12.38 | 0.48 | 19% | ok | 2.3x (38%) |
| blas3 | trsm | f64 | 512x512x512 | 16.130 ms | 8.32 | 0.33 | 13% | ok | 1.6x (26%) |
| blas3 | trsm | f32 | 512x512x512 | 5.553 ms | 24.17 | 0.47 | 19% | ok | 2.2x (37%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.922 ms | 0.53 | 4.28 | 12% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 24.180 ms | 3.70 | 0.17 | 6% | ok | 1.5x (24%) |
| factor | lu | f32 | 512x512 | 14.574 ms | 6.14 | 0.14 | 5% | ok | 1.4x (24%) |
| factor | cholesky | f64 | 512x512 | 33.831 ms | 1.32 | 0.12 | 2% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 30.104 ms | 1.49 | 0.07 | 1% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 34.469 ms | 1.30 | 0.12 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 15.962 ms | 2.80 | 0.26 | 4% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 206.783 ms | 0.87 | 0.02 | 1% | ok | 1.7x (28%) |
| factor | lq | f64 | 512x512 | 59.544 ms | 3.01 | 0.07 | 5% | ok | 1.6x (26%) |
| factor | hessenberg | f64 | 512x512 | 319.578 ms | 1.40 | 0.01 | 2% | ok | 1.5x (26%) |
| factor | householder | f64 | 512x512 | 0.530 ms | 1.98 | 7.91 | 23% | ok | 3.8x (63%) |
| factor | inv | f64 | 512x512 | 180.948 ms | 1.48 | 0.02 | 2% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 19.624 ms | 4.56 | 0.21 | 7% | ok | 1.3x (21%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.423 ms | 0.36 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 145.165 ms | 1.04 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 897.214 ms | 0.17 | 0.00 | 0% | ok | 1.1x (18%) |
| eigen | eigenvalue | f64 | 256x256 | 99.001 ms | 1.69 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 214.213 ms | 1.96 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 358.365 ms | 1.03 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 43.006 ms | 5.85 | 0.02 | 9% | ok | 1.7x (28%) |
| eigen | sylvester | f64 | 256x256 | 504.839 ms | 1.99 | 0.00 | 3% | ok | 1.0x (17%) |
| elementwise | exp | f64 | n=8388608 | 75.883 ms | 0.11 | 1.77 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 108.081 ms | 0.08 | 1.24 | 4% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 178.084 ms | 0.05 | 0.75 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 55.156 ms | 0.15 | 2.43 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 127.127 ms | 0.07 | 1.06 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.861 ms | 3.04 | 31.63 | 91% | ok | 2.7x (45%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.011 ms | 0.87 | 9.04 | 26% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 5.417 ms | 7.73 | 16.25 | 47% | ok | 3.2x (53%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.062 ms | 0.16 | 0.57 | 2% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.512 ms | 0.17 | 1.45 | 4% | ok | 1.1x (18%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 53.139 ms | 0.21 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 97.796 ms | 0.11 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 35.987 ms | 0.31 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 136.596 ms | 0.16 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 157.511 ms | 0.14 | 0.00 | 0% | ok | 1.0x (16%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 60.521 ms | 0.37 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 112.405 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.118 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 16.961 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 180.388 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.897 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 65.557 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 7.747 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 43.054 ms | 0.26 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.109 ms | 0.36 | 1.43 | 4% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 16.384 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.799 ms | 0.67 | 0.04 | 1% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.111 ms | 0.36 | 1.43 | 4% | ok | 1.0x (17%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.558 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 116.121 ms | 0.19 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.761 ms | 0.10 | 0.39 | 1% | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 12.218 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 132.026 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.679 ms | 0.10 | 0.39 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 16.920 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 813.474 ms | 0.38 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 10.465 ms | 0.32 | 1.29 | 4% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 36.986 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 258.000 ms | 1.20 | 0.00 | 2% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.298 ms | 0.33 | 1.31 | 4% | ok | 1.0x (17%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.326 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1892.008 ms | 0.33 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 37.978 ms | 0.09 | 0.35 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 23.558 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1522.588 ms | 0.41 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 37.962 ms | 0.09 | 0.35 | 1% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.192 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 229.104 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.150 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.685 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 51.053 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.835 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 88.393 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.901 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.818 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 57.566 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.338 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 50.543 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 59.180 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.168 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.905 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 44.875 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.522 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.337 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.097 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.401 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.886 ms | 0.91 | 10.16 | 29% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.739 ms | 0.70 | 7.84 | 23% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.519 ms | 0.58 | 6.49 | 19% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 42.058 ms | 0.91 | 6.91 | 20% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 360.685 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 364.373 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 188.474 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.848 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 6.269 ms | 2.68 | 10.71 | 31% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.297 ms | 21.83 | 0.34 | 34% | ok | 5.2x (87%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.319 ms | 12.72 | 13.51 | 39% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.648 ms | 2.97 | 11.88 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.722 ms | 6.16 | 12.33 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.243 ms | 13.50 | 13.50 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.350 ms | 7.14 | 7.14 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 5.137 ms | 52.26 | 0.61 | n/a | ok | 5.1x (84%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.809 ms | 46.21 | 0.36 | n/a | ok | 4.8x (80%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 6.628 ms | 40.50 | 0.24 | n/a | ok | 5.0x (84%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.761 ms | 152.47 | 0.89 | n/a | ok | 4.9x (81%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.535 ms | 59.20 | 0.35 | n/a | ok | 5.2x (86%) |

