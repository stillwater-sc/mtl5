## jetson-orin -- blas, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 64.2 GFLOP/s, f32 128.4 GFLOP/s |
| ... hardware, NEON | f64 64.2, f32 128.4 GFLOP/s |
| ... this build, scalar | f64 32.1, f32 32.1 GFLOP/s |
| bandwidth | read 27.5, update 34.0, triad 32.3 GB/s (ceiling 34.0) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 89% | 126% | 0 | 37% |
| blas2 | 10 | 75% | 123% | 0 | 43% |
| blas3 | 10 | 72% | 87% | 0 | 85% |
| factor | 12 | 13% | 24% | 0 | 26% |
| eigen | 8 | 7% | 19% | 0 | 20% |
| elementwise | 5 | 4% | 7% | 0 | 17% |
| sparse | 5 | 27% | 96% | 0 | 17% |
| direct | 23 | 0% | 6% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 15 | 1% | 13% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 23% | 30% | 0 | 17% |
| mg | 1 | 19% | 19% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 16% | 32% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 70% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.564 ms | 3.68 | 29.40 | 86% | ok | 2.5x (42%) |
| blas1 | dot | f32 | n=8388608 | 2.210 ms | 7.59 | 30.36 | 89% | ok | 3.0x (50%) |
| blas1 | axpy | f64 | n=8388608 | 5.504 ms | 3.05 | 36.58 | 107% | ok | 2.2x (37%) |
| blas1 | axpy | f32 | n=8388608 | 2.698 ms | 6.22 | 37.31 | 110% | ok | 2.2x (37%) |
| blas1 | scale | f64 | n=8388608 | 3.122 ms | 2.69 | 42.99 | 126% | ok | 2.0x (34%) |
| blas1 | two_norm | f64 | n=8388608 | 2.177 ms | 7.71 | 30.83 | 91% | ok | 2.6x (43%) |
| blas1 | one_norm | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 16% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.753 ms | 0.45 | 3.58 | 11% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.502 ms | 0.67 | 5.37 | 16% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 192.057 ms | 0.96 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 4.627 ms | 7.25 | 29.02 | 85% | ok | 2.7x (45%) |
| blas2 | gemv | f32 | n=4096 | 2.276 ms | 14.74 | 29.50 | 87% | ok | 2.6x (43%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 6.447 ms | 5.20 | 20.83 | 61% | ok | 2.0x (34%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.744 ms | 8.96 | 17.93 | 53% | ok | 1.5x (26%) |
| blas2 | symv | f64 | n=4096 | 5.106 ms | 6.57 | 13.16 | 39% | ok | 3.1x (52%) |
| blas2 | trmv | f64 | n=4096 | 2.622 ms | 6.40 | 25.62 | 75% | ok | 3.3x (55%) |
| blas2 | trmv | f32 | n=4096 | 1.310 ms | 12.81 | 25.64 | 75% | ok | 3.7x (61%) |
| blas2 | trsv | f64 | n=4096 | 8.584 ms | 1.95 | 7.83 | 23% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 4.836 ms | 3.47 | 6.94 | 20% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 6.402 ms | 5.24 | 41.94 | 123% | ok | 2.1x (35%) |
| blas3 | gemm | f64 | 512x512x512 | 4.824 ms | 55.65 | 1.30 | 87% | ok | 5.5x (92%) |
| blas3 | gemm | f32 | 512x512x512 | 2.481 ms | 108.20 | 1.27 | 84% | ok | 5.3x (88%) |
| blas3 | gemmt | f64 | 512x512x512 | 4.091 ms | 32.81 | 1.54 | 51% | ok | 4.1x (68%) |
| blas3 | symm | f64 | 512x512x512 | 4.805 ms | 55.87 | 1.31 | 87% | ok | 5.6x (93%) |
| blas3 | syrk | f64 | 512x512x512 | 3.568 ms | 37.62 | 1.18 | 59% | ok | 4.1x (68%) |
| blas3 | syr2k | f64 | 512x512x512 | 5.977 ms | 44.91 | 1.05 | 70% | ok | 4.7x (78%) |
| blas3 | trmm | f64 | 512x512x512 | 2.586 ms | 51.89 | 2.03 | 81% | ok | 5.3x (88%) |
| blas3 | trsm | f64 | 512x512x512 | 2.912 ms | 46.09 | 1.80 | 72% | ok | 5.1x (85%) |
| blas3 | trsm | f32 | 512x512x512 | 1.582 ms | 84.86 | 1.66 | 66% | ok | 4.8x (80%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.522 ms | 0.56 | 4.46 | 13% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 5.743 ms | 15.58 | 0.73 | 24% | ok | 2.4x (39%) |
| factor | lu | f32 | 512x512 | 4.583 ms | 19.53 | 0.46 | 15% | ok | 1.9x (31%) |
| factor | cholesky | f64 | 512x512 | 4.178 ms | 10.71 | 1.00 | 17% | ok | 1.9x (31%) |
| factor | cholesky | f32 | 512x512 | 3.215 ms | 13.92 | 0.65 | 11% | ok | 1.6x (26%) |
| factor | ldlt | f64 | 512x512 | 32.476 ms | 1.38 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 16.209 ms | 2.76 | 0.26 | 4% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 13.765 ms | 13.00 | 0.30 | 20% | ok | 2.1x (34%) |
| factor | lq | f64 | 512x512 | 59.498 ms | 3.01 | 0.07 | 5% | ok | 1.5x (26%) |
| factor | hessenberg | f64 | 512x512 | 318.923 ms | 1.40 | 0.01 | 2% | ok | 1.5x (24%) |
| factor | householder | f64 | 512x512 | 0.544 ms | 1.93 | 7.71 | 23% | ok | 3.8x (64%) |
| factor | inv | f64 | 512x512 | 165.864 ms | 1.62 | 0.03 | 3% | ok | 1.0x (17%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 10.993 ms | 8.14 | 0.38 | 13% | ok | 1.4x (23%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 7.101 ms | 3.15 | 0.15 | 5% | ok | 1.5x (25%) |
| eigen | eigen_symmetric | f64 | 256x256 | 144.871 ms | 1.04 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 958.930 ms | 0.16 | 0.00 | 0% | ok | 0.9x (16%) |
| eigen | eigenvalue | f64 | 256x256 | 56.514 ms | 2.97 | 0.02 | 5% | ok | 1.1x (19%) |
| eigen | schur | f64 | 256x256 | 74.545 ms | 5.63 | 0.01 | 9% | ok | 1.2x (20%) |
| eigen | svd | f64 | 256x256 | 29.784 ms | 12.39 | 0.04 | 19% | ok | 1.5x (24%) |
| eigen | expm | f64 | 256x256 | 36.923 ms | 6.82 | 0.03 | 11% | ok | 1.5x (26%) |
| eigen | sylvester | f64 | 256x256 | 211.317 ms | 4.76 | 0.01 | 7% | ok | 1.2x (20%) |
| elementwise | exp | f64 | n=8388608 | 75.609 ms | 0.11 | 1.78 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 108.265 ms | 0.08 | 1.24 | 4% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 177.802 ms | 0.05 | 0.75 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 55.089 ms | 0.15 | 2.44 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 126.282 ms | 0.07 | 1.06 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.837 ms | 3.13 | 32.52 | 96% | ok | 2.8x (46%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.005 ms | 0.87 | 9.06 | 27% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.693 ms | 8.92 | 18.76 | 55% | ok | 3.7x (62%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.080 ms | 0.16 | 0.57 | 2% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.719 ms | 0.17 | 1.42 | 4% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 54.810 ms | 0.20 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 100.453 ms | 0.11 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 36.097 ms | 0.31 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 132.663 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 152.294 ms | 0.15 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 60.739 ms | 0.37 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 105.208 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 11.994 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 16.794 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 181.026 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.900 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 66.230 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 7.923 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 44.181 ms | 0.25 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.180 ms | 0.35 | 1.39 | 4% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 16.569 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.845 ms | 0.66 | 0.04 | 1% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.121 ms | 0.36 | 1.43 | 4% | ok | 1.0x (17%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.601 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 114.952 ms | 0.19 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.889 ms | 0.10 | 0.38 | 1% | ok | 1.0x (16%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 12.257 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 128.090 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.816 ms | 0.10 | 0.39 | 1% | ok | 1.0x (17%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 97.701 ms | 0.11 | 0.01 | 0% | ok | 0.3x (5%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 5.948 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 89.955 ms | 0.12 | 0.01 | 0% | ok | 0.2x (4%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.445 ms | 0.52 | 2.09 | 6% | ok | 0.9x (16%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 17.036 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 826.831 ms | 0.37 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 10.322 ms | 0.33 | 1.31 | 4% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 37.042 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 257.233 ms | 1.20 | 0.00 | 2% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.255 ms | 0.33 | 1.31 | 4% | ok | 1.0x (17%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.104 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1893.960 ms | 0.33 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 38.046 ms | 0.09 | 0.35 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 23.597 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1514.163 ms | 0.41 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 37.779 ms | 0.09 | 0.36 | 1% | ok | 1.0x (17%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 11.392 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 420.349 ms | 0.73 | 0.00 | 1% | ok | 0.2x (3%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 3.058 ms | 1.10 | 4.41 | 13% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.188 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 167.502 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 78.278 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.902 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 51.273 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.910 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.008 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.004 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 50.035 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 57.646 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.547 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 50.746 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 59.646 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.170 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.892 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 44.525 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.520 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.410 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.111 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 4.197 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.872 ms | 0.91 | 10.21 | 30% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.759 ms | 0.70 | 7.80 | 23% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.545 ms | 0.58 | 6.45 | 19% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 44.122 ms | 0.87 | 6.59 | 19% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 352.437 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 357.512 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 194.909 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 21.103 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 12.501 ms | 1.34 | 5.37 | 16% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.906 ms | 20.80 | 0.32 | 32% | ok | 5.8x (97%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 4.564 ms | 3.68 | 3.91 | 11% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.616 ms | 2.99 | 11.95 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.700 ms | 6.21 | 12.43 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.665 ms | 2.96 | 2.96 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.758 ms | 9.54 | 9.54 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 9.090 ms | 29.53 | 0.35 | n/a | ok | 4.2x (70%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 7.073 ms | 37.95 | 0.30 | n/a | ok | 5.8x (97%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.478 ms | 49.00 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.478 ms | 49.00 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 10.309 ms | 26.04 | 0.15 | n/a | ok | 5.8x (97%) |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

