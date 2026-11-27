## jetson-orin -- blas, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 64.3 GFLOP/s, f32 128.5 GFLOP/s |
| ... hardware, NEON | f64 64.3, f32 128.5 GFLOP/s |
| ... this build, scalar | f64 32.1, f32 32.1 GFLOP/s |
| bandwidth | read 27.2, update 33.9, triad 32.1 GB/s (ceiling 33.9) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 90% | 124% | 0 | 37% |
| blas2 | 10 | 69% | 124% | 0 | 43% |
| blas3 | 10 | 71% | 87% | 0 | 85% |
| factor | 12 | 13% | 24% | 0 | 26% |
| eigen | 8 | 7% | 19% | 0 | 20% |
| elementwise | 5 | 4% | 7% | 0 | 17% |
| sparse | 5 | 27% | 96% | 0 | 17% |
| direct | 23 | 1% | 6% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 15 | 1% | 13% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 23% | 30% | 0 | 17% |
| mg | 1 | 20% | 20% | 0 | 18% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 16% | 32% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 70% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.580 ms | 3.66 | 29.31 | 87% | ok | 2.5x (42%) |
| blas1 | dot | f32 | n=8388608 | 2.206 ms | 7.60 | 30.42 | 90% | ok | 2.7x (45%) |
| blas1 | axpy | f64 | n=8388608 | 5.561 ms | 3.02 | 36.20 | 107% | ok | 2.2x (37%) |
| blas1 | axpy | f32 | n=8388608 | 2.718 ms | 6.17 | 37.04 | 109% | ok | 2.2x (37%) |
| blas1 | scale | f64 | n=8388608 | 3.186 ms | 2.63 | 42.12 | 124% | ok | 2.0x (33%) |
| blas1 | two_norm | f64 | n=8388608 | 2.200 ms | 7.62 | 30.50 | 90% | ok | 2.6x (43%) |
| blas1 | one_norm | f64 | n=8388608 | 12.502 ms | 0.67 | 5.37 | 16% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.752 ms | 0.45 | 3.58 | 11% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.502 ms | 0.67 | 5.37 | 16% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 190.938 ms | 0.97 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 4.620 ms | 7.26 | 29.06 | 86% | ok | 2.7x (46%) |
| blas2 | gemv | f32 | n=4096 | 2.279 ms | 14.72 | 29.46 | 87% | ok | 2.6x (43%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 6.456 ms | 5.20 | 20.80 | 61% | ok | 2.0x (34%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.828 ms | 8.76 | 17.54 | 52% | ok | 1.5x (25%) |
| blas2 | symv | f64 | n=4096 | 5.238 ms | 6.41 | 12.82 | 38% | ok | 3.0x (51%) |
| blas2 | trmv | f64 | n=4096 | 2.842 ms | 5.90 | 23.64 | 70% | ok | 3.0x (50%) |
| blas2 | trmv | f32 | n=4096 | 1.435 ms | 11.69 | 23.41 | 69% | ok | 3.4x (57%) |
| blas2 | trsv | f64 | n=4096 | 8.476 ms | 1.98 | 7.93 | 23% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 4.746 ms | 3.53 | 7.08 | 21% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 6.401 ms | 5.24 | 41.95 | 124% | ok | 2.1x (35%) |
| blas3 | gemm | f64 | 512x512x512 | 4.826 ms | 55.62 | 1.30 | 87% | ok | 5.5x (92%) |
| blas3 | gemm | f32 | 512x512x512 | 2.474 ms | 108.52 | 1.27 | 84% | ok | 5.3x (88%) |
| blas3 | gemmt | f64 | 512x512x512 | 4.082 ms | 32.88 | 1.54 | 51% | ok | 4.1x (68%) |
| blas3 | symm | f64 | 512x512x512 | 4.778 ms | 56.18 | 1.32 | 87% | ok | 5.6x (93%) |
| blas3 | syrk | f64 | 512x512x512 | 3.620 ms | 37.08 | 1.16 | 58% | ok | 4.0x (67%) |
| blas3 | syr2k | f64 | 512x512x512 | 5.966 ms | 44.99 | 1.05 | 70% | ok | 4.7x (79%) |
| blas3 | trmm | f64 | 512x512x512 | 2.593 ms | 51.75 | 2.02 | 81% | ok | 5.3x (88%) |
| blas3 | trsm | f64 | 512x512x512 | 2.922 ms | 45.93 | 1.79 | 71% | ok | 5.1x (85%) |
| blas3 | trsm | f32 | 512x512x512 | 1.570 ms | 85.48 | 1.67 | 67% | ok | 4.8x (80%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.431 ms | 0.56 | 4.50 | 13% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 5.693 ms | 15.72 | 0.74 | 24% | ok | 2.4x (40%) |
| factor | lu | f32 | 512x512 | 4.600 ms | 19.45 | 0.46 | 15% | ok | 1.8x (30%) |
| factor | cholesky | f64 | 512x512 | 4.156 ms | 10.77 | 1.01 | 17% | ok | 1.9x (32%) |
| factor | cholesky | f32 | 512x512 | 3.225 ms | 13.87 | 0.65 | 11% | ok | 1.5x (26%) |
| factor | ldlt | f64 | 512x512 | 31.992 ms | 1.40 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 15.950 ms | 2.81 | 0.26 | 4% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 13.720 ms | 13.04 | 0.31 | 20% | ok | 2.1x (35%) |
| factor | lq | f64 | 512x512 | 59.345 ms | 3.02 | 0.07 | 5% | ok | 1.5x (26%) |
| factor | hessenberg | f64 | 512x512 | 317.461 ms | 1.41 | 0.01 | 2% | ok | 1.5x (25%) |
| factor | householder | f64 | 512x512 | 0.535 ms | 1.96 | 7.84 | 23% | ok | 3.9x (64%) |
| factor | inv | f64 | 512x512 | 163.252 ms | 1.64 | 0.03 | 3% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 10.930 ms | 8.19 | 0.38 | 13% | ok | 1.4x (23%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 7.076 ms | 3.16 | 0.15 | 5% | ok | 1.5x (25%) |
| eigen | eigen_symmetric | f64 | 256x256 | 144.142 ms | 1.05 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 938.183 ms | 0.16 | 0.00 | 0% | ok | 1.0x (16%) |
| eigen | eigenvalue | f64 | 256x256 | 56.554 ms | 2.97 | 0.02 | 5% | ok | 1.1x (19%) |
| eigen | schur | f64 | 256x256 | 74.261 ms | 5.65 | 0.01 | 9% | ok | 1.2x (20%) |
| eigen | svd | f64 | 256x256 | 29.690 ms | 12.43 | 0.04 | 19% | ok | 1.5x (24%) |
| eigen | expm | f64 | 256x256 | 36.302 ms | 6.93 | 0.03 | 11% | ok | 1.5x (26%) |
| eigen | sylvester | f64 | 256x256 | 210.190 ms | 4.79 | 0.01 | 7% | ok | 1.2x (20%) |
| elementwise | exp | f64 | n=8388608 | 75.654 ms | 0.11 | 1.77 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 108.025 ms | 0.08 | 1.24 | 4% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 177.688 ms | 0.05 | 0.76 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 55.138 ms | 0.15 | 2.43 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 126.345 ms | 0.07 | 1.06 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.838 ms | 3.12 | 32.50 | 96% | ok | 2.8x (46%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.022 ms | 0.87 | 9.01 | 27% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.714 ms | 8.88 | 18.68 | 55% | ok | 3.7x (61%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.422 ms | 0.16 | 0.56 | 2% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.193 ms | 0.18 | 1.49 | 4% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 53.805 ms | 0.21 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 88.416 ms | 0.13 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 36.003 ms | 0.31 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 135.088 ms | 0.17 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 153.820 ms | 0.15 | 0.00 | 0% | ok | 1.0x (16%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 60.707 ms | 0.37 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 106.012 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.116 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.005 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 181.427 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.894 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 66.443 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 7.767 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 43.378 ms | 0.26 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.124 ms | 0.36 | 1.42 | 4% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 16.380 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 16.821 ms | 0.67 | 0.04 | 1% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.092 ms | 0.36 | 1.45 | 4% | ok | 1.0x (17%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.612 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 116.521 ms | 0.19 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.584 ms | 0.10 | 0.40 | 1% | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 12.485 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 129.871 ms | 0.17 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 7.499 ms | 0.10 | 0.40 | 1% | ok | 1.0x (17%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 30.043 ms | 0.37 | 0.02 | 1% | ok | 0.9x (16%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 6.004 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 22.400 ms | 0.50 | 0.03 | 1% | ok | 0.9x (15%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.458 ms | 0.52 | 2.07 | 6% | ok | 0.9x (16%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 16.843 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 845.413 ms | 0.37 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 9.840 ms | 0.34 | 1.37 | 4% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 36.857 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 257.557 ms | 1.20 | 0.00 | 2% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 10.011 ms | 0.34 | 1.35 | 4% | ok | 1.0x (17%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.208 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1912.444 ms | 0.32 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.620 ms | 0.09 | 0.37 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 23.757 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1522.761 ms | 0.41 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 36.499 ms | 0.09 | 0.37 | 1% | ok | 1.0x (17%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 11.450 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 406.474 ms | 0.76 | 0.00 | 1% | ok | 0.2x (3%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 3.131 ms | 1.08 | 4.30 | 13% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.015 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 229.242 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 77.790 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.755 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.906 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.808 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 88.994 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.866 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.748 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 57.585 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.438 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 50.623 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 59.239 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.168 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.676 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 44.451 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.499 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.303 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.096 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 4.204 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.925 ms | 0.89 | 10.03 | 30% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.738 ms | 0.70 | 7.84 | 23% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.515 ms | 0.58 | 6.50 | 19% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 42.072 ms | 0.91 | 6.91 | 20% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 352.069 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 357.331 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 193.946 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 21.030 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 12.504 ms | 1.34 | 5.37 | 16% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.907 ms | 20.80 | 0.32 | 32% | ok | 5.8x (97%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 5.607 ms | 2.99 | 3.18 | 9% | ok | 0.8x (14%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.693 ms | 2.95 | 11.79 | n/a | ok | 1.0x (16%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.715 ms | 6.18 | 12.36 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.673 ms | 2.96 | 2.96 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.757 ms | 9.55 | 9.55 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 9.093 ms | 29.52 | 0.35 | n/a | ok | 4.2x (70%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 7.067 ms | 37.98 | 0.30 | n/a | ok | 5.8x (97%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.483 ms | 48.96 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.481 ms | 48.98 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 10.302 ms | 26.06 | 0.15 | n/a | ok | 5.8x (97%) |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

