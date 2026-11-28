## xeon-e5-2420v2 -- blas, 6 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 59.0 GFLOP/s, f32 118.1 GFLOP/s |
| ... hardware, SSE2 | f64 59.0, f32 118.1 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.7 GFLOP/s |
| bandwidth | read 11.4, update 10.6, triad 8.4 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 53% | 107% | 0 | 84% |
| blas2 | 10 | 103% | 108% | 0 | 20% |
| blas3 | 10 | 126% | 158% | 0 | 77% |
| factor | 12 | 18% | 40% | 0 | 25% |
| eigen | 8 | 9% | 23% | 0 | 17% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 69% | 106% | 0 | 17% |
| direct | 19 | 1% | 31% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 4% | 17% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 74% | 81% | 0 | 17% |
| mg | 1 | 70% | 70% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 16% |
| mixed | 3 | 22% | 68% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 85% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 33.511 ms | 0.50 | 4.01 | 35% | ok | 5.6x (93%) |
| blas1 | dot | f32 | n=8388608 | 5.513 ms | 3.04 | 12.17 | 107% | ok | 5.1x (84%) |
| blas1 | axpy | f64 | n=8388608 | 38.978 ms | 0.43 | 5.17 | 45% | ok | 5.5x (91%) |
| blas1 | axpy | f32 | n=8388608 | 16.765 ms | 1.00 | 6.00 | 53% | ok | 5.5x (91%) |
| blas1 | scale | f64 | n=8388608 | 12.100 ms | 0.69 | 11.09 | 98% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 33.078 ms | 0.51 | 2.03 | 18% | ok | 5.6x (94%) |
| blas1 | one_norm | f64 | n=8388608 | 9.783 ms | 0.86 | 6.86 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.026 ms | 0.64 | 5.15 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.704 ms | 0.86 | 6.92 | 61% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 279.009 ms | 0.66 | 0.48 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 11.290 ms | 2.97 | 11.89 | 105% | ok | 1.3x (21%) |
| blas2 | gemv | f32 | n=4096 | 5.462 ms | 6.14 | 12.29 | 108% | ok | 1.2x (19%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 10.931 ms | 3.07 | 12.28 | 108% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.691 ms | 5.90 | 11.80 | 104% | ok | 1.2x (20%) |
| blas2 | symv | f64 | n=4096 | 5.791 ms | 5.79 | 11.60 | 102% | ok | 1.2x (21%) |
| blas2 | trmv | f64 | n=4096 | 5.723 ms | 2.93 | 11.74 | 103% | ok | 1.3x (22%) |
| blas2 | trmv | f32 | n=4096 | 2.971 ms | 5.65 | 11.30 | 99% | ok | 1.3x (21%) |
| blas2 | trsv | f64 | n=4096 | 7.450 ms | 2.25 | 9.02 | 79% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 3.537 ms | 4.74 | 9.50 | 84% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 25.844 ms | 1.30 | 10.39 | 91% | ok | 1.1x (19%) |
| blas3 | gemm | f64 | 512x512x512 | 2.884 ms | 93.06 | 2.18 | 158% | ok | 4.9x (81%) |
| blas3 | gemm | f32 | 512x512x512 | 1.811 ms | 148.25 | 1.74 | 126% | ok | 5.0x (83%) |
| blas3 | gemmt | f64 | 512x512x512 | 2.580 ms | 52.03 | 2.44 | 88% | ok | 3.5x (58%) |
| blas3 | symm | f64 | 512x512x512 | 2.930 ms | 91.61 | 2.15 | 155% | ok | 4.8x (80%) |
| blas3 | syrk | f64 | 512x512x512 | 2.534 ms | 52.96 | 1.66 | 90% | ok | 3.1x (51%) |
| blas3 | syr2k | f64 | 512x512x512 | 3.718 ms | 72.19 | 1.69 | 122% | ok | 4.1x (68%) |
| blas3 | trmm | f64 | 512x512x512 | 1.623 ms | 82.71 | 3.23 | 140% | ok | 4.7x (78%) |
| blas3 | trsm | f64 | 512x512x512 | 1.767 ms | 75.96 | 2.97 | 129% | ok | 4.6x (77%) |
| blas3 | trsm | f32 | 512x512x512 | 1.255 ms | 106.96 | 2.09 | 91% | ok | 4.2x (70%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.690 ms | 0.14 | 1.10 | 10% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 5.615 ms | 15.93 | 0.75 | 27% | ok | 1.6x (27%) |
| factor | lu | f32 | 512x512 | 3.907 ms | 22.90 | 0.54 | 19% | ok | 1.6x (27%) |
| factor | cholesky | f64 | 512x512 | 4.152 ms | 10.78 | 1.01 | 18% | ok | 1.4x (23%) |
| factor | cholesky | f32 | 512x512 | 3.077 ms | 14.54 | 0.68 | 12% | ok | 1.3x (21%) |
| factor | ldlt | f64 | 512x512 | 25.070 ms | 1.78 | 0.17 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 13.367 ms | 3.35 | 0.31 | 6% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 13.017 ms | 13.75 | 0.32 | 23% | ok | 1.5x (25%) |
| factor | lq | f64 | 512x512 | 48.458 ms | 3.69 | 0.09 | 6% | ok | 1.5x (25%) |
| factor | hessenberg | f64 | 512x512 | 399.180 ms | 1.12 | 0.01 | 2% | ok | 1.6x (26%) |
| factor | householder | f64 | 512x512 | 0.918 ms | 1.14 | 4.57 | 40%\* | ok | 3.5x (59%) |
| factor | inv | f64 | 512x512 | 132.179 ms | 2.03 | 0.03 | 3% | ok | 1.0x (16%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 8.437 ms | 10.61 | 0.50 | 18% | ok | 1.5x (25%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 5.898 ms | 3.79 | 0.18 | 6% | ok | 1.1x (18%) |
| eigen | eigen_symmetric | f64 | 256x256 | 173.868 ms | 0.87 | 0.01 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 996.770 ms | 0.15 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 56.997 ms | 2.94 | 0.02 | 5% | ok | 0.9x (16%) |
| eigen | schur | f64 | 256x256 | 75.576 ms | 5.55 | 0.01 | 9% | ok | 1.0x (16%) |
| eigen | svd | f64 | 256x256 | 26.868 ms | 13.74 | 0.04 | 23% | ok | 1.1x (18%) |
| eigen | expm | f64 | 256x256 | 27.388 ms | 9.19 | 0.04 | 16% | ok | 1.3x (22%) |
| eigen | sylvester | f64 | 256x256 | 192.234 ms | 5.24 | 0.01 | 9% | ok | 1.0x (17%) |
| elementwise | exp | f64 | n=8388608 | 151.270 ms | 0.06 | 0.89 | 8% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 145.734 ms | 0.06 | 0.92 | 8% | ok | 1.0x (16%) |
| elementwise | sin | f64 | n=8388608 | 170.749 ms | 0.05 | 0.79 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 90.431 ms | 0.09 | 1.48 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 2469.917 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.268 ms | 1.15 | 12.01 | 106% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.120 ms | 0.84 | 8.73 | 77% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.304 ms | 3.70 | 7.79 | 69% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.125 ms | 0.20 | 0.71 | 6%\* | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.973 ms | 0.34 | 2.79 | 25% | ok | 1.0x (16%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 40.711 ms | 0.27 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 69.378 ms | 0.16 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 25.083 ms | 0.45 | 0.03 | 1% | ok | 1.0x (16%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 117.087 ms | 0.19 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 130.899 ms | 0.17 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 47.560 ms | 0.47 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 81.336 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.921 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.154 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 126.360 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.522 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.149 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.285 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.316 ms | 0.34 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.852 ms | 0.89 | 3.55 | 31%\* | ok | 0.9x (16%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.496 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.568 ms | 0.89 | 0.06 | 2% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.874 ms | 0.86 | 3.46 | 30%\* | ok | 0.9x (16%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.414 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 103.793 ms | 0.22 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.841 ms | 0.20 | 0.79 | 7%\* | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.829 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 113.763 ms | 0.20 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.797 ms | 0.20 | 0.80 | 7%\* | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 12.921 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 634.055 ms | 0.49 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 6.995 ms | 0.48 | 1.93 | 17%\* | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.670 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 203.009 ms | 1.52 | 0.00 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.025 ms | 0.48 | 1.92 | 17%\* | ok | 1.0x (16%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.424 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1696.103 ms | 0.36 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.920 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 17.920 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1497.284 ms | 0.41 | 0.00 | 1% | ok | 1.1x (18%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.473 ms | 0.10 | 0.42 | 4%\* | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 230.935 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 404.912 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 127.757 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 103.748 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 122.127 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 79.902 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 533.889 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 174.583 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.496 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 202.192 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 192.837 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 211.428 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 307.914 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.172 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.352 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 32.517 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 2.906 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | ildl | f64 | 128^2 grid | 2.737 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.092 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 3.533 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.482 ms | 0.75 | 8.42 | 74% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.179 ms | 0.82 | 9.22 | 81% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.801 ms | 0.69 | 7.71 | 68% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 36.545 ms | 1.05 | 7.95 | 70% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 140.890 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 147.607 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 130.809 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.542 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 187.515 ms | 0.09 | 0.36 | 3% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 20.487 ms | 13.10 | 0.20 | 22% | ok | 5.8x (97%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.308 ms | 7.27 | 7.73 | 68% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.510 ms | 2.58 | 10.31 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.137 ms | 5.35 | 10.70 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.985 ms | 3.37 | 3.37 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.977 ms | 3.37 | 3.37 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.565 ms | 23.21 | 0.27 | n/a | ok | 5.1x (85%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 8.229 ms | 32.62 | 0.25 | n/a | ok | 5.5x (92%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.314 ms | 62.22 | 0.36 | n/a | ok | 5.5x (91%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.312 ms | 62.25 | 0.36 | n/a | ok | 5.5x (92%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.003 ms | 20.64 | 0.12 | n/a | ok | 5.4x (90%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

