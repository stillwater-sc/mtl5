## arduino-ventuno-q -- blas, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bf315e4604cf` |
| FMA peak (ceiling) | f64 67.6 GFLOP/s, f32 135.0 GFLOP/s |
| ... hardware, NEON | f64 67.6, f32 135.0 GFLOP/s |
| ... this build, scalar | f64 33.8, f32 33.8 GFLOP/s |
| bandwidth | read 22.8, update 24.9, triad 22.5 GB/s (ceiling 24.9) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 97% | 114% | 0 | 28% |
| blas2 | 10 | 63% | 105% | 0 | 36% |
| blas3 | 10 | 74% | 81% | 0 | 89% |
| factor | 12 | 18% | 36% | 0 | 44% |
| eigen | 8 | 11% | 28% | 0 | 33% |
| elementwise | 5 | 6% | 9% | 0 | 28% |
| sparse | 5 | 55% | 91% | 0 | 27% |
| direct | 23 | 1% | 11% | 0 | 27% |
| ordering | 4 | n/a | n/a | 0 | 27% |
| direct3d | 15 | 2% | 24% | 0 | 27% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 44% | 59% | 0 | 28% |
| mg | 1 | 36% | 36% | 0 | 24% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 31% | 37% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 53% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.560 ms | 3.02 | 24.14 | 97% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.808 ms | 5.97 | 23.90 | 96% | ok | 1.6x (40%) |
| blas1 | axpy | f64 | n=8388608 | 8.054 ms | 2.08 | 25.00 | 100% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 3.932 ms | 4.27 | 25.60 | 103% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 4.703 ms | 1.78 | 28.54 | 114% | ok | 1.2x (29%) |
| blas1 | two_norm | f64 | n=8388608 | 2.618 ms | 6.41 | 25.63 | 103% | ok | 1.2x (30%) |
| blas1 | one_norm | f64 | n=8388608 | 7.294 ms | 1.15 | 9.20 | 37% | ok | 1.1x (28%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.472 ms | 0.54 | 4.34 | 17% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.592 ms | 1.10 | 8.84 | 35% | ok | 1.1x (27%) |
| blas1 | dot_k3 | f64 | n=8388608 | 171.935 ms | 1.07 | 0.78 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 5.106 ms | 6.57 | 26.30 | 105% | ok | 1.8x (45%) |
| blas2 | gemv | f32 | n=4096 | 2.600 ms | 12.91 | 25.83 | 104% | ok | 1.5x (36%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 8.796 ms | 3.81 | 15.27 | 61% | ok | 1.1x (27%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.545 ms | 7.38 | 14.77 | 59% | ok | 0.8x (21%) |
| blas2 | symv | f64 | n=4096 | 6.368 ms | 5.27 | 10.55 | 42% | ok | 1.8x (46%) |
| blas2 | trmv | f64 | n=4096 | 3.638 ms | 4.61 | 18.47 | 74% | ok | 1.7x (43%) |
| blas2 | trmv | f32 | n=4096 | 2.153 ms | 7.79 | 15.60 | 63% | ok | 1.5x (39%) |
| blas2 | trsv | f64 | n=4096 | 5.895 ms | 2.85 | 11.40 | 46% | ok | 1.1x (27%) |
| blas2 | trsv | f32 | n=4096 | 3.148 ms | 5.33 | 10.67 | 43% | ok | 1.0x (26%) |
| blas2 | ger | f64 | n=4096 | 14.905 ms | 2.25 | 18.01 | 72% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 5.040 ms | 53.26 | 1.25 | 79% | ok | 3.6x (90%) |
| blas3 | gemm | f32 | 512x512x512 | 2.455 ms | 109.36 | 1.28 | 81% | ok | 3.6x (90%) |
| blas3 | gemmt | f64 | 512x512x512 | 3.467 ms | 38.72 | 1.81 | 57% | ok | 3.3x (82%) |
| blas3 | symm | f64 | 512x512x512 | 5.102 ms | 52.62 | 1.23 | 78% | ok | 3.6x (89%) |
| blas3 | syrk | f64 | 512x512x512 | 3.270 ms | 41.04 | 1.28 | 61% | ok | 3.0x (75%) |
| blas3 | syr2k | f64 | 512x512x512 | 5.742 ms | 46.75 | 1.10 | 69% | ok | 3.3x (84%) |
| blas3 | trmm | f64 | 512x512x512 | 2.519 ms | 53.28 | 2.08 | 79% | ok | 3.6x (91%) |
| blas3 | trsm | f64 | 512x512x512 | 2.670 ms | 50.26 | 1.96 | 74% | ok | 3.6x (91%) |
| blas3 | trsm | f32 | 512x512x512 | 1.453 ms | 92.40 | 1.80 | 68% | ok | 3.4x (86%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 23.299 ms | 0.23 | 1.82 | 7% | ok | 1.1x (29%) |
| factor | lu | f64 | 512x512 | 4.171 ms | 21.45 | 1.01 | 32% | ok | 2.2x (55%) |
| factor | lu | f32 | 512x512 | 3.151 ms | 28.40 | 0.67 | 21% | ok | 1.8x (44%) |
| factor | cholesky | f64 | 512x512 | 3.061 ms | 14.61 | 1.37 | 22% | ok | 1.8x (44%) |
| factor | cholesky | f32 | 512x512 | 2.030 ms | 22.04 | 1.03 | 16% | ok | 1.6x (40%) |
| factor | ldlt | f64 | 512x512 | 20.285 ms | 2.21 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 60.801 ms | 0.74 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 9.793 ms | 18.27 | 0.43 | 27% | ok | 2.0x (51%) |
| factor | lq | f64 | 512x512 | 36.537 ms | 4.90 | 0.11 | 7% | ok | 1.7x (42%) |
| factor | hessenberg | f64 | 512x512 | 200.759 ms | 2.23 | 0.02 | 3% | ok | 1.9x (48%) |
| factor | householder | f64 | 512x512 | 0.463 ms | 2.27 | 9.06 | 36% | ok | 3.4x (84%) |
| factor | inv | f64 | 512x512 | 111.848 ms | 2.40 | 0.04 | 4% | ok | 1.1x (29%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.454 ms | 12.00 | 0.56 | 18% | ok | 1.4x (34%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 5.388 ms | 4.15 | 0.19 | 6% | ok | 1.4x (34%) |
| eigen | eigen_symmetric | f64 | 256x256 | 87.186 ms | 1.73 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 535.913 ms | 0.28 | 0.00 | 0% | ok | 1.1x (28%) |
| eigen | eigenvalue | f64 | 256x256 | 37.284 ms | 4.50 | 0.03 | 7% | ok | 1.2x (30%) |
| eigen | schur | f64 | 256x256 | 48.084 ms | 8.72 | 0.02 | 13% | ok | 1.3x (33%) |
| eigen | svd | f64 | 256x256 | 19.720 ms | 18.72 | 0.05 | 28% | ok | 1.5x (37%) |
| eigen | expm | f64 | 256x256 | 23.770 ms | 10.59 | 0.04 | 16% | ok | 1.6x (40%) |
| eigen | sylvester | f64 | 256x256 | 137.018 ms | 7.35 | 0.01 | 11% | ok | 1.3x (32%) |
| elementwise | exp | f64 | n=8388608 | 73.529 ms | 0.11 | 1.83 | 7% | ok | 1.2x (29%) |
| elementwise | log | f64 | n=8388608 | 94.928 ms | 0.09 | 1.41 | 6% | ok | 1.1x (27%) |
| elementwise | sin | f64 | n=8388608 | 130.939 ms | 0.06 | 1.03 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 60.739 ms | 0.14 | 2.21 | 9% | ok | 1.1x (28%) |
| elementwise | erf | f64 | n=8388608 | 103.454 ms | 0.08 | 1.30 | 5% | ok | 1.1x (28%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.200 ms | 2.18 | 22.70 | 91% | ok | 1.4x (34%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.998 ms | 1.31 | 13.63 | 55% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.199 ms | 6.76 | 14.20 | 57% | ok | 1.9x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.932 ms | 0.27 | 0.96 | 4% | ok | 1.1x (27%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 9.006 ms | 0.22 | 1.85 | 7% | ok | 1.0x (24%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 162.927 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 52.170 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.433 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 94.318 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 129.495 ms | 0.17 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.819 ms | 0.56 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 185.564 ms | 0.01 | 0.00 | 0% | ok | 1.1x (27%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.726 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.415 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.631 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.698 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.396 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 155.142 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.997 ms | 0.38 | 1.51 | 6% | ok | 0.9x (23%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.677 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.394 ms | 1.08 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.181 ms | 0.35 | 1.39 | 6% | ok | 0.8x (20%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.003 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 80.892 ms | 0.28 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 6.236 ms | 0.12 | 0.49 | 2% | ok | 1.0x (24%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.330 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 111.051 ms | 0.20 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.717 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 14.939 ms | 0.75 | 0.05 | 1% | ok | 1.1x (28%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 3.507 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 10.156 ms | 1.10 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.100 ms | 0.69 | 2.75 | 11% | ok | 1.1x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 19.109 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 14674.879 ms | 0.02 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.995 ms | 0.42 | 1.69 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 41.724 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 183.697 ms | 1.68 | 0.00 | 2% | ok | 1.1x (27%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.858 ms | 0.43 | 1.71 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.596 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1600.214 ms | 0.39 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.674 ms | 0.12 | 0.49 | 2% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 25.851 ms | 0.00 | 0.00 | n/a | ok | 1.1x (29%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2379.090 ms | 0.26 | 0.00 | 0% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.330 ms | 0.12 | 0.48 | 2% | ok | 1.0x (25%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 6.694 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 28.466 ms | 10.84 | 0.03 | 16% | ok | 1.6x (40%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 2.272 ms | 1.48 | 5.93 | 24% | ok | 1.1x (27%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 63.911 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 96.228 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.913 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.365 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.407 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.888 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.894 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.663 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.278 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 34.001 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.635 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.927 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.893 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.103 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.649 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.618 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.879 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.779 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.726 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.998 ms | 1.31 | 14.68 | 59% | ok | 1.0x (26%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 4.304 ms | 0.61 | 6.81 | 27% | ok | 1.1x (28%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.667 ms | 0.98 | 10.99 | 44% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.654 ms | 1.18 | 8.90 | 36% | ok | 1.0x (24%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 207.240 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 211.894 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 130.721 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 12.112 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.228 ms | 2.32 | 9.28 | 37% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.767 ms | 21.03 | 0.33 | 31% | ok | 3.8x (96%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.639 ms | 6.36 | 6.76 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.835 ms | 5.92 | 23.67 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.388 ms | 12.09 | 24.17 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.263 ms | 5.14 | 5.14 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.042 ms | 16.10 | 16.10 | n/a | ok | 1.1x (27%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.743 ms | 22.86 | 0.27 | n/a | ok | 2.1x (53%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.679 ms | 40.19 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.147 ms | 52.16 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.151 ms | 52.11 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 9.758 ms | 27.51 | 0.16 | n/a | ok | 3.9x (99%) |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

