## arduino-ventuno-q -- blas, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 67.6 GFLOP/s, f32 134.9 GFLOP/s |
| ... hardware, NEON | f64 67.6, f32 134.9 GFLOP/s |
| ... this build, scalar | f64 33.9, f32 33.7 GFLOP/s |
| bandwidth | read 22.5, update 24.6, triad 22.4 GB/s (ceiling 24.6) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 97% | 115% | 0 | 28% |
| blas2 | 10 | 65% | 118% | 0 | 36% |
| blas3 | 10 | 74% | 81% | 0 | 88% |
| factor | 12 | 18% | 36% | 0 | 43% |
| eigen | 8 | 11% | 28% | 0 | 33% |
| elementwise | 5 | 5% | 8% | 0 | 27% |
| sparse | 5 | 58% | 92% | 0 | 28% |
| direct | 23 | 1% | 11% | 0 | 27% |
| ordering | 5 | n/a | n/a | 0 | 28% |
| direct3d | 15 | 3% | 24% | 0 | 27% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 52% | 60% | 0 | 27% |
| mg | 1 | 37% | 37% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 32% | 38% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 52% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.563 ms | 3.02 | 24.12 | 98% | ok | 1.0x (25%) |
| blas1 | dot | f32 | n=8388608 | 2.812 ms | 5.97 | 23.86 | 97% | ok | 1.5x (38%) |
| blas1 | axpy | f64 | n=8388608 | 8.075 ms | 2.08 | 24.93 | 102% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 4.207 ms | 3.99 | 23.92 | 97% | ok | 0.9x (24%) |
| blas1 | scale | f64 | n=8388608 | 4.773 ms | 1.76 | 28.12 | 115% | ok | 1.1x (28%) |
| blas1 | two_norm | f64 | n=8388608 | 2.620 ms | 6.40 | 25.61 | 104% | ok | 1.2x (30%) |
| blas1 | one_norm | f64 | n=8388608 | 7.368 ms | 1.14 | 9.11 | 37% | ok | 1.1x (27%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.414 ms | 0.54 | 4.35 | 18% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.385 ms | 1.14 | 9.09 | 37% | ok | 1.2x (29%) |
| blas1 | dot_k3 | f64 | n=8388608 | 182.180 ms | 1.01 | 0.74 | 3% | ok | 1.0x (26%) |
| blas2 | gemv | f64 | n=4096 | 5.807 ms | 5.78 | 23.13 | 94% | ok | 1.6x (39%) |
| blas2 | gemv | f32 | n=4096 | 2.611 ms | 12.85 | 25.71 | 105% | ok | 1.4x (36%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 8.415 ms | 3.99 | 15.96 | 65% | ok | 1.1x (28%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.814 ms | 6.97 | 13.95 | 57% | ok | 0.8x (19%) |
| blas2 | symv | f64 | n=4096 | 6.410 ms | 5.24 | 10.48 | 43% | ok | 1.8x (46%) |
| blas2 | trmv | f64 | n=4096 | 3.638 ms | 4.61 | 18.46 | 75% | ok | 1.7x (43%) |
| blas2 | trmv | f32 | n=4096 | 2.166 ms | 7.74 | 15.50 | 63% | ok | 1.5x (38%) |
| blas2 | trsv | f64 | n=4096 | 5.983 ms | 2.80 | 11.23 | 46% | ok | 1.1x (27%) |
| blas2 | trsv | f32 | n=4096 | 3.140 ms | 5.34 | 10.70 | 44% | ok | 1.0x (26%) |
| blas2 | ger | f64 | n=4096 | 9.252 ms | 3.63 | 29.02 | 118% | ok | 1.2x (30%) |
| blas3 | gemm | f64 | 512x512x512 | 5.022 ms | 53.45 | 1.25 | 79% | ok | 3.5x (88%) |
| blas3 | gemm | f32 | 512x512x512 | 2.457 ms | 109.24 | 1.28 | 81% | ok | 3.5x (88%) |
| blas3 | gemmt | f64 | 512x512x512 | 3.391 ms | 39.58 | 1.86 | 59% | ok | 3.4x (84%) |
| blas3 | symm | f64 | 512x512x512 | 5.033 ms | 53.34 | 1.25 | 79% | ok | 3.6x (89%) |
| blas3 | syrk | f64 | 512x512x512 | 3.260 ms | 41.17 | 1.29 | 61% | ok | 3.0x (75%) |
| blas3 | syr2k | f64 | 512x512x512 | 5.769 ms | 46.53 | 1.09 | 69% | ok | 3.3x (83%) |
| blas3 | trmm | f64 | 512x512x512 | 2.520 ms | 53.26 | 2.08 | 79% | ok | 3.6x (90%) |
| blas3 | trsm | f64 | 512x512x512 | 2.682 ms | 50.05 | 1.96 | 74% | ok | 3.6x (90%) |
| blas3 | trsm | f32 | 512x512x512 | 1.460 ms | 91.94 | 1.80 | 68% | ok | 3.4x (86%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 28.585 ms | 0.19 | 1.49 | 6% | ok | 0.9x (23%) |
| factor | lu | f64 | 512x512 | 4.208 ms | 21.26 | 1.00 | 31% | ok | 2.2x (55%) |
| factor | lu | f32 | 512x512 | 3.122 ms | 28.66 | 0.67 | 21% | ok | 1.8x (44%) |
| factor | cholesky | f64 | 512x512 | 2.962 ms | 15.11 | 1.42 | 22% | ok | 1.8x (45%) |
| factor | cholesky | f32 | 512x512 | 1.991 ms | 22.47 | 1.05 | 17% | ok | 1.6x (41%) |
| factor | ldlt | f64 | 512x512 | 19.164 ms | 2.33 | 0.22 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 10.486 ms | 4.27 | 0.40 | 6% | ok | 1.1x (27%) |
| factor | qr | f64 | 512x512 | 9.616 ms | 18.61 | 0.44 | 28% | ok | 2.1x (51%) |
| factor | lq | f64 | 512x512 | 36.907 ms | 4.85 | 0.11 | 7% | ok | 1.7x (41%) |
| factor | hessenberg | f64 | 512x512 | 201.563 ms | 2.22 | 0.02 | 3% | ok | 1.7x (43%) |
| factor | householder | f64 | 512x512 | 0.470 ms | 2.23 | 8.92 | 36% | ok | 3.3x (81%) |
| factor | inv | f64 | 512x512 | 115.377 ms | 2.33 | 0.04 | 3% | ok | 1.2x (29%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.201 ms | 12.43 | 0.58 | 18% | ok | 1.4x (35%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 5.390 ms | 4.15 | 0.19 | 6% | ok | 1.4x (34%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.953 ms | 1.74 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 546.814 ms | 0.28 | 0.00 | 0% | ok | 1.1x (27%) |
| eigen | eigenvalue | f64 | 256x256 | 37.310 ms | 4.50 | 0.03 | 7% | ok | 1.2x (30%) |
| eigen | schur | f64 | 256x256 | 47.930 ms | 8.75 | 0.02 | 13% | ok | 1.3x (33%) |
| eigen | svd | f64 | 256x256 | 19.632 ms | 18.80 | 0.05 | 28% | ok | 1.5x (37%) |
| eigen | expm | f64 | 256x256 | 23.221 ms | 10.84 | 0.05 | 16% | ok | 1.6x (40%) |
| eigen | sylvester | f64 | 256x256 | 136.035 ms | 7.40 | 0.01 | 11% | ok | 1.3x (32%) |
| elementwise | exp | f64 | n=8388608 | 81.893 ms | 0.10 | 1.64 | 7% | ok | 1.0x (26%) |
| elementwise | log | f64 | n=8388608 | 101.014 ms | 0.08 | 1.33 | 5% | ok | 1.1x (26%) |
| elementwise | sin | f64 | n=8388608 | 136.430 ms | 0.06 | 0.98 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 66.471 ms | 0.13 | 2.02 | 8% | ok | 1.1x (27%) |
| elementwise | erf | f64 | n=8388608 | 106.876 ms | 0.08 | 1.26 | 5% | ok | 1.1x (28%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.211 ms | 2.16 | 22.48 | 92% | ok | 1.4x (35%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.914 ms | 1.37 | 14.23 | 58% | ok | 1.1x (28%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.088 ms | 6.88 | 14.46 | 59% | ok | 2.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.588 ms | 0.28 | 0.99 | 4% | ok | 1.1x (27%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 9.169 ms | 0.22 | 1.82 | 7% | ok | 1.0x (26%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 33.678 ms | 0.33 | 0.02 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 59.412 ms | 0.19 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.141 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 93.179 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 112.096 ms | 0.20 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 38.525 ms | 0.58 | 0.02 | 1% | ok | 1.1x (27%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 67.628 ms | 0.03 | 0.00 | 0% | ok | 1.1x (28%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.046 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.365 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 112.088 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.523 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.493 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.033 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 26.075 ms | 0.43 | 0.03 | 1% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.249 ms | 0.34 | 1.34 | 5% | ok | 0.8x (20%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.289 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.660 ms | 1.05 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.118 ms | 0.36 | 1.43 | 6% | ok | 0.8x (20%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.011 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 79.877 ms | 0.28 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.718 ms | 0.13 | 0.53 | 2% | ok | 1.1x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.923 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 95.270 ms | 0.23 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.770 ms | 0.13 | 0.52 | 2% | ok | 1.0x (26%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 14.908 ms | 0.75 | 0.05 | 1% | ok | 1.1x (28%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 3.504 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 10.064 ms | 1.11 | 0.07 | 2% | ok | 1.1x (27%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.101 ms | 0.69 | 2.75 | 11% | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 10.690 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 606.110 ms | 0.51 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.868 ms | 0.43 | 1.71 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 25.056 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 164.658 ms | 1.87 | 0.01 | 3% | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.806 ms | 0.43 | 1.73 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.833 ms | 0.00 | 0.00 | n/a | ok | 1.1x (26%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1569.148 ms | 0.39 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.746 ms | 0.11 | 0.45 | 2% | ok | 1.1x (26%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 15.229 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1110.263 ms | 0.56 | 0.00 | 1% | ok | 1.1x (28%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.736 ms | 0.11 | 0.45 | 2% | ok | 1.1x (27%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 6.719 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 28.072 ms | 10.99 | 0.03 | 16% | ok | 1.6x (39%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 2.283 ms | 1.48 | 5.90 | 24% | ok | 1.1x (26%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 63.641 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 107.531 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.143 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.263 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.516 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.732 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 55.573 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.009 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 30.796 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 35.527 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.575 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 31.669 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 37.649 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.103 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.579 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.350 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.923 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.809 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.073 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.743 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.990 ms | 1.32 | 14.74 | 60% | ok | 1.1x (27%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.293 ms | 1.14 | 12.79 | 52% | ok | 1.1x (27%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.693 ms | 0.97 | 10.89 | 44% | ok | 1.1x (27%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.055 ms | 1.20 | 9.07 | 37% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 211.391 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 214.044 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 130.682 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.878 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.214 ms | 2.33 | 9.30 | 38% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.325 ms | 21.78 | 0.34 | 32% | ok | 3.9x (99%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.638 ms | 6.36 | 6.76 | 28% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.788 ms | 6.02 | 24.07 | n/a | ok | 1.0x (26%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.391 ms | 12.06 | 24.13 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.264 ms | 5.14 | 5.14 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.053 ms | 15.94 | 15.94 | n/a | ok | 1.1x (27%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.777 ms | 22.79 | 0.27 | n/a | ok | 2.1x (52%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.681 ms | 40.18 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.142 ms | 52.21 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.143 ms | 52.20 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 9.738 ms | 27.56 | 0.16 | n/a | ok | 3.9x (99%) |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

