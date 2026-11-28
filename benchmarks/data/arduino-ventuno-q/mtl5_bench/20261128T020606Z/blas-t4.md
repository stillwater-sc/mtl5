## arduino-ventuno-q -- blas, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 67.8 GFLOP/s, f32 135.6 GFLOP/s |
| ... hardware, NEON | f64 67.8, f32 135.6 GFLOP/s |
| ... this build, scalar | f64 33.8, f32 33.7 GFLOP/s |
| bandwidth | read 22.3, update 24.6, triad 22.2 GB/s (ceiling 24.6) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 98% | 110% | 0 | 28% |
| blas2 | 10 | 63% | 119% | 0 | 36% |
| blas3 | 10 | 72% | 80% | 0 | 88% |
| factor | 12 | 18% | 37% | 0 | 44% |
| eigen | 8 | 11% | 28% | 0 | 33% |
| elementwise | 5 | 5% | 8% | 0 | 27% |
| sparse | 5 | 57% | 92% | 0 | 28% |
| direct | 23 | 1% | 11% | 0 | 27% |
| ordering | 5 | n/a | n/a | 0 | 28% |
| direct3d | 15 | 3% | 24% | 0 | 27% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 52% | 62% | 0 | 27% |
| mg | 1 | 36% | 36% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 32% | 38% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 53% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.584 ms | 3.00 | 24.04 | 98% | ok | 1.0x (25%) |
| blas1 | dot | f32 | n=8388608 | 2.819 ms | 5.95 | 23.81 | 97% | ok | 1.6x (40%) |
| blas1 | axpy | f64 | n=8388608 | 8.153 ms | 2.06 | 24.69 | 100% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 4.189 ms | 4.00 | 24.03 | 98% | ok | 1.0x (24%) |
| blas1 | scale | f64 | n=8388608 | 4.956 ms | 1.69 | 27.08 | 110% | ok | 1.1x (28%) |
| blas1 | two_norm | f64 | n=8388608 | 2.641 ms | 6.35 | 25.41 | 103% | ok | 1.2x (30%) |
| blas1 | one_norm | f64 | n=8388608 | 7.246 ms | 1.16 | 9.26 | 38% | ok | 1.1x (28%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.441 ms | 0.54 | 4.35 | 18% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.427 ms | 1.13 | 9.04 | 37% | ok | 1.2x (29%) |
| blas1 | dot_k3 | f64 | n=8388608 | 181.882 ms | 1.01 | 0.74 | 3% | ok | 1.1x (27%) |
| blas2 | gemv | f64 | n=4096 | 5.086 ms | 6.60 | 26.40 | 107% | ok | 1.8x (44%) |
| blas2 | gemv | f32 | n=4096 | 2.610 ms | 12.86 | 25.72 | 105% | ok | 1.4x (36%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 8.841 ms | 3.80 | 15.19 | 62% | ok | 1.1x (27%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.724 ms | 7.10 | 14.21 | 58% | ok | 0.8x (20%) |
| blas2 | symv | f64 | n=4096 | 6.407 ms | 5.24 | 10.48 | 43% | ok | 1.8x (45%) |
| blas2 | trmv | f64 | n=4096 | 3.631 ms | 4.62 | 18.50 | 75% | ok | 1.7x (43%) |
| blas2 | trmv | f32 | n=4096 | 2.177 ms | 7.71 | 15.43 | 63% | ok | 1.5x (37%) |
| blas2 | trsv | f64 | n=4096 | 5.984 ms | 2.80 | 11.22 | 46% | ok | 1.1x (27%) |
| blas2 | trsv | f32 | n=4096 | 3.055 ms | 5.49 | 11.00 | 45% | ok | 1.1x (27%) |
| blas2 | ger | f64 | n=4096 | 9.199 ms | 3.65 | 29.19 | 119% | ok | 1.2x (31%) |
| blas3 | gemm | f64 | 512x512x512 | 5.074 ms | 52.90 | 1.24 | 78% | ok | 3.5x (89%) |
| blas3 | gemm | f32 | 512x512x512 | 2.471 ms | 108.65 | 1.27 | 80% | ok | 3.5x (88%) |
| blas3 | gemmt | f64 | 512x512x512 | 3.415 ms | 39.30 | 1.84 | 58% | ok | 3.3x (83%) |
| blas3 | symm | f64 | 512x512x512 | 5.073 ms | 52.91 | 1.24 | 78% | ok | 3.6x (90%) |
| blas3 | syrk | f64 | 512x512x512 | 3.283 ms | 40.88 | 1.28 | 60% | ok | 3.0x (74%) |
| blas3 | syr2k | f64 | 512x512x512 | 5.881 ms | 45.65 | 1.07 | 67% | ok | 3.3x (82%) |
| blas3 | trmm | f64 | 512x512x512 | 2.588 ms | 51.86 | 2.03 | 76% | ok | 3.5x (89%) |
| blas3 | trsm | f64 | 512x512x512 | 2.742 ms | 48.95 | 1.91 | 72% | ok | 3.5x (88%) |
| blas3 | trsm | f32 | 512x512x512 | 1.462 ms | 91.81 | 1.79 | 68% | ok | 3.4x (86%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 30.171 ms | 0.18 | 1.41 | 6% | ok | 1.1x (27%) |
| factor | lu | f64 | 512x512 | 4.181 ms | 21.40 | 1.00 | 32% | ok | 2.2x (55%) |
| factor | lu | f32 | 512x512 | 3.127 ms | 28.61 | 0.67 | 21% | ok | 1.8x (44%) |
| factor | cholesky | f64 | 512x512 | 2.963 ms | 15.10 | 1.42 | 22% | ok | 1.8x (45%) |
| factor | cholesky | f32 | 512x512 | 2.000 ms | 22.37 | 1.05 | 16% | ok | 1.6x (40%) |
| factor | ldlt | f64 | 512x512 | 19.366 ms | 2.31 | 0.22 | 3% | ok | 1.1x (27%) |
| factor | ldlt_bk | f64 | 512x512 | 10.525 ms | 4.25 | 0.40 | 6% | ok | 1.1x (27%) |
| factor | qr | f64 | 512x512 | 9.739 ms | 18.37 | 0.43 | 27% | ok | 2.0x (51%) |
| factor | lq | f64 | 512x512 | 36.710 ms | 4.87 | 0.11 | 7% | ok | 1.7x (41%) |
| factor | hessenberg | f64 | 512x512 | 199.941 ms | 2.24 | 0.02 | 3% | ok | 1.7x (44%) |
| factor | householder | f64 | 512x512 | 0.457 ms | 2.30 | 9.17 | 37% | ok | 3.4x (84%) |
| factor | inv | f64 | 512x512 | 125.169 ms | 2.14 | 0.03 | 3% | ok | 1.1x (26%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.195 ms | 12.44 | 0.58 | 18% | ok | 1.4x (35%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 5.384 ms | 4.15 | 0.19 | 6% | ok | 1.4x (34%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.929 ms | 1.74 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 526.637 ms | 0.29 | 0.00 | 0% | ok | 1.1x (28%) |
| eigen | eigenvalue | f64 | 256x256 | 37.302 ms | 4.50 | 0.03 | 7% | ok | 1.2x (30%) |
| eigen | schur | f64 | 256x256 | 47.850 ms | 8.77 | 0.02 | 13% | ok | 1.3x (33%) |
| eigen | svd | f64 | 256x256 | 19.704 ms | 18.73 | 0.05 | 28% | ok | 1.5x (37%) |
| eigen | expm | f64 | 256x256 | 23.147 ms | 10.87 | 0.05 | 16% | ok | 1.6x (40%) |
| eigen | sylvester | f64 | 256x256 | 135.957 ms | 7.40 | 0.01 | 11% | ok | 1.3x (32%) |
| elementwise | exp | f64 | n=8388608 | 82.605 ms | 0.10 | 1.62 | 7% | ok | 1.1x (27%) |
| elementwise | log | f64 | n=8388608 | 101.725 ms | 0.08 | 1.32 | 5% | ok | 1.1x (27%) |
| elementwise | sin | f64 | n=8388608 | 136.492 ms | 0.06 | 0.98 | 4% | ok | 1.1x (27%) |
| elementwise | sqrt | f64 | n=8388608 | 67.352 ms | 0.12 | 1.99 | 8% | ok | 1.1x (27%) |
| elementwise | erf | f64 | n=8388608 | 108.258 ms | 0.08 | 1.24 | 5% | ok | 1.1x (27%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.206 ms | 2.17 | 22.57 | 92% | ok | 1.4x (34%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.958 ms | 1.34 | 13.91 | 57% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.194 ms | 6.76 | 14.21 | 58% | ok | 1.9x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.619 ms | 0.28 | 0.99 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 9.112 ms | 0.22 | 1.83 | 7% | ok | 1.0x (24%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 33.316 ms | 0.34 | 0.02 | 0% | ok | 1.1x (27%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 59.717 ms | 0.19 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.236 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 93.162 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 111.481 ms | 0.20 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 38.385 ms | 0.58 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 67.565 ms | 0.03 | 0.00 | 0% | ok | 1.1x (28%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.672 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.252 ms | 0.00 | 0.00 | n/a | ok | 1.2x (29%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 111.733 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.524 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.750 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 4.995 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 26.231 ms | 0.43 | 0.03 | 1% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.258 ms | 0.33 | 1.34 | 5% | ok | 0.8x (21%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.130 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.654 ms | 1.05 | 0.07 | 2% | ok | 1.1x (27%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.165 ms | 0.35 | 1.40 | 6% | ok | 0.8x (20%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.979 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 79.203 ms | 0.28 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.766 ms | 0.13 | 0.52 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 7.869 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 94.212 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.759 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 14.990 ms | 0.75 | 0.05 | 1% | ok | 1.1x (28%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 3.499 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 10.188 ms | 1.10 | 0.07 | 2% | ok | 1.1x (27%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.094 ms | 0.69 | 2.77 | 11% | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 10.752 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 604.697 ms | 0.51 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.832 ms | 0.43 | 1.72 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 24.772 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 165.383 ms | 1.87 | 0.01 | 3% | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.836 ms | 0.43 | 1.72 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.621 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1589.010 ms | 0.39 | 0.00 | 1% | ok | 1.1x (28%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.556 ms | 0.11 | 0.46 | 2% | ok | 1.1x (26%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 14.920 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1103.079 ms | 0.56 | 0.00 | 1% | ok | 1.1x (28%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.050 ms | 0.12 | 0.46 | 2% | ok | 1.1x (27%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 6.643 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 28.049 ms | 11.00 | 0.03 | 16% | ok | 1.6x (39%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 2.270 ms | 1.48 | 5.94 | 24% | ok | 1.1x (26%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 63.674 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 108.219 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.692 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.390 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.524 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.713 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 55.569 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 56.949 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.210 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 35.562 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.483 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 31.651 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 37.546 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.103 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.576 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.108 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.881 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.812 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.074 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.757 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.929 ms | 1.36 | 15.20 | 62% | ok | 1.0x (26%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.279 ms | 1.15 | 12.87 | 52% | ok | 1.1x (27%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.682 ms | 0.98 | 10.94 | 44% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.459 ms | 1.18 | 8.95 | 36% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 211.021 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 212.522 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 131.850 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.935 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.183 ms | 2.34 | 9.34 | 38% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.560 ms | 21.37 | 0.33 | 32% | ok | 3.9x (97%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.644 ms | 6.35 | 6.74 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.919 ms | 5.75 | 22.99 | n/a | ok | 1.0x (24%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.393 ms | 12.04 | 24.09 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.270 ms | 5.13 | 5.13 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.040 ms | 16.14 | 16.14 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.670 ms | 23.00 | 0.27 | n/a | ok | 2.1x (53%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.680 ms | 40.18 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.150 ms | 52.13 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.154 ms | 52.08 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 9.750 ms | 27.53 | 0.16 | n/a | ok | 4.0x (99%) |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

