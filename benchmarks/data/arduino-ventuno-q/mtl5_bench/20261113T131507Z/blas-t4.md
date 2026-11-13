## arduino-ventuno-q -- blas, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 67.8 GFLOP/s, f32 135.0 GFLOP/s |
| ... hardware, NEON | f64 67.8, f32 135.0 GFLOP/s |
| ... this build, scalar | f64 33.9, f32 33.7 GFLOP/s |
| bandwidth | read 22.6, update 24.6, triad 22.4 GB/s (ceiling 24.6) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 98% | 116% | 0 | 28% |
| blas2 | 10 | 63% | 108% | 0 | 37% |
| blas3 | 10 | 74% | 81% | 0 | 88% |
| factor | 12 | 18% | 37% | 0 | 44% |
| eigen | 8 | 11% | 28% | 0 | 33% |
| elementwise | 5 | 6% | 9% | 0 | 27% |
| sparse | 5 | 56% | 91% | 0 | 28% |
| direct | 23 | 1% | 11% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 27% |
| direct3d | 15 | 2% | 24% | 0 | 27% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 45% | 61% | 0 | 28% |
| mg | 1 | 36% | 36% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 32% | 38% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 53% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.550 ms | 3.02 | 24.18 | 98% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.809 ms | 5.97 | 23.89 | 97% | ok | 1.6x (40%) |
| blas1 | axpy | f64 | n=8388608 | 7.923 ms | 2.12 | 25.41 | 103% | ok | 1.0x (25%) |
| blas1 | axpy | f32 | n=8388608 | 3.966 ms | 4.23 | 25.38 | 103% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 4.714 ms | 1.78 | 28.47 | 116% | ok | 1.2x (29%) |
| blas1 | two_norm | f64 | n=8388608 | 2.627 ms | 6.39 | 25.55 | 104% | ok | 1.2x (31%) |
| blas1 | one_norm | f64 | n=8388608 | 7.408 ms | 1.13 | 9.06 | 37% | ok | 1.2x (29%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.518 ms | 0.54 | 4.32 | 18% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.457 ms | 1.12 | 9.00 | 37% | ok | 1.1x (27%) |
| blas1 | dot_k3 | f64 | n=8388608 | 165.155 ms | 1.12 | 0.81 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 5.060 ms | 6.63 | 26.54 | 108% | ok | 1.8x (45%) |
| blas2 | gemv | f32 | n=4096 | 2.610 ms | 12.85 | 25.72 | 105% | ok | 1.5x (37%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 8.708 ms | 3.85 | 15.42 | 63% | ok | 1.1x (27%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.608 ms | 7.28 | 14.57 | 59% | ok | 0.8x (20%) |
| blas2 | symv | f64 | n=4096 | 6.390 ms | 5.25 | 10.51 | 43% | ok | 1.8x (45%) |
| blas2 | trmv | f64 | n=4096 | 3.641 ms | 4.61 | 18.45 | 75% | ok | 1.7x (43%) |
| blas2 | trmv | f32 | n=4096 | 2.171 ms | 7.73 | 15.47 | 63% | ok | 1.5x (38%) |
| blas2 | trsv | f64 | n=4096 | 6.044 ms | 2.78 | 11.11 | 45% | ok | 1.1x (26%) |
| blas2 | trsv | f32 | n=4096 | 3.078 ms | 5.45 | 10.91 | 44% | ok | 1.0x (26%) |
| blas2 | ger | f64 | n=4096 | 14.823 ms | 2.26 | 18.11 | 74% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 5.072 ms | 52.93 | 1.24 | 78% | ok | 3.5x (88%) |
| blas3 | gemm | f32 | 512x512x512 | 2.469 ms | 108.72 | 1.27 | 81% | ok | 3.5x (88%) |
| blas3 | gemmt | f64 | 512x512x512 | 6.506 ms | 20.63 | 0.97 | 30% | ok | 1.8x (44%) |
| blas3 | symm | f64 | 512x512x512 | 5.109 ms | 52.54 | 1.23 | 77% | ok | 3.5x (89%) |
| blas3 | syrk | f64 | 512x512x512 | 3.293 ms | 40.76 | 1.27 | 60% | ok | 3.0x (75%) |
| blas3 | syr2k | f64 | 512x512x512 | 5.813 ms | 46.18 | 1.08 | 68% | ok | 3.3x (83%) |
| blas3 | trmm | f64 | 512x512x512 | 2.546 ms | 52.73 | 2.06 | 78% | ok | 3.6x (89%) |
| blas3 | trsm | f64 | 512x512x512 | 2.690 ms | 49.89 | 1.95 | 74% | ok | 3.6x (91%) |
| blas3 | trsm | f32 | 512x512x512 | 1.464 ms | 91.68 | 1.79 | 68% | ok | 3.4x (86%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 24.733 ms | 0.21 | 1.72 | 7% | ok | 1.0x (26%) |
| factor | lu | f64 | 512x512 | 4.111 ms | 21.76 | 1.02 | 32% | ok | 2.2x (56%) |
| factor | lu | f32 | 512x512 | 3.132 ms | 28.56 | 0.67 | 21% | ok | 1.8x (44%) |
| factor | cholesky | f64 | 512x512 | 3.006 ms | 14.88 | 1.40 | 22% | ok | 1.8x (45%) |
| factor | cholesky | f32 | 512x512 | 2.016 ms | 22.20 | 1.04 | 16% | ok | 1.6x (40%) |
| factor | ldlt | f64 | 512x512 | 20.319 ms | 2.20 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 60.997 ms | 0.73 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 9.678 ms | 18.49 | 0.43 | 27% | ok | 2.1x (51%) |
| factor | lq | f64 | 512x512 | 36.543 ms | 4.90 | 0.11 | 7% | ok | 1.7x (41%) |
| factor | hessenberg | f64 | 512x512 | 199.979 ms | 2.24 | 0.02 | 3% | ok | 1.8x (44%) |
| factor | householder | f64 | 512x512 | 0.462 ms | 2.27 | 9.07 | 37% | ok | 3.4x (85%) |
| factor | inv | f64 | 512x512 | 123.948 ms | 2.17 | 0.03 | 3% | ok | 1.1x (26%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.393 ms | 12.10 | 0.57 | 18% | ok | 1.4x (35%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 5.382 ms | 4.16 | 0.19 | 6% | ok | 1.4x (34%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.889 ms | 1.74 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 526.810 ms | 0.29 | 0.00 | 0% | ok | 1.1x (28%) |
| eigen | eigenvalue | f64 | 256x256 | 37.219 ms | 4.51 | 0.03 | 7% | ok | 1.2x (30%) |
| eigen | schur | f64 | 256x256 | 47.958 ms | 8.75 | 0.02 | 13% | ok | 1.3x (33%) |
| eigen | svd | f64 | 256x256 | 19.597 ms | 18.83 | 0.05 | 28% | ok | 1.5x (38%) |
| eigen | expm | f64 | 256x256 | 23.598 ms | 10.66 | 0.04 | 16% | ok | 1.6x (40%) |
| eigen | sylvester | f64 | 256x256 | 136.346 ms | 7.38 | 0.01 | 11% | ok | 1.3x (32%) |
| elementwise | exp | f64 | n=8388608 | 75.329 ms | 0.11 | 1.78 | 7% | ok | 1.2x (30%) |
| elementwise | log | f64 | n=8388608 | 97.613 ms | 0.09 | 1.38 | 6% | ok | 1.1x (26%) |
| elementwise | sin | f64 | n=8388608 | 135.814 ms | 0.06 | 0.99 | 4% | ok | 1.1x (27%) |
| elementwise | sqrt | f64 | n=8388608 | 58.898 ms | 0.14 | 2.28 | 9% | ok | 1.2x (30%) |
| elementwise | erf | f64 | n=8388608 | 105.347 ms | 0.08 | 1.27 | 5% | ok | 1.1x (27%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.216 ms | 2.15 | 22.38 | 91% | ok | 1.4x (34%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.960 ms | 1.34 | 13.89 | 56% | ok | 1.1x (28%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.320 ms | 6.63 | 13.93 | 57% | ok | 1.9x (48%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.659 ms | 0.28 | 0.99 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.995 ms | 0.22 | 1.85 | 8% | ok | 1.0x (24%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 163.070 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 52.079 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.463 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 95.332 ms | 0.23 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 128.899 ms | 0.17 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.751 ms | 0.56 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 184.731 ms | 0.01 | 0.00 | 0% | ok | 1.1x (27%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.121 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.116 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.634 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.592 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.366 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 154.844 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.979 ms | 0.38 | 1.53 | 6% | ok | 0.9x (23%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.790 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.395 ms | 1.08 | 0.07 | 2% | ok | 1.1x (27%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.983 ms | 0.38 | 1.53 | 6% | ok | 0.9x (22%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.063 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 80.118 ms | 0.28 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.689 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.024 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 110.241 ms | 0.20 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.663 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 14.880 ms | 0.75 | 0.05 | 1% | ok | 1.1x (28%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 3.511 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 10.082 ms | 1.11 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.096 ms | 0.69 | 2.76 | 11% | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 19.264 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 14669.533 ms | 0.02 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 8.022 ms | 0.42 | 1.68 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 42.498 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 183.808 ms | 1.68 | 0.00 | 2% | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.842 ms | 0.43 | 1.72 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.791 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1613.493 ms | 0.38 | 0.00 | 1% | ok | 1.1x (28%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.566 ms | 0.12 | 0.49 | 2% | ok | 1.1x (27%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 26.745 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2394.236 ms | 0.26 | 0.00 | 0% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.947 ms | 0.12 | 0.48 | 2% | ok | 1.0x (26%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 6.670 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 28.119 ms | 10.97 | 0.03 | 16% | ok | 1.6x (40%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 2.276 ms | 1.48 | 5.92 | 24% | ok | 1.1x (26%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.006 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 96.122 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.398 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.238 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.243 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.568 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.917 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.718 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.407 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.910 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.513 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.713 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.314 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.103 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.653 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.814 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.872 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.800 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.064 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.684 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.945 ms | 1.35 | 15.08 | 61% | ok | 1.1x (27%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 4.227 ms | 0.62 | 6.94 | 28% | ok | 1.1x (28%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.677 ms | 0.98 | 10.96 | 45% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.632 ms | 1.18 | 8.91 | 36% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 207.067 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 211.502 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 134.226 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.952 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.204 ms | 2.33 | 9.32 | 38% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.307 ms | 21.81 | 0.34 | 32% | ok | 4.0x (99%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.640 ms | 6.36 | 6.75 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.022 ms | 5.55 | 22.21 | n/a | ok | 0.9x (24%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.426 ms | 11.76 | 23.52 | n/a | ok | 1.0x (24%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.271 ms | 5.13 | 5.13 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.046 ms | 16.04 | 16.04 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.690 ms | 22.96 | 0.27 | n/a | ok | 2.1x (53%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.679 ms | 40.19 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.137 ms | 52.26 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.142 ms | 52.20 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 9.722 ms | 27.61 | 0.16 | n/a | ok | 3.9x (99%) |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

