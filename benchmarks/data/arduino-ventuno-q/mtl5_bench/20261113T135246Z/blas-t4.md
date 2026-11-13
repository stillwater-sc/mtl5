## arduino-ventuno-q -- blas, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `967d20ac6b04` |
| FMA peak (ceiling) | f64 67.4 GFLOP/s, f32 135.0 GFLOP/s |
| ... hardware, NEON | f64 67.4, f32 135.0 GFLOP/s |
| ... this build, scalar | f64 33.7, f32 33.7 GFLOP/s |
| bandwidth | read 22.9, update 24.7, triad 22.7 GB/s (ceiling 24.7) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 98% | 113% | 0 | 28% |
| blas2 | 10 | 63% | 106% | 0 | 36% |
| blas3 | 10 | 75% | 81% | 0 | 88% |
| factor | 12 | 18% | 39% | 0 | 44% |
| eigen | 8 | 11% | 28% | 0 | 33% |
| elementwise | 5 | 5% | 8% | 0 | 28% |
| sparse | 5 | 56% | 91% | 0 | 28% |
| direct | 23 | 1% | 11% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 28% |
| direct3d | 15 | 2% | 24% | 0 | 28% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 44% | 59% | 0 | 28% |
| mg | 1 | 36% | 36% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 31% | 38% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 53% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.541 ms | 3.03 | 24.22 | 98% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.802 ms | 5.99 | 23.95 | 97% | ok | 1.5x (39%) |
| blas1 | axpy | f64 | n=8388608 | 7.953 ms | 2.11 | 25.31 | 102% | ok | 1.0x (25%) |
| blas1 | axpy | f32 | n=8388608 | 3.985 ms | 4.21 | 25.26 | 102% | ok | 1.0x (26%) |
| blas1 | scale | f64 | n=8388608 | 4.798 ms | 1.75 | 27.97 | 113% | ok | 1.1x (29%) |
| blas1 | two_norm | f64 | n=8388608 | 2.624 ms | 6.39 | 25.58 | 103% | ok | 1.2x (30%) |
| blas1 | one_norm | f64 | n=8388608 | 7.777 ms | 1.08 | 8.63 | 35% | ok | 1.0x (26%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.485 ms | 0.54 | 4.33 | 18% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.287 ms | 1.15 | 9.21 | 37% | ok | 1.2x (29%) |
| blas1 | dot_k3 | f64 | n=8388608 | 175.199 ms | 1.05 | 0.77 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 5.136 ms | 6.53 | 26.15 | 106% | ok | 1.8x (44%) |
| blas2 | gemv | f32 | n=4096 | 2.603 ms | 12.89 | 25.79 | 104% | ok | 1.5x (36%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 8.596 ms | 3.90 | 15.62 | 63% | ok | 1.1x (27%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.825 ms | 6.95 | 13.91 | 56% | ok | 0.8x (19%) |
| blas2 | symv | f64 | n=4096 | 6.401 ms | 5.24 | 10.49 | 42% | ok | 1.8x (46%) |
| blas2 | trmv | f64 | n=4096 | 3.656 ms | 4.59 | 18.37 | 74% | ok | 1.7x (43%) |
| blas2 | trmv | f32 | n=4096 | 2.164 ms | 7.75 | 15.52 | 63% | ok | 1.5x (38%) |
| blas2 | trsv | f64 | n=4096 | 5.997 ms | 2.80 | 11.20 | 45% | ok | 1.1x (27%) |
| blas2 | trsv | f32 | n=4096 | 3.052 ms | 5.50 | 11.00 | 45% | ok | 1.1x (27%) |
| blas2 | ger | f64 | n=4096 | 14.909 ms | 2.25 | 18.01 | 73% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 4.986 ms | 53.84 | 1.26 | 80% | ok | 3.6x (90%) |
| blas3 | gemm | f32 | 512x512x512 | 2.444 ms | 109.85 | 1.29 | 81% | ok | 3.5x (88%) |
| blas3 | gemmt | f64 | 512x512x512 | 3.352 ms | 40.04 | 1.88 | 59% | ok | 3.4x (84%) |
| blas3 | symm | f64 | 512x512x512 | 5.031 ms | 53.36 | 1.25 | 79% | ok | 3.6x (90%) |
| blas3 | syrk | f64 | 512x512x512 | 3.245 ms | 41.36 | 1.29 | 61% | ok | 3.0x (76%) |
| blas3 | syr2k | f64 | 512x512x512 | 5.685 ms | 47.22 | 1.11 | 70% | ok | 3.4x (85%) |
| blas3 | trmm | f64 | 512x512x512 | 2.501 ms | 53.67 | 2.10 | 80% | ok | 3.6x (91%) |
| blas3 | trsm | f64 | 512x512x512 | 2.655 ms | 50.55 | 1.97 | 75% | ok | 3.6x (91%) |
| blas3 | trsm | f32 | 512x512x512 | 2.812 ms | 47.73 | 0.93 | 35% | ok | 1.8x (45%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 28.344 ms | 0.19 | 1.50 | 6% | ok | 1.0x (26%) |
| factor | lu | f64 | 512x512 | 4.095 ms | 21.85 | 1.02 | 32% | ok | 2.2x (56%) |
| factor | lu | f32 | 512x512 | 3.119 ms | 28.69 | 0.67 | 21% | ok | 1.8x (44%) |
| factor | cholesky | f64 | 512x512 | 2.970 ms | 15.06 | 1.41 | 22% | ok | 1.8x (45%) |
| factor | cholesky | f32 | 512x512 | 1.996 ms | 22.42 | 1.05 | 17% | ok | 1.6x (41%) |
| factor | ldlt | f64 | 512x512 | 20.291 ms | 2.20 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 61.396 ms | 0.73 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 9.650 ms | 18.55 | 0.43 | 28% | ok | 2.0x (51%) |
| factor | lq | f64 | 512x512 | 36.887 ms | 4.85 | 0.11 | 7% | ok | 1.7x (41%) |
| factor | hessenberg | f64 | 512x512 | 201.530 ms | 2.22 | 0.02 | 3% | ok | 1.7x (44%) |
| factor | householder | f64 | 512x512 | 0.434 ms | 2.42 | 9.67 | 39% | ok | 3.6x (90%) |
| factor | inv | f64 | 512x512 | 113.836 ms | 2.36 | 0.04 | 3% | ok | 1.1x (28%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.342 ms | 12.19 | 0.57 | 18% | ok | 1.4x (35%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 5.389 ms | 4.15 | 0.19 | 6% | ok | 1.4x (34%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.805 ms | 1.74 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 528.715 ms | 0.29 | 0.00 | 0% | ok | 1.2x (29%) |
| eigen | eigenvalue | f64 | 256x256 | 37.466 ms | 4.48 | 0.03 | 7% | ok | 1.2x (30%) |
| eigen | schur | f64 | 256x256 | 47.821 ms | 8.77 | 0.02 | 13% | ok | 1.3x (33%) |
| eigen | svd | f64 | 256x256 | 19.382 ms | 19.04 | 0.05 | 28% | ok | 1.5x (38%) |
| eigen | expm | f64 | 256x256 | 23.579 ms | 10.67 | 0.04 | 16% | ok | 1.6x (40%) |
| eigen | sylvester | f64 | 256x256 | 136.028 ms | 7.40 | 0.01 | 11% | ok | 1.3x (32%) |
| elementwise | exp | f64 | n=8388608 | 80.671 ms | 0.10 | 1.66 | 7% | ok | 1.1x (27%) |
| elementwise | log | f64 | n=8388608 | 99.998 ms | 0.08 | 1.34 | 5% | ok | 1.1x (28%) |
| elementwise | sin | f64 | n=8388608 | 132.528 ms | 0.06 | 1.01 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 64.929 ms | 0.13 | 2.07 | 8% | ok | 1.1x (28%) |
| elementwise | erf | f64 | n=8388608 | 106.420 ms | 0.08 | 1.26 | 5% | ok | 1.1x (27%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.204 ms | 2.17 | 22.61 | 91% | ok | 1.4x (35%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.961 ms | 1.33 | 13.89 | 56% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.155 ms | 6.80 | 14.31 | 58% | ok | 2.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.663 ms | 0.28 | 0.99 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.969 ms | 0.22 | 1.86 | 8% | ok | 1.0x (25%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 162.886 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 51.975 ms | 0.22 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.354 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 94.418 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 129.128 ms | 0.17 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.923 ms | 0.56 | 0.02 | 1% | ok | 1.1x (27%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 184.129 ms | 0.01 | 0.00 | 0% | ok | 1.1x (27%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.744 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.500 ms | 0.00 | 0.00 | n/a | ok | 1.1x (29%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.627 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.195 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.311 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 155.379 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.017 ms | 0.37 | 1.50 | 6% | ok | 0.9x (22%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.821 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.350 ms | 1.08 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.996 ms | 0.38 | 1.52 | 6% | ok | 0.9x (21%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.965 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 79.637 ms | 0.28 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.685 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.053 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 110.179 ms | 0.20 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.711 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 14.798 ms | 0.76 | 0.05 | 1% | ok | 1.1x (28%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 3.495 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 9.970 ms | 1.12 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.096 ms | 0.69 | 2.76 | 11% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 18.950 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 14664.925 ms | 0.02 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 8.023 ms | 0.42 | 1.68 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 41.749 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 184.717 ms | 1.67 | 0.00 | 2% | ok | 1.1x (29%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.825 ms | 0.43 | 1.72 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.632 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1477.609 ms | 0.42 | 0.00 | 1% | ok | 1.2x (29%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.922 ms | 0.12 | 0.48 | 2% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 26.100 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2257.167 ms | 0.27 | 0.00 | 0% | ok | 1.1x (26%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.119 ms | 0.12 | 0.48 | 2% | ok | 1.0x (25%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 6.655 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 27.722 ms | 11.13 | 0.03 | 17% | ok | 1.6x (40%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 2.274 ms | 1.48 | 5.93 | 24% | ok | 1.1x (27%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.185 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 96.293 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.518 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.185 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.308 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.602 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.845 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.936 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.387 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.796 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.618 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.747 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.214 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.102 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.641 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilut | f64 | 128^2 grid | 27.727 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.867 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.781 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.065 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.732 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.018 ms | 1.30 | 14.53 | 59% | ok | 1.0x (26%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 4.253 ms | 0.62 | 6.90 | 28% | ok | 1.1x (28%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.667 ms | 0.98 | 11.00 | 44% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.373 ms | 1.19 | 8.98 | 36% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 207.642 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 212.611 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 137.968 ms | 0.00 | 0.00 | n/a | ok | 1.1x (26%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.928 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.196 ms | 2.33 | 9.33 | 38% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.850 ms | 20.89 | 0.33 | 31% | ok | 3.8x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.637 ms | 6.36 | 6.76 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.804 ms | 5.98 | 23.93 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.378 ms | 12.17 | 24.35 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.274 ms | 5.12 | 5.12 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.039 ms | 16.15 | 16.15 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.622 ms | 23.10 | 0.27 | n/a | ok | 2.1x (53%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.687 ms | 40.14 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.136 ms | 52.26 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.136 ms | 52.27 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 9.736 ms | 27.57 | 0.16 | n/a | ok | 3.9x (99%) |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

