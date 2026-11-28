## xeon-e5-2420v2 -- native, 6 threads, class large

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 59.1 GFLOP/s, f32 118.2 GFLOP/s |
| ... hardware, SSE2 | f64 59.1, f32 118.2 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.7 GFLOP/s |
| bandwidth | read 11.4, update 10.7, triad 8.5 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 53% | 107% | 0 | 84% |
| blas2 | 10 | 79% | 107% | 0 | 16% |
| blas3 | 10 | 7% | 39% | 0 | 17% |
| factor | 12 | 6% | 40% | 0 | 53% |
| eigen | 8 | 2% | 12% | 0 | 19% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 64% | 97% | 0 | 17% |
| direct | 19 | 1% | 15% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 3% | 14% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 69% | 76% | 0 | 17% |
| mg | 1 | 59% | 59% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 22% | 65% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 87% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 131.944 ms | 0.51 | 4.07 | 36% | ok | 5.6x (93%) |
| blas1 | dot | f32 | n=33554432 | 22.067 ms | 3.04 | 12.16 | 107% | ok | 5.1x (84%) |
| blas1 | axpy | f64 | n=33554432 | 151.983 ms | 0.44 | 5.30 | 46% | ok | 5.6x (93%) |
| blas1 | axpy | f32 | n=33554432 | 66.998 ms | 1.00 | 6.01 | 53% | ok | 5.5x (91%) |
| blas1 | scale | f64 | n=33554432 | 53.956 ms | 0.62 | 9.95 | 87% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=33554432 | 132.581 ms | 0.51 | 2.02 | 18% | ok | 5.7x (94%) |
| blas1 | one_norm | f64 | n=33554432 | 39.065 ms | 0.86 | 6.87 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=33554432 | 52.538 ms | 0.64 | 5.11 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=33554432 | 39.766 ms | 0.84 | 6.75 | 59% | ok | 1.0x (16%) |
| blas1 | dot_k3 | f64 | n=33554432 | 1113.978 ms | 0.66 | 0.48 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=8192 | 44.043 ms | 3.05 | 12.19 | 107% | ok | 1.3x (21%) |
| blas2 | gemv | f32 | n=8192 | 22.116 ms | 6.07 | 12.14 | 106% | ok | 1.5x (26%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 43.881 ms | 3.06 | 12.24 | 107% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 21.909 ms | 6.13 | 12.26 | 107% | ok | 1.2x (21%) |
| blas2 | symv | f64 | n=8192 | 57.306 ms | 2.34 | 4.69 | 41% | ok | 0.9x (15%) |
| blas2 | trmv | f64 | n=8192 | 29.995 ms | 2.24 | 8.95 | 79% | ok | 1.0x (16%) |
| blas2 | trmv | f32 | n=8192 | 19.633 ms | 3.42 | 6.84 | 60% | ok | 0.9x (15%) |
| blas2 | trsv | f64 | n=8192 | 30.043 ms | 2.23 | 8.94 | 78% | ok | 1.0x (16%) |
| blas2 | trsv | f32 | n=8192 | 18.067 ms | 3.71 | 7.43 | 65% | ok | 1.0x (16%) |
| blas2 | ger | f64 | n=8192 | 120.038 ms | 1.12 | 8.95 | 78% | ok | 1.0x (16%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 92.258 ms | 23.28 | 0.27 | 39% | ok | 5.6x (93%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 51.307 ms | 41.86 | 0.25 | 35% | ok | 5.7x (94%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 276.942 ms | 3.88 | 0.09 | 7% | ok | 1.0x (17%) |
| blas3 | symm | f64 | 1024x1024x1024 | 566.905 ms | 3.79 | 0.04 | 6% | ok | 1.0x (17%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 309.704 ms | 3.47 | 0.05 | 6% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 590.272 ms | 3.64 | 0.04 | 6% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 256.762 ms | 4.18 | 0.08 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 258.863 ms | 4.15 | 0.08 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 127.966 ms | 8.39 | 0.08 | 7% | ok | 1.0x (17%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 132.239 ms | 0.13 | 1.01 | 9% | ok | 1.0x (17%) |
| factor | lu | f64 | 1024x1024 | 70.492 ms | 10.15 | 0.24 | 17% | ok | 3.5x (58%) |
| factor | lu | f32 | 1024x1024 | 41.146 ms | 17.40 | 0.20 | 15% | ok | 3.2x (53%) |
| factor | cholesky | f64 | 1024x1024 | 96.576 ms | 3.71 | 0.17 | 6% | ok | 2.1x (35%) |
| factor | cholesky | f32 | 1024x1024 | 93.417 ms | 3.83 | 0.09 | 3% | ok | 2.1x (36%) |
| factor | ldlt | f64 | 1024x1024 | 211.071 ms | 1.70 | 0.08 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 1024x1024 | 105.395 ms | 3.40 | 0.16 | 6% | ok | 1.1x (18%) |
| factor | qr | f64 | 1024x1024 | 1626.430 ms | 0.88 | 0.01 | 1% | ok | 4.2x (70%) |
| factor | lq | f64 | 1024x1024 | 149.458 ms | 9.58 | 0.11 | 16% | ok | 3.9x (65%) |
| factor | hessenberg | f64 | 1024x1024 | 1869.287 ms | 1.91 | 0.01 | 3% | ok | 4.1x (69%) |
| factor | householder | f64 | 1024x1024 | 3.720 ms | 1.13 | 4.51 | 40% | ok | 5.4x (91%) |
| factor | inv | f64 | 1024x1024 | 1004.629 ms | 2.14 | 0.02 | 4% | ok | 1.2x (19%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 56.068 ms | 12.77 | 0.30 | 22% | ok | 2.7x (44%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 415.734 ms | 0.43 | 0.01 | 1% | ok | 1.5x (25%) |
| eigen | eigen_symmetric | f64 | 512x512 | 1843.592 ms | 0.66 | 0.00 | 1% | ok | 1.1x (19%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 14764.464 ms | 0.08 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 512x512 | 859.410 ms | 1.56 | 0.00 | 3% | ok | 1.3x (21%) |
| eigen | schur | f64 | 512x512 | 2602.027 ms | 1.29 | 0.00 | 2% | ok | 1.2x (19%) |
| eigen | svd | f64 | 512x512 | 2675.193 ms | 1.10 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 512x512 | 273.603 ms | 7.36 | 0.02 | 12% | ok | 2.6x (44%) |
| eigen | sylvester | f64 | 512x512 | 5950.041 ms | 1.35 | 0.00 | 2% | ok | 1.2x (19%) |
| elementwise | exp | f64 | n=33554432 | 605.779 ms | 0.06 | 0.89 | 8% | ok | 1.1x (18%) |
| elementwise | log | f64 | n=33554432 | 561.545 ms | 0.06 | 0.96 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=33554432 | 685.244 ms | 0.05 | 0.78 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=33554432 | 363.236 ms | 0.09 | 1.48 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=33554432 | 9939.398 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 9.855 ms | 1.06 | 11.06 | 97% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 13.047 ms | 0.80 | 8.35 | 73% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 48.385 ms | 3.46 | 7.28 | 64% | ok | 1.3x (21%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 102.301 ms | 0.13 | 0.45 | 4% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 37.663 ms | 0.27 | 2.23 | 20% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 305.859 ms | 0.35 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 558.546 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 156.474 ms | 0.68 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 905.552 ms | 0.24 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 855.649 ms | 0.25 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 386.043 ms | 0.55 | 0.01 | 1% | ok | 0.9x (16%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 1154.214 ms | 0.02 | 0.00 | 0% | ok | 1.0x (16%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 43.441 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 63.247 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 580.837 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 2.396 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 252.615 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 26.663 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 270.141 ms | 0.40 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 9.172 ms | 0.44 | 1.75 | 15% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 54.583 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 93.003 ms | 1.15 | 0.03 | 2% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 9.225 ms | 0.44 | 1.74 | 15% | ok | 1.0x (16%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 27.213 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 834.543 ms | 0.26 | 0.00 | 0% | ok | 1.0x (16%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.260 ms | 0.13 | 0.53 | 5% | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 41.178 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 781.863 ms | 0.27 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.272 ms | 0.13 | 0.53 | 5% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 23.696 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1962.714 ms | 0.49 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 19.437 ms | 0.39 | 1.54 | 14% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 61.591 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 607.337 ms | 1.60 | 0.00 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 19.481 ms | 0.38 | 1.54 | 14% | ok | 1.0x (16%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 23.960 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 5766.766 ms | 0.34 | 0.00 | 1% | ok | 1.0x (16%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 88.042 ms | 0.09 | 0.34 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 33.789 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 4931.278 ms | 0.39 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 87.776 ms | 0.09 | 0.34 | 3% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 1741.883 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 2992.927 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 766.588 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 887.666 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 923.967 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 576.894 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 4859.506 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 1229.463 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 722.349 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 1427.147 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 1296.930 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 1496.126 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 2065.505 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.689 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 256^2 grid | 18.224 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 256^2 grid | 149.402 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 256^2 grid | 11.985 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 256^2 grid | 11.241 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 256^2 grid | 0.377 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 9.372 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 14.898 ms | 0.70 | 7.88 | 69% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 13.478 ms | 0.78 | 8.71 | 76% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 16.037 ms | 0.65 | 7.32 | 64% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 173.078 ms | 0.90 | 6.75 | 59% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 563.914 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 577.510 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 1423.534 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 172.500 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=33554432 | 749.453 ms | 0.09 | 0.36 | 3% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 162.890 ms | 13.18 | 0.10 | 22% | ok | 5.6x (93%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 9.634 ms | 6.97 | 7.40 | 65% | ok | 1.0x (16%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 26.074 ms | 2.57 | 10.29 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.662 ms | 5.30 | 10.60 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 20.409 ms | 3.29 | 3.29 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 20.452 ms | 3.28 | 3.28 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 87.316 ms | 24.59 | 0.14 | n/a | ok | 5.2x (87%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 65.251 ms | 32.91 | 0.13 | n/a | ok | 5.6x (93%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 33.772 ms | 63.59 | 0.19 | n/a | ok | 5.6x (93%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 33.744 ms | 63.64 | 0.19 | n/a | ok | 5.6x (93%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 103.137 ms | 20.82 | 0.06 | n/a | ok | 5.5x (91%) |

