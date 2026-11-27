## xeon-e5-2420v2 -- blas, 6 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 59.0 GFLOP/s, f32 118.2 GFLOP/s |
| ... hardware, SSE2 | f64 59.0, f32 118.2 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 2.5 GFLOP/s |
| bandwidth | read 11.4, update 10.6, triad 8.5 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 53% | 107% | 0 | 83% |
| blas2 | 10 | 101% | 108% | 0 | 20% |
| blas3 | 10 | 129% | 158% | 0 | 77% |
| factor | 12 | 18% | 40% | 0 | 25% |
| eigen | 8 | 9% | 23% | 0 | 17% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 68% | 105% | 0 | 17% |
| direct | 19 | 1% | 31% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 4% | 17% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 74% | 83% | 0 | 17% |
| mg | 1 | 72% | 72% | 0 | 18% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 22% | 68% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 83% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 32.950 ms | 0.51 | 4.07 | 36% | ok | 6.0x (101%) |
| blas1 | dot | f32 | n=8388608 | 5.491 ms | 3.06 | 12.22 | 107% | ok | 5.0x (83%) |
| blas1 | axpy | f64 | n=8388608 | 38.410 ms | 0.44 | 5.24 | 46% | ok | 5.5x (92%) |
| blas1 | axpy | f32 | n=8388608 | 16.739 ms | 1.00 | 6.01 | 53% | ok | 5.6x (93%) |
| blas1 | scale | f64 | n=8388608 | 12.036 ms | 0.70 | 11.15 | 98% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 32.980 ms | 0.51 | 2.03 | 18% | ok | 5.9x (98%) |
| blas1 | one_norm | f64 | n=8388608 | 9.734 ms | 0.86 | 6.89 | 61% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.095 ms | 0.64 | 5.12 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.761 ms | 0.86 | 6.88 | 60% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 278.191 ms | 0.66 | 0.48 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 11.313 ms | 2.97 | 11.87 | 104% | ok | 1.3x (21%) |
| blas2 | gemv | f32 | n=4096 | 5.473 ms | 6.13 | 12.27 | 108% | ok | 1.1x (19%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 10.924 ms | 3.07 | 12.29 | 108% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.690 ms | 5.90 | 11.80 | 104% | ok | 1.2x (19%) |
| blas2 | symv | f64 | n=4096 | 5.829 ms | 5.76 | 11.52 | 101% | ok | 1.2x (20%) |
| blas2 | trmv | f64 | n=4096 | 5.927 ms | 2.83 | 11.33 | 100% | ok | 1.3x (21%) |
| blas2 | trmv | f32 | n=4096 | 3.019 ms | 5.56 | 11.12 | 98% | ok | 1.2x (21%) |
| blas2 | trsv | f64 | n=4096 | 7.785 ms | 2.16 | 8.63 | 76% | ok | 1.0x (16%) |
| blas2 | trsv | f32 | n=4096 | 3.376 ms | 4.97 | 9.95 | 87% | ok | 1.1x (18%) |
| blas2 | ger | f64 | n=4096 | 26.058 ms | 1.29 | 10.30 | 91% | ok | 1.1x (18%) |
| blas3 | gemm | f64 | 512x512x512 | 2.876 ms | 93.33 | 2.19 | 158% | ok | 4.9x (82%) |
| blas3 | gemm | f32 | 512x512x512 | 1.738 ms | 154.48 | 1.81 | 131% | ok | 4.6x (77%) |
| blas3 | gemmt | f64 | 512x512x512 | 2.532 ms | 53.01 | 2.48 | 90% | ok | 3.5x (58%) |
| blas3 | symm | f64 | 512x512x512 | 2.877 ms | 93.31 | 2.19 | 158% | ok | 4.9x (82%) |
| blas3 | syrk | f64 | 512x512x512 | 2.508 ms | 53.52 | 1.67 | 91% | ok | 3.1x (52%) |
| blas3 | syr2k | f64 | 512x512x512 | 3.731 ms | 71.95 | 1.69 | 122% | ok | 4.0x (67%) |
| blas3 | trmm | f64 | 512x512x512 | 1.617 ms | 83.03 | 3.24 | 141% | ok | 4.7x (78%) |
| blas3 | trsm | f64 | 512x512x512 | 1.761 ms | 76.21 | 2.98 | 129% | ok | 4.6x (77%) |
| blas3 | trsm | f32 | 512x512x512 | 1.238 ms | 108.44 | 2.12 | 92% | ok | 4.3x (71%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.230 ms | 0.14 | 1.11 | 10% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 5.524 ms | 16.20 | 0.76 | 27% | ok | 1.7x (28%) |
| factor | lu | f32 | 512x512 | 3.810 ms | 23.49 | 0.55 | 20% | ok | 1.7x (28%) |
| factor | cholesky | f64 | 512x512 | 4.101 ms | 10.91 | 1.02 | 18% | ok | 1.4x (24%) |
| factor | cholesky | f32 | 512x512 | 3.022 ms | 14.81 | 0.69 | 13% | ok | 1.3x (21%) |
| factor | ldlt | f64 | 512x512 | 25.511 ms | 1.75 | 0.16 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 13.476 ms | 3.32 | 0.31 | 6% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 13.387 ms | 13.37 | 0.31 | 23% | ok | 1.5x (25%) |
| factor | lq | f64 | 512x512 | 49.665 ms | 3.60 | 0.08 | 6% | ok | 1.5x (25%) |
| factor | hessenberg | f64 | 512x512 | 406.258 ms | 1.10 | 0.01 | 2% | ok | 1.5x (25%) |
| factor | householder | f64 | 512x512 | 0.915 ms | 1.15 | 4.59 | 40%\* | ok | 3.6x (59%) |
| factor | inv | f64 | 512x512 | 134.334 ms | 2.00 | 0.03 | 3% | ok | 1.0x (16%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 8.425 ms | 10.62 | 0.50 | 18% | ok | 1.5x (25%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 5.807 ms | 3.85 | 0.18 | 7% | ok | 1.1x (18%) |
| eigen | eigen_symmetric | f64 | 256x256 | 173.146 ms | 0.87 | 0.01 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1016.530 ms | 0.15 | 0.00 | 0% | ok | 1.0x (16%) |
| eigen | eigenvalue | f64 | 256x256 | 57.597 ms | 2.91 | 0.02 | 5% | ok | 0.9x (16%) |
| eigen | schur | f64 | 256x256 | 78.037 ms | 5.37 | 0.01 | 9% | ok | 0.9x (15%) |
| eigen | svd | f64 | 256x256 | 26.866 ms | 13.74 | 0.04 | 23% | ok | 1.1x (18%) |
| eigen | expm | f64 | 256x256 | 27.085 ms | 9.29 | 0.04 | 16% | ok | 1.3x (22%) |
| eigen | sylvester | f64 | 256x256 | 196.842 ms | 5.11 | 0.01 | 9% | ok | 1.0x (16%) |
| elementwise | exp | f64 | n=8388608 | 151.615 ms | 0.06 | 0.89 | 8% | ok | 1.1x (18%) |
| elementwise | log | f64 | n=8388608 | 139.619 ms | 0.06 | 0.96 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 174.750 ms | 0.05 | 0.77 | 7% | ok | 1.0x (16%) |
| elementwise | sqrt | f64 | n=8388608 | 89.523 ms | 0.09 | 1.50 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 2476.815 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.281 ms | 1.15 | 11.94 | 105% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.139 ms | 0.83 | 8.68 | 76% | ok | 1.0x (16%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.332 ms | 3.70 | 7.77 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.307 ms | 0.20 | 0.71 | 6%\* | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.943 ms | 0.34 | 2.80 | 25% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 40.583 ms | 0.28 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 68.805 ms | 0.16 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.406 ms | 0.42 | 0.03 | 1% | ok | 0.9x (16%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 117.664 ms | 0.19 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 131.719 ms | 0.17 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 57.437 ms | 0.39 | 0.01 | 1% | ok | 0.8x (14%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 82.619 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.942 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.152 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 124.832 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.538 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.807 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.206 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.335 ms | 0.34 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.856 ms | 0.88 | 3.53 | 31%\* | ok | 1.0x (16%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.618 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.542 ms | 0.89 | 0.06 | 2% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.875 ms | 0.86 | 3.46 | 30%\* | ok | 1.0x (16%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.354 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 105.132 ms | 0.21 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.873 ms | 0.20 | 0.78 | 7%\* | ok | 1.0x (16%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.812 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 116.375 ms | 0.19 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.840 ms | 0.20 | 0.79 | 7%\* | ok | 1.0x (16%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 12.898 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 635.465 ms | 0.49 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.030 ms | 0.48 | 1.92 | 17%\* | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.747 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 208.593 ms | 1.48 | 0.00 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.031 ms | 0.48 | 1.92 | 17%\* | ok | 1.0x (16%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.517 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1739.581 ms | 0.35 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.796 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 17.882 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1533.475 ms | 0.40 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.777 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 231.958 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 599.331 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 127.833 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 105.366 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 127.466 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 79.960 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 528.316 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 175.244 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.363 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 205.008 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 193.285 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 213.188 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 306.791 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.172 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.340 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 33.249 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 2.895 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 2.760 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.081 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 3.395 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.494 ms | 0.75 | 8.39 | 74% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.097 ms | 0.85 | 9.47 | 83% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.793 ms | 0.69 | 7.73 | 68% | ok | 1.1x (18%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.387 ms | 1.09 | 8.21 | 72% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 140.984 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 148.388 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 129.715 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.373 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 188.239 ms | 0.09 | 0.36 | 3% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 20.507 ms | 13.09 | 0.20 | 22% | ok | 5.6x (94%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.321 ms | 7.23 | 7.68 | 68% | ok | 1.0x (16%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.480 ms | 2.59 | 10.36 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.126 ms | 5.37 | 10.73 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.983 ms | 3.37 | 3.37 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.981 ms | 3.37 | 3.37 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.523 ms | 23.30 | 0.27 | n/a | ok | 5.0x (83%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 8.208 ms | 32.70 | 0.26 | n/a | ok | 5.5x (92%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.317 ms | 62.18 | 0.36 | n/a | ok | 5.5x (91%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.320 ms | 62.14 | 0.36 | n/a | ok | 5.5x (91%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.070 ms | 20.54 | 0.12 | n/a | ok | 5.6x (94%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

