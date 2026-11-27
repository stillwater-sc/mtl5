## i7-12700k -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.3 GFLOP/s |
| bandwidth | read 33.3, update 38.5, triad 32.6 GB/s (ceiling 38.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 43% | 98% | 0 |
| blas2 | 10 | 100% | 109% | 0 |
| blas3 | 10 | 81% | 94% | 0 |
| factor | 12 | 24% | 39% | 0 |
| eigen | 8 | 24% | 71% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 49% | 95% | 0 |
| direct | 31 | 4% | 32% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 21 | 24% | 69% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 62% | 77% | 0 |
| mg | 1 | 57% | 57% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 36% | 84% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.705 ms | 2.94 | 23.53 | 61% | ok |
| blas1 | dot | f32 | n=8388608 | 4.275 ms | 3.92 | 15.70 | 41% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.394 ms | 2.62 | 31.49 | 82% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.704 ms | 3.57 | 21.40 | 56% | ok |
| blas1 | scale | f64 | n=8388608 | 3.571 ms | 2.35 | 37.59 | 98% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.954 ms | 3.39 | 13.55 | 35% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.069 ms | 2.06 | 16.49 | 43% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.070 ms | 1.19 | 9.49 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 4.020 ms | 2.09 | 16.69 | 43% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 76.730 ms | 2.41 | 1.75 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.508 ms | 9.57 | 38.28 | 100% | ok |
| blas2 | gemv | f32 | n=4096 | 1.602 ms | 20.95 | 41.92 | 109% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.621 ms | 9.27 | 37.08 | 96% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.650 ms | 20.33 | 40.68 | 106% | ok |
| blas2 | symv | f64 | n=4096 | 1.922 ms | 17.46 | 34.96 | 91% | ok |
| blas2 | trmv | f64 | n=4096 | 1.859 ms | 9.02 | 36.13 | 94% | ok |
| blas2 | trmv | f32 | n=4096 | 0.946 ms | 17.74 | 35.52 | 92% | ok |
| blas2 | trsv | f64 | n=4096 | 1.796 ms | 9.34 | 37.41 | 97% | ok |
| blas2 | trsv | f32 | n=4096 | 0.872 ms | 19.23 | 38.50 | 100% | ok |
| blas2 | ger | f64 | n=4096 | 6.845 ms | 4.90 | 39.22 | 102% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.705 ms | 72.46 | 1.70 | 93% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.832 ms | 146.51 | 1.72 | 94% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.577 ms | 52.09 | 2.44 | 67% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.713 ms | 72.30 | 1.69 | 92% | ok |
| blas3 | syrk | f64 | 512x512x512 | 2.548 ms | 52.68 | 1.65 | 67% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 4.718 ms | 56.90 | 1.33 | 73% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.107 ms | 63.69 | 2.49 | 81% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.131 ms | 62.99 | 2.46 | 80% | ok |
| blas3 | trsm | f32 | 512x512x512 | 0.961 ms | 139.67 | 2.73 | 89% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.146 ms | 0.40 | 3.23 | 8% | ok |
| factor | lu | f64 | 512x512 | 3.280 ms | 27.28 | 1.28 | 35% | ok |
| factor | lu | f32 | 512x512 | 1.824 ms | 49.06 | 1.15 | 31% | ok |
| factor | cholesky | f64 | 512x512 | 2.052 ms | 21.80 | 2.04 | 28% | ok |
| factor | cholesky | f32 | 512x512 | 1.169 ms | 38.27 | 1.79 | 24% | ok |
| factor | ldlt | f64 | 512x512 | 7.506 ms | 5.96 | 0.56 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 4.952 ms | 9.03 | 0.85 | 12% | ok |
| factor | qr | f64 | 512x512 | 5.891 ms | 30.38 | 0.71 | 39% | ok |
| factor | lq | f64 | 512x512 | 25.490 ms | 7.02 | 0.16 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 269.980 ms | 1.66 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.626 ms | 0.65 | 2.58 | 7%\* | ok |
| factor | inv | f64 | 512x512 | 45.955 ms | 5.84 | 0.09 | 7% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 3.810 ms | 23.49 | 1.10 | 30% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.734 ms | 12.90 | 0.60 | 16% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 59.040 ms | 2.56 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 751.796 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 16.051 ms | 10.45 | 0.07 | 13% | ok |
| eigen | schur | f64 | 256x256 | 20.926 ms | 20.04 | 0.05 | 26% | ok |
| eigen | svd | f64 | 256x256 | 6.682 ms | 55.23 | 0.16 | 71% | ok |
| eigen | expm | f64 | 256x256 | 11.150 ms | 22.57 | 0.09 | 29% | ok |
| eigen | sylvester | f64 | 256x256 | 53.314 ms | 18.88 | 0.03 | 24% | ok |
| elementwise | exp | f64 | n=8388608 | 39.559 ms | 0.21 | 3.39 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 46.283 ms | 0.18 | 2.90 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 60.476 ms | 0.14 | 2.22 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 29.196 ms | 0.29 | 4.60 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 202.323 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.749 ms | 3.50 | 36.37 | 95% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.919 ms | 2.85 | 29.63 | 77% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.657 ms | 8.99 | 18.91 | 49% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.918 ms | 0.66 | 2.34 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.329 ms | 1.51 | 12.52 | 33%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 14.921 ms | 0.75 | 0.05 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 24.972 ms | 0.45 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.391 ms | 1.33 | 0.09 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.103 ms | 0.56 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.699 ms | 0.50 | 0.02 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.556 ms | 1.35 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 24.997 ms | 0.08 | 0.01 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.362 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.985 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 58.116 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.186 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.125 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.096 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 12.079 ms | 0.93 | 0.06 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.248 ms | 3.05 | 12.18 | 32%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.071 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.058 ms | 2.76 | 0.18 | 4% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.252 ms | 3.00 | 12.00 | 31%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.088 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 37.153 ms | 0.60 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.638 ms | 1.18 | 4.74 | 12%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.453 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.563 ms | 0.58 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.634 ms | 1.19 | 4.77 | 12%\* | ok |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 5.729 ms | 1.95 | 0.13 | 2% | ok |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 1.867 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 3.480 ms | 3.21 | 0.21 | 4% | ok |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 0.328 ms | 2.31 | 9.23 | 24%\* | ok |
| direct | ref_pardiso_spd | f64 | 96^2 grid, nnz 45696 | 4.532 ms | 2.47 | 0.16 | 3% | ok |
| direct | ref_pardiso_spd.analyze | f64 | 96^2 grid, nnz 45696 | 2.110 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_pardiso_spd.factor | f64 | 96^2 grid, nnz 45696 | 1.907 ms | 5.87 | 0.38 | 7% | ok |
| direct | ref_pardiso_spd.solve | f64 | 96^2 grid, nnz 45696 | 0.457 ms | 1.65 | 6.61 | 17%\* | ok |
| direct | ref_pardiso_lu | f64 | 96^2 grid, nnz 45696 | 7.446 ms | 3.00 | 0.10 | 4% | ok |
| direct | ref_pardiso_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.561 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_pardiso_lu.factor | f64 | 96^2 grid, nnz 45696 | 3.247 ms | 6.89 | 0.23 | 9% | ok |
| direct | ref_pardiso_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.573 ms | 1.32 | 5.28 | 14%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.942 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 204.533 ms | 1.51 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.450 ms | 2.32 | 9.30 | 24%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.662 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 60.040 ms | 5.14 | 0.01 | 7% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.456 ms | 2.31 | 9.25 | 24%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.877 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 553.107 ms | 1.12 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.521 ms | 0.45 | 1.79 | 5%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.262 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 451.616 ms | 1.37 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.525 ms | 0.45 | 1.79 | 5%\* | ok |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 3.883 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 11.337 ms | 27.22 | 0.08 | 35% | ok |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 0.507 ms | 6.65 | 26.60 | 69%\* | ok |
| direct3d | ref_pardiso_spd.analyze | f64 | 20^3 grid, nnz 53600 | 7.967 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_pardiso_spd.factor | f64 | 20^3 grid, nnz 53600 | 9.703 ms | 31.81 | 0.09 | 41% | ok |
| direct3d | ref_pardiso_spd.solve | f64 | 20^3 grid, nnz 53600 | 0.870 ms | 3.87 | 15.50 | 40%\* | ok |
| direct3d | ref_pardiso_lu.analyze | f64 | 20^3 grid, nnz 53600 | 9.050 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_pardiso_lu.factor | f64 | 20^3 grid, nnz 53600 | 19.178 ms | 32.18 | 0.04 | 41% | ok |
| direct3d | ref_pardiso_lu.solve | f64 | 20^3 grid, nnz 53600 | 1.037 ms | 3.25 | 13.00 | 34%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 31.359 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 48.204 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 23.051 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.633 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 15.177 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 14.181 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.737 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.453 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.381 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 18.253 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.724 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 16.197 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 19.230 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.041 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.537 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 11.671 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.091 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.026 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.023 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.089 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.988 ms | 2.65 | 29.68 | 77% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.221 ms | 2.14 | 24.01 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.564 ms | 1.67 | 18.75 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.286 ms | 2.89 | 21.88 | 57% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 30.662 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.198 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.373 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.978 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.880 ms | 3.44 | 13.75 | 36% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 38.993 ms | 6.88 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.555 ms | 30.25 | 32.14 | 84%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.300 ms | 7.29 | 29.18 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.999 ms | 16.79 | 33.59 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.725 ms | 9.73 | 9.73 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.724 ms | 9.73 | 9.73 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.492 ms | 15.35 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.821 ms | 12.89 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.229 ms | 26.24 | 0.15 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.233 ms | 26.23 | 0.15 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 26.885 ms | 9.98 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

