## i7-12700k -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `8867b44b3f52` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.3 GFLOP/s |
| bandwidth | read 33.1, update 38.5, triad 32.5 GB/s (ceiling 38.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 44% | 98% | 0 |
| blas2 | 10 | 97% | 107% | 0 |
| blas3 | 10 | 81% | 94% | 0 |
| factor | 12 | 24% | 40% | 0 |
| eigen | 8 | 24% | 71% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 48% | 96% | 0 |
| direct | 31 | 3% | 32% | 0 |
| ordering | 4 | n/a | n/a | 0 |
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
| blas1 | dot | f64 | n=8388608 | 5.725 ms | 2.93 | 23.45 | 61% | ok |
| blas1 | dot | f32 | n=8388608 | 4.273 ms | 3.93 | 15.71 | 41% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.407 ms | 2.62 | 31.42 | 82% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.929 ms | 3.40 | 20.42 | 53% | ok |
| blas1 | scale | f64 | n=8388608 | 3.571 ms | 2.35 | 37.58 | 98% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.874 ms | 3.44 | 13.77 | 36% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.010 ms | 2.09 | 16.74 | 43% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.031 ms | 1.19 | 9.54 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 3.974 ms | 2.11 | 16.89 | 44% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 76.344 ms | 2.42 | 1.76 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.492 ms | 9.61 | 38.46 | 100% | ok |
| blas2 | gemv | f32 | n=4096 | 1.647 ms | 20.37 | 40.77 | 106% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.622 ms | 9.26 | 37.08 | 96% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.628 ms | 20.61 | 41.25 | 107% | ok |
| blas2 | symv | f64 | n=4096 | 1.915 ms | 17.52 | 35.08 | 91% | ok |
| blas2 | trmv | f64 | n=4096 | 1.851 ms | 9.06 | 36.29 | 94% | ok |
| blas2 | trmv | f32 | n=4096 | 0.940 ms | 17.86 | 35.75 | 93% | ok |
| blas2 | trsv | f64 | n=4096 | 1.789 ms | 9.38 | 37.55 | 97% | ok |
| blas2 | trsv | f32 | n=4096 | 0.870 ms | 19.28 | 38.60 | 100% | ok |
| blas2 | ger | f64 | n=4096 | 7.682 ms | 4.37 | 34.95 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.700 ms | 72.55 | 1.70 | 93% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.831 ms | 146.64 | 1.72 | 94% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.580 ms | 52.02 | 2.44 | 66% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.709 ms | 72.38 | 1.70 | 92% | ok |
| blas3 | syrk | f64 | 512x512x512 | 2.545 ms | 52.74 | 1.65 | 67% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 4.732 ms | 56.73 | 1.33 | 72% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.109 ms | 63.65 | 2.49 | 81% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.131 ms | 63.00 | 2.46 | 80% | ok |
| blas3 | trsm | f32 | 512x512x512 | 0.963 ms | 139.44 | 2.72 | 89% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.099 ms | 0.41 | 3.24 | 8% | ok |
| factor | lu | f64 | 512x512 | 3.304 ms | 27.08 | 1.27 | 35% | ok |
| factor | lu | f32 | 512x512 | 1.863 ms | 48.04 | 1.13 | 31% | ok |
| factor | cholesky | f64 | 512x512 | 2.039 ms | 21.94 | 2.06 | 28% | ok |
| factor | cholesky | f32 | 512x512 | 1.188 ms | 37.66 | 1.77 | 24% | ok |
| factor | ldlt | f64 | 512x512 | 7.531 ms | 5.94 | 0.56 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 46.453 ms | 0.96 | 0.09 | 1% | ok |
| factor | qr | f64 | 512x512 | 5.767 ms | 31.03 | 0.73 | 40% | ok |
| factor | lq | f64 | 512x512 | 25.575 ms | 7.00 | 0.16 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 268.326 ms | 1.67 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.640 ms | 0.64 | 2.56 | 7%\* | ok |
| factor | inv | f64 | 512x512 | 45.689 ms | 5.88 | 0.09 | 8% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 3.826 ms | 23.39 | 1.10 | 30% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.699 ms | 13.16 | 0.62 | 17% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 58.911 ms | 2.56 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 745.632 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 16.480 ms | 10.18 | 0.06 | 13% | ok |
| eigen | schur | f64 | 256x256 | 21.027 ms | 19.95 | 0.05 | 25% | ok |
| eigen | svd | f64 | 256x256 | 6.652 ms | 55.48 | 0.16 | 71% | ok |
| eigen | expm | f64 | 256x256 | 11.209 ms | 22.45 | 0.09 | 29% | ok |
| eigen | sylvester | f64 | 256x256 | 53.349 ms | 18.87 | 0.03 | 24% | ok |
| elementwise | exp | f64 | n=8388608 | 39.232 ms | 0.21 | 3.42 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 45.474 ms | 0.18 | 2.95 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 59.953 ms | 0.14 | 2.24 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 28.816 ms | 0.29 | 4.66 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 201.453 ms | 0.04 | 0.67 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.740 ms | 3.54 | 36.82 | 96% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.928 ms | 2.82 | 29.35 | 76% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.751 ms | 8.81 | 18.53 | 48% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.865 ms | 0.67 | 2.36 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.251 ms | 1.60 | 13.31 | 35%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 91.124 ms | 0.12 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.477 ms | 0.44 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.940 ms | 1.25 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.414 ms | 0.57 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 53.590 ms | 0.42 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.821 ms | 1.33 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 99.895 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.340 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.988 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.225 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.096 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.354 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 88.464 ms | 0.13 | 0.01 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.246 ms | 3.07 | 12.28 | 32%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.513 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.203 ms | 2.66 | 0.17 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.248 ms | 3.05 | 12.19 | 32%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.110 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.234 ms | 0.63 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.634 ms | 1.19 | 4.77 | 12%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.926 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.738 ms | 0.47 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.628 ms | 1.20 | 4.82 | 13%\* | ok |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 5.649 ms | 1.98 | 0.13 | 3% | ok |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 1.894 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 3.403 ms | 3.29 | 0.21 | 4% | ok |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 0.327 ms | 2.31 | 9.24 | 24%\* | ok |
| direct | ref_pardiso_spd | f64 | 96^2 grid, nnz 45696 | 4.527 ms | 2.47 | 0.16 | 3% | ok |
| direct | ref_pardiso_spd.analyze | f64 | 96^2 grid, nnz 45696 | 2.109 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_pardiso_spd.factor | f64 | 96^2 grid, nnz 45696 | 1.909 ms | 5.86 | 0.38 | 7% | ok |
| direct | ref_pardiso_spd.solve | f64 | 96^2 grid, nnz 45696 | 0.458 ms | 1.65 | 6.61 | 17%\* | ok |
| direct | ref_pardiso_lu | f64 | 96^2 grid, nnz 45696 | 7.444 ms | 3.01 | 0.10 | 4% | ok |
| direct | ref_pardiso_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.562 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_pardiso_lu.factor | f64 | 96^2 grid, nnz 45696 | 3.244 ms | 6.90 | 0.23 | 9% | ok |
| direct | ref_pardiso_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.577 ms | 1.31 | 5.24 | 14%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 9.482 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 7435.072 ms | 0.04 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.438 ms | 2.34 | 9.37 | 24%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 18.667 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.422 ms | 4.45 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.445 ms | 2.33 | 9.33 | 24%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.891 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 551.473 ms | 1.12 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.511 ms | 0.45 | 1.79 | 5%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.729 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 665.388 ms | 0.93 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.513 ms | 0.45 | 1.79 | 5%\* | ok |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 3.890 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 11.428 ms | 27.00 | 0.08 | 34% | ok |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 0.507 ms | 6.64 | 26.57 | 69%\* | ok |
| direct3d | ref_pardiso_spd.analyze | f64 | 20^3 grid, nnz 53600 | 7.969 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_pardiso_spd.factor | f64 | 20^3 grid, nnz 53600 | 9.684 ms | 31.87 | 0.09 | 41% | ok |
| direct3d | ref_pardiso_spd.solve | f64 | 20^3 grid, nnz 53600 | 0.868 ms | 3.88 | 15.53 | 40%\* | ok |
| direct3d | ref_pardiso_lu.analyze | f64 | 20^3 grid, nnz 53600 | 9.054 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_pardiso_lu.factor | f64 | 20^3 grid, nnz 53600 | 19.083 ms | 32.34 | 0.04 | 41% | ok |
| direct3d | ref_pardiso_lu.solve | f64 | 20^3 grid, nnz 53600 | 1.038 ms | 3.24 | 12.98 | 34%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 31.154 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 32.942 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.933 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.499 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.997 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 14.013 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.979 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.242 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.254 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 17.228 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.513 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.919 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.252 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.034 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.604 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.306 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.077 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.016 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.023 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.371 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.982 ms | 2.66 | 29.85 | 77% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.219 ms | 2.15 | 24.06 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.563 ms | 1.67 | 18.76 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.216 ms | 2.91 | 21.99 | 57% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 30.538 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.806 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.477 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 5.028 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.859 ms | 3.45 | 13.81 | 36% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 38.978 ms | 6.89 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.554 ms | 30.30 | 32.20 | 84%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.308 ms | 7.27 | 29.08 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.991 ms | 16.92 | 33.85 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.721 ms | 9.75 | 9.75 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.720 ms | 9.76 | 9.76 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.419 ms | 15.41 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.803 ms | 12.90 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.224 ms | 26.26 | 0.15 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.222 ms | 26.26 | 0.15 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 26.791 ms | 10.02 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

