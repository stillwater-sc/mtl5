## i7-12700k -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.3 GFLOP/s |
| bandwidth | read 32.2, update 38.4, triad 32.4 GB/s (ceiling 38.4) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 44% | 98% | 0 |
| blas2 | 10 | 100% | 107% | 0 |
| blas3 | 10 | 81% | 94% | 0 |
| factor | 12 | 24% | 40% | 0 |
| eigen | 8 | 24% | 71% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 49% | 95% | 0 |
| direct | 31 | 4% | 32% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 21 | 24% | 69% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 63% | 77% | 0 |
| mg | 1 | 57% | 57% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 36% | 84% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.768 ms | 2.91 | 23.27 | 61% | ok |
| blas1 | dot | f32 | n=8388608 | 4.291 ms | 3.91 | 15.64 | 41% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.526 ms | 2.57 | 30.85 | 80% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.715 ms | 3.56 | 21.35 | 56% | ok |
| blas1 | scale | f64 | n=8388608 | 3.577 ms | 2.35 | 37.52 | 98% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.895 ms | 3.43 | 13.71 | 36% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.016 ms | 2.09 | 16.71 | 44% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.039 ms | 1.19 | 9.53 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 3.983 ms | 2.11 | 16.85 | 44% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 76.970 ms | 2.40 | 1.74 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.500 ms | 9.59 | 38.37 | 100% | ok |
| blas2 | gemv | f32 | n=4096 | 1.657 ms | 20.25 | 40.53 | 106% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.637 ms | 9.22 | 36.92 | 96% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.634 ms | 20.53 | 41.08 | 107% | ok |
| blas2 | symv | f64 | n=4096 | 1.922 ms | 17.45 | 34.94 | 91% | ok |
| blas2 | trmv | f64 | n=4096 | 1.861 ms | 9.02 | 36.10 | 94% | ok |
| blas2 | trmv | f32 | n=4096 | 0.947 ms | 17.72 | 35.48 | 92% | ok |
| blas2 | trsv | f64 | n=4096 | 1.797 ms | 9.34 | 37.39 | 97% | ok |
| blas2 | trsv | f32 | n=4096 | 0.870 ms | 19.29 | 38.62 | 101% | ok |
| blas2 | ger | f64 | n=4096 | 6.829 ms | 4.91 | 39.32 | 102% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.710 ms | 72.35 | 1.70 | 92% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.833 ms | 146.44 | 1.72 | 94% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.569 ms | 52.24 | 2.45 | 67% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.715 ms | 72.26 | 1.69 | 92% | ok |
| blas3 | syrk | f64 | 512x512x512 | 2.543 ms | 52.77 | 1.65 | 67% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 4.716 ms | 56.92 | 1.33 | 73% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.107 ms | 63.71 | 2.49 | 81% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.134 ms | 62.89 | 2.46 | 80% | ok |
| blas3 | trsm | f32 | 512x512x512 | 0.962 ms | 139.49 | 2.72 | 89% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.270 ms | 0.40 | 3.20 | 8% | ok |
| factor | lu | f64 | 512x512 | 3.278 ms | 27.29 | 1.28 | 35% | ok |
| factor | lu | f32 | 512x512 | 1.852 ms | 48.32 | 1.13 | 31% | ok |
| factor | cholesky | f64 | 512x512 | 2.044 ms | 21.89 | 2.05 | 28% | ok |
| factor | cholesky | f32 | 512x512 | 1.204 ms | 37.16 | 1.74 | 24% | ok |
| factor | ldlt | f64 | 512x512 | 7.516 ms | 5.95 | 0.56 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 4.893 ms | 9.14 | 0.86 | 12% | ok |
| factor | qr | f64 | 512x512 | 5.739 ms | 31.18 | 0.73 | 40% | ok |
| factor | lq | f64 | 512x512 | 26.043 ms | 6.87 | 0.16 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 267.971 ms | 1.67 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.605 ms | 0.65 | 2.61 | 7%\* | ok |
| factor | inv | f64 | 512x512 | 46.081 ms | 5.83 | 0.09 | 7% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 3.790 ms | 23.61 | 1.11 | 30% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.735 ms | 12.89 | 0.60 | 16% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 58.842 ms | 2.57 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 755.357 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 16.092 ms | 10.43 | 0.07 | 13% | ok |
| eigen | schur | f64 | 256x256 | 21.073 ms | 19.90 | 0.05 | 25% | ok |
| eigen | svd | f64 | 256x256 | 6.676 ms | 55.29 | 0.16 | 71% | ok |
| eigen | expm | f64 | 256x256 | 11.185 ms | 22.50 | 0.09 | 29% | ok |
| eigen | sylvester | f64 | 256x256 | 52.932 ms | 19.02 | 0.03 | 24% | ok |
| elementwise | exp | f64 | n=8388608 | 39.386 ms | 0.21 | 3.41 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 46.245 ms | 0.18 | 2.90 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 60.125 ms | 0.14 | 2.23 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 29.013 ms | 0.29 | 4.63 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 202.302 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.744 ms | 3.52 | 36.60 | 95% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.925 ms | 2.83 | 29.44 | 77% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.654 ms | 9.00 | 18.92 | 49% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.856 ms | 0.67 | 2.37 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.313 ms | 1.53 | 12.68 | 33%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 14.514 ms | 0.77 | 0.05 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.008 ms | 0.45 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.436 ms | 1.33 | 0.09 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.772 ms | 0.55 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.270 ms | 0.51 | 0.02 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.727 ms | 1.34 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 24.980 ms | 0.08 | 0.01 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.345 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.012 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 58.214 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.186 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.227 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.084 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 12.751 ms | 0.88 | 0.06 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.246 ms | 3.08 | 12.31 | 32%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.077 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.059 ms | 2.76 | 0.18 | 4% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.249 ms | 3.04 | 12.14 | 32%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.068 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 36.573 ms | 0.61 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.638 ms | 1.19 | 4.74 | 12%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.432 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.396 ms | 0.58 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.642 ms | 1.18 | 4.71 | 12%\* | ok |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 5.728 ms | 1.95 | 0.13 | 2% | ok |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 1.879 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 3.481 ms | 3.21 | 0.21 | 4% | ok |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 0.327 ms | 2.31 | 9.24 | 24%\* | ok |
| direct | ref_pardiso_spd | f64 | 96^2 grid, nnz 45696 | 4.545 ms | 2.46 | 0.16 | 3% | ok |
| direct | ref_pardiso_spd.analyze | f64 | 96^2 grid, nnz 45696 | 2.121 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_pardiso_spd.factor | f64 | 96^2 grid, nnz 45696 | 1.912 ms | 5.85 | 0.38 | 7% | ok |
| direct | ref_pardiso_spd.solve | f64 | 96^2 grid, nnz 45696 | 0.459 ms | 1.65 | 6.59 | 17%\* | ok |
| direct | ref_pardiso_lu | f64 | 96^2 grid, nnz 45696 | 7.454 ms | 3.00 | 0.10 | 4% | ok |
| direct | ref_pardiso_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.663 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_pardiso_lu.factor | f64 | 96^2 grid, nnz 45696 | 3.239 ms | 6.91 | 0.23 | 9% | ok |
| direct | ref_pardiso_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.579 ms | 1.31 | 5.22 | 14%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.897 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 207.295 ms | 1.49 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.447 ms | 2.33 | 9.31 | 24%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.671 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 59.760 ms | 5.16 | 0.01 | 7% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.466 ms | 2.30 | 9.19 | 24%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.884 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 559.974 ms | 1.10 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.537 ms | 0.45 | 1.79 | 5%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.247 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 452.239 ms | 1.36 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.540 ms | 0.45 | 1.79 | 5%\* | ok |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 3.894 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 11.557 ms | 26.70 | 0.07 | 34% | ok |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 0.508 ms | 6.63 | 26.53 | 69%\* | ok |
| direct3d | ref_pardiso_spd.analyze | f64 | 20^3 grid, nnz 53600 | 7.964 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_pardiso_spd.factor | f64 | 20^3 grid, nnz 53600 | 9.689 ms | 31.85 | 0.09 | 41% | ok |
| direct3d | ref_pardiso_spd.solve | f64 | 20^3 grid, nnz 53600 | 0.871 ms | 3.87 | 15.47 | 40%\* | ok |
| direct3d | ref_pardiso_lu.analyze | f64 | 20^3 grid, nnz 53600 | 9.038 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_pardiso_lu.factor | f64 | 20^3 grid, nnz 53600 | 19.170 ms | 32.19 | 0.04 | 41% | ok |
| direct3d | ref_pardiso_lu.solve | f64 | 20^3 grid, nnz 53600 | 1.033 ms | 3.26 | 13.05 | 34%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 31.323 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 34.956 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 23.187 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.613 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 15.156 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 14.148 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.703 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.417 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.340 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 18.292 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.709 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 16.246 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 19.084 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.041 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.540 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 11.613 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.087 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.025 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.028 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.149 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.986 ms | 2.65 | 29.73 | 77% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.221 ms | 2.14 | 24.02 | 63% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.563 ms | 1.67 | 18.77 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.348 ms | 2.88 | 21.78 | 57% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 30.508 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.002 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.264 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.949 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.863 ms | 3.45 | 13.80 | 36% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 38.986 ms | 6.89 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.555 ms | 30.25 | 32.14 | 84%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.304 ms | 7.28 | 29.13 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.001 ms | 16.76 | 33.53 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.722 ms | 9.74 | 9.74 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.722 ms | 9.74 | 9.74 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.422 ms | 15.41 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.831 ms | 12.89 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.231 ms | 26.24 | 0.15 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.228 ms | 26.25 | 0.15 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 26.837 ms | 10.00 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

