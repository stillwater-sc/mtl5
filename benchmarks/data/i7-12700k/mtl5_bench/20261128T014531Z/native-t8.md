## i7-12700k -- native, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 600.0 GFLOP/s, f32 1200.3 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.0, f32 1200.3 GFLOP/s |
| ... this build, scalar | f64 9.9, f32 10.0 GFLOP/s |
| bandwidth | read 83.1, update 78.0, triad 76.8 GB/s (ceiling 83.1) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 89% | 125% | 0 | 29% |
| blas2 | 10 | 42% | 134% | 0 | 13% |
| blas3 | 10 | 2% | 17% | 0 | 14% |
| factor | 12 | 2% | 12% | 0 | 14% |
| eigen | 8 | 1% | 4% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 290% | 0 | 13% |
| direct | 19 | 0% | 14% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 10% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 36% | 0 | 13% |
| mg | 1 | 26% | 26% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 39% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 94% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.401 ms | 11.97 | 95.78 | 115% | ok | 4.0x (50%) |
| blas1 | dot | f32 | n=8388608 | 0.648 ms | 25.89 | 103.57 | 125% | ok | 6.5x (82%) |
| blas1 | axpy | f64 | n=8388608 | 2.765 ms | 6.07 | 72.82 | 88% | ok | 2.3x (29%) |
| blas1 | axpy | f32 | n=8388608 | 1.364 ms | 12.30 | 73.78 | 89% | ok | 3.5x (43%) |
| blas1 | scale | f64 | n=8388608 | 1.776 ms | 4.72 | 75.57 | 91% | ok | 2.0x (25%) |
| blas1 | two_norm | f64 | n=8388608 | 0.748 ms | 22.42 | 89.67 | 108% | ok | 6.5x (82%) |
| blas1 | one_norm | f64 | n=8388608 | 4.011 ms | 2.09 | 16.73 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.953 ms | 1.21 | 9.65 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 4.008 ms | 2.09 | 16.74 | 20% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 76.351 ms | 2.42 | 1.76 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.506 ms | 22.29 | 89.19 | 107% | ok | 2.5x (31%) |
| blas2 | gemv | f32 | n=4096 | 0.602 ms | 55.72 | 111.49 | 134% | ok | 4.6x (57%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.571 ms | 21.36 | 85.48 | 103% | ok | 3.0x (38%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.856 ms | 39.19 | 78.41 | 94% | ok | 2.6x (32%) |
| blas2 | symv | f64 | n=4096 | 3.148 ms | 10.66 | 21.34 | 26% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 2.015 ms | 8.33 | 33.34 | 40% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=4096 | 1.417 ms | 11.84 | 23.70 | 29% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 1.969 ms | 8.52 | 34.12 | 41% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.375 ms | 12.20 | 24.42 | 29% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.677 ms | 4.37 | 34.98 | 42% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 2.659 ms | 100.95 | 2.37 | 17% | ok | 7.6x (94%) |
| blas3 | gemm | f32 | 512x512x512 | 1.359 ms | 197.55 | 2.32 | 16% | ok | 7.6x (95%) |
| blas3 | gemmt | f64 | 512x512x512 | 10.952 ms | 12.26 | 0.57 | 2% | ok | 1.1x (14%) |
| blas3 | symm | f64 | 512x512x512 | 19.083 ms | 14.07 | 0.33 | 2% | ok | 1.0x (13%) |
| blas3 | syrk | f64 | 512x512x512 | 12.000 ms | 11.18 | 0.35 | 2% | ok | 1.1x (14%) |
| blas3 | syr2k | f64 | 512x512x512 | 20.171 ms | 13.31 | 0.31 | 2% | ok | 1.2x (15%) |
| blas3 | trmm | f64 | 512x512x512 | 9.288 ms | 14.45 | 0.56 | 2% | ok | 1.0x (13%) |
| blas3 | trsm | f64 | 512x512x512 | 9.417 ms | 14.25 | 0.56 | 2% | ok | 1.0x (13%) |
| blas3 | trsm | f32 | 512x512x512 | 5.135 ms | 26.14 | 0.51 | 2% | ok | 1.0x (13%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.010 ms | 0.41 | 3.26 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 6.881 ms | 13.00 | 0.61 | 2% | ok | 1.7x (21%) |
| factor | lu | f32 | 512x512 | 3.729 ms | 24.00 | 0.56 | 2% | ok | 1.2x (14%) |
| factor | cholesky | f64 | 512x512 | 7.142 ms | 6.26 | 0.59 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 6.473 ms | 6.91 | 0.32 | 1% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 7.230 ms | 6.19 | 0.58 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 4.678 ms | 9.56 | 0.90 | 2% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 129.316 ms | 1.38 | 0.03 | 0% | ok | 1.8x (22%) |
| factor | lq | f64 | 512x512 | 16.675 ms | 10.73 | 0.25 | 2% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 157.873 ms | 2.83 | 0.03 | 0% | ok | 1.7x (21%) |
| factor | householder | f64 | 512x512 | 0.407 ms | 2.58 | 10.31 | 12%\* | ok | 4.0x (50%) |
| factor | inv | f64 | 512x512 | 48.301 ms | 5.56 | 0.09 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 5.566 ms | 16.08 | 0.75 | 3% | ok | 1.1x (14%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 18.045 ms | 1.24 | 0.06 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 57.941 ms | 2.61 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 740.111 ms | 0.20 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 31.585 ms | 5.31 | 0.03 | 1% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 79.611 ms | 5.27 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 108.406 ms | 3.40 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 10.770 ms | 23.37 | 0.10 | 4% | ok | 2.4x (30%) |
| eigen | sylvester | f64 | 256x256 | 176.456 ms | 5.70 | 0.01 | 1% | ok | 1.1x (13%) |
| elementwise | exp | f64 | n=8388608 | 38.750 ms | 0.22 | 3.46 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 45.137 ms | 0.19 | 2.97 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 60.566 ms | 0.14 | 2.22 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.500 ms | 0.29 | 4.71 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 203.041 ms | 0.04 | 0.66 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.113 ms | 23.17 | 241.07 | 290% ! | ok | 6.6x (83%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.904 ms | 2.89 | 30.11 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.310 ms | 31.96 | 67.21 | 81% | ok | 3.6x (45%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 5.026 ms | 0.65 | 2.29 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.528 ms | 1.31 | 10.89 | 13%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 14.372 ms | 0.78 | 0.05 | 0% | ok | 1.1x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 24.602 ms | 0.45 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.407 ms | 1.33 | 0.09 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.286 ms | 0.56 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 43.820 ms | 0.51 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.346 ms | 1.37 | 0.04 | 0% | ok | 1.1x (14%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 24.300 ms | 0.08 | 0.01 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.292 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.852 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 57.027 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.181 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.949 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.059 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 12.100 ms | 0.92 | 0.06 | 0% | ok | 1.1x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.254 ms | 2.97 | 11.89 | 14%\* | ok | 1.0x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 3.980 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 3.974 ms | 2.82 | 0.18 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.258 ms | 2.93 | 11.73 | 14%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.026 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 36.913 ms | 0.61 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.616 ms | 1.23 | 4.91 | 6%\* | ok | 1.3x (16%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.356 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.263 ms | 0.58 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.616 ms | 1.23 | 4.91 | 6%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.789 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 196.270 ms | 1.57 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.628 ms | 2.07 | 8.28 | 10%\* | ok | 0.9x (11%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.416 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 58.814 ms | 5.25 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.651 ms | 2.04 | 8.16 | 10%\* | ok | 0.9x (11%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.778 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 546.760 ms | 1.13 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.710 ms | 0.44 | 1.75 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.092 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 462.749 ms | 1.33 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.813 ms | 0.43 | 1.72 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.590 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 33.915 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.608 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.860 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.780 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.843 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.157 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.005 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.992 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 17.856 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.346 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 15.899 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 18.655 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.039 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.494 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 11.219 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.064 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.996 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.026 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.711 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.977 ms | 2.68 | 30.02 | 36% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.217 ms | 2.15 | 24.11 | 29% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.537 ms | 1.70 | 19.09 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.568 ms | 2.83 | 21.42 | 26% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.797 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 30.624 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 32.836 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.910 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 4.813 ms | 3.49 | 13.94 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 5.126 ms | 52.37 | 0.82 | 9% | ok | 7.6x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.554 ms | 30.30 | 32.19 | 39%\* | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.283 ms | 7.35 | 29.40 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.973 ms | 17.25 | 34.49 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.688 ms | 9.94 | 9.94 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.687 ms | 9.95 | 9.95 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.287 ms | 117.38 | 1.38 | n/a | ok | 7.7x (96%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.746 ms | 97.77 | 0.76 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.355 ms | 198.05 | 1.16 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.355 ms | 198.18 | 1.16 | n/a | ok | 7.6x (94%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.541 ms | 75.81 | 0.44 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

