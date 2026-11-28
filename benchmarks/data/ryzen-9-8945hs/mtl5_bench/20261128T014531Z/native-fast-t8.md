## ryzen-9-8945hs -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 504.8 GFLOP/s, f32 1007.3 GFLOP/s |
| ... hardware, AVX-512F | f64 504.8, f32 1007.3 GFLOP/s |
| ... this build, SSE2 | f64 319.9, f32 639.6 GFLOP/s |
| bandwidth | read 52.5, update 54.0, triad 40.6 GB/s (ceiling 54.0) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 110% | 135% | 0 | 14% |
| blas2 | 10 | 65% | 129% | 0 | 15% |
| blas3 | 10 | 8% | 19% | 0 | 37% |
| factor | 12 | 1% | 6% | 0 | 20% |
| eigen | 8 | 0% | 2% | 0 | 13% |
| elementwise | 5 | 7% | 10% | 0 | 13% |
| sparse | 5 | 16% | 283% | 0 | 13% |
| direct | 19 | 0% | 12% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 11% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 28% | 30% | 0 | 12% |
| mg | 1 | 29% | 29% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 57% | 57% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 74% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.260 ms | 7.42 | 59.39 | 110% | ok | 1.1x (14%) |
| blas1 | dot | f32 | n=8388608 | 0.921 ms | 18.21 | 72.83 | 135% | ok | 1.4x (17%) |
| blas1 | axpy | f64 | n=8388608 | 3.446 ms | 4.87 | 58.42 | 108% | ok | 1.1x (13%) |
| blas1 | axpy | f32 | n=8388608 | 1.598 ms | 10.50 | 62.98 | 117% | ok | 1.1x (14%) |
| blas1 | scale | f64 | n=8388608 | 2.048 ms | 4.10 | 65.54 | 121% | ok | 1.1x (14%) |
| blas1 | two_norm | f64 | n=8388608 | 0.920 ms | 18.23 | 72.90 | 135% | ok | 1.3x (16%) |
| blas1 | one_norm | f64 | n=8388608 | 6.399 ms | 1.31 | 10.49 | 19% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.411 ms | 1.31 | 10.47 | 19% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 6.401 ms | 1.31 | 10.48 | 19% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 101.225 ms | 1.82 | 1.33 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 2.311 ms | 14.52 | 58.11 | 108% | ok | 1.5x (19%) |
| blas2 | gemv | f32 | n=4096 | 0.962 ms | 34.90 | 69.83 | 129% | ok | 1.9x (23%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.339 ms | 14.35 | 57.41 | 106% | ok | 1.8x (23%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.182 ms | 28.39 | 56.81 | 105% | ok | 2.0x (24%) |
| blas2 | symv | f64 | n=4096 | 3.442 ms | 9.75 | 19.52 | 36% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 2.033 ms | 8.25 | 33.05 | 61% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 0.951 ms | 17.64 | 35.31 | 65% | ok | 1.2x (15%) |
| blas2 | trsv | f64 | n=4096 | 1.951 ms | 8.60 | 34.43 | 64% | ok | 1.1x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.013 ms | 16.57 | 33.17 | 61% | ok | 1.2x (15%) |
| blas2 | ger | f64 | n=4096 | 16.228 ms | 2.07 | 16.55 | 31% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 2.741 ms | 97.93 | 2.30 | 19% | ok | 5.1x (64%) |
| blas3 | gemm | f32 | 512x512x512 | 1.509 ms | 177.94 | 2.09 | 18% | ok | 4.6x (58%) |
| blas3 | gemmt | f64 | 512x512x512 | 5.358 ms | 25.05 | 1.17 | 5% | ok | 1.8x (22%) |
| blas3 | symm | f64 | 512x512x512 | 3.807 ms | 70.50 | 1.65 | 14% | ok | 4.0x (50%) |
| blas3 | syrk | f64 | 512x512x512 | 3.143 ms | 42.70 | 1.33 | 8% | ok | 3.0x (37%) |
| blas3 | syr2k | f64 | 512x512x512 | 5.696 ms | 47.13 | 1.10 | 9% | ok | 3.2x (40%) |
| blas3 | trmm | f64 | 512x512x512 | 6.484 ms | 20.70 | 0.81 | 4% | ok | 1.8x (23%) |
| blas3 | trsm | f64 | 512x512x512 | 6.789 ms | 19.77 | 0.77 | 4% | ok | 1.8x (22%) |
| blas3 | trsm | f32 | 512x512x512 | 5.319 ms | 25.23 | 0.49 | 3% | ok | 1.5x (19%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.540 ms | 0.50 | 4.03 | 7% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 26.292 ms | 3.40 | 0.16 | 1% | ok | 1.5x (19%) |
| factor | lu | f32 | 512x512 | 24.745 ms | 3.62 | 0.08 | 0% | ok | 1.6x (20%) |
| factor | cholesky | f64 | 512x512 | 14.337 ms | 3.12 | 0.29 | 1% | ok | 1.0x (12%) |
| factor | cholesky | f32 | 512x512 | 14.017 ms | 3.19 | 0.15 | 0% | ok | 1.0x (12%) |
| factor | ldlt | f64 | 512x512 | 14.487 ms | 3.09 | 0.29 | 1% | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 12.656 ms | 3.53 | 0.33 | 1% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 506.351 ms | 0.35 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 51.007 ms | 3.51 | 0.08 | 1% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 583.245 ms | 0.77 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.251 ms | 0.84 | 3.35 | 6%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 98.585 ms | 2.72 | 0.04 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 28.128 ms | 3.18 | 0.15 | 1% | ok | 1.6x (20%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 47.973 ms | 0.47 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 166.050 ms | 0.91 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 669.105 ms | 0.23 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 256x256 | 74.575 ms | 2.25 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 147.864 ms | 2.84 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 150.675 ms | 2.45 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 22.046 ms | 11.42 | 0.05 | 2% | ok | 1.5x (18%) |
| eigen | sylvester | f64 | 256x256 | 327.649 ms | 3.07 | 0.00 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 36.173 ms | 0.23 | 3.71 | 7% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 50.622 ms | 0.17 | 2.65 | 5% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 47.716 ms | 0.18 | 2.81 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 25.483 ms | 0.33 | 5.27 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 37.905 ms | 0.22 | 3.54 | 7% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.178 ms | 14.71 | 153.06 | 283% ! | ok | 5.9x (73%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.514 ms | 0.74 | 7.75 | 14% | ok | 0.4x (6%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.268 ms | 18.46 | 38.81 | 72% | ok | 3.6x (45%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 21.155 ms | 0.15 | 0.54 | 1%\* | ok | 0.9x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.879 ms | 1.07 | 8.86 | 16%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 22.293 ms | 0.50 | 0.03 | 0% | ok | 1.1x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 35.666 ms | 0.31 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 16.613 ms | 0.67 | 0.04 | 0% | ok | 1.1x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 66.700 ms | 0.34 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 68.372 ms | 0.33 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 29.843 ms | 0.75 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 46.079 ms | 0.04 | 0.00 | 0% | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.576 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.977 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 75.533 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.231 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.878 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.949 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.639 ms | 0.63 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.454 ms | 1.67 | 6.66 | 12%\* | ok | 1.7x (21%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.118 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.116 ms | 1.57 | 0.10 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.506 ms | 1.49 | 5.97 | 11%\* | ok | 1.9x (23%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.062 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 60.910 ms | 0.37 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.612 ms | 0.47 | 1.88 | 3%\* | ok | 1.0x (12%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.936 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.320 ms | 0.38 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.137 ms | 0.66 | 2.66 | 5%\* | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 6.947 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 349.997 ms | 0.88 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.194 ms | 1.54 | 6.14 | 11%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.151 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 84.032 ms | 3.67 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.276 ms | 1.48 | 5.92 | 11%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.541 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1111.822 ms | 0.56 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.372 ms | 0.30 | 1.19 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.477 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 626.815 ms | 0.98 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.327 ms | 0.30 | 1.19 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 41.267 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 59.652 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 36.029 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.139 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.779 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 19.942 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 43.346 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 37.221 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 21.972 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 23.607 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 22.236 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 22.343 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 27.272 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.056 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.961 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 28.979 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 2.213 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 2.128 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.036 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.816 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.783 ms | 1.47 | 16.45 | 30% | ok | 0.9x (11%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.913 ms | 1.37 | 15.33 | 28% | ok | 1.0x (12%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.239 ms | 1.17 | 13.10 | 24% | ok | 1.0x (12%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.689 ms | 2.06 | 15.55 | 29% | ok | 1.1x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.874 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 42.523 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 72.078 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.140 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 2.178 ms | 7.70 | 30.82 | 57% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 3.143 ms | 85.42 | 1.33 | 17% | ok | 6.2x (77%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.574 ms | 29.24 | 31.07 | 57% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.491 ms | 11.26 | 45.02 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.594 ms | 28.26 | 56.53 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.431 ms | 38.96 | 38.96 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.386 ms | 43.43 | 43.43 | n/a | ok | 1.1x (14%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.829 ms | 94.89 | 1.11 | n/a | ok | 6.8x (85%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 3.017 ms | 88.98 | 0.70 | n/a | ok | 7.3x (92%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 3.039 ms | 88.34 | 0.52 | n/a | ok | 7.4x (92%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.773 ms | 347.35 | 2.04 | n/a | ok | 5.9x (74%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.768 ms | 349.66 | 2.05 | n/a | ok | 5.9x (74%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

