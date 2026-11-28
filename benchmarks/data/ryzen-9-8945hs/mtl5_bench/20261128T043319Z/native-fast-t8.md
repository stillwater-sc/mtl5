## ryzen-9-8945hs -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 506.0 GFLOP/s, f32 1011.8 GFLOP/s |
| ... hardware, AVX-512F | f64 506.0, f32 1011.8 GFLOP/s |
| ... this build, SSE2 | f64 318.2, f32 635.4 GFLOP/s |
| bandwidth | read 52.9, update 53.2, triad 41.1 GB/s (ceiling 53.2) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 112% | 128% | 0 | 13% |
| blas2 | 10 | 65% | 119% | 0 | 13% |
| blas3 | 10 | 9% | 19% | 0 | 39% |
| factor | 12 | 1% | 6% | 0 | 19% |
| eigen | 8 | 0% | 2% | 0 | 13% |
| elementwise | 5 | 7% | 10% | 0 | 13% |
| sparse | 5 | 33% | 99% | 0 | 14% |
| direct | 19 | 0% | 17% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 11% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 25% | 33% | 0 | 13% |
| mg | 1 | 30% | 30% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 58% | 58% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 74% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.246 ms | 7.47 | 59.75 | 112% | ok | 1.1x (13%) |
| blas1 | dot | f32 | n=8388608 | 0.987 ms | 17.00 | 67.99 | 128% | ok | 1.3x (16%) |
| blas1 | axpy | f64 | n=8388608 | 3.477 ms | 4.83 | 57.90 | 109% | ok | 1.0x (13%) |
| blas1 | axpy | f32 | n=8388608 | 1.623 ms | 10.34 | 62.02 | 117% | ok | 1.1x (13%) |
| blas1 | scale | f64 | n=8388608 | 1.982 ms | 4.23 | 67.72 | 127% | ok | 1.2x (15%) |
| blas1 | two_norm | f64 | n=8388608 | 1.046 ms | 16.05 | 64.19 | 121% | ok | 1.1x (14%) |
| blas1 | one_norm | f64 | n=8388608 | 6.416 ms | 1.31 | 10.46 | 20% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.402 ms | 1.31 | 10.48 | 20% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 6.394 ms | 1.31 | 10.50 | 20% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 111.461 ms | 1.66 | 1.20 | 2% | ok | 1.0x (12%) |
| blas2 | gemv | f64 | n=4096 | 2.306 ms | 14.55 | 58.24 | 110% | ok | 1.5x (19%) |
| blas2 | gemv | f32 | n=4096 | 1.064 ms | 31.52 | 63.08 | 119% | ok | 1.7x (21%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.337 ms | 14.36 | 57.47 | 108% | ok | 1.7x (21%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.209 ms | 27.76 | 55.55 | 104% | ok | 1.7x (21%) |
| blas2 | symv | f64 | n=4096 | 3.729 ms | 9.00 | 18.01 | 34% | ok | 0.9x (11%) |
| blas2 | trmv | f64 | n=4096 | 1.959 ms | 8.57 | 34.30 | 65% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 1.163 ms | 14.42 | 28.88 | 54% | ok | 0.9x (11%) |
| blas2 | trsv | f64 | n=4096 | 1.937 ms | 8.66 | 34.69 | 65% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.186 ms | 14.15 | 28.33 | 53% | ok | 1.0x (12%) |
| blas2 | ger | f64 | n=4096 | 16.311 ms | 2.06 | 16.46 | 31% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.739 ms | 98.02 | 2.30 | 19% | ok | 5.0x (63%) |
| blas3 | gemm | f32 | 512x512x512 | 1.628 ms | 164.88 | 1.93 | 16% | ok | 4.3x (53%) |
| blas3 | gemmt | f64 | 512x512x512 | 4.338 ms | 30.94 | 1.45 | 6% | ok | 2.1x (27%) |
| blas3 | symm | f64 | 512x512x512 | 3.873 ms | 69.31 | 1.62 | 14% | ok | 3.9x (49%) |
| blas3 | syrk | f64 | 512x512x512 | 2.948 ms | 45.53 | 1.42 | 9% | ok | 3.1x (39%) |
| blas3 | syr2k | f64 | 512x512x512 | 5.538 ms | 48.47 | 1.14 | 10% | ok | 3.3x (41%) |
| blas3 | trmm | f64 | 512x512x512 | 6.534 ms | 20.54 | 0.80 | 4% | ok | 1.8x (23%) |
| blas3 | trsm | f64 | 512x512x512 | 6.867 ms | 19.55 | 0.76 | 4% | ok | 1.8x (22%) |
| blas3 | trsm | f32 | 512x512x512 | 5.401 ms | 24.85 | 0.49 | 2% | ok | 1.5x (19%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 11.238 ms | 0.47 | 3.78 | 7% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 25.800 ms | 3.47 | 0.16 | 1% | ok | 1.6x (19%) |
| factor | lu | f32 | 512x512 | 24.747 ms | 3.62 | 0.08 | 0% | ok | 1.5x (19%) |
| factor | cholesky | f64 | 512x512 | 14.285 ms | 3.13 | 0.29 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 13.692 ms | 3.27 | 0.15 | 0% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 14.393 ms | 3.11 | 0.29 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 12.850 ms | 3.48 | 0.33 | 1% | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 499.295 ms | 0.36 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 51.320 ms | 3.49 | 0.08 | 1% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 568.995 ms | 0.79 | 0.01 | 0% | ok | 1.6x (21%) |
| factor | householder | f64 | 512x512 | 1.250 ms | 0.84 | 3.36 | 6%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 97.624 ms | 2.75 | 0.04 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 27.907 ms | 3.21 | 0.15 | 1% | ok | 1.5x (19%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 49.450 ms | 0.45 | 0.02 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 160.273 ms | 0.94 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 652.489 ms | 0.23 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 73.519 ms | 2.28 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | schur | f64 | 256x256 | 141.285 ms | 2.97 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 149.640 ms | 2.47 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 21.870 ms | 11.51 | 0.05 | 2% | ok | 1.5x (18%) |
| eigen | sylvester | f64 | 256x256 | 313.226 ms | 3.21 | 0.01 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 33.366 ms | 0.25 | 4.02 | 8% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 51.704 ms | 0.16 | 2.60 | 5% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 49.045 ms | 0.17 | 2.74 | 5% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 26.344 ms | 0.32 | 5.09 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 38.773 ms | 0.22 | 3.46 | 7% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.515 ms | 5.08 | 52.89 | 99% | ok | 1.8x (22%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.549 ms | 1.69 | 17.57 | 33% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.281 ms | 18.36 | 38.60 | 73% | ok | 3.5x (43%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 22.349 ms | 0.15 | 0.51 | 1%\* | ok | 0.9x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.883 ms | 1.07 | 8.84 | 17%\* | ok | 1.1x (14%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 21.944 ms | 0.51 | 0.03 | 0% | ok | 1.1x (14%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 34.672 ms | 0.32 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 16.089 ms | 0.70 | 0.05 | 0% | ok | 1.1x (14%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 65.285 ms | 0.34 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 68.681 ms | 0.33 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 30.301 ms | 0.74 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 42.535 ms | 0.04 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.575 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.916 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 74.892 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.228 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.400 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.925 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.887 ms | 0.63 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.343 ms | 2.21 | 8.83 | 17%\* | ok | 1.1x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.127 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.244 ms | 1.54 | 0.10 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.352 ms | 2.15 | 8.59 | 16%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.089 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 57.795 ms | 0.39 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.245 ms | 0.61 | 2.43 | 5%\* | ok | 1.1x (14%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.924 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.512 ms | 0.38 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.298 ms | 0.58 | 2.33 | 4%\* | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 6.913 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 345.801 ms | 0.89 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.283 ms | 1.48 | 5.90 | 11%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 16.806 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 84.386 ms | 3.66 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.361 ms | 1.43 | 5.71 | 11%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.609 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1094.344 ms | 0.56 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.220 ms | 0.30 | 1.20 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.568 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 611.229 ms | 1.01 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.145 ms | 0.30 | 1.21 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 41.197 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 58.480 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 35.961 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.070 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.677 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 19.931 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 43.231 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 37.219 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 21.940 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 23.659 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 22.227 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 22.321 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 27.228 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.055 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 3.038 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 27.785 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 2.230 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 2.169 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.037 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.818 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.688 ms | 1.55 | 17.37 | 33% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.215 ms | 1.18 | 13.24 | 25% | ok | 0.9x (11%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.232 ms | 1.17 | 13.14 | 25% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.515 ms | 2.08 | 15.70 | 30% | ok | 1.1x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.590 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.240 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 65.176 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.060 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 2.173 ms | 7.72 | 30.88 | 58% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 3.125 ms | 85.91 | 1.34 | 17% | ok | 6.2x (77%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.578 ms | 29.02 | 30.83 | 58% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.425 ms | 11.77 | 47.08 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.598 ms | 28.07 | 56.15 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.430 ms | 39.00 | 39.00 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.387 ms | 43.34 | 43.34 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.878 ms | 93.27 | 1.09 | n/a | ok | 6.6x (82%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 3.009 ms | 89.21 | 0.70 | n/a | ok | 7.3x (91%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 3.039 ms | 88.33 | 0.52 | n/a | ok | 7.4x (93%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.772 ms | 347.53 | 2.04 | n/a | ok | 6.9x (86%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.767 ms | 349.94 | 2.05 | n/a | ok | 5.9x (74%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

