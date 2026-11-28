## ryzen-9-8945hs -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 556.5 GFLOP/s, f32 1012.4 GFLOP/s |
| ... hardware, AVX-512F | f64 556.5, f32 1012.4 GFLOP/s |
| ... this build, scalar | f64 14.6, f32 14.6 GFLOP/s |
| bandwidth | read 52.6, update 53.2, triad 40.9 GB/s (ceiling 53.2) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 82% | 127% | 0 | 54% |
| blas2 | 10 | 103% | 119% | 0 | 18% |
| blas3 | 10 | 43% | 71% | 0 | 67% |
| factor | 12 | 1% | 6% | 0 | 12% |
| eigen | 8 | 2% | 5% | 0 | 11% |
| elementwise | 5 | 6% | 10% | 0 | 13% |
| sparse | 5 | 20% | 280% | 0 | 13% |
| direct | 19 | 0% | 16% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 11% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 31% | 0 | 13% |
| mg | 1 | 29% | 29% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 10% | 18% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 96% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.286 ms | 7.34 | 58.71 | 110% | ok | 4.9x (61%) |
| blas1 | dot | f32 | n=8388608 | 1.540 ms | 10.90 | 43.58 | 82% | ok | 8.0x (99%) |
| blas1 | axpy | f64 | n=8388608 | 3.500 ms | 4.79 | 57.52 | 108% | ok | 4.3x (54%) |
| blas1 | axpy | f32 | n=8388608 | 1.922 ms | 8.73 | 52.36 | 98% | ok | 7.8x (97%) |
| blas1 | scale | f64 | n=8388608 | 1.984 ms | 4.23 | 67.64 | 127% | ok | 1.3x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 1.542 ms | 10.88 | 43.52 | 82% | ok | 8.0x (99%) |
| blas1 | one_norm | f64 | n=8388608 | 6.421 ms | 1.31 | 10.45 | 20% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.440 ms | 1.30 | 10.42 | 20% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 6.394 ms | 1.31 | 10.50 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 102.948 ms | 1.79 | 1.30 | 2% | ok | 1.1x (14%) |
| blas2 | gemv | f64 | n=4096 | 2.491 ms | 13.47 | 53.91 | 101% | ok | 1.5x (18%) |
| blas2 | gemv | f32 | n=4096 | 1.173 ms | 28.61 | 57.24 | 108% | ok | 1.3x (16%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.662 ms | 12.60 | 50.44 | 95% | ok | 1.6x (20%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.290 ms | 26.02 | 52.06 | 98% | ok | 1.2x (15%) |
| blas2 | symv | f64 | n=4096 | 1.226 ms | 27.38 | 54.80 | 103% | ok | 1.6x (20%) |
| blas2 | trmv | f64 | n=4096 | 1.226 ms | 13.68 | 54.77 | 103% | ok | 1.5x (19%) |
| blas2 | trmv | f32 | n=4096 | 0.530 ms | 31.63 | 63.32 | 119% | ok | 2.0x (26%) |
| blas2 | trsv | f64 | n=4096 | 1.706 ms | 9.84 | 39.38 | 74% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.921 ms | 18.22 | 36.48 | 69% | ok | 1.1x (13%) |
| blas2 | ger | f64 | n=4096 | 4.675 ms | 7.18 | 57.44 | 108% | ok | 1.4x (18%) |
| blas3 | gemm | f64 | 512x512x512 | 0.682 ms | 393.83 | 9.23 | 71% | ok | 6.7x (84%) |
| blas3 | gemm | f32 | 512x512x512 | 0.385 ms | 697.96 | 8.18 | 69% | ok | 6.2x (78%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.674 ms | 80.19 | 3.76 | 14% | ok | 2.0x (26%) |
| blas3 | symm | f64 | 512x512x512 | 0.682 ms | 393.66 | 9.23 | 71% | ok | 7.0x (87%) |
| blas3 | syrk | f64 | 512x512x512 | 1.991 ms | 67.40 | 2.11 | 12% | ok | 1.9x (23%) |
| blas3 | syr2k | f64 | 512x512x512 | 2.703 ms | 99.31 | 2.33 | 18% | ok | 2.2x (28%) |
| blas3 | trmm | f64 | 512x512x512 | 0.442 ms | 303.94 | 11.87 | 55% | ok | 5.3x (67%) |
| blas3 | trsm | f64 | 512x512x512 | 0.603 ms | 222.47 | 8.69 | 40% | ok | 5.8x (73%) |
| blas3 | trsm | f32 | 512x512x512 | 0.307 ms | 437.33 | 8.54 | 43% | ok | 5.3x (67%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 11.179 ms | 0.47 | 3.80 | 7% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 11.121 ms | 8.05 | 0.38 | 1% | ok | 0.6x (7%) |
| factor | lu | f32 | 512x512 | 5.079 ms | 17.62 | 0.41 | 2% | ok | 0.6x (8%) |
| factor | cholesky | f64 | 512x512 | 6.985 ms | 6.41 | 0.60 | 1% | ok | 0.8x (10%) |
| factor | cholesky | f32 | 512x512 | 2.446 ms | 18.29 | 0.86 | 2% | ok | 0.9x (11%) |
| factor | ldlt | f64 | 512x512 | 14.820 ms | 3.02 | 0.28 | 1% | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 12.647 ms | 3.54 | 0.33 | 1% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 14.107 ms | 12.69 | 0.30 | 2% | ok | 0.9x (11%) |
| factor | lq | f64 | 512x512 | 50.925 ms | 3.51 | 0.08 | 1% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 573.863 ms | 0.78 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.260 ms | 0.83 | 3.33 | 6%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 82.035 ms | 3.27 | 0.05 | 1% | ok | 0.9x (12%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 8.768 ms | 10.21 | 0.48 | 2% | ok | 0.8x (10%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 4.272 ms | 5.24 | 0.25 | 1% | ok | 0.5x (6%) |
| eigen | eigen_symmetric | f64 | 256x256 | 161.594 ms | 0.93 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 675.672 ms | 0.22 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 256x256 | 25.125 ms | 6.68 | 0.04 | 1% | ok | 0.8x (11%) |
| eigen | schur | f64 | 256x256 | 31.277 ms | 13.41 | 0.03 | 2% | ok | 0.9x (11%) |
| eigen | svd | f64 | 256x256 | 13.537 ms | 27.27 | 0.08 | 5% | ok | 0.8x (10%) |
| eigen | expm | f64 | 256x256 | 20.662 ms | 12.18 | 0.05 | 2% | ok | 0.9x (12%) |
| eigen | sylvester | f64 | 256x256 | 99.522 ms | 10.11 | 0.02 | 2% | ok | 0.9x (11%) |
| elementwise | exp | f64 | n=8388608 | 33.237 ms | 0.25 | 4.04 | 8% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 52.943 ms | 0.16 | 2.54 | 5% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 47.736 ms | 0.18 | 2.81 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 26.478 ms | 0.32 | 5.07 | 10% | ok | 1.0x (12%) |
| elementwise | erf | f64 | n=8388608 | 42.353 ms | 0.20 | 3.17 | 6% | ok | 1.1x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.183 ms | 14.33 | 149.04 | 280% ! | ok | 5.1x (63%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.505 ms | 1.04 | 10.87 | 20% | ok | 0.6x (8%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.247 ms | 18.64 | 39.18 | 74% | ok | 4.3x (54%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 21.528 ms | 0.15 | 0.53 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.885 ms | 1.06 | 8.83 | 17%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 24.071 ms | 0.46 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 39.005 ms | 0.29 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 20.241 ms | 0.55 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 65.743 ms | 0.34 | 0.01 | 0% | ok | 1.1x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 70.477 ms | 0.32 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 30.176 ms | 0.74 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 42.748 ms | 0.04 | 0.00 | 0%\* | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.603 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.976 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 74.883 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.244 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.998 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.720 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.677 ms | 0.63 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.346 ms | 2.19 | 8.74 | 16%\* | ok | 1.0x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.254 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.286 ms | 1.54 | 0.10 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.346 ms | 2.19 | 8.74 | 16%\* | ok | 1.1x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.143 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 58.205 ms | 0.38 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.333 ms | 0.57 | 2.27 | 4%\* | ok | 0.9x (12%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.946 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.859 ms | 0.37 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.280 ms | 0.59 | 2.36 | 4%\* | ok | 0.9x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 7.032 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 344.571 ms | 0.90 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.275 ms | 1.48 | 5.92 | 11%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.939 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 84.565 ms | 3.65 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.318 ms | 1.45 | 5.81 | 11%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.712 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1101.795 ms | 0.56 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.298 ms | 0.30 | 1.19 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.559 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 625.856 ms | 0.99 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.230 ms | 0.30 | 1.20 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 49.162 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 73.784 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 39.841 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.191 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 27.200 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.233 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 66.550 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 43.215 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.942 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 32.822 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 30.917 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 31.928 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 41.334 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.054 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.950 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 27.149 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 2.220 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 2.129 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.035 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.183 ms | 0.00 | 0.00 | n/a | ok | 0.8x (10%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.760 ms | 1.49 | 16.67 | 31% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.909 ms | 1.37 | 15.37 | 29% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.249 ms | 1.16 | 13.04 | 25% | ok | 1.0x (12%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.543 ms | 2.07 | 15.67 | 29% | ok | 1.1x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.560 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.329 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 64.778 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.745 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 12.254 ms | 1.37 | 5.48 | 10% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.875 ms | 55.06 | 0.86 | 10% | ok | 7.7x (96%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.838 ms | 9.13 | 9.70 | 18% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.038 ms | 4.16 | 16.62 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.245 ms | 5.17 | 10.34 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.122 ms | 7.91 | 7.91 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.122 ms | 7.91 | 7.91 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.269 ms | 211.58 | 2.48 | n/a | ok | 8.1x (101%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.024 ms | 53.43 | 0.42 | n/a | ok | 7.9x (98%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.507 ms | 59.55 | 0.35 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.513 ms | 59.48 | 0.35 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.511 ms | 59.50 | 0.35 | n/a | ok | 7.6x (96%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

