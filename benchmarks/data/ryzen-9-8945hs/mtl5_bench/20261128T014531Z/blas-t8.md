## ryzen-9-8945hs -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 557.0 GFLOP/s, f32 918.4 GFLOP/s |
| ... hardware, AVX-512F | f64 557.0, f32 918.4 GFLOP/s |
| ... this build, scalar | f64 14.6, f32 14.6 GFLOP/s |
| bandwidth | read 52.4, update 53.8, triad 40.8 GB/s (ceiling 53.8) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 81% | 124% | 0 | 54% |
| blas2 | 10 | 102% | 115% | 0 | 18% |
| blas3 | 10 | 46% | 75% | 0 | 64% |
| factor | 12 | 1% | 6% | 0 | 11% |
| eigen | 8 | 2% | 5% | 0 | 12% |
| elementwise | 5 | 6% | 10% | 0 | 13% |
| sparse | 5 | 16% | 291% | 0 | 13% |
| direct | 19 | 0% | 13% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 11% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 29% | 31% | 0 | 16% |
| mg | 1 | 29% | 29% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 10% | 18% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 95% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.374 ms | 7.07 | 56.52 | 105% | ok | 5.0x (62%) |
| blas1 | dot | f32 | n=8388608 | 1.546 ms | 10.86 | 43.42 | 81% | ok | 7.9x (99%) |
| blas1 | axpy | f64 | n=8388608 | 3.493 ms | 4.80 | 57.64 | 107% | ok | 4.3x (54%) |
| blas1 | axpy | f32 | n=8388608 | 1.915 ms | 8.76 | 52.56 | 98% | ok | 7.9x (98%) |
| blas1 | scale | f64 | n=8388608 | 2.008 ms | 4.18 | 66.85 | 124% | ok | 1.3x (16%) |
| blas1 | two_norm | f64 | n=8388608 | 1.543 ms | 10.87 | 43.48 | 81% | ok | 7.9x (99%) |
| blas1 | one_norm | f64 | n=8388608 | 6.398 ms | 1.31 | 10.49 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.406 ms | 1.31 | 10.48 | 19% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 6.397 ms | 1.31 | 10.49 | 20% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 110.016 ms | 1.68 | 1.22 | 2% | ok | 0.9x (12%) |
| blas2 | gemv | f64 | n=4096 | 2.608 ms | 12.86 | 51.49 | 96% | ok | 1.3x (16%) |
| blas2 | gemv | f32 | n=4096 | 1.128 ms | 29.74 | 59.51 | 111% | ok | 1.4x (18%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.765 ms | 12.14 | 48.57 | 90% | ok | 1.4x (18%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.233 ms | 27.22 | 54.47 | 101% | ok | 1.2x (15%) |
| blas2 | symv | f64 | n=4096 | 1.219 ms | 27.53 | 55.12 | 102% | ok | 1.6x (20%) |
| blas2 | trmv | f64 | n=4096 | 1.220 ms | 13.75 | 55.04 | 102% | ok | 1.4x (18%) |
| blas2 | trmv | f32 | n=4096 | 0.544 ms | 30.83 | 61.73 | 115% | ok | 1.7x (21%) |
| blas2 | trsv | f64 | n=4096 | 2.042 ms | 8.22 | 32.90 | 61% | ok | 0.9x (11%) |
| blas2 | trsv | f32 | n=4096 | 1.124 ms | 14.93 | 29.88 | 56% | ok | 0.8x (11%) |
| blas2 | ger | f64 | n=4096 | 4.567 ms | 7.35 | 58.80 | 109% | ok | 1.5x (18%) |
| blas3 | gemm | f64 | 512x512x512 | 0.691 ms | 388.31 | 9.10 | 70% | ok | 6.7x (84%) |
| blas3 | gemm | f32 | 512x512x512 | 0.391 ms | 685.66 | 8.04 | 75% | ok | 6.2x (77%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.655 ms | 81.12 | 3.80 | 15% | ok | 2.2x (27%) |
| blas3 | symm | f64 | 512x512x512 | 0.722 ms | 371.85 | 8.72 | 67% | ok | 6.7x (84%) |
| blas3 | syrk | f64 | 512x512x512 | 2.083 ms | 64.45 | 2.01 | 12% | ok | 1.9x (23%) |
| blas3 | syr2k | f64 | 512x512x512 | 2.718 ms | 98.76 | 2.31 | 18% | ok | 2.3x (29%) |
| blas3 | trmm | f64 | 512x512x512 | 0.475 ms | 282.39 | 11.03 | 51% | ok | 5.1x (63%) |
| blas3 | trsm | f64 | 512x512x512 | 0.655 ms | 204.91 | 8.00 | 37% | ok | 5.4x (68%) |
| blas3 | trsm | f32 | 512x512x512 | 0.319 ms | 420.48 | 8.21 | 46% | ok | 5.2x (64%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.423 ms | 0.51 | 4.07 | 8% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 11.026 ms | 8.12 | 0.38 | 1% | ok | 0.6x (7%) |
| factor | lu | f32 | 512x512 | 5.173 ms | 17.30 | 0.41 | 2% | ok | 0.6x (8%) |
| factor | cholesky | f64 | 512x512 | 6.909 ms | 6.48 | 0.61 | 1% | ok | 0.8x (10%) |
| factor | cholesky | f32 | 512x512 | 2.638 ms | 16.96 | 0.79 | 2% | ok | 0.9x (11%) |
| factor | ldlt | f64 | 512x512 | 14.827 ms | 3.02 | 0.28 | 1% | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 12.735 ms | 3.51 | 0.33 | 1% | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 14.152 ms | 12.65 | 0.30 | 2% | ok | 0.9x (11%) |
| factor | lq | f64 | 512x512 | 51.677 ms | 3.46 | 0.08 | 1% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 594.306 ms | 0.75 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.267 ms | 0.83 | 3.31 | 6%\* | ok | 3.9x (48%) |
| factor | inv | f64 | 512x512 | 83.196 ms | 3.23 | 0.05 | 1% | ok | 0.9x (11%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 8.959 ms | 9.99 | 0.47 | 2% | ok | 0.8x (10%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 4.303 ms | 5.20 | 0.24 | 1% | ok | 0.5x (6%) |
| eigen | eigen_symmetric | f64 | 256x256 | 165.575 ms | 0.91 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 634.864 ms | 0.24 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 256x256 | 25.233 ms | 6.65 | 0.04 | 1% | ok | 0.8x (11%) |
| eigen | schur | f64 | 256x256 | 31.383 ms | 13.36 | 0.03 | 2% | ok | 0.9x (11%) |
| eigen | svd | f64 | 256x256 | 13.609 ms | 27.12 | 0.08 | 5% | ok | 0.8x (10%) |
| eigen | expm | f64 | 256x256 | 20.546 ms | 12.25 | 0.05 | 2% | ok | 1.0x (12%) |
| eigen | sylvester | f64 | 256x256 | 99.280 ms | 10.14 | 0.02 | 2% | ok | 0.9x (12%) |
| elementwise | exp | f64 | n=8388608 | 32.721 ms | 0.26 | 4.10 | 8% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 52.341 ms | 0.16 | 2.56 | 5% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 46.934 ms | 0.18 | 2.86 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 25.535 ms | 0.33 | 5.26 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 41.364 ms | 0.20 | 3.24 | 6% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.174 ms | 15.07 | 156.77 | 291% ! | ok | 5.1x (64%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 5.844 ms | 0.45 | 4.66 | 9% | ok | 0.3x (3%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.325 ms | 18.01 | 37.87 | 70% | ok | 3.4x (43%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 19.709 ms | 0.17 | 0.58 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.895 ms | 1.06 | 8.79 | 16%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 24.456 ms | 0.46 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 39.754 ms | 0.28 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 20.093 ms | 0.56 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 65.218 ms | 0.34 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 70.611 ms | 0.32 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 30.193 ms | 0.74 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 42.906 ms | 0.04 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.604 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.918 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 75.338 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.242 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.790 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.735 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.719 ms | 0.63 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.443 ms | 1.71 | 6.82 | 13%\* | ok | 2.2x (27%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.224 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.354 ms | 1.52 | 0.10 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.637 ms | 1.19 | 4.75 | 9%\* | ok | 1.4x (18%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.065 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 57.078 ms | 0.39 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.345 ms | 0.56 | 2.25 | 4%\* | ok | 1.2x (15%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.901 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.642 ms | 0.38 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.125 ms | 0.67 | 2.69 | 5%\* | ok | 1.2x (15%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 7.054 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 412.633 ms | 0.75 | 0.00 | 0% | ok | 0.8x (10%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.602 ms | 1.29 | 5.18 | 10%\* | ok | 0.8x (10%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.195 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 84.586 ms | 3.65 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.207 ms | 1.53 | 6.11 | 11%\* | ok | 1.0x (13%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.731 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1130.630 ms | 0.55 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.346 ms | 0.30 | 1.19 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.771 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 643.704 ms | 0.96 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.277 ms | 0.30 | 1.20 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 49.407 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 74.352 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 40.033 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.194 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 27.471 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.598 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 68.460 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 43.497 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.151 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.057 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 31.284 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 32.323 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.061 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.055 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.956 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 45.199 ms | 0.00 | 0.00 | n/a | ok | 0.6x (8%) |
| pc | ic_0 | f64 | 128^2 grid | 3.779 ms | 0.00 | 0.00 | n/a | ok | 0.6x (7%) |
| pc | ildl | f64 | 128^2 grid | 3.700 ms | 0.00 | 0.00 | n/a | ok | 0.6x (7%) |
| pc | ssor | f64 | 128^2 grid | 0.036 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.648 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.755 ms | 1.49 | 16.71 | 31% | ok | 1.0x (12%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.909 ms | 1.37 | 15.36 | 29% | ok | 1.3x (16%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.247 ms | 1.16 | 13.05 | 24% | ok | 1.3x (16%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.713 ms | 2.05 | 15.53 | 29% | ok | 1.1x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.552 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.386 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 70.797 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.862 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 12.267 ms | 1.37 | 5.47 | 10% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.855 ms | 55.29 | 0.86 | 10% | ok | 7.8x (97%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.831 ms | 9.16 | 9.73 | 18% | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.078 ms | 4.11 | 16.46 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.244 ms | 5.17 | 10.34 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.123 ms | 7.90 | 7.90 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.122 ms | 7.91 | 7.91 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.340 ms | 200.38 | 2.35 | n/a | ok | 7.6x (96%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.999 ms | 53.69 | 0.42 | n/a | ok | 7.7x (96%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.516 ms | 59.45 | 0.35 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.511 ms | 59.50 | 0.35 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.535 ms | 59.20 | 0.35 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

