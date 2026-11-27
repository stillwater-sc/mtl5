## ryzen-9-8945hs -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `57512adcb8fe` |
| FMA peak (ceiling) | f64 488.3 GFLOP/s, f32 1009.4 GFLOP/s |
| ... hardware, AVX-512F | f64 488.3, f32 1009.4 GFLOP/s |
| ... this build, scalar | f64 12.4, f32 14.6 GFLOP/s |
| bandwidth | read 52.5, update 53.1, triad 40.9 GB/s (ceiling 53.1) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 82% | 125% | 0 | 53% |
| blas2 | 10 | 99% | 107% | 0 | 17% |
| blas3 | 10 | 45% | 81% | 0 | 68% |
| factor | 12 | 2% | 6% | 0 | 12% |
| eigen | 8 | 2% | 6% | 0 | 12% |
| elementwise | 5 | 6% | 10% | 0 | 13% |
| sparse | 5 | 32% | 281% | 0 | 12% |
| direct | 19 | 0% | 16% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 11% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 30% | 33% | 0 | 13% |
| mg | 1 | 30% | 30% | 0 | 15% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 11% | 18% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 96% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.305 ms | 7.28 | 58.23 | 110% | ok | 6.0x (74%) |
| blas1 | dot | f32 | n=8388608 | 1.540 ms | 10.90 | 43.58 | 82% | ok | 7.9x (99%) |
| blas1 | axpy | f64 | n=8388608 | 3.517 ms | 4.77 | 57.24 | 108% | ok | 4.3x (53%) |
| blas1 | axpy | f32 | n=8388608 | 1.726 ms | 9.72 | 58.34 | 110% | ok | 7.4x (93%) |
| blas1 | scale | f64 | n=8388608 | 2.017 ms | 4.16 | 66.56 | 125% | ok | 2.2x (28%) |
| blas1 | two_norm | f64 | n=8388608 | 1.806 ms | 9.29 | 37.16 | 70% | ok | 8.1x (101%) |
| blas1 | one_norm | f64 | n=8388608 | 6.395 ms | 1.31 | 10.49 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.405 ms | 1.31 | 10.48 | 20% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 6.397 ms | 1.31 | 10.49 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 105.238 ms | 1.75 | 1.28 | 2% | ok | 1.1x (13%) |
| blas2 | gemv | f64 | n=4096 | 2.584 ms | 12.98 | 51.97 | 98% | ok | 1.3x (16%) |
| blas2 | gemv | f32 | n=4096 | 1.188 ms | 28.25 | 56.52 | 107% | ok | 1.3x (17%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.735 ms | 12.27 | 49.10 | 93% | ok | 1.3x (16%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.322 ms | 25.37 | 50.77 | 96% | ok | 1.1x (14%) |
| blas2 | symv | f64 | n=4096 | 1.277 ms | 26.28 | 52.61 | 99% | ok | 1.7x (21%) |
| blas2 | trmv | f64 | n=4096 | 1.247 ms | 13.45 | 53.86 | 102% | ok | 1.5x (19%) |
| blas2 | trmv | f32 | n=4096 | 0.610 ms | 27.52 | 55.09 | 104% | ok | 1.6x (20%) |
| blas2 | trsv | f64 | n=4096 | 1.778 ms | 9.43 | 37.77 | 71% | ok | 1.1x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.925 ms | 18.14 | 36.31 | 68% | ok | 1.2x (15%) |
| blas2 | ger | f64 | n=4096 | 4.717 ms | 7.11 | 56.92 | 107% | ok | 1.4x (18%) |
| blas3 | gemm | f64 | 512x512x512 | 0.682 ms | 393.83 | 9.23 | 81% | ok | 6.7x (84%) |
| blas3 | gemm | f32 | 512x512x512 | 0.386 ms | 695.25 | 8.15 | 69% | ok | 6.2x (77%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.683 ms | 79.73 | 3.74 | 16% | ok | 2.0x (25%) |
| blas3 | symm | f64 | 512x512x512 | 0.682 ms | 393.31 | 9.22 | 81% | ok | 7.0x (87%) |
| blas3 | syrk | f64 | 512x512x512 | 1.989 ms | 67.48 | 2.11 | 14% | ok | 1.9x (23%) |
| blas3 | syr2k | f64 | 512x512x512 | 2.703 ms | 99.32 | 2.33 | 20% | ok | 2.3x (28%) |
| blas3 | trmm | f64 | 512x512x512 | 0.440 ms | 305.25 | 11.92 | 63% | ok | 5.4x (67%) |
| blas3 | trsm | f64 | 512x512x512 | 0.609 ms | 220.57 | 8.62 | 45% | ok | 5.8x (72%) |
| blas3 | trsm | f32 | 512x512x512 | 0.302 ms | 445.02 | 8.69 | 44% | ok | 5.4x (68%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.533 ms | 0.50 | 4.03 | 8% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 11.060 ms | 8.09 | 0.38 | 2% | ok | 0.6x (7%) |
| factor | lu | f32 | 512x512 | 5.074 ms | 17.63 | 0.41 | 2% | ok | 0.6x (7%) |
| factor | cholesky | f64 | 512x512 | 7.219 ms | 6.20 | 0.58 | 1% | ok | 0.8x (9%) |
| factor | cholesky | f32 | 512x512 | 2.424 ms | 18.46 | 0.87 | 2% | ok | 0.9x (11%) |
| factor | ldlt | f64 | 512x512 | 14.814 ms | 3.02 | 0.28 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 12.621 ms | 3.54 | 0.33 | 1% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 14.135 ms | 12.66 | 0.30 | 3% | ok | 0.9x (11%) |
| factor | lq | f64 | 512x512 | 50.924 ms | 3.51 | 0.08 | 1% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 572.763 ms | 0.78 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.247 ms | 0.84 | 3.36 | 6%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 83.338 ms | 3.22 | 0.05 | 1% | ok | 0.9x (12%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 8.803 ms | 10.17 | 0.48 | 2% | ok | 0.8x (10%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 4.313 ms | 5.19 | 0.24 | 1% | ok | 0.5x (6%) |
| eigen | eigen_symmetric | f64 | 256x256 | 157.223 ms | 0.96 | 0.01 | 0% | ok | 1.1x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 628.579 ms | 0.24 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 256x256 | 25.410 ms | 6.60 | 0.04 | 1% | ok | 0.8x (10%) |
| eigen | schur | f64 | 256x256 | 31.545 ms | 13.30 | 0.03 | 3% | ok | 0.9x (11%) |
| eigen | svd | f64 | 256x256 | 13.432 ms | 27.48 | 0.08 | 6% | ok | 0.8x (10%) |
| eigen | expm | f64 | 256x256 | 20.831 ms | 12.08 | 0.05 | 2% | ok | 0.9x (12%) |
| eigen | sylvester | f64 | 256x256 | 100.247 ms | 10.04 | 0.02 | 2% | ok | 0.9x (12%) |
| elementwise | exp | f64 | n=8388608 | 32.864 ms | 0.26 | 4.08 | 8% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 52.330 ms | 0.16 | 2.56 | 5% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 47.158 ms | 0.18 | 2.85 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 25.753 ms | 0.33 | 5.21 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 41.552 ms | 0.20 | 3.23 | 6% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.182 ms | 14.35 | 149.29 | 281% ! | ok | 5.1x (64%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.581 ms | 1.66 | 17.22 | 32% | ok | 1.0x (12%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.269 ms | 18.46 | 38.81 | 73% | ok | 3.6x (45%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.749 ms | 0.16 | 0.55 | 1%\* | ok | 0.9x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.999 ms | 1.00 | 8.33 | 16%\* | ok | 1.0x (12%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 23.858 ms | 0.47 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 38.888 ms | 0.29 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 25.829 ms | 0.43 | 0.03 | 0% | ok | 0.7x (9%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 65.285 ms | 0.34 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 69.724 ms | 0.32 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 30.398 ms | 0.74 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 42.923 ms | 0.04 | 0.00 | 0% | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.569 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.997 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 75.623 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.239 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.848 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.711 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 18.075 ms | 0.62 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.349 ms | 2.17 | 8.68 | 16%\* | ok | 1.0x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.343 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 6.964 ms | 1.61 | 0.10 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.346 ms | 2.18 | 8.74 | 16%\* | ok | 1.0x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.073 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 57.859 ms | 0.39 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.367 ms | 0.55 | 2.21 | 4%\* | ok | 1.0x (12%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.960 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.284 ms | 0.38 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.376 ms | 0.55 | 2.20 | 4%\* | ok | 0.9x (11%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 6.954 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 340.076 ms | 0.91 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.256 ms | 1.49 | 5.97 | 11%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.562 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 86.248 ms | 3.58 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.357 ms | 1.43 | 5.72 | 11%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.557 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1121.991 ms | 0.55 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.287 ms | 0.30 | 1.19 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.500 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 630.656 ms | 0.98 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.255 ms | 0.30 | 1.20 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 50.960 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 106.447 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 40.409 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.805 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 28.265 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.735 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 85.839 ms | 0.00 | 0.00 | n/a | ok | 0.8x (10%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 45.423 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 26.394 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 34.828 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.912 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 34.041 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 44.645 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.058 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.969 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 28.215 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 2.233 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 128^2 grid | 2.155 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.030 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.718 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.696 ms | 1.54 | 17.30 | 33% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.858 ms | 1.41 | 15.79 | 30% | ok | 1.1x (14%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.332 ms | 1.12 | 12.58 | 24% | ok | 1.0x (12%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.411 ms | 2.09 | 15.79 | 30% | ok | 1.2x (15%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.758 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.219 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 70.002 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 10.098 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 14.379 ms | 1.17 | 4.67 | 9% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.859 ms | 55.24 | 0.86 | 11% | ok | 7.8x (97%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.837 ms | 9.13 | 9.70 | 18% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.075 ms | 4.12 | 16.47 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.259 ms | 5.15 | 10.30 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.122 ms | 7.90 | 7.90 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.122 ms | 7.91 | 7.91 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.277 ms | 210.26 | 2.46 | n/a | ok | 8.2x (102%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.006 ms | 53.62 | 0.42 | n/a | ok | 7.7x (96%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.531 ms | 59.24 | 0.35 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.529 ms | 59.27 | 0.35 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.519 ms | 59.39 | 0.35 | n/a | ok | 7.7x (96%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

