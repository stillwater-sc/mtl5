## ryzen-7-8845hs -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `57512adcb8fe` |
| FMA peak (ceiling) | f64 624.6 GFLOP/s, f32 1248.4 GFLOP/s |
| ... hardware, AVX-512F | f64 624.6, f32 1248.4 GFLOP/s |
| ... this build, scalar | f64 12.2, f32 14.4 GFLOP/s |
| bandwidth | read 53.2, update 53.6, triad 41.1 GB/s (ceiling 53.6) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 94% | 127% | 0 | 43% |
| blas2 | 10 | 103% | 112% | 0 | 18% |
| blas3 | 10 | 41% | 71% | 0 | 63% |
| factor | 12 | 2% | 8% | 0 | 10% |
| eigen | 8 | 2% | 5% | 0 | 11% |
| elementwise | 5 | 7% | 11% | 0 | 11% |
| sparse | 5 | 37% | 252% | 0 | 36% |
| direct | 19 | 0% | 19% | 0 | 11% |
| ordering | 5 | n/a | n/a | 0 | 11% |
| direct3d | 12 | 2% | 11% | 0 | 11% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 30% | 34% | 0 | 12% |
| mg | 1 | 34% | 34% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 10% | 21% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 77% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.260 ms | 7.42 | 59.39 | 111% | ok | 5.0x (62%) |
| blas1 | dot | f32 | n=8388608 | 1.334 ms | 12.57 | 50.30 | 94% | ok | 7.4x (92%) |
| blas1 | axpy | f64 | n=8388608 | 3.493 ms | 4.80 | 57.63 | 107% | ok | 3.4x (43%) |
| blas1 | axpy | f32 | n=8388608 | 1.728 ms | 9.71 | 58.24 | 109% | ok | 6.0x (75%) |
| blas1 | scale | f64 | n=8388608 | 1.968 ms | 4.26 | 68.20 | 127% | ok | 1.2x (15%) |
| blas1 | two_norm | f64 | n=8388608 | 1.554 ms | 10.80 | 43.20 | 81% | ok | 7.4x (92%) |
| blas1 | one_norm | f64 | n=8388608 | 5.069 ms | 1.66 | 13.24 | 25% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.100 ms | 1.64 | 13.16 | 25% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 5.065 ms | 1.66 | 13.25 | 25% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 87.362 ms | 2.11 | 1.54 | 3% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 2.461 ms | 13.64 | 54.57 | 102% | ok | 1.4x (18%) |
| blas2 | gemv | f32 | n=4096 | 1.172 ms | 28.63 | 57.28 | 107% | ok | 1.3x (17%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.604 ms | 12.88 | 51.56 | 96% | ok | 1.4x (18%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.264 ms | 26.56 | 53.14 | 99% | ok | 1.2x (15%) |
| blas2 | symv | f64 | n=4096 | 1.219 ms | 27.52 | 55.09 | 103% | ok | 1.6x (20%) |
| blas2 | trmv | f64 | n=4096 | 1.218 ms | 13.78 | 55.16 | 103% | ok | 1.5x (18%) |
| blas2 | trmv | f32 | n=4096 | 0.561 ms | 29.90 | 59.86 | 112% | ok | 1.7x (21%) |
| blas2 | trsv | f64 | n=4096 | 1.679 ms | 9.99 | 40.02 | 75% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.858 ms | 19.55 | 39.13 | 73% | ok | 1.0x (12%) |
| blas2 | ger | f64 | n=4096 | 4.707 ms | 7.13 | 57.04 | 106% | ok | 1.4x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 0.610 ms | 440.42 | 10.32 | 71% | ok | 6.0x (75%) |
| blas3 | gemm | f32 | 512x512x512 | 0.336 ms | 797.97 | 9.35 | 64% | ok | 5.7x (71%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.362 ms | 98.58 | 4.62 | 16% | ok | 2.0x (25%) |
| blas3 | symm | f64 | 512x512x512 | 0.612 ms | 438.55 | 10.28 | 70% | ok | 6.2x (78%) |
| blas3 | syrk | f64 | 512x512x512 | 1.723 ms | 77.89 | 2.43 | 12% | ok | 1.7x (22%) |
| blas3 | syr2k | f64 | 512x512x512 | 2.397 ms | 111.99 | 2.62 | 18% | ok | 2.0x (25%) |
| blas3 | trmm | f64 | 512x512x512 | 0.393 ms | 341.96 | 13.36 | 55% | ok | 4.8x (60%) |
| blas3 | trsm | f64 | 512x512x512 | 0.537 ms | 250.08 | 9.77 | 40% | ok | 5.3x (66%) |
| blas3 | trsm | f32 | 512x512x512 | 0.265 ms | 506.29 | 9.89 | 41% | ok | 5.1x (63%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 8.974 ms | 0.59 | 4.73 | 9% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 9.201 ms | 9.72 | 0.46 | 2% | ok | 0.5x (7%) |
| factor | lu | f32 | 512x512 | 4.340 ms | 20.62 | 0.48 | 2% | ok | 0.6x (7%) |
| factor | cholesky | f64 | 512x512 | 5.904 ms | 7.58 | 0.71 | 1%\* | ok | 0.7x (9%) |
| factor | cholesky | f32 | 512x512 | 2.283 ms | 19.59 | 0.92 | 2%\* | ok | 0.7x (9%) |
| factor | ldlt | f64 | 512x512 | 11.855 ms | 3.77 | 0.35 | 1%\* | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 10.708 ms | 4.18 | 0.39 | 1%\* | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 12.949 ms | 13.82 | 0.32 | 2% | ok | 0.8x (10%) |
| factor | lq | f64 | 512x512 | 43.436 ms | 4.12 | 0.10 | 1% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 470.415 ms | 0.95 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.040 ms | 1.01 | 4.03 | 8%\* | ok | 3.7x (47%) |
| factor | inv | f64 | 512x512 | 77.869 ms | 3.45 | 0.05 | 1% | ok | 0.8x (10%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.787 ms | 11.49 | 0.54 | 2% | ok | 0.7x (9%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 3.958 ms | 5.65 | 0.26 | 1% | ok | 0.4x (5%) |
| eigen | eigen_symmetric | f64 | 256x256 | 118.701 ms | 1.27 | 0.01 | 0% | ok | 1.1x (14%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 490.934 ms | 0.31 | 0.00 | 0% | ok | 1.1x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 21.976 ms | 7.63 | 0.05 | 1% | ok | 0.8x (10%) |
| eigen | schur | f64 | 256x256 | 27.047 ms | 15.51 | 0.04 | 2% | ok | 0.8x (10%) |
| eigen | svd | f64 | 256x256 | 12.713 ms | 29.03 | 0.08 | 5% | ok | 0.7x (9%) |
| eigen | expm | f64 | 256x256 | 17.500 ms | 14.38 | 0.06 | 2% | ok | 0.9x (11%) |
| eigen | sylvester | f64 | 256x256 | 88.511 ms | 11.37 | 0.02 | 2% | ok | 0.8x (11%) |
| elementwise | exp | f64 | n=8388608 | 32.825 ms | 0.26 | 4.09 | 8% | ok | 0.8x (10%) |
| elementwise | log | f64 | n=8388608 | 46.594 ms | 0.18 | 2.88 | 5% | ok | 0.9x (11%) |
| elementwise | sin | f64 | n=8388608 | 41.862 ms | 0.20 | 3.21 | 6% | ok | 0.9x (11%) |
| elementwise | sqrt | f64 | n=8388608 | 23.179 ms | 0.36 | 5.79 | 11% | ok | 0.9x (11%) |
| elementwise | erf | f64 | n=8388608 | 38.259 ms | 0.22 | 3.51 | 7% | ok | 0.9x (11%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.201 ms | 13.01 | 135.34 | 252% ! | ok | 4.3x (54%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.355 ms | 1.93 | 20.09 | 37% | ok | 4.4x (55%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.308 ms | 18.15 | 38.15 | 71% | ok | 2.8x (36%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 17.215 ms | 0.19 | 0.67 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.884 ms | 1.07 | 8.83 | 16%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 20.911 ms | 0.53 | 0.03 | 0% | ok | 0.9x (11%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 33.408 ms | 0.33 | 0.02 | 0% | ok | 0.9x (11%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 17.233 ms | 0.65 | 0.04 | 0% | ok | 0.8x (10%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 57.354 ms | 0.39 | 0.01 | 0% | ok | 0.9x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 65.051 ms | 0.34 | 0.01 | 0% | ok | 0.9x (11%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 28.479 ms | 0.79 | 0.03 | 0% | ok | 0.9x (11%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 41.663 ms | 0.05 | 0.00 | 0%\* | ok | 0.9x (11%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.006 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.116 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 63.916 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.252 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.926 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.334 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 15.315 ms | 0.73 | 0.05 | 0% | ok | 0.9x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.310 ms | 2.44 | 9.76 | 18%\* | ok | 0.9x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 7.174 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 6.570 ms | 1.70 | 0.11 | 0% | ok | 0.9x (11%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.299 ms | 2.53 | 10.10 | 19%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.623 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 53.034 ms | 0.42 | 0.01 | 0% | ok | 0.9x (11%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.448 ms | 0.52 | 2.09 | 4%\* | ok | 0.8x (10%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.132 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 55.583 ms | 0.40 | 0.01 | 0% | ok | 0.9x (11%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.401 ms | 0.54 | 2.16 | 4%\* | ok | 0.8x (10%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.822 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 270.058 ms | 1.14 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.282 ms | 1.48 | 5.90 | 11%\* | ok | 0.9x (11%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 16.935 ms | 0.00 | 0.00 | n/a | ok | 0.8x (10%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 80.480 ms | 3.83 | 0.01 | 1% | ok | 0.9x (11%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.414 ms | 1.40 | 5.58 | 10%\* | ok | 0.8x (10%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.988 ms | 0.00 | 0.00 | n/a | ok | 0.7x (8%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1205.107 ms | 0.51 | 0.00 | 0% | ok | 0.8x (9%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 12.290 ms | 0.27 | 1.10 | 2%\* | ok | 0.9x (11%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.894 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 545.278 ms | 1.13 | 0.00 | 0% | ok | 0.9x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.922 ms | 0.31 | 1.23 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 41.132 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 87.624 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 33.109 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 18.069 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 23.085 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 18.474 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 60.680 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 36.340 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 20.850 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 28.299 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 26.769 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 28.022 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 37.166 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.050 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.595 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| pc | ilut | f64 | 128^2 grid | 25.513 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| pc | ic_0 | f64 | 128^2 grid | 1.930 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| pc | ildl | f64 | 128^2 grid | 1.862 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.025 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.965 ms | 0.00 | 0.00 | n/a | ok | 0.7x (9%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.845 ms | 1.42 | 15.89 | 30% | ok | 0.9x (11%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.618 ms | 1.62 | 18.13 | 34% | ok | 0.9x (12%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.853 ms | 1.41 | 15.83 | 30% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.736 ms | 2.44 | 18.47 | 34% | ok | 1.0x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 33.935 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 34.739 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 78.944 ms | 0.00 | 0.00 | n/a | ok | 0.7x (8%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.418 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 11.959 ms | 1.40 | 5.61 | 10% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.546 ms | 59.04 | 0.92 | 9% | ok | 6.5x (81%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.583 ms | 10.60 | 11.26 | 21% | ok | 0.9x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.416 ms | 4.91 | 19.64 | n/a | ok | 0.9x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.039 ms | 5.52 | 11.04 | n/a | ok | 0.9x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.479 ms | 11.34 | 11.34 | n/a | ok | 0.9x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.479 ms | 11.34 | 11.34 | n/a | ok | 0.9x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.365 ms | 196.60 | 2.30 | n/a | ok | 6.1x (77%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.724 ms | 56.83 | 0.44 | n/a | ok | 6.4x (81%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.239 ms | 63.32 | 0.37 | n/a | ok | 6.5x (81%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.235 ms | 63.39 | 0.37 | n/a | ok | 6.5x (81%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.272 ms | 62.83 | 0.37 | n/a | ok | 6.4x (80%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

