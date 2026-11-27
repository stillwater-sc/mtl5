# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,8, CPUs 0,2,4,6,8,10,12,14, clock policy power scheme copy of Balanced with processor state 100-100% and EPP 0, boost aggressive; AC, 2026-11-27T14:48:53Z.

## ryzen-9-8945hs -- native, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `57512adcb8fe` |
| FMA peak (ceiling) | f64 77.8 GFLOP/s, f32 126.0 GFLOP/s |
| ... hardware, AVX-512F | f64 77.8, f32 126.0 GFLOP/s |
| ... this build, scalar | f64 1.9, f32 1.9 GFLOP/s |
| bandwidth | read 33.4, update 46.6, triad 32.8 GB/s (ceiling 46.6) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 23% | 109% | 0 |
| blas2 | 10 | 52% | 87% | 0 |
| blas3 | 10 | 12% | 21% | 0 |
| factor | 12 | 3% | 5% | 0 |
| eigen | 8 | 3% | 9% | 0 |
| elementwise | 5 | 8% | 11% | 0 |
| sparse | 5 | 26% | 64% | 0 |
| direct | 19 | 1% | 18% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 3% | 13% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 34% | 38% | 0 |
| mg | 1 | 32% | 32% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 12% | 21% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 12.266 ms | 1.37 | 10.94 | 24% | ok |
| blas1 | dot | f32 | n=8388608 | 12.255 ms | 1.37 | 5.48 | 12% | ok |
| blas1 | axpy | f64 | n=8388608 | 13.403 ms | 1.25 | 15.02 | 32% | ok |
| blas1 | axpy | f32 | n=8388608 | 12.788 ms | 1.31 | 7.87 | 17% | ok |
| blas1 | scale | f64 | n=8388608 | 2.636 ms | 3.18 | 50.91 | 109% | ok |
| blas1 | two_norm | f64 | n=8388608 | 11.204 ms | 1.50 | 5.99 | 13% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.400 ms | 1.31 | 10.49 | 23% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.413 ms | 1.31 | 10.46 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 6.393 ms | 1.31 | 10.50 | 23% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 108.704 ms | 1.70 | 1.23 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.299 ms | 10.17 | 40.71 | 87% | ok |
| blas2 | gemv | f32 | n=4096 | 2.757 ms | 12.17 | 24.35 | 52% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.092 ms | 4.73 | 18.94 | 41% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.328 ms | 14.41 | 28.84 | 62% | ok |
| blas2 | symv | f64 | n=4096 | 3.718 ms | 9.03 | 18.07 | 39% | ok |
| blas2 | trmv | f64 | n=4096 | 2.044 ms | 8.21 | 32.87 | 71% | ok |
| blas2 | trmv | f32 | n=4096 | 1.878 ms | 8.93 | 17.88 | 38% | ok |
| blas2 | trsv | f64 | n=4096 | 2.054 ms | 8.17 | 32.71 | 70% | ok |
| blas2 | trsv | f32 | n=4096 | 1.902 ms | 8.82 | 17.65 | 38% | ok |
| blas2 | ger | f64 | n=4096 | 16.354 ms | 2.05 | 16.42 | 35% | ok |
| blas3 | gemm | f64 | 512x512x512 | 19.480 ms | 13.78 | 0.32 | 18% | ok |
| blas3 | gemm | f32 | 512x512x512 | 10.000 ms | 26.84 | 0.31 | 21% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 13.809 ms | 9.72 | 0.46 | 12% | ok |
| blas3 | symm | f64 | 512x512x512 | 19.769 ms | 13.58 | 0.32 | 17% | ok |
| blas3 | syrk | f64 | 512x512x512 | 16.730 ms | 8.02 | 0.25 | 10% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 28.317 ms | 9.48 | 0.22 | 12% | ok |
| blas3 | trmm | f64 | 512x512x512 | 35.227 ms | 3.81 | 0.15 | 5% | ok |
| blas3 | trsm | f64 | 512x512x512 | 32.592 ms | 4.12 | 0.16 | 5% | ok |
| blas3 | trsm | f32 | 512x512x512 | 35.566 ms | 3.77 | 0.07 | 3% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.328 ms | 0.51 | 4.11 | 9% | ok |
| factor | lu | f64 | 512x512 | 40.499 ms | 2.21 | 0.10 | 3% | ok |
| factor | lu | f32 | 512x512 | 38.127 ms | 2.35 | 0.06 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 14.384 ms | 3.11 | 0.29 | 4% | ok |
| factor | cholesky | f32 | 512x512 | 13.726 ms | 3.26 | 0.15 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 14.451 ms | 3.10 | 0.29 | 4% | ok |
| factor | ldlt_bk | f64 | 512x512 | 12.511 ms | 3.58 | 0.34 | 5% | ok |
| factor | qr | f64 | 512x512 | 819.690 ms | 0.22 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 81.669 ms | 2.19 | 0.05 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 944.321 ms | 0.47 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.918 ms | 0.21 | 0.85 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 111.310 ms | 2.41 | 0.04 | 3% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 42.356 ms | 2.11 | 0.10 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 49.317 ms | 0.45 | 0.02 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 163.593 ms | 0.92 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 633.582 ms | 0.24 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 75.987 ms | 2.21 | 0.01 | 3% | ok |
| eigen | schur | f64 | 256x256 | 141.796 ms | 2.96 | 0.01 | 4% | ok |
| eigen | svd | f64 | 256x256 | 150.711 ms | 2.45 | 0.01 | 3% | ok |
| eigen | expm | f64 | 256x256 | 37.030 ms | 6.80 | 0.03 | 9% | ok |
| eigen | sylvester | f64 | 256x256 | 328.580 ms | 3.06 | 0.00 | 4% | ok |
| elementwise | exp | f64 | n=8388608 | 32.601 ms | 0.26 | 4.12 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 50.907 ms | 0.16 | 2.64 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 47.148 ms | 0.18 | 2.85 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 25.626 ms | 0.33 | 5.24 | 11% | ok |
| elementwise | erf | f64 | n=8388608 | 37.938 ms | 0.22 | 3.54 | 8% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.911 ms | 2.87 | 29.88 | 64% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.561 ms | 1.68 | 17.44 | 37% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 7.290 ms | 5.74 | 12.08 | 26% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 19.834 ms | 0.16 | 0.58 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.992 ms | 1.01 | 8.36 | 18%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 23.503 ms | 0.48 | 0.03 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 38.705 ms | 0.29 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 19.387 ms | 0.58 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 65.361 ms | 0.34 | 0.01 | 0% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 69.397 ms | 0.32 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 30.057 ms | 0.74 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 44.948 ms | 0.04 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.523 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.941 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 75.232 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.244 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.970 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.602 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.549 ms | 0.64 | 0.04 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.365 ms | 2.07 | 8.29 | 18%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.035 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 6.924 ms | 1.62 | 0.11 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.364 ms | 2.08 | 8.30 | 18%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.048 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 57.537 ms | 0.39 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.323 ms | 0.57 | 2.29 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.939 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.261 ms | 0.38 | 0.01 | 0% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.314 ms | 0.58 | 2.30 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 7.016 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 354.374 ms | 0.87 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.205 ms | 1.53 | 6.11 | 13%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.376 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 84.752 ms | 3.64 | 0.01 | 5% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.417 ms | 1.39 | 5.58 | 12%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.581 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1129.848 ms | 0.55 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.178 ms | 0.30 | 1.21 | 3%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.471 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 618.183 ms | 1.00 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.190 ms | 0.30 | 1.20 | 3%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 50.157 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 105.501 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 39.990 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.339 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 27.504 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.430 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 69.202 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 43.547 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.138 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 32.694 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 31.173 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 32.628 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.451 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.066 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.925 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 28.190 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.215 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.166 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.029 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.838 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.671 ms | 1.57 | 17.55 | 38% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.849 ms | 1.42 | 15.86 | 34% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.213 ms | 1.18 | 13.25 | 28% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 19.573 ms | 1.96 | 14.85 | 32% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.829 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.376 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 64.649 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.640 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 12.253 ms | 1.37 | 5.48 | 12% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 36.834 ms | 7.29 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.833 ms | 9.15 | 9.72 | 21% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.052 ms | 4.14 | 16.56 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.243 ms | 5.17 | 10.35 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.726 ms | 9.72 | 9.72 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.726 ms | 9.72 | 9.72 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 9.884 ms | 27.16 | 0.32 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 38.474 ms | 6.98 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 34.626 ms | 7.75 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 34.688 ms | 7.74 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 34.564 ms | 7.77 | 0.05 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## ryzen-9-8945hs -- native, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `57512adcb8fe` |
| FMA peak (ceiling) | f64 501.0 GFLOP/s, f32 1004.2 GFLOP/s |
| ... hardware, AVX-512F | f64 501.0, f32 1004.2 GFLOP/s |
| ... this build, scalar | f64 14.7, f32 13.3 GFLOP/s |
| bandwidth | read 52.3, update 52.9, triad 41.2 GB/s (ceiling 52.9) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 90% | 126% | 0 | 47% |
| blas2 | 10 | 63% | 113% | 0 | 13% |
| blas3 | 10 | 2% | 21% | 0 | 12% |
| factor | 12 | 1% | 6% | 0 | 19% |
| eigen | 8 | 0% | 2% | 0 | 13% |
| elementwise | 5 | 7% | 10% | 0 | 13% |
| sparse | 5 | 23% | 284% | 0 | 13% |
| direct | 19 | 0% | 17% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 11% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 26% | 30% | 0 | 12% |
| mg | 1 | 30% | 30% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 11% | 18% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 95% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.293 ms | 7.32 | 58.53 | 111% | ok | 5.3x (67%) |
| blas1 | dot | f32 | n=8388608 | 1.541 ms | 10.89 | 43.55 | 82% | ok | 8.0x (99%) |
| blas1 | axpy | f64 | n=8388608 | 3.544 ms | 4.73 | 56.82 | 107% | ok | 3.8x (47%) |
| blas1 | axpy | f32 | n=8388608 | 1.769 ms | 9.49 | 56.92 | 108% | ok | 7.2x (90%) |
| blas1 | scale | f64 | n=8388608 | 2.018 ms | 4.16 | 66.52 | 126% | ok | 1.3x (16%) |
| blas1 | two_norm | f64 | n=8388608 | 1.409 ms | 11.90 | 47.61 | 90% | ok | 7.9x (99%) |
| blas1 | one_norm | f64 | n=8388608 | 6.400 ms | 1.31 | 10.49 | 20% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.415 ms | 1.31 | 10.46 | 20% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 6.399 ms | 1.31 | 10.49 | 20% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 110.232 ms | 1.67 | 1.22 | 2% | ok | 1.0x (12%) |
| blas2 | gemv | f64 | n=4096 | 2.426 ms | 13.83 | 55.36 | 105% | ok | 1.4x (17%) |
| blas2 | gemv | f32 | n=4096 | 1.123 ms | 29.88 | 59.79 | 113% | ok | 2.5x (31%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.332 ms | 14.39 | 57.58 | 109% | ok | 3.0x (38%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.153 ms | 29.11 | 58.25 | 110% | ok | 2.0x (25%) |
| blas2 | symv | f64 | n=4096 | 3.570 ms | 9.40 | 18.81 | 36% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 2.054 ms | 8.17 | 32.71 | 62% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=4096 | 1.761 ms | 9.53 | 19.07 | 36% | ok | 1.1x (13%) |
| blas2 | trsv | f64 | n=4096 | 2.028 ms | 8.27 | 33.12 | 63% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.800 ms | 9.32 | 18.66 | 35% | ok | 1.1x (13%) |
| blas2 | ger | f64 | n=4096 | 16.322 ms | 2.06 | 16.45 | 31% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 2.523 ms | 106.40 | 2.49 | 21% | ok | 7.7x (97%) |
| blas3 | gemm | f32 | 512x512x512 | 1.260 ms | 212.98 | 2.50 | 21% | ok | 7.9x (99%) |
| blas3 | gemmt | f64 | 512x512x512 | 13.959 ms | 9.62 | 0.45 | 2% | ok | 1.0x (12%) |
| blas3 | symm | f64 | 512x512x512 | 20.229 ms | 13.27 | 0.31 | 3% | ok | 1.0x (12%) |
| blas3 | syrk | f64 | 512x512x512 | 16.819 ms | 7.98 | 0.25 | 2% | ok | 1.0x (12%) |
| blas3 | syr2k | f64 | 512x512x512 | 28.464 ms | 9.43 | 0.22 | 2% | ok | 1.0x (12%) |
| blas3 | trmm | f64 | 512x512x512 | 37.080 ms | 3.62 | 0.14 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f64 | 512x512x512 | 33.922 ms | 3.96 | 0.15 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f32 | 512x512x512 | 35.199 ms | 3.81 | 0.07 | 0% | ok | 1.0x (13%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.565 ms | 0.50 | 4.02 | 8% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 25.914 ms | 3.45 | 0.16 | 1% | ok | 1.6x (20%) |
| factor | lu | f32 | 512x512 | 24.548 ms | 3.65 | 0.09 | 0% | ok | 1.6x (19%) |
| factor | cholesky | f64 | 512x512 | 14.382 ms | 3.11 | 0.29 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 13.724 ms | 3.26 | 0.15 | 0% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 14.472 ms | 3.09 | 0.29 | 1% | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 12.680 ms | 3.53 | 0.33 | 1% | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 501.112 ms | 0.36 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 51.927 ms | 3.45 | 0.08 | 1% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 576.186 ms | 0.78 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.251 ms | 0.84 | 3.35 | 6%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 97.073 ms | 2.77 | 0.04 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 28.162 ms | 3.18 | 0.15 | 1% | ok | 1.5x (19%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 47.703 ms | 0.47 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 161.536 ms | 0.93 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 620.388 ms | 0.24 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 70.726 ms | 2.37 | 0.01 | 0% | ok | 1.1x (13%) |
| eigen | schur | f64 | 256x256 | 146.965 ms | 2.85 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 152.128 ms | 2.43 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | expm | f64 | 256x256 | 22.227 ms | 11.32 | 0.05 | 2% | ok | 1.7x (21%) |
| eigen | sylvester | f64 | 256x256 | 325.296 ms | 3.09 | 0.00 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 32.590 ms | 0.26 | 4.12 | 8% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 50.756 ms | 0.17 | 2.64 | 5% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 50.257 ms | 0.17 | 2.67 | 5% | ok | 0.9x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 25.457 ms | 0.33 | 5.27 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 38.384 ms | 0.22 | 3.50 | 7% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.181 ms | 14.43 | 150.11 | 284% ! | ok | 5.0x (63%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.266 ms | 1.16 | 12.02 | 23% | ok | 0.7x (9%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.302 ms | 18.20 | 38.26 | 72% | ok | 3.2x (40%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.401 ms | 0.16 | 0.56 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.852 ms | 1.08 | 8.99 | 17%\* | ok | 1.1x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 23.846 ms | 0.47 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 38.835 ms | 0.29 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 19.430 ms | 0.58 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 64.922 ms | 0.34 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 82.637 ms | 0.27 | 0.01 | 0% | ok | 0.8x (10%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 34.286 ms | 0.65 | 0.02 | 0% | ok | 0.9x (11%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 43.250 ms | 0.04 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.608 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.913 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 76.117 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.242 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.949 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.691 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.713 ms | 0.63 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.344 ms | 2.20 | 8.79 | 17%\* | ok | 1.1x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.173 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.027 ms | 1.59 | 0.10 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.347 ms | 2.18 | 8.71 | 16%\* | ok | 1.0x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.039 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 60.428 ms | 0.37 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.331 ms | 0.57 | 2.27 | 4%\* | ok | 1.0x (12%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.900 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 60.038 ms | 0.37 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.309 ms | 0.58 | 2.31 | 4%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 6.972 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 342.741 ms | 0.90 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.259 ms | 1.49 | 5.97 | 11%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 16.913 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 84.388 ms | 3.66 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.319 ms | 1.45 | 5.81 | 11%\* | ok | 1.0x (13%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.534 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1138.679 ms | 0.54 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.295 ms | 0.30 | 1.19 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.482 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 623.518 ms | 0.99 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.480 ms | 0.29 | 1.17 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 49.766 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 102.939 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 39.746 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.163 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 27.497 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.350 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 69.018 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 43.169 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.997 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 32.826 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 30.986 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 32.264 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.230 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.054 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.962 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 27.813 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 2.201 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 2.153 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.030 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.837 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.092 ms | 1.25 | 14.02 | 26% | ok | 0.8x (10%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.849 ms | 1.42 | 15.86 | 30% | ok | 1.0x (12%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.238 ms | 1.17 | 13.11 | 25% | ok | 1.0x (12%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.364 ms | 2.09 | 15.83 | 30% | ok | 1.1x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.839 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.343 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 68.275 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.617 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 12.245 ms | 1.37 | 5.48 | 10% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.811 ms | 55.80 | 0.87 | 11% | ok | 7.7x (96%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.835 ms | 9.14 | 9.71 | 18% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.960 ms | 3.38 | 13.53 | n/a | ok | 0.8x (10%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.246 ms | 5.17 | 10.34 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.727 ms | 9.72 | 9.72 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.726 ms | 9.72 | 9.72 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.246 ms | 215.52 | 2.53 | n/a | ok | 7.9x (99%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.029 ms | 53.38 | 0.42 | n/a | ok | 7.7x (96%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.541 ms | 59.12 | 0.35 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.555 ms | 58.93 | 0.35 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.553 ms | 58.95 | 0.35 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

## ryzen-9-8945hs -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `57512adcb8fe` |
| FMA peak (ceiling) | f64 63.3 GFLOP/s, f32 126.3 GFLOP/s |
| ... hardware, AVX-512F | f64 63.3, f32 126.3 GFLOP/s |
| ... this build, SSE2 | f64 40.7, f32 81.5 GFLOP/s |
| bandwidth | read 45.3, update 51.2, triad 37.8 GB/s (ceiling 51.2) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 106% | 119% | 0 |
| blas2 | 10 | 65% | 74% | 0 |
| blas3 | 10 | 23% | 31% | 0 |
| factor | 12 | 3% | 5% | 0 |
| eigen | 8 | 4% | 12% | 0 |
| elementwise | 5 | 7% | 10% | 0 |
| sparse | 5 | 21% | 57% | 0 |
| direct | 19 | 1% | 17% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 12% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 31% | 34% | 0 |
| mg | 1 | 29% | 29% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 60% | 61% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 2.473 ms | 6.78 | 54.27 | 106% | ok |
| blas1 | dot | f32 | n=8388608 | 1.252 ms | 13.39 | 53.58 | 105% | ok |
| blas1 | axpy | f64 | n=8388608 | 3.590 ms | 4.67 | 56.08 | 109% | ok |
| blas1 | axpy | f32 | n=8388608 | 1.654 ms | 10.14 | 60.87 | 119% | ok |
| blas1 | scale | f64 | n=8388608 | 2.325 ms | 3.61 | 57.74 | 113% | ok |
| blas1 | two_norm | f64 | n=8388608 | 1.188 ms | 14.12 | 56.48 | 110% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.402 ms | 1.31 | 10.48 | 20% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.411 ms | 1.31 | 10.47 | 20% | ok |
| blas1 | sum | f64 | n=8388608 | 6.397 ms | 1.31 | 10.49 | 20% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 101.564 ms | 1.82 | 1.32 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.539 ms | 9.48 | 37.94 | 74% | ok |
| blas2 | gemv | f32 | n=4096 | 1.888 ms | 17.78 | 35.57 | 69% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 4.014 ms | 8.36 | 33.45 | 65% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.171 ms | 15.45 | 30.92 | 60% | ok |
| blas2 | symv | f64 | n=4096 | 3.455 ms | 9.71 | 19.44 | 38% | ok |
| blas2 | trmv | f64 | n=4096 | 1.902 ms | 8.82 | 35.31 | 69% | ok |
| blas2 | trmv | f32 | n=4096 | 0.994 ms | 16.87 | 33.78 | 66% | ok |
| blas2 | trsv | f64 | n=4096 | 2.054 ms | 8.17 | 32.71 | 64% | ok |
| blas2 | trsv | f32 | n=4096 | 1.007 ms | 16.66 | 33.35 | 65% | ok |
| blas2 | ger | f64 | n=4096 | 16.272 ms | 2.06 | 16.50 | 32% | ok |
| blas3 | gemm | f64 | 512x512x512 | 13.827 ms | 19.41 | 0.46 | 31% | ok |
| blas3 | gemm | f32 | 512x512x512 | 6.970 ms | 38.51 | 0.45 | 30% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 9.322 ms | 14.40 | 0.67 | 23% | ok |
| blas3 | symm | f64 | 512x512x512 | 15.019 ms | 17.87 | 0.42 | 28% | ok |
| blas3 | syrk | f64 | 512x512x512 | 9.311 ms | 14.41 | 0.45 | 23% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 18.255 ms | 14.71 | 0.34 | 23% | ok |
| blas3 | trmm | f64 | 512x512x512 | 11.805 ms | 11.37 | 0.44 | 18% | ok |
| blas3 | trsm | f64 | 512x512x512 | 12.439 ms | 10.79 | 0.42 | 17% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.381 ms | 16.01 | 0.31 | 13% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.527 ms | 0.50 | 4.03 | 8% | ok |
| factor | lu | f64 | 512x512 | 40.555 ms | 2.21 | 0.10 | 3% | ok |
| factor | lu | f32 | 512x512 | 37.722 ms | 2.37 | 0.06 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 14.326 ms | 3.12 | 0.29 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 13.796 ms | 3.24 | 0.15 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 14.441 ms | 3.10 | 0.29 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 12.902 ms | 3.47 | 0.33 | 5% | ok |
| factor | qr | f64 | 512x512 | 817.831 ms | 0.22 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 81.421 ms | 2.20 | 0.05 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 947.925 ms | 0.47 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.940 ms | 0.21 | 0.85 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 110.972 ms | 2.42 | 0.04 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 41.913 ms | 2.13 | 0.10 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 48.172 ms | 0.46 | 0.02 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 163.483 ms | 0.92 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 628.141 ms | 0.24 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 75.788 ms | 2.21 | 0.01 | 3% | ok |
| eigen | schur | f64 | 256x256 | 146.649 ms | 2.86 | 0.01 | 5% | ok |
| eigen | svd | f64 | 256x256 | 150.286 ms | 2.46 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 32.233 ms | 7.81 | 0.03 | 12% | ok |
| eigen | sylvester | f64 | 256x256 | 339.140 ms | 2.97 | 0.00 | 5% | ok |
| elementwise | exp | f64 | n=8388608 | 32.669 ms | 0.26 | 4.11 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 50.761 ms | 0.17 | 2.64 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 47.294 ms | 0.18 | 2.84 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 25.536 ms | 0.33 | 5.26 | 10% | ok |
| elementwise | erf | f64 | n=8388608 | 38.203 ms | 0.22 | 3.51 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.933 ms | 2.81 | 29.20 | 57% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.585 ms | 1.65 | 17.18 | 34% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 8.050 ms | 5.20 | 10.94 | 21% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 19.314 ms | 0.17 | 0.60 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 2.036 ms | 0.99 | 8.18 | 16%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 23.159 ms | 0.48 | 0.03 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 36.202 ms | 0.31 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 18.050 ms | 0.62 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 65.923 ms | 0.34 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 69.336 ms | 0.32 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 29.746 ms | 0.75 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 43.971 ms | 0.04 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.576 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.952 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 76.886 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.234 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.598 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.603 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.473 ms | 0.64 | 0.04 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.365 ms | 2.07 | 8.30 | 16%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.053 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 6.922 ms | 1.62 | 0.11 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.347 ms | 2.18 | 8.71 | 17%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.057 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 60.237 ms | 0.37 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.301 ms | 0.58 | 2.33 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.898 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 58.674 ms | 0.38 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.185 ms | 0.64 | 2.55 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 6.923 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 352.182 ms | 0.88 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.272 ms | 1.48 | 5.93 | 12%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.015 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 84.309 ms | 3.66 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.309 ms | 1.46 | 5.84 | 11%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.558 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1140.251 ms | 0.54 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.144 ms | 0.30 | 1.21 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.656 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 620.578 ms | 0.99 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.166 ms | 0.30 | 1.21 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 41.507 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 78.146 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 35.892 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.137 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.763 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 20.079 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 43.766 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 37.574 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.052 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 23.762 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 22.305 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 22.390 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 27.406 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.055 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.921 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 28.980 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.204 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.118 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.030 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.817 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.694 ms | 1.55 | 17.32 | 34% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.845 ms | 1.42 | 15.90 | 31% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.741 ms | 0.96 | 10.70 | 21% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 19.835 ms | 1.94 | 14.65 | 29% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.869 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.532 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 65.479 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.066 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 2.188 ms | 7.67 | 30.68 | 60% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 19.303 ms | 13.91 | 0.22 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.574 ms | 29.22 | 31.05 | 61% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.464 ms | 11.46 | 45.84 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.594 ms | 28.24 | 56.49 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.430 ms | 39.04 | 39.04 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.385 ms | 43.62 | 43.62 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 18.889 ms | 14.21 | 0.17 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 21.985 ms | 12.21 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 22.476 ms | 11.94 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.580 ms | 58.61 | 0.34 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.553 ms | 58.96 | 0.35 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## ryzen-9-8945hs -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `57512adcb8fe` |
| FMA peak (ceiling) | f64 460.4 GFLOP/s, f32 1005.2 GFLOP/s |
| ... hardware, AVX-512F | f64 460.4, f32 1005.2 GFLOP/s |
| ... this build, SSE2 | f64 319.8, f32 638.1 GFLOP/s |
| bandwidth | read 52.6, update 52.8, triad 40.6 GB/s (ceiling 52.8) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 112% | 127% | 0 | 13% |
| blas2 | 10 | 67% | 121% | 0 | 13% |
| blas3 | 10 | 10% | 21% | 0 | 39% |
| factor | 12 | 1% | 6% | 0 | 19% |
| eigen | 8 | 1% | 2% | 0 | 13% |
| elementwise | 5 | 7% | 10% | 0 | 13% |
| sparse | 5 | 34% | 291% | 0 | 13% |
| direct | 19 | 0% | 17% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 11% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 10% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 28% | 30% | 0 | 13% |
| mg | 1 | 30% | 30% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 58% | 59% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 74% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.266 ms | 7.40 | 59.24 | 112% | ok | 1.1x (14%) |
| blas1 | dot | f32 | n=8388608 | 1.012 ms | 16.58 | 66.31 | 126% | ok | 1.2x (15%) |
| blas1 | axpy | f64 | n=8388608 | 3.471 ms | 4.83 | 58.00 | 110% | ok | 1.0x (13%) |
| blas1 | axpy | f32 | n=8388608 | 1.618 ms | 10.37 | 62.22 | 118% | ok | 1.0x (13%) |
| blas1 | scale | f64 | n=8388608 | 2.003 ms | 4.19 | 67.02 | 127% | ok | 1.2x (15%) |
| blas1 | two_norm | f64 | n=8388608 | 1.012 ms | 16.58 | 66.31 | 126% | ok | 1.2x (15%) |
| blas1 | one_norm | f64 | n=8388608 | 6.397 ms | 1.31 | 10.49 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.408 ms | 1.31 | 10.47 | 20% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 6.394 ms | 1.31 | 10.50 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 109.407 ms | 1.69 | 1.23 | 2% | ok | 0.9x (12%) |
| blas2 | gemv | f64 | n=4096 | 2.306 ms | 14.55 | 58.24 | 110% | ok | 1.5x (19%) |
| blas2 | gemv | f32 | n=4096 | 1.049 ms | 31.99 | 64.01 | 121% | ok | 1.8x (22%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.332 ms | 14.39 | 57.59 | 109% | ok | 1.7x (22%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.201 ms | 27.93 | 55.89 | 106% | ok | 1.8x (23%) |
| blas2 | symv | f64 | n=4096 | 3.777 ms | 8.88 | 17.79 | 34% | ok | 0.9x (11%) |
| blas2 | trmv | f64 | n=4096 | 1.925 ms | 8.72 | 34.90 | 66% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=4096 | 1.028 ms | 16.32 | 32.68 | 62% | ok | 1.0x (12%) |
| blas2 | trsv | f64 | n=4096 | 1.908 ms | 8.79 | 35.21 | 67% | ok | 1.1x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.004 ms | 16.71 | 33.46 | 63% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 16.273 ms | 2.06 | 16.50 | 31% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.768 ms | 96.97 | 2.27 | 21% | ok | 5.0x (62%) |
| blas3 | gemm | f32 | 512x512x512 | 1.498 ms | 179.14 | 2.10 | 18% | ok | 4.7x (58%) |
| blas3 | gemmt | f64 | 512x512x512 | 4.011 ms | 33.46 | 1.57 | 7% | ok | 2.3x (29%) |
| blas3 | symm | f64 | 512x512x512 | 3.718 ms | 72.19 | 1.69 | 16% | ok | 4.0x (50%) |
| blas3 | syrk | f64 | 512x512x512 | 2.951 ms | 45.48 | 1.42 | 10% | ok | 3.2x (39%) |
| blas3 | syr2k | f64 | 512x512x512 | 5.546 ms | 48.40 | 1.13 | 11% | ok | 3.3x (41%) |
| blas3 | trmm | f64 | 512x512x512 | 6.539 ms | 20.52 | 0.80 | 4% | ok | 1.8x (23%) |
| blas3 | trsm | f64 | 512x512x512 | 7.212 ms | 18.61 | 0.73 | 4% | ok | 1.7x (22%) |
| blas3 | trsm | f32 | 512x512x512 | 5.736 ms | 23.40 | 0.46 | 2% | ok | 1.5x (18%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.240 ms | 0.52 | 4.15 | 8% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 25.892 ms | 3.46 | 0.16 | 1% | ok | 1.6x (20%) |
| factor | lu | f32 | 512x512 | 24.410 ms | 3.67 | 0.09 | 0% | ok | 1.5x (19%) |
| factor | cholesky | f64 | 512x512 | 14.349 ms | 3.12 | 0.29 | 1% | ok | 1.0x (12%) |
| factor | cholesky | f32 | 512x512 | 13.723 ms | 3.26 | 0.15 | 0% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 14.442 ms | 3.10 | 0.29 | 1% | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 12.609 ms | 3.55 | 0.33 | 1% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 500.235 ms | 0.36 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 50.586 ms | 3.54 | 0.08 | 1% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 579.084 ms | 0.77 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.253 ms | 0.84 | 3.35 | 6%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 96.576 ms | 2.78 | 0.04 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 28.231 ms | 3.17 | 0.15 | 1% | ok | 1.5x (19%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 47.694 ms | 0.47 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 163.226 ms | 0.93 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 630.339 ms | 0.24 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 256x256 | 74.373 ms | 2.26 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 147.416 ms | 2.85 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 150.546 ms | 2.45 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | expm | f64 | 256x256 | 22.154 ms | 11.36 | 0.05 | 2% | ok | 1.5x (18%) |
| eigen | sylvester | f64 | 256x256 | 330.640 ms | 3.04 | 0.00 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 32.449 ms | 0.26 | 4.14 | 8% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 50.645 ms | 0.17 | 2.65 | 5% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 47.002 ms | 0.18 | 2.86 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 25.430 ms | 0.33 | 5.28 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 38.331 ms | 0.22 | 3.50 | 7% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.177 ms | 14.77 | 153.67 | 291% ! | ok | 5.3x (66%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.518 ms | 1.72 | 17.94 | 34% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.279 ms | 18.38 | 38.64 | 73% | ok | 3.5x (44%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 19.677 ms | 0.17 | 0.58 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.990 ms | 1.01 | 8.36 | 16%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 21.989 ms | 0.51 | 0.03 | 0% | ok | 1.1x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 34.261 ms | 0.33 | 0.02 | 0% | ok | 1.1x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 15.864 ms | 0.71 | 0.05 | 0% | ok | 1.1x (14%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 68.897 ms | 0.32 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 67.626 ms | 0.33 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 30.192 ms | 0.74 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 42.979 ms | 0.04 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.627 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.997 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 74.857 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.239 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.323 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.698 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.661 ms | 0.63 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.345 ms | 2.19 | 8.77 | 17%\* | ok | 1.1x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.181 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 6.947 ms | 1.61 | 0.11 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.345 ms | 2.19 | 8.76 | 17%\* | ok | 1.0x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.061 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 57.248 ms | 0.39 | 0.01 | 0% | ok | 1.1x (13%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.282 ms | 0.59 | 2.36 | 4%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.922 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 58.644 ms | 0.38 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.541 ms | 0.49 | 1.96 | 4%\* | ok | 0.8x (10%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 7.005 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 349.585 ms | 0.88 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.306 ms | 1.46 | 5.84 | 11%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.085 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 84.563 ms | 3.65 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.391 ms | 1.41 | 5.64 | 11%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.561 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1106.888 ms | 0.56 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.276 ms | 0.30 | 1.20 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.580 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 615.742 ms | 1.00 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.280 ms | 0.30 | 1.19 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 41.471 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 83.242 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 36.045 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 20.844 ms | 0.00 | 0.00 | n/a | ok | 0.8x (10%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 27.446 ms | 0.00 | 0.00 | n/a | ok | 0.8x (10%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 24.108 ms | 0.00 | 0.00 | n/a | ok | 0.8x (10%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 55.864 ms | 0.00 | 0.00 | n/a | ok | 0.8x (10%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 43.780 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.984 ms | 0.00 | 0.00 | n/a | ok | 0.8x (11%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 24.874 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 26.646 ms | 0.00 | 0.00 | n/a | ok | 0.8x (10%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 26.795 ms | 0.00 | 0.00 | n/a | ok | 0.8x (10%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 34.301 ms | 0.00 | 0.00 | n/a | ok | 0.8x (10%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.091 ms | 0.00 | 0.00 | n/a | ok | 0.6x (8%) |
| pc | ilu_0 | f64 | 128^2 grid | 3.005 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 29.049 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 2.212 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 128^2 grid | 2.130 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.030 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.816 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.975 ms | 1.33 | 14.85 | 28% | ok | 0.9x (11%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.826 ms | 1.43 | 16.06 | 30% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.232 ms | 1.17 | 13.14 | 25% | ok | 1.2x (15%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.501 ms | 2.08 | 15.71 | 30% | ok | 1.1x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.891 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.824 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 77.187 ms | 0.00 | 0.00 | n/a | ok | 0.8x (11%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.173 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 2.175 ms | 7.71 | 30.86 | 58% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 3.089 ms | 86.89 | 1.36 | 19% | ok | 6.2x (78%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.577 ms | 29.08 | 30.89 | 59% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.547 ms | 10.85 | 43.38 | n/a | ok | 0.9x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.593 ms | 28.28 | 56.56 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.431 ms | 38.90 | 38.90 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.390 ms | 43.02 | 43.02 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.812 ms | 95.46 | 1.12 | n/a | ok | 6.7x (84%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 3.020 ms | 88.88 | 0.69 | n/a | ok | 7.3x (91%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 3.058 ms | 87.77 | 0.51 | n/a | ok | 7.3x (92%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.774 ms | 346.82 | 2.03 | n/a | ok | 5.9x (74%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.770 ms | 348.53 | 2.04 | n/a | ok | 5.9x (74%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

## ryzen-9-8945hs -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `57512adcb8fe` |
| FMA peak (ceiling) | f64 64.2 GFLOP/s, f32 136.0 GFLOP/s |
| ... hardware, AVX-512F | f64 64.2, f32 136.0 GFLOP/s |
| ... this build, scalar | f64 1.6, f32 1.9 GFLOP/s |
| bandwidth | read 36.5, update 52.0, triad 37.7 GB/s (ceiling 52.0) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 20% | 57% | 0 |
| blas2 | 10 | 75% | 88% | 0 |
| blas3 | 10 | 68% | 91% | 0 |
| factor | 12 | 13% | 22% | 0 |
| eigen | 8 | 17% | 53% | 0 |
| elementwise | 5 | 6% | 10% | 0 |
| sparse | 5 | 21% | 56% | 0 |
| direct | 19 | 1% | 17% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 12% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 27% | 33% | 0 |
| mg | 1 | 25% | 25% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 11% | 19% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 13.737 ms | 1.22 | 9.77 | 19% | ok |
| blas1 | dot | f32 | n=8388608 | 12.235 ms | 1.37 | 5.49 | 11% | ok |
| blas1 | axpy | f64 | n=8388608 | 14.982 ms | 1.12 | 13.44 | 26% | ok |
| blas1 | axpy | f32 | n=8388608 | 12.800 ms | 1.31 | 7.86 | 15% | ok |
| blas1 | scale | f64 | n=8388608 | 4.498 ms | 1.87 | 29.84 | 57% | ok |
| blas1 | two_norm | f64 | n=8388608 | 14.550 ms | 1.15 | 4.61 | 9% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.402 ms | 1.31 | 10.48 | 20% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.417 ms | 1.31 | 10.46 | 20% | ok |
| blas1 | sum | f64 | n=8388608 | 6.404 ms | 1.31 | 10.48 | 20% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 112.552 ms | 1.64 | 1.19 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.263 ms | 10.28 | 41.15 | 79% | ok |
| blas2 | gemv | f32 | n=4096 | 1.570 ms | 21.37 | 42.75 | 82% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.435 ms | 9.77 | 39.09 | 75% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.464 ms | 22.92 | 45.86 | 88% | ok |
| blas2 | symv | f64 | n=4096 | 2.112 ms | 15.88 | 31.80 | 61% | ok |
| blas2 | trmv | f64 | n=4096 | 1.900 ms | 8.83 | 35.36 | 68% | ok |
| blas2 | trmv | f32 | n=4096 | 0.999 ms | 16.80 | 33.63 | 65% | ok |
| blas2 | trsv | f64 | n=4096 | 1.921 ms | 8.73 | 34.97 | 67% | ok |
| blas2 | trsv | f32 | n=4096 | 1.083 ms | 15.49 | 31.01 | 60% | ok |
| blas2 | ger | f64 | n=4096 | 6.630 ms | 5.06 | 40.50 | 78% | ok |
| blas3 | gemm | f64 | 512x512x512 | 4.577 ms | 58.65 | 1.37 | 91% | ok |
| blas3 | gemm | f32 | 512x512x512 | 2.377 ms | 112.94 | 1.32 | 83% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 3.423 ms | 39.22 | 1.84 | 61% | ok |
| blas3 | symm | f64 | 512x512x512 | 4.756 ms | 56.44 | 1.32 | 88% | ok |
| blas3 | syrk | f64 | 512x512x512 | 3.718 ms | 36.10 | 1.13 | 56% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 6.107 ms | 43.96 | 1.03 | 68% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.361 ms | 56.85 | 2.22 | 89% | ok |
| blas3 | trsm | f64 | 512x512x512 | 3.501 ms | 38.34 | 1.50 | 60% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.644 ms | 81.66 | 1.59 | 60% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.495 ms | 0.51 | 4.05 | 8% | ok |
| factor | lu | f64 | 512x512 | 6.330 ms | 14.14 | 0.66 | 22% | ok |
| factor | lu | f32 | 512x512 | 3.039 ms | 29.44 | 0.69 | 22% | ok |
| factor | cholesky | f64 | 512x512 | 5.460 ms | 8.19 | 0.77 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 2.087 ms | 21.44 | 1.01 | 16% | ok |
| factor | ldlt | f64 | 512x512 | 14.818 ms | 3.02 | 0.28 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 12.944 ms | 3.46 | 0.32 | 5% | ok |
| factor | qr | f64 | 512x512 | 12.907 ms | 13.87 | 0.32 | 22% | ok |
| factor | lq | f64 | 512x512 | 81.412 ms | 2.20 | 0.05 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 936.525 ms | 0.48 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.910 ms | 0.21 | 0.85 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 77.050 ms | 3.48 | 0.05 | 5% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.954 ms | 12.87 | 0.60 | 20% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 2.165 ms | 10.33 | 0.48 | 16% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 165.385 ms | 0.91 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 619.665 ms | 0.24 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 20.874 ms | 8.04 | 0.05 | 13% | ok |
| eigen | schur | f64 | 256x256 | 27.233 ms | 15.40 | 0.04 | 24% | ok |
| eigen | svd | f64 | 256x256 | 10.871 ms | 33.95 | 0.10 | 53% | ok |
| eigen | expm | f64 | 256x256 | 19.572 ms | 12.86 | 0.05 | 20% | ok |
| eigen | sylvester | f64 | 256x256 | 93.066 ms | 10.82 | 0.02 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 32.660 ms | 0.26 | 4.11 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 52.815 ms | 0.16 | 2.54 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 48.000 ms | 0.17 | 2.80 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 26.007 ms | 0.32 | 5.16 | 10% | ok |
| elementwise | erf | f64 | n=8388608 | 42.014 ms | 0.20 | 3.19 | 6% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.938 ms | 2.79 | 29.02 | 56% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.534 ms | 1.71 | 17.75 | 34% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 8.086 ms | 5.18 | 10.89 | 21% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 19.651 ms | 0.17 | 0.59 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.921 ms | 1.04 | 8.67 | 17%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 23.781 ms | 0.47 | 0.03 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 38.983 ms | 0.29 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 19.219 ms | 0.58 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 64.649 ms | 0.35 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 69.652 ms | 0.32 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 29.999 ms | 0.75 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 42.887 ms | 0.04 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.542 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.977 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 75.249 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.241 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.767 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.590 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.551 ms | 0.64 | 0.04 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.349 ms | 2.17 | 8.66 | 17%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.040 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.168 ms | 1.56 | 0.10 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.351 ms | 2.16 | 8.62 | 17%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.056 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 57.298 ms | 0.39 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.302 ms | 0.58 | 2.32 | 4%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.121 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.704 ms | 0.37 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.249 ms | 0.61 | 2.42 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 6.960 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 346.051 ms | 0.89 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.222 ms | 1.52 | 6.07 | 12%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 16.816 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 84.295 ms | 3.66 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.278 ms | 1.48 | 5.92 | 11%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.597 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1119.007 ms | 0.55 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.229 ms | 0.30 | 1.20 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.415 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 635.899 ms | 0.97 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.180 ms | 0.30 | 1.21 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 51.016 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 105.781 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 40.390 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.841 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 28.439 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.889 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 72.016 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 44.605 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.629 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 34.788 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.972 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 34.118 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 44.471 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.057 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.925 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 28.710 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.224 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.145 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.029 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.695 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.696 ms | 1.54 | 17.29 | 33% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.126 ms | 1.23 | 13.80 | 27% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.300 ms | 1.14 | 12.75 | 25% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 22.634 ms | 1.70 | 12.84 | 25% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 41.174 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.332 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 67.942 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 10.106 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 14.381 ms | 1.17 | 4.67 | 9% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 37.737 ms | 7.11 | 0.11 | 11% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.836 ms | 9.14 | 9.71 | 19% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.041 ms | 4.15 | 16.61 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.314 ms | 5.06 | 10.13 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.122 ms | 7.91 | 7.91 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.122 ms | 7.90 | 7.90 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 10.411 ms | 25.78 | 0.30 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 38.426 ms | 6.99 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 34.775 ms | 7.72 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 34.777 ms | 7.72 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 34.653 ms | 7.75 | 0.05 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

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

