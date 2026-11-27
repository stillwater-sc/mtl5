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

