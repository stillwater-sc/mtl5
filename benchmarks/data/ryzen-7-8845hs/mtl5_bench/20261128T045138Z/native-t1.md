## ryzen-7-8845hs -- native, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 80.1 GFLOP/s, f32 160.1 GFLOP/s |
| ... hardware, AVX-512F | f64 80.1, f32 160.1 GFLOP/s |
| ... this build, scalar | f64 1.6, f32 1.9 GFLOP/s |
| bandwidth | read 43.3, update 53.6, triad 39.1 GB/s (ceiling 53.6) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 25% | 109% | 0 |
| blas2 | 10 | 55% | 82% | 0 |
| blas3 | 10 | 15% | 20% | 0 |
| factor | 12 | 3% | 5% | 0 |
| eigen | 8 | 4% | 11% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 25% | 57% | 0 |
| direct | 19 | 1% | 20% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 13% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 33% | 36% | 0 |
| mg | 1 | 33% | 33% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 12% | 23% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.469 ms | 1.46 | 11.70 | 22% | ok |
| blas1 | dot | f32 | n=8388608 | 9.782 ms | 1.72 | 6.86 | 13% | ok |
| blas1 | axpy | f64 | n=8388608 | 11.908 ms | 1.41 | 16.91 | 32% | ok |
| blas1 | axpy | f32 | n=8388608 | 11.889 ms | 1.41 | 8.47 | 16% | ok |
| blas1 | scale | f64 | n=8388608 | 2.288 ms | 3.67 | 58.65 | 109% | ok |
| blas1 | two_norm | f64 | n=8388608 | 10.594 ms | 1.58 | 6.33 | 12% | ok |
| blas1 | one_norm | f64 | n=8388608 | 5.061 ms | 1.66 | 13.26 | 25% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 5.073 ms | 1.65 | 13.23 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 5.061 ms | 1.66 | 13.26 | 25% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 92.443 ms | 2.00 | 1.45 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.042 ms | 11.03 | 44.15 | 82% | ok |
| blas2 | gemv | f32 | n=4096 | 2.283 ms | 14.70 | 29.41 | 55% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.065 ms | 4.75 | 19.01 | 35% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.207 ms | 15.20 | 30.42 | 57% | ok |
| blas2 | symv | f64 | n=4096 | 2.822 ms | 11.89 | 23.80 | 44% | ok |
| blas2 | trmv | f64 | n=4096 | 1.688 ms | 9.94 | 39.79 | 74% | ok |
| blas2 | trmv | f32 | n=4096 | 1.484 ms | 11.30 | 22.63 | 42% | ok |
| blas2 | trsv | f64 | n=4096 | 1.714 ms | 9.79 | 39.19 | 73% | ok |
| blas2 | trsv | f32 | n=4096 | 1.505 ms | 11.15 | 22.32 | 42% | ok |
| blas2 | ger | f64 | n=4096 | 13.519 ms | 2.48 | 19.86 | 37% | ok |
| blas3 | gemm | f64 | 512x512x512 | 16.478 ms | 16.29 | 0.38 | 20% | ok |
| blas3 | gemm | f32 | 512x512x512 | 8.235 ms | 32.60 | 0.38 | 20% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 11.320 ms | 11.86 | 0.56 | 15% | ok |
| blas3 | symm | f64 | 512x512x512 | 17.036 ms | 15.76 | 0.37 | 20% | ok |
| blas3 | syrk | f64 | 512x512x512 | 13.551 ms | 9.90 | 0.31 | 12% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 22.779 ms | 11.78 | 0.28 | 15% | ok |
| blas3 | trmm | f64 | 512x512x512 | 25.092 ms | 5.35 | 0.21 | 7% | ok |
| blas3 | trsm | f64 | 512x512x512 | 26.707 ms | 5.03 | 0.20 | 6% | ok |
| blas3 | trsm | f32 | 512x512x512 | 25.969 ms | 5.17 | 0.10 | 3% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.196 ms | 0.58 | 4.62 | 9% | ok |
| factor | lu | f64 | 512x512 | 32.711 ms | 2.74 | 0.13 | 3% | ok |
| factor | lu | f32 | 512x512 | 31.041 ms | 2.88 | 0.07 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 11.350 ms | 3.94 | 0.37 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 10.894 ms | 4.11 | 0.19 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 11.432 ms | 3.91 | 0.37 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 10.228 ms | 4.37 | 0.41 | 5% | ok |
| factor | qr | f64 | 512x512 | 639.962 ms | 0.28 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 64.637 ms | 2.77 | 0.06 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 738.391 ms | 0.61 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 3.906 ms | 0.27 | 1.07 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 89.159 ms | 3.01 | 0.05 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 33.928 ms | 2.64 | 0.12 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 37.645 ms | 0.59 | 0.03 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 128.335 ms | 1.18 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 522.008 ms | 0.29 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 59.466 ms | 2.82 | 0.02 | 4% | ok |
| eigen | schur | f64 | 256x256 | 110.522 ms | 3.80 | 0.01 | 5% | ok |
| eigen | svd | f64 | 256x256 | 117.815 ms | 3.13 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 29.835 ms | 8.43 | 0.04 | 11% | ok |
| eigen | sylvester | f64 | 256x256 | 259.733 ms | 3.88 | 0.01 | 5% | ok |
| elementwise | exp | f64 | n=8388608 | 27.585 ms | 0.30 | 4.87 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 41.941 ms | 0.20 | 3.20 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 39.378 ms | 0.21 | 3.41 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 21.331 ms | 0.39 | 6.29 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 31.669 ms | 0.26 | 4.24 | 8% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.892 ms | 2.93 | 30.53 | 57% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.322 ms | 1.98 | 20.60 | 38% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.471 ms | 6.47 | 13.61 | 25% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 18.263 ms | 0.18 | 0.63 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.692 ms | 1.19 | 9.84 | 18%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 18.845 ms | 0.59 | 0.04 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 30.850 ms | 0.36 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 15.525 ms | 0.72 | 0.05 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 53.620 ms | 0.42 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 57.981 ms | 0.39 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 24.636 ms | 0.91 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 37.651 ms | 0.05 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.649 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.655 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 59.700 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.226 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.169 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.915 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.199 ms | 0.79 | 0.05 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.279 ms | 2.71 | 10.85 | 20%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.542 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 5.981 ms | 1.87 | 0.12 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.289 ms | 2.62 | 10.48 | 20%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.316 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.644 ms | 0.47 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.142 ms | 0.66 | 2.65 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.757 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 49.203 ms | 0.45 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.116 ms | 0.68 | 2.71 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.567 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 253.130 ms | 1.22 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.994 ms | 1.69 | 6.76 | 13%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 13.766 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 68.540 ms | 4.50 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.031 ms | 1.66 | 6.63 | 12%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.156 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 883.164 ms | 0.70 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.438 ms | 0.32 | 1.29 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.543 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 495.209 ms | 1.25 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.386 ms | 0.32 | 1.30 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 40.388 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 61.005 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 32.127 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.299 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.444 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.993 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 58.391 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 35.016 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 20.180 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 26.997 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 25.647 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 26.829 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 35.547 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.043 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.407 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 22.336 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.816 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.737 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.029 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.652 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.652 ms | 1.58 | 17.75 | 33% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.527 ms | 1.71 | 19.20 | 36% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.829 ms | 1.43 | 16.04 | 30% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 16.215 ms | 2.37 | 17.93 | 33% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.445 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.823 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 51.936 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.974 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 11.383 ms | 1.47 | 5.90 | 11% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 29.078 ms | 9.23 | 0.14 | 12% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.469 ms | 11.42 | 12.13 | 23% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.197 ms | 5.25 | 20.99 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.881 ms | 5.82 | 11.64 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.388 ms | 12.08 | 12.08 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.389 ms | 12.08 | 12.08 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 8.072 ms | 33.25 | 0.39 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 30.303 ms | 8.86 | 0.07 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 27.287 ms | 9.84 | 0.06 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 27.513 ms | 9.76 | 0.06 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 27.287 ms | 9.84 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

