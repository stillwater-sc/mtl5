# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,8, CPUs 0,2,4,6,8,10,12,14, clock policy power scheme copy of Balanced with processor state 100-100% and EPP 0, boost aggressive; AC, 2026-11-28T01:58:21Z.

## ryzen-9-8945hs -- native, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 67.3 GFLOP/s, f32 126.6 GFLOP/s |
| ... hardware, AVX-512F | f64 67.3, f32 126.6 GFLOP/s |
| ... this build, scalar | f64 1.6, f32 1.9 GFLOP/s |
| bandwidth | read 36.9, update 51.5, triad 37.7 GB/s (ceiling 51.5) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 20% | 97% | 0 |
| blas2 | 10 | 38% | 76% | 0 |
| blas3 | 10 | 12% | 21% | 0 |
| factor | 12 | 3% | 5% | 0 |
| eigen | 8 | 4% | 10% | 0 |
| elementwise | 5 | 7% | 10% | 0 |
| sparse | 5 | 17% | 52% | 0 |
| direct | 19 | 1% | 15% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 12% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 30% | 34% | 0 |
| mg | 1 | 28% | 28% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 11% | 19% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 14.391 ms | 1.17 | 9.33 | 18% | ok |
| blas1 | dot | f32 | n=8388608 | 12.524 ms | 1.34 | 5.36 | 10% | ok |
| blas1 | axpy | f64 | n=8388608 | 14.964 ms | 1.12 | 13.45 | 26% | ok |
| blas1 | axpy | f32 | n=8388608 | 14.930 ms | 1.12 | 6.74 | 13% | ok |
| blas1 | scale | f64 | n=8388608 | 2.685 ms | 3.12 | 49.98 | 97% | ok |
| blas1 | two_norm | f64 | n=8388608 | 13.881 ms | 1.21 | 4.83 | 9% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.401 ms | 1.31 | 10.48 | 20% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.420 ms | 1.31 | 10.45 | 20% | ok |
| blas1 | sum | f64 | n=8388608 | 6.400 ms | 1.31 | 10.49 | 20% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 111.678 ms | 1.65 | 1.20 | 2% | ok |
| blas2 | gemv | f64 | n=4096 | 3.426 ms | 9.79 | 39.20 | 76% | ok |
| blas2 | gemv | f32 | n=4096 | 3.392 ms | 9.89 | 19.80 | 38% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.784 ms | 4.31 | 17.25 | 34% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.369 ms | 14.17 | 28.35 | 55% | ok |
| blas2 | symv | f64 | n=4096 | 3.936 ms | 8.53 | 17.07 | 33% | ok |
| blas2 | trmv | f64 | n=4096 | 2.048 ms | 8.19 | 32.80 | 64% | ok |
| blas2 | trmv | f32 | n=4096 | 1.913 ms | 8.77 | 17.56 | 34% | ok |
| blas2 | trsv | f64 | n=4096 | 2.078 ms | 8.07 | 32.33 | 63% | ok |
| blas2 | trsv | f32 | n=4096 | 1.905 ms | 8.81 | 17.63 | 34% | ok |
| blas2 | ger | f64 | n=4096 | 16.330 ms | 2.05 | 16.44 | 32% | ok |
| blas3 | gemm | f64 | 512x512x512 | 19.407 ms | 13.83 | 0.32 | 21% | ok |
| blas3 | gemm | f32 | 512x512x512 | 9.964 ms | 26.94 | 0.32 | 21% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 39.454 ms | 3.40 | 0.16 | 5% | ok |
| blas3 | symm | f64 | 512x512x512 | 20.361 ms | 13.18 | 0.31 | 20% | ok |
| blas3 | syrk | f64 | 512x512x512 | 16.916 ms | 7.93 | 0.25 | 12% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 28.780 ms | 9.33 | 0.22 | 14% | ok |
| blas3 | trmm | f64 | 512x512x512 | 31.445 ms | 4.27 | 0.17 | 6% | ok |
| blas3 | trsm | f64 | 512x512x512 | 35.773 ms | 3.75 | 0.15 | 6% | ok |
| blas3 | trsm | f32 | 512x512x512 | 32.170 ms | 4.17 | 0.08 | 3% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.297 ms | 0.52 | 4.12 | 8% | ok |
| factor | lu | f64 | 512x512 | 40.679 ms | 2.20 | 0.10 | 3% | ok |
| factor | lu | f32 | 512x512 | 38.156 ms | 2.35 | 0.05 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 14.469 ms | 3.09 | 0.29 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 13.732 ms | 3.26 | 0.15 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 14.468 ms | 3.09 | 0.29 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 13.619 ms | 3.29 | 0.31 | 5% | ok |
| factor | qr | f64 | 512x512 | 815.187 ms | 0.22 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 82.454 ms | 2.17 | 0.05 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 955.920 ms | 0.47 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.915 ms | 0.21 | 0.85 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 118.338 ms | 2.27 | 0.04 | 3% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 41.903 ms | 2.14 | 0.10 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 49.183 ms | 0.45 | 0.02 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 162.778 ms | 0.93 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 650.150 ms | 0.23 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 72.152 ms | 2.33 | 0.01 | 3% | ok |
| eigen | schur | f64 | 256x256 | 149.440 ms | 2.81 | 0.01 | 4% | ok |
| eigen | svd | f64 | 256x256 | 150.833 ms | 2.45 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 37.005 ms | 6.80 | 0.03 | 10% | ok |
| eigen | sylvester | f64 | 256x256 | 337.215 ms | 2.99 | 0.00 | 4% | ok |
| elementwise | exp | f64 | n=8388608 | 34.068 ms | 0.25 | 3.94 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 51.005 ms | 0.16 | 2.63 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 48.114 ms | 0.17 | 2.79 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 25.618 ms | 0.33 | 5.24 | 10% | ok |
| elementwise | erf | f64 | n=8388608 | 38.091 ms | 0.22 | 3.52 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.018 ms | 2.57 | 26.74 | 52% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 7.627 ms | 0.34 | 3.57 | 7% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 7.927 ms | 5.28 | 11.11 | 22% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.152 ms | 0.16 | 0.57 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.890 ms | 1.06 | 8.81 | 17%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 23.763 ms | 0.47 | 0.03 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 39.091 ms | 0.29 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 20.171 ms | 0.55 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 66.201 ms | 0.34 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 70.561 ms | 0.32 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 29.991 ms | 0.75 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 44.241 ms | 0.04 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.608 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 7.034 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 75.795 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.238 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 18.008 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.656 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.584 ms | 0.64 | 0.04 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.392 ms | 1.93 | 7.72 | 15%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.184 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.219 ms | 1.55 | 0.10 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.389 ms | 1.94 | 7.77 | 15%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.209 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 57.104 ms | 0.39 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.602 ms | 0.47 | 1.89 | 4%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.022 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.097 ms | 0.38 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.003 ms | 0.75 | 3.02 | 6%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 7.127 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 390.663 ms | 0.79 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.171 ms | 1.55 | 6.21 | 12%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.832 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 84.707 ms | 3.64 | 0.01 | 5% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.142 ms | 1.57 | 6.29 | 12%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.619 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1118.594 ms | 0.55 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.230 ms | 0.30 | 1.20 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.781 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 629.588 ms | 0.98 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.254 ms | 0.30 | 1.20 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 51.579 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 78.438 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 41.195 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 22.087 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 28.691 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 23.066 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 74.694 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 45.110 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.865 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 35.030 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 33.294 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 34.755 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 46.967 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.056 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 3.033 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 28.169 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.299 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.209 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.037 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.802 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.688 ms | 1.55 | 17.38 | 34% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.926 ms | 1.36 | 15.22 | 30% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.259 ms | 1.16 | 12.98 | 25% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 19.986 ms | 1.92 | 14.54 | 28% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.899 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.417 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 74.531 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 10.120 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 14.394 ms | 1.17 | 4.66 | 9% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 36.858 ms | 7.28 | 0.11 | 11% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.834 ms | 9.15 | 9.72 | 19% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.045 ms | 4.15 | 16.59 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.241 ms | 5.18 | 10.35 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.728 ms | 9.71 | 9.71 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.728 ms | 9.71 | 9.71 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 10.043 ms | 26.73 | 0.31 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 38.453 ms | 6.98 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 34.751 ms | 7.72 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 34.644 ms | 7.75 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 34.523 ms | 7.78 | 0.05 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## ryzen-9-8945hs -- native, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 504.0 GFLOP/s, f32 1010.5 GFLOP/s |
| ... hardware, AVX-512F | f64 504.0, f32 1010.5 GFLOP/s |
| ... this build, scalar | f64 12.5, f32 14.5 GFLOP/s |
| bandwidth | read 52.5, update 53.1, triad 40.8 GB/s (ceiling 53.1) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 82% | 124% | 0 | 53% |
| blas2 | 10 | 61% | 132% | 0 | 13% |
| blas3 | 10 | 2% | 21% | 0 | 13% |
| factor | 12 | 1% | 6% | 0 | 20% |
| eigen | 8 | 0% | 2% | 0 | 13% |
| elementwise | 5 | 7% | 10% | 0 | 13% |
| sparse | 5 | 28% | 103% | 0 | 25% |
| direct | 19 | 0% | 16% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 10% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 35% | 0 | 13% |
| mg | 1 | 29% | 29% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 11% | 18% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 95% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.305 ms | 7.28 | 58.22 | 110% | ok | 6.2x (78%) |
| blas1 | dot | f32 | n=8388608 | 1.545 ms | 10.86 | 43.45 | 82% | ok | 8.1x (101%) |
| blas1 | axpy | f64 | n=8388608 | 3.546 ms | 4.73 | 56.78 | 107% | ok | 4.2x (53%) |
| blas1 | axpy | f32 | n=8388608 | 1.928 ms | 8.70 | 52.21 | 98% | ok | 7.7x (97%) |
| blas1 | scale | f64 | n=8388608 | 2.043 ms | 4.11 | 65.69 | 124% | ok | 1.3x (16%) |
| blas1 | two_norm | f64 | n=8388608 | 1.748 ms | 9.60 | 38.38 | 72% | ok | 7.9x (99%) |
| blas1 | one_norm | f64 | n=8388608 | 6.448 ms | 1.30 | 10.41 | 20% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.461 ms | 1.30 | 10.39 | 20% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 6.444 ms | 1.30 | 10.41 | 20% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 104.095 ms | 1.77 | 1.29 | 2% | ok | 1.1x (13%) |
| blas2 | gemv | f64 | n=4096 | 2.426 ms | 13.83 | 55.35 | 104% | ok | 1.4x (18%) |
| blas2 | gemv | f32 | n=4096 | 0.959 ms | 34.98 | 69.99 | 132% | ok | 3.5x (44%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.342 ms | 14.33 | 57.33 | 108% | ok | 3.3x (42%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.133 ms | 29.61 | 59.25 | 112% | ok | 2.1x (26%) |
| blas2 | symv | f64 | n=4096 | 3.799 ms | 8.83 | 17.68 | 33% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 2.108 ms | 7.96 | 31.87 | 60% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=4096 | 1.893 ms | 8.86 | 17.74 | 33% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 2.071 ms | 8.10 | 32.44 | 61% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.896 ms | 8.85 | 17.71 | 33% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 16.397 ms | 2.05 | 16.38 | 31% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.524 ms | 106.37 | 2.49 | 21% | ok | 7.7x (96%) |
| blas3 | gemm | f32 | 512x512x512 | 1.260 ms | 213.08 | 2.50 | 21% | ok | 7.9x (99%) |
| blas3 | gemmt | f64 | 512x512x512 | 14.203 ms | 9.45 | 0.44 | 2% | ok | 2.8x (35%) |
| blas3 | symm | f64 | 512x512x512 | 20.244 ms | 13.26 | 0.31 | 3% | ok | 1.0x (13%) |
| blas3 | syrk | f64 | 512x512x512 | 17.038 ms | 7.88 | 0.25 | 2% | ok | 1.0x (12%) |
| blas3 | syr2k | f64 | 512x512x512 | 28.918 ms | 9.28 | 0.22 | 2% | ok | 1.0x (12%) |
| blas3 | trmm | f64 | 512x512x512 | 30.836 ms | 4.35 | 0.17 | 1% | ok | 1.0x (13%) |
| blas3 | trsm | f64 | 512x512x512 | 35.426 ms | 3.79 | 0.15 | 1% | ok | 1.0x (13%) |
| blas3 | trsm | f32 | 512x512x512 | 32.233 ms | 4.16 | 0.08 | 0% | ok | 1.0x (12%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.317 ms | 0.51 | 4.12 | 8% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 25.944 ms | 3.45 | 0.16 | 1% | ok | 1.6x (20%) |
| factor | lu | f32 | 512x512 | 24.427 ms | 3.66 | 0.09 | 0% | ok | 1.6x (20%) |
| factor | cholesky | f64 | 512x512 | 14.342 ms | 3.12 | 0.29 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 13.762 ms | 3.25 | 0.15 | 0% | ok | 1.0x (12%) |
| factor | ldlt | f64 | 512x512 | 14.466 ms | 3.09 | 0.29 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 12.478 ms | 3.59 | 0.34 | 1% | ok | 1.1x (14%) |
| factor | qr | f64 | 512x512 | 499.604 ms | 0.36 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 51.137 ms | 3.50 | 0.08 | 1% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 570.661 ms | 0.78 | 0.01 | 0% | ok | 1.7x (21%) |
| factor | householder | f64 | 512x512 | 1.251 ms | 0.84 | 3.35 | 6%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 96.983 ms | 2.77 | 0.04 | 1% | ok | 1.2x (15%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 27.640 ms | 3.24 | 0.15 | 1% | ok | 1.5x (19%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 48.730 ms | 0.46 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 166.325 ms | 0.91 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 622.937 ms | 0.24 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 75.904 ms | 2.21 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | schur | f64 | 256x256 | 154.491 ms | 2.71 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 150.758 ms | 2.45 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 22.382 ms | 11.24 | 0.05 | 2% | ok | 1.7x (21%) |
| eigen | sylvester | f64 | 256x256 | 332.534 ms | 3.03 | 0.00 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 32.784 ms | 0.26 | 4.09 | 8% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 50.838 ms | 0.17 | 2.64 | 5% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 47.996 ms | 0.17 | 2.80 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 25.582 ms | 0.33 | 5.25 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 38.104 ms | 0.22 | 3.52 | 7% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.501 ms | 5.23 | 54.40 | 103% | ok | 2.0x (25%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.866 ms | 1.40 | 14.59 | 28% | ok | 4.1x (51%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.281 ms | 18.36 | 38.60 | 73% | ok | 3.5x (43%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 19.693 ms | 0.17 | 0.58 | 1%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.933 ms | 1.04 | 8.61 | 16%\* | ok | 1.0x (12%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 23.862 ms | 0.47 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 38.906 ms | 0.29 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 19.096 ms | 0.59 | 0.04 | 0% | ok | 1.1x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 66.214 ms | 0.34 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 69.308 ms | 0.32 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 30.576 ms | 0.73 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 43.304 ms | 0.04 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.705 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 7.034 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 75.754 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.242 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.927 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.692 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.461 ms | 0.64 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.348 ms | 2.17 | 8.70 | 16%\* | ok | 1.1x (14%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.181 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 6.987 ms | 1.60 | 0.10 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.371 ms | 2.04 | 8.16 | 15%\* | ok | 1.0x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.067 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 57.715 ms | 0.39 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.391 ms | 0.54 | 2.17 | 4%\* | ok | 1.2x (14%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.943 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.981 ms | 0.37 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.646 ms | 0.46 | 1.84 | 3%\* | ok | 0.6x (8%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 7.009 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 362.872 ms | 0.85 | 0.00 | 0% | ok | 1.1x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.551 ms | 1.32 | 5.28 | 10%\* | ok | 0.9x (11%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.354 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 85.560 ms | 3.61 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.634 ms | 1.28 | 5.12 | 10%\* | ok | 0.8x (10%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.535 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1120.050 ms | 0.55 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.588 ms | 0.29 | 1.16 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.650 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 709.485 ms | 0.87 | 0.00 | 0% | ok | 0.9x (11%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.447 ms | 0.29 | 1.18 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 51.605 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 78.981 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 40.939 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 22.150 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 28.604 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 23.009 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 76.210 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 45.032 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.806 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 34.945 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 33.050 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 34.613 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 45.655 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.054 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.972 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 28.315 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 2.207 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 2.138 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.036 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.803 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.589 ms | 1.65 | 18.46 | 35% | ok | 1.1x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.930 ms | 1.36 | 15.20 | 29% | ok | 1.0x (12%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.252 ms | 1.16 | 13.02 | 25% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.994 ms | 2.02 | 15.30 | 29% | ok | 1.1x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 41.041 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.483 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 70.446 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 10.046 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 14.471 ms | 1.16 | 4.64 | 9% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.808 ms | 55.83 | 0.87 | 11% | ok | 7.7x (96%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.834 ms | 9.15 | 9.72 | 18% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.055 ms | 4.14 | 16.55 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.238 ms | 5.18 | 10.36 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.728 ms | 9.71 | 9.71 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.727 ms | 9.72 | 9.72 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.319 ms | 203.53 | 2.39 | n/a | ok | 7.6x (95%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.033 ms | 53.34 | 0.42 | n/a | ok | 7.6x (96%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.522 ms | 59.37 | 0.35 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.522 ms | 59.36 | 0.35 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.528 ms | 59.28 | 0.35 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## ryzen-9-8945hs -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 63.4 GFLOP/s, f32 126.5 GFLOP/s |
| ... hardware, AVX-512F | f64 63.4, f32 126.5 GFLOP/s |
| ... this build, SSE2 | f64 40.8, f32 81.4 GFLOP/s |
| bandwidth | read 44.4, update 48.5, triad 37.4 GB/s (ceiling 48.5) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 111% | 119% | 0 |
| blas2 | 10 | 66% | 77% | 0 |
| blas3 | 10 | 23% | 30% | 0 |
| factor | 12 | 3% | 6% | 0 |
| eigen | 8 | 4% | 12% | 0 |
| elementwise | 5 | 7% | 11% | 0 |
| sparse | 5 | 22% | 54% | 0 |
| direct | 19 | 1% | 8% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 13% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 32% | 39% | 0 |
| mg | 1 | 30% | 30% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 64% | 64% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 2.516 ms | 6.67 | 53.34 | 110% | ok |
| blas1 | dot | f32 | n=8388608 | 1.249 ms | 13.43 | 53.73 | 111% | ok |
| blas1 | axpy | f64 | n=8388608 | 3.648 ms | 4.60 | 55.19 | 114% | ok |
| blas1 | axpy | f32 | n=8388608 | 1.753 ms | 9.57 | 57.42 | 119% | ok |
| blas1 | scale | f64 | n=8388608 | 2.344 ms | 3.58 | 57.25 | 118% | ok |
| blas1 | two_norm | f64 | n=8388608 | 1.169 ms | 14.35 | 57.40 | 118% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.399 ms | 1.31 | 10.49 | 22% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.406 ms | 1.31 | 10.48 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 6.423 ms | 1.31 | 10.45 | 22% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 101.404 ms | 1.82 | 1.32 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.580 ms | 9.37 | 37.51 | 77% | ok |
| blas2 | gemv | f32 | n=4096 | 1.801 ms | 18.63 | 37.27 | 77% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 4.230 ms | 7.93 | 31.75 | 66% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.316 ms | 14.49 | 28.99 | 60% | ok |
| blas2 | symv | f64 | n=4096 | 3.606 ms | 9.31 | 18.63 | 38% | ok |
| blas2 | trmv | f64 | n=4096 | 2.078 ms | 8.07 | 32.32 | 67% | ok |
| blas2 | trmv | f32 | n=4096 | 1.143 ms | 14.67 | 29.38 | 61% | ok |
| blas2 | trsv | f64 | n=4096 | 2.098 ms | 8.00 | 32.02 | 66% | ok |
| blas2 | trsv | f32 | n=4096 | 1.226 ms | 13.69 | 27.41 | 57% | ok |
| blas2 | ger | f64 | n=4096 | 16.269 ms | 2.06 | 16.50 | 34% | ok |
| blas3 | gemm | f64 | 512x512x512 | 14.016 ms | 19.15 | 0.45 | 30% | ok |
| blas3 | gemm | f32 | 512x512x512 | 6.961 ms | 38.56 | 0.45 | 30% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 9.395 ms | 14.29 | 0.67 | 23% | ok |
| blas3 | symm | f64 | 512x512x512 | 15.194 ms | 17.67 | 0.41 | 28% | ok |
| blas3 | syrk | f64 | 512x512x512 | 9.299 ms | 14.43 | 0.45 | 23% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 18.056 ms | 14.87 | 0.35 | 23% | ok |
| blas3 | trmm | f64 | 512x512x512 | 11.804 ms | 11.37 | 0.44 | 18% | ok |
| blas3 | trsm | f64 | 512x512x512 | 12.012 ms | 11.17 | 0.44 | 18% | ok |
| blas3 | trsm | f32 | 512x512x512 | 7.988 ms | 16.80 | 0.33 | 13% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.558 ms | 0.50 | 4.02 | 8% | ok |
| factor | lu | f64 | 512x512 | 40.443 ms | 2.21 | 0.10 | 3% | ok |
| factor | lu | f32 | 512x512 | 38.978 ms | 2.30 | 0.05 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 14.299 ms | 3.13 | 0.29 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 13.714 ms | 3.26 | 0.15 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 14.436 ms | 3.10 | 0.29 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 12.679 ms | 3.53 | 0.33 | 6% | ok |
| factor | qr | f64 | 512x512 | 823.534 ms | 0.22 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 81.683 ms | 2.19 | 0.05 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 939.169 ms | 0.48 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.888 ms | 0.21 | 0.86 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 110.040 ms | 2.44 | 0.04 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 44.106 ms | 2.03 | 0.10 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 48.625 ms | 0.46 | 0.02 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 162.039 ms | 0.93 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 643.712 ms | 0.23 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 75.110 ms | 2.23 | 0.01 | 4% | ok |
| eigen | schur | f64 | 256x256 | 147.411 ms | 2.85 | 0.01 | 4% | ok |
| eigen | svd | f64 | 256x256 | 151.419 ms | 2.44 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 32.272 ms | 7.80 | 0.03 | 12% | ok |
| eigen | sylvester | f64 | 256x256 | 329.892 ms | 3.05 | 0.00 | 5% | ok |
| elementwise | exp | f64 | n=8388608 | 36.458 ms | 0.23 | 3.68 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 50.928 ms | 0.16 | 2.64 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 48.061 ms | 0.17 | 2.79 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 25.489 ms | 0.33 | 5.27 | 11% | ok |
| elementwise | erf | f64 | n=8388608 | 37.700 ms | 0.22 | 3.56 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.042 ms | 2.51 | 26.12 | 54% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.555 ms | 1.68 | 17.51 | 36% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 8.177 ms | 5.12 | 10.77 | 22% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 19.800 ms | 0.16 | 0.58 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.947 ms | 1.03 | 8.55 | 18%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 23.915 ms | 0.47 | 0.03 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 36.053 ms | 0.31 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 17.899 ms | 0.62 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 65.004 ms | 0.34 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 69.871 ms | 0.32 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 29.862 ms | 0.75 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 44.265 ms | 0.04 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.559 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 7.061 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 78.159 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.231 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.596 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.614 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.522 ms | 0.64 | 0.04 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.764 ms | 0.99 | 3.96 | 8%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.117 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.076 ms | 1.58 | 0.10 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.948 ms | 0.80 | 3.19 | 7%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.177 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 58.854 ms | 0.38 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.591 ms | 0.48 | 1.90 | 4%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.093 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.207 ms | 0.38 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.125 ms | 0.67 | 2.69 | 6%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 7.066 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 348.877 ms | 0.88 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.189 ms | 1.54 | 6.16 | 13%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 16.935 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 86.666 ms | 3.56 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.239 ms | 1.50 | 6.02 | 12%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.795 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1147.161 ms | 0.54 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.218 ms | 0.30 | 1.20 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.569 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 624.397 ms | 0.99 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.209 ms | 0.30 | 1.20 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 41.372 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 58.803 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 36.163 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.102 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.707 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 20.044 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 43.385 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 37.052 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 21.947 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 23.689 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 22.209 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 22.369 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 27.403 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.058 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.929 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 28.658 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.245 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.141 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.036 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.822 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.567 ms | 1.67 | 18.71 | 39% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.911 ms | 1.37 | 15.35 | 32% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.236 ms | 1.17 | 13.11 | 27% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 19.845 ms | 1.94 | 14.65 | 30% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 49.780 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.512 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 71.936 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.075 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 2.175 ms | 7.71 | 30.86 | 64% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 19.334 ms | 13.88 | 0.22 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.572 ms | 29.33 | 31.16 | 64% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.479 ms | 11.35 | 45.39 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.599 ms | 28.02 | 56.05 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.430 ms | 38.98 | 38.98 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.444 ms | 37.79 | 37.79 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 19.127 ms | 14.03 | 0.16 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 22.103 ms | 12.14 | 0.09 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 22.445 ms | 11.96 | 0.07 | n/a | ok |
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

## ryzen-9-8945hs -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 78.8 GFLOP/s, f32 126.6 GFLOP/s |
| ... hardware, AVX-512F | f64 78.8, f32 126.6 GFLOP/s |
| ... this build, scalar | f64 1.9, f32 1.9 GFLOP/s |
| bandwidth | read 36.6, update 51.5, triad 37.8 GB/s (ceiling 51.5) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 20% | 99% | 0 |
| blas2 | 10 | 75% | 90% | 0 |
| blas3 | 10 | 64% | 88% | 0 |
| factor | 12 | 10% | 22% | 0 |
| eigen | 8 | 14% | 43% | 0 |
| elementwise | 5 | 6% | 10% | 0 |
| sparse | 5 | 21% | 59% | 0 |
| direct | 19 | 1% | 7% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 12% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 23% | 34% | 0 |
| mg | 1 | 28% | 28% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 11% | 19% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.775 ms | 1.42 | 11.40 | 22% | ok |
| blas1 | dot | f32 | n=8388608 | 12.240 ms | 1.37 | 5.48 | 11% | ok |
| blas1 | axpy | f64 | n=8388608 | 14.965 ms | 1.12 | 13.45 | 26% | ok |
| blas1 | axpy | f32 | n=8388608 | 15.067 ms | 1.11 | 6.68 | 13% | ok |
| blas1 | scale | f64 | n=8388608 | 2.642 ms | 3.18 | 50.81 | 99% | ok |
| blas1 | two_norm | f64 | n=8388608 | 12.264 ms | 1.37 | 5.47 | 11% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.399 ms | 1.31 | 10.49 | 20% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.410 ms | 1.31 | 10.47 | 20% | ok |
| blas1 | sum | f64 | n=8388608 | 6.396 ms | 1.31 | 10.49 | 20% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 101.404 ms | 1.82 | 1.32 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.278 ms | 10.24 | 40.97 | 79% | ok |
| blas2 | gemv | f32 | n=4096 | 1.633 ms | 20.55 | 41.11 | 80% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.940 ms | 8.52 | 34.08 | 66% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.443 ms | 23.26 | 46.54 | 90% | ok |
| blas2 | symv | f64 | n=4096 | 1.985 ms | 16.90 | 33.83 | 66% | ok |
| blas2 | trmv | f64 | n=4096 | 1.747 ms | 9.60 | 38.45 | 75% | ok |
| blas2 | trmv | f32 | n=4096 | 0.928 ms | 18.08 | 36.19 | 70% | ok |
| blas2 | trsv | f64 | n=4096 | 1.753 ms | 9.57 | 38.33 | 74% | ok |
| blas2 | trsv | f32 | n=4096 | 0.946 ms | 17.73 | 35.49 | 69% | ok |
| blas2 | ger | f64 | n=4096 | 6.640 ms | 5.05 | 40.44 | 78% | ok |
| blas3 | gemm | f64 | 512x512x512 | 4.660 ms | 57.60 | 1.35 | 73% | ok |
| blas3 | gemm | f32 | 512x512x512 | 2.412 ms | 111.27 | 1.30 | 88% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 3.566 ms | 37.64 | 1.76 | 48% | ok |
| blas3 | symm | f64 | 512x512x512 | 4.860 ms | 55.23 | 1.29 | 70% | ok |
| blas3 | syrk | f64 | 512x512x512 | 3.883 ms | 34.57 | 1.08 | 44% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 6.237 ms | 43.04 | 1.01 | 55% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.409 ms | 55.72 | 2.18 | 71% | ok |
| blas3 | trsm | f64 | 512x512x512 | 3.542 ms | 37.89 | 1.48 | 48% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.646 ms | 81.52 | 1.59 | 64% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.276 ms | 0.52 | 4.13 | 8% | ok |
| factor | lu | f64 | 512x512 | 6.321 ms | 14.16 | 0.66 | 18% | ok |
| factor | lu | f32 | 512x512 | 3.156 ms | 28.35 | 0.66 | 22% | ok |
| factor | cholesky | f64 | 512x512 | 5.456 ms | 8.20 | 0.77 | 10% | ok |
| factor | cholesky | f32 | 512x512 | 2.266 ms | 19.74 | 0.93 | 16% | ok |
| factor | ldlt | f64 | 512x512 | 14.818 ms | 3.02 | 0.28 | 4% | ok |
| factor | ldlt_bk | f64 | 512x512 | 12.431 ms | 3.60 | 0.34 | 5% | ok |
| factor | qr | f64 | 512x512 | 12.936 ms | 13.83 | 0.32 | 18% | ok |
| factor | lq | f64 | 512x512 | 80.965 ms | 2.21 | 0.05 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 941.376 ms | 0.48 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.896 ms | 0.21 | 0.86 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 76.172 ms | 3.52 | 0.06 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.957 ms | 12.86 | 0.60 | 16% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 2.205 ms | 10.14 | 0.48 | 13% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 162.599 ms | 0.93 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 630.339 ms | 0.24 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 21.223 ms | 7.91 | 0.05 | 10% | ok |
| eigen | schur | f64 | 256x256 | 27.656 ms | 15.17 | 0.04 | 19% | ok |
| eigen | svd | f64 | 256x256 | 10.896 ms | 33.87 | 0.10 | 43% | ok |
| eigen | expm | f64 | 256x256 | 19.609 ms | 12.83 | 0.05 | 16% | ok |
| eigen | sylvester | f64 | 256x256 | 92.602 ms | 10.87 | 0.02 | 14% | ok |
| elementwise | exp | f64 | n=8388608 | 32.560 ms | 0.26 | 4.12 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 52.011 ms | 0.16 | 2.58 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 47.070 ms | 0.18 | 2.85 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 25.593 ms | 0.33 | 5.24 | 10% | ok |
| elementwise | erf | f64 | n=8388608 | 41.695 ms | 0.20 | 3.22 | 6% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.894 ms | 2.93 | 30.47 | 59% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.545 ms | 1.69 | 17.63 | 34% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 7.952 ms | 5.27 | 11.07 | 21% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 19.446 ms | 0.17 | 0.59 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.937 ms | 1.04 | 8.59 | 17%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 24.259 ms | 0.46 | 0.03 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 40.765 ms | 0.27 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 20.218 ms | 0.55 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 65.444 ms | 0.34 | 0.01 | 0% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 70.356 ms | 0.32 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 30.014 ms | 0.75 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 43.238 ms | 0.04 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.577 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 7.086 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 75.477 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.242 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.824 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.598 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.523 ms | 0.64 | 0.04 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.959 ms | 0.79 | 3.15 | 6%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.055 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.010 ms | 1.60 | 0.10 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.896 ms | 0.84 | 3.37 | 7%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.178 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 57.803 ms | 0.39 | 0.01 | 0% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.608 ms | 0.47 | 1.88 | 4%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.014 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 58.973 ms | 0.38 | 0.01 | 0% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.381 ms | 0.55 | 2.19 | 4%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 6.975 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 338.853 ms | 0.91 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.182 ms | 1.54 | 6.18 | 12%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.204 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 85.718 ms | 3.60 | 0.01 | 5% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.266 ms | 1.49 | 5.95 | 12%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.856 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1120.360 ms | 0.55 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.214 ms | 0.30 | 1.20 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.740 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 639.654 ms | 0.96 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.274 ms | 0.30 | 1.20 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 49.498 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 74.664 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 40.050 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.178 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 27.530 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.454 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 68.623 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 43.784 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.119 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.252 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 31.447 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 32.544 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.784 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.054 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.925 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 27.869 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.222 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.137 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.036 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.664 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.691 ms | 1.55 | 17.34 | 34% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.437 ms | 1.07 | 12.04 | 23% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.847 ms | 0.92 | 10.30 | 20% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 20.130 ms | 1.91 | 14.44 | 28% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 43.177 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 43.899 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 73.262 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.826 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 12.256 ms | 1.37 | 5.48 | 11% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 37.760 ms | 7.11 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.835 ms | 9.14 | 9.71 | 19% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.051 ms | 4.14 | 16.56 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.246 ms | 5.17 | 10.34 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.123 ms | 7.90 | 7.90 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.124 ms | 7.90 | 7.90 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 10.240 ms | 26.21 | 0.31 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 38.355 ms | 7.00 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 34.581 ms | 7.76 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 34.815 ms | 7.71 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 34.571 ms | 7.76 | 0.05 | n/a | ok |

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

