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

