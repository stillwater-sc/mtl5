## ryzen-9-8945hs -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 63.3 GFLOP/s, f32 126.5 GFLOP/s |
| ... hardware, AVX-512F | f64 63.3, f32 126.5 GFLOP/s |
| ... this build, SSE2 | f64 40.8, f32 81.6 GFLOP/s |
| bandwidth | read 45.3, update 51.8, triad 37.9 GB/s (ceiling 51.8) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 107% | 112% | 0 |
| blas2 | 10 | 64% | 74% | 0 |
| blas3 | 10 | 23% | 31% | 0 |
| factor | 12 | 3% | 6% | 0 |
| eigen | 8 | 4% | 12% | 0 |
| elementwise | 5 | 7% | 10% | 0 |
| sparse | 5 | 22% | 58% | 0 |
| direct | 19 | 1% | 17% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 12% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 29% | 32% | 0 |
| mg | 1 | 28% | 28% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 59% | 60% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 2.410 ms | 6.96 | 55.68 | 107% | ok |
| blas1 | dot | f32 | n=8388608 | 1.243 ms | 13.49 | 53.98 | 104% | ok |
| blas1 | axpy | f64 | n=8388608 | 3.600 ms | 4.66 | 55.93 | 108% | ok |
| blas1 | axpy | f32 | n=8388608 | 1.738 ms | 9.65 | 57.93 | 112% | ok |
| blas1 | scale | f64 | n=8388608 | 2.317 ms | 3.62 | 57.93 | 112% | ok |
| blas1 | two_norm | f64 | n=8388608 | 1.182 ms | 14.19 | 56.77 | 110% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.393 ms | 1.31 | 10.50 | 20% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.409 ms | 1.31 | 10.47 | 20% | ok |
| blas1 | sum | f64 | n=8388608 | 6.391 ms | 1.31 | 10.50 | 20% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 111.455 ms | 1.66 | 1.20 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.488 ms | 9.62 | 38.50 | 74% | ok |
| blas2 | gemv | f32 | n=4096 | 1.810 ms | 18.54 | 37.10 | 72% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.969 ms | 8.45 | 33.84 | 65% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.078 ms | 16.15 | 32.31 | 62% | ok |
| blas2 | symv | f64 | n=4096 | 3.385 ms | 9.91 | 19.84 | 38% | ok |
| blas2 | trmv | f64 | n=4096 | 2.027 ms | 8.28 | 33.14 | 64% | ok |
| blas2 | trmv | f32 | n=4096 | 1.059 ms | 15.84 | 31.72 | 61% | ok |
| blas2 | trsv | f64 | n=4096 | 2.018 ms | 8.31 | 33.28 | 64% | ok |
| blas2 | trsv | f32 | n=4096 | 1.172 ms | 14.31 | 28.66 | 55% | ok |
| blas2 | ger | f64 | n=4096 | 16.232 ms | 2.07 | 16.54 | 32% | ok |
| blas3 | gemm | f64 | 512x512x512 | 13.799 ms | 19.45 | 0.46 | 31% | ok |
| blas3 | gemm | f32 | 512x512x512 | 6.934 ms | 38.71 | 0.45 | 31% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 9.273 ms | 14.47 | 0.68 | 23% | ok |
| blas3 | symm | f64 | 512x512x512 | 15.046 ms | 17.84 | 0.42 | 28% | ok |
| blas3 | syrk | f64 | 512x512x512 | 9.196 ms | 14.60 | 0.46 | 23% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 18.008 ms | 14.91 | 0.35 | 24% | ok |
| blas3 | trmm | f64 | 512x512x512 | 12.002 ms | 11.18 | 0.44 | 18% | ok |
| blas3 | trsm | f64 | 512x512x512 | 12.243 ms | 10.96 | 0.43 | 17% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.057 ms | 16.66 | 0.33 | 13% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 11.016 ms | 0.48 | 3.86 | 7% | ok |
| factor | lu | f64 | 512x512 | 40.111 ms | 2.23 | 0.10 | 4% | ok |
| factor | lu | f32 | 512x512 | 37.897 ms | 2.36 | 0.06 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 14.294 ms | 3.13 | 0.29 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 13.693 ms | 3.27 | 0.15 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 14.409 ms | 3.10 | 0.29 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 12.802 ms | 3.49 | 0.33 | 6% | ok |
| factor | qr | f64 | 512x512 | 816.753 ms | 0.22 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 80.973 ms | 2.21 | 0.05 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 934.306 ms | 0.48 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.937 ms | 0.21 | 0.85 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 111.010 ms | 2.42 | 0.04 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 42.454 ms | 2.11 | 0.10 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 48.210 ms | 0.46 | 0.02 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 160.681 ms | 0.94 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 664.346 ms | 0.23 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 73.519 ms | 2.28 | 0.01 | 4% | ok |
| eigen | schur | f64 | 256x256 | 143.561 ms | 2.92 | 0.01 | 5% | ok |
| eigen | svd | f64 | 256x256 | 150.930 ms | 2.45 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 32.046 ms | 7.85 | 0.03 | 12% | ok |
| eigen | sylvester | f64 | 256x256 | 324.692 ms | 3.10 | 0.00 | 5% | ok |
| elementwise | exp | f64 | n=8388608 | 33.382 ms | 0.25 | 4.02 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 51.620 ms | 0.16 | 2.60 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 48.800 ms | 0.17 | 2.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 26.443 ms | 0.32 | 5.08 | 10% | ok |
| elementwise | erf | f64 | n=8388608 | 39.005 ms | 0.22 | 3.44 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.911 ms | 2.87 | 29.89 | 58% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.588 ms | 1.65 | 17.14 | 33% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 7.873 ms | 5.32 | 11.18 | 22% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.900 ms | 0.16 | 0.55 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 2.059 ms | 0.97 | 8.08 | 16%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 23.980 ms | 0.47 | 0.03 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 35.593 ms | 0.31 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 17.700 ms | 0.63 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 65.266 ms | 0.34 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 70.601 ms | 0.32 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 30.236 ms | 0.74 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 43.941 ms | 0.04 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.522 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.916 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 76.124 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.224 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.322 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.593 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.653 ms | 0.63 | 0.04 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.364 ms | 2.08 | 8.30 | 16%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.027 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.249 ms | 1.54 | 0.10 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.345 ms | 2.19 | 8.78 | 17%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.093 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 58.210 ms | 0.38 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.345 ms | 0.56 | 2.25 | 4%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.951 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 60.641 ms | 0.37 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.268 ms | 0.60 | 2.39 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 6.927 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 357.822 ms | 0.86 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.213 ms | 1.52 | 6.09 | 12%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.220 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 85.049 ms | 3.63 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.275 ms | 1.48 | 5.93 | 11%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.584 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1090.926 ms | 0.57 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.200 ms | 0.30 | 1.20 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.598 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 613.321 ms | 1.01 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.053 ms | 0.30 | 1.22 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 41.283 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 58.741 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 35.989 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.109 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.732 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 19.953 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 43.348 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 37.280 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 21.998 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 23.669 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 22.215 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 22.365 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 27.191 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.055 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.929 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 27.357 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.239 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.157 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.036 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.818 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.742 ms | 1.50 | 16.84 | 32% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.943 ms | 1.35 | 15.09 | 29% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.238 ms | 1.17 | 13.10 | 25% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 19.784 ms | 1.94 | 14.69 | 28% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.551 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.146 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 66.886 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.132 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 2.186 ms | 7.67 | 30.70 | 59% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 19.255 ms | 13.94 | 0.22 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.575 ms | 29.18 | 31.01 | 60% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.471 ms | 11.41 | 45.62 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.596 ms | 28.14 | 56.29 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.430 ms | 39.05 | 39.05 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.388 ms | 43.22 | 43.22 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 18.912 ms | 14.19 | 0.17 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 21.932 ms | 12.24 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 22.598 ms | 11.88 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.308 ms | 50.57 | 0.30 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.558 ms | 58.89 | 0.35 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

