# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,8, CPUs 0,2,4,6,8,10,12,14, clock policy power scheme copy of Balanced with processor state 100-100% and EPP 0, boost aggressive; AC, 2026-11-27T14:48:16Z.

## ryzen-7-8845hs -- native, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `57512adcb8fe` |
| FMA peak (ceiling) | f64 80.0 GFLOP/s, f32 160.1 GFLOP/s |
| ... hardware, AVX-512F | f64 80.0, f32 160.1 GFLOP/s |
| ... this build, scalar | f64 1.9, f32 1.9 GFLOP/s |
| bandwidth | read 43.4, update 53.2, triad 38.9 GB/s (ceiling 53.2) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 25% | 110% | 0 |
| blas2 | 10 | 56% | 83% | 0 |
| blas3 | 10 | 15% | 21% | 0 |
| factor | 12 | 3% | 5% | 0 |
| eigen | 8 | 4% | 11% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 26% | 58% | 0 |
| direct | 19 | 1% | 20% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 13% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 37% | 39% | 0 |
| mg | 1 | 31% | 31% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 13% | 23% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 9.811 ms | 1.71 | 13.68 | 26% | ok |
| blas1 | dot | f32 | n=8388608 | 10.159 ms | 1.65 | 6.61 | 12% | ok |
| blas1 | axpy | f64 | n=8388608 | 11.917 ms | 1.41 | 16.89 | 32% | ok |
| blas1 | axpy | f32 | n=8388608 | 10.259 ms | 1.64 | 9.81 | 18% | ok |
| blas1 | scale | f64 | n=8388608 | 2.290 ms | 3.66 | 58.62 | 110% | ok |
| blas1 | two_norm | f64 | n=8388608 | 9.436 ms | 1.78 | 7.11 | 13% | ok |
| blas1 | one_norm | f64 | n=8388608 | 5.065 ms | 1.66 | 13.25 | 25% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 5.075 ms | 1.65 | 13.22 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 5.064 ms | 1.66 | 13.25 | 25% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 82.278 ms | 2.24 | 1.63 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.044 ms | 11.02 | 44.12 | 83% | ok |
| blas2 | gemv | f32 | n=4096 | 2.256 ms | 14.87 | 29.76 | 56% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 6.412 ms | 5.23 | 20.94 | 39% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.100 ms | 15.98 | 31.97 | 60% | ok |
| blas2 | symv | f64 | n=4096 | 2.899 ms | 11.58 | 23.17 | 44% | ok |
| blas2 | trmv | f64 | n=4096 | 1.685 ms | 9.95 | 39.86 | 75% | ok |
| blas2 | trmv | f32 | n=4096 | 1.476 ms | 11.37 | 22.75 | 43% | ok |
| blas2 | trsv | f64 | n=4096 | 1.684 ms | 9.96 | 39.89 | 75% | ok |
| blas2 | trsv | f32 | n=4096 | 1.489 ms | 11.27 | 22.56 | 42% | ok |
| blas2 | ger | f64 | n=4096 | 13.461 ms | 2.49 | 19.95 | 37% | ok |
| blas3 | gemm | f64 | 512x512x512 | 15.982 ms | 16.80 | 0.39 | 21% | ok |
| blas3 | gemm | f32 | 512x512x512 | 8.186 ms | 32.79 | 0.38 | 20% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 11.263 ms | 11.92 | 0.56 | 15% | ok |
| blas3 | symm | f64 | 512x512x512 | 17.116 ms | 15.68 | 0.37 | 20% | ok |
| blas3 | syrk | f64 | 512x512x512 | 13.580 ms | 9.88 | 0.31 | 12% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 22.804 ms | 11.77 | 0.28 | 15% | ok |
| blas3 | trmm | f64 | 512x512x512 | 25.921 ms | 5.18 | 0.20 | 6% | ok |
| blas3 | trsm | f64 | 512x512x512 | 26.502 ms | 5.06 | 0.20 | 6% | ok |
| blas3 | trsm | f32 | 512x512x512 | 26.329 ms | 5.10 | 0.10 | 3% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 8.551 ms | 0.62 | 4.97 | 9% | ok |
| factor | lu | f64 | 512x512 | 32.477 ms | 2.76 | 0.13 | 3% | ok |
| factor | lu | f32 | 512x512 | 31.290 ms | 2.86 | 0.07 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 11.380 ms | 3.93 | 0.37 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 10.903 ms | 4.10 | 0.19 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 11.459 ms | 3.90 | 0.37 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 10.176 ms | 4.40 | 0.41 | 5% | ok |
| factor | qr | f64 | 512x512 | 649.401 ms | 0.28 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 64.230 ms | 2.79 | 0.07 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 750.824 ms | 0.60 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 3.899 ms | 0.27 | 1.08 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 89.141 ms | 3.01 | 0.05 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 34.174 ms | 2.62 | 0.12 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 38.391 ms | 0.58 | 0.03 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 126.775 ms | 1.19 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 517.931 ms | 0.29 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 59.667 ms | 2.81 | 0.02 | 4% | ok |
| eigen | schur | f64 | 256x256 | 114.142 ms | 3.67 | 0.01 | 5% | ok |
| eigen | svd | f64 | 256x256 | 118.184 ms | 3.12 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 29.878 ms | 8.42 | 0.04 | 11% | ok |
| eigen | sylvester | f64 | 256x256 | 256.218 ms | 3.93 | 0.01 | 5% | ok |
| elementwise | exp | f64 | n=8388608 | 31.301 ms | 0.27 | 4.29 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 41.164 ms | 0.20 | 3.26 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 37.896 ms | 0.22 | 3.54 | 7% | ok |
| elementwise | sqrt | f64 | n=8388608 | 20.537 ms | 0.41 | 6.54 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 30.901 ms | 0.27 | 4.34 | 8% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.884 ms | 2.96 | 30.81 | 58% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.294 ms | 2.02 | 21.05 | 40% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.430 ms | 6.51 | 13.69 | 26% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 15.775 ms | 0.21 | 0.73 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.889 ms | 1.06 | 8.81 | 17%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 18.721 ms | 0.60 | 0.04 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 30.507 ms | 0.37 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 14.875 ms | 0.75 | 0.05 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 53.242 ms | 0.42 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 56.106 ms | 0.40 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 24.140 ms | 0.93 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 38.070 ms | 0.05 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.637 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.602 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 59.892 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.224 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.080 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.925 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.109 ms | 0.79 | 0.05 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.284 ms | 2.66 | 10.66 | 20%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.795 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 6.882 ms | 1.63 | 0.11 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.280 ms | 2.70 | 10.82 | 20%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.296 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 46.916 ms | 0.48 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.167 ms | 0.65 | 2.59 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.705 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.634 ms | 0.46 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.127 ms | 0.67 | 2.68 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.576 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 251.792 ms | 1.23 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.985 ms | 1.70 | 6.79 | 13%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 14.016 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 68.481 ms | 4.51 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.026 ms | 1.66 | 6.65 | 12%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.010 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 893.878 ms | 0.69 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.508 ms | 0.32 | 1.28 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.362 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 494.563 ms | 1.25 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.581 ms | 0.32 | 1.27 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 39.208 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 81.420 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 31.548 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 16.771 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 21.770 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.674 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 54.949 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.371 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 19.778 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 26.006 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 24.572 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 25.542 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 33.563 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.044 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.370 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 22.328 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.801 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.728 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.023 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.678 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.403 ms | 1.87 | 20.91 | 39% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.493 ms | 1.75 | 19.64 | 37% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.783 ms | 1.47 | 16.45 | 31% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 17.539 ms | 2.19 | 16.57 | 31% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.199 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.732 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 51.434 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.750 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 9.772 ms | 1.72 | 6.87 | 13% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 29.068 ms | 9.23 | 0.14 | 12% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.476 ms | 11.36 | 12.07 | 23% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.208 ms | 5.23 | 20.92 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.884 ms | 5.82 | 11.64 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.387 ms | 12.09 | 12.09 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.388 ms | 12.09 | 12.09 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 8.122 ms | 33.05 | 0.39 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 30.511 ms | 8.80 | 0.07 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 27.333 ms | 9.82 | 0.06 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 27.317 ms | 9.83 | 0.06 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 27.302 ms | 9.83 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## ryzen-7-8845hs -- native, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `57512adcb8fe` |
| FMA peak (ceiling) | f64 628.0 GFLOP/s, f32 1254.7 GFLOP/s |
| ... hardware, AVX-512F | f64 628.0, f32 1254.7 GFLOP/s |
| ... this build, scalar | f64 14.6, f32 14.4 GFLOP/s |
| bandwidth | read 53.1, update 53.6, triad 41.4 GB/s (ceiling 53.6) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 98% | 124% | 0 | 42% |
| blas2 | 10 | 74% | 109% | 0 | 13% |
| blas3 | 10 | 2% | 18% | 0 | 12% |
| factor | 12 | 1% | 8% | 0 | 18% |
| eigen | 8 | 0% | 2% | 0 | 13% |
| elementwise | 5 | 8% | 12% | 0 | 12% |
| sparse | 5 | 35% | 291% | 0 | 15% |
| direct | 19 | 0% | 20% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 13% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 37% | 37% | 0 | 12% |
| mg | 1 | 36% | 36% | 0 | 15% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 13% | 22% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 88% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.381 ms | 7.05 | 56.37 | 105% | ok | 4.1x (52%) |
| blas1 | dot | f32 | n=8388608 | 1.335 ms | 12.57 | 50.26 | 94% | ok | 7.6x (95%) |
| blas1 | axpy | f64 | n=8388608 | 3.524 ms | 4.76 | 57.13 | 106% | ok | 3.4x (42%) |
| blas1 | axpy | f32 | n=8388608 | 1.734 ms | 9.68 | 58.06 | 108% | ok | 5.9x (74%) |
| blas1 | scale | f64 | n=8388608 | 2.025 ms | 4.14 | 66.27 | 124% | ok | 1.1x (14%) |
| blas1 | two_norm | f64 | n=8388608 | 1.282 ms | 13.08 | 52.33 | 98% | ok | 7.4x (92%) |
| blas1 | one_norm | f64 | n=8388608 | 5.085 ms | 1.65 | 13.20 | 25% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.127 ms | 1.64 | 13.09 | 24% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 5.064 ms | 1.66 | 13.25 | 25% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 81.506 ms | 2.26 | 1.65 | 3% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 2.378 ms | 14.11 | 56.46 | 105% | ok | 1.3x (16%) |
| blas2 | gemv | f32 | n=4096 | 1.170 ms | 28.69 | 57.41 | 107% | ok | 1.9x (24%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.288 ms | 14.66 | 58.69 | 109% | ok | 2.8x (35%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.150 ms | 29.19 | 58.40 | 109% | ok | 1.8x (23%) |
| blas2 | symv | f64 | n=4096 | 2.970 ms | 11.30 | 22.62 | 42% | ok | 1.0x (12%) |
| blas2 | trmv | f64 | n=4096 | 1.713 ms | 9.79 | 39.22 | 73% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=4096 | 1.489 ms | 11.27 | 22.56 | 42% | ok | 1.0x (12%) |
| blas2 | trsv | f64 | n=4096 | 1.683 ms | 9.97 | 39.91 | 74% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.504 ms | 11.15 | 22.33 | 42% | ok | 1.0x (12%) |
| blas2 | ger | f64 | n=4096 | 13.555 ms | 2.48 | 19.81 | 37% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.482 ms | 108.17 | 2.54 | 17% | ok | 6.4x (81%) |
| blas3 | gemm | f32 | 512x512x512 | 1.200 ms | 223.70 | 2.62 | 18% | ok | 6.8x (85%) |
| blas3 | gemmt | f64 | 512x512x512 | 11.402 ms | 11.77 | 0.55 | 2% | ok | 1.0x (12%) |
| blas3 | symm | f64 | 512x512x512 | 17.316 ms | 15.50 | 0.36 | 2% | ok | 1.0x (12%) |
| blas3 | syrk | f64 | 512x512x512 | 13.689 ms | 9.80 | 0.31 | 2% | ok | 1.0x (12%) |
| blas3 | syr2k | f64 | 512x512x512 | 22.923 ms | 11.71 | 0.27 | 2% | ok | 1.0x (12%) |
| blas3 | trmm | f64 | 512x512x512 | 25.951 ms | 5.17 | 0.20 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f64 | 512x512x512 | 26.629 ms | 5.04 | 0.20 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f32 | 512x512x512 | 26.623 ms | 5.04 | 0.10 | 0% | ok | 1.0x (12%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 8.592 ms | 0.62 | 4.94 | 9% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 22.124 ms | 4.04 | 0.19 | 1% | ok | 1.5x (18%) |
| factor | lu | f32 | 512x512 | 20.711 ms | 4.32 | 0.10 | 0% | ok | 1.5x (19%) |
| factor | cholesky | f64 | 512x512 | 11.601 ms | 3.86 | 0.36 | 1%\* | ok | 1.0x (12%) |
| factor | cholesky | f32 | 512x512 | 10.948 ms | 4.09 | 0.19 | 0%\* | ok | 1.0x (12%) |
| factor | ldlt | f64 | 512x512 | 11.456 ms | 3.91 | 0.37 | 1%\* | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 10.546 ms | 4.24 | 0.40 | 1%\* | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 395.898 ms | 0.45 | 0.01 | 0% | ok | 1.6x (21%) |
| factor | lq | f64 | 512x512 | 41.727 ms | 4.29 | 0.10 | 1% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 461.723 ms | 0.97 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.004 ms | 1.05 | 4.18 | 8%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 77.955 ms | 3.44 | 0.05 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 23.298 ms | 3.84 | 0.18 | 1% | ok | 1.5x (18%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 36.805 ms | 0.61 | 0.03 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 127.568 ms | 1.18 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 498.172 ms | 0.30 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 58.349 ms | 2.88 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 108.802 ms | 3.85 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 119.595 ms | 3.09 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | expm | f64 | 256x256 | 18.392 ms | 13.68 | 0.06 | 2% | ok | 1.6x (20%) |
| eigen | sylvester | f64 | 256x256 | 260.721 ms | 3.86 | 0.01 | 1% | ok | 1.0x (12%) |
| elementwise | exp | f64 | n=8388608 | 27.144 ms | 0.31 | 4.94 | 9% | ok | 1.2x (14%) |
| elementwise | log | f64 | n=8388608 | 41.687 ms | 0.20 | 3.22 | 6% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 38.425 ms | 0.22 | 3.49 | 7% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 20.554 ms | 0.41 | 6.53 | 12% | ok | 1.0x (12%) |
| elementwise | erf | f64 | n=8388608 | 31.278 ms | 0.27 | 4.29 | 8% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.174 ms | 15.01 | 156.14 | 291% ! | ok | 5.1x (63%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.435 ms | 1.82 | 18.97 | 35% | ok | 0.9x (11%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.253 ms | 18.59 | 39.08 | 73% | ok | 2.9x (36%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.595 ms | 0.20 | 0.69 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.612 ms | 1.25 | 10.33 | 19%\* | ok | 1.2x (15%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 19.182 ms | 0.58 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 31.295 ms | 0.36 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 15.094 ms | 0.74 | 0.05 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 53.785 ms | 0.42 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 57.994 ms | 0.39 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 25.457 ms | 0.88 | 0.03 | 0% | ok | 0.9x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 38.188 ms | 0.05 | 0.00 | 0%\* | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.738 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.645 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 60.395 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.227 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.360 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.997 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.364 ms | 0.78 | 0.05 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.278 ms | 2.72 | 10.88 | 20%\* | ok | 1.0x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.655 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 5.667 ms | 1.97 | 0.13 | 0% | ok | 1.2x (15%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.282 ms | 2.68 | 10.73 | 20%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.336 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.463 ms | 0.47 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.128 ms | 0.67 | 2.68 | 5%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.821 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.399 ms | 0.46 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.132 ms | 0.67 | 2.67 | 5%\* | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.728 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 258.520 ms | 1.19 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.009 ms | 1.68 | 6.71 | 13%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 13.933 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.054 ms | 4.47 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.052 ms | 1.64 | 6.57 | 12%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.085 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 906.412 ms | 0.68 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.599 ms | 0.32 | 1.27 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.469 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 499.461 ms | 1.24 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.504 ms | 0.32 | 1.28 | 2%\* | ok | 1.0x (13%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 39.142 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 80.251 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 31.525 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 16.750 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 21.766 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.708 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 54.976 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.419 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 19.806 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 25.943 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 24.584 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 25.616 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 33.573 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.044 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.457 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 25.598 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| pc | ic_0 | f64 | 128^2 grid | 1.794 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 1.739 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.024 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.692 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.480 ms | 1.77 | 19.82 | 37% | ok | 0.9x (12%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.493 ms | 1.75 | 19.64 | 37% | ok | 1.0x (12%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.921 ms | 1.36 | 15.27 | 28% | ok | 0.9x (12%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.953 ms | 2.57 | 19.44 | 36% | ok | 1.2x (15%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.324 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.660 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 52.726 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.758 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 9.812 ms | 1.71 | 6.84 | 13% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.004 ms | 67.04 | 1.05 | 11% | ok | 7.3x (91%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.495 ms | 11.22 | 11.92 | 22% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.223 ms | 5.21 | 20.82 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.894 ms | 5.80 | 11.59 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.402 ms | 11.96 | 11.96 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.403 ms | 11.96 | 11.96 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.158 ms | 231.85 | 2.72 | n/a | ok | 7.0x (88%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.136 ms | 64.90 | 0.51 | n/a | ok | 7.4x (92%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 3.751 ms | 71.56 | 0.42 | n/a | ok | 7.3x (91%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.761 ms | 71.38 | 0.42 | n/a | ok | 7.3x (91%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.766 ms | 71.28 | 0.42 | n/a | ok | 7.2x (91%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

## ryzen-7-8845hs -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `57512adcb8fe` |
| FMA peak (ceiling) | f64 80.0 GFLOP/s, f32 160.0 GFLOP/s |
| ... hardware, AVX-512F | f64 80.0, f32 160.0 GFLOP/s |
| ... this build, SSE2 | f64 40.0, f32 80.0 GFLOP/s |
| bandwidth | read 46.8, update 53.5, triad 38.8 GB/s (ceiling 53.5) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 105% | 129% | 0 |
| blas2 | 10 | 66% | 75% | 0 |
| blas3 | 10 | 23% | 30% | 0 |
| factor | 12 | 3% | 5% | 0 |
| eigen | 8 | 4% | 12% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 26% | 59% | 0 |
| direct | 19 | 1% | 21% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 13% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 38% | 39% | 0 |
| mg | 1 | 34% | 34% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 70% | 70% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 2.403 ms | 6.98 | 55.87 | 105% | ok |
| blas1 | dot | f32 | n=8388608 | 1.212 ms | 13.84 | 55.37 | 104% | ok |
| blas1 | axpy | f64 | n=8388608 | 3.515 ms | 4.77 | 57.28 | 107% | ok |
| blas1 | axpy | f32 | n=8388608 | 1.656 ms | 10.13 | 60.77 | 114% | ok |
| blas1 | scale | f64 | n=8388608 | 1.940 ms | 4.32 | 69.18 | 129% | ok |
| blas1 | two_norm | f64 | n=8388608 | 1.170 ms | 14.33 | 57.34 | 107% | ok |
| blas1 | one_norm | f64 | n=8388608 | 5.065 ms | 1.66 | 13.25 | 25% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 5.075 ms | 1.65 | 13.22 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 5.064 ms | 1.66 | 13.25 | 25% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 87.759 ms | 2.10 | 1.53 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.368 ms | 9.96 | 39.87 | 75% | ok |
| blas2 | gemv | f32 | n=4096 | 1.768 ms | 18.98 | 37.98 | 71% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.987 ms | 8.42 | 33.68 | 63% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.109 ms | 15.91 | 31.83 | 60% | ok |
| blas2 | symv | f64 | n=4096 | 3.230 ms | 10.39 | 20.79 | 39% | ok |
| blas2 | trmv | f64 | n=4096 | 1.802 ms | 9.31 | 37.28 | 70% | ok |
| blas2 | trmv | f32 | n=4096 | 0.936 ms | 17.93 | 35.90 | 67% | ok |
| blas2 | trsv | f64 | n=4096 | 1.895 ms | 8.85 | 35.44 | 66% | ok |
| blas2 | trsv | f32 | n=4096 | 1.080 ms | 15.54 | 31.10 | 58% | ok |
| blas2 | ger | f64 | n=4096 | 13.583 ms | 2.47 | 19.77 | 37% | ok |
| blas3 | gemm | f64 | 512x512x512 | 11.020 ms | 24.36 | 0.57 | 30% | ok |
| blas3 | gemm | f32 | 512x512x512 | 5.572 ms | 48.17 | 0.56 | 30% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 7.429 ms | 18.07 | 0.85 | 23% | ok |
| blas3 | symm | f64 | 512x512x512 | 11.971 ms | 22.42 | 0.53 | 28% | ok |
| blas3 | syrk | f64 | 512x512x512 | 7.385 ms | 18.17 | 0.57 | 23% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 14.181 ms | 18.93 | 0.44 | 24% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.441 ms | 14.22 | 0.56 | 18% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.728 ms | 13.80 | 0.54 | 17% | ok |
| blas3 | trsm | f32 | 512x512x512 | 6.373 ms | 21.06 | 0.41 | 13% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 8.528 ms | 0.62 | 4.98 | 9% | ok |
| factor | lu | f64 | 512x512 | 32.693 ms | 2.74 | 0.13 | 3% | ok |
| factor | lu | f32 | 512x512 | 31.082 ms | 2.88 | 0.07 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 11.393 ms | 3.93 | 0.37 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 10.915 ms | 4.10 | 0.19 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 11.479 ms | 3.90 | 0.37 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 10.472 ms | 4.27 | 0.40 | 5% | ok |
| factor | qr | f64 | 512x512 | 644.980 ms | 0.28 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 64.341 ms | 2.78 | 0.07 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 738.831 ms | 0.61 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 3.930 ms | 0.27 | 1.07 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 89.138 ms | 3.01 | 0.05 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 33.774 ms | 2.65 | 0.12 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 37.000 ms | 0.60 | 0.03 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 129.442 ms | 1.17 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 506.147 ms | 0.30 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 60.780 ms | 2.76 | 0.02 | 3% | ok |
| eigen | schur | f64 | 256x256 | 110.515 ms | 3.80 | 0.01 | 5% | ok |
| eigen | svd | f64 | 256x256 | 118.836 ms | 3.11 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 25.937 ms | 9.70 | 0.04 | 12% | ok |
| eigen | sylvester | f64 | 256x256 | 267.908 ms | 3.76 | 0.01 | 5% | ok |
| elementwise | exp | f64 | n=8388608 | 27.023 ms | 0.31 | 4.97 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 42.067 ms | 0.20 | 3.19 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 38.232 ms | 0.22 | 3.51 | 7% | ok |
| elementwise | sqrt | f64 | n=8388608 | 20.608 ms | 0.41 | 6.51 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 31.096 ms | 0.27 | 4.32 | 8% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.868 ms | 3.01 | 31.36 | 59% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.490 ms | 1.76 | 18.28 | 34% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.446 ms | 6.50 | 13.66 | 26% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.348 ms | 0.20 | 0.70 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.703 ms | 1.18 | 9.78 | 18%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 18.474 ms | 0.61 | 0.04 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 27.896 ms | 0.40 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 13.639 ms | 0.82 | 0.05 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 52.842 ms | 0.42 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 55.500 ms | 0.40 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 24.162 ms | 0.93 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 38.858 ms | 0.05 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.654 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.574 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 61.272 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.223 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.070 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.926 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.596 ms | 0.77 | 0.05 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.275 ms | 2.75 | 11.01 | 21%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.565 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 5.655 ms | 1.98 | 0.13 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.290 ms | 2.61 | 10.44 | 20%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.290 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 46.823 ms | 0.48 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.181 ms | 0.64 | 2.56 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.754 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.569 ms | 0.46 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.085 ms | 0.70 | 2.79 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.546 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 256.804 ms | 1.20 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.001 ms | 1.68 | 6.73 | 13%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 13.516 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 68.757 ms | 4.49 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.053 ms | 1.64 | 6.56 | 12%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.051 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 962.159 ms | 0.64 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.561 ms | 0.32 | 1.28 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.415 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 497.158 ms | 1.24 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.422 ms | 0.32 | 1.29 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 32.605 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 61.426 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 28.280 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 13.489 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 17.944 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 15.732 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 34.403 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 29.347 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 17.322 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 18.974 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 17.563 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.635 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.574 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.045 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.434 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 22.527 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.832 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.748 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.024 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.664 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.400 ms | 1.87 | 20.94 | 39% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.461 ms | 1.79 | 20.07 | 38% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.784 ms | 1.47 | 16.44 | 31% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.884 ms | 2.42 | 18.30 | 34% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.223 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.691 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 52.471 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 6.514 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 1.785 ms | 9.40 | 37.59 | 70% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 15.297 ms | 17.55 | 0.27 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.477 ms | 35.14 | 37.33 | 70% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.222 ms | 13.73 | 54.94 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.467 ms | 35.90 | 71.80 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.349 ms | 48.07 | 48.07 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.319 ms | 52.66 | 52.66 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 14.988 ms | 17.91 | 0.21 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 17.347 ms | 15.47 | 0.12 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 17.739 ms | 15.13 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.722 ms | 72.12 | 0.42 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.698 ms | 72.60 | 0.43 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## ryzen-7-8845hs -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `57512adcb8fe` |
| FMA peak (ceiling) | f64 625.9 GFLOP/s, f32 1254.7 GFLOP/s |
| ... hardware, AVX-512F | f64 625.9, f32 1254.7 GFLOP/s |
| ... this build, SSE2 | f64 313.7, f32 624.8 GFLOP/s |
| bandwidth | read 53.0, update 53.5, triad 41.2 GB/s (ceiling 53.5) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 112% | 129% | 0 | 13% |
| blas2 | 10 | 70% | 111% | 0 | 14% |
| blas3 | 10 | 9% | 19% | 0 | 38% |
| factor | 12 | 1% | 8% | 0 | 18% |
| eigen | 8 | 0% | 2% | 0 | 13% |
| elementwise | 5 | 8% | 12% | 0 | 12% |
| sparse | 5 | 39% | 286% | 0 | 14% |
| direct | 19 | 0% | 20% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 12% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 33% | 37% | 0 | 12% |
| mg | 1 | 36% | 36% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 69% | 70% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 70% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.237 ms | 7.50 | 60.00 | 112% | ok | 1.1x (13%) |
| blas1 | dot | f32 | n=8388608 | 0.971 ms | 17.28 | 69.11 | 129% | ok | 1.2x (16%) |
| blas1 | axpy | f64 | n=8388608 | 3.460 ms | 4.85 | 58.19 | 109% | ok | 1.0x (13%) |
| blas1 | axpy | f32 | n=8388608 | 1.623 ms | 10.34 | 62.03 | 116% | ok | 1.0x (13%) |
| blas1 | scale | f64 | n=8388608 | 1.965 ms | 4.27 | 68.31 | 128% | ok | 1.0x (12%) |
| blas1 | two_norm | f64 | n=8388608 | 1.119 ms | 14.99 | 59.98 | 112% | ok | 1.0x (13%) |
| blas1 | one_norm | f64 | n=8388608 | 5.066 ms | 1.66 | 13.25 | 25% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.098 ms | 1.65 | 13.16 | 25% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 5.063 ms | 1.66 | 13.25 | 25% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 87.996 ms | 2.10 | 1.53 | 3% | ok | 1.0x (12%) |
| blas2 | gemv | f64 | n=4096 | 2.291 ms | 14.65 | 58.63 | 110% | ok | 1.5x (18%) |
| blas2 | gemv | f32 | n=4096 | 1.134 ms | 29.59 | 59.22 | 111% | ok | 1.6x (19%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.288 ms | 14.67 | 58.70 | 110% | ok | 1.7x (22%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.193 ms | 28.13 | 56.29 | 105% | ok | 1.8x (22%) |
| blas2 | symv | f64 | n=4096 | 3.209 ms | 10.46 | 20.94 | 39% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 1.886 ms | 8.89 | 35.61 | 67% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=4096 | 1.100 ms | 15.25 | 30.54 | 57% | ok | 0.9x (11%) |
| blas2 | trsv | f64 | n=4096 | 1.795 ms | 9.35 | 37.42 | 70% | ok | 1.1x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.955 ms | 17.58 | 35.18 | 66% | ok | 1.1x (14%) |
| blas2 | ger | f64 | n=4096 | 13.705 ms | 2.45 | 19.59 | 37% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.228 ms | 120.49 | 2.82 | 19% | ok | 4.9x (62%) |
| blas3 | gemm | f32 | 512x512x512 | 1.216 ms | 220.75 | 2.59 | 18% | ok | 4.6x (57%) |
| blas3 | gemmt | f64 | 512x512x512 | 3.091 ms | 43.42 | 2.04 | 7% | ok | 2.4x (30%) |
| blas3 | symm | f64 | 512x512x512 | 2.950 ms | 90.99 | 2.13 | 15% | ok | 4.1x (51%) |
| blas3 | syrk | f64 | 512x512x512 | 2.401 ms | 55.91 | 1.75 | 9% | ok | 3.1x (38%) |
| blas3 | syr2k | f64 | 512x512x512 | 4.616 ms | 58.15 | 1.36 | 9% | ok | 3.1x (38%) |
| blas3 | trmm | f64 | 512x512x512 | 5.545 ms | 24.21 | 0.95 | 4% | ok | 1.7x (21%) |
| blas3 | trsm | f64 | 512x512x512 | 5.748 ms | 23.35 | 0.91 | 4% | ok | 1.7x (21%) |
| blas3 | trsm | f32 | 512x512x512 | 4.460 ms | 30.09 | 0.59 | 2% | ok | 1.4x (18%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.000 ms | 0.59 | 4.72 | 9% | ok | 0.9x (12%) |
| factor | lu | f64 | 512x512 | 21.979 ms | 4.07 | 0.19 | 1% | ok | 1.5x (19%) |
| factor | lu | f32 | 512x512 | 21.101 ms | 4.24 | 0.10 | 0% | ok | 1.5x (18%) |
| factor | cholesky | f64 | 512x512 | 11.453 ms | 3.91 | 0.37 | 1%\* | ok | 1.0x (12%) |
| factor | cholesky | f32 | 512x512 | 10.933 ms | 4.09 | 0.19 | 0%\* | ok | 1.0x (12%) |
| factor | ldlt | f64 | 512x512 | 11.520 ms | 3.88 | 0.36 | 1%\* | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 10.443 ms | 4.28 | 0.40 | 1%\* | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 396.684 ms | 0.45 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 42.355 ms | 4.23 | 0.10 | 1% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 456.711 ms | 0.98 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.014 ms | 1.04 | 4.14 | 8%\* | ok | 3.9x (48%) |
| factor | inv | f64 | 512x512 | 78.644 ms | 3.41 | 0.05 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 24.020 ms | 3.73 | 0.17 | 1% | ok | 1.4x (18%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.368 ms | 0.57 | 0.03 | 0% | ok | 0.9x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 126.698 ms | 1.19 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 506.464 ms | 0.30 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 256x256 | 60.667 ms | 2.77 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 114.870 ms | 3.65 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 120.048 ms | 3.07 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | expm | f64 | 256x256 | 18.316 ms | 13.74 | 0.06 | 2% | ok | 1.4x (18%) |
| eigen | sylvester | f64 | 256x256 | 264.603 ms | 3.80 | 0.01 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 27.600 ms | 0.30 | 4.86 | 9% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 41.859 ms | 0.20 | 3.21 | 6% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 38.732 ms | 0.22 | 3.47 | 6% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 20.542 ms | 0.41 | 6.53 | 12% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 31.462 ms | 0.27 | 4.27 | 8% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.178 ms | 14.69 | 152.81 | 286% ! | ok | 4.9x (61%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.291 ms | 2.03 | 21.08 | 39% | ok | 1.2x (14%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.246 ms | 18.64 | 39.20 | 73% | ok | 2.9x (36%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.866 ms | 0.19 | 0.68 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.624 ms | 1.24 | 10.25 | 19%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 18.524 ms | 0.60 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 28.050 ms | 0.40 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 13.412 ms | 0.83 | 0.05 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 54.467 ms | 0.41 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 56.525 ms | 0.40 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 25.282 ms | 0.88 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 38.133 ms | 0.05 | 0.00 | 0%\* | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.752 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.639 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 60.563 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.228 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.100 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.995 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.595 ms | 0.77 | 0.05 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.279 ms | 2.71 | 10.83 | 20%\* | ok | 1.0x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.667 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 5.750 ms | 1.95 | 0.13 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.282 ms | 2.69 | 10.74 | 20%\* | ok | 1.0x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.317 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.367 ms | 0.47 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.128 ms | 0.67 | 2.68 | 5%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.801 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.497 ms | 0.46 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.136 ms | 0.67 | 2.66 | 5%\* | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.634 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 260.146 ms | 1.19 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.019 ms | 1.67 | 6.68 | 12%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 13.996 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.300 ms | 4.45 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.064 ms | 1.63 | 6.53 | 12%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.080 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 907.634 ms | 0.68 | 0.00 | 0% | ok | 1.1x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.576 ms | 0.32 | 1.27 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.488 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 494.820 ms | 1.25 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.595 ms | 0.32 | 1.27 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 32.654 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 61.684 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 28.390 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 13.512 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 17.988 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 15.749 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 34.601 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 29.399 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 17.367 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 18.645 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 17.595 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.691 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.649 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.044 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.404 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 22.679 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 1.804 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 1.758 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.024 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.666 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.478 ms | 1.77 | 19.84 | 37% | ok | 0.9x (12%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.642 ms | 1.59 | 17.86 | 33% | ok | 0.9x (11%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.761 ms | 1.49 | 16.65 | 31% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.983 ms | 2.57 | 19.40 | 36% | ok | 1.1x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.328 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.803 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 51.087 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 6.477 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 1.786 ms | 9.39 | 37.57 | 70% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 2.558 ms | 104.94 | 1.64 | 17% | ok | 6.0x (75%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.484 ms | 34.68 | 36.85 | 69% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.247 ms | 13.46 | 53.83 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.469 ms | 35.77 | 71.54 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.351 ms | 47.85 | 47.85 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.321 ms | 52.35 | 52.35 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.346 ms | 114.41 | 1.34 | n/a | ok | 6.4x (80%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.519 ms | 106.57 | 0.83 | n/a | ok | 6.9x (86%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 2.543 ms | 105.58 | 0.62 | n/a | ok | 7.0x (87%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.661 ms | 406.35 | 2.38 | n/a | ok | 5.6x (70%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.659 ms | 407.21 | 2.39 | n/a | ok | 5.6x (70%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

## ryzen-7-8845hs -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `57512adcb8fe` |
| FMA peak (ceiling) | f64 80.0 GFLOP/s, f32 160.1 GFLOP/s |
| ... hardware, AVX-512F | f64 80.0, f32 160.1 GFLOP/s |
| ... this build, scalar | f64 1.6, f32 1.9 GFLOP/s |
| bandwidth | read 42.5, update 53.3, triad 38.7 GB/s (ceiling 53.3) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 25% | 108% | 0 |
| blas2 | 10 | 72% | 85% | 0 |
| blas3 | 10 | 71% | 92% | 0 |
| factor | 12 | 13% | 23% | 0 |
| eigen | 8 | 17% | 53% | 0 |
| elementwise | 5 | 7% | 12% | 0 |
| sparse | 5 | 18% | 59% | 0 |
| direct | 19 | 1% | 19% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 13% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 35% | 37% | 0 |
| mg | 1 | 34% | 34% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 11% | 22% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.211 ms | 1.50 | 11.97 | 22% | ok |
| blas1 | dot | f32 | n=8388608 | 9.859 ms | 1.70 | 6.81 | 13% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.043 ms | 1.39 | 16.72 | 31% | ok |
| blas1 | axpy | f32 | n=8388608 | 10.366 ms | 1.62 | 9.71 | 18% | ok |
| blas1 | scale | f64 | n=8388608 | 2.322 ms | 3.61 | 57.79 | 108% | ok |
| blas1 | two_norm | f64 | n=8388608 | 11.483 ms | 1.46 | 5.84 | 11% | ok |
| blas1 | one_norm | f64 | n=8388608 | 5.066 ms | 1.66 | 13.25 | 25% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 5.074 ms | 1.65 | 13.23 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 5.065 ms | 1.66 | 13.25 | 25% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 89.415 ms | 2.06 | 1.50 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.490 ms | 9.61 | 38.48 | 72% | ok |
| blas2 | gemv | f32 | n=4096 | 1.563 ms | 21.47 | 42.97 | 81% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.765 ms | 8.91 | 35.67 | 67% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.486 ms | 22.59 | 45.20 | 85% | ok |
| blas2 | symv | f64 | n=4096 | 1.936 ms | 17.33 | 34.70 | 65% | ok |
| blas2 | trmv | f64 | n=4096 | 1.781 ms | 9.42 | 37.72 | 71% | ok |
| blas2 | trmv | f32 | n=4096 | 0.943 ms | 17.79 | 35.62 | 67% | ok |
| blas2 | trsv | f64 | n=4096 | 1.754 ms | 9.56 | 38.29 | 72% | ok |
| blas2 | trsv | f32 | n=4096 | 0.851 ms | 19.72 | 39.48 | 74% | ok |
| blas2 | ger | f64 | n=4096 | 6.380 ms | 5.26 | 42.09 | 79% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.650 ms | 73.54 | 1.72 | 92% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.901 ms | 141.22 | 1.65 | 88% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.743 ms | 48.92 | 2.29 | 61% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.796 ms | 70.71 | 1.66 | 88% | ok |
| blas3 | syrk | f64 | 512x512x512 | 3.013 ms | 44.55 | 1.39 | 56% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 4.757 ms | 56.42 | 1.32 | 71% | ok |
| blas3 | trmm | f64 | 512x512x512 | 1.898 ms | 70.71 | 2.76 | 88% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.853 ms | 47.05 | 1.84 | 59% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.345 ms | 99.79 | 1.95 | 62% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 8.813 ms | 0.60 | 4.82 | 9% | ok |
| factor | lu | f64 | 512x512 | 4.970 ms | 18.01 | 0.84 | 23% | ok |
| factor | lu | f32 | 512x512 | 2.422 ms | 36.95 | 0.87 | 23% | ok |
| factor | cholesky | f64 | 512x512 | 4.336 ms | 10.32 | 0.97 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 1.663 ms | 26.91 | 1.26 | 17% | ok |
| factor | ldlt | f64 | 512x512 | 11.748 ms | 3.81 | 0.36 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 10.597 ms | 4.22 | 0.40 | 5% | ok |
| factor | qr | f64 | 512x512 | 10.249 ms | 17.46 | 0.41 | 22% | ok |
| factor | lq | f64 | 512x512 | 65.219 ms | 2.74 | 0.06 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 741.440 ms | 0.60 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 3.880 ms | 0.27 | 1.08 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 60.082 ms | 4.47 | 0.07 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 5.512 ms | 16.23 | 0.76 | 20% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.694 ms | 13.21 | 0.62 | 17% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 129.651 ms | 1.16 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 521.773 ms | 0.29 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 17.033 ms | 9.85 | 0.06 | 12% | ok |
| eigen | schur | f64 | 256x256 | 22.193 ms | 18.90 | 0.05 | 24% | ok |
| eigen | svd | f64 | 256x256 | 8.665 ms | 42.60 | 0.12 | 53% | ok |
| eigen | expm | f64 | 256x256 | 15.504 ms | 16.23 | 0.07 | 20% | ok |
| eigen | sylvester | f64 | 256x256 | 74.783 ms | 13.46 | 0.02 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 27.020 ms | 0.31 | 4.97 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 42.417 ms | 0.20 | 3.16 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 38.297 ms | 0.22 | 3.50 | 7% | ok |
| elementwise | sqrt | f64 | n=8388608 | 20.729 ms | 0.40 | 6.47 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 34.156 ms | 0.25 | 3.93 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.862 ms | 3.04 | 31.60 | 59% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 5.928 ms | 0.44 | 4.59 | 9% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.574 ms | 6.37 | 13.39 | 25% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.852 ms | 0.19 | 0.68 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.691 ms | 1.19 | 9.84 | 18%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 18.983 ms | 0.59 | 0.04 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 30.617 ms | 0.37 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 14.223 ms | 0.79 | 0.05 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 53.170 ms | 0.42 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 56.183 ms | 0.40 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 24.888 ms | 0.90 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 37.943 ms | 0.05 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.681 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.621 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 60.272 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.226 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.234 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.933 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.234 ms | 0.79 | 0.05 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.294 ms | 2.57 | 10.28 | 19%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.588 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 5.721 ms | 1.96 | 0.13 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.291 ms | 2.60 | 10.38 | 19%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.293 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 46.818 ms | 0.48 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.131 ms | 0.67 | 2.67 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.764 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.765 ms | 0.46 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.098 ms | 0.69 | 2.75 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.608 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 261.770 ms | 1.18 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.966 ms | 1.71 | 6.85 | 13%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 14.012 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.454 ms | 4.44 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.019 ms | 1.67 | 6.68 | 13%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.072 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 905.149 ms | 0.68 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.541 ms | 0.32 | 1.28 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.427 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 512.274 ms | 1.20 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.565 ms | 0.32 | 1.28 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 40.275 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 83.924 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 32.133 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.339 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.426 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 18.096 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 58.085 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 35.334 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 20.225 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 27.475 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 25.988 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 27.098 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 35.688 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.044 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.436 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 22.168 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.823 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.759 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.024 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.343 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.571 ms | 1.67 | 18.67 | 35% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.493 ms | 1.75 | 19.64 | 37% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.891 ms | 1.38 | 15.51 | 29% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.938 ms | 2.41 | 18.24 | 34% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.347 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.940 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 52.860 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.197 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 11.467 ms | 1.46 | 5.85 | 11% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 29.556 ms | 9.08 | 0.14 | 11% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.489 ms | 11.27 | 11.97 | 22% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.204 ms | 5.24 | 20.95 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.864 ms | 5.86 | 11.72 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.391 ms | 12.06 | 12.06 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.390 ms | 12.07 | 12.07 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 8.397 ms | 31.97 | 0.37 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 30.447 ms | 8.82 | 0.07 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 27.407 ms | 9.79 | 0.06 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 27.408 ms | 9.79 | 0.06 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 27.388 ms | 9.80 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

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

