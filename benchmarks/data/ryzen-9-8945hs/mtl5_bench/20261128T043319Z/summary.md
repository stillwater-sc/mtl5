# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,8, CPUs 0,2,4,6,8,10,12,14, clock policy power scheme copy of Balanced with processor state 100-100% and EPP 0, boost aggressive; AC, 2026-11-28T04:52:25Z.

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
| bandwidth | read 36.8, update 52.0, triad 38.1 GB/s (ceiling 52.0) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 19% | 98% | 0 |
| blas2 | 10 | 46% | 80% | 0 |
| blas3 | 10 | 14% | 21% | 0 |
| factor | 12 | 3% | 5% | 0 |
| eigen | 8 | 4% | 10% | 0 |
| elementwise | 5 | 7% | 10% | 0 |
| sparse | 5 | 21% | 56% | 0 |
| direct | 19 | 1% | 16% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 12% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 29% | 32% | 0 |
| mg | 1 | 28% | 28% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 11% | 19% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 14.413 ms | 1.16 | 9.31 | 18% | ok |
| blas1 | dot | f32 | n=8388608 | 12.260 ms | 1.37 | 5.47 | 11% | ok |
| blas1 | axpy | f64 | n=8388608 | 14.954 ms | 1.12 | 13.46 | 26% | ok |
| blas1 | axpy | f32 | n=8388608 | 14.927 ms | 1.12 | 6.74 | 13% | ok |
| blas1 | scale | f64 | n=8388608 | 2.633 ms | 3.19 | 50.98 | 98% | ok |
| blas1 | two_norm | f64 | n=8388608 | 13.857 ms | 1.21 | 4.84 | 9% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.397 ms | 1.31 | 10.49 | 20% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.407 ms | 1.31 | 10.47 | 20% | ok |
| blas1 | sum | f64 | n=8388608 | 6.731 ms | 1.25 | 9.97 | 19% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 105.674 ms | 1.75 | 1.27 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.236 ms | 10.37 | 41.49 | 80% | ok |
| blas2 | gemv | f32 | n=4096 | 2.824 ms | 11.88 | 23.78 | 46% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 6.717 ms | 5.00 | 19.99 | 38% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.346 ms | 14.30 | 28.62 | 55% | ok |
| blas2 | symv | f64 | n=4096 | 3.271 ms | 10.26 | 20.54 | 40% | ok |
| blas2 | trmv | f64 | n=4096 | 1.892 ms | 8.87 | 35.50 | 68% | ok |
| blas2 | trmv | f32 | n=4096 | 1.772 ms | 9.47 | 18.96 | 36% | ok |
| blas2 | trsv | f64 | n=4096 | 1.917 ms | 8.75 | 35.03 | 67% | ok |
| blas2 | trsv | f32 | n=4096 | 1.783 ms | 9.41 | 18.84 | 36% | ok |
| blas2 | ger | f64 | n=4096 | 16.216 ms | 2.07 | 16.56 | 32% | ok |
| blas3 | gemm | f64 | 512x512x512 | 19.897 ms | 13.49 | 0.32 | 20% | ok |
| blas3 | gemm | f32 | 512x512x512 | 9.982 ms | 26.89 | 0.32 | 21% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 13.854 ms | 9.69 | 0.45 | 14% | ok |
| blas3 | symm | f64 | 512x512x512 | 20.761 ms | 12.93 | 0.30 | 19% | ok |
| blas3 | syrk | f64 | 512x512x512 | 16.821 ms | 7.98 | 0.25 | 12% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 27.785 ms | 9.66 | 0.23 | 14% | ok |
| blas3 | trmm | f64 | 512x512x512 | 30.852 ms | 4.35 | 0.17 | 6% | ok |
| blas3 | trsm | f64 | 512x512x512 | 35.255 ms | 3.81 | 0.15 | 6% | ok |
| blas3 | trsm | f32 | 512x512x512 | 32.292 ms | 4.16 | 0.08 | 3% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 11.029 ms | 0.48 | 3.85 | 7% | ok |
| factor | lu | f64 | 512x512 | 40.270 ms | 2.22 | 0.10 | 3% | ok |
| factor | lu | f32 | 512x512 | 38.472 ms | 2.33 | 0.05 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 14.282 ms | 3.13 | 0.29 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 13.706 ms | 3.26 | 0.15 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 14.401 ms | 3.11 | 0.29 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 12.907 ms | 3.47 | 0.32 | 5% | ok |
| factor | qr | f64 | 512x512 | 812.120 ms | 0.22 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 81.397 ms | 2.20 | 0.05 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 941.316 ms | 0.48 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.897 ms | 0.21 | 0.86 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 111.347 ms | 2.41 | 0.04 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 41.717 ms | 2.14 | 0.10 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 49.186 ms | 0.45 | 0.02 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 165.909 ms | 0.91 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 647.404 ms | 0.23 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 76.433 ms | 2.20 | 0.01 | 3% | ok |
| eigen | schur | f64 | 256x256 | 142.310 ms | 2.95 | 0.01 | 4% | ok |
| eigen | svd | f64 | 256x256 | 149.260 ms | 2.47 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 37.007 ms | 6.80 | 0.03 | 10% | ok |
| eigen | sylvester | f64 | 256x256 | 336.396 ms | 2.99 | 0.00 | 4% | ok |
| elementwise | exp | f64 | n=8388608 | 33.460 ms | 0.25 | 4.01 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 54.554 ms | 0.15 | 2.46 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 48.879 ms | 0.17 | 2.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 26.413 ms | 0.32 | 5.08 | 10% | ok |
| elementwise | erf | f64 | n=8388608 | 38.535 ms | 0.22 | 3.48 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.930 ms | 2.82 | 29.29 | 56% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.686 ms | 1.55 | 16.15 | 31% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 7.937 ms | 5.28 | 11.09 | 21% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 21.756 ms | 0.15 | 0.53 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 2.073 ms | 0.97 | 8.03 | 15%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 23.097 ms | 0.48 | 0.03 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 38.679 ms | 0.29 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 19.886 ms | 0.56 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 65.473 ms | 0.34 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 69.797 ms | 0.32 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 29.995 ms | 0.75 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 43.371 ms | 0.04 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.555 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.972 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 76.702 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.260 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.915 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.594 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 16.837 ms | 0.66 | 0.04 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.370 ms | 2.04 | 8.16 | 16%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.029 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.279 ms | 1.54 | 0.10 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.364 ms | 2.08 | 8.31 | 16%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.074 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 57.634 ms | 0.39 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.295 ms | 0.58 | 2.34 | 4%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.904 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.564 ms | 0.38 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.260 ms | 0.60 | 2.40 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 6.974 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 344.143 ms | 0.90 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.221 ms | 1.52 | 6.07 | 12%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.143 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 84.361 ms | 3.66 | 0.01 | 5% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.271 ms | 1.48 | 5.93 | 11%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.643 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1103.621 ms | 0.56 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.117 ms | 0.30 | 1.21 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.692 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 605.275 ms | 1.02 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.152 ms | 0.30 | 1.21 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 51.188 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 77.660 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 40.900 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 22.073 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 28.543 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.946 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 74.508 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 50.533 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.757 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 34.766 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.965 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 34.463 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 45.428 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.055 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.914 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 27.720 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.241 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.156 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.036 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.801 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.757 ms | 1.49 | 16.69 | 32% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.968 ms | 1.33 | 14.90 | 29% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.224 ms | 1.18 | 13.19 | 25% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 19.757 ms | 1.95 | 14.71 | 28% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.585 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.145 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 65.482 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.987 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 14.412 ms | 1.16 | 4.66 | 9% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 36.650 ms | 7.32 | 0.11 | 11% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.836 ms | 9.14 | 9.71 | 19% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.274 ms | 3.92 | 15.70 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.232 ms | 5.19 | 10.38 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.727 ms | 9.72 | 9.72 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.727 ms | 9.71 | 9.71 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 9.994 ms | 26.86 | 0.31 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 38.347 ms | 7.00 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 34.561 ms | 7.77 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 34.542 ms | 7.77 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 34.464 ms | 7.79 | 0.05 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## ryzen-9-8945hs -- native, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 506.3 GFLOP/s, f32 1011.7 GFLOP/s |
| ... hardware, AVX-512F | f64 506.3, f32 1011.7 GFLOP/s |
| ... this build, scalar | f64 12.5, f32 14.5 GFLOP/s |
| bandwidth | read 52.7, update 53.4, triad 41.0 GB/s (ceiling 53.4) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 82% | 126% | 0 | 53% |
| blas2 | 10 | 61% | 108% | 0 | 12% |
| blas3 | 10 | 2% | 21% | 0 | 13% |
| factor | 12 | 1% | 6% | 0 | 19% |
| eigen | 8 | 0% | 2% | 0 | 13% |
| elementwise | 5 | 6% | 10% | 0 | 13% |
| sparse | 5 | 25% | 290% | 0 | 14% |
| direct | 19 | 0% | 16% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 11% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 25% | 31% | 0 | 12% |
| mg | 1 | 29% | 29% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 11% | 18% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 95% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.297 ms | 7.30 | 58.43 | 109% | ok | 6.3x (78%) |
| blas1 | dot | f32 | n=8388608 | 1.541 ms | 10.88 | 43.53 | 82% | ok | 8.0x (99%) |
| blas1 | axpy | f64 | n=8388608 | 3.532 ms | 4.75 | 57.00 | 107% | ok | 4.2x (53%) |
| blas1 | axpy | f32 | n=8388608 | 1.916 ms | 8.76 | 52.54 | 98% | ok | 7.8x (97%) |
| blas1 | scale | f64 | n=8388608 | 1.990 ms | 4.21 | 67.44 | 126% | ok | 1.3x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 1.740 ms | 9.64 | 38.57 | 72% | ok | 8.0x (100%) |
| blas1 | one_norm | f64 | n=8388608 | 6.416 ms | 1.31 | 10.46 | 20% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.443 ms | 1.30 | 10.42 | 20% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 6.394 ms | 1.31 | 10.49 | 20% | ok | 1.1x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 108.433 ms | 1.70 | 1.24 | 2% | ok | 1.0x (12%) |
| blas2 | gemv | f64 | n=4096 | 2.414 ms | 13.90 | 55.64 | 104% | ok | 1.3x (17%) |
| blas2 | gemv | f32 | n=4096 | 1.183 ms | 28.36 | 56.74 | 106% | ok | 2.4x (30%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.330 ms | 14.40 | 57.64 | 108% | ok | 2.9x (36%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.163 ms | 28.84 | 57.72 | 108% | ok | 2.0x (25%) |
| blas2 | symv | f64 | n=4096 | 3.879 ms | 8.65 | 17.32 | 32% | ok | 0.8x (11%) |
| blas2 | trmv | f64 | n=4096 | 2.129 ms | 7.88 | 31.56 | 59% | ok | 0.9x (11%) |
| blas2 | trmv | f32 | n=4096 | 1.911 ms | 8.78 | 17.57 | 33% | ok | 0.9x (12%) |
| blas2 | trsv | f64 | n=4096 | 2.067 ms | 8.12 | 32.49 | 61% | ok | 0.9x (12%) |
| blas2 | trsv | f32 | n=4096 | 1.913 ms | 8.77 | 17.55 | 33% | ok | 0.9x (12%) |
| blas2 | ger | f64 | n=4096 | 16.457 ms | 2.04 | 16.32 | 31% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.521 ms | 106.47 | 2.50 | 21% | ok | 7.9x (99%) |
| blas3 | gemm | f32 | 512x512x512 | 1.244 ms | 215.70 | 2.53 | 21% | ok | 8.0x (100%) |
| blas3 | gemmt | f64 | 512x512x512 | 13.956 ms | 9.62 | 0.45 | 2% | ok | 1.0x (12%) |
| blas3 | symm | f64 | 512x512x512 | 20.648 ms | 13.00 | 0.30 | 3% | ok | 1.0x (13%) |
| blas3 | syrk | f64 | 512x512x512 | 16.772 ms | 8.00 | 0.25 | 2% | ok | 1.0x (13%) |
| blas3 | syr2k | f64 | 512x512x512 | 28.433 ms | 9.44 | 0.22 | 2% | ok | 1.0x (12%) |
| blas3 | trmm | f64 | 512x512x512 | 30.685 ms | 4.37 | 0.17 | 1% | ok | 1.0x (13%) |
| blas3 | trsm | f64 | 512x512x512 | 35.242 ms | 3.81 | 0.15 | 1% | ok | 1.0x (13%) |
| blas3 | trsm | f32 | 512x512x512 | 32.146 ms | 4.18 | 0.08 | 0% | ok | 1.0x (13%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 11.203 ms | 0.47 | 3.79 | 7% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 25.940 ms | 3.45 | 0.16 | 1% | ok | 1.6x (19%) |
| factor | lu | f32 | 512x512 | 23.963 ms | 3.73 | 0.09 | 0% | ok | 1.6x (20%) |
| factor | cholesky | f64 | 512x512 | 14.268 ms | 3.14 | 0.29 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 13.691 ms | 3.27 | 0.15 | 0% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 14.384 ms | 3.11 | 0.29 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 13.072 ms | 3.42 | 0.32 | 1% | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 501.841 ms | 0.36 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 50.862 ms | 3.52 | 0.08 | 1% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 568.612 ms | 0.79 | 0.01 | 0% | ok | 1.7x (21%) |
| factor | householder | f64 | 512x512 | 1.250 ms | 0.84 | 3.36 | 6%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 96.925 ms | 2.77 | 0.04 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 27.412 ms | 3.26 | 0.15 | 1% | ok | 1.5x (19%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 49.422 ms | 0.45 | 0.02 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 163.895 ms | 0.92 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 643.626 ms | 0.23 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 76.657 ms | 2.19 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | schur | f64 | 256x256 | 142.029 ms | 2.95 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 181.679 ms | 2.03 | 0.01 | 0% | ok | 0.8x (10%) |
| eigen | expm | f64 | 256x256 | 24.666 ms | 10.20 | 0.04 | 2% | ok | 1.5x (19%) |
| eigen | sylvester | f64 | 256x256 | 324.617 ms | 3.10 | 0.00 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 39.706 ms | 0.21 | 3.38 | 6% | ok | 0.8x (11%) |
| elementwise | log | f64 | n=8388608 | 53.108 ms | 0.16 | 2.53 | 5% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 48.494 ms | 0.17 | 2.77 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 26.401 ms | 0.32 | 5.08 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 38.566 ms | 0.22 | 3.48 | 7% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.176 ms | 14.87 | 154.72 | 290% ! | ok | 5.3x (66%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.063 ms | 1.27 | 13.20 | 25% | ok | 0.8x (10%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.259 ms | 18.54 | 38.98 | 73% | ok | 3.5x (44%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 21.124 ms | 0.15 | 0.54 | 1%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.883 ms | 1.07 | 8.84 | 17%\* | ok | 1.1x (14%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 23.311 ms | 0.48 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 39.368 ms | 0.28 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 19.500 ms | 0.57 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 65.227 ms | 0.34 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 69.882 ms | 0.32 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 29.839 ms | 0.75 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 42.445 ms | 0.04 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.628 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 7.059 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 75.790 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.235 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.707 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.684 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.064 ms | 0.66 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.344 ms | 2.20 | 8.80 | 16%\* | ok | 1.1x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.182 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.283 ms | 1.54 | 0.10 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.345 ms | 2.19 | 8.76 | 16%\* | ok | 1.1x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.137 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.304 ms | 0.38 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.286 ms | 0.59 | 2.35 | 4%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.936 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.817 ms | 0.37 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.258 ms | 0.60 | 2.41 | 5%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 7.182 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 346.734 ms | 0.89 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.248 ms | 1.50 | 5.99 | 11%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.625 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 92.537 ms | 3.33 | 0.01 | 1% | ok | 0.9x (11%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.338 ms | 1.44 | 5.76 | 11%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.640 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1112.142 ms | 0.55 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.230 ms | 0.30 | 1.20 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.641 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 608.671 ms | 1.01 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.251 ms | 0.30 | 1.20 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 51.263 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 77.373 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 40.648 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.984 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 28.459 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.806 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 73.999 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 44.450 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.622 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 34.687 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.894 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 34.390 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 45.391 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.055 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 3.092 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| pc | ilut | f64 | 128^2 grid | 27.502 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 2.244 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 128^2 grid | 2.169 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.036 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.805 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.751 ms | 1.50 | 16.75 | 31% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.164 ms | 1.21 | 13.55 | 25% | ok | 0.9x (11%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.235 ms | 1.17 | 13.12 | 25% | ok | 1.0x (12%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.551 ms | 2.07 | 15.67 | 29% | ok | 1.1x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.490 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.175 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 67.978 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.999 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 14.382 ms | 1.17 | 4.67 | 9% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.785 ms | 56.10 | 0.88 | 11% | ok | 7.7x (96%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.837 ms | 9.13 | 9.70 | 18% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.999 ms | 4.20 | 16.78 | n/a | ok | 1.1x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.232 ms | 5.19 | 10.38 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.727 ms | 9.72 | 9.72 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.726 ms | 9.72 | 9.72 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.265 ms | 212.27 | 2.49 | n/a | ok | 7.9x (99%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.995 ms | 53.74 | 0.42 | n/a | ok | 7.7x (96%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.509 ms | 59.54 | 0.35 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.502 ms | 59.62 | 0.35 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.530 ms | 59.26 | 0.35 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

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

## ryzen-9-8945hs -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 506.0 GFLOP/s, f32 1011.8 GFLOP/s |
| ... hardware, AVX-512F | f64 506.0, f32 1011.8 GFLOP/s |
| ... this build, SSE2 | f64 318.2, f32 635.4 GFLOP/s |
| bandwidth | read 52.9, update 53.2, triad 41.1 GB/s (ceiling 53.2) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 112% | 128% | 0 | 13% |
| blas2 | 10 | 65% | 119% | 0 | 13% |
| blas3 | 10 | 9% | 19% | 0 | 39% |
| factor | 12 | 1% | 6% | 0 | 19% |
| eigen | 8 | 0% | 2% | 0 | 13% |
| elementwise | 5 | 7% | 10% | 0 | 13% |
| sparse | 5 | 33% | 99% | 0 | 14% |
| direct | 19 | 0% | 17% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 11% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 25% | 33% | 0 | 13% |
| mg | 1 | 30% | 30% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 58% | 58% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 74% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.246 ms | 7.47 | 59.75 | 112% | ok | 1.1x (13%) |
| blas1 | dot | f32 | n=8388608 | 0.987 ms | 17.00 | 67.99 | 128% | ok | 1.3x (16%) |
| blas1 | axpy | f64 | n=8388608 | 3.477 ms | 4.83 | 57.90 | 109% | ok | 1.0x (13%) |
| blas1 | axpy | f32 | n=8388608 | 1.623 ms | 10.34 | 62.02 | 117% | ok | 1.1x (13%) |
| blas1 | scale | f64 | n=8388608 | 1.982 ms | 4.23 | 67.72 | 127% | ok | 1.2x (15%) |
| blas1 | two_norm | f64 | n=8388608 | 1.046 ms | 16.05 | 64.19 | 121% | ok | 1.1x (14%) |
| blas1 | one_norm | f64 | n=8388608 | 6.416 ms | 1.31 | 10.46 | 20% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.402 ms | 1.31 | 10.48 | 20% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 6.394 ms | 1.31 | 10.50 | 20% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 111.461 ms | 1.66 | 1.20 | 2% | ok | 1.0x (12%) |
| blas2 | gemv | f64 | n=4096 | 2.306 ms | 14.55 | 58.24 | 110% | ok | 1.5x (19%) |
| blas2 | gemv | f32 | n=4096 | 1.064 ms | 31.52 | 63.08 | 119% | ok | 1.7x (21%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.337 ms | 14.36 | 57.47 | 108% | ok | 1.7x (21%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.209 ms | 27.76 | 55.55 | 104% | ok | 1.7x (21%) |
| blas2 | symv | f64 | n=4096 | 3.729 ms | 9.00 | 18.01 | 34% | ok | 0.9x (11%) |
| blas2 | trmv | f64 | n=4096 | 1.959 ms | 8.57 | 34.30 | 65% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 1.163 ms | 14.42 | 28.88 | 54% | ok | 0.9x (11%) |
| blas2 | trsv | f64 | n=4096 | 1.937 ms | 8.66 | 34.69 | 65% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.186 ms | 14.15 | 28.33 | 53% | ok | 1.0x (12%) |
| blas2 | ger | f64 | n=4096 | 16.311 ms | 2.06 | 16.46 | 31% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.739 ms | 98.02 | 2.30 | 19% | ok | 5.0x (63%) |
| blas3 | gemm | f32 | 512x512x512 | 1.628 ms | 164.88 | 1.93 | 16% | ok | 4.3x (53%) |
| blas3 | gemmt | f64 | 512x512x512 | 4.338 ms | 30.94 | 1.45 | 6% | ok | 2.1x (27%) |
| blas3 | symm | f64 | 512x512x512 | 3.873 ms | 69.31 | 1.62 | 14% | ok | 3.9x (49%) |
| blas3 | syrk | f64 | 512x512x512 | 2.948 ms | 45.53 | 1.42 | 9% | ok | 3.1x (39%) |
| blas3 | syr2k | f64 | 512x512x512 | 5.538 ms | 48.47 | 1.14 | 10% | ok | 3.3x (41%) |
| blas3 | trmm | f64 | 512x512x512 | 6.534 ms | 20.54 | 0.80 | 4% | ok | 1.8x (23%) |
| blas3 | trsm | f64 | 512x512x512 | 6.867 ms | 19.55 | 0.76 | 4% | ok | 1.8x (22%) |
| blas3 | trsm | f32 | 512x512x512 | 5.401 ms | 24.85 | 0.49 | 2% | ok | 1.5x (19%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 11.238 ms | 0.47 | 3.78 | 7% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 25.800 ms | 3.47 | 0.16 | 1% | ok | 1.6x (19%) |
| factor | lu | f32 | 512x512 | 24.747 ms | 3.62 | 0.08 | 0% | ok | 1.5x (19%) |
| factor | cholesky | f64 | 512x512 | 14.285 ms | 3.13 | 0.29 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 13.692 ms | 3.27 | 0.15 | 0% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 14.393 ms | 3.11 | 0.29 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 12.850 ms | 3.48 | 0.33 | 1% | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 499.295 ms | 0.36 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 51.320 ms | 3.49 | 0.08 | 1% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 568.995 ms | 0.79 | 0.01 | 0% | ok | 1.6x (21%) |
| factor | householder | f64 | 512x512 | 1.250 ms | 0.84 | 3.36 | 6%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 97.624 ms | 2.75 | 0.04 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 27.907 ms | 3.21 | 0.15 | 1% | ok | 1.5x (19%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 49.450 ms | 0.45 | 0.02 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 160.273 ms | 0.94 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 652.489 ms | 0.23 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 73.519 ms | 2.28 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | schur | f64 | 256x256 | 141.285 ms | 2.97 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 149.640 ms | 2.47 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 21.870 ms | 11.51 | 0.05 | 2% | ok | 1.5x (18%) |
| eigen | sylvester | f64 | 256x256 | 313.226 ms | 3.21 | 0.01 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 33.366 ms | 0.25 | 4.02 | 8% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 51.704 ms | 0.16 | 2.60 | 5% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 49.045 ms | 0.17 | 2.74 | 5% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 26.344 ms | 0.32 | 5.09 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 38.773 ms | 0.22 | 3.46 | 7% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.515 ms | 5.08 | 52.89 | 99% | ok | 1.8x (22%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.549 ms | 1.69 | 17.57 | 33% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.281 ms | 18.36 | 38.60 | 73% | ok | 3.5x (43%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 22.349 ms | 0.15 | 0.51 | 1%\* | ok | 0.9x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.883 ms | 1.07 | 8.84 | 17%\* | ok | 1.1x (14%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 21.944 ms | 0.51 | 0.03 | 0% | ok | 1.1x (14%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 34.672 ms | 0.32 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 16.089 ms | 0.70 | 0.05 | 0% | ok | 1.1x (14%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 65.285 ms | 0.34 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 68.681 ms | 0.33 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 30.301 ms | 0.74 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 42.535 ms | 0.04 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.575 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.916 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 74.892 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.228 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.400 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.925 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.887 ms | 0.63 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.343 ms | 2.21 | 8.83 | 17%\* | ok | 1.1x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.127 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.244 ms | 1.54 | 0.10 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.352 ms | 2.15 | 8.59 | 16%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.089 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 57.795 ms | 0.39 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.245 ms | 0.61 | 2.43 | 5%\* | ok | 1.1x (14%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.924 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.512 ms | 0.38 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.298 ms | 0.58 | 2.33 | 4%\* | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 6.913 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 345.801 ms | 0.89 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.283 ms | 1.48 | 5.90 | 11%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 16.806 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 84.386 ms | 3.66 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.361 ms | 1.43 | 5.71 | 11%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.609 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1094.344 ms | 0.56 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.220 ms | 0.30 | 1.20 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.568 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 611.229 ms | 1.01 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.145 ms | 0.30 | 1.21 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 41.197 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 58.480 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 35.961 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.070 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.677 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 19.931 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 43.231 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 37.219 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 21.940 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 23.659 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 22.227 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 22.321 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 27.228 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.055 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 3.038 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 27.785 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 2.230 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 2.169 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.037 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.818 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.688 ms | 1.55 | 17.37 | 33% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.215 ms | 1.18 | 13.24 | 25% | ok | 0.9x (11%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.232 ms | 1.17 | 13.14 | 25% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.515 ms | 2.08 | 15.70 | 30% | ok | 1.1x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.590 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.240 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 65.176 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.060 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 2.173 ms | 7.72 | 30.88 | 58% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 3.125 ms | 85.91 | 1.34 | 17% | ok | 6.2x (77%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.578 ms | 29.02 | 30.83 | 58% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.425 ms | 11.77 | 47.08 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.598 ms | 28.07 | 56.15 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.430 ms | 39.00 | 39.00 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.387 ms | 43.34 | 43.34 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.878 ms | 93.27 | 1.09 | n/a | ok | 6.6x (82%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 3.009 ms | 89.21 | 0.70 | n/a | ok | 7.3x (91%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 3.039 ms | 88.33 | 0.52 | n/a | ok | 7.4x (93%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.772 ms | 347.53 | 2.04 | n/a | ok | 6.9x (86%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.767 ms | 349.94 | 2.05 | n/a | ok | 5.9x (74%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## ryzen-9-8945hs -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 79.0 GFLOP/s, f32 126.4 GFLOP/s |
| ... hardware, AVX-512F | f64 79.0, f32 126.4 GFLOP/s |
| ... this build, scalar | f64 1.9, f32 1.9 GFLOP/s |
| bandwidth | read 36.9, update 51.8, triad 37.8 GB/s (ceiling 51.8) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 20% | 98% | 0 |
| blas2 | 10 | 71% | 85% | 0 |
| blas3 | 10 | 65% | 89% | 0 |
| factor | 12 | 10% | 23% | 0 |
| eigen | 8 | 14% | 43% | 0 |
| elementwise | 5 | 6% | 10% | 0 |
| sparse | 5 | 18% | 57% | 0 |
| direct | 19 | 1% | 17% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 12% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 29% | 31% | 0 |
| mg | 1 | 28% | 28% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 11% | 19% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.232 ms | 1.49 | 11.95 | 23% | ok |
| blas1 | dot | f32 | n=8388608 | 12.256 ms | 1.37 | 5.48 | 11% | ok |
| blas1 | axpy | f64 | n=8388608 | 14.984 ms | 1.12 | 13.44 | 26% | ok |
| blas1 | axpy | f32 | n=8388608 | 14.940 ms | 1.12 | 6.74 | 13% | ok |
| blas1 | scale | f64 | n=8388608 | 2.638 ms | 3.18 | 50.87 | 98% | ok |
| blas1 | two_norm | f64 | n=8388608 | 12.260 ms | 1.37 | 5.47 | 11% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.394 ms | 1.31 | 10.49 | 20% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.413 ms | 1.31 | 10.47 | 20% | ok |
| blas1 | sum | f64 | n=8388608 | 6.397 ms | 1.31 | 10.49 | 20% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 111.558 ms | 1.65 | 1.20 | 2% | ok |
| blas2 | gemv | f64 | n=4096 | 3.645 ms | 9.21 | 36.84 | 71% | ok |
| blas2 | gemv | f32 | n=4096 | 1.520 ms | 22.08 | 44.18 | 85% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 4.216 ms | 7.96 | 31.85 | 61% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.558 ms | 21.54 | 43.11 | 83% | ok |
| blas2 | symv | f64 | n=4096 | 1.925 ms | 17.43 | 34.90 | 67% | ok |
| blas2 | trmv | f64 | n=4096 | 1.825 ms | 9.19 | 36.81 | 71% | ok |
| blas2 | trmv | f32 | n=4096 | 1.083 ms | 15.50 | 31.03 | 60% | ok |
| blas2 | trsv | f64 | n=4096 | 1.773 ms | 9.46 | 37.89 | 73% | ok |
| blas2 | trsv | f32 | n=4096 | 0.983 ms | 17.07 | 34.18 | 66% | ok |
| blas2 | ger | f64 | n=4096 | 6.639 ms | 5.05 | 40.44 | 78% | ok |
| blas3 | gemm | f64 | 512x512x512 | 4.573 ms | 58.70 | 1.38 | 74% | ok |
| blas3 | gemm | f32 | 512x512x512 | 2.386 ms | 112.52 | 1.32 | 89% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 3.421 ms | 39.24 | 1.84 | 50% | ok |
| blas3 | symm | f64 | 512x512x512 | 4.751 ms | 56.50 | 1.32 | 72% | ok |
| blas3 | syrk | f64 | 512x512x512 | 3.702 ms | 36.26 | 1.13 | 46% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 5.996 ms | 44.77 | 1.05 | 57% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.362 ms | 56.83 | 2.22 | 72% | ok |
| blas3 | trsm | f64 | 512x512x512 | 3.501 ms | 38.34 | 1.50 | 49% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.639 ms | 81.92 | 1.60 | 65% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.867 ms | 0.49 | 3.91 | 8% | ok |
| factor | lu | f64 | 512x512 | 6.332 ms | 14.13 | 0.66 | 18% | ok |
| factor | lu | f32 | 512x512 | 3.091 ms | 28.95 | 0.68 | 23% | ok |
| factor | cholesky | f64 | 512x512 | 5.469 ms | 8.18 | 0.77 | 10% | ok |
| factor | cholesky | f32 | 512x512 | 2.162 ms | 20.70 | 0.97 | 16% | ok |
| factor | ldlt | f64 | 512x512 | 14.783 ms | 3.03 | 0.28 | 4% | ok |
| factor | ldlt_bk | f64 | 512x512 | 12.812 ms | 3.49 | 0.33 | 4% | ok |
| factor | qr | f64 | 512x512 | 12.955 ms | 13.81 | 0.32 | 17% | ok |
| factor | lq | f64 | 512x512 | 80.638 ms | 2.22 | 0.05 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 935.701 ms | 0.48 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.933 ms | 0.21 | 0.85 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 76.131 ms | 3.53 | 0.06 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.937 ms | 12.90 | 0.60 | 16% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 2.178 ms | 10.27 | 0.48 | 13% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 160.457 ms | 0.94 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 647.134 ms | 0.23 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 21.106 ms | 7.95 | 0.05 | 10% | ok |
| eigen | schur | f64 | 256x256 | 27.364 ms | 15.33 | 0.04 | 19% | ok |
| eigen | svd | f64 | 256x256 | 10.875 ms | 33.94 | 0.10 | 43% | ok |
| eigen | expm | f64 | 256x256 | 19.522 ms | 12.89 | 0.05 | 16% | ok |
| eigen | sylvester | f64 | 256x256 | 90.189 ms | 11.16 | 0.02 | 14% | ok |
| elementwise | exp | f64 | n=8388608 | 33.413 ms | 0.25 | 4.02 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 53.140 ms | 0.16 | 2.53 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 47.901 ms | 0.18 | 2.80 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 26.476 ms | 0.32 | 5.07 | 10% | ok |
| elementwise | erf | f64 | n=8388608 | 45.035 ms | 0.19 | 2.98 | 6% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.927 ms | 2.82 | 29.36 | 57% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.536 ms | 1.70 | 17.72 | 34% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 9.686 ms | 4.32 | 9.09 | 18% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 21.385 ms | 0.15 | 0.54 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.914 ms | 1.05 | 8.70 | 17%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 23.761 ms | 0.47 | 0.03 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 38.876 ms | 0.29 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 19.606 ms | 0.57 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 70.147 ms | 0.32 | 0.01 | 0% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 69.874 ms | 0.32 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 29.809 ms | 0.75 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 42.799 ms | 0.04 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.571 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.933 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 75.003 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.245 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.715 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.604 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.602 ms | 0.64 | 0.04 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.348 ms | 2.18 | 8.70 | 17%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.055 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.203 ms | 1.55 | 0.10 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.366 ms | 2.06 | 8.25 | 16%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.097 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.023 ms | 0.38 | 0.01 | 0% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.229 ms | 0.61 | 2.46 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.891 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.536 ms | 0.38 | 0.01 | 0% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.210 ms | 0.63 | 2.50 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 6.963 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 337.330 ms | 0.91 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.216 ms | 1.52 | 6.08 | 12%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.161 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 85.624 ms | 3.60 | 0.01 | 5% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.278 ms | 1.48 | 5.92 | 11%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.591 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1109.012 ms | 0.56 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.267 ms | 0.30 | 1.20 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.522 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 626.061 ms | 0.99 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.072 ms | 0.30 | 1.22 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 49.123 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 73.650 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 39.768 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.025 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 27.378 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.402 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 66.670 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 43.524 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.004 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 32.893 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 31.010 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 31.962 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.048 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.055 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.933 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 27.100 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.241 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.159 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.036 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.661 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.842 ms | 1.42 | 15.92 | 31% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.962 ms | 1.33 | 14.95 | 29% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.237 ms | 1.17 | 13.11 | 25% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 19.807 ms | 1.94 | 14.67 | 28% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.547 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.322 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 67.663 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.813 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 12.258 ms | 1.37 | 5.47 | 11% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 37.330 ms | 7.19 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.835 ms | 9.14 | 9.72 | 19% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.999 ms | 4.20 | 16.78 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.247 ms | 5.17 | 10.33 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.122 ms | 7.91 | 7.91 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.122 ms | 7.91 | 7.91 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 10.216 ms | 26.28 | 0.31 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 39.455 ms | 6.80 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 34.554 ms | 7.77 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 34.618 ms | 7.75 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 34.482 ms | 7.78 | 0.05 | n/a | ok |

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
| FMA peak (ceiling) | f64 556.5 GFLOP/s, f32 1012.4 GFLOP/s |
| ... hardware, AVX-512F | f64 556.5, f32 1012.4 GFLOP/s |
| ... this build, scalar | f64 14.6, f32 14.6 GFLOP/s |
| bandwidth | read 52.6, update 53.2, triad 40.9 GB/s (ceiling 53.2) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 82% | 127% | 0 | 54% |
| blas2 | 10 | 103% | 119% | 0 | 18% |
| blas3 | 10 | 43% | 71% | 0 | 67% |
| factor | 12 | 1% | 6% | 0 | 12% |
| eigen | 8 | 2% | 5% | 0 | 11% |
| elementwise | 5 | 6% | 10% | 0 | 13% |
| sparse | 5 | 20% | 280% | 0 | 13% |
| direct | 19 | 0% | 16% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 11% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 31% | 0 | 13% |
| mg | 1 | 29% | 29% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 10% | 18% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 96% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.286 ms | 7.34 | 58.71 | 110% | ok | 4.9x (61%) |
| blas1 | dot | f32 | n=8388608 | 1.540 ms | 10.90 | 43.58 | 82% | ok | 8.0x (99%) |
| blas1 | axpy | f64 | n=8388608 | 3.500 ms | 4.79 | 57.52 | 108% | ok | 4.3x (54%) |
| blas1 | axpy | f32 | n=8388608 | 1.922 ms | 8.73 | 52.36 | 98% | ok | 7.8x (97%) |
| blas1 | scale | f64 | n=8388608 | 1.984 ms | 4.23 | 67.64 | 127% | ok | 1.3x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 1.542 ms | 10.88 | 43.52 | 82% | ok | 8.0x (99%) |
| blas1 | one_norm | f64 | n=8388608 | 6.421 ms | 1.31 | 10.45 | 20% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.440 ms | 1.30 | 10.42 | 20% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 6.394 ms | 1.31 | 10.50 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 102.948 ms | 1.79 | 1.30 | 2% | ok | 1.1x (14%) |
| blas2 | gemv | f64 | n=4096 | 2.491 ms | 13.47 | 53.91 | 101% | ok | 1.5x (18%) |
| blas2 | gemv | f32 | n=4096 | 1.173 ms | 28.61 | 57.24 | 108% | ok | 1.3x (16%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.662 ms | 12.60 | 50.44 | 95% | ok | 1.6x (20%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.290 ms | 26.02 | 52.06 | 98% | ok | 1.2x (15%) |
| blas2 | symv | f64 | n=4096 | 1.226 ms | 27.38 | 54.80 | 103% | ok | 1.6x (20%) |
| blas2 | trmv | f64 | n=4096 | 1.226 ms | 13.68 | 54.77 | 103% | ok | 1.5x (19%) |
| blas2 | trmv | f32 | n=4096 | 0.530 ms | 31.63 | 63.32 | 119% | ok | 2.0x (26%) |
| blas2 | trsv | f64 | n=4096 | 1.706 ms | 9.84 | 39.38 | 74% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.921 ms | 18.22 | 36.48 | 69% | ok | 1.1x (13%) |
| blas2 | ger | f64 | n=4096 | 4.675 ms | 7.18 | 57.44 | 108% | ok | 1.4x (18%) |
| blas3 | gemm | f64 | 512x512x512 | 0.682 ms | 393.83 | 9.23 | 71% | ok | 6.7x (84%) |
| blas3 | gemm | f32 | 512x512x512 | 0.385 ms | 697.96 | 8.18 | 69% | ok | 6.2x (78%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.674 ms | 80.19 | 3.76 | 14% | ok | 2.0x (26%) |
| blas3 | symm | f64 | 512x512x512 | 0.682 ms | 393.66 | 9.23 | 71% | ok | 7.0x (87%) |
| blas3 | syrk | f64 | 512x512x512 | 1.991 ms | 67.40 | 2.11 | 12% | ok | 1.9x (23%) |
| blas3 | syr2k | f64 | 512x512x512 | 2.703 ms | 99.31 | 2.33 | 18% | ok | 2.2x (28%) |
| blas3 | trmm | f64 | 512x512x512 | 0.442 ms | 303.94 | 11.87 | 55% | ok | 5.3x (67%) |
| blas3 | trsm | f64 | 512x512x512 | 0.603 ms | 222.47 | 8.69 | 40% | ok | 5.8x (73%) |
| blas3 | trsm | f32 | 512x512x512 | 0.307 ms | 437.33 | 8.54 | 43% | ok | 5.3x (67%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 11.179 ms | 0.47 | 3.80 | 7% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 11.121 ms | 8.05 | 0.38 | 1% | ok | 0.6x (7%) |
| factor | lu | f32 | 512x512 | 5.079 ms | 17.62 | 0.41 | 2% | ok | 0.6x (8%) |
| factor | cholesky | f64 | 512x512 | 6.985 ms | 6.41 | 0.60 | 1% | ok | 0.8x (10%) |
| factor | cholesky | f32 | 512x512 | 2.446 ms | 18.29 | 0.86 | 2% | ok | 0.9x (11%) |
| factor | ldlt | f64 | 512x512 | 14.820 ms | 3.02 | 0.28 | 1% | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 12.647 ms | 3.54 | 0.33 | 1% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 14.107 ms | 12.69 | 0.30 | 2% | ok | 0.9x (11%) |
| factor | lq | f64 | 512x512 | 50.925 ms | 3.51 | 0.08 | 1% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 573.863 ms | 0.78 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.260 ms | 0.83 | 3.33 | 6%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 82.035 ms | 3.27 | 0.05 | 1% | ok | 0.9x (12%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 8.768 ms | 10.21 | 0.48 | 2% | ok | 0.8x (10%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 4.272 ms | 5.24 | 0.25 | 1% | ok | 0.5x (6%) |
| eigen | eigen_symmetric | f64 | 256x256 | 161.594 ms | 0.93 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 675.672 ms | 0.22 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 256x256 | 25.125 ms | 6.68 | 0.04 | 1% | ok | 0.8x (11%) |
| eigen | schur | f64 | 256x256 | 31.277 ms | 13.41 | 0.03 | 2% | ok | 0.9x (11%) |
| eigen | svd | f64 | 256x256 | 13.537 ms | 27.27 | 0.08 | 5% | ok | 0.8x (10%) |
| eigen | expm | f64 | 256x256 | 20.662 ms | 12.18 | 0.05 | 2% | ok | 0.9x (12%) |
| eigen | sylvester | f64 | 256x256 | 99.522 ms | 10.11 | 0.02 | 2% | ok | 0.9x (11%) |
| elementwise | exp | f64 | n=8388608 | 33.237 ms | 0.25 | 4.04 | 8% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 52.943 ms | 0.16 | 2.54 | 5% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 47.736 ms | 0.18 | 2.81 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 26.478 ms | 0.32 | 5.07 | 10% | ok | 1.0x (12%) |
| elementwise | erf | f64 | n=8388608 | 42.353 ms | 0.20 | 3.17 | 6% | ok | 1.1x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.183 ms | 14.33 | 149.04 | 280% ! | ok | 5.1x (63%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.505 ms | 1.04 | 10.87 | 20% | ok | 0.6x (8%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.247 ms | 18.64 | 39.18 | 74% | ok | 4.3x (54%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 21.528 ms | 0.15 | 0.53 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.885 ms | 1.06 | 8.83 | 17%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 24.071 ms | 0.46 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 39.005 ms | 0.29 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 20.241 ms | 0.55 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 65.743 ms | 0.34 | 0.01 | 0% | ok | 1.1x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 70.477 ms | 0.32 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 30.176 ms | 0.74 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 42.748 ms | 0.04 | 0.00 | 0%\* | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.603 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.976 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 74.883 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.244 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.998 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.720 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.677 ms | 0.63 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.346 ms | 2.19 | 8.74 | 16%\* | ok | 1.0x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.254 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.286 ms | 1.54 | 0.10 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.346 ms | 2.19 | 8.74 | 16%\* | ok | 1.1x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.143 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 58.205 ms | 0.38 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.333 ms | 0.57 | 2.27 | 4%\* | ok | 0.9x (12%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.946 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.859 ms | 0.37 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.280 ms | 0.59 | 2.36 | 4%\* | ok | 0.9x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 7.032 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 344.571 ms | 0.90 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.275 ms | 1.48 | 5.92 | 11%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.939 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 84.565 ms | 3.65 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.318 ms | 1.45 | 5.81 | 11%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.712 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1101.795 ms | 0.56 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.298 ms | 0.30 | 1.19 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.559 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 625.856 ms | 0.99 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.230 ms | 0.30 | 1.20 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 49.162 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 73.784 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 39.841 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.191 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 27.200 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.233 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 66.550 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 43.215 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.942 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 32.822 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 30.917 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 31.928 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 41.334 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.054 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.950 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 27.149 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 2.220 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 2.129 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.035 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.183 ms | 0.00 | 0.00 | n/a | ok | 0.8x (10%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.760 ms | 1.49 | 16.67 | 31% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.909 ms | 1.37 | 15.37 | 29% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.249 ms | 1.16 | 13.04 | 25% | ok | 1.0x (12%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.543 ms | 2.07 | 15.67 | 29% | ok | 1.1x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.560 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.329 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 64.778 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.745 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 12.254 ms | 1.37 | 5.48 | 10% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.875 ms | 55.06 | 0.86 | 10% | ok | 7.7x (96%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.838 ms | 9.13 | 9.70 | 18% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.038 ms | 4.16 | 16.62 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.245 ms | 5.17 | 10.34 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.122 ms | 7.91 | 7.91 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.122 ms | 7.91 | 7.91 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.269 ms | 211.58 | 2.48 | n/a | ok | 8.1x (101%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.024 ms | 53.43 | 0.42 | n/a | ok | 7.9x (98%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.507 ms | 59.55 | 0.35 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.513 ms | 59.48 | 0.35 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.511 ms | 59.50 | 0.35 | n/a | ok | 7.6x (96%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

