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

