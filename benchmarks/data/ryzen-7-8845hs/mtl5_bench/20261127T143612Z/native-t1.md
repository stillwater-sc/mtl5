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

