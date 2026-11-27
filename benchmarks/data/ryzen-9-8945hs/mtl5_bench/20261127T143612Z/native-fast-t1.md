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

