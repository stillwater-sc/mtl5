# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,8, CPUs 0,2,4,6,8,10,12,14, clock policy power scheme copy of Balanced with processor state 100-100% and EPP 0, boost aggressive; AC, 2026-11-28T02:15:42Z.

## ryzen-7-8845hs -- native, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 80.0 GFLOP/s, f32 160.1 GFLOP/s |
| ... hardware, AVX-512F | f64 80.0, f32 160.1 GFLOP/s |
| ... this build, scalar | f64 1.6, f32 1.9 GFLOP/s |
| bandwidth | read 43.2, update 53.6, triad 39.1 GB/s (ceiling 53.6) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 25% | 109% | 0 |
| blas2 | 10 | 55% | 84% | 0 |
| blas3 | 10 | 15% | 21% | 0 |
| factor | 12 | 3% | 5% | 0 |
| eigen | 8 | 4% | 10% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 25% | 59% | 0 |
| direct | 19 | 1% | 21% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 13% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 36% | 36% | 0 |
| mg | 1 | 35% | 35% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 12% | 23% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.511 ms | 1.46 | 11.66 | 22% | ok |
| blas1 | dot | f32 | n=8388608 | 9.849 ms | 1.70 | 6.81 | 13% | ok |
| blas1 | axpy | f64 | n=8388608 | 11.970 ms | 1.40 | 16.82 | 31% | ok |
| blas1 | axpy | f32 | n=8388608 | 11.981 ms | 1.40 | 8.40 | 16% | ok |
| blas1 | scale | f64 | n=8388608 | 2.295 ms | 3.65 | 58.47 | 109% | ok |
| blas1 | two_norm | f64 | n=8388608 | 11.018 ms | 1.52 | 6.09 | 11% | ok |
| blas1 | one_norm | f64 | n=8388608 | 5.062 ms | 1.66 | 13.26 | 25% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 5.075 ms | 1.65 | 13.22 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 5.062 ms | 1.66 | 13.26 | 25% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 90.589 ms | 2.04 | 1.48 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 2.972 ms | 11.29 | 45.18 | 84% | ok |
| blas2 | gemv | f32 | n=4096 | 2.253 ms | 14.89 | 29.80 | 56% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.054 ms | 4.76 | 19.04 | 36% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.292 ms | 14.64 | 29.29 | 55% | ok |
| blas2 | symv | f64 | n=4096 | 2.798 ms | 11.99 | 24.01 | 45% | ok |
| blas2 | trmv | f64 | n=4096 | 1.707 ms | 9.83 | 39.34 | 73% | ok |
| blas2 | trmv | f32 | n=4096 | 1.478 ms | 11.35 | 22.72 | 42% | ok |
| blas2 | trsv | f64 | n=4096 | 1.674 ms | 10.02 | 40.14 | 75% | ok |
| blas2 | trsv | f32 | n=4096 | 1.534 ms | 10.94 | 21.90 | 41% | ok |
| blas2 | ger | f64 | n=4096 | 13.453 ms | 2.49 | 19.96 | 37% | ok |
| blas3 | gemm | f64 | 512x512x512 | 16.033 ms | 16.74 | 0.39 | 21% | ok |
| blas3 | gemm | f32 | 512x512x512 | 8.312 ms | 32.29 | 0.38 | 20% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 11.312 ms | 11.87 | 0.56 | 15% | ok |
| blas3 | symm | f64 | 512x512x512 | 17.083 ms | 15.71 | 0.37 | 20% | ok |
| blas3 | syrk | f64 | 512x512x512 | 13.546 ms | 9.91 | 0.31 | 12% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 22.700 ms | 11.83 | 0.28 | 15% | ok |
| blas3 | trmm | f64 | 512x512x512 | 25.178 ms | 5.33 | 0.21 | 7% | ok |
| blas3 | trsm | f64 | 512x512x512 | 26.806 ms | 5.01 | 0.20 | 6% | ok |
| blas3 | trsm | f32 | 512x512x512 | 26.133 ms | 5.14 | 0.10 | 3% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.173 ms | 0.58 | 4.63 | 9% | ok |
| factor | lu | f64 | 512x512 | 32.862 ms | 2.72 | 0.13 | 3% | ok |
| factor | lu | f32 | 512x512 | 31.065 ms | 2.88 | 0.07 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 11.362 ms | 3.94 | 0.37 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 10.913 ms | 4.10 | 0.19 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 11.426 ms | 3.92 | 0.37 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 10.238 ms | 4.37 | 0.41 | 5% | ok |
| factor | qr | f64 | 512x512 | 644.379 ms | 0.28 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 65.160 ms | 2.75 | 0.06 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 739.871 ms | 0.60 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 3.874 ms | 0.27 | 1.08 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 89.033 ms | 3.02 | 0.05 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 34.149 ms | 2.62 | 0.12 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 37.590 ms | 0.60 | 0.03 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 127.019 ms | 1.19 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 514.592 ms | 0.29 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 58.890 ms | 2.85 | 0.02 | 4% | ok |
| eigen | schur | f64 | 256x256 | 111.962 ms | 3.75 | 0.01 | 5% | ok |
| eigen | svd | f64 | 256x256 | 118.708 ms | 3.11 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 29.974 ms | 8.40 | 0.03 | 10% | ok |
| eigen | sylvester | f64 | 256x256 | 268.831 ms | 3.74 | 0.01 | 5% | ok |
| elementwise | exp | f64 | n=8388608 | 28.180 ms | 0.30 | 4.76 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 42.146 ms | 0.20 | 3.18 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 39.896 ms | 0.21 | 3.36 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 21.328 ms | 0.39 | 6.29 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 31.505 ms | 0.27 | 4.26 | 8% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.858 ms | 3.05 | 31.74 | 59% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.331 ms | 1.97 | 20.46 | 38% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.520 ms | 6.42 | 13.50 | 25% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 18.345 ms | 0.18 | 0.63 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.719 ms | 1.17 | 9.68 | 18%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 19.326 ms | 0.58 | 0.04 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 30.941 ms | 0.36 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 16.046 ms | 0.70 | 0.05 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 53.905 ms | 0.42 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 57.847 ms | 0.39 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 24.546 ms | 0.91 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 38.269 ms | 0.05 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.635 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.624 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 60.623 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.224 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 13.981 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.914 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.238 ms | 0.79 | 0.05 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.275 ms | 2.75 | 11.01 | 21%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.569 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 5.958 ms | 1.88 | 0.12 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.277 ms | 2.73 | 10.93 | 20%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.336 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.768 ms | 0.47 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.095 ms | 0.69 | 2.76 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.785 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 49.257 ms | 0.45 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.104 ms | 0.68 | 2.74 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.545 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 255.358 ms | 1.21 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.984 ms | 1.70 | 6.79 | 13%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 13.939 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.181 ms | 4.46 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.020 ms | 1.67 | 6.67 | 12%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.159 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 890.142 ms | 0.69 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.525 ms | 0.32 | 1.28 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.560 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 495.771 ms | 1.24 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.509 ms | 0.32 | 1.28 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 40.668 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 60.928 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 32.252 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.488 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.493 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 18.016 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 58.609 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 35.256 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 20.241 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 27.279 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 25.842 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 27.017 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 35.881 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.044 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.462 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 21.871 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.803 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.758 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.029 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.655 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.512 ms | 1.73 | 19.40 | 36% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.500 ms | 1.74 | 19.55 | 36% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.791 ms | 1.46 | 16.38 | 31% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.662 ms | 2.45 | 18.56 | 35% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.382 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.864 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 53.332 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.020 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 11.438 ms | 1.47 | 5.87 | 11% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 29.055 ms | 9.24 | 0.14 | 12% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.475 ms | 11.38 | 12.09 | 23% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.197 ms | 5.25 | 20.99 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.881 ms | 5.82 | 11.65 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.395 ms | 12.03 | 12.03 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.396 ms | 12.02 | 12.02 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 8.207 ms | 32.71 | 0.38 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 30.290 ms | 8.86 | 0.07 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 27.361 ms | 9.81 | 0.06 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 27.310 ms | 9.83 | 0.06 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 27.992 ms | 9.59 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## ryzen-7-8845hs -- native, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 627.5 GFLOP/s, f32 1254.8 GFLOP/s |
| ... hardware, AVX-512F | f64 627.5, f32 1254.8 GFLOP/s |
| ... this build, scalar | f64 12.3, f32 14.3 GFLOP/s |
| bandwidth | read 53.1, update 53.3, triad 41.3 GB/s (ceiling 53.3) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 94% | 128% | 0 | 43% |
| blas2 | 10 | 74% | 110% | 0 | 13% |
| blas3 | 10 | 2% | 18% | 0 | 12% |
| factor | 12 | 1% | 8% | 0 | 18% |
| eigen | 8 | 0% | 2% | 0 | 12% |
| elementwise | 5 | 8% | 12% | 0 | 12% |
| sparse | 5 | 37% | 298% | 0 | 13% |
| direct | 19 | 0% | 21% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 12% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 35% | 36% | 0 | 12% |
| mg | 1 | 36% | 36% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 11% | 22% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 88% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.273 ms | 7.38 | 59.04 | 111% | ok | 5.1x (63%) |
| blas1 | dot | f32 | n=8388608 | 1.336 ms | 12.56 | 50.22 | 94% | ok | 7.4x (92%) |
| blas1 | axpy | f64 | n=8388608 | 3.520 ms | 4.77 | 57.19 | 107% | ok | 3.4x (43%) |
| blas1 | axpy | f32 | n=8388608 | 1.750 ms | 9.59 | 57.54 | 108% | ok | 6.8x (86%) |
| blas1 | scale | f64 | n=8388608 | 1.970 ms | 4.26 | 68.13 | 128% | ok | 1.2x (15%) |
| blas1 | two_norm | f64 | n=8388608 | 1.497 ms | 11.20 | 44.82 | 84% | ok | 7.4x (92%) |
| blas1 | one_norm | f64 | n=8388608 | 5.061 ms | 1.66 | 13.26 | 25% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.100 ms | 1.64 | 13.16 | 25% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 5.062 ms | 1.66 | 13.26 | 25% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 84.509 ms | 2.18 | 1.59 | 3% | ok | 1.1x (13%) |
| blas2 | gemv | f64 | n=4096 | 2.377 ms | 14.11 | 56.48 | 106% | ok | 1.3x (16%) |
| blas2 | gemv | f32 | n=4096 | 1.184 ms | 28.35 | 56.73 | 107% | ok | 1.9x (24%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.285 ms | 14.68 | 58.76 | 110% | ok | 3.1x (39%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.150 ms | 29.18 | 58.39 | 110% | ok | 2.0x (25%) |
| blas2 | symv | f64 | n=4096 | 2.924 ms | 11.48 | 22.98 | 43% | ok | 1.0x (12%) |
| blas2 | trmv | f64 | n=4096 | 1.712 ms | 9.80 | 39.25 | 74% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=4096 | 1.482 ms | 11.32 | 22.66 | 43% | ok | 1.0x (12%) |
| blas2 | trsv | f64 | n=4096 | 1.753 ms | 9.57 | 38.32 | 72% | ok | 1.0x (12%) |
| blas2 | trsv | f32 | n=4096 | 1.520 ms | 11.04 | 22.09 | 41% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 13.675 ms | 2.45 | 19.63 | 37% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.455 ms | 109.34 | 2.56 | 17% | ok | 6.5x (82%) |
| blas3 | gemm | f32 | 512x512x512 | 1.184 ms | 226.78 | 2.66 | 18% | ok | 7.0x (88%) |
| blas3 | gemmt | f64 | 512x512x512 | 11.581 ms | 11.59 | 0.54 | 2% | ok | 1.0x (12%) |
| blas3 | symm | f64 | 512x512x512 | 17.277 ms | 15.54 | 0.36 | 2% | ok | 1.0x (12%) |
| blas3 | syrk | f64 | 512x512x512 | 13.717 ms | 9.78 | 0.31 | 2% | ok | 1.0x (12%) |
| blas3 | syr2k | f64 | 512x512x512 | 23.113 ms | 11.61 | 0.27 | 2% | ok | 1.0x (12%) |
| blas3 | trmm | f64 | 512x512x512 | 25.359 ms | 5.29 | 0.21 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f64 | 512x512x512 | 26.965 ms | 4.98 | 0.19 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f32 | 512x512x512 | 26.296 ms | 5.10 | 0.10 | 0% | ok | 1.0x (12%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.446 ms | 0.56 | 4.50 | 8% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 22.345 ms | 4.00 | 0.19 | 1% | ok | 1.5x (18%) |
| factor | lu | f32 | 512x512 | 21.193 ms | 4.22 | 0.10 | 0% | ok | 1.5x (18%) |
| factor | cholesky | f64 | 512x512 | 11.369 ms | 3.94 | 0.37 | 1%\* | ok | 1.0x (12%) |
| factor | cholesky | f32 | 512x512 | 10.890 ms | 4.11 | 0.19 | 0%\* | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 11.424 ms | 3.92 | 0.37 | 1%\* | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 10.383 ms | 4.31 | 0.40 | 1%\* | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 396.484 ms | 0.45 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 41.870 ms | 4.27 | 0.10 | 1% | ok | 1.6x (19%) |
| factor | hessenberg | f64 | 512x512 | 457.529 ms | 0.98 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.012 ms | 1.04 | 4.14 | 8%\* | ok | 3.8x (48%) |
| factor | inv | f64 | 512x512 | 78.307 ms | 3.43 | 0.05 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 23.827 ms | 3.76 | 0.18 | 1% | ok | 1.4x (18%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 38.797 ms | 0.58 | 0.03 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 129.465 ms | 1.17 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 520.204 ms | 0.29 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 256x256 | 61.755 ms | 2.72 | 0.02 | 0% | ok | 1.0x (12%) |
| eigen | schur | f64 | 256x256 | 116.456 ms | 3.60 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 119.498 ms | 3.09 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | expm | f64 | 256x256 | 18.604 ms | 13.53 | 0.06 | 2% | ok | 1.6x (20%) |
| eigen | sylvester | f64 | 256x256 | 262.427 ms | 3.84 | 0.01 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 28.050 ms | 0.30 | 4.79 | 9% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 42.386 ms | 0.20 | 3.17 | 6% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 40.267 ms | 0.21 | 3.33 | 6% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 21.457 ms | 0.39 | 6.26 | 12% | ok | 1.0x (12%) |
| elementwise | erf | f64 | n=8388608 | 31.690 ms | 0.26 | 4.24 | 8% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.171 ms | 15.26 | 158.78 | 298% ! | ok | 5.0x (63%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.398 ms | 1.87 | 19.48 | 37% | ok | 1.0x (12%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.235 ms | 18.74 | 39.39 | 74% | ok | 2.9x (36%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 18.562 ms | 0.18 | 0.62 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.660 ms | 1.21 | 10.03 | 19%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 19.765 ms | 0.57 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 31.585 ms | 0.35 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 15.435 ms | 0.72 | 0.05 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 54.244 ms | 0.41 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 58.308 ms | 0.38 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 25.623 ms | 0.87 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 38.663 ms | 0.05 | 0.00 | 0%\* | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.739 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.721 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 61.729 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.227 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.309 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.986 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.245 ms | 0.79 | 0.05 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.281 ms | 2.69 | 10.77 | 20%\* | ok | 1.0x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.706 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 6.032 ms | 1.85 | 0.12 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.276 ms | 2.74 | 10.94 | 21%\* | ok | 1.0x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.376 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.518 ms | 0.46 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.137 ms | 0.67 | 2.66 | 5%\* | ok | 1.0x (12%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.852 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 49.140 ms | 0.46 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.102 ms | 0.69 | 2.74 | 5%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.639 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 257.469 ms | 1.20 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.025 ms | 1.66 | 6.66 | 12%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 14.020 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.556 ms | 4.44 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.065 ms | 1.63 | 6.53 | 12%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.206 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 896.055 ms | 0.69 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.576 ms | 0.32 | 1.27 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.609 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 499.133 ms | 1.24 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.535 ms | 0.32 | 1.28 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 40.683 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 61.271 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 32.392 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.356 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.497 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 18.015 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 58.760 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 35.196 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 20.218 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 27.320 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 25.934 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 27.164 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 36.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.044 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.519 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 25.270 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| pc | ic_0 | f64 | 128^2 grid | 1.807 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 128^2 grid | 1.873 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.030 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.659 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.519 ms | 1.72 | 19.31 | 36% | ok | 1.0x (12%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.560 ms | 1.68 | 18.79 | 35% | ok | 1.0x (12%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.183 ms | 1.20 | 13.43 | 25% | ok | 0.8x (10%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.967 ms | 2.57 | 19.42 | 36% | ok | 1.0x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.652 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 33.057 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 53.580 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.018 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 11.437 ms | 1.47 | 5.87 | 11% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.021 ms | 66.75 | 1.04 | 11% | ok | 7.2x (90%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.497 ms | 11.21 | 11.91 | 22% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.669 ms | 4.57 | 18.29 | n/a | ok | 0.9x (11%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.895 ms | 5.80 | 11.59 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.409 ms | 11.91 | 11.91 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.408 ms | 11.91 | 11.91 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.164 ms | 230.67 | 2.70 | n/a | ok | 7.1x (88%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.159 ms | 64.54 | 0.50 | n/a | ok | 7.3x (91%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 3.770 ms | 71.20 | 0.42 | n/a | ok | 7.3x (91%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.778 ms | 71.06 | 0.42 | n/a | ok | 7.2x (90%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.797 ms | 70.69 | 0.41 | n/a | ok | 7.4x (92%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

## ryzen-7-8845hs -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 80.1 GFLOP/s, f32 160.1 GFLOP/s |
| ... hardware, AVX-512F | f64 80.1, f32 160.1 GFLOP/s |
| ... this build, SSE2 | f64 40.0, f32 80.1 GFLOP/s |
| bandwidth | read 47.0, update 53.4, triad 39.1 GB/s (ceiling 53.4) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 108% | 131% | 0 |
| blas2 | 10 | 67% | 77% | 0 |
| blas3 | 10 | 22% | 30% | 0 |
| factor | 12 | 3% | 5% | 0 |
| eigen | 8 | 4% | 12% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 18% | 59% | 0 |
| direct | 19 | 1% | 20% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 13% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 36% | 38% | 0 |
| mg | 1 | 34% | 34% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 70% | 70% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 2.354 ms | 7.13 | 57.00 | 107% | ok |
| blas1 | dot | f32 | n=8388608 | 1.168 ms | 14.37 | 57.48 | 108% | ok |
| blas1 | axpy | f64 | n=8388608 | 3.428 ms | 4.89 | 58.72 | 110% | ok |
| blas1 | axpy | f32 | n=8388608 | 1.654 ms | 10.14 | 60.85 | 114% | ok |
| blas1 | scale | f64 | n=8388608 | 1.927 ms | 4.35 | 69.65 | 131% | ok |
| blas1 | two_norm | f64 | n=8388608 | 1.164 ms | 14.41 | 57.65 | 108% | ok |
| blas1 | one_norm | f64 | n=8388608 | 5.062 ms | 1.66 | 13.26 | 25% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 5.072 ms | 1.65 | 13.23 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 5.061 ms | 1.66 | 13.26 | 25% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 89.277 ms | 2.07 | 1.50 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.274 ms | 10.25 | 41.02 | 77% | ok |
| blas2 | gemv | f32 | n=4096 | 1.753 ms | 19.14 | 38.30 | 72% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.815 ms | 8.80 | 35.20 | 66% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.244 ms | 14.96 | 29.93 | 56% | ok |
| blas2 | symv | f64 | n=4096 | 2.821 ms | 11.89 | 23.81 | 45% | ok |
| blas2 | trmv | f64 | n=4096 | 1.756 ms | 9.56 | 38.26 | 72% | ok |
| blas2 | trmv | f32 | n=4096 | 0.925 ms | 18.14 | 36.31 | 68% | ok |
| blas2 | trsv | f64 | n=4096 | 1.885 ms | 8.90 | 35.63 | 67% | ok |
| blas2 | trsv | f32 | n=4096 | 0.942 ms | 17.82 | 35.67 | 67% | ok |
| blas2 | ger | f64 | n=4096 | 13.559 ms | 2.47 | 19.80 | 37% | ok |
| blas3 | gemm | f64 | 512x512x512 | 11.033 ms | 24.33 | 0.57 | 30% | ok |
| blas3 | gemm | f32 | 512x512x512 | 5.569 ms | 48.20 | 0.56 | 30% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 7.457 ms | 18.00 | 0.84 | 22% | ok |
| blas3 | symm | f64 | 512x512x512 | 12.012 ms | 22.35 | 0.52 | 28% | ok |
| blas3 | syrk | f64 | 512x512x512 | 7.464 ms | 17.98 | 0.56 | 22% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 14.294 ms | 18.78 | 0.44 | 23% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.741 ms | 13.78 | 0.54 | 17% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.831 ms | 13.65 | 0.53 | 17% | ok |
| blas3 | trsm | f32 | 512x512x512 | 6.481 ms | 20.71 | 0.40 | 13% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.473 ms | 0.56 | 4.48 | 8% | ok |
| factor | lu | f64 | 512x512 | 32.782 ms | 2.73 | 0.13 | 3% | ok |
| factor | lu | f32 | 512x512 | 30.971 ms | 2.89 | 0.07 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 11.398 ms | 3.93 | 0.37 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 10.894 ms | 4.11 | 0.19 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 11.454 ms | 3.91 | 0.37 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 10.305 ms | 4.34 | 0.41 | 5% | ok |
| factor | qr | f64 | 512x512 | 645.855 ms | 0.28 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 64.558 ms | 2.77 | 0.06 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 738.848 ms | 0.61 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 3.900 ms | 0.27 | 1.08 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 87.594 ms | 3.06 | 0.05 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 33.910 ms | 2.64 | 0.12 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 37.386 ms | 0.60 | 0.03 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 127.662 ms | 1.18 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 517.786 ms | 0.29 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 56.359 ms | 2.98 | 0.02 | 4% | ok |
| eigen | schur | f64 | 256x256 | 111.693 ms | 3.76 | 0.01 | 5% | ok |
| eigen | svd | f64 | 256x256 | 119.351 ms | 3.09 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 25.746 ms | 9.77 | 0.04 | 12% | ok |
| eigen | sylvester | f64 | 256x256 | 265.954 ms | 3.78 | 0.01 | 5% | ok |
| elementwise | exp | f64 | n=8388608 | 27.781 ms | 0.30 | 4.83 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 42.142 ms | 0.20 | 3.18 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 39.545 ms | 0.21 | 3.39 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 21.298 ms | 0.39 | 6.30 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 31.517 ms | 0.27 | 4.26 | 8% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.871 ms | 3.00 | 31.26 | 59% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 5.182 ms | 0.51 | 5.25 | 10% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.523 ms | 6.42 | 13.50 | 25% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 17.666 ms | 0.18 | 0.65 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.715 ms | 1.17 | 9.71 | 18%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 18.979 ms | 0.59 | 0.04 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 28.299 ms | 0.40 | 0.03 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 14.356 ms | 0.78 | 0.05 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 53.492 ms | 0.42 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 57.969 ms | 0.39 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 25.317 ms | 0.88 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 38.815 ms | 0.05 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.806 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.767 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 61.395 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.227 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 13.987 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.126 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.437 ms | 0.77 | 0.05 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.279 ms | 2.71 | 10.85 | 20%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.553 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 5.973 ms | 1.87 | 0.12 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.289 ms | 2.61 | 10.45 | 20%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.527 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.581 ms | 0.47 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.092 ms | 0.69 | 2.77 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.187 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 49.007 ms | 0.46 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.181 ms | 0.64 | 2.56 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.820 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 266.464 ms | 1.16 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.964 ms | 1.72 | 6.86 | 13%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 14.206 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.295 ms | 4.45 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.037 ms | 1.65 | 6.62 | 12%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.320 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 913.911 ms | 0.68 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.498 ms | 0.32 | 1.28 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.891 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 507.067 ms | 1.22 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.527 ms | 0.32 | 1.28 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 32.632 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 45.648 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 28.429 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 13.476 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 17.921 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 15.703 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 34.461 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 29.220 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 17.290 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 18.609 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 17.484 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.610 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.621 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.044 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.692 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 25.050 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.147 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.914 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.030 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.744 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.447 ms | 1.81 | 20.27 | 38% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.538 ms | 1.70 | 19.07 | 36% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.769 ms | 1.48 | 16.58 | 31% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 16.199 ms | 2.37 | 17.94 | 34% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.428 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.904 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 54.488 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 6.578 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 1.788 ms | 9.38 | 37.53 | 70% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 15.226 ms | 17.63 | 0.28 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.474 ms | 35.39 | 37.61 | 70% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.244 ms | 13.48 | 53.92 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.475 ms | 35.35 | 70.70 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.349 ms | 48.02 | 48.02 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.317 ms | 52.94 | 52.94 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 14.988 ms | 17.91 | 0.21 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 17.340 ms | 15.48 | 0.12 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 17.797 ms | 15.08 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.713 ms | 72.29 | 0.42 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.696 ms | 72.64 | 0.43 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## ryzen-7-8845hs -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 625.9 GFLOP/s, f32 1248.4 GFLOP/s |
| ... hardware, AVX-512F | f64 625.9, f32 1248.4 GFLOP/s |
| ... this build, SSE2 | f64 312.6, f32 624.5 GFLOP/s |
| bandwidth | read 53.2, update 53.6, triad 41.2 GB/s (ceiling 53.6) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 112% | 129% | 0 | 13% |
| blas2 | 10 | 70% | 112% | 0 | 13% |
| blas3 | 10 | 9% | 18% | 0 | 38% |
| factor | 12 | 1% | 8% | 0 | 18% |
| eigen | 8 | 0% | 2% | 0 | 13% |
| elementwise | 5 | 8% | 12% | 0 | 12% |
| sparse | 5 | 39% | 289% | 0 | 36% |
| direct | 19 | 0% | 20% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 13% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 14% |
| smoother | 3 | 34% | 36% | 0 | 12% |
| mg | 1 | 36% | 36% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 69% | 70% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 70% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.235 ms | 7.51 | 60.04 | 112% | ok | 1.1x (13%) |
| blas1 | dot | f32 | n=8388608 | 0.973 ms | 17.25 | 69.00 | 129% | ok | 1.2x (15%) |
| blas1 | axpy | f64 | n=8388608 | 3.481 ms | 4.82 | 57.83 | 108% | ok | 1.0x (12%) |
| blas1 | axpy | f32 | n=8388608 | 1.621 ms | 10.35 | 62.09 | 116% | ok | 1.0x (13%) |
| blas1 | scale | f64 | n=8388608 | 1.964 ms | 4.27 | 68.33 | 127% | ok | 1.0x (12%) |
| blas1 | two_norm | f64 | n=8388608 | 1.082 ms | 15.51 | 62.03 | 116% | ok | 1.1x (13%) |
| blas1 | one_norm | f64 | n=8388608 | 5.064 ms | 1.66 | 13.25 | 25% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.098 ms | 1.65 | 13.16 | 25% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 5.061 ms | 1.66 | 13.26 | 25% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 89.277 ms | 2.07 | 1.50 | 3% | ok | 1.0x (12%) |
| blas2 | gemv | f64 | n=4096 | 2.289 ms | 14.66 | 58.67 | 109% | ok | 1.4x (18%) |
| blas2 | gemv | f32 | n=4096 | 1.121 ms | 29.95 | 59.92 | 112% | ok | 1.6x (20%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.285 ms | 14.69 | 58.77 | 110% | ok | 1.7x (21%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.183 ms | 28.36 | 56.76 | 106% | ok | 1.9x (24%) |
| blas2 | symv | f64 | n=4096 | 2.951 ms | 11.37 | 22.77 | 42% | ok | 1.0x (12%) |
| blas2 | trmv | f64 | n=4096 | 1.889 ms | 8.88 | 35.56 | 66% | ok | 0.9x (12%) |
| blas2 | trmv | f32 | n=4096 | 0.945 ms | 17.76 | 35.55 | 66% | ok | 1.0x (12%) |
| blas2 | trsv | f64 | n=4096 | 1.891 ms | 8.87 | 35.53 | 66% | ok | 1.0x (12%) |
| blas2 | trsv | f32 | n=4096 | 0.894 ms | 18.76 | 37.57 | 70% | ok | 1.1x (13%) |
| blas2 | ger | f64 | n=4096 | 13.580 ms | 2.47 | 19.77 | 37% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.337 ms | 114.84 | 2.69 | 18% | ok | 4.7x (59%) |
| blas3 | gemm | f32 | 512x512x512 | 1.231 ms | 218.03 | 2.56 | 17% | ok | 4.5x (57%) |
| blas3 | gemmt | f64 | 512x512x512 | 2.799 ms | 47.96 | 2.25 | 8% | ok | 2.7x (33%) |
| blas3 | symm | f64 | 512x512x512 | 3.063 ms | 87.63 | 2.05 | 14% | ok | 3.9x (49%) |
| blas3 | syrk | f64 | 512x512x512 | 2.427 ms | 55.31 | 1.73 | 9% | ok | 3.1x (38%) |
| blas3 | syr2k | f64 | 512x512x512 | 4.638 ms | 57.87 | 1.36 | 9% | ok | 3.1x (39%) |
| blas3 | trmm | f64 | 512x512x512 | 5.533 ms | 24.26 | 0.95 | 4% | ok | 1.8x (22%) |
| blas3 | trsm | f64 | 512x512x512 | 5.750 ms | 23.34 | 0.91 | 4% | ok | 1.7x (21%) |
| blas3 | trsm | f32 | 512x512x512 | 4.447 ms | 30.18 | 0.59 | 2% | ok | 1.5x (18%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.391 ms | 0.57 | 4.52 | 8% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 22.152 ms | 4.04 | 0.19 | 1% | ok | 1.5x (18%) |
| factor | lu | f32 | 512x512 | 21.033 ms | 4.25 | 0.10 | 0% | ok | 1.5x (18%) |
| factor | cholesky | f64 | 512x512 | 11.418 ms | 3.92 | 0.37 | 1%\* | ok | 1.0x (12%) |
| factor | cholesky | f32 | 512x512 | 10.925 ms | 4.09 | 0.19 | 0%\* | ok | 1.0x (12%) |
| factor | ldlt | f64 | 512x512 | 11.483 ms | 3.90 | 0.37 | 1%\* | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 10.565 ms | 4.23 | 0.40 | 1%\* | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 398.693 ms | 0.45 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 41.964 ms | 4.26 | 0.10 | 1% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 458.252 ms | 0.98 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.021 ms | 1.03 | 4.11 | 8%\* | ok | 3.8x (48%) |
| factor | inv | f64 | 512x512 | 78.354 ms | 3.43 | 0.05 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 23.696 ms | 3.78 | 0.18 | 1% | ok | 1.4x (18%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 38.166 ms | 0.59 | 0.03 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 128.378 ms | 1.18 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 510.010 ms | 0.30 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 57.047 ms | 2.94 | 0.02 | 0% | ok | 1.0x (12%) |
| eigen | schur | f64 | 256x256 | 109.391 ms | 3.83 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 119.360 ms | 3.09 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | expm | f64 | 256x256 | 18.057 ms | 13.94 | 0.06 | 2% | ok | 1.4x (18%) |
| eigen | sylvester | f64 | 256x256 | 259.216 ms | 3.88 | 0.01 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 28.085 ms | 0.30 | 4.78 | 9% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 42.629 ms | 0.20 | 3.15 | 6% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 40.250 ms | 0.21 | 3.33 | 6% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 21.360 ms | 0.39 | 6.28 | 12% | ok | 1.0x (12%) |
| elementwise | erf | f64 | n=8388608 | 31.820 ms | 0.26 | 4.22 | 8% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.176 ms | 14.90 | 154.98 | 289% ! | ok | 5.0x (62%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.319 ms | 1.99 | 20.65 | 39% | ok | 3.9x (49%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.248 ms | 18.63 | 39.17 | 73% | ok | 2.9x (36%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 18.214 ms | 0.18 | 0.63 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.659 ms | 1.21 | 10.03 | 19%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 18.384 ms | 0.61 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 28.164 ms | 0.40 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 13.281 ms | 0.84 | 0.06 | 0% | ok | 1.1x (14%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 54.221 ms | 0.41 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 57.548 ms | 0.39 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 25.145 ms | 0.89 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 38.311 ms | 0.05 | 0.00 | 0%\* | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.745 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.683 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 61.983 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.227 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.366 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.004 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.531 ms | 0.77 | 0.05 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.286 ms | 2.64 | 10.56 | 20%\* | ok | 1.0x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.690 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 6.031 ms | 1.85 | 0.12 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.282 ms | 2.69 | 10.74 | 20%\* | ok | 1.0x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.415 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.002 ms | 0.47 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.106 ms | 0.68 | 2.74 | 5%\* | ok | 1.0x (12%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.836 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 49.264 ms | 0.45 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.137 ms | 0.66 | 2.66 | 5%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.636 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 260.486 ms | 1.18 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.001 ms | 1.68 | 6.73 | 13%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 14.141 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.599 ms | 4.43 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.070 ms | 1.63 | 6.51 | 12%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.187 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 923.971 ms | 0.67 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.530 ms | 0.32 | 1.28 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.564 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 503.477 ms | 1.23 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.499 ms | 0.32 | 1.28 | 2%\* | ok | 1.0x (13%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 32.629 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 46.085 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 28.534 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 13.499 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 17.958 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 15.715 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 34.541 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 29.290 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 17.336 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 18.611 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 17.552 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.688 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.555 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.043 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.457 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | ilut | f64 | 128^2 grid | 22.789 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | ic_0 | f64 | 128^2 grid | 1.800 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| pc | ildl | f64 | 128^2 grid | 1.755 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | ssor | f64 | 128^2 grid | 0.030 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.672 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.620 ms | 1.62 | 18.10 | 34% | ok | 0.9x (11%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.521 ms | 1.72 | 19.29 | 36% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.794 ms | 1.46 | 16.34 | 30% | ok | 1.0x (12%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.266 ms | 2.52 | 19.04 | 36% | ok | 1.1x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.617 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 33.062 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 54.904 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 6.549 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 1.784 ms | 9.40 | 37.61 | 70% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 2.627 ms | 102.17 | 1.60 | 16% | ok | 5.8x (72%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.480 ms | 34.92 | 37.11 | 69% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.213 ms | 13.83 | 55.33 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.476 ms | 35.28 | 70.55 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.349 ms | 48.00 | 48.00 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.318 ms | 52.69 | 52.69 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.421 ms | 110.88 | 1.30 | n/a | ok | 6.2x (77%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.526 ms | 106.26 | 0.83 | n/a | ok | 6.9x (86%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 2.588 ms | 103.71 | 0.61 | n/a | ok | 6.9x (86%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.665 ms | 403.60 | 2.36 | n/a | ok | 5.6x (70%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.664 ms | 404.03 | 2.37 | n/a | ok | 5.6x (70%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

## ryzen-7-8845hs -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 80.1 GFLOP/s, f32 160.1 GFLOP/s |
| ... hardware, AVX-512F | f64 80.1, f32 160.1 GFLOP/s |
| ... this build, scalar | f64 1.9, f32 1.9 GFLOP/s |
| bandwidth | read 43.1, update 53.2, triad 38.8 GB/s (ceiling 53.2) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 25% | 110% | 0 |
| blas2 | 10 | 78% | 90% | 0 |
| blas3 | 10 | 70% | 92% | 0 |
| factor | 12 | 13% | 23% | 0 |
| eigen | 8 | 17% | 53% | 0 |
| elementwise | 5 | 7% | 12% | 0 |
| sparse | 5 | 26% | 60% | 0 |
| direct | 19 | 1% | 20% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 13% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 32% | 34% | 0 |
| mg | 1 | 30% | 30% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 13% | 23% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 9.138 ms | 1.84 | 14.69 | 28% | ok |
| blas1 | dot | f32 | n=8388608 | 9.852 ms | 1.70 | 6.81 | 13% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.029 ms | 1.39 | 16.74 | 31% | ok |
| blas1 | axpy | f32 | n=8388608 | 12.009 ms | 1.40 | 8.38 | 16% | ok |
| blas1 | scale | f64 | n=8388608 | 2.292 ms | 3.66 | 58.55 | 110% | ok |
| blas1 | two_norm | f64 | n=8388608 | 9.838 ms | 1.71 | 6.82 | 13% | ok |
| blas1 | one_norm | f64 | n=8388608 | 5.063 ms | 1.66 | 13.25 | 25% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 5.071 ms | 1.65 | 13.23 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 5.061 ms | 1.66 | 13.26 | 25% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 83.257 ms | 2.22 | 1.61 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.221 ms | 10.42 | 41.68 | 78% | ok |
| blas2 | gemv | f32 | n=4096 | 1.409 ms | 23.82 | 47.66 | 90% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.367 ms | 9.97 | 39.89 | 75% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.496 ms | 22.43 | 44.87 | 84% | ok |
| blas2 | symv | f64 | n=4096 | 1.818 ms | 18.45 | 36.95 | 69% | ok |
| blas2 | trmv | f64 | n=4096 | 1.592 ms | 10.54 | 42.21 | 79% | ok |
| blas2 | trmv | f32 | n=4096 | 0.884 ms | 18.99 | 38.01 | 71% | ok |
| blas2 | trsv | f64 | n=4096 | 1.717 ms | 9.77 | 39.12 | 74% | ok |
| blas2 | trsv | f32 | n=4096 | 0.866 ms | 19.38 | 38.79 | 73% | ok |
| blas2 | ger | f64 | n=4096 | 6.365 ms | 5.27 | 42.19 | 79% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.646 ms | 73.62 | 1.73 | 92% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.899 ms | 141.36 | 1.66 | 88% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.737 ms | 49.03 | 2.30 | 61% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.795 ms | 70.73 | 1.66 | 88% | ok |
| blas3 | syrk | f64 | 512x512x512 | 2.934 ms | 45.75 | 1.43 | 57% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 4.756 ms | 56.44 | 1.32 | 70% | ok |
| blas3 | trmm | f64 | 512x512x512 | 1.897 ms | 70.74 | 2.76 | 88% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.847 ms | 47.15 | 1.84 | 59% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.347 ms | 99.66 | 1.95 | 62% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.382 ms | 0.57 | 4.53 | 9% | ok |
| factor | lu | f64 | 512x512 | 4.977 ms | 17.98 | 0.84 | 22% | ok |
| factor | lu | f32 | 512x512 | 2.436 ms | 36.72 | 0.86 | 23% | ok |
| factor | cholesky | f64 | 512x512 | 4.346 ms | 10.29 | 0.97 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 1.676 ms | 26.70 | 1.25 | 17% | ok |
| factor | ldlt | f64 | 512x512 | 11.748 ms | 3.81 | 0.36 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 10.441 ms | 4.28 | 0.40 | 5% | ok |
| factor | qr | f64 | 512x512 | 10.185 ms | 17.57 | 0.41 | 22% | ok |
| factor | lq | f64 | 512x512 | 64.665 ms | 2.77 | 0.06 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 737.984 ms | 0.61 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 3.912 ms | 0.27 | 1.07 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 60.173 ms | 4.46 | 0.07 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 5.532 ms | 16.17 | 0.76 | 20% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.696 ms | 13.19 | 0.62 | 16% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 129.234 ms | 1.17 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 525.892 ms | 0.29 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 16.932 ms | 9.91 | 0.06 | 12% | ok |
| eigen | schur | f64 | 256x256 | 22.235 ms | 18.86 | 0.05 | 24% | ok |
| eigen | svd | f64 | 256x256 | 8.719 ms | 42.33 | 0.12 | 53% | ok |
| eigen | expm | f64 | 256x256 | 15.714 ms | 16.02 | 0.07 | 20% | ok |
| eigen | sylvester | f64 | 256x256 | 72.498 ms | 13.89 | 0.02 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 27.797 ms | 0.30 | 4.83 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 43.232 ms | 0.19 | 3.10 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 39.164 ms | 0.21 | 3.43 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 21.388 ms | 0.39 | 6.28 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 34.552 ms | 0.24 | 3.88 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.852 ms | 3.07 | 31.98 | 60% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.481 ms | 1.77 | 18.39 | 35% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.434 ms | 6.51 | 13.68 | 26% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 17.923 ms | 0.18 | 0.64 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.701 ms | 1.18 | 9.79 | 18%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 19.401 ms | 0.58 | 0.04 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 31.080 ms | 0.36 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 15.543 ms | 0.72 | 0.05 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 54.208 ms | 0.41 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 57.425 ms | 0.39 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 26.356 ms | 0.85 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 37.657 ms | 0.05 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.707 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.673 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 60.210 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.227 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.347 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.950 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.287 ms | 0.78 | 0.05 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.278 ms | 2.72 | 10.90 | 20%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.602 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 6.017 ms | 1.86 | 0.12 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.291 ms | 2.59 | 10.38 | 20%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.391 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.512 ms | 0.47 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.113 ms | 0.68 | 2.72 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.854 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 49.289 ms | 0.45 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.066 ms | 0.71 | 2.84 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.593 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 255.657 ms | 1.21 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.984 ms | 1.70 | 6.79 | 13%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 13.819 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.567 ms | 4.44 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.024 ms | 1.66 | 6.66 | 13%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.185 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 899.491 ms | 0.69 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.408 ms | 0.32 | 1.29 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.534 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 517.755 ms | 1.19 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.468 ms | 0.32 | 1.29 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 38.966 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 57.677 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 31.662 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 16.588 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 21.523 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.578 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 53.064 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.148 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 19.689 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 25.912 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 24.460 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 25.308 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 32.904 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.044 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.462 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 22.322 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.811 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.773 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.030 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.337 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.612 ms | 1.62 | 18.20 | 34% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.735 ms | 1.51 | 16.90 | 32% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.869 ms | 0.91 | 10.22 | 19% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.448 ms | 2.08 | 15.76 | 30% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.482 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.999 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 52.952 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.811 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 9.796 ms | 1.71 | 6.85 | 13% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 29.502 ms | 9.10 | 0.14 | 11% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.482 ms | 11.32 | 12.03 | 23% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.201 ms | 5.24 | 20.96 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.865 ms | 5.86 | 11.71 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.389 ms | 12.08 | 12.08 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.392 ms | 12.05 | 12.05 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 8.387 ms | 32.00 | 0.38 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 30.396 ms | 8.83 | 0.07 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 27.390 ms | 9.80 | 0.06 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 27.400 ms | 9.80 | 0.06 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 27.371 ms | 9.81 | 0.06 | n/a | ok |

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
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 625.8 GFLOP/s, f32 1249.7 GFLOP/s |
| ... hardware, AVX-512F | f64 625.8, f32 1249.7 GFLOP/s |
| ... this build, scalar | f64 14.4, f32 14.4 GFLOP/s |
| bandwidth | read 53.1, update 53.6, triad 41.4 GB/s (ceiling 53.6) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 93% | 125% | 0 | 43% |
| blas2 | 10 | 102% | 111% | 0 | 16% |
| blas3 | 10 | 41% | 71% | 0 | 64% |
| factor | 12 | 2% | 8% | 0 | 11% |
| eigen | 8 | 2% | 5% | 0 | 11% |
| elementwise | 5 | 7% | 12% | 0 | 12% |
| sparse | 5 | 32% | 116% | 0 | 13% |
| direct | 19 | 0% | 20% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 13% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 34% | 35% | 0 | 14% |
| mg | 1 | 35% | 35% | 0 | 15% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 13% | 22% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 87% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.243 ms | 7.48 | 59.84 | 112% | ok | 4.1x (51%) |
| blas1 | dot | f32 | n=8388608 | 1.345 ms | 12.48 | 49.90 | 93% | ok | 7.3x (92%) |
| blas1 | axpy | f64 | n=8388608 | 3.513 ms | 4.78 | 57.31 | 107% | ok | 3.4x (43%) |
| blas1 | axpy | f32 | n=8388608 | 1.738 ms | 9.65 | 57.92 | 108% | ok | 6.9x (86%) |
| blas1 | scale | f64 | n=8388608 | 1.997 ms | 4.20 | 67.20 | 125% | ok | 1.1x (14%) |
| blas1 | two_norm | f64 | n=8388608 | 1.348 ms | 12.44 | 49.78 | 93% | ok | 7.3x (91%) |
| blas1 | one_norm | f64 | n=8388608 | 5.090 ms | 1.65 | 13.18 | 25% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.099 ms | 1.65 | 13.16 | 25% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 5.079 ms | 1.65 | 13.21 | 25% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 89.991 ms | 2.05 | 1.49 | 3% | ok | 0.9x (12%) |
| blas2 | gemv | f64 | n=4096 | 2.569 ms | 13.06 | 52.28 | 98% | ok | 1.3x (16%) |
| blas2 | gemv | f32 | n=4096 | 1.159 ms | 28.94 | 57.91 | 108% | ok | 1.2x (15%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.686 ms | 12.49 | 49.99 | 93% | ok | 1.3x (16%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.226 ms | 27.38 | 54.78 | 102% | ok | 1.2x (15%) |
| blas2 | symv | f64 | n=4096 | 1.238 ms | 27.11 | 54.26 | 101% | ok | 1.5x (18%) |
| blas2 | trmv | f64 | n=4096 | 1.201 ms | 13.97 | 55.93 | 104% | ok | 1.3x (17%) |
| blas2 | trmv | f32 | n=4096 | 0.563 ms | 29.79 | 59.64 | 111% | ok | 1.6x (20%) |
| blas2 | trsv | f64 | n=4096 | 1.678 ms | 10.00 | 40.03 | 75% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.874 ms | 19.19 | 38.42 | 72% | ok | 1.0x (12%) |
| blas2 | ger | f64 | n=4096 | 4.662 ms | 7.20 | 57.59 | 107% | ok | 1.4x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 0.607 ms | 442.16 | 10.36 | 71% | ok | 6.0x (75%) |
| blas3 | gemm | f32 | 512x512x512 | 0.338 ms | 793.48 | 9.30 | 63% | ok | 5.6x (70%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.367 ms | 98.21 | 4.60 | 16% | ok | 2.0x (25%) |
| blas3 | symm | f64 | 512x512x512 | 0.611 ms | 439.70 | 10.31 | 70% | ok | 6.2x (78%) |
| blas3 | syrk | f64 | 512x512x512 | 1.723 ms | 77.91 | 2.43 | 12% | ok | 1.7x (21%) |
| blas3 | syr2k | f64 | 512x512x512 | 2.395 ms | 112.10 | 2.63 | 18% | ok | 2.0x (25%) |
| blas3 | trmm | f64 | 512x512x512 | 0.400 ms | 335.96 | 13.12 | 54% | ok | 4.7x (59%) |
| blas3 | trsm | f64 | 512x512x512 | 0.544 ms | 246.77 | 9.64 | 39% | ok | 5.2x (65%) |
| blas3 | trsm | f32 | 512x512x512 | 0.263 ms | 510.33 | 9.97 | 41% | ok | 5.1x (64%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.553 ms | 0.56 | 4.45 | 8% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 9.180 ms | 9.75 | 0.46 | 2% | ok | 0.5x (7%) |
| factor | lu | f32 | 512x512 | 4.384 ms | 20.41 | 0.48 | 2% | ok | 0.6x (7%) |
| factor | cholesky | f64 | 512x512 | 6.019 ms | 7.43 | 0.70 | 1%\* | ok | 0.7x (9%) |
| factor | cholesky | f32 | 512x512 | 2.305 ms | 19.41 | 0.91 | 2%\* | ok | 0.7x (9%) |
| factor | ldlt | f64 | 512x512 | 11.832 ms | 3.78 | 0.35 | 1%\* | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 10.591 ms | 4.22 | 0.40 | 1%\* | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 11.835 ms | 15.12 | 0.35 | 2% | ok | 0.9x (11%) |
| factor | lq | f64 | 512x512 | 41.901 ms | 4.27 | 0.10 | 1% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 456.350 ms | 0.98 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.016 ms | 1.03 | 4.13 | 8%\* | ok | 3.8x (48%) |
| factor | inv | f64 | 512x512 | 66.375 ms | 4.04 | 0.06 | 1% | ok | 0.9x (11%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.289 ms | 12.28 | 0.58 | 2% | ok | 0.8x (9%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 3.612 ms | 6.19 | 0.29 | 1% | ok | 0.5x (6%) |
| eigen | eigen_symmetric | f64 | 256x256 | 131.576 ms | 1.15 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 507.990 ms | 0.30 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 21.027 ms | 7.98 | 0.05 | 1% | ok | 0.8x (10%) |
| eigen | schur | f64 | 256x256 | 26.607 ms | 15.76 | 0.04 | 3% | ok | 0.8x (10%) |
| eigen | svd | f64 | 256x256 | 12.055 ms | 30.62 | 0.09 | 5% | ok | 0.7x (9%) |
| eigen | expm | f64 | 256x256 | 17.627 ms | 14.28 | 0.06 | 2% | ok | 0.9x (11%) |
| eigen | sylvester | f64 | 256x256 | 81.857 ms | 12.30 | 0.02 | 2% | ok | 0.9x (11%) |
| elementwise | exp | f64 | n=8388608 | 28.449 ms | 0.29 | 4.72 | 9% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 43.808 ms | 0.19 | 3.06 | 6% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 39.507 ms | 0.21 | 3.40 | 6% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 21.545 ms | 0.39 | 6.23 | 12% | ok | 1.0x (12%) |
| elementwise | erf | f64 | n=8388608 | 35.303 ms | 0.24 | 3.80 | 7% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.439 ms | 5.96 | 62.00 | 116% | ok | 1.9x (24%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.612 ms | 1.62 | 16.89 | 32% | ok | 0.9x (11%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.248 ms | 18.63 | 39.17 | 73% | ok | 2.9x (36%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 18.538 ms | 0.18 | 0.62 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.607 ms | 1.25 | 10.36 | 19%\* | ok | 1.1x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 19.952 ms | 0.56 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 31.758 ms | 0.35 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 15.961 ms | 0.70 | 0.05 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 55.333 ms | 0.40 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 58.235 ms | 0.38 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 27.787 ms | 0.81 | 0.03 | 0% | ok | 0.9x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 38.532 ms | 0.05 | 0.00 | 0%\* | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.759 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.703 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 61.220 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.233 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.578 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.979 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.505 ms | 0.77 | 0.05 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.282 ms | 2.68 | 10.71 | 20%\* | ok | 1.0x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.709 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 6.049 ms | 1.85 | 0.12 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.286 ms | 2.65 | 10.59 | 20%\* | ok | 1.0x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.356 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.525 ms | 0.46 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.043 ms | 0.73 | 2.90 | 5%\* | ok | 1.1x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.853 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 50.193 ms | 0.45 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.093 ms | 0.69 | 2.77 | 5%\* | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.651 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 260.230 ms | 1.19 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.009 ms | 1.68 | 6.71 | 13%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 14.403 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 70.628 ms | 4.37 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.045 ms | 1.65 | 6.59 | 12%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.235 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 902.118 ms | 0.68 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.520 ms | 0.32 | 1.28 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.593 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 510.662 ms | 1.21 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.568 ms | 0.32 | 1.28 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 38.988 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 57.954 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 31.584 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 16.644 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 21.651 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.652 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 53.697 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.374 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 19.768 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 26.052 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 24.549 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 25.374 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 33.087 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.043 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.401 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 23.177 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 1.838 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 128^2 grid | 1.779 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.029 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.357 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.598 ms | 1.64 | 18.36 | 34% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.572 ms | 1.66 | 18.65 | 35% | ok | 1.1x (14%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.083 ms | 1.26 | 14.08 | 26% | ok | 1.4x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.310 ms | 2.51 | 18.99 | 35% | ok | 1.2x (15%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.702 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 33.154 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 55.775 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.824 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 9.878 ms | 1.70 | 6.79 | 13% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.072 ms | 65.92 | 1.03 | 11% | ok | 7.2x (91%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.502 ms | 11.17 | 11.86 | 22% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.267 ms | 5.14 | 20.54 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.871 ms | 5.84 | 11.69 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.396 ms | 12.01 | 12.01 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.396 ms | 12.02 | 12.02 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.208 ms | 222.21 | 2.60 | n/a | ok | 6.9x (87%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.184 ms | 64.15 | 0.50 | n/a | ok | 7.3x (91%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 3.785 ms | 70.91 | 0.42 | n/a | ok | 7.2x (90%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.790 ms | 70.83 | 0.42 | n/a | ok | 7.2x (90%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.809 ms | 70.47 | 0.41 | n/a | ok | 7.2x (90%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

