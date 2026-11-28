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

