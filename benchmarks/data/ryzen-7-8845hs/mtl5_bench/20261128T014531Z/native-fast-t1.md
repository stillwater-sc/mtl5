## ryzen-7-8845hs -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 79.2 GFLOP/s, f32 159.0 GFLOP/s |
| ... hardware, AVX-512F | f64 79.2, f32 159.0 GFLOP/s |
| ... this build, SSE2 | f64 39.6, f32 79.9 GFLOP/s |
| bandwidth | read 40.9, update 46.5, triad 34.1 GB/s (ceiling 46.5) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 100% | 117% | 0 |
| blas2 | 10 | 76% | 81% | 0 |
| blas3 | 10 | 23% | 30% | 0 |
| factor | 12 | 3% | 5% | 0 |
| eigen | 8 | 4% | 12% | 0 |
| elementwise | 5 | 9% | 14% | 0 |
| sparse | 5 | 29% | 66% | 0 |
| direct | 19 | 1% | 22% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 3% | 15% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 41% | 41% | 0 |
| mg | 1 | 39% | 39% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 81% | 81% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 2.895 ms | 5.80 | 46.37 | 100% | ok |
| blas1 | dot | f32 | n=8388608 | 1.519 ms | 11.04 | 44.18 | 95% | ok |
| blas1 | axpy | f64 | n=8388608 | 4.203 ms | 3.99 | 47.90 | 103% | ok |
| blas1 | axpy | f32 | n=8388608 | 2.022 ms | 8.30 | 49.78 | 107% | ok |
| blas1 | scale | f64 | n=8388608 | 2.461 ms | 3.41 | 54.54 | 117% | ok |
| blas1 | two_norm | f64 | n=8388608 | 1.377 ms | 12.18 | 48.73 | 105% | ok |
| blas1 | one_norm | f64 | n=8388608 | 5.127 ms | 1.64 | 13.09 | 28% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.244 ms | 1.34 | 10.75 | 23% | ok |
| blas1 | sum | f64 | n=8388608 | 5.124 ms | 1.64 | 13.10 | 28% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 98.609 ms | 1.87 | 1.36 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.692 ms | 9.09 | 36.38 | 78% | ok |
| blas2 | gemv | f32 | n=4096 | 1.775 ms | 18.90 | 37.82 | 81% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.972 ms | 8.45 | 33.81 | 73% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.997 ms | 16.80 | 33.62 | 72% | ok |
| blas2 | symv | f64 | n=4096 | 2.730 ms | 12.29 | 24.60 | 53% | ok |
| blas2 | trmv | f64 | n=4096 | 1.898 ms | 8.84 | 35.39 | 76% | ok |
| blas2 | trmv | f32 | n=4096 | 0.956 ms | 17.55 | 35.14 | 76% | ok |
| blas2 | trsv | f64 | n=4096 | 1.832 ms | 9.16 | 36.66 | 79% | ok |
| blas2 | trsv | f32 | n=4096 | 0.963 ms | 17.42 | 34.87 | 75% | ok |
| blas2 | ger | f64 | n=4096 | 13.712 ms | 2.45 | 19.58 | 42% | ok |
| blas3 | gemm | f64 | 512x512x512 | 11.219 ms | 23.93 | 0.56 | 30% | ok |
| blas3 | gemm | f32 | 512x512x512 | 5.659 ms | 47.44 | 0.56 | 30% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 7.710 ms | 17.41 | 0.82 | 22% | ok |
| blas3 | symm | f64 | 512x512x512 | 12.211 ms | 21.98 | 0.52 | 28% | ok |
| blas3 | syrk | f64 | 512x512x512 | 7.474 ms | 17.96 | 0.56 | 23% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 14.459 ms | 18.57 | 0.44 | 23% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.702 ms | 13.83 | 0.54 | 17% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.688 ms | 13.85 | 0.54 | 17% | ok |
| blas3 | trsm | f32 | 512x512x512 | 6.460 ms | 20.78 | 0.41 | 13% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 8.893 ms | 0.60 | 4.78 | 10% | ok |
| factor | lu | f64 | 512x512 | 32.895 ms | 2.72 | 0.13 | 3% | ok |
| factor | lu | f32 | 512x512 | 32.398 ms | 2.76 | 0.06 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 11.391 ms | 3.93 | 0.37 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 10.895 ms | 4.11 | 0.19 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 11.431 ms | 3.91 | 0.37 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 10.414 ms | 4.30 | 0.40 | 5% | ok |
| factor | qr | f64 | 512x512 | 650.452 ms | 0.28 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 64.487 ms | 2.78 | 0.07 | 4% | ok |
| factor | hessenberg | f64 | 512x512 | 745.886 ms | 0.60 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 3.937 ms | 0.27 | 1.07 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 88.567 ms | 3.03 | 0.05 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 33.827 ms | 2.65 | 0.12 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 37.326 ms | 0.60 | 0.03 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 131.031 ms | 1.15 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 509.109 ms | 0.30 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 60.823 ms | 2.76 | 0.02 | 3% | ok |
| eigen | schur | f64 | 256x256 | 110.037 ms | 3.81 | 0.01 | 5% | ok |
| eigen | svd | f64 | 256x256 | 118.573 ms | 3.11 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 25.940 ms | 9.70 | 0.04 | 12% | ok |
| eigen | sylvester | f64 | 256x256 | 263.817 ms | 3.82 | 0.01 | 5% | ok |
| elementwise | exp | f64 | n=8388608 | 27.009 ms | 0.31 | 4.97 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 41.384 ms | 0.20 | 3.24 | 7% | ok |
| elementwise | sin | f64 | n=8388608 | 38.934 ms | 0.22 | 3.45 | 7% | ok |
| elementwise | sqrt | f64 | n=8388608 | 20.509 ms | 0.41 | 6.54 | 14% | ok |
| elementwise | erf | f64 | n=8388608 | 30.969 ms | 0.27 | 4.33 | 9% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.887 ms | 2.95 | 30.70 | 66% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.321 ms | 1.98 | 20.62 | 44% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.603 ms | 6.34 | 13.33 | 29% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.151 ms | 0.20 | 0.71 | 2%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.719 ms | 1.17 | 9.68 | 21%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 18.747 ms | 0.60 | 0.04 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 28.465 ms | 0.39 | 0.03 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 13.624 ms | 0.82 | 0.05 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 55.662 ms | 0.40 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 56.700 ms | 0.39 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 24.507 ms | 0.91 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 38.194 ms | 0.05 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.655 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.595 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 60.527 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.223 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.056 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.924 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.373 ms | 0.78 | 0.05 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.290 ms | 2.60 | 10.42 | 22%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.562 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 5.587 ms | 2.00 | 0.13 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.292 ms | 2.59 | 10.35 | 22%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.269 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.393 ms | 0.47 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.167 ms | 0.65 | 2.59 | 6%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.746 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.768 ms | 0.46 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.120 ms | 0.67 | 2.70 | 6%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.586 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 258.664 ms | 1.19 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.980 ms | 1.70 | 6.81 | 15%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 13.527 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.409 ms | 4.45 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.015 ms | 1.67 | 6.69 | 14%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.072 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 897.791 ms | 0.69 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.621 ms | 0.32 | 1.27 | 3%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.411 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 496.157 ms | 1.24 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.494 ms | 0.32 | 1.28 | 3%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 32.625 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 45.816 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 28.540 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 13.511 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 17.985 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 15.779 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 34.474 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 29.511 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 17.522 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 18.722 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 17.670 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 19.438 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.572 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.044 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.392 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 22.727 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.818 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.745 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.029 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.669 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.536 ms | 1.70 | 19.09 | 41% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.545 ms | 1.69 | 18.98 | 41% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.796 ms | 1.46 | 16.33 | 35% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 16.142 ms | 2.38 | 18.01 | 39% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.215 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.735 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 52.640 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 6.516 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 1.786 ms | 9.39 | 37.56 | 81% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 15.285 ms | 17.56 | 0.27 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.476 ms | 35.25 | 37.45 | 81% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.244 ms | 13.49 | 53.96 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.471 ms | 35.59 | 71.18 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.349 ms | 48.14 | 48.14 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.318 ms | 52.81 | 52.81 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 14.999 ms | 17.90 | 0.21 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 17.363 ms | 15.46 | 0.12 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 17.736 ms | 15.14 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.722 ms | 72.12 | 0.42 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.698 ms | 72.59 | 0.43 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

