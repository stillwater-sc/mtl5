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

