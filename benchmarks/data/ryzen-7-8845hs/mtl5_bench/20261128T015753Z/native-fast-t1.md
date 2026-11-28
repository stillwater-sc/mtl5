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

