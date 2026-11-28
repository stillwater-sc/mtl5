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
| bandwidth | read 46.9, update 53.2, triad 38.8 GB/s (ceiling 53.2) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 104% | 131% | 0 |
| blas2 | 10 | 64% | 80% | 0 |
| blas3 | 10 | 23% | 30% | 0 |
| factor | 12 | 3% | 5% | 0 |
| eigen | 8 | 4% | 12% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 25% | 58% | 0 |
| direct | 19 | 1% | 21% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 13% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 31% | 37% | 0 |
| mg | 1 | 34% | 34% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 71% | 71% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 2.424 ms | 6.92 | 55.36 | 104% | ok |
| blas1 | dot | f32 | n=8388608 | 1.219 ms | 13.76 | 55.06 | 103% | ok |
| blas1 | axpy | f64 | n=8388608 | 3.486 ms | 4.81 | 57.76 | 109% | ok |
| blas1 | axpy | f32 | n=8388608 | 1.669 ms | 10.05 | 60.30 | 113% | ok |
| blas1 | scale | f64 | n=8388608 | 1.932 ms | 4.34 | 69.45 | 131% | ok |
| blas1 | two_norm | f64 | n=8388608 | 1.162 ms | 14.43 | 57.73 | 108% | ok |
| blas1 | one_norm | f64 | n=8388608 | 5.062 ms | 1.66 | 13.26 | 25% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 5.072 ms | 1.65 | 13.23 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 5.062 ms | 1.66 | 13.26 | 25% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 88.684 ms | 2.08 | 1.51 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.349 ms | 10.02 | 40.09 | 75% | ok |
| blas2 | gemv | f32 | n=4096 | 1.569 ms | 21.39 | 42.80 | 80% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.955 ms | 8.48 | 33.95 | 64% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.153 ms | 15.58 | 31.18 | 59% | ok |
| blas2 | symv | f64 | n=4096 | 2.790 ms | 12.03 | 24.08 | 45% | ok |
| blas2 | trmv | f64 | n=4096 | 1.817 ms | 9.23 | 36.97 | 69% | ok |
| blas2 | trmv | f32 | n=4096 | 1.090 ms | 15.39 | 30.80 | 58% | ok |
| blas2 | trsv | f64 | n=4096 | 1.795 ms | 9.35 | 37.43 | 70% | ok |
| blas2 | trsv | f32 | n=4096 | 1.086 ms | 15.45 | 30.93 | 58% | ok |
| blas2 | ger | f64 | n=4096 | 13.596 ms | 2.47 | 19.75 | 37% | ok |
| blas3 | gemm | f64 | 512x512x512 | 11.027 ms | 24.34 | 0.57 | 30% | ok |
| blas3 | gemm | f32 | 512x512x512 | 5.593 ms | 48.00 | 0.56 | 30% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 7.494 ms | 17.91 | 0.84 | 22% | ok |
| blas3 | symm | f64 | 512x512x512 | 12.040 ms | 22.30 | 0.52 | 28% | ok |
| blas3 | syrk | f64 | 512x512x512 | 7.407 ms | 18.12 | 0.57 | 23% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 14.202 ms | 18.90 | 0.44 | 24% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.708 ms | 13.82 | 0.54 | 17% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.830 ms | 13.65 | 0.53 | 17% | ok |
| blas3 | trsm | f32 | 512x512x512 | 6.441 ms | 20.84 | 0.41 | 13% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.439 ms | 0.56 | 4.50 | 8% | ok |
| factor | lu | f64 | 512x512 | 32.853 ms | 2.72 | 0.13 | 3% | ok |
| factor | lu | f32 | 512x512 | 31.140 ms | 2.87 | 0.07 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 11.346 ms | 3.94 | 0.37 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 10.888 ms | 4.11 | 0.19 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 11.423 ms | 3.92 | 0.37 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 10.331 ms | 4.33 | 0.41 | 5% | ok |
| factor | qr | f64 | 512x512 | 649.118 ms | 0.28 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 65.056 ms | 2.75 | 0.06 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 745.110 ms | 0.60 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 3.913 ms | 0.27 | 1.07 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 88.622 ms | 3.03 | 0.05 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 34.025 ms | 2.63 | 0.12 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 37.564 ms | 0.60 | 0.03 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 127.855 ms | 1.18 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 516.999 ms | 0.29 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 55.724 ms | 3.01 | 0.02 | 4% | ok |
| eigen | schur | f64 | 256x256 | 114.140 ms | 3.67 | 0.01 | 5% | ok |
| eigen | svd | f64 | 256x256 | 118.189 ms | 3.12 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 25.631 ms | 9.82 | 0.04 | 12% | ok |
| eigen | sylvester | f64 | 256x256 | 257.632 ms | 3.91 | 0.01 | 5% | ok |
| elementwise | exp | f64 | n=8388608 | 27.505 ms | 0.30 | 4.88 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 42.005 ms | 0.20 | 3.20 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 39.402 ms | 0.21 | 3.41 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 21.319 ms | 0.39 | 6.30 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 31.505 ms | 0.27 | 4.26 | 8% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.883 ms | 2.96 | 30.85 | 58% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.429 ms | 1.83 | 19.06 | 36% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.719 ms | 6.23 | 13.10 | 25% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 17.742 ms | 0.18 | 0.65 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.968 ms | 1.02 | 8.46 | 16%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 18.993 ms | 0.59 | 0.04 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 28.032 ms | 0.40 | 0.03 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 13.529 ms | 0.83 | 0.05 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 56.133 ms | 0.40 | 0.01 | 0% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 56.922 ms | 0.39 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 24.904 ms | 0.90 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 37.553 ms | 0.05 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.614 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.572 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 60.305 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.222 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 13.932 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.931 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.701 ms | 0.76 | 0.05 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.274 ms | 2.76 | 11.05 | 21%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.641 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 5.962 ms | 1.88 | 0.12 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.274 ms | 2.76 | 11.03 | 21%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.317 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.431 ms | 0.47 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.155 ms | 0.65 | 2.62 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.773 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 49.667 ms | 0.45 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.052 ms | 0.72 | 2.88 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.580 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 258.243 ms | 1.19 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.986 ms | 1.70 | 6.78 | 13%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 14.054 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 68.947 ms | 4.48 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.007 ms | 1.68 | 6.71 | 13%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.159 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 898.538 ms | 0.69 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.416 ms | 0.32 | 1.29 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.526 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 497.577 ms | 1.24 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.457 ms | 0.32 | 1.29 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 32.654 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 45.638 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 28.331 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 13.486 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 17.930 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 15.730 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 34.401 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 29.491 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 17.312 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 18.607 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 17.499 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.631 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.526 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.045 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.382 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 23.022 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.798 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.738 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.030 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.665 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.472 ms | 1.78 | 19.92 | 37% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.776 ms | 1.47 | 16.52 | 31% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.792 ms | 1.46 | 16.37 | 31% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 16.137 ms | 2.38 | 18.01 | 34% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.399 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.845 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 51.939 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 6.456 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 1.779 ms | 9.43 | 37.72 | 71% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 15.236 ms | 17.62 | 0.28 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.473 ms | 35.49 | 37.71 | 71% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.220 ms | 13.75 | 55.00 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.474 ms | 35.39 | 70.78 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.348 ms | 48.25 | 48.25 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.316 ms | 53.06 | 53.06 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 14.931 ms | 17.98 | 0.21 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 17.330 ms | 15.49 | 0.12 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 17.715 ms | 15.15 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.697 ms | 72.61 | 0.43 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.677 ms | 73.01 | 0.43 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

