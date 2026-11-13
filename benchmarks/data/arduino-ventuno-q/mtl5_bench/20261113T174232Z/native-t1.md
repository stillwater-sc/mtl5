## arduino-ventuno-q -- native, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bf315e4604cf` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.6 GFLOP/s |
| ... this build, scalar | f64 8.4, f32 8.4 GFLOP/s |
| bandwidth | read 22.6, update 24.8, triad 23.2 GB/s (ceiling 24.8) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 83% | 104% | 0 |
| blas2 | 10 | 57% | 70% | 0 |
| blas3 | 10 | 29% | 32% | 0 |
| factor | 12 | 12% | 32% | 0 |
| eigen | 8 | 15% | 22% | 0 |
| elementwise | 5 | 5% | 7% | 0 |
| sparse | 5 | 29% | 66% | 0 |
| direct | 19 | 2% | 7% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 9% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 57% | 0 |
| mg | 1 | 36% | 36% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 34% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.681 ms | 2.95 | 23.63 | 95% | ok |
| blas1 | dot | f32 | n=8388608 | 4.318 ms | 3.89 | 15.54 | 63% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.798 ms | 2.15 | 25.82 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.008 ms | 4.19 | 25.12 | 101% | ok |
| blas1 | scale | f64 | n=8388608 | 5.228 ms | 1.60 | 25.67 | 104% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.256 ms | 5.15 | 20.61 | 83% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.075 ms | 1.04 | 8.31 | 34% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.415 ms | 0.48 | 3.85 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.604 ms | 0.97 | 7.80 | 31% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 183.296 ms | 1.01 | 0.73 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 7.854 ms | 4.27 | 17.10 | 69% | ok |
| blas2 | gemv | f32 | n=4096 | 5.704 ms | 5.88 | 11.77 | 48% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 9.967 ms | 3.37 | 13.47 | 54% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.731 ms | 7.09 | 14.19 | 57% | ok |
| blas2 | symv | f64 | n=4096 | 16.302 ms | 2.06 | 4.12 | 17% | ok |
| blas2 | trmv | f64 | n=4096 | 4.321 ms | 3.88 | 15.55 | 63% | ok |
| blas2 | trmv | f32 | n=4096 | 4.825 ms | 3.48 | 6.96 | 28% | ok |
| blas2 | trsv | f64 | n=4096 | 4.547 ms | 3.69 | 14.77 | 60% | ok |
| blas2 | trsv | f32 | n=4096 | 4.750 ms | 3.53 | 7.07 | 29% | ok |
| blas2 | ger | f64 | n=4096 | 15.440 ms | 2.17 | 17.39 | 70% | ok |
| blas3 | gemm | f64 | 512x512x512 | 52.611 ms | 5.10 | 0.12 | 30% | ok |
| blas3 | gemm | f32 | 512x512x512 | 24.662 ms | 10.88 | 0.13 | 32% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 41.264 ms | 3.25 | 0.15 | 19% | ok |
| blas3 | symm | f64 | 512x512x512 | 56.131 ms | 4.78 | 0.11 | 28% | ok |
| blas3 | syrk | f64 | 512x512x512 | 40.642 ms | 3.30 | 0.10 | 20% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 84.792 ms | 3.17 | 0.07 | 19% | ok |
| blas3 | trmm | f64 | 512x512x512 | 27.044 ms | 4.96 | 0.19 | 30% | ok |
| blas3 | trsm | f64 | 512x512x512 | 26.672 ms | 5.03 | 0.20 | 30% | ok |
| blas3 | trsm | f32 | 512x512x512 | 13.596 ms | 9.87 | 0.19 | 29% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 28.784 ms | 0.18 | 1.48 | 6% | ok |
| factor | lu | f64 | 512x512 | 24.787 ms | 3.61 | 0.17 | 21% | ok |
| factor | lu | f32 | 512x512 | 12.974 ms | 6.90 | 0.16 | 21% | ok |
| factor | cholesky | f64 | 512x512 | 20.947 ms | 2.14 | 0.20 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 19.764 ms | 2.26 | 0.11 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 22.796 ms | 1.96 | 0.18 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 68.771 ms | 0.65 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 256.732 ms | 0.70 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 61.272 ms | 2.92 | 0.07 | 17% | ok |
| factor | hessenberg | f64 | 512x512 | 355.022 ms | 1.26 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.574 ms | 0.67 | 2.67 | 11% | ok |
| factor | inv | f64 | 512x512 | 147.940 ms | 1.81 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.795 ms | 5.33 | 0.25 | 32% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.898 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.674 ms | 1.55 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 594.218 ms | 0.25 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.223 ms | 2.50 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 151.585 ms | 2.77 | 0.01 | 16% | ok |
| eigen | svd | f64 | 256x256 | 232.321 ms | 1.59 | 0.00 | 9% | ok |
| eigen | expm | f64 | 256x256 | 67.531 ms | 3.73 | 0.02 | 22% | ok |
| eigen | sylvester | f64 | 256x256 | 380.027 ms | 2.65 | 0.00 | 16% | ok |
| elementwise | exp | f64 | n=8388608 | 87.858 ms | 0.10 | 1.53 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 111.165 ms | 0.08 | 1.21 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 149.882 ms | 0.06 | 0.90 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 73.161 ms | 0.11 | 1.83 | 7% | ok |
| elementwise | erf | f64 | n=8388608 | 111.544 ms | 0.08 | 1.20 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.665 ms | 1.57 | 16.35 | 66% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.203 ms | 1.19 | 12.36 | 50% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.131 ms | 3.45 | 7.26 | 29% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.088 ms | 0.25 | 0.88 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.400 ms | 0.24 | 1.98 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 181.824 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 65.357 ms | 0.17 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.576 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 102.468 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 139.369 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.184 ms | 0.52 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 201.160 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.820 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.529 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.695 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.454 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.007 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 174.227 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.818 ms | 0.42 | 1.66 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.844 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.449 ms | 0.98 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.786 ms | 0.42 | 1.69 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.421 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 87.613 ms | 0.26 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.734 ms | 0.13 | 0.53 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 10.016 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 118.707 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.792 ms | 0.13 | 0.52 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 21.933 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 16532.922 ms | 0.02 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.457 ms | 0.45 | 1.81 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 47.491 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 207.691 ms | 1.49 | 0.00 | 9% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.626 ms | 0.44 | 1.77 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.838 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1771.404 ms | 0.35 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.443 ms | 0.12 | 0.47 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 30.197 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2449.395 ms | 0.25 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.051 ms | 0.12 | 0.48 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.772 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 104.701 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 53.979 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 29.254 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 35.018 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 34.038 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 64.314 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.810 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.584 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 37.301 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.563 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.874 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 46.855 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.112 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.962 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 31.123 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.160 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.984 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.065 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.754 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.074 ms | 1.26 | 14.14 | 57% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.486 ms | 1.05 | 11.80 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.949 ms | 0.89 | 9.94 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.502 ms | 1.18 | 8.94 | 36% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 236.482 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 273.434 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 147.387 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.265 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 8.037 ms | 2.09 | 8.35 | 34% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 48.767 ms | 5.50 | 0.09 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.941 ms | 5.70 | 6.06 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.848 ms | 5.89 | 23.56 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.393 ms | 12.04 | 24.09 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.645 ms | 4.60 | 4.60 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.173 ms | 14.31 | 14.31 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 24.084 ms | 11.15 | 0.13 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 26.339 ms | 10.19 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 20.287 ms | 13.23 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 20.293 ms | 13.23 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 38.488 ms | 6.97 | 0.04 | n/a | ok |

