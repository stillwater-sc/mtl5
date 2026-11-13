# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,4, CPUs 0,1,2,3, clock policy governor performance on the A78C policies; performance 902-2112 MHz on cpus 0,1; performance 940-2361 MHz on cpus 2,3, 2026-11-13T18:20:08Z.

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

## arduino-ventuno-q -- native, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bf315e4604cf` |
| FMA peak (ceiling) | f64 67.8 GFLOP/s, f32 134.9 GFLOP/s |
| ... hardware, NEON | f64 67.8, f32 134.9 GFLOP/s |
| ... this build, scalar | f64 33.8, f32 33.8 GFLOP/s |
| bandwidth | read 23.0, update 24.7, triad 22.7 GB/s (ceiling 24.7) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 98% | 106% | 0 | 28% |
| blas2 | 10 | 63% | 88% | 0 | 27% |
| blas3 | 10 | 8% | 17% | 0 | 28% |
| factor | 12 | 4% | 39% | 0 | 36% |
| eigen | 8 | 4% | 10% | 0 | 28% |
| elementwise | 5 | 5% | 8% | 0 | 28% |
| sparse | 5 | 56% | 92% | 0 | 28% |
| direct | 19 | 0% | 6% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 27% |
| direct3d | 12 | 2% | 7% | 0 | 28% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 28% |
| smoother | 3 | 52% | 59% | 0 | 27% |
| mg | 1 | 36% | 36% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 31% | 38% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 51% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.556 ms | 3.02 | 24.16 | 98% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.799 ms | 5.99 | 23.97 | 97% | ok | 1.5x (39%) |
| blas1 | axpy | f64 | n=8388608 | 8.233 ms | 2.04 | 24.45 | 99% | ok | 0.9x (24%) |
| blas1 | axpy | f32 | n=8388608 | 4.024 ms | 4.17 | 25.01 | 101% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 5.113 ms | 1.64 | 26.25 | 106% | ok | 1.0x (26%) |
| blas1 | two_norm | f64 | n=8388608 | 2.617 ms | 6.41 | 25.65 | 104% | ok | 1.2x (31%) |
| blas1 | one_norm | f64 | n=8388608 | 7.331 ms | 1.14 | 9.15 | 37% | ok | 1.1x (28%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.560 ms | 0.54 | 4.31 | 17% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.410 ms | 1.13 | 9.06 | 37% | ok | 1.2x (29%) |
| blas1 | dot_k3 | f64 | n=8388608 | 168.062 ms | 1.10 | 0.80 | 3% | ok | 1.1x (27%) |
| blas2 | gemv | f64 | n=4096 | 6.676 ms | 5.03 | 20.12 | 81% | ok | 1.2x (29%) |
| blas2 | gemv | f32 | n=4096 | 3.081 ms | 10.89 | 21.80 | 88% | ok | 1.9x (46%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.769 ms | 4.32 | 17.28 | 70% | ok | 1.3x (32%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.649 ms | 5.94 | 11.89 | 48% | ok | 0.8x (21%) |
| blas2 | symv | f64 | n=4096 | 15.010 ms | 2.24 | 4.48 | 18% | ok | 1.1x (27%) |
| blas2 | trmv | f64 | n=4096 | 4.374 ms | 3.84 | 15.36 | 62% | ok | 1.0x (25%) |
| blas2 | trmv | f32 | n=4096 | 4.369 ms | 3.84 | 7.69 | 31% | ok | 1.1x (28%) |
| blas2 | trsv | f64 | n=4096 | 4.339 ms | 3.87 | 15.48 | 63% | ok | 1.0x (26%) |
| blas2 | trsv | f32 | n=4096 | 4.353 ms | 3.85 | 7.72 | 31% | ok | 1.1x (27%) |
| blas2 | ger | f64 | n=4096 | 15.039 ms | 2.23 | 17.85 | 72% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 27.219 ms | 9.86 | 0.23 | 15% | ok | 1.9x (48%) |
| blas3 | gemm | f32 | 512x512x512 | 11.550 ms | 23.24 | 0.27 | 17% | ok | 2.1x (53%) |
| blas3 | gemmt | f64 | 512x512x512 | 40.583 ms | 3.31 | 0.16 | 5% | ok | 1.0x (25%) |
| blas3 | symm | f64 | 512x512x512 | 49.009 ms | 5.48 | 0.13 | 8% | ok | 1.1x (29%) |
| blas3 | syrk | f64 | 512x512x512 | 40.384 ms | 3.32 | 0.10 | 5% | ok | 1.0x (25%) |
| blas3 | syr2k | f64 | 512x512x512 | 82.731 ms | 3.24 | 0.08 | 5% | ok | 1.0x (26%) |
| blas3 | trmm | f64 | 512x512x512 | 24.171 ms | 5.55 | 0.22 | 8% | ok | 1.1x (28%) |
| blas3 | trsm | f64 | 512x512x512 | 23.894 ms | 5.62 | 0.22 | 8% | ok | 1.1x (28%) |
| blas3 | trsm | f32 | 512x512x512 | 11.159 ms | 12.03 | 0.23 | 9% | ok | 1.2x (30%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 25.451 ms | 0.21 | 1.67 | 7% | ok | 1.1x (28%) |
| factor | lu | f64 | 512x512 | 17.312 ms | 5.17 | 0.24 | 8% | ok | 1.4x (36%) |
| factor | lu | f32 | 512x512 | 8.960 ms | 9.99 | 0.23 | 7% | ok | 1.4x (36%) |
| factor | cholesky | f64 | 512x512 | 18.582 ms | 2.41 | 0.23 | 4% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 17.579 ms | 2.55 | 0.12 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 20.274 ms | 2.21 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 61.075 ms | 0.73 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 148.124 ms | 1.21 | 0.03 | 2% | ok | 1.7x (43%) |
| factor | lq | f64 | 512x512 | 36.670 ms | 4.88 | 0.11 | 7% | ok | 1.7x (42%) |
| factor | hessenberg | f64 | 512x512 | 202.009 ms | 2.21 | 0.02 | 3% | ok | 1.8x (44%) |
| factor | householder | f64 | 512x512 | 0.439 ms | 2.39 | 9.54 | 39% | ok | 3.6x (90%) |
| factor | inv | f64 | 512x512 | 127.813 ms | 2.10 | 0.03 | 3% | ok | 1.2x (29%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.349 ms | 7.25 | 0.34 | 11% | ok | 1.4x (34%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.531 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.874 ms | 1.74 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 527.507 ms | 0.29 | 0.00 | 0% | ok | 1.1x (28%) |
| eigen | eigenvalue | f64 | 256x256 | 60.093 ms | 2.79 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 134.219 ms | 3.12 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.403 ms | 1.80 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 38.219 ms | 6.58 | 0.03 | 10% | ok | 1.8x (44%) |
| eigen | sylvester | f64 | 256x256 | 316.501 ms | 3.18 | 0.00 | 5% | ok | 1.2x (30%) |
| elementwise | exp | f64 | n=8388608 | 81.559 ms | 0.10 | 1.65 | 7% | ok | 1.1x (27%) |
| elementwise | log | f64 | n=8388608 | 99.961 ms | 0.08 | 1.34 | 5% | ok | 1.1x (28%) |
| elementwise | sin | f64 | n=8388608 | 135.132 ms | 0.06 | 0.99 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 66.418 ms | 0.13 | 2.02 | 8% | ok | 1.1x (28%) |
| elementwise | erf | f64 | n=8388608 | 107.102 ms | 0.08 | 1.25 | 5% | ok | 1.0x (26%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.199 ms | 2.18 | 22.71 | 92% | ok | 1.4x (35%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.954 ms | 1.34 | 13.94 | 56% | ok | 1.1x (28%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.011 ms | 6.97 | 14.65 | 59% | ok | 2.0x (50%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.574 ms | 0.28 | 0.99 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.601 ms | 0.23 | 1.94 | 8% | ok | 1.0x (24%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 163.328 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 58.512 ms | 0.19 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.380 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 92.220 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 124.112 ms | 0.18 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 38.899 ms | 0.58 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 185.001 ms | 0.01 | 0.00 | 0% | ok | 1.1x (27%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.755 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.575 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.637 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.749 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.361 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 155.660 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.000 ms | 0.38 | 1.51 | 6% | ok | 0.9x (23%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.653 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.317 ms | 1.08 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.992 ms | 0.38 | 1.52 | 6% | ok | 0.9x (22%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.998 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 78.203 ms | 0.29 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.554 ms | 0.14 | 0.54 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.069 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 106.565 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.555 ms | 0.14 | 0.54 | 2% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 19.485 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 14695.201 ms | 0.02 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.631 ms | 0.44 | 1.77 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 41.857 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 181.443 ms | 1.70 | 0.00 | 3% | ok | 1.1x (29%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.953 ms | 0.42 | 1.69 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.912 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1582.879 ms | 0.39 | 0.00 | 1% | ok | 1.1x (28%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.613 ms | 0.12 | 0.49 | 2% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 26.292 ms | 0.00 | 0.00 | n/a | ok | 1.1x (29%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2286.196 ms | 0.27 | 0.00 | 0% | ok | 1.1x (27%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 26.997 ms | 0.12 | 0.50 | 2% | ok | 1.0x (26%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 63.391 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 95.684 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.373 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.054 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.195 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.649 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.661 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.614 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.026 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.557 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.336 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.489 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.177 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.104 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.479 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilut | f64 | 128^2 grid | 27.470 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.863 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ildl | f64 | 128^2 grid | 2.746 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.577 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.002 ms | 1.31 | 14.65 | 59% | ok | 1.0x (26%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.288 ms | 1.14 | 12.82 | 52% | ok | 1.1x (27%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.658 ms | 0.98 | 11.03 | 45% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.640 ms | 1.18 | 8.90 | 36% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 208.548 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 241.245 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 135.885 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.941 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.210 ms | 2.33 | 9.31 | 38% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.845 ms | 20.90 | 0.33 | 31% | ok | 3.8x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.657 ms | 6.31 | 6.71 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.812 ms | 5.97 | 23.87 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.357 ms | 12.36 | 24.72 | n/a | ok | 1.0x (26%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.268 ms | 5.13 | 5.13 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.037 ms | 16.19 | 16.19 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.831 ms | 22.69 | 0.27 | n/a | ok | 2.0x (51%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.677 ms | 40.20 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.139 ms | 52.24 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.136 ms | 52.26 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 9.765 ms | 27.49 | 0.16 | n/a | ok | 3.9x (99%) |

## arduino-ventuno-q -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON: -march=armv8.2-a+dotprod+fp16+rcpc -mtune=cortex-a78c]`, SIMD NEON_WITHOUT_AES |
| commit | `bf315e4604cf` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.6 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 16.8, f32 33.6 GFLOP/s |
| bandwidth | read 22.3, update 24.7, triad 23.1 GB/s (ceiling 24.7) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 94% | 105% | 0 |
| blas2 | 10 | 66% | 72% | 0 |
| blas3 | 10 | 45% | 58% | 0 |
| factor | 12 | 12% | 31% | 0 |
| eigen | 8 | 15% | 31% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 29% | 66% | 0 |
| direct | 19 | 1% | 7% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 9% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 57% | 0 |
| mg | 1 | 36% | 36% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 67% | 132% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.736 ms | 2.92 | 23.40 | 95% | ok |
| blas1 | dot | f32 | n=8388608 | 2.886 ms | 5.81 | 23.25 | 94% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.752 ms | 2.16 | 25.97 | 105% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.917 ms | 4.28 | 25.70 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 5.314 ms | 1.58 | 25.26 | 102% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.141 ms | 5.34 | 21.37 | 87% | ok |
| blas1 | one_norm | f64 | n=8388608 | 9.014 ms | 0.93 | 7.44 | 30% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.631 ms | 0.48 | 3.81 | 15% | ok |
| blas1 | sum | f64 | n=8388608 | 8.317 ms | 1.01 | 8.07 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 199.963 ms | 0.92 | 0.67 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 8.195 ms | 4.09 | 16.39 | 66% | ok |
| blas2 | gemv | f32 | n=4096 | 4.073 ms | 8.24 | 16.48 | 67% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.866 ms | 4.27 | 17.07 | 69% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.971 ms | 8.45 | 16.91 | 68% | ok |
| blas2 | symv | f64 | n=4096 | 16.071 ms | 2.09 | 4.18 | 17% | ok |
| blas2 | trmv | f64 | n=4096 | 4.888 ms | 3.43 | 13.74 | 56% | ok |
| blas2 | trmv | f32 | n=4096 | 2.688 ms | 6.24 | 12.49 | 51% | ok |
| blas2 | trsv | f64 | n=4096 | 4.768 ms | 3.52 | 14.09 | 57% | ok |
| blas2 | trsv | f32 | n=4096 | 2.750 ms | 6.10 | 12.21 | 49% | ok |
| blas2 | ger | f64 | n=4096 | 15.196 ms | 2.21 | 17.67 | 72% | ok |
| blas3 | gemm | f64 | 512x512x512 | 28.268 ms | 9.50 | 0.22 | 57% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.743 ms | 19.53 | 0.23 | 58% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 18.060 ms | 7.43 | 0.35 | 44% | ok |
| blas3 | symm | f64 | 512x512x512 | 29.148 ms | 9.21 | 0.22 | 55% | ok |
| blas3 | syrk | f64 | 512x512x512 | 41.873 ms | 3.21 | 0.10 | 19% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 84.832 ms | 3.16 | 0.07 | 19% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.708 ms | 7.58 | 0.30 | 45% | ok |
| blas3 | trsm | f64 | 512x512x512 | 17.964 ms | 7.47 | 0.29 | 44% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.493 ms | 15.80 | 0.31 | 47% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 26.493 ms | 0.20 | 1.60 | 6% | ok |
| factor | lu | f64 | 512x512 | 24.351 ms | 3.67 | 0.17 | 22% | ok |
| factor | lu | f32 | 512x512 | 13.046 ms | 6.86 | 0.16 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 22.522 ms | 1.99 | 0.19 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 21.018 ms | 2.13 | 0.10 | 6% | ok |
| factor | ldlt | f64 | 512x512 | 21.556 ms | 2.08 | 0.19 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 68.900 ms | 0.65 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 246.684 ms | 0.73 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 60.865 ms | 2.94 | 0.07 | 17% | ok |
| factor | hessenberg | f64 | 512x512 | 346.236 ms | 1.29 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.488 ms | 0.71 | 2.82 | 11% | ok |
| factor | inv | f64 | 512x512 | 145.303 ms | 1.85 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.922 ms | 5.29 | 0.25 | 31% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 40.346 ms | 0.55 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 99.011 ms | 1.53 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 629.162 ms | 0.24 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 68.115 ms | 2.46 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 152.798 ms | 2.74 | 0.01 | 16% | ok |
| eigen | svd | f64 | 256x256 | 230.284 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 48.838 ms | 5.15 | 0.02 | 31% | ok |
| eigen | sylvester | f64 | 256x256 | 360.924 ms | 2.79 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 86.781 ms | 0.10 | 1.55 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 103.559 ms | 0.08 | 1.30 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 151.687 ms | 0.06 | 0.88 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 70.501 ms | 0.12 | 1.90 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 116.501 ms | 0.07 | 1.15 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.667 ms | 1.57 | 16.34 | 66% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.157 ms | 1.21 | 12.62 | 51% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.131 ms | 3.45 | 7.26 | 29% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.231 ms | 0.25 | 0.87 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.721 ms | 0.23 | 1.91 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 183.979 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 59.821 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.753 ms | 0.42 | 0.03 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 103.895 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 140.800 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.534 ms | 0.51 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 202.838 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.878 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 14.506 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.703 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.572 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.101 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 175.307 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.828 ms | 0.41 | 1.65 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.943 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.638 ms | 0.96 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.693 ms | 0.45 | 1.79 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.566 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 89.222 ms | 0.25 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.936 ms | 0.13 | 0.51 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 10.251 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 124.352 ms | 0.18 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.884 ms | 0.13 | 0.51 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 22.505 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 16814.876 ms | 0.02 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.613 ms | 0.44 | 1.77 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 46.962 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 215.839 ms | 1.43 | 0.00 | 9% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.752 ms | 0.43 | 1.74 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.924 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1706.270 ms | 0.36 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.645 ms | 0.12 | 0.47 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 30.158 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2437.504 ms | 0.25 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.273 ms | 0.12 | 0.48 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.626 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 104.119 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 54.675 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 29.033 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.641 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 33.872 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 63.811 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.627 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.522 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 37.471 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.609 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.596 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 46.938 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.109 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.932 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 31.260 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.083 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.957 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.064 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.744 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.086 ms | 1.25 | 14.06 | 57% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.500 ms | 1.05 | 11.73 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.944 ms | 0.89 | 9.96 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.277 ms | 1.19 | 9.01 | 36% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 233.946 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 270.470 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 145.570 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.177 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.086 ms | 4.11 | 16.43 | 67% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 41.577 ms | 6.46 | 0.10 | 38% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.755 ms | 22.21 | 23.60 | 132% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.838 ms | 5.91 | 23.65 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.395 ms | 12.03 | 24.06 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.695 ms | 24.13 | 24.13 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.525 ms | 11.00 | 11.00 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.356 ms | 15.47 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 16.687 ms | 16.09 | 0.13 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 16.591 ms | 16.18 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.678 ms | 72.99 | 0.43 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.558 ms | 19.80 | 0.12 | n/a | ok |

## arduino-ventuno-q -- native-fast, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON: -march=armv8.2-a+dotprod+fp16+rcpc -mtune=cortex-a78c]`, SIMD NEON_WITHOUT_AES |
| commit | `bf315e4604cf` |
| FMA peak (ceiling) | f64 67.5 GFLOP/s, f32 135.0 GFLOP/s |
| ... hardware, NEON | f64 67.4, f32 134.9 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 67.5, f32 135.0 GFLOP/s |
| bandwidth | read 23.0, update 24.5, triad 22.4 GB/s (ceiling 24.5) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 98% | 106% | 0 | 27% |
| blas2 | 10 | 58% | 90% | 0 | 26% |
| blas3 | 10 | 28% | 55% | 0 | 59% |
| factor | 12 | 3% | 40% | 0 | 35% |
| eigen | 8 | 4% | 13% | 0 | 29% |
| elementwise | 5 | 5% | 9% | 0 | 27% |
| sparse | 5 | 56% | 92% | 0 | 28% |
| direct | 19 | 0% | 6% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 29% |
| direct3d | 12 | 2% | 7% | 0 | 28% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 53% | 62% | 0 | 28% |
| mg | 1 | 37% | 37% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 74% | 76% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 96% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.549 ms | 3.02 | 24.19 | 99% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.806 ms | 5.98 | 23.91 | 98% | ok | 1.0x (26%) |
| blas1 | axpy | f64 | n=8388608 | 8.216 ms | 2.04 | 24.50 | 100% | ok | 0.9x (24%) |
| blas1 | axpy | f32 | n=8388608 | 4.332 ms | 3.87 | 23.24 | 95% | ok | 0.9x (23%) |
| blas1 | scale | f64 | n=8388608 | 5.181 ms | 1.62 | 25.90 | 106% | ok | 1.0x (26%) |
| blas1 | two_norm | f64 | n=8388608 | 2.633 ms | 6.37 | 25.48 | 104% | ok | 1.2x (30%) |
| blas1 | one_norm | f64 | n=8388608 | 7.291 ms | 1.15 | 9.20 | 38% | ok | 1.2x (31%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.795 ms | 0.53 | 4.25 | 17% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.661 ms | 1.10 | 8.76 | 36% | ok | 1.1x (27%) |
| blas1 | dot_k3 | f64 | n=8388608 | 164.683 ms | 1.12 | 0.82 | 3% | ok | 1.2x (30%) |
| blas2 | gemv | f64 | n=4096 | 6.699 ms | 5.01 | 20.04 | 82% | ok | 1.2x (31%) |
| blas2 | gemv | f32 | n=4096 | 3.060 ms | 10.97 | 21.94 | 90% | ok | 1.3x (33%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.686 ms | 4.37 | 17.47 | 71% | ok | 1.0x (26%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.709 ms | 7.13 | 14.26 | 58% | ok | 0.8x (21%) |
| blas2 | symv | f64 | n=4096 | 15.270 ms | 2.20 | 4.40 | 18% | ok | 1.1x (26%) |
| blas2 | trmv | f64 | n=4096 | 4.693 ms | 3.57 | 14.31 | 58% | ok | 1.0x (26%) |
| blas2 | trmv | f32 | n=4096 | 2.586 ms | 6.49 | 12.99 | 53% | ok | 1.0x (26%) |
| blas2 | trsv | f64 | n=4096 | 4.831 ms | 3.47 | 13.91 | 57% | ok | 1.0x (25%) |
| blas2 | trsv | f32 | n=4096 | 2.645 ms | 6.34 | 12.70 | 52% | ok | 1.0x (26%) |
| blas2 | ger | f64 | n=4096 | 14.461 ms | 2.32 | 18.57 | 76% | ok | 1.1x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 7.430 ms | 36.13 | 0.85 | 54% | ok | 3.8x (95%) |
| blas3 | gemm | f32 | 512x512x512 | 3.589 ms | 74.79 | 0.88 | 55% | ok | 3.8x (96%) |
| blas3 | gemmt | f64 | 512x512x512 | 5.951 ms | 22.56 | 1.06 | 33% | ok | 3.0x (76%) |
| blas3 | symm | f64 | 512x512x512 | 8.135 ms | 33.00 | 0.77 | 49% | ok | 3.6x (90%) |
| blas3 | syrk | f64 | 512x512x512 | 37.168 ms | 3.61 | 0.11 | 5% | ok | 1.1x (28%) |
| blas3 | syr2k | f64 | 512x512x512 | 81.419 ms | 3.30 | 0.08 | 5% | ok | 1.0x (26%) |
| blas3 | trmm | f64 | 512x512x512 | 7.655 ms | 17.53 | 0.68 | 26% | ok | 2.3x (58%) |
| blas3 | trsm | f64 | 512x512x512 | 8.221 ms | 16.33 | 0.64 | 24% | ok | 2.2x (55%) |
| blas3 | trsm | f32 | 512x512x512 | 3.604 ms | 37.24 | 0.73 | 28% | ok | 2.4x (59%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 25.242 ms | 0.21 | 1.68 | 7% | ok | 1.0x (26%) |
| factor | lu | f64 | 512x512 | 17.170 ms | 5.21 | 0.24 | 8% | ok | 1.4x (35%) |
| factor | lu | f32 | 512x512 | 8.986 ms | 9.96 | 0.23 | 7% | ok | 1.5x (36%) |
| factor | cholesky | f64 | 512x512 | 20.102 ms | 2.23 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 18.795 ms | 2.38 | 0.11 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 18.995 ms | 2.36 | 0.22 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 61.866 ms | 0.72 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 145.931 ms | 1.23 | 0.03 | 2% | ok | 1.7x (42%) |
| factor | lq | f64 | 512x512 | 35.448 ms | 5.05 | 0.12 | 7% | ok | 1.7x (43%) |
| factor | hessenberg | f64 | 512x512 | 198.763 ms | 2.25 | 0.02 | 3% | ok | 1.7x (44%) |
| factor | householder | f64 | 512x512 | 0.430 ms | 2.44 | 9.74 | 40% | ok | 3.5x (86%) |
| factor | inv | f64 | 512x512 | 127.041 ms | 2.11 | 0.03 | 3% | ok | 1.1x (29%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.146 ms | 7.37 | 0.35 | 11% | ok | 1.4x (35%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.640 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.764 ms | 1.74 | 0.01 | 3% | ok | 1.1x (29%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 528.176 ms | 0.29 | 0.00 | 0% | ok | 1.2x (30%) |
| eigen | eigenvalue | f64 | 256x256 | 60.584 ms | 2.77 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 135.279 ms | 3.10 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 204.974 ms | 1.80 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 29.048 ms | 8.66 | 0.04 | 13% | ok | 1.7x (42%) |
| eigen | sylvester | f64 | 256x256 | 315.028 ms | 3.20 | 0.00 | 5% | ok | 1.1x (29%) |
| elementwise | exp | f64 | n=8388608 | 79.608 ms | 0.11 | 1.69 | 7% | ok | 1.1x (27%) |
| elementwise | log | f64 | n=8388608 | 101.458 ms | 0.08 | 1.32 | 5% | ok | 1.0x (26%) |
| elementwise | sin | f64 | n=8388608 | 134.417 ms | 0.06 | 1.00 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 59.836 ms | 0.14 | 2.24 | 9% | ok | 1.2x (29%) |
| elementwise | erf | f64 | n=8388608 | 105.937 ms | 0.08 | 1.27 | 5% | ok | 1.1x (27%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.207 ms | 2.17 | 22.55 | 92% | ok | 1.4x (35%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.969 ms | 1.33 | 13.83 | 56% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.133 ms | 6.83 | 14.36 | 59% | ok | 2.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.892 ms | 0.27 | 0.97 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.896 ms | 0.23 | 1.87 | 8% | ok | 1.0x (25%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 164.714 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 53.684 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.525 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 95.054 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 126.742 ms | 0.18 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.375 ms | 0.57 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 183.722 ms | 0.01 | 0.00 | 0% | ok | 1.1x (28%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.740 ms | 0.00 | 0.00 | n/a | ok | 1.1x (29%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.138 ms | 0.00 | 0.00 | n/a | ok | 1.2x (30%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.633 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 39.007 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.497 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 157.345 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.332 ms | 0.32 | 1.30 | 5% | ok | 0.8x (20%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.789 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.369 ms | 1.08 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.199 ms | 0.34 | 1.38 | 6% | ok | 0.8x (19%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.937 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 79.883 ms | 0.28 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.653 ms | 0.13 | 0.54 | 2% | ok | 1.1x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.075 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 108.373 ms | 0.21 | 0.01 | 0% | ok | 1.1x (29%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.688 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 19.348 ms | 0.00 | 0.00 | n/a | ok | 1.2x (29%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 14979.374 ms | 0.02 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.873 ms | 0.43 | 1.71 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 41.982 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 186.582 ms | 1.65 | 0.00 | 2% | ok | 1.2x (29%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.646 ms | 0.44 | 1.76 | 7% | ok | 1.0x (25%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.586 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1535.724 ms | 0.40 | 0.00 | 1% | ok | 1.1x (28%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.272 ms | 0.12 | 0.49 | 2% | ok | 1.1x (26%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 26.129 ms | 0.00 | 0.00 | n/a | ok | 1.2x (29%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2290.915 ms | 0.27 | 0.00 | 0% | ok | 1.1x (27%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.059 ms | 0.12 | 0.48 | 2% | ok | 1.0x (25%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.100 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 97.382 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 49.206 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.209 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.383 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.535 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.238 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.599 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.241 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.960 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.404 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.487 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.119 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.102 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.453 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilut | f64 | 128^2 grid | 27.997 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.795 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ildl | f64 | 128^2 grid | 2.717 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.065 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.591 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.933 ms | 1.35 | 15.17 | 62% | ok | 1.1x (27%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.268 ms | 1.15 | 12.93 | 53% | ok | 1.1x (28%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.655 ms | 0.99 | 11.05 | 45% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 31.992 ms | 1.20 | 9.09 | 37% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 211.234 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 245.743 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 135.125 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.809 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 3.694 ms | 4.54 | 18.16 | 74% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 10.698 ms | 25.09 | 0.39 | 37% | ok | 3.9x (97%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.962 ms | 17.44 | 18.53 | 76% | ok | 0.8x (20%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.810 ms | 5.97 | 23.88 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.369 ms | 12.25 | 24.50 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.677 ms | 24.78 | 24.78 | n/a | ok | 1.0x (26%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.361 ms | 12.33 | 12.33 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 4.478 ms | 59.94 | 0.70 | n/a | ok | 3.9x (97%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.303 ms | 62.39 | 0.49 | n/a | ok | 3.9x (97%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.195 ms | 63.99 | 0.37 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.961 ms | 279.39 | 1.64 | n/a | ok | 3.8x (96%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.324 ms | 80.75 | 0.47 | n/a | ok | 4.1x (102%) |

## arduino-ventuno-q -- blas, 1 thread, class medium

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
| bandwidth | read 22.3, update 24.6, triad 23.0 GB/s (ceiling 24.6) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 89% | 104% | 0 |
| blas2 | 10 | 58% | 73% | 0 |
| blas3 | 10 | 83% | 93% | 0 |
| factor | 12 | 41% | 58% | 0 |
| eigen | 8 | 34% | 75% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 30% | 68% | 0 |
| direct | 23 | 2% | 11% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 15 | 7% | 42% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 40% | 56% | 0 |
| mg | 1 | 37% | 37% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 34% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.697 ms | 2.95 | 23.56 | 96% | ok |
| blas1 | dot | f32 | n=8388608 | 4.506 ms | 3.72 | 14.89 | 60% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.820 ms | 2.15 | 25.74 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.954 ms | 4.24 | 25.46 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 5.402 ms | 1.55 | 24.85 | 101% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.049 ms | 5.50 | 22.01 | 89% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.279 ms | 1.01 | 8.11 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.437 ms | 0.48 | 3.85 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.453 ms | 0.99 | 7.94 | 32% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 191.304 ms | 0.96 | 0.70 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 8.985 ms | 3.73 | 14.95 | 61% | ok |
| blas2 | gemv | f32 | n=4096 | 3.819 ms | 8.79 | 17.58 | 71% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 9.325 ms | 3.60 | 14.40 | 58% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.717 ms | 9.03 | 18.06 | 73% | ok |
| blas2 | symv | f64 | n=4096 | 11.531 ms | 2.91 | 5.83 | 24% | ok |
| blas2 | trmv | f64 | n=4096 | 6.296 ms | 2.66 | 10.67 | 43% | ok |
| blas2 | trmv | f32 | n=4096 | 3.262 ms | 5.14 | 10.30 | 42% | ok |
| blas2 | trsv | f64 | n=4096 | 6.375 ms | 2.63 | 10.54 | 43% | ok |
| blas2 | trsv | f32 | n=4096 | 3.155 ms | 5.32 | 10.65 | 43% | ok |
| blas2 | ger | f64 | n=4096 | 15.409 ms | 2.18 | 17.42 | 71% | ok |
| blas3 | gemm | f64 | 512x512x512 | 17.720 ms | 15.15 | 0.36 | 90% | ok |
| blas3 | gemm | f32 | 512x512x512 | 8.641 ms | 31.07 | 0.36 | 93% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 11.387 ms | 11.79 | 0.55 | 70% | ok |
| blas3 | symm | f64 | 512x512x512 | 17.933 ms | 14.97 | 0.35 | 89% | ok |
| blas3 | syrk | f64 | 512x512x512 | 9.857 ms | 13.62 | 0.43 | 81% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 19.194 ms | 13.99 | 0.33 | 83% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.059 ms | 14.82 | 0.58 | 88% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.710 ms | 13.82 | 0.54 | 82% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.006 ms | 26.81 | 0.52 | 80% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 26.407 ms | 0.20 | 1.61 | 7% | ok |
| factor | lu | f64 | 512x512 | 9.159 ms | 9.77 | 0.46 | 58% | ok |
| factor | lu | f32 | 512x512 | 5.506 ms | 16.25 | 0.38 | 48% | ok |
| factor | cholesky | f64 | 512x512 | 5.387 ms | 8.30 | 0.78 | 50% | ok |
| factor | cholesky | f32 | 512x512 | 3.216 ms | 13.91 | 0.65 | 41% | ok |
| factor | ldlt | f64 | 512x512 | 22.599 ms | 1.98 | 0.19 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 68.372 ms | 0.65 | 0.06 | 4% | ok |
| factor | qr | f64 | 512x512 | 19.763 ms | 9.06 | 0.21 | 54% | ok |
| factor | lq | f64 | 512x512 | 60.637 ms | 2.95 | 0.07 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 349.518 ms | 1.28 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.550 ms | 0.68 | 2.71 | 11% | ok |
| factor | inv | f64 | 512x512 | 131.166 ms | 2.05 | 0.03 | 12% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 10.242 ms | 8.74 | 0.41 | 52% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 7.361 ms | 3.04 | 0.14 | 18% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 97.025 ms | 1.56 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 626.441 ms | 0.24 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 45.139 ms | 3.72 | 0.02 | 22% | ok |
| eigen | schur | f64 | 256x256 | 63.383 ms | 6.62 | 0.02 | 39% | ok |
| eigen | svd | f64 | 256x256 | 29.428 ms | 12.54 | 0.04 | 75% | ok |
| eigen | expm | f64 | 256x256 | 38.007 ms | 6.62 | 0.03 | 39% | ok |
| eigen | sylvester | f64 | 256x256 | 175.947 ms | 5.72 | 0.01 | 34% | ok |
| elementwise | exp | f64 | n=8388608 | 83.479 ms | 0.10 | 1.61 | 7% | ok |
| elementwise | log | f64 | n=8388608 | 107.659 ms | 0.08 | 1.25 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 150.444 ms | 0.06 | 0.89 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 68.978 ms | 0.12 | 1.95 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 112.370 ms | 0.07 | 1.19 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.637 ms | 1.60 | 16.63 | 68% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.161 ms | 1.21 | 12.60 | 51% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.998 ms | 3.49 | 7.34 | 30% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 12.756 ms | 0.26 | 0.90 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.806 ms | 0.23 | 1.89 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 181.317 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 57.470 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.357 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 104.893 ms | 0.21 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 142.881 ms | 0.16 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 43.859 ms | 0.51 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 201.371 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.793 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.690 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.700 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.531 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.909 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 174.088 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.847 ms | 0.41 | 1.64 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.747 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.453 ms | 0.98 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.728 ms | 0.44 | 1.75 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.431 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 89.806 ms | 0.25 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.928 ms | 0.13 | 0.51 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 10.010 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 120.746 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.968 ms | 0.13 | 0.51 | 2% | ok |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 16.746 ms | 0.67 | 0.04 | 4% | ok |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 3.892 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 11.253 ms | 0.99 | 0.06 | 6% | ok |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.161 ms | 0.65 | 2.60 | 11% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 21.428 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 16474.028 ms | 0.02 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.697 ms | 0.44 | 1.75 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 46.632 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 205.652 ms | 1.50 | 0.00 | 9% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.546 ms | 0.45 | 1.79 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.870 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1705.322 ms | 0.36 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.087 ms | 0.12 | 0.46 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 29.688 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2446.518 ms | 0.25 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.648 ms | 0.12 | 0.47 | 2% | ok |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 7.423 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 44.133 ms | 6.99 | 0.02 | 42% | ok |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 2.405 ms | 1.40 | 5.60 | 23% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.716 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 104.150 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 53.896 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 28.639 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.562 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 33.777 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 64.284 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 63.996 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.891 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 37.568 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.758 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.885 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 47.638 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.112 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 5.050 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 31.124 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.126 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 3.046 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.066 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.972 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.117 ms | 1.24 | 13.85 | 56% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 4.621 ms | 0.57 | 6.35 | 26% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.948 ms | 0.89 | 9.95 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 31.966 ms | 1.20 | 9.09 | 37% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 229.321 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 234.081 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 143.721 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.402 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 8.024 ms | 2.09 | 8.36 | 34% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 48.613 ms | 5.52 | 0.09 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.936 ms | 5.71 | 6.07 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.848 ms | 5.89 | 23.56 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.392 ms | 12.05 | 24.10 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.643 ms | 4.61 | 4.61 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.169 ms | 14.35 | 14.35 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 24.595 ms | 10.91 | 0.13 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 26.363 ms | 10.18 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 20.303 ms | 13.22 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 20.299 ms | 13.22 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 38.478 ms | 6.98 | 0.04 | n/a | ok |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

## arduino-ventuno-q -- blas, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bf315e4604cf` |
| FMA peak (ceiling) | f64 67.6 GFLOP/s, f32 134.9 GFLOP/s |
| ... hardware, NEON | f64 67.6, f32 134.9 GFLOP/s |
| ... this build, scalar | f64 33.7, f32 33.8 GFLOP/s |
| bandwidth | read 23.0, update 24.5, triad 22.5 GB/s (ceiling 24.5) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 99% | 109% | 0 | 28% |
| blas2 | 10 | 64% | 107% | 0 | 37% |
| blas3 | 10 | 75% | 81% | 0 | 88% |
| factor | 12 | 18% | 38% | 0 | 44% |
| eigen | 8 | 11% | 28% | 0 | 33% |
| elementwise | 5 | 6% | 8% | 0 | 26% |
| sparse | 5 | 57% | 93% | 0 | 28% |
| direct | 23 | 1% | 11% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 28% |
| direct3d | 15 | 2% | 24% | 0 | 27% |
| krylov | 13 | n/a | n/a | 0 | 27% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 45% | 61% | 0 | 27% |
| mg | 1 | 37% | 37% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 27% |
| mixed | 3 | 31% | 38% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 53% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.554 ms | 3.02 | 24.17 | 99% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.815 ms | 5.96 | 23.84 | 97% | ok | 1.6x (40%) |
| blas1 | axpy | f64 | n=8388608 | 8.272 ms | 2.03 | 24.34 | 99% | ok | 0.9x (24%) |
| blas1 | axpy | f32 | n=8388608 | 4.091 ms | 4.10 | 24.60 | 100% | ok | 1.0x (24%) |
| blas1 | scale | f64 | n=8388608 | 5.019 ms | 1.67 | 26.74 | 109% | ok | 1.1x (27%) |
| blas1 | two_norm | f64 | n=8388608 | 2.619 ms | 6.40 | 25.62 | 105% | ok | 1.2x (29%) |
| blas1 | one_norm | f64 | n=8388608 | 7.690 ms | 1.09 | 8.73 | 36% | ok | 1.1x (27%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.473 ms | 0.54 | 4.34 | 18% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.449 ms | 1.13 | 9.01 | 37% | ok | 1.1x (28%) |
| blas1 | dot_k3 | f64 | n=8388608 | 170.810 ms | 1.08 | 0.79 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 5.111 ms | 6.56 | 26.27 | 107% | ok | 1.8x (44%) |
| blas2 | gemv | f32 | n=4096 | 2.596 ms | 12.92 | 25.86 | 106% | ok | 1.5x (37%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 8.592 ms | 3.91 | 15.63 | 64% | ok | 1.1x (27%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.938 ms | 6.80 | 13.60 | 55% | ok | 0.8x (19%) |
| blas2 | symv | f64 | n=4096 | 6.494 ms | 5.17 | 10.34 | 42% | ok | 1.8x (44%) |
| blas2 | trmv | f64 | n=4096 | 3.660 ms | 4.58 | 18.36 | 75% | ok | 1.7x (43%) |
| blas2 | trmv | f32 | n=4096 | 2.204 ms | 7.61 | 15.24 | 62% | ok | 1.5x (37%) |
| blas2 | trsv | f64 | n=4096 | 5.994 ms | 2.80 | 11.21 | 46% | ok | 1.1x (27%) |
| blas2 | trsv | f32 | n=4096 | 3.078 ms | 5.45 | 10.91 | 45% | ok | 1.0x (26%) |
| blas2 | ger | f64 | n=4096 | 14.703 ms | 2.28 | 18.26 | 75% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 5.010 ms | 53.58 | 1.26 | 79% | ok | 3.5x (88%) |
| blas3 | gemm | f32 | 512x512x512 | 2.442 ms | 109.93 | 1.29 | 81% | ok | 3.5x (88%) |
| blas3 | gemmt | f64 | 512x512x512 | 3.412 ms | 39.33 | 1.84 | 58% | ok | 3.3x (83%) |
| blas3 | symm | f64 | 512x512x512 | 5.106 ms | 52.58 | 1.23 | 78% | ok | 3.5x (88%) |
| blas3 | syrk | f64 | 512x512x512 | 3.262 ms | 41.15 | 1.29 | 61% | ok | 3.0x (76%) |
| blas3 | syr2k | f64 | 512x512x512 | 5.739 ms | 46.77 | 1.10 | 69% | ok | 3.3x (84%) |
| blas3 | trmm | f64 | 512x512x512 | 2.516 ms | 53.34 | 2.08 | 79% | ok | 3.6x (90%) |
| blas3 | trsm | f64 | 512x512x512 | 2.657 ms | 50.51 | 1.97 | 75% | ok | 3.7x (91%) |
| blas3 | trsm | f32 | 512x512x512 | 1.446 ms | 92.79 | 1.81 | 69% | ok | 3.5x (87%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 29.514 ms | 0.18 | 1.44 | 6% | ok | 0.9x (22%) |
| factor | lu | f64 | 512x512 | 4.106 ms | 21.79 | 1.02 | 32% | ok | 2.2x (56%) |
| factor | lu | f32 | 512x512 | 3.135 ms | 28.54 | 0.67 | 21% | ok | 1.8x (44%) |
| factor | cholesky | f64 | 512x512 | 2.978 ms | 15.02 | 1.41 | 22% | ok | 1.8x (45%) |
| factor | cholesky | f32 | 512x512 | 2.003 ms | 22.34 | 1.05 | 17% | ok | 1.6x (40%) |
| factor | ldlt | f64 | 512x512 | 20.205 ms | 2.21 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 60.592 ms | 0.74 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 9.641 ms | 18.56 | 0.44 | 27% | ok | 2.0x (51%) |
| factor | lq | f64 | 512x512 | 36.377 ms | 4.92 | 0.12 | 7% | ok | 1.7x (42%) |
| factor | hessenberg | f64 | 512x512 | 199.692 ms | 2.24 | 0.02 | 3% | ok | 1.8x (44%) |
| factor | householder | f64 | 512x512 | 0.446 ms | 2.35 | 9.40 | 38% | ok | 3.5x (87%) |
| factor | inv | f64 | 512x512 | 116.185 ms | 2.31 | 0.04 | 3% | ok | 1.1x (28%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.381 ms | 12.12 | 0.57 | 18% | ok | 1.4x (35%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 5.379 ms | 4.16 | 0.19 | 6% | ok | 1.4x (34%) |
| eigen | eigen_symmetric | f64 | 256x256 | 87.133 ms | 1.73 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 539.104 ms | 0.28 | 0.00 | 0% | ok | 1.2x (29%) |
| eigen | eigenvalue | f64 | 256x256 | 37.224 ms | 4.51 | 0.03 | 7% | ok | 1.2x (30%) |
| eigen | schur | f64 | 256x256 | 47.925 ms | 8.75 | 0.02 | 13% | ok | 1.3x (33%) |
| eigen | svd | f64 | 256x256 | 19.602 ms | 18.83 | 0.05 | 28% | ok | 1.5x (38%) |
| eigen | expm | f64 | 256x256 | 23.763 ms | 10.59 | 0.04 | 16% | ok | 1.6x (40%) |
| eigen | sylvester | f64 | 256x256 | 136.135 ms | 7.39 | 0.01 | 11% | ok | 1.3x (32%) |
| elementwise | exp | f64 | n=8388608 | 81.017 ms | 0.10 | 1.66 | 7% | ok | 1.0x (26%) |
| elementwise | log | f64 | n=8388608 | 98.493 ms | 0.09 | 1.36 | 6% | ok | 1.1x (27%) |
| elementwise | sin | f64 | n=8388608 | 134.180 ms | 0.06 | 1.00 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 66.690 ms | 0.13 | 2.01 | 8% | ok | 1.0x (26%) |
| elementwise | erf | f64 | n=8388608 | 107.022 ms | 0.08 | 1.25 | 5% | ok | 1.0x (26%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.200 ms | 2.18 | 22.70 | 93% | ok | 1.4x (34%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.953 ms | 1.34 | 13.94 | 57% | ok | 1.1x (28%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.128 ms | 6.83 | 14.37 | 59% | ok | 2.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.584 ms | 0.28 | 0.99 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.951 ms | 0.22 | 1.86 | 8% | ok | 1.0x (25%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 163.315 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 52.088 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.461 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 93.877 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 128.229 ms | 0.17 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.730 ms | 0.56 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 185.655 ms | 0.01 | 0.00 | 0% | ok | 1.1x (27%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.782 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.088 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.633 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.837 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.336 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 155.405 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.304 ms | 0.33 | 1.31 | 5% | ok | 0.8x (20%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.645 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.353 ms | 1.08 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.156 ms | 0.35 | 1.40 | 6% | ok | 0.8x (20%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.060 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 80.173 ms | 0.28 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.706 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.238 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 109.607 ms | 0.20 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.694 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 14.788 ms | 0.76 | 0.05 | 1% | ok | 1.1x (28%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 3.503 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 9.942 ms | 1.13 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 1.099 ms | 0.69 | 2.75 | 11% | ok | 1.1x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 18.952 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 14667.743 ms | 0.02 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.976 ms | 0.42 | 1.69 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 41.845 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 187.033 ms | 1.65 | 0.00 | 2% | ok | 1.1x (27%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.789 ms | 0.43 | 1.73 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.881 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1583.298 ms | 0.39 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.524 ms | 0.12 | 0.47 | 2% | ok | 1.0x (25%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 26.417 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2308.999 ms | 0.27 | 0.00 | 0% | ok | 1.1x (26%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.194 ms | 0.12 | 0.48 | 2% | ok | 1.0x (25%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 6.695 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 28.052 ms | 11.00 | 0.03 | 16% | ok | 1.6x (39%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 2.286 ms | 1.47 | 5.90 | 24% | ok | 1.1x (26%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.027 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 96.507 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 49.051 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.637 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.606 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.856 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.863 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 58.505 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.967 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 34.039 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.683 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.890 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.356 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.102 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.644 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.746 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.852 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.779 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.065 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.700 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.973 ms | 1.33 | 14.86 | 61% | ok | 1.1x (27%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 4.286 ms | 0.61 | 6.84 | 28% | ok | 1.1x (27%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.677 ms | 0.98 | 10.96 | 45% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.201 ms | 1.19 | 9.03 | 37% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 209.027 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 214.362 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 135.143 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 12.006 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.220 ms | 2.32 | 9.29 | 38% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.781 ms | 21.00 | 0.33 | 31% | ok | 3.8x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.655 ms | 6.32 | 6.71 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.805 ms | 5.98 | 23.92 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.370 ms | 12.25 | 24.49 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.269 ms | 5.13 | 5.13 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.040 ms | 16.14 | 16.14 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.538 ms | 23.26 | 0.27 | n/a | ok | 2.1x (53%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 6.680 ms | 40.18 | 0.31 | n/a | ok | 3.9x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.140 ms | 52.23 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.138 ms | 52.24 | 0.31 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 9.723 ms | 27.61 | 0.16 | n/a | ok | 4.0x (99%) |

Skipped in this build:

- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

