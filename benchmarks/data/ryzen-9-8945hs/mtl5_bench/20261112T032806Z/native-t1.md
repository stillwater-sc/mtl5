## ryzen-9-8945hs -- native, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 78.4 GFLOP/s, f32 126.4 GFLOP/s |
| ... hardware, AVX-512F | f64 78.4, f32 126.4 GFLOP/s |
| ... this build, scalar | f64 1.9, f32 1.9 GFLOP/s |
| bandwidth | read 36.8, update 51.7, triad 37.8 GB/s (ceiling 51.7) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 20% | 60% | 0 |
| blas2 | 10 | 44% | 77% | 0 |
| blas3 | 10 | 12% | 21% | 0 |
| factor | 12 | 3% | 4% | 0 |
| eigen | 8 | 3% | 9% | 0 |
| elementwise | 5 | 7% | 10% | 0 |
| sparse | 5 | 22% | 56% | 0 |
| direct | 7 | 0% | 1% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 31% | 32% | 0 |
| mg | 1 | 29% | 29% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 11% | 20% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 12.291 ms | 1.37 | 10.92 | 21% | ok |
| blas1 | dot | f32 | n=8388608 | 12.257 ms | 1.37 | 5.47 | 11% | ok |
| blas1 | axpy | f64 | n=8388608 | 14.961 ms | 1.12 | 13.46 | 26% | ok |
| blas1 | axpy | f32 | n=8388608 | 14.906 ms | 1.13 | 6.75 | 13% | ok |
| blas1 | scale | f64 | n=8388608 | 4.296 ms | 1.95 | 31.24 | 60% | ok |
| blas1 | two_norm | f64 | n=8388608 | 11.731 ms | 1.43 | 5.72 | 11% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.391 ms | 1.31 | 10.50 | 20% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.405 ms | 1.31 | 10.48 | 20% | ok |
| blas1 | sum | f64 | n=8388608 | 6.391 ms | 1.31 | 10.50 | 20% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 109.635 ms | 1.68 | 1.22 | 2% | ok |
| blas2 | gemv | f64 | n=4096 | 3.382 ms | 9.92 | 39.71 | 77% | ok |
| blas2 | gemv | f32 | n=4096 | 2.967 ms | 11.31 | 22.63 | 44% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.779 ms | 4.31 | 17.26 | 33% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.538 ms | 13.22 | 26.45 | 51% | ok |
| blas2 | symv | f64 | n=4096 | 10.435 ms | 3.22 | 6.44 | 12% | ok |
| blas2 | trmv | f64 | n=4096 | 2.044 ms | 8.21 | 32.87 | 64% | ok |
| blas2 | trmv | f32 | n=4096 | 1.839 ms | 9.12 | 18.27 | 35% | ok |
| blas2 | trsv | f64 | n=4096 | 2.034 ms | 8.25 | 33.02 | 64% | ok |
| blas2 | trsv | f32 | n=4096 | 1.861 ms | 9.02 | 18.05 | 35% | ok |
| blas2 | ger | f64 | n=4096 | 16.267 ms | 2.06 | 16.51 | 32% | ok |
| blas3 | gemm | f64 | 512x512x512 | 20.011 ms | 13.41 | 0.31 | 17% | ok |
| blas3 | gemm | f32 | 512x512x512 | 9.957 ms | 26.96 | 0.32 | 21% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 13.757 ms | 9.76 | 0.46 | 12% | ok |
| blas3 | symm | f64 | 512x512x512 | 21.110 ms | 12.72 | 0.30 | 16% | ok |
| blas3 | syrk | f64 | 512x512x512 | 16.862 ms | 7.96 | 0.25 | 10% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 29.066 ms | 9.24 | 0.22 | 12% | ok |
| blas3 | trmm | f64 | 512x512x512 | 30.766 ms | 4.36 | 0.17 | 6% | ok |
| blas3 | trsm | f64 | 512x512x512 | 35.235 ms | 3.81 | 0.15 | 5% | ok |
| blas3 | trsm | f32 | 512x512x512 | 32.124 ms | 4.18 | 0.08 | 3% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.902 ms | 0.49 | 3.90 | 8% | ok |
| factor | lu | f64 | 512x512 | 40.693 ms | 2.20 | 0.10 | 3% | ok |
| factor | lu | f32 | 512x512 | 38.427 ms | 2.33 | 0.05 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 14.327 ms | 3.12 | 0.29 | 4% | ok |
| factor | cholesky | f32 | 512x512 | 13.712 ms | 3.26 | 0.15 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 14.412 ms | 3.10 | 0.29 | 4% | ok |
| factor | ldlt_bk | f64 | 512x512 | 203.839 ms | 0.22 | 0.02 | 0% | ok |
| factor | qr | f64 | 512x512 | 810.843 ms | 0.22 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 83.463 ms | 2.14 | 0.05 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 933.562 ms | 0.48 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.961 ms | 0.21 | 0.85 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 110.021 ms | 2.44 | 0.04 | 3% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 41.893 ms | 2.14 | 0.10 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 49.700 ms | 0.45 | 0.02 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 164.127 ms | 0.92 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 638.466 ms | 0.24 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 75.241 ms | 2.23 | 0.01 | 3% | ok |
| eigen | schur | f64 | 256x256 | 144.383 ms | 2.90 | 0.01 | 4% | ok |
| eigen | svd | f64 | 256x256 | 149.555 ms | 2.47 | 0.01 | 3% | ok |
| eigen | expm | f64 | 256x256 | 37.397 ms | 6.73 | 0.03 | 9% | ok |
| eigen | sylvester | f64 | 256x256 | 329.581 ms | 3.05 | 0.00 | 4% | ok |
| elementwise | exp | f64 | n=8388608 | 39.838 ms | 0.21 | 3.37 | 7% | ok |
| elementwise | log | f64 | n=8388608 | 52.323 ms | 0.16 | 2.57 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 48.041 ms | 0.17 | 2.79 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 26.584 ms | 0.32 | 5.05 | 10% | ok |
| elementwise | erf | f64 | n=8388608 | 38.837 ms | 0.22 | 3.46 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.932 ms | 2.81 | 29.22 | 56% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.156 ms | 1.21 | 12.63 | 24% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 7.913 ms | 5.29 | 11.13 | 22% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 22.429 ms | 0.15 | 0.51 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.960 ms | 1.02 | 8.49 | 16%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 138.306 ms | 0.08 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 39.556 ms | 0.28 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 22.580 ms | 0.50 | 0.03 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 67.183 ms | 0.33 | 0.01 | 0% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 91.196 ms | 0.25 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 30.980 ms | 0.72 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 95.396 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.575 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.975 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.502 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.880 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 49.762 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 54.686 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 38.497 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.068 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 27.404 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.309 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 64.560 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 42.778 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.174 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 29.857 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 28.791 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 32.173 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 39.159 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.046 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.935 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 28.281 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.260 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.162 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.028 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.807 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.776 ms | 1.47 | 16.51 | 32% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.829 ms | 1.43 | 16.03 | 31% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.221 ms | 1.18 | 13.21 | 26% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 19.376 ms | 1.98 | 15.00 | 29% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.561 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 58.794 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 66.987 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.700 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 11.733 ms | 1.43 | 5.72 | 11% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 36.662 ms | 7.32 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.725 ms | 9.73 | 10.33 | 20% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.276 ms | 3.92 | 15.69 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.213 ms | 5.22 | 10.44 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.727 ms | 9.71 | 9.71 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.727 ms | 9.71 | 9.71 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 9.867 ms | 27.21 | 0.32 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 38.326 ms | 7.00 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 38.661 ms | 6.94 | 0.04 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 38.664 ms | 6.94 | 0.04 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 34.464 ms | 7.79 | 0.05 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

