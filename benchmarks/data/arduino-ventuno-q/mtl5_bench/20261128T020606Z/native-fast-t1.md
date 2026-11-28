## arduino-ventuno-q -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON: -march=armv8.2-a+dotprod+fp16+rcpc -mtune=cortex-a78c]`, SIMD NEON_WITHOUT_AES |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 16.8 GFLOP/s, f32 33.6 GFLOP/s |
| ... hardware, NEON | f64 16.8, f32 33.6 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 16.8, f32 33.6 GFLOP/s |
| bandwidth | read 22.2, update 24.4, triad 22.9 GB/s (ceiling 24.4) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 95% | 105% | 0 |
| blas2 | 10 | 66% | 73% | 0 |
| blas3 | 10 | 48% | 58% | 0 |
| factor | 12 | 13% | 31% | 0 |
| eigen | 8 | 15% | 32% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 30% | 69% | 0 |
| direct | 19 | 2% | 7% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 3% | 10% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 56% | 0 |
| mg | 1 | 37% | 37% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 68% | 133% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.742 ms | 2.92 | 23.37 | 96% | ok |
| blas1 | dot | f32 | n=8388608 | 2.892 ms | 5.80 | 23.21 | 95% | ok |
| blas1 | axpy | f64 | n=8388608 | 7.876 ms | 2.13 | 25.56 | 105% | ok |
| blas1 | axpy | f32 | n=8388608 | 3.986 ms | 4.21 | 25.25 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 5.384 ms | 1.56 | 24.93 | 102% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.096 ms | 5.42 | 21.68 | 89% | ok |
| blas1 | one_norm | f64 | n=8388608 | 8.427 ms | 1.00 | 7.96 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 17.666 ms | 0.47 | 3.80 | 16% | ok |
| blas1 | sum | f64 | n=8388608 | 8.121 ms | 1.03 | 8.26 | 34% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 199.284 ms | 0.93 | 0.67 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 8.302 ms | 4.04 | 16.18 | 66% | ok |
| blas2 | gemv | f32 | n=4096 | 4.065 ms | 8.25 | 16.52 | 68% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.833 ms | 4.28 | 17.14 | 70% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.928 ms | 8.54 | 17.09 | 70% | ok |
| blas2 | symv | f64 | n=4096 | 12.084 ms | 2.78 | 5.56 | 23% | ok |
| blas2 | trmv | f64 | n=4096 | 4.656 ms | 3.60 | 14.43 | 59% | ok |
| blas2 | trmv | f32 | n=4096 | 2.718 ms | 6.17 | 12.36 | 51% | ok |
| blas2 | trsv | f64 | n=4096 | 5.032 ms | 3.33 | 13.35 | 55% | ok |
| blas2 | trsv | f32 | n=4096 | 2.774 ms | 6.05 | 12.11 | 50% | ok |
| blas2 | ger | f64 | n=4096 | 15.008 ms | 2.24 | 17.89 | 73% | ok |
| blas3 | gemm | f64 | 512x512x512 | 28.393 ms | 9.45 | 0.22 | 56% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.747 ms | 19.53 | 0.23 | 58% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 17.972 ms | 7.47 | 0.35 | 44% | ok |
| blas3 | symm | f64 | 512x512x512 | 29.263 ms | 9.17 | 0.21 | 55% | ok |
| blas3 | syrk | f64 | 512x512x512 | 16.570 ms | 8.10 | 0.25 | 48% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 32.612 ms | 8.23 | 0.19 | 49% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.711 ms | 7.58 | 0.30 | 45% | ok |
| blas3 | trsm | f64 | 512x512x512 | 17.926 ms | 7.49 | 0.29 | 45% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.483 ms | 15.82 | 0.31 | 47% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 32.871 ms | 0.16 | 1.29 | 5% | ok |
| factor | lu | f64 | 512x512 | 24.091 ms | 3.71 | 0.17 | 22% | ok |
| factor | lu | f32 | 512x512 | 13.049 ms | 6.86 | 0.16 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 22.368 ms | 2.00 | 0.19 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 20.992 ms | 2.13 | 0.10 | 6% | ok |
| factor | ldlt | f64 | 512x512 | 21.279 ms | 2.10 | 0.20 | 13% | ok |
| factor | ldlt_bk | f64 | 512x512 | 10.860 ms | 4.12 | 0.39 | 25% | ok |
| factor | qr | f64 | 512x512 | 242.043 ms | 0.74 | 0.02 | 4% | ok |
| factor | lq | f64 | 512x512 | 60.992 ms | 2.93 | 0.07 | 17% | ok |
| factor | hessenberg | f64 | 512x512 | 339.701 ms | 1.32 | 0.01 | 8% | ok |
| factor | householder | f64 | 512x512 | 1.515 ms | 0.69 | 2.77 | 11% | ok |
| factor | inv | f64 | 512x512 | 147.623 ms | 1.82 | 0.03 | 11% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.948 ms | 5.28 | 0.25 | 31% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.854 ms | 0.56 | 0.03 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 96.990 ms | 1.56 | 0.01 | 9% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 590.375 ms | 0.26 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 67.545 ms | 2.48 | 0.02 | 15% | ok |
| eigen | schur | f64 | 256x256 | 151.006 ms | 2.78 | 0.01 | 17% | ok |
| eigen | svd | f64 | 256x256 | 230.606 ms | 1.60 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 47.504 ms | 5.30 | 0.02 | 32% | ok |
| eigen | sylvester | f64 | 256x256 | 362.446 ms | 2.78 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 89.350 ms | 0.09 | 1.50 | 6% | ok |
| elementwise | log | f64 | n=8388608 | 111.113 ms | 0.08 | 1.21 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 149.887 ms | 0.06 | 0.90 | 4% | ok |
| elementwise | sqrt | f64 | n=8388608 | 72.285 ms | 0.12 | 1.86 | 8% | ok |
| elementwise | erf | f64 | n=8388608 | 118.240 ms | 0.07 | 1.14 | 5% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.629 ms | 1.61 | 16.72 | 69% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.161 ms | 1.21 | 12.60 | 52% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 12.161 ms | 3.44 | 7.24 | 30% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 13.254 ms | 0.25 | 0.87 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.787 ms | 0.23 | 1.89 | 8% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 38.072 ms | 0.29 | 0.02 | 2% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 58.493 ms | 0.19 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.362 ms | 0.42 | 0.03 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 103.776 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 122.734 ms | 0.18 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 42.433 ms | 0.53 | 0.02 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 79.768 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.697 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 13.210 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 125.008 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.573 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.331 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.657 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 30.167 ms | 0.37 | 0.02 | 2% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.819 ms | 0.42 | 1.66 | 7% | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 12.260 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 11.865 ms | 0.94 | 0.06 | 6% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.702 ms | 0.44 | 1.78 | 7% | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.448 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 88.608 ms | 0.25 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.812 ms | 0.13 | 0.52 | 2% | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.178 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 103.692 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.872 ms | 0.13 | 0.52 | 2% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 12.039 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 675.339 ms | 0.46 | 0.00 | 3% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.362 ms | 0.46 | 1.83 | 7% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 27.794 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 187.140 ms | 1.65 | 0.00 | 10% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.396 ms | 0.46 | 1.82 | 7% | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.548 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1692.541 ms | 0.36 | 0.00 | 2% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.779 ms | 0.12 | 0.47 | 2% | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 16.801 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1194.101 ms | 0.52 | 0.00 | 3% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.283 ms | 0.12 | 0.48 | 2% | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 70.996 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 116.577 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 54.695 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 29.084 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 34.646 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 34.345 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 62.793 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 64.044 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.885 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 39.361 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 35.784 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 34.739 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 41.030 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.110 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.991 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 31.192 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.117 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.981 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.076 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.775 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.136 ms | 1.23 | 13.73 | 56% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.517 ms | 1.04 | 11.65 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.985 ms | 0.88 | 9.83 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 31.934 ms | 1.20 | 9.10 | 37% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 233.620 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 235.081 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 142.823 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 13.188 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.069 ms | 4.12 | 16.49 | 68% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 41.407 ms | 6.48 | 0.10 | 39% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.751 ms | 22.33 | 23.72 | 133% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.816 ms | 5.96 | 23.83 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.394 ms | 12.03 | 24.06 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.696 ms | 24.11 | 24.11 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.522 ms | 11.02 | 11.02 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.292 ms | 15.52 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 16.695 ms | 16.08 | 0.13 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 16.586 ms | 16.18 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.611 ms | 74.34 | 0.44 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.185 ms | 20.36 | 0.12 | n/a | ok |

