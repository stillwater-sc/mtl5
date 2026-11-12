## jetson-orin -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 10.7 GFLOP/s, f32 21.5 GFLOP/s |
| ... hardware, NEON | f64 10.7, f32 21.5 GFLOP/s |
| ... this build, NEON | f64 10.7, f32 21.5 GFLOP/s |
| bandwidth | read 11.5, update 16.2, triad 14.8 GB/s (ceiling 16.2) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 73% | 131% | 0 |
| blas2 | 10 | 48% | 124% | 0 |
| blas3 | 10 | 50% | 61% | 0 |
| factor | 12 | 13% | 27% | 0 |
| eigen | 8 | 16% | 33% | 0 |
| elementwise | 5 | 8% | 15% | 0 |
| sparse | 5 | 31% | 74% | 0 |
| direct | 7 | 1% | 3% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 49% | 56% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 66% | 119% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.566 ms | 1.45 | 11.60 | 72% | ok |
| blas1 | dot | f32 | n=8388608 | 5.672 ms | 2.96 | 11.83 | 73% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.227 ms | 1.37 | 16.47 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 5.979 ms | 2.81 | 16.84 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 6.337 ms | 1.32 | 21.18 | 131% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.598 ms | 3.00 | 11.99 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.501 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.751 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.503 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 192.296 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 13.922 ms | 2.41 | 9.65 | 60% | ok |
| blas2 | gemv | f32 | n=4096 | 6.269 ms | 5.35 | 10.71 | 66% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 50.052 ms | 0.67 | 2.68 | 17% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 24.652 ms | 1.36 | 2.72 | 17% | ok |
| blas2 | symv | f64 | n=4096 | 22.082 ms | 1.52 | 3.04 | 19% | ok |
| blas2 | trmv | f64 | n=4096 | 9.049 ms | 1.85 | 7.42 | 46% | ok |
| blas2 | trmv | f32 | n=4096 | 4.330 ms | 3.87 | 7.76 | 48% | ok |
| blas2 | trsv | f64 | n=4096 | 9.282 ms | 1.81 | 7.24 | 45% | ok |
| blas2 | trsv | f32 | n=4096 | 4.264 ms | 3.93 | 7.88 | 49% | ok |
| blas2 | ger | f64 | n=4096 | 13.401 ms | 2.50 | 20.04 | 124% | ok |
| blas3 | gemm | f64 | 512x512x512 | 41.288 ms | 6.50 | 0.15 | 61% | ok |
| blas3 | gemm | f32 | 512x512x512 | 20.477 ms | 13.11 | 0.15 | 61% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 25.834 ms | 5.20 | 0.24 | 48% | ok |
| blas3 | symm | f64 | 512x512x512 | 42.560 ms | 6.31 | 0.15 | 59% | ok |
| blas3 | syrk | f64 | 512x512x512 | 49.944 ms | 2.69 | 0.08 | 25% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 92.722 ms | 2.90 | 0.07 | 27% | ok |
| blas3 | trmm | f64 | 512x512x512 | 24.957 ms | 5.38 | 0.21 | 50% | ok |
| blas3 | trsm | f64 | 512x512x512 | 25.768 ms | 5.21 | 0.20 | 49% | ok |
| blas3 | trsm | f32 | 512x512x512 | 12.595 ms | 10.66 | 0.21 | 50% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.633 ms | 0.55 | 4.41 | 27% | ok |
| factor | lu | f64 | 512x512 | 35.522 ms | 2.52 | 0.12 | 23% | ok |
| factor | lu | f32 | 512x512 | 20.248 ms | 4.42 | 0.10 | 21% | ok |
| factor | cholesky | f64 | 512x512 | 33.959 ms | 1.32 | 0.12 | 12% | ok |
| factor | cholesky | f32 | 512x512 | 30.034 ms | 1.49 | 0.07 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 32.294 ms | 1.39 | 0.13 | 13% | ok |
| factor | ldlt_bk | f64 | 512x512 | 85.848 ms | 0.52 | 0.05 | 5% | ok |
| factor | qr | f64 | 512x512 | 333.709 ms | 0.54 | 0.01 | 5% | ok |
| factor | lq | f64 | 512x512 | 92.005 ms | 1.95 | 0.05 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 470.798 ms | 0.95 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 1.995 ms | 0.53 | 2.10 | 13% | ok |
| factor | inv | f64 | 512x512 | 191.363 ms | 1.40 | 0.02 | 13% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 30.445 ms | 2.94 | 0.14 | 27% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.220 ms | 0.37 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 144.446 ms | 1.05 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 904.419 ms | 0.17 | 0.00 | 2% | ok |
| eigen | eigenvalue | f64 | 256x256 | 98.464 ms | 1.70 | 0.01 | 16% | ok |
| eigen | schur | f64 | 256x256 | 215.345 ms | 1.95 | 0.00 | 18% | ok |
| eigen | svd | f64 | 256x256 | 358.885 ms | 1.03 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 72.043 ms | 3.49 | 0.01 | 33% | ok |
| eigen | sylvester | f64 | 256x256 | 516.424 ms | 1.95 | 0.00 | 18% | ok |
| elementwise | exp | f64 | n=8388608 | 76.034 ms | 0.11 | 1.77 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.148 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 178.607 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.112 ms | 0.15 | 2.44 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 126.586 ms | 0.07 | 1.06 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.283 ms | 1.15 | 11.93 | 74% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.075 ms | 0.85 | 8.85 | 55% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.316 ms | 2.42 | 5.08 | 31% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.833 ms | 0.16 | 0.55 | 3% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.825 ms | 0.17 | 1.41 | 9% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 277.967 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 99.500 ms | 0.11 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.503 ms | 0.30 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 134.874 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 191.193 ms | 0.12 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 61.861 ms | 0.36 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 291.572 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.258 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 16.988 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.135 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 66.428 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.151 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 148.007 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 77.989 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.487 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.744 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.650 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.320 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.076 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.951 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.831 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.635 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.323 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 68.075 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.166 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.737 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 44.908 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.501 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.227 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.089 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.230 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.231 ms | 0.81 | 9.08 | 56% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.740 ms | 0.70 | 7.84 | 49% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.515 ms | 0.58 | 6.50 | 40% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 45.031 ms | 0.85 | 6.45 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 353.745 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 419.285 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 197.613 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.815 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 6.263 ms | 2.68 | 10.72 | 66% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 64.044 ms | 4.19 | 0.07 | 39% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.310 ms | 12.81 | 13.61 | 119% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.604 ms | 2.99 | 11.97 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.684 ms | 6.25 | 12.50 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.208 ms | 13.89 | 13.89 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.348 ms | 7.15 | 7.15 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 26.016 ms | 10.32 | 0.12 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 27.808 ms | 9.65 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 33.397 ms | 8.04 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 8.563 ms | 31.35 | 0.18 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 23.385 ms | 11.48 | 0.07 | n/a | ok |

