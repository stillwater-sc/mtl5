## jetson-orin -- native, 1 thread, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 10.7 GFLOP/s, f32 21.5 GFLOP/s |
| ... hardware, NEON | f64 10.7, f32 21.5 GFLOP/s |
| ... this build, scalar | f64 5.4, f32 5.4 GFLOP/s |
| bandwidth | read 11.5, update 16.2, triad 14.8 GB/s (ceiling 16.2) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 72% | 131% | 0 |
| blas2 | 10 | 33% | 124% | 0 |
| blas3 | 10 | 33% | 33% | 0 |
| factor | 12 | 13% | 27% | 0 |
| eigen | 8 | 16% | 23% | 0 |
| elementwise | 5 | 8% | 15% | 0 |
| sparse | 5 | 31% | 73% | 0 |
| direct | 7 | 1% | 3% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 60% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 33% | 34% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.539 ms | 1.45 | 11.63 | 72% | ok |
| blas1 | dot | f32 | n=8388608 | 6.005 ms | 2.79 | 11.18 | 69% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.258 ms | 1.37 | 16.42 | 101% | ok |
| blas1 | axpy | f32 | n=8388608 | 5.964 ms | 2.81 | 16.88 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 6.336 ms | 1.32 | 21.18 | 131% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.605 ms | 2.99 | 11.97 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 18.750 ms | 0.45 | 3.58 | 22% | ok |
| blas1 | sum | f64 | n=8388608 | 12.500 ms | 0.67 | 5.37 | 33% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 192.367 ms | 0.96 | 0.70 | 9% | ok |
| blas2 | gemv | f64 | n=4096 | 21.711 ms | 1.55 | 6.18 | 38% | ok |
| blas2 | gemv | f32 | n=4096 | 12.416 ms | 2.70 | 5.41 | 33% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 16.004 ms | 2.10 | 8.39 | 52% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.275 ms | 5.35 | 10.70 | 66% | ok |
| blas2 | symv | f64 | n=4096 | 21.330 ms | 1.57 | 3.15 | 19% | ok |
| blas2 | trmv | f64 | n=4096 | 13.093 ms | 1.28 | 5.13 | 32% | ok |
| blas2 | trmv | f32 | n=4096 | 26.561 ms | 0.63 | 1.26 | 8% | ok |
| blas2 | trsv | f64 | n=4096 | 13.112 ms | 1.28 | 5.12 | 32% | ok |
| blas2 | trsv | f32 | n=4096 | 27.035 ms | 0.62 | 1.24 | 8% | ok |
| blas2 | ger | f64 | n=4096 | 13.406 ms | 2.50 | 20.03 | 124% | ok |
| blas3 | gemm | f64 | 512x512x512 | 75.215 ms | 3.57 | 0.08 | 33% | ok |
| blas3 | gemm | f32 | 512x512x512 | 38.166 ms | 7.03 | 0.08 | 33% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 49.261 ms | 2.72 | 0.13 | 25% | ok |
| blas3 | symm | f64 | 512x512x512 | 75.813 ms | 3.54 | 0.08 | 33% | ok |
| blas3 | syrk | f64 | 512x512x512 | 51.808 ms | 2.59 | 0.08 | 24% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 93.019 ms | 2.89 | 0.07 | 27% | ok |
| blas3 | trmm | f64 | 512x512x512 | 37.963 ms | 3.54 | 0.14 | 33% | ok |
| blas3 | trsm | f64 | 512x512x512 | 38.344 ms | 3.50 | 0.14 | 33% | ok |
| blas3 | trsm | f32 | 512x512x512 | 19.279 ms | 6.96 | 0.14 | 32% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.564 ms | 0.56 | 4.44 | 27% | ok |
| factor | lu | f64 | 512x512 | 36.144 ms | 2.48 | 0.12 | 23% | ok |
| factor | lu | f32 | 512x512 | 20.469 ms | 4.37 | 0.10 | 20% | ok |
| factor | cholesky | f64 | 512x512 | 31.825 ms | 1.41 | 0.13 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 30.100 ms | 1.49 | 0.07 | 7% | ok |
| factor | ldlt | f64 | 512x512 | 34.450 ms | 1.30 | 0.12 | 12% | ok |
| factor | ldlt_bk | f64 | 512x512 | 85.787 ms | 0.52 | 0.05 | 5% | ok |
| factor | qr | f64 | 512x512 | 334.500 ms | 0.53 | 0.01 | 5% | ok |
| factor | lq | f64 | 512x512 | 91.967 ms | 1.95 | 0.05 | 18% | ok |
| factor | hessenberg | f64 | 512x512 | 473.116 ms | 0.95 | 0.01 | 9% | ok |
| factor | householder | f64 | 512x512 | 2.013 ms | 0.52 | 2.08 | 13% | ok |
| factor | inv | f64 | 512x512 | 192.447 ms | 1.39 | 0.02 | 13% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 30.696 ms | 2.91 | 0.14 | 27% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.380 ms | 0.36 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 144.863 ms | 1.04 | 0.01 | 10% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 947.450 ms | 0.16 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 98.408 ms | 1.70 | 0.01 | 16% | ok |
| eigen | schur | f64 | 256x256 | 214.235 ms | 1.96 | 0.00 | 18% | ok |
| eigen | svd | f64 | 256x256 | 358.627 ms | 1.03 | 0.00 | 10% | ok |
| eigen | expm | f64 | 256x256 | 102.700 ms | 2.45 | 0.01 | 23% | ok |
| eigen | sylvester | f64 | 256x256 | 536.759 ms | 1.88 | 0.00 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 75.790 ms | 0.11 | 1.77 | 11% | ok |
| elementwise | log | f64 | n=8388608 | 108.000 ms | 0.08 | 1.24 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 178.948 ms | 0.05 | 0.75 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 55.073 ms | 0.15 | 2.44 | 15% | ok |
| elementwise | erf | f64 | n=8388608 | 126.536 ms | 0.07 | 1.06 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.298 ms | 1.14 | 11.85 | 73% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.063 ms | 0.85 | 8.89 | 55% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 17.596 ms | 2.38 | 5.00 | 31% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.012 ms | 0.16 | 0.57 | 4% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.815 ms | 0.17 | 1.41 | 9% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 278.019 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 99.276 ms | 0.11 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.576 ms | 0.30 | 0.02 | 3% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 129.032 ms | 0.17 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 184.741 ms | 0.12 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 62.381 ms | 0.36 | 0.01 | 3% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 290.265 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.118 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.450 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.134 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 66.791 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.085 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 148.114 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 77.906 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.588 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.776 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.796 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.785 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.108 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 50.066 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.847 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.707 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.636 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 68.819 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.167 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 6.829 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 45.473 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.510 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 4.250 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.090 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.235 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.991 ms | 0.87 | 9.80 | 60% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.735 ms | 0.70 | 7.85 | 48% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 7.751 ms | 0.34 | 3.78 | 23% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 44.854 ms | 0.86 | 6.48 | 40% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 351.669 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 410.617 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 201.091 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 21.148 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 12.501 ms | 1.34 | 5.37 | 33% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 75.423 ms | 3.56 | 0.06 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 4.566 ms | 3.67 | 3.90 | 34% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.731 ms | 2.93 | 11.71 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.729 ms | 6.15 | 12.30 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.643 ms | 2.97 | 2.97 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.758 ms | 9.54 | 9.54 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 37.858 ms | 7.09 | 0.08 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 41.098 ms | 6.53 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 31.863 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 31.874 ms | 8.42 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 60.131 ms | 4.46 | 0.03 | n/a | ok |

