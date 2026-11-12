## jetson-orin -- native-fast, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 64.3 GFLOP/s, f32 128.6 GFLOP/s |
| ... hardware, NEON | f64 64.3, f32 128.6 GFLOP/s |
| ... this build, NEON | f64 64.3, f32 128.6 GFLOP/s |
| bandwidth | read 29.6, update 34.5, triad 32.8 GB/s (ceiling 34.5) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 94% | 137% | 0 | 40% |
| blas2 | 10 | 23% | 88% | 0 | 17% |
| blas3 | 10 | 19% | 51% | 0 | 38% |
| factor | 12 | 2% | 22% | 0 | 20% |
| eigen | 8 | 3% | 8% | 0 | 17% |
| elementwise | 5 | 4% | 7% | 0 | 17% |
| sparse | 5 | 25% | 91% | 0 | 18% |
| direct | 7 | 0% | 0% | 0 | 17% |
| ordering | 4 | n/a | n/a | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 23% | 27% | 0 | 17% |
| mg | 1 | 20% | 20% | 0 | 18% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 34% | 40% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 80% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.254 ms | 3.94 | 31.55 | 92% | ok | 2.7x (45%) |
| blas1 | dot | f32 | n=8388608 | 2.065 ms | 8.13 | 32.50 | 94% | ok | 2.7x (46%) |
| blas1 | axpy | f64 | n=8388608 | 5.015 ms | 3.35 | 40.15 | 116% | ok | 2.4x (41%) |
| blas1 | axpy | f32 | n=8388608 | 2.486 ms | 6.75 | 40.48 | 117% | ok | 2.4x (40%) |
| blas1 | scale | f64 | n=8388608 | 2.842 ms | 2.95 | 47.23 | 137% | ok | 2.2x (37%) |
| blas1 | two_norm | f64 | n=8388608 | 2.036 ms | 8.24 | 32.97 | 96% | ok | 2.8x (46%) |
| blas1 | one_norm | f64 | n=8388608 | 12.504 ms | 0.67 | 5.37 | 16% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.757 ms | 0.45 | 3.58 | 10% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.504 ms | 0.67 | 5.37 | 16% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 191.694 ms | 0.96 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 5.150 ms | 6.52 | 26.07 | 76% | ok | 2.7x (45%) |
| blas2 | gemv | f32 | n=4096 | 2.218 ms | 15.13 | 30.27 | 88% | ok | 2.8x (47%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 11.407 ms | 2.94 | 11.77 | 34% | ok | 4.4x (73%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 9.384 ms | 3.58 | 7.15 | 21% | ok | 2.6x (44%) |
| blas2 | symv | f64 | n=4096 | 21.446 ms | 1.56 | 3.13 | 9% | ok | 1.0x (17%) |
| blas2 | trmv | f64 | n=4096 | 8.978 ms | 1.87 | 7.48 | 22% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 4.285 ms | 3.92 | 7.84 | 23% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 9.055 ms | 1.85 | 7.42 | 22% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 4.176 ms | 4.02 | 8.04 | 23% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 13.362 ms | 2.51 | 20.09 | 58% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 8.212 ms | 32.69 | 0.77 | 51% | ok | 5.0x (84%) |
| blas3 | gemm | f32 | 512x512x512 | 4.110 ms | 65.31 | 0.77 | 51% | ok | 5.0x (83%) |
| blas3 | gemmt | f64 | 512x512x512 | 7.623 ms | 17.61 | 0.83 | 27% | ok | 3.4x (56%) |
| blas3 | symm | f64 | 512x512x512 | 9.158 ms | 29.31 | 0.69 | 46% | ok | 4.6x (77%) |
| blas3 | syrk | f64 | 512x512x512 | 50.996 ms | 2.63 | 0.08 | 4% | ok | 1.0x (16%) |
| blas3 | syr2k | f64 | 512x512x512 | 93.274 ms | 2.88 | 0.07 | 4% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 512x512x512 | 10.875 ms | 12.34 | 0.48 | 19% | ok | 2.3x (38%) |
| blas3 | trsm | f64 | 512x512x512 | 11.667 ms | 11.50 | 0.45 | 18% | ok | 2.2x (37%) |
| blas3 | trsm | f32 | 512x512x512 | 5.718 ms | 23.47 | 0.46 | 18% | ok | 2.2x (37%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.676 ms | 0.55 | 4.39 | 13% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 26.586 ms | 3.37 | 0.16 | 5% | ok | 1.3x (22%) |
| factor | lu | f32 | 512x512 | 16.603 ms | 5.39 | 0.13 | 4% | ok | 1.2x (20%) |
| factor | cholesky | f64 | 512x512 | 33.987 ms | 1.32 | 0.12 | 2% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 30.179 ms | 1.48 | 0.07 | 1% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 32.261 ms | 1.39 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 86.633 ms | 0.52 | 0.05 | 1% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 207.341 ms | 0.86 | 0.02 | 1% | ok | 1.6x (27%) |
| factor | lq | f64 | 512x512 | 61.112 ms | 2.93 | 0.07 | 5% | ok | 1.5x (25%) |
| factor | hessenberg | f64 | 512x512 | 304.859 ms | 1.47 | 0.01 | 2% | ok | 1.5x (26%) |
| factor | householder | f64 | 512x512 | 0.544 ms | 1.93 | 7.71 | 22% | ok | 3.7x (61%) |
| factor | inv | f64 | 512x512 | 182.763 ms | 1.47 | 0.02 | 2% | ok | 1.0x (17%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 26.118 ms | 3.43 | 0.16 | 5% | ok | 1.2x (19%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 60.996 ms | 0.37 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 144.305 ms | 1.05 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 900.812 ms | 0.17 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 98.476 ms | 1.70 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 213.700 ms | 1.96 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 358.027 ms | 1.03 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 50.135 ms | 5.02 | 0.02 | 8% | ok | 1.4x (24%) |
| eigen | sylvester | f64 | 256x256 | 499.925 ms | 2.01 | 0.00 | 3% | ok | 1.0x (17%) |
| elementwise | exp | f64 | n=8388608 | 75.211 ms | 0.11 | 1.78 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 107.315 ms | 0.08 | 1.25 | 4% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 177.101 ms | 0.05 | 0.76 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 54.810 ms | 0.15 | 2.45 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 126.133 ms | 0.07 | 1.06 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.864 ms | 3.03 | 31.51 | 91% | ok | 2.6x (44%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.127 ms | 0.84 | 8.71 | 25% | ok | 1.0x (16%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 5.613 ms | 7.46 | 15.69 | 46% | ok | 3.1x (51%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.211 ms | 0.16 | 0.57 | 2% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.158 ms | 0.18 | 1.49 | 4% | ok | 1.1x (18%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 277.428 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 99.051 ms | 0.11 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.314 ms | 0.30 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 135.416 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 190.845 ms | 0.12 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 72.865 ms | 0.31 | 0.01 | 0% | ok | 0.8x (14%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 288.246 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.072 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 16.933 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.145 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 65.173 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 105.783 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 147.396 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 77.779 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.495 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.559 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.397 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 88.718 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.118 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.658 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.824 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.407 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.156 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 68.233 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.166 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.732 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 44.856 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.504 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.228 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.090 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.254 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.197 ms | 0.82 | 9.17 | 27% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.737 ms | 0.70 | 7.85 | 23% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.514 ms | 0.58 | 6.50 | 19% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 42.313 ms | 0.91 | 6.87 | 20% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 351.319 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 411.405 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 199.138 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 21.055 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| mixed | dot | f32>f64 | n=8388608 | 6.259 ms | 2.68 | 10.72 | 31% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.278 ms | 21.86 | 0.34 | 34% | ok | 5.2x (87%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.292 ms | 12.99 | 13.80 | 40% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.633 ms | 2.98 | 11.91 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.648 ms | 6.34 | 12.67 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.199 ms | 14.00 | 14.00 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.348 ms | 7.15 | 7.15 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 5.155 ms | 52.08 | 0.61 | n/a | ok | 5.0x (84%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.316 ms | 50.50 | 0.39 | n/a | ok | 5.2x (87%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 6.473 ms | 41.47 | 0.24 | n/a | ok | 5.2x (86%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.782 ms | 150.65 | 0.88 | n/a | ok | 4.8x (80%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.562 ms | 58.84 | 0.34 | n/a | ok | 5.1x (85%) |

