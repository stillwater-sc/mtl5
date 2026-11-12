## jetson-orin -- blas, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 64.2 GFLOP/s, f32 128.4 GFLOP/s |
| ... hardware, NEON | f64 64.2, f32 128.4 GFLOP/s |
| ... this build, scalar | f64 32.1, f32 32.1 GFLOP/s |
| bandwidth | read 27.4, update 34.5, triad 33.0 GB/s (ceiling 34.5) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 92% | 140% | 0 | 42% |
| blas2 | 10 | 60% | 86% | 0 | 43% |
| blas3 | 10 | 72% | 88% | 0 | 86% |
| factor | 12 | 11% | 25% | 0 | 27% |
| eigen | 8 | 7% | 19% | 0 | 20% |
| elementwise | 5 | 4% | 7% | 0 | 17% |
| sparse | 5 | 26% | 96% | 0 | 18% |
| direct | 7 | 0% | 1% | 0 | 17% |
| ordering | 4 | n/a | n/a | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 23% | 28% | 0 | 17% |
| mg | 1 | 20% | 20% | 0 | 18% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 16% | 32% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 70% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.222 ms | 3.97 | 31.79 | 92% | ok | 2.8x (46%) |
| blas1 | dot | f32 | n=8388608 | 2.229 ms | 7.53 | 30.10 | 87% | ok | 3.0x (49%) |
| blas1 | axpy | f64 | n=8388608 | 4.922 ms | 3.41 | 40.90 | 119% | ok | 2.5x (42%) |
| blas1 | axpy | f32 | n=8388608 | 2.378 ms | 7.06 | 42.34 | 123% | ok | 2.5x (42%) |
| blas1 | scale | f64 | n=8388608 | 2.770 ms | 3.03 | 48.46 | 140% | ok | 2.3x (38%) |
| blas1 | two_norm | f64 | n=8388608 | 2.034 ms | 8.25 | 33.00 | 96% | ok | 2.8x (46%) |
| blas1 | one_norm | f64 | n=8388608 | 12.499 ms | 0.67 | 5.37 | 16% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.749 ms | 0.45 | 3.58 | 10% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.498 ms | 0.67 | 5.37 | 16% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 192.154 ms | 0.96 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 4.620 ms | 7.26 | 29.07 | 84% | ok | 2.7x (46%) |
| blas2 | gemv | f32 | n=4096 | 2.272 ms | 14.77 | 29.55 | 86% | ok | 2.6x (43%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 6.504 ms | 5.16 | 20.65 | 60% | ok | 2.0x (33%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 3.761 ms | 8.92 | 17.85 | 52% | ok | 1.6x (26%) |
| blas2 | symv | f64 | n=4096 | 5.256 ms | 6.38 | 12.78 | 37% | ok | 3.0x (51%) |
| blas2 | trmv | f64 | n=4096 | 2.927 ms | 5.73 | 22.95 | 66% | ok | 3.0x (49%) |
| blas2 | trmv | f32 | n=4096 | 1.442 ms | 11.63 | 23.29 | 67% | ok | 3.4x (56%) |
| blas2 | trsv | f64 | n=4096 | 8.547 ms | 1.96 | 7.86 | 23% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 4.868 ms | 3.45 | 6.90 | 20% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 13.411 ms | 2.50 | 20.02 | 58% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 4.817 ms | 55.72 | 1.31 | 87% | ok | 5.5x (92%) |
| blas3 | gemm | f32 | 512x512x512 | 2.470 ms | 108.66 | 1.27 | 85% | ok | 5.3x (89%) |
| blas3 | gemmt | f64 | 512x512x512 | 4.086 ms | 32.85 | 1.54 | 51% | ok | 4.1x (68%) |
| blas3 | symm | f64 | 512x512x512 | 4.755 ms | 56.45 | 1.32 | 88% | ok | 5.6x (94%) |
| blas3 | syrk | f64 | 512x512x512 | 3.592 ms | 37.36 | 1.17 | 58% | ok | 4.0x (67%) |
| blas3 | syr2k | f64 | 512x512x512 | 6.209 ms | 43.23 | 1.01 | 67% | ok | 4.5x (76%) |
| blas3 | trmm | f64 | 512x512x512 | 2.573 ms | 52.17 | 2.04 | 81% | ok | 5.3x (88%) |
| blas3 | trsm | f64 | 512x512x512 | 2.899 ms | 46.31 | 1.81 | 72% | ok | 5.2x (86%) |
| blas3 | trsm | f32 | 512x512x512 | 1.569 ms | 85.57 | 1.67 | 67% | ok | 4.8x (81%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.246 ms | 0.57 | 4.59 | 13% | ok | 1.1x (18%) |
| factor | lu | f64 | 512x512 | 5.679 ms | 15.76 | 0.74 | 25% | ok | 2.4x (40%) |
| factor | lu | f32 | 512x512 | 4.542 ms | 19.70 | 0.46 | 15% | ok | 1.9x (31%) |
| factor | cholesky | f64 | 512x512 | 4.181 ms | 10.70 | 1.00 | 17% | ok | 1.9x (31%) |
| factor | cholesky | f32 | 512x512 | 3.146 ms | 14.22 | 0.67 | 11% | ok | 1.6x (27%) |
| factor | ldlt | f64 | 512x512 | 32.224 ms | 1.39 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 83.669 ms | 0.53 | 0.05 | 1% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 13.457 ms | 13.30 | 0.31 | 21% | ok | 2.1x (35%) |
| factor | lq | f64 | 512x512 | 61.635 ms | 2.90 | 0.07 | 5% | ok | 1.5x (25%) |
| factor | hessenberg | f64 | 512x512 | 321.671 ms | 1.39 | 0.01 | 2% | ok | 1.5x (25%) |
| factor | householder | f64 | 512x512 | 0.544 ms | 1.93 | 7.71 | 22% | ok | 3.7x (61%) |
| factor | inv | f64 | 512x512 | 165.769 ms | 1.62 | 0.03 | 3% | ok | 1.0x (17%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 14.755 ms | 6.06 | 0.28 | 9% | ok | 1.2x (20%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 6.960 ms | 3.21 | 0.15 | 5% | ok | 1.5x (25%) |
| eigen | eigen_symmetric | f64 | 256x256 | 144.951 ms | 1.04 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 923.716 ms | 0.16 | 0.00 | 0% | ok | 1.0x (16%) |
| eigen | eigenvalue | f64 | 256x256 | 56.087 ms | 2.99 | 0.02 | 5% | ok | 1.1x (19%) |
| eigen | schur | f64 | 256x256 | 73.843 ms | 5.68 | 0.01 | 9% | ok | 1.2x (20%) |
| eigen | svd | f64 | 256x256 | 29.696 ms | 12.43 | 0.04 | 19% | ok | 1.5x (24%) |
| eigen | expm | f64 | 256x256 | 37.525 ms | 6.71 | 0.03 | 10% | ok | 1.5x (25%) |
| eigen | sylvester | f64 | 256x256 | 210.412 ms | 4.78 | 0.01 | 7% | ok | 1.2x (20%) |
| elementwise | exp | f64 | n=8388608 | 75.696 ms | 0.11 | 1.77 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 108.161 ms | 0.08 | 1.24 | 4% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 178.387 ms | 0.05 | 0.75 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 54.973 ms | 0.15 | 2.44 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 126.267 ms | 0.07 | 1.06 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.819 ms | 3.19 | 33.23 | 96% | ok | 2.8x (46%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.062 ms | 0.85 | 8.89 | 26% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 5.800 ms | 7.22 | 15.18 | 44% | ok | 3.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.569 ms | 0.16 | 0.56 | 2% | ok | 1.0x (16%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.168 ms | 0.18 | 1.49 | 4% | ok | 1.1x (18%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 280.166 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 101.441 ms | 0.11 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.359 ms | 0.30 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 131.332 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 190.234 ms | 0.12 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 61.993 ms | 0.36 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 287.318 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.184 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.341 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.128 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 65.800 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 105.510 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 147.132 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 77.895 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.363 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.695 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.514 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.319 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.112 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.829 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.729 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.610 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.773 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 68.763 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.166 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.680 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 44.929 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.498 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.230 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.090 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 4.078 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.005 ms | 0.87 | 9.76 | 28% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.737 ms | 0.70 | 7.85 | 23% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.520 ms | 0.58 | 6.49 | 19% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 42.106 ms | 0.91 | 6.90 | 20% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 349.169 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 354.940 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 192.255 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 21.039 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 12.501 ms | 1.34 | 5.37 | 16% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.966 ms | 20.70 | 0.32 | 32% | ok | 5.8x (97%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 5.603 ms | 2.99 | 3.18 | 9% | ok | 0.8x (14%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.592 ms | 3.00 | 12.00 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.678 ms | 6.27 | 12.53 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.643 ms | 2.97 | 2.97 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.758 ms | 9.54 | 9.54 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 9.065 ms | 29.61 | 0.35 | n/a | ok | 4.2x (70%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 7.079 ms | 37.92 | 0.30 | n/a | ok | 5.8x (97%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.485 ms | 48.94 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.478 ms | 49.01 | 0.29 | n/a | ok | 5.8x (97%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 10.299 ms | 26.07 | 0.15 | n/a | ok | 5.8x (97%) |

