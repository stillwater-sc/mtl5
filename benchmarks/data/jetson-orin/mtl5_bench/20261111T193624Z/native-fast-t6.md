## jetson-orin -- native-fast, 6 threads, class medium

| | |
|---|---|
| CPU | ARMv8 Processor rev 1 (v8l) (aarch64, 6 logical cores) |
| OS | Linux Ubuntu 22.04.5 LTS (5.15.148-tegra) |
| compiler | GCC 11.4.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD NEON |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 64.3 GFLOP/s, f32 128.6 GFLOP/s |
| ... hardware, NEON | f64 64.3, f32 128.4 GFLOP/s |
| ... this build, NEON | f64 64.3, f32 128.6 GFLOP/s |
| bandwidth | read 27.6, update 35.0, triad 32.7 GB/s (ceiling 35.0) |
| last-level cache | 2 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 93% | 134% | 0 | 37% |
| blas2 | 10 | 23% | 86% | 0 | 17% |
| blas3 | 10 | 19% | 51% | 0 | 38% |
| factor | 12 | 2% | 22% | 0 | 21% |
| eigen | 8 | 3% | 9% | 0 | 17% |
| elementwise | 5 | 4% | 7% | 0 | 17% |
| sparse | 5 | 25% | 95% | 0 | 17% |
| direct | 7 | 0% | 0% | 0 | 17% |
| ordering | 4 | n/a | n/a | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 22% | 26% | 0 | 17% |
| mg | 1 | 20% | 20% | 0 | 18% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 34% | 39% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 80% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 4.253 ms | 3.94 | 31.56 | 90% | ok | 2.7x (45%) |
| blas1 | dot | f32 | n=8388608 | 2.060 ms | 8.14 | 32.57 | 93% | ok | 2.7x (46%) |
| blas1 | axpy | f64 | n=8388608 | 5.502 ms | 3.05 | 36.59 | 105% | ok | 2.2x (37%) |
| blas1 | axpy | f32 | n=8388608 | 2.657 ms | 6.32 | 37.89 | 108% | ok | 2.3x (38%) |
| blas1 | scale | f64 | n=8388608 | 2.855 ms | 2.94 | 47.01 | 134% | ok | 2.2x (37%) |
| blas1 | two_norm | f64 | n=8388608 | 2.041 ms | 8.22 | 32.88 | 94% | ok | 2.7x (46%) |
| blas1 | one_norm | f64 | n=8388608 | 12.501 ms | 0.67 | 5.37 | 15% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 18.751 ms | 0.45 | 3.58 | 10% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 12.502 ms | 0.67 | 5.37 | 15% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 191.558 ms | 0.96 | 0.70 | 2% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 5.057 ms | 6.63 | 26.55 | 76% | ok | 2.8x (46%) |
| blas2 | gemv | f32 | n=4096 | 2.222 ms | 15.10 | 30.21 | 86% | ok | 3.0x (50%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 18.911 ms | 1.77 | 7.10 | 20% | ok | 2.7x (44%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 9.522 ms | 3.52 | 7.05 | 20% | ok | 2.6x (43%) |
| blas2 | symv | f64 | n=4096 | 21.393 ms | 1.57 | 3.14 | 9% | ok | 1.0x (17%) |
| blas2 | trmv | f64 | n=4096 | 9.044 ms | 1.86 | 7.43 | 21% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 4.257 ms | 3.94 | 7.89 | 23% | ok | 1.0x (16%) |
| blas2 | trsv | f64 | n=4096 | 9.577 ms | 1.75 | 7.01 | 20% | ok | 0.9x (16%) |
| blas2 | trsv | f32 | n=4096 | 4.200 ms | 3.99 | 8.00 | 23% | ok | 1.0x (16%) |
| blas2 | ger | f64 | n=4096 | 13.402 ms | 2.50 | 20.03 | 57% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 8.274 ms | 32.44 | 0.76 | 50% | ok | 5.0x (84%) |
| blas3 | gemm | f32 | 512x512x512 | 4.125 ms | 65.07 | 0.76 | 51% | ok | 5.0x (83%) |
| blas3 | gemmt | f64 | 512x512x512 | 7.709 ms | 17.41 | 0.82 | 27% | ok | 3.3x (56%) |
| blas3 | symm | f64 | 512x512x512 | 9.104 ms | 29.49 | 0.69 | 46% | ok | 4.6x (77%) |
| blas3 | syrk | f64 | 512x512x512 | 49.901 ms | 2.69 | 0.08 | 4% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 512x512x512 | 90.339 ms | 2.97 | 0.07 | 5% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 512x512x512 | 10.877 ms | 12.34 | 0.48 | 19% | ok | 2.3x (38%) |
| blas3 | trsm | f64 | 512x512x512 | 11.698 ms | 11.47 | 0.45 | 18% | ok | 2.2x (37%) |
| blas3 | trsm | f32 | 512x512x512 | 5.724 ms | 23.45 | 0.46 | 18% | ok | 2.2x (37%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.750 ms | 0.54 | 4.36 | 12% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 26.033 ms | 3.44 | 0.16 | 5% | ok | 1.4x (23%) |
| factor | lu | f32 | 512x512 | 16.179 ms | 5.53 | 0.13 | 4% | ok | 1.3x (21%) |
| factor | cholesky | f64 | 512x512 | 33.916 ms | 1.32 | 0.12 | 2% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 30.108 ms | 1.49 | 0.07 | 1% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 32.205 ms | 1.39 | 0.13 | 2% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 86.309 ms | 0.52 | 0.05 | 1% | ok | 1.0x (16%) |
| factor | qr | f64 | 512x512 | 209.831 ms | 0.85 | 0.02 | 1% | ok | 1.5x (26%) |
| factor | lq | f64 | 512x512 | 60.552 ms | 2.96 | 0.07 | 5% | ok | 1.5x (25%) |
| factor | hessenberg | f64 | 512x512 | 307.632 ms | 1.45 | 0.01 | 2% | ok | 1.5x (25%) |
| factor | householder | f64 | 512x512 | 0.542 ms | 1.94 | 7.74 | 22% | ok | 3.6x (59%) |
| factor | inv | f64 | 512x512 | 185.322 ms | 1.45 | 0.02 | 2% | ok | 1.0x (17%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 25.927 ms | 3.45 | 0.16 | 5% | ok | 1.2x (20%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.077 ms | 0.37 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 144.468 ms | 1.05 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 902.138 ms | 0.17 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 98.714 ms | 1.70 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 214.253 ms | 1.96 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 357.977 ms | 1.03 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 43.498 ms | 5.79 | 0.02 | 9% | ok | 1.7x (28%) |
| eigen | sylvester | f64 | 256x256 | 497.407 ms | 2.02 | 0.00 | 3% | ok | 1.0x (17%) |
| elementwise | exp | f64 | n=8388608 | 75.222 ms | 0.11 | 1.78 | 5% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 107.265 ms | 0.08 | 1.25 | 4% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 177.477 ms | 0.05 | 0.76 | 2% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 54.823 ms | 0.15 | 2.45 | 7% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 126.825 ms | 0.07 | 1.06 | 3% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.822 ms | 3.18 | 33.13 | 95% | ok | 2.8x (46%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.125 ms | 0.84 | 8.71 | 25% | ok | 1.0x (16%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 5.783 ms | 7.24 | 15.23 | 44% | ok | 3.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.460 ms | 0.16 | 0.56 | 2% | ok | 1.0x (16%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 11.346 ms | 0.18 | 1.47 | 4% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 277.387 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 98.910 ms | 0.11 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 37.297 ms | 0.30 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 136.520 ms | 0.16 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 192.055 ms | 0.12 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 74.029 ms | 0.30 | 0.01 | 0% | ok | 0.8x (14%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 290.624 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 12.100 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 17.072 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 1.144 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 66.264 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 106.080 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 146.893 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 77.955 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 42.457 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 50.575 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 49.611 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 89.143 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.964 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 49.752 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 54.702 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 52.609 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.169 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 68.535 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.167 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 6.733 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 44.961 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 4.483 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 4.215 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.090 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.222 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.188 ms | 0.82 | 9.20 | 26% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.738 ms | 0.70 | 7.85 | 22% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.512 ms | 0.58 | 6.50 | 19% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 42.461 ms | 0.91 | 6.85 | 20% | ok | 1.1x (18%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 350.561 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 411.241 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 193.162 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 20.812 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 6.260 ms | 2.68 | 10.72 | 31% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 12.277 ms | 21.86 | 0.34 | 34% | ok | 5.2x (87%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.295 ms | 12.96 | 13.77 | 39% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 5.662 ms | 2.96 | 11.85 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.695 ms | 6.22 | 12.45 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.221 ms | 13.75 | 13.75 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.350 ms | 7.14 | 7.14 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 5.145 ms | 52.18 | 0.61 | n/a | ok | 5.1x (84%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.310 ms | 50.55 | 0.39 | n/a | ok | 5.2x (87%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 6.465 ms | 41.52 | 0.24 | n/a | ok | 5.2x (86%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.777 ms | 151.08 | 0.89 | n/a | ok | 4.8x (80%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.543 ms | 59.09 | 0.35 | n/a | ok | 5.2x (86%) |

