## arduino-ventuno-q -- native, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 67.7 GFLOP/s, f32 135.6 GFLOP/s |
| ... hardware, NEON | f64 67.7, f32 135.6 GFLOP/s |
| ... this build, scalar | f64 26.9, f32 33.7 GFLOP/s |
| bandwidth | read 22.8, update 24.6, triad 22.6 GB/s (ceiling 24.6) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 91% | 107% | 0 | 28% |
| blas2 | 10 | 54% | 73% | 0 | 28% |
| blas3 | 10 | 9% | 16% | 0 | 29% |
| factor | 12 | 2% | 16% | 0 | 28% |
| eigen | 8 | 4% | 9% | 0 | 28% |
| elementwise | 5 | 5% | 8% | 0 | 27% |
| sparse | 5 | 55% | 72% | 0 | 27% |
| direct | 7 | 0% | 1% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 28% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 28% |
| smoother | 3 | 53% | 62% | 0 | 27% |
| mg | 1 | 33% | 33% | 0 | 23% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 30% | 38% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 51% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.985 ms | 2.80 | 22.42 | 91% | ok | 0.9x (24%) |
| blas1 | dot | f32 | n=8388608 | 3.022 ms | 5.55 | 22.21 | 90% | ok | 1.4x (36%) |
| blas1 | axpy | f64 | n=8388608 | 8.305 ms | 2.02 | 24.24 | 99% | ok | 0.9x (24%) |
| blas1 | axpy | f32 | n=8388608 | 4.322 ms | 3.88 | 23.29 | 95% | ok | 0.9x (23%) |
| blas1 | scale | f64 | n=8388608 | 5.094 ms | 1.65 | 26.35 | 107% | ok | 1.0x (26%) |
| blas1 | two_norm | f64 | n=8388608 | 2.868 ms | 5.85 | 23.40 | 95% | ok | 1.1x (28%) |
| blas1 | one_norm | f64 | n=8388608 | 7.299 ms | 1.15 | 9.19 | 37% | ok | 1.1x (28%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.514 ms | 0.54 | 4.33 | 18% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.870 ms | 1.07 | 8.53 | 35% | ok | 1.1x (27%) |
| blas1 | dot_k3 | f64 | n=8388608 | 177.417 ms | 1.04 | 0.76 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 7.803 ms | 4.30 | 17.21 | 70% | ok | 1.2x (30%) |
| blas2 | gemv | f32 | n=4096 | 4.002 ms | 8.39 | 16.78 | 68% | ok | 1.8x (44%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 8.170 ms | 4.11 | 16.44 | 67% | ok | 1.2x (30%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.143 ms | 6.52 | 13.05 | 53% | ok | 0.9x (22%) |
| blas2 | symv | f64 | n=4096 | 16.181 ms | 2.07 | 4.15 | 17% | ok | 1.1x (27%) |
| blas2 | trmv | f64 | n=4096 | 5.076 ms | 3.31 | 13.23 | 54% | ok | 1.0x (25%) |
| blas2 | trmv | f32 | n=4096 | 6.115 ms | 2.74 | 5.49 | 22% | ok | 1.1x (28%) |
| blas2 | trsv | f64 | n=4096 | 5.052 ms | 3.32 | 13.30 | 54% | ok | 1.0x (26%) |
| blas2 | trsv | f32 | n=4096 | 18.653 ms | 0.90 | 1.80 | 7% | ok | 1.1x (28%) |
| blas2 | ger | f64 | n=4096 | 14.996 ms | 2.24 | 17.90 | 73% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 27.283 ms | 9.84 | 0.23 | 15% | ok | 1.9x (48%) |
| blas3 | gemm | f32 | 512x512x512 | 12.039 ms | 22.30 | 0.26 | 16% | ok | 2.1x (52%) |
| blas3 | gemmt | f64 | 512x512x512 | 36.081 ms | 3.72 | 0.17 | 5% | ok | 1.1x (27%) |
| blas3 | symm | f64 | 512x512x512 | 51.602 ms | 5.20 | 0.12 | 8% | ok | 1.0x (26%) |
| blas3 | syrk | f64 | 512x512x512 | 37.254 ms | 3.60 | 0.11 | 5% | ok | 1.1x (27%) |
| blas3 | syr2k | f64 | 512x512x512 | 75.659 ms | 3.55 | 0.08 | 5% | ok | 1.0x (26%) |
| blas3 | trmm | f64 | 512x512x512 | 22.329 ms | 6.01 | 0.23 | 9% | ok | 1.2x (29%) |
| blas3 | trsm | f64 | 512x512x512 | 22.892 ms | 5.86 | 0.23 | 9% | ok | 1.1x (29%) |
| blas3 | trsm | f32 | 512x512x512 | 10.829 ms | 12.39 | 0.24 | 9% | ok | 1.1x (28%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 25.407 ms | 0.21 | 1.67 | 7% | ok | 1.2x (30%) |
| factor | lu | f64 | 512x512 | 110.776 ms | 0.81 | 0.04 | 1% | ok | 0.2x (6%) |
| factor | lu | f32 | 512x512 | 90.738 ms | 0.99 | 0.02 | 1% | ok | 0.1x (4%) |
| factor | cholesky | f64 | 512x512 | 19.931 ms | 2.24 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 17.579 ms | 2.55 | 0.12 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 19.155 ms | 2.34 | 0.22 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 60.709 ms | 0.74 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 198.339 ms | 0.90 | 0.02 | 1% | ok | 1.3x (31%) |
| factor | lq | f64 | 512x512 | 129.678 ms | 1.38 | 0.03 | 2% | ok | 0.5x (12%) |
| factor | hessenberg | f64 | 512x512 | 438.860 ms | 1.02 | 0.01 | 2% | ok | 0.8x (20%) |
| factor | householder | f64 | 512x512 | 1.095 ms | 0.96 | 3.83 | 16% | ok | 1.4x (35%) |
| factor | inv | f64 | 512x512 | 223.001 ms | 1.20 | 0.02 | 2% | ok | 0.7x (17%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 104.699 ms | 0.85 | 0.04 | 1% | ok | 0.2x (5%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.514 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 87.746 ms | 1.72 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 533.152 ms | 0.28 | 0.00 | 0% | ok | 1.1x (28%) |
| eigen | eigenvalue | f64 | 256x256 | 60.552 ms | 2.77 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 134.448 ms | 3.12 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.690 ms | 1.79 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 42.363 ms | 5.94 | 0.02 | 9% | ok | 1.6x (40%) |
| eigen | sylvester | f64 | 256x256 | 319.788 ms | 3.15 | 0.00 | 5% | ok | 1.2x (29%) |
| elementwise | exp | f64 | n=8388608 | 80.467 ms | 0.10 | 1.67 | 7% | ok | 1.1x (27%) |
| elementwise | log | f64 | n=8388608 | 99.791 ms | 0.08 | 1.34 | 5% | ok | 1.1x (28%) |
| elementwise | sin | f64 | n=8388608 | 135.635 ms | 0.06 | 0.99 | 4% | ok | 1.1x (27%) |
| elementwise | sqrt | f64 | n=8388608 | 66.721 ms | 0.13 | 2.01 | 8% | ok | 1.1x (27%) |
| elementwise | erf | f64 | n=8388608 | 106.810 ms | 0.08 | 1.26 | 5% | ok | 1.1x (28%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.535 ms | 1.70 | 17.73 | 72% | ok | 1.1x (27%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.970 ms | 1.33 | 13.82 | 56% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.486 ms | 6.46 | 13.57 | 55% | ok | 1.9x (47%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.783 ms | 0.28 | 0.98 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 10.136 ms | 0.20 | 1.64 | 7% | ok | 0.9x (22%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 163.032 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 53.227 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.190 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 92.385 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 126.087 ms | 0.18 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.094 ms | 0.57 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 181.535 ms | 0.01 | 0.00 | 0% | ok | 1.1x (28%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.689 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 11.986 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.633 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.432 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 63.630 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 97.271 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.420 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 25.990 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.313 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.581 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.214 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.159 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.081 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.715 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.371 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.541 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.538 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.099 ms | 0.00 | 0.00 | n/a | ok | 1.1x (29%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.357 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilut | f64 | 128^2 grid | 27.176 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.776 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.696 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.062 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.555 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.936 ms | 1.35 | 15.15 | 62% | ok | 1.1x (27%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.264 ms | 1.16 | 12.95 | 53% | ok | 1.1x (27%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 4.512 ms | 0.58 | 6.50 | 26% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.993 ms | 1.07 | 8.08 | 33% | ok | 0.9x (23%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 209.179 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 243.163 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 151.555 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.892 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 7.202 ms | 2.33 | 9.32 | 38% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 13.187 ms | 20.36 | 0.32 | 30% | ok | 3.7x (92%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.637 ms | 6.36 | 6.76 | 27% | ok | 1.1x (28%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.809 ms | 5.97 | 23.89 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.365 ms | 12.29 | 24.58 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 3.271 ms | 5.13 | 5.13 | n/a | ok | 1.1x (28%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.044 ms | 16.07 | 16.07 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 12.094 ms | 22.20 | 0.26 | n/a | ok | 2.0x (51%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 7.463 ms | 35.97 | 0.28 | n/a | ok | 3.5x (88%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.909 ms | 45.43 | 0.27 | n/a | ok | 3.4x (86%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.879 ms | 45.66 | 0.27 | n/a | ok | 3.5x (86%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 10.475 ms | 25.63 | 0.15 | n/a | ok | 3.7x (92%) |

