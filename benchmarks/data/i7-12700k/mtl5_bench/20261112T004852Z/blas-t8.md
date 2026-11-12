## i7-12700k -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 600.9 GFLOP/s, f32 1201.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.9, f32 1201.6 GFLOP/s |
| ... this build, scalar | f64 10.1, f32 10.0 GFLOP/s |
| bandwidth | read 83.7, update 78.3, triad 76.3 GB/s (ceiling 83.7) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 84% | 122% | 0 | 28% |
| blas2 | 10 | 118% | 201% | 0 | 32% |
| blas3 | 10 | 66% | 85% | 0 | 78% |
| factor | 12 | 5% | 12% | 0 | 17% |
| eigen | 8 | 3% | 10% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 293% | 0 | 13% |
| direct | 7 | 0% | 0% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 36% | 0 | 13% |
| mg | 1 | 25% | 25% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 19% |
| mixed | 3 | 17% | 39% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 86% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.609 ms | 10.43 | 83.44 | 100% | ok | 3.6x (45%) |
| blas1 | dot | f32 | n=8388608 | 0.657 ms | 25.53 | 102.12 | 122% | ok | 6.5x (81%) |
| blas1 | axpy | f64 | n=8388608 | 2.850 ms | 5.89 | 70.65 | 84% | ok | 2.3x (28%) |
| blas1 | axpy | f32 | n=8388608 | 1.429 ms | 11.74 | 70.46 | 84% | ok | 3.5x (43%) |
| blas1 | scale | f64 | n=8388608 | 1.907 ms | 4.40 | 70.39 | 84% | ok | 1.9x (24%) |
| blas1 | two_norm | f64 | n=8388608 | 0.741 ms | 22.66 | 90.62 | 108% | ok | 6.4x (80%) |
| blas1 | one_norm | f64 | n=8388608 | 3.959 ms | 2.12 | 16.95 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.958 ms | 1.21 | 9.64 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.933 ms | 2.13 | 17.06 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 74.627 ms | 2.47 | 1.80 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.417 ms | 23.67 | 94.74 | 113% | ok | 2.4x (30%) |
| blas2 | gemv | f32 | n=4096 | 0.546 ms | 61.41 | 122.88 | 147% | ok | 2.9x (36%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.425 ms | 23.55 | 94.25 | 113% | ok | 2.5x (32%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.683 ms | 49.13 | 98.30 | 118% | ok | 2.4x (30%) |
| blas2 | symv | f64 | n=4096 | 0.630 ms | 53.25 | 106.61 | 127% | ok | 3.0x (38%) |
| blas2 | trmv | f64 | n=4096 | 0.603 ms | 27.82 | 111.40 | 133% | ok | 3.1x (38%) |
| blas2 | trmv | f32 | n=4096 | 0.199 ms | 84.20 | 168.56 | 201% ! | ok | 4.8x (60%) |
| blas2 | trsv | f64 | n=4096 | 1.765 ms | 9.51 | 38.07 | 46% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.888 ms | 18.89 | 37.82 | 45% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.761 ms | 4.32 | 34.60 | 41% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 0.539 ms | 497.70 | 11.66 | 83% | ok | 6.9x (87%) |
| blas3 | gemm | f32 | 512x512x512 | 0.263 ms | 1019.10 | 11.94 | 85% | ok | 6.9x (87%) |
| blas3 | gemmt | f64 | 512x512x512 | 0.702 ms | 191.22 | 8.96 | 32% | ok | 3.7x (46%) |
| blas3 | symm | f64 | 512x512x512 | 0.539 ms | 497.92 | 11.67 | 83% | ok | 6.9x (86%) |
| blas3 | syrk | f64 | 512x512x512 | 0.763 ms | 175.86 | 5.50 | 29% | ok | 3.4x (42%) |
| blas3 | syr2k | f64 | 512x512x512 | 1.119 ms | 239.97 | 5.62 | 40% | ok | 4.2x (53%) |
| blas3 | trmm | f64 | 512x512x512 | 0.340 ms | 395.18 | 15.44 | 66% | ok | 6.2x (78%) |
| blas3 | trsm | f64 | 512x512x512 | 0.337 ms | 398.34 | 15.56 | 66% | ok | 6.3x (79%) |
| blas3 | trsm | f32 | 512x512x512 | 0.159 ms | 841.55 | 16.44 | 70% | ok | 6.0x (76%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.915 ms | 0.41 | 3.29 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 2.059 ms | 43.45 | 2.04 | 7% | ok | 1.6x (20%) |
| factor | lu | f32 | 512x512 | 1.324 ms | 67.57 | 1.58 | 6% | ok | 1.3x (17%) |
| factor | cholesky | f64 | 512x512 | 1.552 ms | 28.83 | 2.70 | 5% | ok | 1.3x (17%) |
| factor | cholesky | f32 | 512x512 | 1.016 ms | 44.03 | 2.06 | 4% | ok | 1.5x (18%) |
| factor | ldlt | f64 | 512x512 | 7.399 ms | 6.05 | 0.57 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 55.560 ms | 0.81 | 0.08 | 0% | ok | 0.8x (10%) |
| factor | qr | f64 | 512x512 | 4.175 ms | 42.86 | 1.00 | 7% | ok | 1.4x (17%) |
| factor | lq | f64 | 512x512 | 17.820 ms | 10.04 | 0.24 | 2% | ok | 1.5x (18%) |
| factor | hessenberg | f64 | 512x512 | 186.018 ms | 2.41 | 0.02 | 0% | ok | 1.4x (18%) |
| factor | householder | f64 | 512x512 | 0.433 ms | 2.43 | 9.69 | 12%\* | ok | 3.8x (47%) |
| factor | inv | f64 | 512x512 | 46.443 ms | 5.78 | 0.09 | 1% | ok | 1.0x (12%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 2.723 ms | 32.86 | 1.54 | 5% | ok | 1.4x (17%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.640 ms | 13.64 | 0.64 | 2% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 58.679 ms | 2.57 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 784.279 ms | 0.19 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 256x256 | 17.379 ms | 9.65 | 0.06 | 2% | ok | 0.9x (12%) |
| eigen | schur | f64 | 256x256 | 21.867 ms | 19.18 | 0.05 | 3% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 6.192 ms | 59.60 | 0.17 | 10% | ok | 1.1x (13%) |
| eigen | expm | f64 | 256x256 | 8.863 ms | 28.40 | 0.12 | 5% | ok | 1.3x (16%) |
| eigen | sylvester | f64 | 256x256 | 55.482 ms | 18.14 | 0.03 | 3% | ok | 1.0x (12%) |
| elementwise | exp | f64 | n=8388608 | 38.285 ms | 0.22 | 3.51 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 44.850 ms | 0.19 | 2.99 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 58.355 ms | 0.14 | 2.30 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.086 ms | 0.30 | 4.78 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 199.494 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.111 ms | 23.53 | 244.76 | 293% ! | ok | 6.7x (84%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.907 ms | 2.89 | 30.02 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.409 ms | 29.72 | 62.48 | 75% | ok | 3.4x (42%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.826 ms | 0.68 | 2.38 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.408 ms | 1.43 | 11.82 | 14%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 89.857 ms | 0.12 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.280 ms | 0.44 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 9.016 ms | 1.24 | 0.08 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.178 ms | 0.57 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 52.380 ms | 0.43 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.751 ms | 1.34 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 86.319 ms | 0.02 | 0.00 | 0% | ok | 1.2x (14%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.279 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.936 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.226 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 16.776 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.460 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 32.040 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.381 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.264 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.657 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.715 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.610 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.669 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.931 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 16.935 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.133 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.564 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 20.853 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.032 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.550 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 12.027 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.052 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.984 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.116 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.973 ms | 2.69 | 30.14 | 36% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.198 ms | 2.18 | 24.48 | 29% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.536 ms | 1.70 | 19.10 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.934 ms | 2.76 | 20.86 | 25% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.445 ms | 0.00 | 0.00 | n/a | ok | 1.6x (20%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 30.701 ms | 0.00 | 0.00 | n/a | ok | 1.6x (19%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 37.426 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.888 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 4.858 ms | 3.45 | 13.81 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 5.186 ms | 51.77 | 0.81 | 9% | ok | 7.5x (94%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.548 ms | 30.62 | 32.53 | 39%\* | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.223 ms | 7.55 | 30.18 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.971 ms | 17.28 | 34.57 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.690 ms | 9.93 | 9.93 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.686 ms | 9.95 | 9.95 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.535 ms | 105.87 | 1.24 | n/a | ok | 6.9x (86%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.829 ms | 94.90 | 0.74 | n/a | ok | 7.4x (92%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.356 ms | 197.91 | 1.16 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.357 ms | 197.77 | 1.16 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.617 ms | 74.22 | 0.43 | n/a | ok | 7.5x (93%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

