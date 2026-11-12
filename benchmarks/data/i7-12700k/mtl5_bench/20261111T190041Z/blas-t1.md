## i7-12700k -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.3 GFLOP/s |
| bandwidth | read 33.1, update 38.5, triad 32.5 GB/s (ceiling 38.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 44% | 98% | 0 |
| blas2 | 10 | 97% | 106% | 0 |
| blas3 | 10 | 81% | 94% | 0 |
| factor | 12 | 25% | 40% | 0 |
| eigen | 8 | 24% | 71% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 49% | 95% | 0 |
| direct | 7 | 1% | 2% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 62% | 76% | 0 |
| mg | 1 | 56% | 56% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 35% | 83% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.616 ms | 2.99 | 23.90 | 62% | ok |
| blas1 | dot | f32 | n=8388608 | 4.224 ms | 3.97 | 15.89 | 41% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.438 ms | 2.61 | 31.27 | 81% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.926 ms | 3.41 | 20.44 | 53% | ok |
| blas1 | scale | f64 | n=8388608 | 3.573 ms | 2.35 | 37.57 | 98% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.878 ms | 3.44 | 13.76 | 36% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.023 ms | 2.09 | 16.68 | 43% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.036 ms | 1.19 | 9.54 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 3.994 ms | 2.10 | 16.80 | 44% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 76.947 ms | 2.40 | 1.74 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.508 ms | 9.56 | 38.28 | 99% | ok |
| blas2 | gemv | f32 | n=4096 | 1.642 ms | 20.43 | 40.89 | 106% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.628 ms | 9.25 | 37.01 | 96% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.638 ms | 20.48 | 40.98 | 106% | ok |
| blas2 | symv | f64 | n=4096 | 1.914 ms | 17.53 | 35.09 | 91% | ok |
| blas2 | trmv | f64 | n=4096 | 1.856 ms | 9.04 | 36.20 | 94% | ok |
| blas2 | trmv | f32 | n=4096 | 0.943 ms | 17.79 | 35.62 | 93% | ok |
| blas2 | trsv | f64 | n=4096 | 1.795 ms | 9.35 | 37.43 | 97% | ok |
| blas2 | trsv | f32 | n=4096 | 0.876 ms | 19.15 | 38.34 | 100% | ok |
| blas2 | ger | f64 | n=4096 | 7.697 ms | 4.36 | 34.89 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.709 ms | 72.38 | 1.70 | 92% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.832 ms | 146.52 | 1.72 | 94% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.573 ms | 52.16 | 2.45 | 67% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.718 ms | 72.21 | 1.69 | 92% | ok |
| blas3 | syrk | f64 | 512x512x512 | 2.546 ms | 52.72 | 1.65 | 67% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 4.716 ms | 56.92 | 1.33 | 73% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.119 ms | 63.35 | 2.47 | 81% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.143 ms | 62.63 | 2.45 | 80% | ok |
| blas3 | trsm | f32 | 512x512x512 | 0.962 ms | 139.55 | 2.73 | 89% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.272 ms | 0.40 | 3.20 | 8% | ok |
| factor | lu | f64 | 512x512 | 3.248 ms | 27.55 | 1.29 | 35% | ok |
| factor | lu | f32 | 512x512 | 1.885 ms | 47.48 | 1.11 | 30% | ok |
| factor | cholesky | f64 | 512x512 | 2.051 ms | 21.81 | 2.04 | 28% | ok |
| factor | cholesky | f32 | 512x512 | 1.121 ms | 39.92 | 1.87 | 25% | ok |
| factor | ldlt | f64 | 512x512 | 7.444 ms | 6.01 | 0.56 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 46.117 ms | 0.97 | 0.09 | 1% | ok |
| factor | qr | f64 | 512x512 | 5.752 ms | 31.11 | 0.73 | 40% | ok |
| factor | lq | f64 | 512x512 | 25.999 ms | 6.88 | 0.16 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 270.923 ms | 1.65 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.638 ms | 0.64 | 2.56 | 7%\* | ok |
| factor | inv | f64 | 512x512 | 46.193 ms | 5.81 | 0.09 | 7% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 3.785 ms | 23.64 | 1.11 | 30% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.710 ms | 13.08 | 0.61 | 17% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 59.238 ms | 2.55 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 746.680 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 16.216 ms | 10.35 | 0.06 | 13% | ok |
| eigen | schur | f64 | 256x256 | 21.003 ms | 19.97 | 0.05 | 26% | ok |
| eigen | svd | f64 | 256x256 | 6.658 ms | 55.44 | 0.16 | 71% | ok |
| eigen | expm | f64 | 256x256 | 11.271 ms | 22.33 | 0.09 | 29% | ok |
| eigen | sylvester | f64 | 256x256 | 53.545 ms | 18.80 | 0.03 | 24% | ok |
| elementwise | exp | f64 | n=8388608 | 39.558 ms | 0.21 | 3.39 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 45.787 ms | 0.18 | 2.93 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 60.319 ms | 0.14 | 2.23 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 29.137 ms | 0.29 | 4.61 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 202.087 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.747 ms | 3.51 | 36.47 | 95% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.927 ms | 2.82 | 29.36 | 76% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.665 ms | 8.98 | 18.88 | 49% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.850 ms | 0.67 | 2.37 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.267 ms | 1.58 | 13.14 | 34%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 91.327 ms | 0.12 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.635 ms | 0.44 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.915 ms | 1.25 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 41.045 ms | 0.55 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 53.887 ms | 0.42 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.927 ms | 1.32 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 99.990 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.353 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.980 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.220 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.231 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 31.228 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 32.851 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 23.036 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.522 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 15.031 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 14.069 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.802 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.404 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.317 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 17.225 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.500 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 18.011 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.394 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.035 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.546 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.395 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.065 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 0.997 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.022 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.118 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.002 ms | 2.61 | 29.26 | 76% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.222 ms | 2.14 | 23.99 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.565 ms | 1.67 | 18.74 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.440 ms | 2.86 | 21.63 | 56% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 31.798 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 33.267 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 38.901 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 5.014 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.916 ms | 3.41 | 13.65 | 35% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 39.048 ms | 6.87 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.560 ms | 29.97 | 31.84 | 83%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.300 ms | 7.29 | 29.18 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.014 ms | 16.54 | 33.08 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.725 ms | 9.72 | 9.72 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.725 ms | 9.73 | 9.73 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.476 ms | 15.36 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.817 ms | 12.89 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.235 ms | 26.23 | 0.15 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.235 ms | 26.23 | 0.15 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 26.851 ms | 10.00 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

