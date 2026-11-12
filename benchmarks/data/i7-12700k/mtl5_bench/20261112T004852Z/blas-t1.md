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
| bandwidth | read 34.7, update 38.8, triad 32.8 GB/s (ceiling 38.8) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 44% | 96% | 0 |
| blas2 | 10 | 96% | 109% | 0 |
| blas3 | 10 | 81% | 94% | 0 |
| factor | 12 | 19% | 40% | 0 |
| eigen | 8 | 24% | 71% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 48% | 94% | 0 |
| direct | 7 | 1% | 2% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 62% | 76% | 0 |
| mg | 1 | 56% | 56% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 35% | 82% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.834 ms | 2.88 | 23.01 | 59% | ok |
| blas1 | dot | f32 | n=8388608 | 4.262 ms | 3.94 | 15.75 | 41% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.481 ms | 2.59 | 31.06 | 80% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.950 ms | 3.39 | 20.34 | 52% | ok |
| blas1 | scale | f64 | n=8388608 | 3.603 ms | 2.33 | 37.25 | 96% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.765 ms | 3.52 | 14.08 | 36% | ok |
| blas1 | one_norm | f64 | n=8388608 | 3.959 ms | 2.12 | 16.95 | 44% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.075 ms | 1.19 | 9.48 | 24% | ok |
| blas1 | sum | f64 | n=8388608 | 3.937 ms | 2.13 | 17.05 | 44% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 76.428 ms | 2.41 | 1.76 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.421 ms | 9.81 | 39.25 | 101% | ok |
| blas2 | gemv | f32 | n=4096 | 1.591 ms | 21.09 | 42.20 | 109% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.596 ms | 9.33 | 37.34 | 96% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.628 ms | 20.61 | 41.24 | 106% | ok |
| blas2 | symv | f64 | n=4096 | 1.895 ms | 17.71 | 35.45 | 91% | ok |
| blas2 | trmv | f64 | n=4096 | 1.853 ms | 9.06 | 36.26 | 93% | ok |
| blas2 | trmv | f32 | n=4096 | 0.955 ms | 17.58 | 35.19 | 91% | ok |
| blas2 | trsv | f64 | n=4096 | 1.806 ms | 9.29 | 37.20 | 96% | ok |
| blas2 | trsv | f32 | n=4096 | 0.914 ms | 18.35 | 36.74 | 95% | ok |
| blas2 | ger | f64 | n=4096 | 7.774 ms | 4.32 | 34.54 | 89% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.735 ms | 71.87 | 1.68 | 92% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.830 ms | 146.71 | 1.72 | 94% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.583 ms | 51.96 | 2.44 | 66% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.723 ms | 72.09 | 1.69 | 92% | ok |
| blas3 | syrk | f64 | 512x512x512 | 2.567 ms | 52.28 | 1.63 | 67% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 4.738 ms | 56.66 | 1.33 | 72% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.114 ms | 63.49 | 2.48 | 81% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.135 ms | 62.87 | 2.46 | 80% | ok |
| blas3 | trsm | f32 | 512x512x512 | 0.964 ms | 139.25 | 2.72 | 89% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.113 ms | 0.40 | 3.24 | 8% | ok |
| factor | lu | f64 | 512x512 | 3.293 ms | 27.17 | 1.27 | 35% | ok |
| factor | lu | f32 | 512x512 | 1.785 ms | 50.13 | 1.17 | 32% | ok |
| factor | cholesky | f64 | 512x512 | 2.068 ms | 21.63 | 2.03 | 28% | ok |
| factor | cholesky | f32 | 512x512 | 1.502 ms | 29.79 | 1.40 | 19% | ok |
| factor | ldlt | f64 | 512x512 | 7.464 ms | 5.99 | 0.56 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 45.771 ms | 0.98 | 0.09 | 1% | ok |
| factor | qr | f64 | 512x512 | 5.727 ms | 31.25 | 0.73 | 40% | ok |
| factor | lq | f64 | 512x512 | 25.900 ms | 6.91 | 0.16 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 266.529 ms | 1.68 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.643 ms | 0.64 | 2.55 | 7%\* | ok |
| factor | inv | f64 | 512x512 | 46.183 ms | 5.81 | 0.09 | 7% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 3.786 ms | 23.63 | 1.11 | 30% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.716 ms | 13.04 | 0.61 | 17% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 59.156 ms | 2.55 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 759.307 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 16.392 ms | 10.23 | 0.06 | 13% | ok |
| eigen | schur | f64 | 256x256 | 21.025 ms | 19.95 | 0.05 | 25% | ok |
| eigen | svd | f64 | 256x256 | 6.640 ms | 55.59 | 0.16 | 71% | ok |
| eigen | expm | f64 | 256x256 | 11.250 ms | 22.37 | 0.09 | 29% | ok |
| eigen | sylvester | f64 | 256x256 | 53.047 ms | 18.98 | 0.03 | 24% | ok |
| elementwise | exp | f64 | n=8388608 | 39.232 ms | 0.21 | 3.42 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 45.485 ms | 0.18 | 2.95 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 60.015 ms | 0.14 | 2.24 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 28.767 ms | 0.29 | 4.67 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 201.436 ms | 0.04 | 0.67 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.744 ms | 3.52 | 36.58 | 94% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.921 ms | 2.84 | 29.57 | 76% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.738 ms | 8.84 | 18.58 | 48% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.880 ms | 0.67 | 2.36 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.281 ms | 1.57 | 13.00 | 33%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 91.289 ms | 0.12 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.589 ms | 0.44 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.957 ms | 1.25 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.204 ms | 0.56 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 53.614 ms | 0.42 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.916 ms | 1.32 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 99.522 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.378 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.154 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.219 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.159 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 31.337 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 33.214 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 23.038 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.555 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 15.057 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 14.044 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.872 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.462 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.314 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 17.235 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.456 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.878 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.383 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.034 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.546 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.361 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.066 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 0.995 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.023 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.114 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.991 ms | 2.64 | 29.59 | 76% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.218 ms | 2.15 | 24.08 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.563 ms | 1.67 | 18.77 | 48% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.377 ms | 2.87 | 21.73 | 56% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 46.319 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 47.783 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 38.902 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 5.013 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.874 ms | 3.44 | 13.77 | 35% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 38.935 ms | 6.89 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.560 ms | 29.97 | 31.84 | 82%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.272 ms | 7.39 | 29.54 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.985 ms | 17.03 | 34.06 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.721 ms | 9.75 | 9.75 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.722 ms | 9.75 | 9.75 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.484 ms | 15.35 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.811 ms | 12.90 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.312 ms | 26.03 | 0.15 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.268 ms | 26.14 | 0.15 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 26.978 ms | 9.95 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

