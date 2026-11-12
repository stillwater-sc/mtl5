## i7-12700k -- native, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.4 GFLOP/s |
| bandwidth | read 34.9, update 39.0, triad 33.0 GB/s (ceiling 39.0) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 44% | 96% | 0 |
| blas2 | 10 | 78% | 93% | 0 |
| blas3 | 10 | 17% | 18% | 0 |
| factor | 12 | 8% | 14% | 0 |
| eigen | 8 | 4% | 11% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 48% | 93% | 0 |
| direct | 7 | 1% | 2% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 62% | 76% | 0 |
| mg | 1 | 56% | 56% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 35% | 84% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.683 ms | 2.95 | 23.62 | 61% | ok |
| blas1 | dot | f32 | n=8388608 | 4.296 ms | 3.90 | 15.62 | 40% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.373 ms | 2.63 | 31.59 | 81% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.718 ms | 3.56 | 21.34 | 55% | ok |
| blas1 | scale | f64 | n=8388608 | 3.597 ms | 2.33 | 37.32 | 96% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.792 ms | 3.50 | 14.00 | 36% | ok |
| blas1 | one_norm | f64 | n=8388608 | 3.981 ms | 2.11 | 16.86 | 43% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.092 ms | 1.18 | 9.46 | 24% | ok |
| blas1 | sum | f64 | n=8388608 | 3.947 ms | 2.13 | 17.00 | 44% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 76.426 ms | 2.41 | 1.76 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.701 ms | 9.07 | 36.29 | 93% | ok |
| blas2 | gemv | f32 | n=4096 | 2.726 ms | 12.31 | 24.63 | 63% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 4.643 ms | 7.23 | 28.92 | 74% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.199 ms | 15.26 | 30.54 | 78% | ok |
| blas2 | symv | f64 | n=4096 | 6.543 ms | 5.13 | 10.27 | 26% | ok |
| blas2 | trmv | f64 | n=4096 | 2.031 ms | 8.26 | 33.08 | 85% | ok |
| blas2 | trmv | f32 | n=4096 | 1.456 ms | 11.52 | 23.07 | 59% | ok |
| blas2 | trsv | f64 | n=4096 | 2.011 ms | 8.34 | 33.40 | 86% | ok |
| blas2 | trsv | f32 | n=4096 | 1.417 ms | 11.84 | 23.70 | 61% | ok |
| blas2 | ger | f64 | n=4096 | 7.773 ms | 4.32 | 34.54 | 89% | ok |
| blas3 | gemm | f64 | 512x512x512 | 20.145 ms | 13.33 | 0.31 | 17% | ok |
| blas3 | gemm | f32 | 512x512x512 | 10.639 ms | 25.23 | 0.30 | 16% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 11.220 ms | 11.96 | 0.56 | 15% | ok |
| blas3 | symm | f64 | 512x512x512 | 19.468 ms | 13.79 | 0.32 | 18% | ok |
| blas3 | syrk | f64 | 512x512x512 | 13.034 ms | 10.30 | 0.32 | 13% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 23.720 ms | 11.32 | 0.27 | 14% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.631 ms | 13.94 | 0.54 | 18% | ok |
| blas3 | trsm | f64 | 512x512x512 | 10.163 ms | 13.21 | 0.52 | 17% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.164 ms | 25.99 | 0.51 | 17% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.985 ms | 0.41 | 3.27 | 8% | ok |
| factor | lu | f64 | 512x512 | 12.069 ms | 7.41 | 0.35 | 9% | ok |
| factor | lu | f32 | 512x512 | 3.982 ms | 22.47 | 0.53 | 14% | ok |
| factor | cholesky | f64 | 512x512 | 7.459 ms | 6.00 | 0.56 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.602 ms | 6.78 | 0.32 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.576 ms | 5.91 | 0.55 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 50.663 ms | 0.88 | 0.08 | 1% | ok |
| factor | qr | f64 | 512x512 | 223.044 ms | 0.80 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 25.998 ms | 6.88 | 0.16 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 267.225 ms | 1.67 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.653 ms | 0.64 | 2.54 | 7%\* | ok |
| factor | inv | f64 | 512x512 | 54.828 ms | 4.90 | 0.08 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 9.158 ms | 9.77 | 0.46 | 12% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 30.941 ms | 0.72 | 0.03 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 101.985 ms | 1.48 | 0.01 | 2% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1111.295 ms | 0.14 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 51.696 ms | 3.25 | 0.02 | 4% | ok |
| eigen | schur | f64 | 256x256 | 145.661 ms | 2.88 | 0.01 | 4% | ok |
| eigen | svd | f64 | 256x256 | 111.270 ms | 3.32 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 28.303 ms | 8.89 | 0.04 | 11% | ok |
| eigen | sylvester | f64 | 256x256 | 191.941 ms | 5.24 | 0.01 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 39.085 ms | 0.21 | 3.43 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 45.516 ms | 0.18 | 2.95 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 59.704 ms | 0.14 | 2.25 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 28.580 ms | 0.29 | 4.70 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 201.160 ms | 0.04 | 0.67 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.750 ms | 3.49 | 36.29 | 93% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.926 ms | 2.83 | 29.42 | 75% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.679 ms | 8.95 | 18.82 | 48% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.893 ms | 0.67 | 2.35 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.298 ms | 1.55 | 12.82 | 33%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 91.472 ms | 0.12 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.729 ms | 0.43 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.989 ms | 1.24 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.285 ms | 0.56 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 53.782 ms | 0.42 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 17.092 ms | 1.31 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 99.296 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.477 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.089 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.230 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.593 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 31.362 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 34.040 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 23.139 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.599 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 15.154 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 14.129 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 28.099 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.529 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.367 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 17.287 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.513 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.947 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.558 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.034 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.583 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.380 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.071 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.004 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.022 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.694 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.989 ms | 2.65 | 29.64 | 76% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.222 ms | 2.14 | 24.01 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.565 ms | 1.67 | 18.74 | 48% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.425 ms | 2.86 | 21.65 | 56% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 30.053 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 44.413 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 40.163 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 5.031 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.884 ms | 3.44 | 13.74 | 35% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 38.972 ms | 6.89 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.547 ms | 30.70 | 32.62 | 84%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.262 ms | 7.42 | 29.67 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.982 ms | 17.08 | 34.16 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.723 ms | 9.74 | 9.74 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.723 ms | 9.74 | 9.74 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.458 ms | 15.38 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.812 ms | 12.90 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.232 ms | 26.23 | 0.15 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.230 ms | 26.24 | 0.15 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 26.814 ms | 10.01 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

