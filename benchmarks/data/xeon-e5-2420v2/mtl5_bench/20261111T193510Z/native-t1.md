## xeon-e5-2420v2 -- native, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 20.9 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 20.9 GFLOP/s |
| ... this build, scalar | f64 0.1, f32 0.6 GFLOP/s |
| bandwidth | read 10.2, update 10.2, triad 8.0 GB/s (ceiling 10.2) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 24% | 106% | 0 |
| blas2 | 10 | 94% | 100% | 0 |
| blas3 | 10 | 40% | 45% | 0 |
| factor | 12 | 17% | 39% | 0 |
| eigen | 8 | 15% | 28% | 0 |
| elementwise | 5 | 9% | 14% | 0 |
| sparse | 5 | 57% | 108% | 0 |
| direct | 7 | 2% | 4% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 79% | 92% | 0 |
| mg | 1 | 79% | 79% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 23% | 78% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 184.501 ms | 0.09 | 0.73 | 7% | ok |
| blas1 | dot | f32 | n=8388608 | 27.429 ms | 0.61 | 2.45 | 24% | ok |
| blas1 | axpy | f64 | n=8388608 | 218.067 ms | 0.08 | 0.92 | 9% | ok |
| blas1 | axpy | f32 | n=8388608 | 91.685 ms | 0.18 | 1.10 | 11% | ok |
| blas1 | scale | f64 | n=8388608 | 12.459 ms | 0.67 | 10.77 | 106% | ok |
| blas1 | two_norm | f64 | n=8388608 | 185.147 ms | 0.09 | 0.36 | 4% | ok |
| blas1 | one_norm | f64 | n=8388608 | 9.757 ms | 0.86 | 6.88 | 67% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 13.076 ms | 0.64 | 5.13 | 50% | ok |
| blas1 | sum | f64 | n=8388608 | 9.721 ms | 0.86 | 6.90 | 68% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 281.893 ms | 0.65 | 0.48 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 13.142 ms | 2.55 | 10.22 | 100% | ok |
| blas2 | gemv | f32 | n=4096 | 8.246 ms | 4.07 | 8.14 | 80% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 13.398 ms | 2.50 | 10.02 | 98% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.614 ms | 5.07 | 10.15 | 99% | ok |
| blas2 | symv | f64 | n=4096 | 16.147 ms | 2.08 | 4.16 | 41% | ok |
| blas2 | trmv | f64 | n=4096 | 7.017 ms | 2.39 | 9.57 | 94% | ok |
| blas2 | trmv | f32 | n=4096 | 4.556 ms | 3.68 | 7.37 | 72% | ok |
| blas2 | trsv | f64 | n=4096 | 6.840 ms | 2.45 | 9.82 | 96% | ok |
| blas2 | trsv | f32 | n=4096 | 4.329 ms | 3.88 | 7.76 | 76% | ok |
| blas2 | ger | f64 | n=4096 | 28.119 ms | 1.19 | 9.55 | 94% | ok |
| blas3 | gemm | f64 | 512x512x512 | 64.296 ms | 4.17 | 0.10 | 40% | ok |
| blas3 | gemm | f32 | 512x512x512 | 34.799 ms | 7.71 | 0.09 | 37% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 28.665 ms | 4.68 | 0.22 | 45% | ok |
| blas3 | symm | f64 | 512x512x512 | 59.937 ms | 4.48 | 0.10 | 43% | ok |
| blas3 | syrk | f64 | 512x512x512 | 37.686 ms | 3.56 | 0.11 | 34% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 64.406 ms | 4.17 | 0.10 | 40% | ok |
| blas3 | trmm | f64 | 512x512x512 | 33.164 ms | 4.05 | 0.16 | 39% | ok |
| blas3 | trsm | f64 | 512x512x512 | 32.472 ms | 4.13 | 0.16 | 39% | ok |
| blas3 | trsm | f32 | 512x512x512 | 16.230 ms | 8.27 | 0.16 | 40% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.832 ms | 0.14 | 1.09 | 11% | ok |
| factor | lu | f64 | 512x512 | 30.799 ms | 2.91 | 0.14 | 28% | ok |
| factor | lu | f32 | 512x512 | 16.527 ms | 5.41 | 0.13 | 26% | ok |
| factor | cholesky | f64 | 512x512 | 24.842 ms | 1.80 | 0.17 | 17% | ok |
| factor | cholesky | f32 | 512x512 | 23.592 ms | 1.90 | 0.09 | 9% | ok |
| factor | ldlt | f64 | 512x512 | 24.844 ms | 1.80 | 0.17 | 17% | ok |
| factor | ldlt_bk | f64 | 512x512 | 150.471 ms | 0.30 | 0.03 | 3% | ok |
| factor | qr | f64 | 512x512 | 524.547 ms | 0.34 | 0.01 | 3% | ok |
| factor | lq | f64 | 512x512 | 73.535 ms | 2.43 | 0.06 | 23% | ok |
| factor | hessenberg | f64 | 512x512 | 633.534 ms | 0.71 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.210 ms | 0.33 | 1.31 | 13%\* | ok |
| factor | inv | f64 | 512x512 | 151.234 ms | 1.77 | 0.03 | 17% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 21.708 ms | 4.12 | 0.19 | 39% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 62.064 ms | 0.36 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 172.766 ms | 0.87 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1000.721 ms | 0.15 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 108.517 ms | 1.55 | 0.01 | 15% | ok |
| eigen | schur | f64 | 256x256 | 266.236 ms | 1.58 | 0.00 | 15% | ok |
| eigen | svd | f64 | 256x256 | 301.331 ms | 1.22 | 0.00 | 12% | ok |
| eigen | expm | f64 | 256x256 | 85.762 ms | 2.93 | 0.01 | 28% | ok |
| eigen | sylvester | f64 | 256x256 | 609.242 ms | 1.65 | 0.00 | 16% | ok |
| elementwise | exp | f64 | n=8388608 | 151.803 ms | 0.06 | 0.88 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 150.937 ms | 0.06 | 0.89 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 172.986 ms | 0.05 | 0.78 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 90.716 ms | 0.09 | 1.48 | 14% | ok |
| elementwise | erf | f64 | n=8388608 | 2478.330 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.463 ms | 1.06 | 11.06 | 108% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.079 ms | 0.85 | 8.85 | 87% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.127 ms | 2.77 | 5.82 | 57% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.424 ms | 0.20 | 0.70 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.777 ms | 0.35 | 2.88 | 28% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 274.115 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 69.604 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.544 ms | 0.42 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 116.264 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 153.213 ms | 0.15 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 47.885 ms | 0.47 | 0.02 | 4% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 231.569 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.966 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.318 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.631 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.956 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 231.752 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 393.721 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 126.980 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 101.953 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 121.936 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 79.133 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 519.380 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 175.151 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 90.825 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 136.209 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 137.658 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 170.841 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 249.427 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.127 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.393 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 34.283 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.861 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.717 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.075 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.061 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.653 ms | 0.72 | 8.03 | 79% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.109 ms | 0.84 | 9.43 | 92% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.804 ms | 0.69 | 7.71 | 76% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.903 ms | 1.07 | 8.10 | 79% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 140.896 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 189.695 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 147.248 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.413 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 187.790 ms | 0.09 | 0.36 | 4% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 113.327 ms | 2.37 | 0.04 | 23% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.237 ms | 7.50 | 7.97 | 78% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.469 ms | 2.59 | 10.37 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.159 ms | 5.31 | 10.62 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.010 ms | 3.35 | 3.35 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 5.094 ms | 3.29 | 3.29 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 57.725 ms | 4.65 | 0.05 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 45.202 ms | 5.94 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 23.766 ms | 11.29 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 23.701 ms | 11.33 | 0.07 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 71.276 ms | 3.77 | 0.02 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

