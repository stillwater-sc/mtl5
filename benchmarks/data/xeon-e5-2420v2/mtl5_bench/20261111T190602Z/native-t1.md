## xeon-e5-2420v2 -- native, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 21.0 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 21.0 GFLOP/s |
| ... this build, scalar | f64 0.1, f32 0.6 GFLOP/s |
| bandwidth | read 10.2, update 10.2, triad 8.1 GB/s (ceiling 10.2) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 24% | 105% | 0 |
| blas2 | 10 | 93% | 100% | 0 |
| blas3 | 10 | 39% | 45% | 0 |
| factor | 12 | 17% | 39% | 0 |
| eigen | 8 | 14% | 28% | 0 |
| elementwise | 5 | 9% | 14% | 0 |
| sparse | 5 | 57% | 110% | 0 |
| direct | 7 | 2% | 4% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 80% | 91% | 0 |
| mg | 1 | 78% | 78% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 22% | 76% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 184.838 ms | 0.09 | 0.73 | 7% | ok |
| blas1 | dot | f32 | n=8388608 | 26.921 ms | 0.62 | 2.49 | 24% | ok |
| blas1 | axpy | f64 | n=8388608 | 213.500 ms | 0.08 | 0.94 | 9% | ok |
| blas1 | axpy | f32 | n=8388608 | 91.652 ms | 0.18 | 1.10 | 11% | ok |
| blas1 | scale | f64 | n=8388608 | 12.545 ms | 0.67 | 10.70 | 105% | ok |
| blas1 | two_norm | f64 | n=8388608 | 192.924 ms | 0.09 | 0.35 | 3% | ok |
| blas1 | one_norm | f64 | n=8388608 | 9.736 ms | 0.86 | 6.89 | 67% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 12.965 ms | 0.65 | 5.18 | 51% | ok |
| blas1 | sum | f64 | n=8388608 | 9.727 ms | 0.86 | 6.90 | 67% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 281.739 ms | 0.66 | 0.48 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 13.120 ms | 2.56 | 10.24 | 100% | ok |
| blas2 | gemv | f32 | n=4096 | 8.446 ms | 3.97 | 7.95 | 78% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 13.448 ms | 2.50 | 9.99 | 98% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.546 ms | 5.13 | 10.26 | 100% | ok |
| blas2 | symv | f64 | n=4096 | 16.178 ms | 2.07 | 4.15 | 41% | ok |
| blas2 | trmv | f64 | n=4096 | 7.039 ms | 2.38 | 9.54 | 93% | ok |
| blas2 | trmv | f32 | n=4096 | 4.655 ms | 3.60 | 7.22 | 71% | ok |
| blas2 | trsv | f64 | n=4096 | 6.906 ms | 2.43 | 9.73 | 95% | ok |
| blas2 | trsv | f32 | n=4096 | 4.530 ms | 3.70 | 7.41 | 73% | ok |
| blas2 | ger | f64 | n=4096 | 28.141 ms | 1.19 | 9.54 | 93% | ok |
| blas3 | gemm | f64 | 512x512x512 | 65.598 ms | 4.09 | 0.10 | 39% | ok |
| blas3 | gemm | f32 | 512x512x512 | 34.751 ms | 7.72 | 0.09 | 37% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 28.616 ms | 4.69 | 0.22 | 45% | ok |
| blas3 | symm | f64 | 512x512x512 | 60.628 ms | 4.43 | 0.10 | 42% | ok |
| blas3 | syrk | f64 | 512x512x512 | 38.012 ms | 3.53 | 0.11 | 34% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 68.780 ms | 3.90 | 0.09 | 37% | ok |
| blas3 | trmm | f64 | 512x512x512 | 33.175 ms | 4.05 | 0.16 | 39% | ok |
| blas3 | trsm | f64 | 512x512x512 | 32.651 ms | 4.11 | 0.16 | 39% | ok |
| blas3 | trsm | f32 | 512x512x512 | 16.292 ms | 8.24 | 0.16 | 39% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.847 ms | 0.14 | 1.09 | 11% | ok |
| factor | lu | f64 | 512x512 | 30.886 ms | 2.90 | 0.14 | 28% | ok |
| factor | lu | f32 | 512x512 | 16.376 ms | 5.46 | 0.13 | 26% | ok |
| factor | cholesky | f64 | 512x512 | 24.875 ms | 1.80 | 0.17 | 17% | ok |
| factor | cholesky | f32 | 512x512 | 24.049 ms | 1.86 | 0.09 | 9% | ok |
| factor | ldlt | f64 | 512x512 | 25.023 ms | 1.79 | 0.17 | 17% | ok |
| factor | ldlt_bk | f64 | 512x512 | 146.901 ms | 0.30 | 0.03 | 3% | ok |
| factor | qr | f64 | 512x512 | 523.089 ms | 0.34 | 0.01 | 3% | ok |
| factor | lq | f64 | 512x512 | 73.371 ms | 2.44 | 0.06 | 23% | ok |
| factor | hessenberg | f64 | 512x512 | 630.672 ms | 0.71 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.219 ms | 0.33 | 1.30 | 13%\* | ok |
| factor | inv | f64 | 512x512 | 151.131 ms | 1.78 | 0.03 | 17% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 21.802 ms | 4.10 | 0.19 | 39% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 63.762 ms | 0.35 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 169.705 ms | 0.89 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 982.905 ms | 0.15 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 112.882 ms | 1.49 | 0.01 | 14% | ok |
| eigen | schur | f64 | 256x256 | 268.703 ms | 1.56 | 0.00 | 15% | ok |
| eigen | svd | f64 | 256x256 | 301.541 ms | 1.22 | 0.00 | 12% | ok |
| eigen | expm | f64 | 256x256 | 84.698 ms | 2.97 | 0.01 | 28% | ok |
| eigen | sylvester | f64 | 256x256 | 620.384 ms | 1.62 | 0.00 | 15% | ok |
| elementwise | exp | f64 | n=8388608 | 151.432 ms | 0.06 | 0.89 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 140.711 ms | 0.06 | 0.95 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 172.270 ms | 0.05 | 0.78 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 91.225 ms | 0.09 | 1.47 | 14% | ok |
| elementwise | erf | f64 | n=8388608 | 2468.403 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.429 ms | 1.08 | 11.21 | 110% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.059 ms | 0.86 | 8.90 | 87% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.162 ms | 2.76 | 5.81 | 57% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.303 ms | 0.20 | 0.71 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.729 ms | 0.35 | 2.91 | 28% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 275.767 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 69.637 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.450 ms | 0.42 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 116.518 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 155.602 ms | 0.14 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 48.135 ms | 0.46 | 0.02 | 4% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 226.893 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.833 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.098 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.620 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 41.983 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 230.630 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 398.755 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 126.927 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 102.191 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 121.610 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 79.040 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 520.611 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 174.740 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.737 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 135.586 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 136.874 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 174.672 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 250.589 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.127 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.385 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 33.440 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.898 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.736 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.077 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.033 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.580 ms | 0.73 | 8.19 | 80% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.136 ms | 0.83 | 9.35 | 91% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.848 ms | 0.68 | 7.62 | 75% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 36.406 ms | 1.06 | 7.98 | 78% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 150.905 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 206.130 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 162.988 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.975 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 188.867 ms | 0.09 | 0.36 | 3% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 113.774 ms | 2.36 | 0.04 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.284 ms | 7.34 | 7.80 | 76% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.514 ms | 2.58 | 10.30 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.085 ms | 5.44 | 10.88 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.064 ms | 3.31 | 3.31 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 5.094 ms | 3.29 | 3.29 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 57.717 ms | 4.65 | 0.05 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 45.352 ms | 5.92 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 23.658 ms | 11.35 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 23.928 ms | 11.22 | 0.07 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 71.928 ms | 3.73 | 0.02 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

