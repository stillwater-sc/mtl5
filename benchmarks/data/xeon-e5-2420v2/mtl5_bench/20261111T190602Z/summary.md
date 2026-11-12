# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,6, CPUs 0,1,2,3,4,5, clock policy governor performance on every CPU, turbo on; performance 1200-2700 MHz on cpus 0,1,2,3,4,5, 2026-11-11T19:06:02Z.

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

## xeon-e5-2420v2 -- native, 6 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 59.0 GFLOP/s, f32 118.1 GFLOP/s |
| ... hardware, SSE2 | f64 59.0, f32 118.1 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.5 GFLOP/s |
| bandwidth | read 11.4, update 10.6, triad 8.4 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 54% | 107% | 0 | 81% |
| blas2 | 10 | 85% | 107% | 0 | 17% |
| blas3 | 10 | 7% | 39% | 0 | 17% |
| factor | 12 | 3% | 38% | 0 | 19% |
| eigen | 8 | 3% | 11% | 0 | 18% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 68% | 103% | 0 | 17% |
| direct | 7 | 0% | 1% | 0 | 17% |
| ordering | 4 | n/a | n/a | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 70% | 77% | 0 | 16% |
| mg | 1 | 73% | 73% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 18% |
| mixed | 3 | 22% | 67% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 91% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 33.035 ms | 0.51 | 4.06 | 36% | ok | 5.6x (93%) |
| blas1 | dot | f32 | n=8388608 | 5.515 ms | 3.04 | 12.17 | 107% | ok | 4.9x (81%) |
| blas1 | axpy | f64 | n=8388608 | 38.865 ms | 0.43 | 5.18 | 45% | ok | 5.5x (92%) |
| blas1 | axpy | f32 | n=8388608 | 16.368 ms | 1.03 | 6.15 | 54% | ok | 5.6x (93%) |
| blas1 | scale | f64 | n=8388608 | 12.190 ms | 0.69 | 11.01 | 97% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 33.124 ms | 0.51 | 2.03 | 18% | ok | 5.8x (97%) |
| blas1 | one_norm | f64 | n=8388608 | 9.771 ms | 0.86 | 6.87 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.048 ms | 0.64 | 5.14 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.738 ms | 0.86 | 6.89 | 60% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 280.165 ms | 0.66 | 0.48 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 10.976 ms | 3.06 | 12.23 | 107% | ok | 1.2x (20%) |
| blas2 | gemv | f32 | n=4096 | 5.524 ms | 6.07 | 12.15 | 107% | ok | 1.5x (25%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 10.959 ms | 3.06 | 12.25 | 107% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.921 ms | 4.85 | 9.70 | 85% | ok | 0.9x (16%) |
| blas2 | symv | f64 | n=4096 | 16.161 ms | 2.08 | 4.16 | 36% | ok | 1.0x (17%) |
| blas2 | trmv | f64 | n=4096 | 7.239 ms | 2.32 | 9.28 | 81% | ok | 1.0x (16%) |
| blas2 | trmv | f32 | n=4096 | 4.855 ms | 3.46 | 6.92 | 61% | ok | 1.0x (16%) |
| blas2 | trsv | f64 | n=4096 | 7.004 ms | 2.40 | 9.59 | 84% | ok | 1.0x (16%) |
| blas2 | trsv | f32 | n=4096 | 4.751 ms | 3.53 | 7.07 | 62% | ok | 1.0x (16%) |
| blas2 | ger | f64 | n=4096 | 27.851 ms | 1.20 | 9.64 | 85% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 11.598 ms | 23.14 | 0.54 | 39% | ok | 5.7x (94%) |
| blas3 | gemm | f32 | 512x512x512 | 6.435 ms | 41.71 | 0.49 | 35% | ok | 5.4x (90%) |
| blas3 | gemmt | f64 | 512x512x512 | 29.345 ms | 4.57 | 0.21 | 8% | ok | 1.0x (16%) |
| blas3 | symm | f64 | 512x512x512 | 61.011 ms | 4.40 | 0.10 | 7% | ok | 1.0x (17%) |
| blas3 | syrk | f64 | 512x512x512 | 38.337 ms | 3.50 | 0.11 | 6% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 512x512x512 | 65.877 ms | 4.07 | 0.10 | 7% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 512x512x512 | 33.168 ms | 4.05 | 0.16 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f64 | 512x512x512 | 32.426 ms | 4.14 | 0.16 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f32 | 512x512x512 | 16.974 ms | 7.91 | 0.15 | 7% | ok | 1.0x (16%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.535 ms | 0.14 | 1.10 | 10% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 24.017 ms | 3.73 | 0.17 | 6% | ok | 1.3x (21%) |
| factor | lu | f32 | 512x512 | 14.022 ms | 6.38 | 0.15 | 5% | ok | 1.2x (19%) |
| factor | cholesky | f64 | 512x512 | 25.323 ms | 1.77 | 0.17 | 3% | ok | 1.0x (16%) |
| factor | cholesky | f32 | 512x512 | 23.902 ms | 1.87 | 0.09 | 2% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 25.385 ms | 1.76 | 0.17 | 3% | ok | 1.0x (16%) |
| factor | ldlt_bk | f64 | 512x512 | 135.416 ms | 0.33 | 0.03 | 1% | ok | 1.1x (18%) |
| factor | qr | f64 | 512x512 | 321.919 ms | 0.56 | 0.01 | 1% | ok | 1.6x (27%) |
| factor | lq | f64 | 512x512 | 51.133 ms | 3.50 | 0.08 | 6% | ok | 1.4x (24%) |
| factor | hessenberg | f64 | 512x512 | 404.048 ms | 1.11 | 0.01 | 2% | ok | 1.6x (26%) |
| factor | householder | f64 | 512x512 | 0.962 ms | 1.09 | 4.36 | 38%\* | ok | 3.3x (56%) |
| factor | inv | f64 | 512x512 | 148.026 ms | 1.81 | 0.03 | 3% | ok | 1.0x (17%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 19.129 ms | 4.68 | 0.22 | 8% | ok | 1.1x (19%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 60.437 ms | 0.37 | 0.02 | 1% | ok | 1.1x (18%) |
| eigen | eigen_symmetric | f64 | 256x256 | 169.090 ms | 0.89 | 0.01 | 2% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 990.463 ms | 0.15 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 106.743 ms | 1.57 | 0.01 | 3% | ok | 1.1x (18%) |
| eigen | schur | f64 | 256x256 | 263.592 ms | 1.59 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 304.044 ms | 1.21 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 37.434 ms | 6.72 | 0.03 | 11% | ok | 2.3x (38%) |
| eigen | sylvester | f64 | 256x256 | 585.827 ms | 1.72 | 0.00 | 3% | ok | 1.1x (18%) |
| elementwise | exp | f64 | n=8388608 | 152.559 ms | 0.05 | 0.88 | 8% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 142.274 ms | 0.06 | 0.94 | 8% | ok | 1.0x (16%) |
| elementwise | sin | f64 | n=8388608 | 172.044 ms | 0.05 | 0.78 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 90.762 ms | 0.09 | 1.48 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 2471.498 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.313 ms | 1.13 | 11.77 | 103% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.093 ms | 0.85 | 8.80 | 77% | ok | 1.0x (16%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.354 ms | 3.69 | 7.76 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.154 ms | 0.20 | 0.71 | 6%\* | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.845 ms | 0.34 | 2.85 | 25% | ok | 1.0x (16%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 274.391 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 69.378 ms | 0.16 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.931 ms | 0.42 | 0.03 | 1% | ok | 1.0x (16%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 116.980 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 154.391 ms | 0.14 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 48.442 ms | 0.46 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 231.361 ms | 0.01 | 0.00 | 0% | ok | 1.0x (16%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.908 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.268 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.610 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.874 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 231.783 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 393.968 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 126.557 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 102.049 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 121.256 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 79.084 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 518.805 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 174.265 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 90.778 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 136.612 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 137.614 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 172.218 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 250.033 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.141 ms | 0.00 | 0.00 | n/a | ok | 0.9x (15%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.414 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 33.126 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 2.897 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 2.707 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.076 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.143 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.695 ms | 0.71 | 7.94 | 70% | ok | 1.0x (16%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.342 ms | 0.78 | 8.77 | 77% | ok | 0.9x (16%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.832 ms | 0.68 | 7.65 | 67% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 34.936 ms | 1.10 | 8.32 | 73% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 141.782 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 189.310 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 145.148 ms | 0.00 | 0.00 | n/a | ok | 1.1x (19%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.702 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 187.734 ms | 0.09 | 0.36 | 3% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 20.564 ms | 13.05 | 0.20 | 22% | ok | 5.5x (92%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.337 ms | 7.18 | 7.63 | 67% | ok | 1.0x (16%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.447 ms | 2.60 | 10.41 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.145 ms | 5.33 | 10.67 | n/a | ok | 1.0x (16%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.984 ms | 3.37 | 3.37 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 5.013 ms | 3.35 | 3.35 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 10.348 ms | 25.94 | 0.30 | n/a | ok | 5.6x (93%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 8.238 ms | 32.58 | 0.25 | n/a | ok | 5.5x (92%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.336 ms | 61.90 | 0.36 | n/a | ok | 5.5x (91%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.335 ms | 61.93 | 0.36 | n/a | ok | 5.5x (92%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.096 ms | 20.50 | 0.12 | n/a | ok | 5.5x (92%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## xeon-e5-2420v2 -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 21.0 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 21.0 GFLOP/s |
| ... this build, SSE4 | f64 10.5, f32 20.9 GFLOP/s |
| bandwidth | read 10.4, update 10.2, triad 8.0 GB/s (ceiling 10.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 104% | 109% | 0 |
| blas2 | 10 | 91% | 101% | 0 |
| blas3 | 10 | 73% | 95% | 0 |
| factor | 12 | 17% | 39% | 0 |
| eigen | 8 | 15% | 47% | 0 |
| elementwise | 5 | 9% | 14% | 0 |
| sparse | 5 | 56% | 107% | 0 |
| direct | 7 | 2% | 4% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 79% | 91% | 0 |
| mg | 1 | 79% | 79% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 76% | 90% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.873 ms | 1.41 | 11.30 | 109% | ok |
| blas1 | dot | f32 | n=8388608 | 5.948 ms | 2.82 | 11.28 | 109% | ok |
| blas1 | axpy | f64 | n=8388608 | 18.714 ms | 0.90 | 10.76 | 104% | ok |
| blas1 | axpy | f32 | n=8388608 | 9.108 ms | 1.84 | 11.05 | 106% | ok |
| blas1 | scale | f64 | n=8388608 | 12.452 ms | 0.67 | 10.78 | 104% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.929 ms | 2.83 | 11.32 | 109% | ok |
| blas1 | one_norm | f64 | n=8388608 | 9.731 ms | 0.86 | 6.90 | 66% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 13.074 ms | 0.64 | 5.13 | 49% | ok |
| blas1 | sum | f64 | n=8388608 | 9.746 ms | 0.86 | 6.89 | 66% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 275.357 ms | 0.67 | 0.49 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 13.002 ms | 2.58 | 10.33 | 100% | ok |
| blas2 | gemv | f32 | n=4096 | 6.429 ms | 5.22 | 10.44 | 101% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 37.452 ms | 0.90 | 3.59 | 35% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 16.981 ms | 1.98 | 3.95 | 38% | ok |
| blas2 | symv | f64 | n=4096 | 16.333 ms | 2.05 | 4.11 | 40% | ok |
| blas2 | trmv | f64 | n=4096 | 6.984 ms | 2.40 | 9.62 | 93% | ok |
| blas2 | trmv | f32 | n=4096 | 3.597 ms | 4.66 | 9.34 | 90% | ok |
| blas2 | trsv | f64 | n=4096 | 6.878 ms | 2.44 | 9.77 | 94% | ok |
| blas2 | trsv | f32 | n=4096 | 3.606 ms | 4.65 | 9.32 | 90% | ok |
| blas2 | ger | f64 | n=4096 | 28.433 ms | 1.18 | 9.44 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 27.833 ms | 9.64 | 0.23 | 92% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.450 ms | 19.96 | 0.23 | 95% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 16.922 ms | 7.93 | 0.37 | 76% | ok |
| blas3 | symm | f64 | 512x512x512 | 28.905 ms | 9.29 | 0.22 | 88% | ok |
| blas3 | syrk | f64 | 512x512x512 | 36.965 ms | 3.63 | 0.11 | 35% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 63.544 ms | 4.22 | 0.10 | 40% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.583 ms | 7.63 | 0.30 | 73% | ok |
| blas3 | trsm | f64 | 512x512x512 | 18.220 ms | 7.37 | 0.29 | 70% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.841 ms | 15.18 | 0.30 | 72% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.806 ms | 0.14 | 1.09 | 11% | ok |
| factor | lu | f64 | 512x512 | 29.511 ms | 3.03 | 0.14 | 29% | ok |
| factor | lu | f32 | 512x512 | 16.382 ms | 5.46 | 0.13 | 26% | ok |
| factor | cholesky | f64 | 512x512 | 25.477 ms | 1.76 | 0.16 | 17% | ok |
| factor | cholesky | f32 | 512x512 | 24.502 ms | 1.83 | 0.09 | 9% | ok |
| factor | ldlt | f64 | 512x512 | 25.790 ms | 1.73 | 0.16 | 17% | ok |
| factor | ldlt_bk | f64 | 512x512 | 147.732 ms | 0.30 | 0.03 | 3% | ok |
| factor | qr | f64 | 512x512 | 504.336 ms | 0.35 | 0.01 | 3% | ok |
| factor | lq | f64 | 512x512 | 73.246 ms | 2.44 | 0.06 | 23% | ok |
| factor | hessenberg | f64 | 512x512 | 613.815 ms | 0.73 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.197 ms | 0.33 | 1.31 | 13%\* | ok |
| factor | inv | f64 | 512x512 | 153.929 ms | 1.74 | 0.03 | 17% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 21.812 ms | 4.10 | 0.19 | 39% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 61.166 ms | 0.37 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 171.982 ms | 0.88 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 983.557 ms | 0.15 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 108.691 ms | 1.54 | 0.01 | 15% | ok |
| eigen | schur | f64 | 256x256 | 260.876 ms | 1.61 | 0.00 | 15% | ok |
| eigen | svd | f64 | 256x256 | 263.758 ms | 1.40 | 0.00 | 13% | ok |
| eigen | expm | f64 | 256x256 | 51.273 ms | 4.91 | 0.02 | 47% | ok |
| eigen | sylvester | f64 | 256x256 | 582.486 ms | 1.73 | 0.00 | 16% | ok |
| elementwise | exp | f64 | n=8388608 | 151.940 ms | 0.06 | 0.88 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 139.980 ms | 0.06 | 0.96 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 171.195 ms | 0.05 | 0.78 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 90.635 ms | 0.09 | 1.48 | 14% | ok |
| elementwise | erf | f64 | n=8388608 | 2471.853 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.458 ms | 1.06 | 11.08 | 107% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 4.118 ms | 0.64 | 6.61 | 64% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.122 ms | 2.77 | 5.82 | 56% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.362 ms | 0.20 | 0.70 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.796 ms | 0.35 | 2.87 | 28% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 239.595 ms | 0.05 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 70.342 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.482 ms | 0.42 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 118.167 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 155.544 ms | 0.14 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 49.093 ms | 0.46 | 0.01 | 4% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 234.179 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.074 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.277 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.649 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.344 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 80.659 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 112.556 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 64.660 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 32.333 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 38.260 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 37.303 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 69.815 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 74.553 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 37.737 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 43.895 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 42.497 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 45.418 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 54.655 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.130 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.507 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 33.337 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.885 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.806 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.076 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.060 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.586 ms | 0.73 | 8.18 | 79% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.096 ms | 0.85 | 9.47 | 91% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.806 ms | 0.69 | 7.71 | 74% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.622 ms | 1.08 | 8.16 | 79% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 140.101 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 181.276 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 148.458 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 16.226 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 7.161 ms | 2.34 | 9.37 | 90% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 62.050 ms | 4.33 | 0.07 | 41% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.270 ms | 7.39 | 7.85 | 76% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.008 ms | 2.79 | 11.17 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.998 ms | 5.60 | 11.19 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.080 ms | 8.07 | 8.07 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.646 ms | 10.19 | 10.19 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 18.762 ms | 14.31 | 0.17 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 22.024 ms | 12.19 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 22.034 ms | 12.18 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 16.107 ms | 16.67 | 0.10 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 12.410 ms | 21.63 | 0.13 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## xeon-e5-2420v2 -- native-fast, 6 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 59.1 GFLOP/s, f32 118.2 GFLOP/s |
| ... hardware, SSE2 | f64 59.1, f32 118.2 GFLOP/s |
| ... this build, SSE4 | f64 59.0, f32 118.1 GFLOP/s |
| bandwidth | read 11.4, update 10.7, triad 8.5 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 98% | 108% | 0 | 17% |
| blas2 | 10 | 83% | 108% | 0 | 17% |
| blas3 | 10 | 28% | 78% | 0 | 36% |
| factor | 12 | 3% | 38% | 0 | 20% |
| eigen | 8 | 3% | 13% | 0 | 17% |
| elementwise | 5 | 8% | 12% | 0 | 17% |
| sparse | 5 | 55% | 103% | 0 | 17% |
| direct | 7 | 0% | 1% | 0 | 17% |
| ordering | 4 | n/a | n/a | 0 | 16% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 73% | 82% | 0 | 16% |
| mg | 1 | 73% | 73% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 69% | 81% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 79% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 10.917 ms | 1.54 | 12.29 | 108% | ok | 1.1x (18%) |
| blas1 | dot | f32 | n=8388608 | 5.469 ms | 3.07 | 12.27 | 108% | ok | 1.1x (18%) |
| blas1 | axpy | f64 | n=8388608 | 17.918 ms | 0.94 | 11.24 | 98% | ok | 1.0x (17%) |
| blas1 | axpy | f32 | n=8388608 | 8.505 ms | 1.97 | 11.84 | 104% | ok | 1.1x (18%) |
| blas1 | scale | f64 | n=8388608 | 12.184 ms | 0.69 | 11.02 | 97% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 5.455 ms | 3.08 | 12.30 | 108% | ok | 1.1x (18%) |
| blas1 | one_norm | f64 | n=8388608 | 9.769 ms | 0.86 | 6.87 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.020 ms | 0.64 | 5.15 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.759 ms | 0.86 | 6.88 | 60% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 273.764 ms | 0.67 | 0.49 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 10.936 ms | 3.07 | 12.28 | 108% | ok | 1.2x (20%) |
| blas2 | gemv | f32 | n=4096 | 5.594 ms | 6.00 | 12.00 | 105% | ok | 1.1x (19%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 24.788 ms | 1.35 | 5.42 | 47% | ok | 1.5x (25%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 12.992 ms | 2.58 | 5.17 | 45% | ok | 1.3x (22%) |
| blas2 | symv | f64 | n=4096 | 16.360 ms | 2.05 | 4.11 | 36% | ok | 1.0x (17%) |
| blas2 | trmv | f64 | n=4096 | 6.877 ms | 2.44 | 9.77 | 86% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 3.530 ms | 4.75 | 9.52 | 83% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 7.056 ms | 2.38 | 9.52 | 83% | ok | 1.0x (16%) |
| blas2 | trsv | f32 | n=4096 | 3.527 ms | 4.76 | 9.52 | 83% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 27.883 ms | 1.20 | 9.63 | 84% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 5.948 ms | 45.13 | 1.06 | 76% | ok | 4.7x (78%) |
| blas3 | gemm | f32 | 512x512x512 | 2.910 ms | 92.26 | 1.08 | 78% | ok | 4.6x (77%) |
| blas3 | gemmt | f64 | 512x512x512 | 5.336 ms | 25.15 | 1.18 | 43% | ok | 3.2x (53%) |
| blas3 | symm | f64 | 512x512x512 | 6.920 ms | 38.79 | 0.91 | 66% | ok | 4.2x (70%) |
| blas3 | syrk | f64 | 512x512x512 | 37.350 ms | 3.59 | 0.11 | 6% | ok | 1.0x (16%) |
| blas3 | syr2k | f64 | 512x512x512 | 63.719 ms | 4.21 | 0.10 | 7% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 512x512x512 | 8.165 ms | 16.44 | 0.64 | 28% | ok | 2.2x (36%) |
| blas3 | trsm | f64 | 512x512x512 | 8.657 ms | 15.50 | 0.61 | 26% | ok | 2.1x (35%) |
| blas3 | trsm | f32 | 512x512x512 | 4.754 ms | 28.23 | 0.55 | 24% | ok | 1.9x (31%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.432 ms | 0.14 | 1.11 | 10% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 23.340 ms | 3.83 | 0.18 | 6% | ok | 1.3x (21%) |
| factor | lu | f32 | 512x512 | 13.493 ms | 6.63 | 0.16 | 6% | ok | 1.2x (20%) |
| factor | cholesky | f64 | 512x512 | 25.400 ms | 1.76 | 0.17 | 3% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 23.796 ms | 1.88 | 0.09 | 2% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 25.796 ms | 1.73 | 0.16 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 138.345 ms | 0.32 | 0.03 | 1% | ok | 1.1x (18%) |
| factor | qr | f64 | 512x512 | 322.890 ms | 0.55 | 0.01 | 1% | ok | 1.6x (26%) |
| factor | lq | f64 | 512x512 | 51.607 ms | 3.47 | 0.08 | 6% | ok | 1.4x (24%) |
| factor | hessenberg | f64 | 512x512 | 405.770 ms | 1.10 | 0.01 | 2% | ok | 1.5x (25%) |
| factor | householder | f64 | 512x512 | 0.963 ms | 1.09 | 4.36 | 38%\* | ok | 3.3x (55%) |
| factor | inv | f64 | 512x512 | 145.466 ms | 1.85 | 0.03 | 3% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 18.902 ms | 4.73 | 0.22 | 8% | ok | 1.2x (19%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 60.887 ms | 0.37 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 172.094 ms | 0.88 | 0.01 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1004.881 ms | 0.15 | 0.00 | 0% | ok | 1.0x (16%) |
| eigen | eigenvalue | f64 | 256x256 | 104.506 ms | 1.61 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 262.364 ms | 1.60 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 259.741 ms | 1.42 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 32.331 ms | 7.78 | 0.03 | 13% | ok | 1.6x (26%) |
| eigen | sylvester | f64 | 256x256 | 570.426 ms | 1.76 | 0.00 | 3% | ok | 1.0x (17%) |
| elementwise | exp | f64 | n=8388608 | 151.316 ms | 0.06 | 0.89 | 8% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 141.084 ms | 0.06 | 0.95 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 175.435 ms | 0.05 | 0.77 | 7% | ok | 1.0x (16%) |
| elementwise | sqrt | f64 | n=8388608 | 96.207 ms | 0.09 | 1.40 | 12% | ok | 0.9x (16%) |
| elementwise | erf | f64 | n=8388608 | 2479.881 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.315 ms | 1.13 | 11.76 | 103% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 4.301 ms | 0.61 | 6.33 | 55% | ok | 1.0x (16%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.399 ms | 3.67 | 7.72 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 17.231 ms | 0.19 | 0.67 | 6%\* | ok | 0.9x (16%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.839 ms | 0.34 | 2.85 | 25% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 238.483 ms | 0.05 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 70.372 ms | 0.16 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.543 ms | 0.42 | 0.03 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 116.327 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 155.351 ms | 0.14 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 49.233 ms | 0.45 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 235.217 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.276 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 27.547 ms | 0.00 | 0.00 | n/a | ok | 0.4x (7%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.656 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.661 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 80.335 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 112.174 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 64.527 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 32.157 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 38.401 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 37.037 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 66.444 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 74.151 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 37.531 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 43.810 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 42.069 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 46.548 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 54.852 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.130 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.459 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 33.438 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 2.869 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 2.765 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.076 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.200 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.526 ms | 0.74 | 8.32 | 73% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.130 ms | 0.84 | 9.37 | 82% | ok | 1.0x (16%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.904 ms | 0.67 | 7.51 | 66% | ok | 1.0x (16%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 34.987 ms | 1.10 | 8.31 | 73% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 140.147 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 255.962 ms | 0.00 | 0.00 | n/a | ok | 0.7x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 153.168 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 16.273 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 7.249 ms | 2.31 | 9.26 | 81% | ok | 1.0x (16%) |
| mixed | gemm | f32>f64 | 512x512x512 | 11.796 ms | 22.76 | 0.36 | 39% | ok | 5.3x (88%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.259 ms | 7.43 | 7.89 | 69% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.328 ms | 2.65 | 10.61 | n/a | ok | 0.9x (16%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.263 ms | 5.14 | 10.28 | n/a | ok | 0.9x (15%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.080 ms | 8.06 | 8.06 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.652 ms | 10.16 | 10.16 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 3.970 ms | 67.62 | 0.79 | n/a | ok | 4.7x (79%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.555 ms | 58.93 | 0.46 | n/a | ok | 4.8x (81%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.516 ms | 59.44 | 0.35 | n/a | ok | 4.9x (81%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.342 ms | 80.32 | 0.47 | n/a | ok | 4.8x (80%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 2.590 ms | 103.63 | 0.61 | n/a | ok | 4.8x (80%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## xeon-e5-2420v2 -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 21.1 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 21.1 GFLOP/s |
| ... this build, scalar | f64 0.1, f32 0.7 GFLOP/s |
| bandwidth | read 10.2, update 10.3, triad 8.2 GB/s (ceiling 10.3) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 24% | 104% | 0 |
| blas2 | 10 | 92% | 104% | 0 |
| blas3 | 10 | 162% | 181% | 0 |
| factor | 12 | 53% | 92% | 0 |
| eigen | 8 | 50% | 118% | 0 |
| elementwise | 5 | 9% | 14% | 0 |
| sparse | 5 | 56% | 109% | 0 |
| direct | 7 | 2% | 4% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 80% | 92% | 0 |
| mg | 1 | 79% | 79% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 22% | 75% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 182.373 ms | 0.09 | 0.74 | 7% | ok |
| blas1 | dot | f32 | n=8388608 | 26.991 ms | 0.62 | 2.49 | 24% | ok |
| blas1 | axpy | f64 | n=8388608 | 211.121 ms | 0.08 | 0.95 | 9% | ok |
| blas1 | axpy | f32 | n=8388608 | 91.224 ms | 0.18 | 1.10 | 11% | ok |
| blas1 | scale | f64 | n=8388608 | 12.487 ms | 0.67 | 10.75 | 104% | ok |
| blas1 | two_norm | f64 | n=8388608 | 186.301 ms | 0.09 | 0.36 | 3% | ok |
| blas1 | one_norm | f64 | n=8388608 | 9.742 ms | 0.86 | 6.89 | 67% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 13.071 ms | 0.64 | 5.13 | 50% | ok |
| blas1 | sum | f64 | n=8388608 | 9.740 ms | 0.86 | 6.89 | 67% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 277.765 ms | 0.66 | 0.48 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 14.383 ms | 2.33 | 9.34 | 91% | ok |
| blas2 | gemv | f32 | n=4096 | 6.270 ms | 5.35 | 10.71 | 104% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 12.972 ms | 2.59 | 10.35 | 100% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.622 ms | 5.07 | 10.14 | 98% | ok |
| blas2 | symv | f64 | n=4096 | 7.009 ms | 4.79 | 9.58 | 93% | ok |
| blas2 | trmv | f64 | n=4096 | 7.619 ms | 2.20 | 8.82 | 86% | ok |
| blas2 | trmv | f32 | n=4096 | 3.723 ms | 4.51 | 9.02 | 88% | ok |
| blas2 | trsv | f64 | n=4096 | 7.563 ms | 2.22 | 8.88 | 86% | ok |
| blas2 | trsv | f32 | n=4096 | 3.534 ms | 4.75 | 9.50 | 92% | ok |
| blas2 | ger | f64 | n=4096 | 28.726 ms | 1.17 | 9.35 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 14.096 ms | 19.04 | 0.45 | 181% | ok |
| blas3 | gemm | f32 | 512x512x512 | 8.550 ms | 31.40 | 0.37 | 149% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 8.897 ms | 15.09 | 0.71 | 144% | ok |
| blas3 | symm | f64 | 512x512x512 | 14.236 ms | 18.86 | 0.44 | 180% | ok |
| blas3 | syrk | f64 | 512x512x512 | 7.899 ms | 16.99 | 0.53 | 162% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 14.970 ms | 17.93 | 0.42 | 171% | ok |
| blas3 | trmm | f64 | 512x512x512 | 7.486 ms | 17.93 | 0.70 | 171% | ok |
| blas3 | trsm | f64 | 512x512x512 | 8.187 ms | 16.39 | 0.64 | 156% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.534 ms | 24.25 | 0.47 | 115% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.995 ms | 0.14 | 1.09 | 11% | ok |
| factor | lu | f64 | 512x512 | 9.244 ms | 9.68 | 0.45 | 92% | ok |
| factor | lu | f32 | 512x512 | 6.510 ms | 13.75 | 0.32 | 65% | ok |
| factor | cholesky | f64 | 512x512 | 5.818 ms | 7.69 | 0.72 | 73% | ok |
| factor | cholesky | f32 | 512x512 | 4.006 ms | 11.17 | 0.52 | 53% | ok |
| factor | ldlt | f64 | 512x512 | 25.980 ms | 1.72 | 0.16 | 16% | ok |
| factor | ldlt_bk | f64 | 512x512 | 150.052 ms | 0.30 | 0.03 | 3% | ok |
| factor | qr | f64 | 512x512 | 20.094 ms | 8.91 | 0.21 | 85% | ok |
| factor | lq | f64 | 512x512 | 74.552 ms | 2.40 | 0.06 | 23% | ok |
| factor | hessenberg | f64 | 512x512 | 638.219 ms | 0.70 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.275 ms | 0.32 | 1.28 | 12%\* | ok |
| factor | inv | f64 | 512x512 | 131.013 ms | 2.05 | 0.03 | 20% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.818 ms | 6.98 | 0.33 | 67% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 6.216 ms | 3.60 | 0.17 | 34% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 173.988 ms | 0.87 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 985.877 ms | 0.15 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 53.600 ms | 3.13 | 0.02 | 30% | ok |
| eigen | schur | f64 | 256x256 | 72.201 ms | 5.81 | 0.01 | 55% | ok |
| eigen | svd | f64 | 256x256 | 29.866 ms | 12.36 | 0.04 | 118% | ok |
| eigen | expm | f64 | 256x256 | 35.951 ms | 7.00 | 0.03 | 67% | ok |
| eigen | sylvester | f64 | 256x256 | 190.908 ms | 5.27 | 0.01 | 50% | ok |
| elementwise | exp | f64 | n=8388608 | 151.697 ms | 0.06 | 0.88 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 141.023 ms | 0.06 | 0.95 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 170.507 ms | 0.05 | 0.79 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 90.926 ms | 0.09 | 1.48 | 14% | ok |
| elementwise | erf | f64 | n=8388608 | 2480.361 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.419 ms | 1.08 | 11.26 | 109% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.068 ms | 0.85 | 8.87 | 86% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.215 ms | 2.75 | 5.79 | 56% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.219 ms | 0.20 | 0.71 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.743 ms | 0.35 | 2.90 | 28% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 273.354 ms | 0.04 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 70.781 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.259 ms | 0.43 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 116.137 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 156.041 ms | 0.14 | 0.00 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 48.483 ms | 0.46 | 0.02 | 4% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 231.371 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.920 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.187 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.634 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 43.240 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 232.919 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 393.891 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 126.478 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 102.557 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 121.421 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 79.411 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 518.317 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 174.188 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 90.586 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 135.698 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 137.360 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 170.407 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 248.892 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.128 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.436 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 32.890 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.826 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.723 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.077 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 3.423 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.539 ms | 0.74 | 8.29 | 80% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.082 ms | 0.85 | 9.52 | 92% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.797 ms | 0.69 | 7.72 | 75% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.525 ms | 1.08 | 8.18 | 79% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 141.712 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 148.924 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 144.331 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.539 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 187.653 ms | 0.09 | 0.36 | 3% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 114.436 ms | 2.35 | 0.04 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.304 ms | 7.28 | 7.74 | 75% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.501 ms | 2.58 | 10.32 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.157 ms | 5.31 | 10.63 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.982 ms | 3.37 | 3.37 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 5.097 ms | 3.29 | 3.29 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 57.575 ms | 4.66 | 0.05 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 45.060 ms | 5.96 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 23.740 ms | 11.31 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 23.451 ms | 11.45 | 0.07 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 70.283 ms | 3.82 | 0.02 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## xeon-e5-2420v2 -- blas, 6 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 58.9 GFLOP/s, f32 118.2 GFLOP/s |
| ... hardware, SSE2 | f64 58.9, f32 118.2 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.7 GFLOP/s |
| bandwidth | read 11.4, update 10.5, triad 8.4 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 53% | 107% | 0 | 81% |
| blas2 | 10 | 103% | 108% | 0 | 20% |
| blas3 | 10 | 129% | 159% | 0 | 77% |
| factor | 12 | 18% | 38% | 0 | 25% |
| eigen | 8 | 9% | 24% | 0 | 17% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 68% | 103% | 0 | 17% |
| direct | 7 | 0% | 1% | 0 | 17% |
| ordering | 4 | n/a | n/a | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 67% | 82% | 0 | 16% |
| mg | 1 | 73% | 73% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 22% | 68% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 90% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 32.690 ms | 0.51 | 4.11 | 36% | ok | 5.6x (93%) |
| blas1 | dot | f32 | n=8388608 | 5.521 ms | 3.04 | 12.16 | 107% | ok | 4.9x (81%) |
| blas1 | axpy | f64 | n=8388608 | 38.180 ms | 0.44 | 5.27 | 46% | ok | 5.5x (92%) |
| blas1 | axpy | f32 | n=8388608 | 16.570 ms | 1.01 | 6.08 | 53% | ok | 5.5x (92%) |
| blas1 | scale | f64 | n=8388608 | 12.164 ms | 0.69 | 11.03 | 97% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 32.840 ms | 0.51 | 2.04 | 18% | ok | 5.7x (95%) |
| blas1 | one_norm | f64 | n=8388608 | 9.692 ms | 0.87 | 6.92 | 61% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.066 ms | 0.64 | 5.14 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.710 ms | 0.86 | 6.91 | 61% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 277.510 ms | 0.67 | 0.48 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 11.307 ms | 2.97 | 11.88 | 105% | ok | 1.3x (21%) |
| blas2 | gemv | f32 | n=4096 | 5.466 ms | 6.14 | 12.28 | 108% | ok | 1.1x (19%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 10.924 ms | 3.07 | 12.29 | 108% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.691 ms | 5.90 | 11.80 | 104% | ok | 1.2x (19%) |
| blas2 | symv | f64 | n=4096 | 5.779 ms | 5.81 | 11.62 | 102% | ok | 1.2x (20%) |
| blas2 | trmv | f64 | n=4096 | 5.725 ms | 2.93 | 11.73 | 103% | ok | 1.3x (22%) |
| blas2 | trmv | f32 | n=4096 | 2.968 ms | 5.65 | 11.32 | 100% | ok | 1.3x (21%) |
| blas2 | trsv | f64 | n=4096 | 7.433 ms | 2.26 | 9.04 | 80% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 3.476 ms | 4.83 | 9.66 | 85% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 27.922 ms | 1.20 | 9.62 | 85% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 2.870 ms | 93.55 | 2.19 | 159% | ok | 4.9x (82%) |
| blas3 | gemm | f32 | 512x512x512 | 1.762 ms | 152.37 | 1.79 | 129% | ok | 4.9x (81%) |
| blas3 | gemmt | f64 | 512x512x512 | 2.536 ms | 52.93 | 2.48 | 90% | ok | 3.5x (58%) |
| blas3 | symm | f64 | 512x512x512 | 2.863 ms | 93.76 | 2.20 | 159% | ok | 5.0x (83%) |
| blas3 | syrk | f64 | 512x512x512 | 2.517 ms | 53.33 | 1.67 | 91% | ok | 3.1x (52%) |
| blas3 | syr2k | f64 | 512x512x512 | 3.736 ms | 71.85 | 1.68 | 122% | ok | 4.0x (67%) |
| blas3 | trmm | f64 | 512x512x512 | 1.617 ms | 83.02 | 3.24 | 141% | ok | 4.6x (77%) |
| blas3 | trsm | f64 | 512x512x512 | 1.773 ms | 75.69 | 2.96 | 129% | ok | 4.6x (77%) |
| blas3 | trsm | f32 | 512x512x512 | 1.251 ms | 107.26 | 2.09 | 91% | ok | 4.4x (74%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.657 ms | 0.14 | 1.10 | 10% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 5.593 ms | 16.00 | 0.75 | 27% | ok | 1.7x (28%) |
| factor | lu | f32 | 512x512 | 3.865 ms | 23.15 | 0.54 | 20% | ok | 1.7x (28%) |
| factor | cholesky | f64 | 512x512 | 4.129 ms | 10.84 | 1.02 | 18% | ok | 1.4x (23%) |
| factor | cholesky | f32 | 512x512 | 3.086 ms | 14.50 | 0.68 | 12% | ok | 1.3x (22%) |
| factor | ldlt | f64 | 512x512 | 24.996 ms | 1.79 | 0.17 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 137.762 ms | 0.32 | 0.03 | 1% | ok | 1.1x (18%) |
| factor | qr | f64 | 512x512 | 13.176 ms | 13.58 | 0.32 | 23% | ok | 1.5x (25%) |
| factor | lq | f64 | 512x512 | 51.909 ms | 3.45 | 0.08 | 6% | ok | 1.4x (24%) |
| factor | hessenberg | f64 | 512x512 | 415.183 ms | 1.08 | 0.01 | 2% | ok | 1.5x (26%) |
| factor | householder | f64 | 512x512 | 0.967 ms | 1.09 | 4.34 | 38%\* | ok | 3.4x (56%) |
| factor | inv | f64 | 512x512 | 131.496 ms | 2.04 | 0.03 | 3% | ok | 1.0x (17%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 8.442 ms | 10.60 | 0.50 | 18% | ok | 1.5x (25%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 5.764 ms | 3.88 | 0.18 | 7% | ok | 1.1x (18%) |
| eigen | eigen_symmetric | f64 | 256x256 | 171.931 ms | 0.88 | 0.01 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 995.176 ms | 0.15 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 56.534 ms | 2.97 | 0.02 | 5% | ok | 0.9x (16%) |
| eigen | schur | f64 | 256x256 | 75.580 ms | 5.55 | 0.01 | 9% | ok | 1.0x (16%) |
| eigen | svd | f64 | 256x256 | 26.537 ms | 13.91 | 0.04 | 24% | ok | 1.1x (19%) |
| eigen | expm | f64 | 256x256 | 27.345 ms | 9.20 | 0.04 | 16% | ok | 1.3x (22%) |
| eigen | sylvester | f64 | 256x256 | 194.172 ms | 5.18 | 0.01 | 9% | ok | 1.0x (16%) |
| elementwise | exp | f64 | n=8388608 | 152.246 ms | 0.06 | 0.88 | 8% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 140.359 ms | 0.06 | 0.96 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 170.935 ms | 0.05 | 0.79 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 90.533 ms | 0.09 | 1.48 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 2470.984 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.317 ms | 1.13 | 11.75 | 103% | ok | 1.0x (17%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.097 ms | 0.85 | 8.79 | 77% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.339 ms | 3.69 | 7.76 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.120 ms | 0.20 | 0.71 | 6%\* | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.816 ms | 0.35 | 2.86 | 25% | ok | 1.0x (16%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 274.497 ms | 0.04 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 69.834 ms | 0.16 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.474 ms | 0.42 | 0.03 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 116.364 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 154.712 ms | 0.14 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 48.117 ms | 0.46 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 228.643 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.869 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.299 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.614 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.338 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 230.409 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 392.404 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 134.126 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 103.190 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 121.995 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 78.769 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 517.377 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 180.982 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 90.716 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 135.858 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 136.541 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 171.868 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 248.053 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.129 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.441 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 33.003 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 2.830 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 2.717 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.077 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 3.542 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.877 ms | 0.68 | 7.56 | 67% | ok | 0.9x (15%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.156 ms | 0.83 | 9.29 | 82% | ok | 1.0x (16%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.846 ms | 0.68 | 7.63 | 67% | ok | 1.0x (16%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 34.876 ms | 1.10 | 8.33 | 73% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 140.690 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 147.380 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 143.169 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.687 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 192.821 ms | 0.09 | 0.35 | 3% | ok | 1.0x (16%) |
| mixed | gemm | f32>f64 | 512x512x512 | 20.562 ms | 13.05 | 0.20 | 22% | ok | 5.6x (93%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.298 ms | 7.30 | 7.76 | 68% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.532 ms | 2.57 | 10.27 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.138 ms | 5.35 | 10.69 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 5.013 ms | 3.35 | 3.35 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.987 ms | 3.36 | 3.36 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 10.489 ms | 25.59 | 0.30 | n/a | ok | 5.5x (91%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 8.236 ms | 32.59 | 0.25 | n/a | ok | 5.5x (91%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.336 ms | 61.91 | 0.36 | n/a | ok | 5.5x (91%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.330 ms | 62.00 | 0.36 | n/a | ok | 5.4x (90%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.075 ms | 20.53 | 0.12 | n/a | ok | 5.4x (90%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

