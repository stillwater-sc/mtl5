## 12th Gen Intel(R) Core(TM) i7-12700K -- blas, 20 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `e7497e6e4c47` |
| FMA peak (ceiling) | f64 625.8 GFLOP/s, f32 1266.4 GFLOP/s |
| ... hardware, AVX2+FMA | f64 625.8, f32 1266.4 GFLOP/s |
| ... this build, scalar | f64 18.6, f32 18.7 GFLOP/s |
| bandwidth | read 72.1, update 71.2, triad 66.8 GB/s (ceiling 72.1) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 55% | 128% | 0 | 5% |
| blas2 | 6 | 157% | 174% | 0 | 15% |
| blas3 | 10 | 39% | 63% | 0 | 19% |
| factor | 12 | 5% | 10% | 0 | 6% |
| eigen | 8 | 2% | 8% | 0 | 4% |
| elementwise | 5 | 4% | 7% | 0 | 5% |
| sparse | 5 | 41% | 400% | 0 | 5% |
| direct | 7 | 0% | 0% | 0 | 5% |
| ordering | 4 | n/a | n/a | 0 | 5% |
| krylov | 13 | n/a | n/a | 0 | 4% |
| pc | 7 | n/a | n/a | 0 | 5% |
| smoother | 3 | 34% | 42% | 0 | 5% |
| mg | 1 | 28% | 28% | 0 | 5% |
| sparse_eig | 4 | n/a | n/a | 0 | 5% |
| mixed | 3 | 19% | 46% | 0 | 5% |
| integer | 9 | n/a | n/a | 0 | 42% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.459 ms | 11.50 | 91.99 | 128% | ok | 2.5x (12%) |
| blas1 | dot | f32 | n=8388608 | 1.694 ms | 9.91 | 39.62 | 55% | ok | 1.0x (5%) |
| blas1 | axpy | f64 | n=8388608 | 2.483 ms | 6.76 | 81.07 | 112% | ok | 1.8x (9%) |
| blas1 | axpy | f32 | n=8388608 | 1.103 ms | 15.21 | 91.29 | 127% | ok | 2.0x (10%) |
| blas1 | scale | f64 | n=8388608 | 1.556 ms | 5.39 | 86.25 | 120% | ok | 2.1x (10%) |
| blas1 | two_norm | f64 | n=8388608 | 3.435 ms | 4.88 | 19.53 | 27% | ok | 1.0x (5%) |
| blas1 | one_norm | f64 | n=8388608 | 3.968 ms | 2.11 | 16.91 | 23% | ok | 1.0x (5%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.940 ms | 1.21 | 9.67 | 13% | ok | 1.0x (5%) |
| blas1 | sum | f64 | n=8388608 | 3.929 ms | 2.14 | 17.08 | 24% | ok | 1.0x (5%) |
| blas1 | dot_k3 | f64 | n=8388608 | 76.305 ms | 2.42 | 1.76 | 2% | ok | 1.0x (5%) |
| blas2 | gemv | f64 | n=4096 | 1.498 ms | 22.41 | 89.67 | 124% | ok | 2.3x (12%) |
| blas2 | gemv | f32 | n=4096 | 0.537 ms | 62.53 | 125.12 | 174% ! | ok | 2.9x (15%) |
| blas2 | symv | f64 | n=4096 | 0.594 ms | 56.50 | 113.12 | 157% ! | ok | 3.2x (16%) |
| blas2 | trmv | f64 | n=4096 | 0.581 ms | 28.89 | 115.68 | 160% ! | ok | 3.2x (16%) |
| blas2 | trsv | f64 | n=4096 | 1.767 ms | 9.50 | 38.02 | 53% | ok | 1.0x (5%) |
| blas2 | ger | f64 | n=4096 | 7.674 ms | 4.37 | 34.99 | 49% | ok | 1.0x (5%) |
| blas3 | gemm | f64 | 512x512x512 | 0.677 ms | 396.77 | 9.30 | 63% | ok | 5.4x (27%) |
| blas3 | gemm | f32 | 512x512x512 | 0.344 ms | 779.28 | 9.13 | 62% | ok | 5.3x (26%) |
| blas3 | gemmt | f64 | 512x512x512 | 0.982 ms | 136.73 | 6.41 | 22% | ok | 2.6x (13%) |
| blas3 | symm | f64 | 512x512x512 | 0.678 ms | 396.09 | 9.28 | 63% | ok | 5.5x (27%) |
| blas3 | syrk | f64 | 512x512x512 | 0.807 ms | 166.33 | 5.20 | 27% | ok | 3.1x (16%) |
| blas3 | syr2k | f64 | 512x512x512 | 1.446 ms | 185.62 | 4.35 | 30% | ok | 3.2x (16%) |
| blas3 | trmm | f64 | 512x512x512 | 0.565 ms | 237.37 | 9.27 | 38% | ok | 3.7x (19%) |
| blas3 | trsm | f64 | 512x512x512 | 0.547 ms | 245.20 | 9.58 | 39% | ok | 3.9x (19%) |
| blas3 | trsm | f32 | 512x512x512 | 0.244 ms | 549.46 | 10.73 | 43% | ok | 3.9x (20%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.965 ms | 0.41 | 3.28 | 5% | ok | 1.0x (5%) |
| factor | lu | f64 | 512x512 | 2.511 ms | 35.63 | 1.67 | 6% | ok | 1.3x (6%) |
| factor | lu | f32 | 512x512 | 1.310 ms | 68.33 | 1.60 | 5% | ok | 1.3x (7%) |
| factor | cholesky | f64 | 512x512 | 1.530 ms | 29.25 | 2.74 | 5% | ok | 1.3x (6%) |
| factor | cholesky | f32 | 512x512 | 1.008 ms | 44.39 | 2.08 | 4% | ok | 1.2x (6%) |
| factor | ldlt | f64 | 512x512 | 7.202 ms | 6.21 | 0.58 | 1% | ok | 1.0x (5%) |
| factor | ldlt_bk | f64 | 512x512 | 44.241 ms | 1.01 | 0.09 | 0% | ok | 1.0x (5%) |
| factor | qr | f64 | 512x512 | 4.696 ms | 38.11 | 0.89 | 6% | ok | 1.2x (6%) |
| factor | lq | f64 | 512x512 | 18.820 ms | 9.51 | 0.22 | 2% | ok | 1.3x (7%) |
| factor | hessenberg | f64 | 512x512 | 183.059 ms | 2.44 | 0.02 | 0% | ok | 1.4x (7%) |
| factor | householder | f64 | 512x512 | 0.584 ms | 1.80 | 7.18 | 10%\* | ok | 2.6x (13%) |
| factor | inv | f64 | 512x512 | 61.778 ms | 4.35 | 0.07 | 1% | ok | 0.9x (5%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 2.979 ms | 30.03 | 1.41 | 5% | ok | 1.4x (7%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.976 ms | 11.32 | 0.53 | 2% | ok | 0.9x (4%) |
| eigen | eigen_symmetric | f64 | 256x256 | 57.548 ms | 2.62 | 0.02 | 0% | ok | 1.0x (5%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 744.935 ms | 0.20 | 0.00 | 0% | ok | 1.0x (5%) |
| eigen | eigenvalue | f64 | 256x256 | 23.525 ms | 7.13 | 0.04 | 1% | ok | 0.7x (3%) |
| eigen | schur | f64 | 256x256 | 30.742 ms | 13.64 | 0.03 | 2% | ok | 0.7x (3%) |
| eigen | svd | f64 | 256x256 | 7.479 ms | 49.35 | 0.14 | 8% | ok | 0.9x (4%) |
| eigen | expm | f64 | 256x256 | 10.867 ms | 23.16 | 0.10 | 4% | ok | 1.1x (6%) |
| eigen | sylvester | f64 | 256x256 | 72.854 ms | 13.82 | 0.02 | 2% | ok | 0.7x (4%) |
| elementwise | exp | f64 | n=8388608 | 38.662 ms | 0.22 | 3.47 | 5% | ok | 1.0x (5%) |
| elementwise | log | f64 | n=8388608 | 44.763 ms | 0.19 | 3.00 | 4% | ok | 1.0x (5%) |
| elementwise | sin | f64 | n=8388608 | 59.234 ms | 0.14 | 2.27 | 3% | ok | 1.0x (5%) |
| elementwise | sqrt | f64 | n=8388608 | 28.604 ms | 0.29 | 4.69 | 7% | ok | 1.0x (5%) |
| elementwise | erf | f64 | n=8388608 | 203.394 ms | 0.04 | 0.66 | 1% | ok | 1.0x (5%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.094 ms | 27.74 | 288.64 | 400% ! | ok | 7.9x (39%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.920 ms | 2.84 | 29.59 | 41% | ok | 1.0x (5%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.617 ms | 25.90 | 54.46 | 76% | ok | 2.8x (14%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.797 ms | 0.68 | 2.40 | 3%\* | ok | 1.0x (5%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.440 ms | 1.39 | 11.56 | 16%\* | ok | 0.8x (4%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 90.365 ms | 0.12 | 0.01 | 0% | ok | 1.0x (5%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.649 ms | 0.44 | 0.03 | 0% | ok | 1.0x (5%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.988 ms | 1.24 | 0.08 | 0% | ok | 1.0x (5%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.249 ms | 0.56 | 0.02 | 0% | ok | 1.0x (5%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 53.367 ms | 0.42 | 0.01 | 0% | ok | 1.0x (5%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 20.649 ms | 1.08 | 0.04 | 0% | ok | 1.0x (5%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 86.309 ms | 0.02 | 0.00 | 0% | ok | 1.0x (5%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.287 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.876 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.223 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.878 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 35.382 ms | 0.00 | 0.00 | n/a | ok | 0.8x (4%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 34.840 ms | 0.00 | 0.00 | n/a | ok | 0.8x (4%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 25.879 ms | 0.00 | 0.00 | n/a | ok | 0.8x (4%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 13.773 ms | 0.00 | 0.00 | n/a | ok | 0.8x (4%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 16.352 ms | 0.00 | 0.00 | n/a | ok | 0.8x (4%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 15.870 ms | 0.00 | 0.00 | n/a | ok | 0.8x (4%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 29.711 ms | 0.00 | 0.00 | n/a | ok | 0.7x (3%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 27.904 ms | 0.00 | 0.00 | n/a | ok | 0.8x (4%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 16.002 ms | 0.00 | 0.00 | n/a | ok | 0.8x (4%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 19.232 ms | 0.00 | 0.00 | n/a | ok | 0.8x (4%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 18.400 ms | 0.00 | 0.00 | n/a | ok | 0.8x (4%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 19.632 ms | 0.00 | 0.00 | n/a | ok | 0.8x (4%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 22.656 ms | 0.00 | 0.00 | n/a | ok | 0.8x (4%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.043 ms | 0.00 | 0.00 | n/a | ok | 0.8x (4%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.521 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| pc | ilut | f64 | 128^2 grid | 12.002 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| pc | ic_0 | f64 | 128^2 grid | 1.036 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| pc | ildl | f64 | 128^2 grid | 0.979 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| pc | ssor | f64 | 128^2 grid | 0.026 ms | 0.00 | 0.00 | n/a | ok | 0.8x (4%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.139 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.975 ms | 2.68 | 30.07 | 42% | ok | 1.0x (5%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.199 ms | 2.18 | 24.47 | 34% | ok | 1.0x (5%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.536 ms | 1.70 | 19.09 | 26% | ok | 1.0x (5%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.347 ms | 2.68 | 20.26 | 28% | ok | 0.9x (5%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.362 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 31.324 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 37.851 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.561 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| mixed | dot | f32>f64 | n=8388608 | 4.818 ms | 3.48 | 13.93 | 19% | ok | 1.0x (5%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.592 ms | 58.46 | 0.91 | 9% | ok | 8.4x (42%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.543 ms | 30.92 | 32.86 | 46%\* | ok | 1.0x (5%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.292 ms | 7.32 | 29.28 | n/a | ok | 1.0x (5%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.001 ms | 16.76 | 33.52 | n/a | ok | 1.0x (5%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.686 ms | 9.95 | 9.95 | n/a | ok | 1.0x (5%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.686 ms | 9.95 | 9.95 | n/a | ok | 1.0x (5%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.021 ms | 132.83 | 1.56 | n/a | ok | 8.5x (42%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.407 ms | 111.50 | 0.87 | n/a | ok | 8.5x (42%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.080 ms | 248.65 | 1.46 | n/a | ok | 9.3x (46%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.079 ms | 248.88 | 1.46 | n/a | ok | 9.3x (47%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.101 ms | 86.57 | 0.51 | n/a | ok | 8.5x (43%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

