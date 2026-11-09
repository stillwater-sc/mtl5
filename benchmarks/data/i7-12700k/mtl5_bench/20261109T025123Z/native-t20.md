## 12th Gen Intel(R) Core(TM) i7-12700K -- native, 20 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `e7497e6e4c47` |
| FMA peak (ceiling) | f64 626.3 GFLOP/s, f32 1257.2 GFLOP/s |
| ... hardware, AVX2+FMA | f64 626.3, f32 1257.2 GFLOP/s |
| ... this build, scalar | f64 18.6, f32 19.2 GFLOP/s |
| bandwidth | read 73.3, update 71.8, triad 67.3 GB/s (ceiling 73.3) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 88% | 119% | 0 | 12% |
| blas2 | 6 | 47% | 125% | 0 | 5% |
| blas3 | 10 | 2% | 20% | 0 | 5% |
| factor | 12 | 1% | 12% | 0 | 6% |
| eigen | 8 | 1% | 3% | 0 | 6% |
| elementwise | 5 | 4% | 6% | 0 | 5% |
| sparse | 5 | 41% | 247% | 0 | 5% |
| direct | 7 | 0% | 0% | 0 | 5% |
| ordering | 4 | n/a | n/a | 0 | 5% |
| krylov | 13 | n/a | n/a | 0 | 5% |
| pc | 7 | n/a | n/a | 0 | 5% |
| smoother | 3 | 33% | 41% | 0 | 5% |
| mg | 1 | 28% | 28% | 0 | 5% |
| sparse_eig | 4 | n/a | n/a | 0 | 5% |
| mixed | 3 | 19% | 46% | 0 | 5% |
| integer | 9 | n/a | n/a | 0 | 43% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.541 ms | 10.89 | 87.10 | 119% | ok | 3.6x (18%) |
| blas1 | dot | f32 | n=8388608 | 1.043 ms | 16.08 | 64.32 | 88% | ok | 4.0x (20%) |
| blas1 | axpy | f64 | n=8388608 | 2.727 ms | 6.15 | 73.84 | 101% | ok | 2.3x (12%) |
| blas1 | axpy | f32 | n=8388608 | 1.413 ms | 11.87 | 71.22 | 97% | ok | 3.3x (17%) |
| blas1 | scale | f64 | n=8388608 | 1.740 ms | 4.82 | 77.15 | 105% | ok | 2.0x (10%) |
| blas1 | two_norm | f64 | n=8388608 | 1.042 ms | 16.10 | 64.41 | 88% | ok | 4.6x (23%) |
| blas1 | one_norm | f64 | n=8388608 | 3.972 ms | 2.11 | 16.89 | 23% | ok | 1.0x (5%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.922 ms | 1.21 | 9.69 | 13% | ok | 1.0x (5%) |
| blas1 | sum | f64 | n=8388608 | 3.940 ms | 2.13 | 17.03 | 23% | ok | 1.0x (5%) |
| blas1 | dot_k3 | f64 | n=8388608 | 75.646 ms | 2.44 | 1.77 | 2% | ok | 1.0x (5%) |
| blas2 | gemv | f64 | n=4096 | 1.511 ms | 22.21 | 88.87 | 121% | ok | 5.5x (27%) |
| blas2 | gemv | f32 | n=4096 | 0.733 ms | 45.75 | 91.55 | 125% | ok | 9.3x (47%) |
| blas2 | symv | f64 | n=4096 | 6.503 ms | 5.16 | 10.33 | 14% | ok | 1.0x (5%) |
| blas2 | trmv | f64 | n=4096 | 4.333 ms | 3.87 | 15.50 | 21% | ok | 1.0x (5%) |
| blas2 | trsv | f64 | n=4096 | 4.349 ms | 3.86 | 15.45 | 21% | ok | 1.0x (5%) |
| blas2 | ger | f64 | n=4096 | 7.809 ms | 4.30 | 34.38 | 47% | ok | 1.0x (5%) |
| blas3 | gemm | f64 | 512x512x512 | 2.224 ms | 120.69 | 2.83 | 19% | ok | 8.3x (41%) |
| blas3 | gemm | f32 | 512x512x512 | 1.064 ms | 252.30 | 2.96 | 20% | ok | 9.4x (47%) |
| blas3 | gemmt | f64 | 512x512x512 | 9.053 ms | 14.83 | 0.69 | 2% | ok | 1.0x (5%) |
| blas3 | symm | f64 | 512x512x512 | 16.750 ms | 16.03 | 0.38 | 3% | ok | 1.0x (5%) |
| blas3 | syrk | f64 | 512x512x512 | 11.609 ms | 11.56 | 0.36 | 2% | ok | 1.0x (5%) |
| blas3 | syr2k | f64 | 512x512x512 | 19.743 ms | 13.60 | 0.32 | 2% | ok | 1.0x (5%) |
| blas3 | trmm | f64 | 512x512x512 | 9.129 ms | 14.70 | 0.57 | 2% | ok | 1.0x (5%) |
| blas3 | trsm | f64 | 512x512x512 | 9.180 ms | 14.62 | 0.57 | 2% | ok | 1.0x (5%) |
| blas3 | trsm | f32 | 512x512x512 | 4.949 ms | 27.12 | 0.53 | 2% | ok | 1.0x (5%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.942 ms | 0.41 | 3.28 | 4% | ok | 1.0x (5%) |
| factor | lu | f64 | 512x512 | 8.355 ms | 10.71 | 0.50 | 2% | ok | 1.3x (6%) |
| factor | lu | f32 | 512x512 | 5.031 ms | 17.79 | 0.42 | 1% | ok | 1.1x (6%) |
| factor | cholesky | f64 | 512x512 | 7.094 ms | 6.31 | 0.59 | 1% | ok | 1.0x (5%) |
| factor | cholesky | f32 | 512x512 | 6.461 ms | 6.92 | 0.32 | 1% | ok | 1.0x (5%) |
| factor | ldlt | f64 | 512x512 | 7.177 ms | 6.23 | 0.58 | 1% | ok | 1.0x (5%) |
| factor | ldlt_bk | f64 | 512x512 | 45.832 ms | 0.98 | 0.09 | 0% | ok | 1.0x (5%) |
| factor | qr | f64 | 512x512 | 143.044 ms | 1.25 | 0.03 | 0% | ok | 1.5x (8%) |
| factor | lq | f64 | 512x512 | 18.146 ms | 9.86 | 0.23 | 2% | ok | 1.4x (7%) |
| factor | hessenberg | f64 | 512x512 | 180.341 ms | 2.48 | 0.02 | 0% | ok | 1.4x (7%) |
| factor | householder | f64 | 512x512 | 0.465 ms | 2.26 | 9.03 | 12%\* | ok | 3.3x (16%) |
| factor | inv | f64 | 512x512 | 64.260 ms | 4.18 | 0.07 | 1% | ok | 1.0x (5%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.523 ms | 11.89 | 0.56 | 2% | ok | 1.1x (5%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 18.204 ms | 1.23 | 0.06 | 0% | ok | 1.0x (5%) |
| eigen | eigen_symmetric | f64 | 256x256 | 58.322 ms | 2.59 | 0.02 | 0% | ok | 1.2x (6%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 733.295 ms | 0.21 | 0.00 | 0% | ok | 1.3x (6%) |
| eigen | eigenvalue | f64 | 256x256 | 32.827 ms | 5.11 | 0.03 | 1% | ok | 1.0x (5%) |
| eigen | schur | f64 | 256x256 | 80.371 ms | 5.22 | 0.01 | 1% | ok | 1.3x (6%) |
| eigen | svd | f64 | 256x256 | 106.549 ms | 3.46 | 0.01 | 1% | ok | 1.0x (5%) |
| eigen | expm | f64 | 256x256 | 12.602 ms | 19.97 | 0.08 | 3% | ok | 2.3x (12%) |
| eigen | sylvester | f64 | 256x256 | 175.901 ms | 5.72 | 0.01 | 1% | ok | 1.1x (5%) |
| elementwise | exp | f64 | n=8388608 | 38.869 ms | 0.22 | 3.45 | 5% | ok | 1.0x (5%) |
| elementwise | log | f64 | n=8388608 | 44.821 ms | 0.19 | 2.99 | 4% | ok | 1.0x (5%) |
| elementwise | sin | f64 | n=8388608 | 58.924 ms | 0.14 | 2.28 | 3% | ok | 1.0x (5%) |
| elementwise | sqrt | f64 | n=8388608 | 28.469 ms | 0.29 | 4.71 | 6% | ok | 1.0x (5%) |
| elementwise | erf | f64 | n=8388608 | 200.115 ms | 0.04 | 0.67 | 1% | ok | 1.0x (5%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.150 ms | 17.39 | 180.96 | 247% ! | ok | 4.9x (25%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.912 ms | 2.87 | 29.86 | 41% | ok | 1.0x (5%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.596 ms | 26.25 | 55.18 | 75% | ok | 3.1x (15%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.816 ms | 0.68 | 2.39 | 3%\* | ok | 1.0x (5%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.438 ms | 1.40 | 11.58 | 16%\* | ok | 0.8x (4%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 89.742 ms | 0.12 | 0.01 | 0% | ok | 1.0x (5%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.745 ms | 0.43 | 0.03 | 0% | ok | 1.0x (5%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 9.002 ms | 1.24 | 0.08 | 0% | ok | 1.0x (5%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.427 ms | 0.55 | 0.02 | 0% | ok | 1.0x (5%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 53.524 ms | 0.42 | 0.01 | 0% | ok | 1.0x (5%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 20.696 ms | 1.08 | 0.04 | 0% | ok | 0.9x (4%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 86.097 ms | 0.02 | 0.00 | 0% | ok | 1.0x (5%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.272 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.914 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.223 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.755 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.536 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 32.133 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.423 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.317 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.663 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.749 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.415 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.733 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.014 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 16.915 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.098 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.595 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.235 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.033 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.518 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| pc | ilut | f64 | 128^2 grid | 11.979 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| pc | ic_0 | f64 | 128^2 grid | 1.045 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| pc | ildl | f64 | 128^2 grid | 0.975 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| pc | ssor | f64 | 128^2 grid | 0.020 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.710 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.973 ms | 2.69 | 30.15 | 41% | ok | 1.0x (5%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.201 ms | 2.18 | 24.41 | 33% | ok | 1.0x (5%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.536 ms | 1.70 | 19.10 | 26% | ok | 1.0x (5%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.327 ms | 2.68 | 20.29 | 28% | ok | 0.9x (5%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.204 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 43.944 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 39.157 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.923 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| mixed | dot | f32>f64 | n=8388608 | 4.798 ms | 3.50 | 13.99 | 19% | ok | 1.0x (5%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.551 ms | 58.98 | 0.92 | 9% | ok | 8.4x (42%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.533 ms | 31.48 | 33.45 | 46%\* | ok | 1.0x (5%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.271 ms | 7.39 | 29.56 | n/a | ok | 1.0x (5%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.998 ms | 16.81 | 33.63 | n/a | ok | 1.0x (5%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.684 ms | 9.96 | 9.96 | n/a | ok | 1.0x (5%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.685 ms | 9.96 | 9.96 | n/a | ok | 1.0x (5%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.996 ms | 134.46 | 1.58 | n/a | ok | 8.5x (43%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.386 ms | 112.51 | 0.88 | n/a | ok | 8.6x (43%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.077 ms | 249.23 | 1.46 | n/a | ok | 9.3x (47%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.078 ms | 249.09 | 1.46 | n/a | ok | 9.3x (47%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.031 ms | 88.56 | 0.52 | n/a | ok | 8.7x (44%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

