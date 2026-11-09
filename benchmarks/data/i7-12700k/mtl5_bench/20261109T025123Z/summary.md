# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,20, 2026-11-09T03:00:16Z.

## 12th Gen Intel(R) Core(TM) i7-12700K -- native, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `e7497e6e4c47` |
| FMA peak (ceiling) | f64 79.4 GFLOP/s, f32 158.7 GFLOP/s |
| ... hardware, AVX2+FMA | f64 79.4, f32 158.7 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.5 GFLOP/s |
| bandwidth | read 23.7, update 34.4, triad 24.5 GB/s (ceiling 34.4) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 49% | 110% | 0 |
| blas2 | 6 | 45% | 102% | 0 |
| blas3 | 10 | 18% | 20% | 0 |
| factor | 12 | 8% | 14% | 0 |
| eigen | 8 | 5% | 11% | 0 |
| elementwise | 5 | 9% | 14% | 0 |
| sparse | 5 | 52% | 107% | 0 |
| direct | 7 | 1% | 2% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 71% | 88% | 0 |
| mg | 1 | 65% | 65% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 41% | 97% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.540 ms | 3.03 | 24.23 | 70% | ok |
| blas1 | dot | f32 | n=8388608 | 4.166 ms | 4.03 | 16.11 | 47% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.367 ms | 2.64 | 31.62 | 92% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.704 ms | 3.57 | 21.40 | 62% | ok |
| blas1 | scale | f64 | n=8388608 | 3.547 ms | 2.36 | 37.84 | 110% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.821 ms | 3.48 | 13.92 | 40% | ok |
| blas1 | one_norm | f64 | n=8388608 | 3.979 ms | 2.11 | 16.87 | 49% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.932 ms | 1.21 | 9.68 | 28% | ok |
| blas1 | sum | f64 | n=8388608 | 3.944 ms | 2.13 | 17.01 | 49% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 75.569 ms | 2.44 | 1.78 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 8.307 ms | 4.04 | 16.16 | 47% | ok |
| blas2 | gemv | f32 | n=4096 | 6.835 ms | 4.91 | 9.82 | 29% | ok |
| blas2 | symv | f64 | n=4096 | 6.490 ms | 5.17 | 10.35 | 30% | ok |
| blas2 | trmv | f64 | n=4096 | 4.352 ms | 3.86 | 15.44 | 45% | ok |
| blas2 | trsv | f64 | n=4096 | 4.341 ms | 3.87 | 15.48 | 45% | ok |
| blas2 | ger | f64 | n=4096 | 7.662 ms | 4.38 | 35.04 | 102% | ok |
| blas3 | gemm | f64 | 512x512x512 | 18.354 ms | 14.63 | 0.34 | 18% | ok |
| blas3 | gemm | f32 | 512x512x512 | 9.955 ms | 26.97 | 0.32 | 17% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 9.099 ms | 14.75 | 0.69 | 19% | ok |
| blas3 | symm | f64 | 512x512x512 | 16.712 ms | 16.06 | 0.38 | 20% | ok |
| blas3 | syrk | f64 | 512x512x512 | 11.670 ms | 11.50 | 0.36 | 14% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 19.870 ms | 13.51 | 0.32 | 17% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.179 ms | 14.62 | 0.57 | 18% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.199 ms | 14.59 | 0.57 | 18% | ok |
| blas3 | trsm | f32 | 512x512x512 | 4.944 ms | 27.15 | 0.53 | 17% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.953 ms | 0.41 | 3.28 | 10% | ok |
| factor | lu | f64 | 512x512 | 10.449 ms | 8.56 | 0.40 | 11% | ok |
| factor | lu | f32 | 512x512 | 5.685 ms | 15.74 | 0.37 | 10% | ok |
| factor | cholesky | f64 | 512x512 | 7.071 ms | 6.33 | 0.59 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.450 ms | 6.94 | 0.33 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.276 ms | 6.15 | 0.58 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 44.771 ms | 1.00 | 0.09 | 1% | ok |
| factor | qr | f64 | 512x512 | 216.749 ms | 0.83 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 24.916 ms | 7.18 | 0.17 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 255.577 ms | 1.75 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.518 ms | 0.69 | 2.76 | 8%\* | ok |
| factor | inv | f64 | 512x512 | 63.612 ms | 4.22 | 0.07 | 5% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.994 ms | 11.19 | 0.52 | 14% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 19.084 ms | 1.17 | 0.05 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 68.991 ms | 2.19 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 932.061 ms | 0.16 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 34.387 ms | 4.88 | 0.03 | 6% | ok |
| eigen | schur | f64 | 256x256 | 101.449 ms | 4.13 | 0.01 | 5% | ok |
| eigen | svd | f64 | 256x256 | 111.133 ms | 3.32 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 29.397 ms | 8.56 | 0.04 | 11% | ok |
| eigen | sylvester | f64 | 256x256 | 193.394 ms | 5.21 | 0.01 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 38.719 ms | 0.22 | 3.47 | 10% | ok |
| elementwise | log | f64 | n=8388608 | 44.911 ms | 0.19 | 2.99 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 59.051 ms | 0.14 | 2.27 | 7% | ok |
| elementwise | sqrt | f64 | n=8388608 | 28.532 ms | 0.29 | 4.70 | 14% | ok |
| elementwise | erf | f64 | n=8388608 | 200.815 ms | 0.04 | 0.67 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.740 ms | 3.54 | 36.78 | 107% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.910 ms | 2.87 | 29.91 | 87% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.910 ms | 8.53 | 17.93 | 52% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.834 ms | 0.67 | 2.38 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.221 ms | 1.64 | 13.63 | 40%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 90.027 ms | 0.12 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.403 ms | 0.44 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.794 ms | 1.27 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 42.306 ms | 0.53 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 54.449 ms | 0.41 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 18.125 ms | 1.23 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 88.658 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.294 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.855 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.219 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.755 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.492 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 32.185 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.574 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.262 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.637 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.732 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.307 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.728 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.030 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 16.822 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.066 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.467 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 20.826 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.033 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.515 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.100 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.038 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 0.974 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.020 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.711 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.972 ms | 2.69 | 30.17 | 88% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.201 ms | 2.18 | 24.42 | 71% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.560 ms | 1.68 | 18.80 | 55% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.087 ms | 2.94 | 22.21 | 65% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.260 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 44.052 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 39.511 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.964 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.801 ms | 3.49 | 13.98 | 41% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 38.238 ms | 7.02 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.535 ms | 31.38 | 33.34 | 97%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.284 ms | 7.35 | 29.39 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.994 ms | 16.88 | 33.76 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.685 ms | 9.95 | 9.95 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.685 ms | 9.96 | 9.96 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.064 ms | 15.73 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.443 ms | 13.13 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.037 ms | 26.75 | 0.16 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.031 ms | 26.76 | 0.16 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 26.445 ms | 10.15 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

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

## 12th Gen Intel(R) Core(TM) i7-12700K -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `e7497e6e4c47` |
| FMA peak (ceiling) | f64 79.5 GFLOP/s, f32 159.2 GFLOP/s |
| ... hardware, AVX2+FMA | f64 79.5, f32 159.2 GFLOP/s |
| ... this build, AVX2 | f64 79.3, f32 159.1 GFLOP/s |
| bandwidth | read 35.0, update 36.9, triad 29.3 GB/s (ceiling 36.9) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 100% | 116% | 0 |
| blas2 | 6 | 103% | 109% | 0 |
| blas3 | 10 | 57% | 90% | 0 |
| factor | 12 | 8% | 19% | 0 |
| eigen | 8 | 7% | 26% | 0 |
| elementwise | 5 | 8% | 13% | 0 |
| sparse | 5 | 59% | 100% | 0 |
| direct | 7 | 1% | 2% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 62% | 81% | 0 |
| mg | 1 | 55% | 55% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 95% | 159% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 3.642 ms | 4.61 | 36.85 | 100% | ok |
| blas1 | dot | f32 | n=8388608 | 1.679 ms | 9.99 | 39.96 | 108% | ok |
| blas1 | axpy | f64 | n=8388608 | 4.752 ms | 3.53 | 42.37 | 115% | ok |
| blas1 | axpy | f32 | n=8388608 | 2.341 ms | 7.17 | 43.00 | 116% | ok |
| blas1 | scale | f64 | n=8388608 | 3.452 ms | 2.43 | 38.88 | 105% | ok |
| blas1 | two_norm | f64 | n=8388608 | 2.199 ms | 7.63 | 30.52 | 83% | ok |
| blas1 | one_norm | f64 | n=8388608 | 3.992 ms | 2.10 | 16.81 | 46% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.959 ms | 1.21 | 9.64 | 26% | ok |
| blas1 | sum | f64 | n=8388608 | 3.956 ms | 2.12 | 16.96 | 46% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 70.121 ms | 2.63 | 1.91 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.369 ms | 9.96 | 39.86 | 108% | ok |
| blas2 | gemv | f32 | n=4096 | 1.662 ms | 20.19 | 40.40 | 109% | ok |
| blas2 | symv | f64 | n=4096 | 6.173 ms | 5.44 | 10.88 | 29% | ok |
| blas2 | trmv | f64 | n=4096 | 1.762 ms | 9.52 | 38.13 | 103% | ok |
| blas2 | trsv | f64 | n=4096 | 1.794 ms | 9.35 | 37.43 | 101% | ok |
| blas2 | ger | f64 | n=4096 | 7.124 ms | 4.71 | 37.69 | 102% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.752 ms | 71.55 | 1.68 | 90% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.882 ms | 142.63 | 1.67 | 90% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.606 ms | 51.49 | 2.41 | 65% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.991 ms | 67.26 | 1.58 | 85% | ok |
| blas3 | syrk | f64 | 512x512x512 | 10.270 ms | 13.07 | 0.41 | 16% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 17.340 ms | 15.48 | 0.36 | 19% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.979 ms | 45.06 | 1.76 | 57% | ok |
| blas3 | trsm | f64 | 512x512x512 | 3.056 ms | 43.92 | 1.72 | 55% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.497 ms | 89.64 | 1.75 | 56% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.948 ms | 0.41 | 3.28 | 9% | ok |
| factor | lu | f64 | 512x512 | 10.277 ms | 8.71 | 0.41 | 11% | ok |
| factor | lu | f32 | 512x512 | 3.656 ms | 24.48 | 0.57 | 15% | ok |
| factor | cholesky | f64 | 512x512 | 6.902 ms | 6.48 | 0.61 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.454 ms | 6.93 | 0.32 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.059 ms | 6.34 | 0.59 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 44.039 ms | 1.02 | 0.10 | 1% | ok |
| factor | qr | f64 | 512x512 | 213.335 ms | 0.84 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 22.619 ms | 7.91 | 0.19 | 10% | ok |
| factor | hessenberg | f64 | 512x512 | 251.907 ms | 1.78 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.500 ms | 0.70 | 2.80 | 8%\* | ok |
| factor | inv | f64 | 512x512 | 63.004 ms | 4.26 | 0.07 | 5% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.004 ms | 14.90 | 0.70 | 19% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 16.429 ms | 1.36 | 0.06 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 56.560 ms | 2.67 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 746.278 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 30.163 ms | 5.56 | 0.03 | 7% | ok |
| eigen | schur | f64 | 256x256 | 78.935 ms | 5.31 | 0.01 | 7% | ok |
| eigen | svd | f64 | 256x256 | 90.369 ms | 4.08 | 0.01 | 5% | ok |
| eigen | expm | f64 | 256x256 | 12.306 ms | 20.45 | 0.09 | 26% | ok |
| eigen | sylvester | f64 | 256x256 | 174.692 ms | 5.76 | 0.01 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 38.764 ms | 0.22 | 3.46 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 46.015 ms | 0.18 | 2.92 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 58.790 ms | 0.14 | 2.28 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 28.525 ms | 0.29 | 4.71 | 13% | ok |
| elementwise | erf | f64 | n=8388608 | 201.148 ms | 0.04 | 0.67 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.734 ms | 3.56 | 37.07 | 100% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.922 ms | 2.84 | 29.53 | 80% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.032 ms | 10.39 | 21.84 | 59% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.793 ms | 0.68 | 2.40 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.222 ms | 1.64 | 13.62 | 37%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 71.264 ms | 0.16 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 26.851 ms | 0.42 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.842 ms | 1.27 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.262 ms | 0.56 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 55.531 ms | 0.40 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.562 ms | 1.35 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 88.892 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.362 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.922 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.221 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.041 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 27.630 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 28.153 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 20.512 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.447 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.672 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.562 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 20.223 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.564 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.415 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.542 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 14.823 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 15.826 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 18.029 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.032 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.529 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 11.810 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.065 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 0.982 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.691 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.976 ms | 2.68 | 30.06 | 81% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.286 ms | 2.04 | 22.80 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.506 ms | 1.74 | 19.47 | 53% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.210 ms | 2.71 | 20.45 | 55% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 27.754 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 38.760 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 39.083 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.512 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 1.921 ms | 8.73 | 34.93 | 95% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 5.986 ms | 44.84 | 0.70 | 56% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.304 ms | 55.11 | 58.55 | 159%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.914 ms | 8.77 | 35.07 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.753 ms | 22.29 | 44.57 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.275 ms | 60.94 | 60.94 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.255 ms | 65.75 | 65.75 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 3.942 ms | 68.10 | 0.80 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.523 ms | 48.60 | 0.38 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.499 ms | 48.81 | 0.29 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 2.084 ms | 128.80 | 0.75 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 1.860 ms | 144.35 | 0.85 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## 12th Gen Intel(R) Core(TM) i7-12700K -- native-fast, 20 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `e7497e6e4c47` |
| FMA peak (ceiling) | f64 668.5 GFLOP/s, f32 1260.4 GFLOP/s |
| ... hardware, AVX2+FMA | f64 629.7, f32 1259.4 GFLOP/s |
| ... this build, AVX2 | f64 668.5, f32 1260.4 GFLOP/s |
| bandwidth | read 72.6, update 71.5, triad 64.5 GB/s (ceiling 72.6) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 106% | 166% | 0 | 9% |
| blas2 | 6 | 52% | 164% | 0 | 5% |
| blas3 | 10 | 10% | 42% | 0 | 7% |
| factor | 12 | 1% | 12% | 0 | 5% |
| eigen | 8 | 1% | 3% | 0 | 5% |
| elementwise | 5 | 4% | 6% | 0 | 5% |
| sparse | 5 | 41% | 351% | 0 | 5% |
| direct | 7 | 0% | 0% | 0 | 5% |
| ordering | 4 | n/a | n/a | 0 | 5% |
| krylov | 13 | n/a | n/a | 0 | 5% |
| pc | 7 | n/a | n/a | 0 | 5% |
| smoother | 3 | 31% | 42% | 0 | 5% |
| mg | 1 | 26% | 26% | 0 | 5% |
| sparse_eig | 4 | n/a | n/a | 0 | 5% |
| mixed | 3 | 49% | 81% | 0 | 5% |
| integer | 9 | n/a | n/a | 0 | 15% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.644 ms | 10.20 | 81.62 | 112% | ok | 2.2x (11%) |
| blas1 | dot | f32 | n=8388608 | 0.567 ms | 29.57 | 118.27 | 163% ! | ok | 3.0x (15%) |
| blas1 | axpy | f64 | n=8388608 | 2.711 ms | 6.19 | 74.26 | 102% | ok | 1.8x (9%) |
| blas1 | axpy | f32 | n=8388608 | 1.259 ms | 13.32 | 79.93 | 110% | ok | 1.9x (9%) |
| blas1 | scale | f64 | n=8388608 | 1.746 ms | 4.81 | 76.89 | 106% | ok | 2.0x (10%) |
| blas1 | two_norm | f64 | n=8388608 | 0.557 ms | 30.11 | 120.45 | 166% ! | ok | 3.9x (20%) |
| blas1 | one_norm | f64 | n=8388608 | 3.985 ms | 2.11 | 16.84 | 23% | ok | 1.0x (5%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.939 ms | 1.21 | 9.67 | 13% | ok | 1.0x (5%) |
| blas1 | sum | f64 | n=8388608 | 3.954 ms | 2.12 | 16.97 | 23% | ok | 1.0x (5%) |
| blas1 | dot_k3 | f64 | n=8388608 | 70.123 ms | 2.63 | 1.91 | 3% | ok | 1.0x (5%) |
| blas2 | gemv | f64 | n=4096 | 1.598 ms | 21.00 | 84.03 | 116% | ok | 2.1x (11%) |
| blas2 | gemv | f32 | n=4096 | 0.564 ms | 59.53 | 119.13 | 164% ! | ok | 2.9x (15%) |
| blas2 | symv | f64 | n=4096 | 6.216 ms | 5.40 | 10.81 | 15% | ok | 1.0x (5%) |
| blas2 | trmv | f64 | n=4096 | 1.764 ms | 9.51 | 38.09 | 52% | ok | 1.0x (5%) |
| blas2 | trsv | f64 | n=4096 | 1.810 ms | 9.27 | 37.12 | 51% | ok | 1.0x (5%) |
| blas2 | ger | f64 | n=4096 | 7.142 ms | 4.70 | 37.59 | 52% | ok | 1.0x (5%) |
| blas3 | gemm | f64 | 512x512x512 | 1.569 ms | 171.08 | 4.01 | 26% | ok | 2.4x (12%) |
| blas3 | gemm | f32 | 512x512x512 | 0.502 ms | 534.26 | 6.26 | 42% | ok | 3.7x (19%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.832 ms | 73.27 | 3.43 | 11% | ok | 1.4x (7%) |
| blas3 | symm | f64 | 512x512x512 | 1.478 ms | 181.68 | 4.26 | 27% | ok | 2.7x (14%) |
| blas3 | syrk | f64 | 512x512x512 | 11.426 ms | 11.75 | 0.37 | 2% | ok | 0.9x (4%) |
| blas3 | syr2k | f64 | 512x512x512 | 17.816 ms | 15.07 | 0.35 | 2% | ok | 1.0x (5%) |
| blas3 | trmm | f64 | 512x512x512 | 2.416 ms | 55.56 | 2.17 | 8% | ok | 1.2x (6%) |
| blas3 | trsm | f64 | 512x512x512 | 2.144 ms | 62.59 | 2.44 | 9% | ok | 1.4x (7%) |
| blas3 | trsm | f32 | 512x512x512 | 1.065 ms | 126.01 | 2.46 | 10% | ok | 1.4x (7%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.955 ms | 0.41 | 3.28 | 5% | ok | 1.0x (5%) |
| factor | lu | f64 | 512x512 | 8.639 ms | 10.36 | 0.49 | 2% | ok | 1.2x (6%) |
| factor | lu | f32 | 512x512 | 4.264 ms | 20.98 | 0.49 | 2% | ok | 0.9x (4%) |
| factor | cholesky | f64 | 512x512 | 6.975 ms | 6.41 | 0.60 | 1% | ok | 1.0x (5%) |
| factor | cholesky | f32 | 512x512 | 6.415 ms | 6.97 | 0.33 | 1% | ok | 1.0x (5%) |
| factor | ldlt | f64 | 512x512 | 7.077 ms | 6.32 | 0.59 | 1% | ok | 1.0x (5%) |
| factor | ldlt_bk | f64 | 512x512 | 48.645 ms | 0.92 | 0.09 | 0% | ok | 0.9x (5%) |
| factor | qr | f64 | 512x512 | 153.122 ms | 1.17 | 0.03 | 0% | ok | 1.4x (7%) |
| factor | lq | f64 | 512x512 | 16.694 ms | 10.72 | 0.25 | 2% | ok | 1.4x (7%) |
| factor | hessenberg | f64 | 512x512 | 187.219 ms | 2.39 | 0.02 | 0% | ok | 1.3x (7%) |
| factor | householder | f64 | 512x512 | 0.483 ms | 2.17 | 8.68 | 12%\* | ok | 3.1x (16%) |
| factor | inv | f64 | 512x512 | 62.205 ms | 4.32 | 0.07 | 1% | ok | 1.0x (5%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.940 ms | 12.89 | 0.60 | 2% | ok | 0.9x (4%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 16.544 ms | 1.35 | 0.06 | 0% | ok | 1.0x (5%) |
| eigen | eigen_symmetric | f64 | 256x256 | 58.728 ms | 2.57 | 0.02 | 0% | ok | 1.0x (5%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 760.714 ms | 0.20 | 0.00 | 0% | ok | 1.0x (5%) |
| eigen | eigenvalue | f64 | 256x256 | 30.422 ms | 5.51 | 0.03 | 1% | ok | 1.0x (5%) |
| eigen | schur | f64 | 256x256 | 79.354 ms | 5.29 | 0.01 | 1% | ok | 1.0x (5%) |
| eigen | svd | f64 | 256x256 | 93.669 ms | 3.94 | 0.01 | 1% | ok | 1.0x (5%) |
| eigen | expm | f64 | 256x256 | 11.902 ms | 21.14 | 0.09 | 3% | ok | 1.0x (5%) |
| eigen | sylvester | f64 | 256x256 | 171.339 ms | 5.88 | 0.01 | 1% | ok | 1.0x (5%) |
| elementwise | exp | f64 | n=8388608 | 38.796 ms | 0.22 | 3.46 | 5% | ok | 1.0x (5%) |
| elementwise | log | f64 | n=8388608 | 45.838 ms | 0.18 | 2.93 | 4% | ok | 1.0x (5%) |
| elementwise | sin | f64 | n=8388608 | 59.093 ms | 0.14 | 2.27 | 3% | ok | 1.0x (5%) |
| elementwise | sqrt | f64 | n=8388608 | 28.571 ms | 0.29 | 4.70 | 6% | ok | 1.0x (5%) |
| elementwise | erf | f64 | n=8388608 | 201.236 ms | 0.04 | 0.67 | 1% | ok | 1.0x (5%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.107 ms | 24.49 | 254.84 | 351% ! | ok | 6.9x (34%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.911 ms | 2.87 | 29.90 | 41% | ok | 1.0x (5%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.567 ms | 26.73 | 56.19 | 77% | ok | 2.6x (13%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.684 ms | 0.70 | 2.46 | 3%\* | ok | 1.0x (5%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.442 ms | 1.39 | 11.55 | 16%\* | ok | 0.8x (4%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 71.656 ms | 0.16 | 0.01 | 0% | ok | 1.0x (5%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 27.790 ms | 0.40 | 0.03 | 0% | ok | 1.0x (5%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 9.391 ms | 1.19 | 0.08 | 0% | ok | 0.9x (5%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.139 ms | 0.56 | 0.02 | 0% | ok | 1.0x (5%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 56.232 ms | 0.40 | 0.01 | 0% | ok | 1.0x (5%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 21.086 ms | 1.06 | 0.03 | 0% | ok | 0.8x (4%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 86.034 ms | 0.02 | 0.00 | 0% | ok | 1.0x (5%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.382 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.881 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.223 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.884 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.245 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 28.448 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 20.812 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.352 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.665 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.520 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 20.186 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.557 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.489 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.515 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 14.731 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 15.768 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 17.903 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.032 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.516 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| pc | ilut | f64 | 128^2 grid | 11.823 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| pc | ic_0 | f64 | 128^2 grid | 1.056 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| pc | ildl | f64 | 128^2 grid | 1.043 ms | 0.00 | 0.00 | n/a | ok | 0.9x (5%) |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.692 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.973 ms | 2.69 | 30.15 | 42% | ok | 1.0x (5%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.300 ms | 2.01 | 22.56 | 31% | ok | 1.0x (5%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.484 ms | 1.76 | 19.76 | 27% | ok | 1.0x (5%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.128 ms | 2.54 | 19.21 | 26% | ok | 0.9x (5%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 27.871 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 38.711 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 39.177 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.567 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| mixed | dot | f32>f64 | n=8388608 | 1.889 ms | 8.88 | 35.52 | 49% | ok | 1.0x (5%) |
| mixed | gemm | f32>f64 | 512x512x512 | 1.396 ms | 192.30 | 3.00 | 29% | ok | 4.3x (21%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.303 ms | 55.33 | 58.79 | 81%\* | ok | 1.0x (5%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.915 ms | 8.76 | 35.05 | n/a | ok | 1.0x (5%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.746 ms | 22.48 | 44.95 | n/a | ok | 1.0x (5%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.268 ms | 62.55 | 62.55 | n/a | ok | 1.0x (5%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.253 ms | 66.24 | 66.24 | n/a | ok | 1.0x (5%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.280 ms | 209.76 | 2.46 | n/a | ok | 3.1x (15%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 1.411 ms | 190.30 | 1.49 | n/a | ok | 3.9x (20%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.386 ms | 193.67 | 1.13 | n/a | ok | 4.0x (20%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.499 ms | 537.68 | 3.15 | n/a | ok | 4.2x (21%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.457 ms | 587.70 | 3.44 | n/a | ok | 4.1x (20%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

## 12th Gen Intel(R) Core(TM) i7-12700K -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `e7497e6e4c47` |
| FMA peak (ceiling) | f64 79.5 GFLOP/s, f32 159.2 GFLOP/s |
| ... hardware, AVX2+FMA | f64 79.5, f32 159.2 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.3 GFLOP/s |
| bandwidth | read 27.0, update 36.2, triad 30.6 GB/s (ceiling 36.2) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 103% | 125% | 0 |
| blas2 | 6 | 106% | 118% | 0 |
| blas3 | 10 | 80% | 93% | 0 |
| factor | 12 | 24% | 40% | 0 |
| eigen | 8 | 24% | 70% | 0 |
| elementwise | 5 | 8% | 13% | 0 |
| sparse | 5 | 53% | 101% | 0 |
| direct | 7 | 1% | 2% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 67% | 83% | 0 |
| mg | 1 | 61% | 61% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 38% | 90% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 3.608 ms | 4.65 | 37.20 | 103% | ok |
| blas1 | dot | f32 | n=8388608 | 1.677 ms | 10.00 | 40.01 | 110% | ok |
| blas1 | axpy | f64 | n=8388608 | 4.568 ms | 3.67 | 44.08 | 122% | ok |
| blas1 | axpy | f32 | n=8388608 | 2.220 ms | 7.56 | 45.34 | 125% | ok |
| blas1 | scale | f64 | n=8388608 | 3.206 ms | 2.62 | 41.86 | 115% | ok |
| blas1 | two_norm | f64 | n=8388608 | 3.417 ms | 4.91 | 19.64 | 54% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.005 ms | 2.09 | 16.76 | 46% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.962 ms | 1.20 | 9.64 | 27% | ok |
| blas1 | sum | f64 | n=8388608 | 3.955 ms | 2.12 | 16.97 | 47% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 76.158 ms | 2.42 | 1.76 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.490 ms | 9.61 | 38.48 | 106% | ok |
| blas2 | gemv | f32 | n=4096 | 1.572 ms | 21.34 | 42.70 | 118% | ok |
| blas2 | symv | f64 | n=4096 | 1.894 ms | 17.72 | 35.47 | 98% | ok |
| blas2 | trmv | f64 | n=4096 | 1.830 ms | 9.17 | 36.72 | 101% | ok |
| blas2 | trsv | f64 | n=4096 | 1.754 ms | 9.57 | 38.30 | 106% | ok |
| blas2 | ger | f64 | n=4096 | 7.669 ms | 4.38 | 35.01 | 97% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.687 ms | 72.81 | 1.71 | 92% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.822 ms | 147.34 | 1.73 | 93% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.569 ms | 52.24 | 2.45 | 66% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.694 ms | 72.68 | 1.70 | 91% | ok |
| blas3 | syrk | f64 | 512x512x512 | 2.525 ms | 53.17 | 1.66 | 67% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 4.697 ms | 57.15 | 1.34 | 72% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.103 ms | 63.81 | 2.49 | 80% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.120 ms | 63.31 | 2.47 | 80% | ok |
| blas3 | trsm | f32 | 512x512x512 | 0.957 ms | 140.27 | 2.74 | 88% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.848 ms | 0.41 | 3.31 | 9% | ok |
| factor | lu | f64 | 512x512 | 3.191 ms | 28.04 | 1.31 | 35% | ok |
| factor | lu | f32 | 512x512 | 1.750 ms | 51.13 | 1.20 | 32% | ok |
| factor | cholesky | f64 | 512x512 | 1.970 ms | 22.71 | 2.13 | 29% | ok |
| factor | cholesky | f32 | 512x512 | 1.163 ms | 38.46 | 1.80 | 24% | ok |
| factor | ldlt | f64 | 512x512 | 7.151 ms | 6.26 | 0.59 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 44.234 ms | 1.01 | 0.09 | 1% | ok |
| factor | qr | f64 | 512x512 | 5.641 ms | 31.72 | 0.74 | 40% | ok |
| factor | lq | f64 | 512x512 | 25.168 ms | 7.11 | 0.17 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 249.235 ms | 1.80 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.520 ms | 0.69 | 2.76 | 8%\* | ok |
| factor | inv | f64 | 512x512 | 55.740 ms | 4.82 | 0.08 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 4.054 ms | 22.07 | 1.03 | 28% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.699 ms | 13.16 | 0.62 | 17% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 57.876 ms | 2.61 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 739.289 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 16.043 ms | 10.46 | 0.07 | 13% | ok |
| eigen | schur | f64 | 256x256 | 20.892 ms | 20.08 | 0.05 | 25% | ok |
| eigen | svd | f64 | 256x256 | 6.613 ms | 55.82 | 0.16 | 70% | ok |
| eigen | expm | f64 | 256x256 | 12.160 ms | 20.70 | 0.09 | 26% | ok |
| eigen | sylvester | f64 | 256x256 | 52.594 ms | 19.14 | 0.03 | 24% | ok |
| elementwise | exp | f64 | n=8388608 | 38.779 ms | 0.22 | 3.46 | 10% | ok |
| elementwise | log | f64 | n=8388608 | 44.800 ms | 0.19 | 3.00 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 58.758 ms | 0.14 | 2.28 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 28.501 ms | 0.29 | 4.71 | 13% | ok |
| elementwise | erf | f64 | n=8388608 | 199.816 ms | 0.04 | 0.67 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.742 ms | 3.53 | 36.69 | 101% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.916 ms | 2.86 | 29.74 | 82% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.603 ms | 9.10 | 19.13 | 53% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.819 ms | 0.68 | 2.39 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.208 ms | 1.66 | 13.78 | 38%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 89.645 ms | 0.12 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 24.970 ms | 0.45 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.711 ms | 1.28 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.329 ms | 0.55 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 53.382 ms | 0.42 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 20.505 ms | 1.09 | 0.04 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 86.387 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.297 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.869 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.221 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.851 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.571 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 28.596 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 21.659 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.178 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.498 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.091 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 20.569 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 23.408 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.124 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.424 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 14.665 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 15.535 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 17.828 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.033 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.523 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.049 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.045 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 0.971 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.109 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.976 ms | 2.68 | 30.05 | 83% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.200 ms | 2.18 | 24.43 | 67% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.541 ms | 1.70 | 19.04 | 53% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.078 ms | 2.94 | 22.23 | 61% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.222 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 31.512 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 38.682 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.582 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.814 ms | 3.49 | 13.94 | 38% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 38.441 ms | 6.98 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.544 ms | 30.83 | 32.76 | 90%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.260 ms | 7.42 | 29.69 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.995 ms | 16.86 | 33.73 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.687 ms | 9.94 | 9.94 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.696 ms | 9.89 | 9.89 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.117 ms | 15.68 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.450 ms | 13.13 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.024 ms | 26.78 | 0.16 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.050 ms | 26.71 | 0.16 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 26.425 ms | 10.16 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

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

