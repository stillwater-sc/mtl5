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

