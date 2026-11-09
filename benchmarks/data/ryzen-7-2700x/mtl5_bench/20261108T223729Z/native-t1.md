## AMD Ryzen 7 2700X Eight-Core Processor -- native, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 2700X Eight-Core Processor (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.51.36256 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `e096dbbc6788` |
| FMA peak (ceiling) | f64 31.9 GFLOP/s, f32 63.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 31.9, f32 63.6 GFLOP/s |
| ... this build, scalar | f64 1.2, f32 1.2 GFLOP/s |
| bandwidth | read 21.1, update 23.4, triad 19.4 GB/s (ceiling 23.4) |
| last-level cache | 8 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 40% | 80% | 0 |
| blas2 | 6 | 40% | 48% | 0 |
| blas3 | 10 | 26% | 32% | 0 |
| factor | 12 | 4% | 9% | 0 |
| eigen | 8 | 7% | 13% | 0 |
| elementwise | 5 | 10% | 21% | 0 |
| sparse | 5 | 37% | 86% | 0 |
| direct | 7 | 1% | 1% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 53% | 62% | 0 |
| mg | 1 | 51% | 51% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 20% | 33% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 14.194 ms | 1.18 | 9.46 | 40% | ok |
| blas1 | dot | f32 | n=8388608 | 14.387 ms | 1.17 | 4.66 | 20% | ok |
| blas1 | axpy | f64 | n=8388608 | 16.203 ms | 1.04 | 12.43 | 53% | ok |
| blas1 | axpy | f32 | n=8388608 | 14.882 ms | 1.13 | 6.76 | 29% | ok |
| blas1 | scale | f64 | n=8388608 | 7.206 ms | 1.16 | 18.63 | 80% | ok |
| blas1 | two_norm | f64 | n=8388608 | 14.603 ms | 1.15 | 4.60 | 20% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.570 ms | 1.28 | 10.21 | 44% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 8.846 ms | 0.95 | 7.59 | 32% | ok |
| blas1 | sum | f64 | n=8388608 | 6.636 ms | 1.26 | 10.11 | 43% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 116.474 ms | 1.58 | 1.15 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 14.252 ms | 2.35 | 9.42 | 40% | ok |
| blas2 | gemv | f32 | n=4096 | 13.097 ms | 2.56 | 5.13 | 22% | ok |
| blas2 | symv | f64 | n=4096 | 9.995 ms | 3.36 | 6.72 | 29% | ok |
| blas2 | trmv | f64 | n=4096 | 7.160 ms | 2.34 | 9.38 | 40% | ok |
| blas2 | trsv | f64 | n=4096 | 6.939 ms | 2.42 | 9.68 | 41% | ok |
| blas2 | ger | f64 | n=4096 | 24.014 ms | 1.40 | 11.18 | 48% | ok |
| blas3 | gemm | f64 | 512x512x512 | 28.436 ms | 9.44 | 0.22 | 30% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.307 ms | 20.17 | 0.24 | 32% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 15.912 ms | 8.43 | 0.40 | 26% | ok |
| blas3 | symm | f64 | 512x512x512 | 28.809 ms | 9.32 | 0.22 | 29% | ok |
| blas3 | syrk | f64 | 512x512x512 | 18.498 ms | 7.26 | 0.23 | 23% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 29.410 ms | 9.13 | 0.21 | 29% | ok |
| blas3 | trmm | f64 | 512x512x512 | 45.065 ms | 2.98 | 0.12 | 9% | ok |
| blas3 | trsm | f64 | 512x512x512 | 45.826 ms | 2.93 | 0.11 | 9% | ok |
| blas3 | trsm | f32 | 512x512x512 | 50.882 ms | 2.64 | 0.05 | 4% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 19.056 ms | 0.28 | 2.23 | 10% | ok |
| factor | lu | f64 | 512x512 | 148.661 ms | 0.60 | 0.03 | 2% | ok |
| factor | lu | f32 | 512x512 | 150.947 ms | 0.59 | 0.01 | 1% | ok |
| factor | cholesky | f64 | 512x512 | 16.425 ms | 2.72 | 0.26 | 9% | ok |
| factor | cholesky | f32 | 512x512 | 16.857 ms | 2.65 | 0.12 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 16.489 ms | 2.71 | 0.25 | 9% | ok |
| factor | ldlt_bk | f64 | 512x512 | 159.439 ms | 0.28 | 0.03 | 1% | ok |
| factor | qr | f64 | 512x512 | 512.832 ms | 0.35 | 0.01 | 1% | ok |
| factor | lq | f64 | 512x512 | 154.419 ms | 1.16 | 0.03 | 4% | ok |
| factor | hessenberg | f64 | 512x512 | 685.062 ms | 0.65 | 0.01 | 2% | ok |
| factor | householder | f64 | 512x512 | 3.286 ms | 0.32 | 1.28 | 5%\* | ok |
| factor | inv | f64 | 512x512 | 171.541 ms | 1.56 | 0.02 | 5% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 64.490 ms | 1.39 | 0.07 | 4% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 33.554 ms | 0.67 | 0.03 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 111.391 ms | 1.36 | 0.01 | 4% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 586.004 ms | 0.26 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 75.484 ms | 2.22 | 0.01 | 7% | ok |
| eigen | schur | f64 | 256x256 | 173.144 ms | 2.42 | 0.01 | 8% | ok |
| eigen | svd | f64 | 256x256 | 179.009 ms | 2.06 | 0.01 | 6% | ok |
| eigen | expm | f64 | 256x256 | 61.162 ms | 4.11 | 0.02 | 13% | ok |
| eigen | sylvester | f64 | 256x256 | 434.842 ms | 2.31 | 0.00 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 57.329 ms | 0.15 | 2.34 | 10% | ok |
| elementwise | log | f64 | n=8388608 | 75.034 ms | 0.11 | 1.79 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 56.559 ms | 0.15 | 2.37 | 10% | ok |
| elementwise | sqrt | f64 | n=8388608 | 27.251 ms | 0.31 | 4.93 | 21% | ok |
| elementwise | erf | f64 | n=8388608 | 60.953 ms | 0.14 | 2.20 | 9% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.359 ms | 1.93 | 20.04 | 86% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.144 ms | 0.83 | 8.66 | 37% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 7.615 ms | 5.50 | 11.56 | 49% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 21.180 ms | 0.15 | 0.54 | 2% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 2.961 ms | 0.68 | 5.62 | 24% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 185.719 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 53.487 ms | 0.21 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 34.282 ms | 0.33 | 0.02 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 112.160 ms | 0.20 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 121.169 ms | 0.18 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 46.807 ms | 0.48 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 353.341 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 5.922 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 9.345 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.779 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 27.218 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 53.128 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 67.238 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 44.273 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 22.740 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 33.048 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 23.994 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 78.125 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 48.771 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 27.843 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 44.966 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 55.468 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 84.429 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 72.799 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.061 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.382 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 46.639 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 3.400 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 3.559 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.049 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.133 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.358 ms | 1.11 | 12.44 | 53% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.032 ms | 1.29 | 14.43 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.376 ms | 1.10 | 12.35 | 53% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 24.197 ms | 1.59 | 12.01 | 51% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 63.165 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 69.704 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 126.893 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.852 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 14.221 ms | 1.18 | 4.72 | 20% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 53.319 ms | 5.03 | 0.08 | 16% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.300 ms | 7.29 | 7.75 | 33% | ok |
| integer | dot | i32 | n=8388608, |v| <= 15 | 4.358 ms | 3.85 | 15.40 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, |v| <= 15 | 3.988 ms | 4.21 | 8.41 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, |v| <= 15 | 2.635 ms | 6.37 | 6.37 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, |v| <= 15 | 2.236 ms | 7.50 | 7.50 | n/a | ok |
| integer | gemm | i32 | 512^3, |v| <= 2047 | 18.349 ms | 14.63 | 0.17 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, |v| <= 2047 | 47.401 ms | 5.66 | 0.04 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, |v| <= 127 | 38.913 ms | 6.90 | 0.04 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, |v| <= 127 | 39.613 ms | 6.78 | 0.04 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, |v| <= 127 | 38.405 ms | 6.99 | 0.04 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

