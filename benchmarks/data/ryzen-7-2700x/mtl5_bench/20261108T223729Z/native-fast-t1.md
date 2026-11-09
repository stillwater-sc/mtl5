## AMD Ryzen 7 2700X Eight-Core Processor -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 2700X Eight-Core Processor (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.51.36256 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `e096dbbc6788` |
| FMA peak (ceiling) | f64 31.9 GFLOP/s, f32 64.2 GFLOP/s |
| ... hardware, AVX2+FMA | f64 31.9, f32 64.2 GFLOP/s |
| ... this build, SSE2 | f64 31.9, f32 63.9 GFLOP/s |
| bandwidth | read 25.1, update 21.0, triad 18.3 GB/s (ceiling 25.1) |
| last-level cache | 8 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 85% | 106% | 0 |
| blas2 | 6 | 89% | 98% | 0 |
| blas3 | 10 | 27% | 42% | 0 |
| factor | 12 | 3% | 9% | 0 |
| eigen | 8 | 6% | 16% | 0 |
| elementwise | 5 | 9% | 19% | 0 |
| sparse | 5 | 34% | 85% | 0 |
| direct | 7 | 1% | 1% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 52% | 58% | 0 |
| mg | 1 | 47% | 47% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 79% | 84% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.268 ms | 3.18 | 25.48 | 101% | ok |
| blas1 | dot | f32 | n=8388608 | 2.517 ms | 6.67 | 26.66 | 106% | ok |
| blas1 | axpy | f64 | n=8388608 | 8.908 ms | 1.88 | 22.60 | 90% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.689 ms | 3.58 | 21.47 | 85% | ok |
| blas1 | scale | f64 | n=8388608 | 6.799 ms | 1.23 | 19.74 | 79% | ok |
| blas1 | two_norm | f64 | n=8388608 | 2.823 ms | 5.94 | 23.77 | 95% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.628 ms | 1.27 | 10.12 | 40% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 22.107 ms | 0.38 | 3.04 | 12% | ok |
| blas1 | sum | f64 | n=8388608 | 6.638 ms | 1.26 | 10.11 | 40% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 119.442 ms | 1.55 | 1.12 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 5.478 ms | 6.12 | 24.51 | 98% | ok |
| blas2 | gemv | f32 | n=4096 | 2.811 ms | 11.94 | 23.89 | 95% | ok |
| blas2 | symv | f64 | n=4096 | 10.340 ms | 3.25 | 6.50 | 26% | ok |
| blas2 | trmv | f64 | n=4096 | 2.997 ms | 5.60 | 22.41 | 89% | ok |
| blas2 | trsv | f64 | n=4096 | 3.172 ms | 5.29 | 21.18 | 84% | ok |
| blas2 | ger | f64 | n=4096 | 24.293 ms | 1.38 | 11.05 | 44% | ok |
| blas3 | gemm | f64 | 512x512x512 | 21.348 ms | 12.57 | 0.29 | 39% | ok |
| blas3 | gemm | f32 | 512x512x512 | 9.929 ms | 27.04 | 0.32 | 42% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 13.463 ms | 9.97 | 0.47 | 31% | ok |
| blas3 | symm | f64 | 512x512x512 | 22.346 ms | 12.01 | 0.28 | 38% | ok |
| blas3 | syrk | f64 | 512x512x512 | 19.065 ms | 7.04 | 0.22 | 22% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 30.963 ms | 8.67 | 0.20 | 27% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.742 ms | 7.56 | 0.30 | 24% | ok |
| blas3 | trsm | f64 | 512x512x512 | 21.624 ms | 6.21 | 0.24 | 19% | ok |
| blas3 | trsm | f32 | 512x512x512 | 11.764 ms | 11.41 | 0.22 | 18% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 16.838 ms | 0.32 | 2.52 | 10% | ok |
| factor | lu | f64 | 512x512 | 149.146 ms | 0.60 | 0.03 | 2% | ok |
| factor | lu | f32 | 512x512 | 51.625 ms | 1.73 | 0.04 | 3% | ok |
| factor | cholesky | f64 | 512x512 | 16.437 ms | 2.72 | 0.26 | 9% | ok |
| factor | cholesky | f32 | 512x512 | 16.436 ms | 2.72 | 0.13 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 16.202 ms | 2.76 | 0.26 | 9% | ok |
| factor | ldlt_bk | f64 | 512x512 | 159.488 ms | 0.28 | 0.03 | 1% | ok |
| factor | qr | f64 | 512x512 | 511.967 ms | 0.35 | 0.01 | 1% | ok |
| factor | lq | f64 | 512x512 | 172.004 ms | 1.04 | 0.02 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 678.472 ms | 0.66 | 0.01 | 2% | ok |
| factor | householder | f64 | 512x512 | 3.289 ms | 0.32 | 1.28 | 5%\* | ok |
| factor | inv | f64 | 512x512 | 206.977 ms | 1.30 | 0.02 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 153.007 ms | 0.58 | 0.03 | 2% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.822 ms | 0.56 | 0.03 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 113.651 ms | 1.33 | 0.01 | 4% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 610.571 ms | 0.25 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 68.393 ms | 2.45 | 0.02 | 8% | ok |
| eigen | schur | f64 | 256x256 | 220.307 ms | 1.90 | 0.00 | 6% | ok |
| eigen | svd | f64 | 256x256 | 183.620 ms | 2.01 | 0.01 | 6% | ok |
| eigen | expm | f64 | 256x256 | 50.017 ms | 5.03 | 0.02 | 16% | ok |
| eigen | sylvester | f64 | 256x256 | 393.788 ms | 2.56 | 0.00 | 8% | ok |
| elementwise | exp | f64 | n=8388608 | 61.654 ms | 0.14 | 2.18 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 70.100 ms | 0.12 | 1.91 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 59.464 ms | 0.14 | 2.26 | 9% | ok |
| elementwise | sqrt | f64 | n=8388608 | 28.556 ms | 0.29 | 4.70 | 19% | ok |
| elementwise | erf | f64 | n=8388608 | 61.858 ms | 0.14 | 2.17 | 9% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.270 ms | 2.06 | 21.44 | 85% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.167 ms | 0.83 | 8.60 | 34% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 7.504 ms | 5.58 | 11.73 | 47% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 21.023 ms | 0.15 | 0.55 | 2% | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 2.982 ms | 0.67 | 5.58 | 22% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 172.342 ms | 0.06 | 0.00 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 52.812 ms | 0.21 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 32.804 ms | 0.34 | 0.02 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 100.850 ms | 0.22 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 127.068 ms | 0.18 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 53.139 ms | 0.42 | 0.01 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 402.178 ms | 0.00 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 6.007 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 9.421 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.877 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 26.042 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 43.186 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 51.599 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 38.306 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 18.357 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 25.135 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 21.382 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 53.390 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 38.915 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.594 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 34.267 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 36.995 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 54.484 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 45.719 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.074 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.586 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 43.890 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 4.183 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 3.554 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.039 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.180 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.253 ms | 1.16 | 13.02 | 52% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.010 ms | 1.30 | 14.59 | 58% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.393 ms | 1.09 | 12.26 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 24.617 ms | 1.56 | 11.81 | 47% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 51.213 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 70.644 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 119.792 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 10.019 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 3.196 ms | 5.25 | 21.00 | 84% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 25.600 ms | 10.49 | 0.16 | 33% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.898 ms | 18.67 | 19.84 | 79% | ok |
| integer | dot | i32 | n=8388608, |v| <= 15 | 2.816 ms | 5.96 | 23.83 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, |v| <= 15 | 1.203 ms | 13.94 | 27.89 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, |v| <= 15 | 0.857 ms | 19.57 | 19.57 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, |v| <= 15 | 0.677 ms | 24.76 | 24.76 | n/a | ok |
| integer | gemm | i32 | 512^3, |v| <= 2047 | 27.443 ms | 9.78 | 0.11 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, |v| <= 2047 | 27.521 ms | 9.75 | 0.08 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, |v| <= 127 | 29.075 ms | 9.23 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, |v| <= 127 | 7.638 ms | 35.14 | 0.21 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, |v| <= 127 | 6.695 ms | 40.09 | 0.23 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

