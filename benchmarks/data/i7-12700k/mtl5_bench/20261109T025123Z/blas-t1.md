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

