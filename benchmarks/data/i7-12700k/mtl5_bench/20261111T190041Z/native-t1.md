## i7-12700k -- native, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.4 GFLOP/s |
| bandwidth | read 33.0, update 38.5, triad 32.5 GB/s (ceiling 38.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 44% | 97% | 0 |
| blas2 | 10 | 78% | 93% | 0 |
| blas3 | 10 | 17% | 18% | 0 |
| factor | 12 | 8% | 18% | 0 |
| eigen | 8 | 6% | 12% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 49% | 95% | 0 |
| direct | 7 | 1% | 2% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 62% | 77% | 0 |
| mg | 1 | 57% | 57% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 34% | 85% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.857 ms | 2.86 | 22.91 | 59% | ok |
| blas1 | dot | f32 | n=8388608 | 4.218 ms | 3.98 | 15.91 | 41% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.351 ms | 2.64 | 31.70 | 82% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.725 ms | 3.55 | 21.30 | 55% | ok |
| blas1 | scale | f64 | n=8388608 | 3.576 ms | 2.35 | 37.54 | 97% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.888 ms | 3.43 | 13.73 | 36% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.037 ms | 2.08 | 16.62 | 43% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.042 ms | 1.19 | 9.53 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 3.995 ms | 2.10 | 16.80 | 44% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 77.567 ms | 2.38 | 1.73 | 4% | ok |
| blas2 | gemv | f64 | n=4096 | 3.756 ms | 8.93 | 35.75 | 93% | ok |
| blas2 | gemv | f32 | n=4096 | 2.728 ms | 12.30 | 24.62 | 64% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 4.811 ms | 6.97 | 27.91 | 72% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.248 ms | 14.93 | 29.87 | 78% | ok |
| blas2 | symv | f64 | n=4096 | 6.593 ms | 5.09 | 10.19 | 26% | ok |
| blas2 | trmv | f64 | n=4096 | 2.034 ms | 8.25 | 33.02 | 86% | ok |
| blas2 | trmv | f32 | n=4096 | 1.453 ms | 11.55 | 23.12 | 60% | ok |
| blas2 | trsv | f64 | n=4096 | 2.038 ms | 8.23 | 32.96 | 86% | ok |
| blas2 | trsv | f32 | n=4096 | 1.414 ms | 11.87 | 23.75 | 62% | ok |
| blas2 | ger | f64 | n=4096 | 7.717 ms | 4.35 | 34.79 | 90% | ok |
| blas3 | gemm | f64 | 512x512x512 | 20.198 ms | 13.29 | 0.31 | 17% | ok |
| blas3 | gemm | f32 | 512x512x512 | 10.618 ms | 25.28 | 0.30 | 16% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 10.773 ms | 12.46 | 0.58 | 16% | ok |
| blas3 | symm | f64 | 512x512x512 | 19.298 ms | 13.91 | 0.33 | 18% | ok |
| blas3 | syrk | f64 | 512x512x512 | 13.043 ms | 10.29 | 0.32 | 13% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 23.572 ms | 11.39 | 0.27 | 15% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.743 ms | 13.78 | 0.54 | 18% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.708 ms | 13.83 | 0.54 | 18% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.142 ms | 26.10 | 0.51 | 17% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.214 ms | 0.40 | 3.21 | 8% | ok |
| factor | lu | f64 | 512x512 | 11.625 ms | 7.70 | 0.36 | 10% | ok |
| factor | lu | f32 | 512x512 | 4.233 ms | 21.14 | 0.50 | 13% | ok |
| factor | cholesky | f64 | 512x512 | 7.345 ms | 6.09 | 0.57 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.611 ms | 6.77 | 0.32 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.458 ms | 6.00 | 0.56 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 47.970 ms | 0.93 | 0.09 | 1% | ok |
| factor | qr | f64 | 512x512 | 227.701 ms | 0.79 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 25.974 ms | 6.89 | 0.16 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 273.695 ms | 1.63 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.638 ms | 0.64 | 2.56 | 7%\* | ok |
| factor | inv | f64 | 512x512 | 54.709 ms | 4.91 | 0.08 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.243 ms | 14.33 | 0.67 | 18% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 18.156 ms | 1.23 | 0.06 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 59.144 ms | 2.55 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 743.077 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 32.983 ms | 5.09 | 0.03 | 6% | ok |
| eigen | schur | f64 | 256x256 | 81.936 ms | 5.12 | 0.01 | 7% | ok |
| eigen | svd | f64 | 256x256 | 110.166 ms | 3.35 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 26.373 ms | 9.54 | 0.04 | 12% | ok |
| eigen | sylvester | f64 | 256x256 | 186.214 ms | 5.41 | 0.01 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 39.618 ms | 0.21 | 3.39 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 46.018 ms | 0.18 | 2.92 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 60.309 ms | 0.14 | 2.23 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 29.098 ms | 0.29 | 4.61 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 203.386 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.746 ms | 3.51 | 36.50 | 95% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.943 ms | 2.77 | 28.87 | 75% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.649 ms | 9.01 | 18.94 | 49% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.874 ms | 0.67 | 2.36 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.278 ms | 1.57 | 13.03 | 34%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 91.448 ms | 0.12 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.587 ms | 0.44 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.901 ms | 1.26 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.158 ms | 0.56 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 54.223 ms | 0.41 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.808 ms | 1.33 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 101.709 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.364 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.987 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.230 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.082 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 31.258 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 33.126 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 23.161 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.540 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 15.050 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 14.074 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.765 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.335 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.321 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 17.229 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.456 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.854 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.322 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.035 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.555 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.376 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.069 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.005 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.022 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.692 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.992 ms | 2.64 | 29.56 | 77% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.221 ms | 2.14 | 24.02 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.564 ms | 1.67 | 18.75 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.339 ms | 2.88 | 21.79 | 57% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.970 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 44.412 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 39.759 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 5.079 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 5.136 ms | 3.27 | 13.07 | 34% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 39.013 ms | 6.88 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.545 ms | 30.80 | 32.72 | 85%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.305 ms | 7.28 | 29.12 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.009 ms | 16.62 | 33.25 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.724 ms | 9.73 | 9.73 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.725 ms | 9.73 | 9.73 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.466 ms | 15.37 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.813 ms | 12.90 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.230 ms | 26.24 | 0.15 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.230 ms | 26.24 | 0.15 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 26.812 ms | 10.01 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

