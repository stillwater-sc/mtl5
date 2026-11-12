## i7-12700k -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `bfa1c2822688` |
| FMA peak (ceiling) | f64 600.8 GFLOP/s, f32 1201.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.8, f32 1201.5 GFLOP/s |
| ... this build, AVX2 | f64 600.8, f32 1201.6 GFLOP/s |
| bandwidth | read 82.0, update 78.5, triad 76.5 GB/s (ceiling 82.0) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 88% | 146% | 0 | 22% |
| blas2 | 10 | 49% | 150% | 0 | 13% |
| blas3 | 10 | 11% | 62% | 0 | 16% |
| factor | 12 | 1% | 12% | 0 | 13% |
| eigen | 8 | 1% | 4% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 302% | 0 | 13% |
| direct | 7 | 0% | 0% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 28% | 37% | 0 | 13% |
| mg | 1 | 24% | 24% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 49% | 71% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 79% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.431 ms | 11.72 | 93.79 | 114% | ok | 2.6x (32%) |
| blas1 | dot | f32 | n=8388608 | 0.561 ms | 29.88 | 119.54 | 146% | ok | 3.0x (38%) |
| blas1 | axpy | f64 | n=8388608 | 2.781 ms | 6.03 | 72.40 | 88% | ok | 1.7x (22%) |
| blas1 | axpy | f32 | n=8388608 | 1.360 ms | 12.33 | 74.00 | 90% | ok | 1.7x (22%) |
| blas1 | scale | f64 | n=8388608 | 1.893 ms | 4.43 | 70.91 | 86% | ok | 1.8x (23%) |
| blas1 | two_norm | f64 | n=8388608 | 0.572 ms | 29.35 | 117.39 | 143% | ok | 3.9x (48%) |
| blas1 | one_norm | f64 | n=8388608 | 3.991 ms | 2.10 | 16.81 | 21% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.940 ms | 1.21 | 9.67 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.932 ms | 2.13 | 17.07 | 21% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 70.036 ms | 2.64 | 1.92 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.438 ms | 23.34 | 93.41 | 114% | ok | 2.4x (30%) |
| blas2 | gemv | f32 | n=4096 | 0.546 ms | 61.42 | 122.90 | 150% | ok | 3.1x (39%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.164 ms | 15.50 | 62.05 | 76% | ok | 4.4x (54%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.919 ms | 36.51 | 73.05 | 89% | ok | 4.4x (55%) |
| blas2 | symv | f64 | n=4096 | 6.176 ms | 5.43 | 10.88 | 13% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 1.758 ms | 9.54 | 38.21 | 47% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 0.843 ms | 19.89 | 39.82 | 49% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 1.797 ms | 9.34 | 37.39 | 46% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.862 ms | 19.46 | 38.95 | 48% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.105 ms | 4.72 | 37.79 | 46% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 1.167 ms | 230.07 | 5.39 | 38% | ok | 3.2x (41%) |
| blas3 | gemm | f32 | 512x512x512 | 0.360 ms | 745.23 | 8.73 | 62% | ok | 5.3x (66%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.250 ms | 107.34 | 5.03 | 18% | ok | 2.1x (26%) |
| blas3 | symm | f64 | 512x512x512 | 1.142 ms | 235.00 | 5.51 | 39% | ok | 3.5x (44%) |
| blas3 | syrk | f64 | 512x512x512 | 10.572 ms | 12.70 | 0.40 | 2% | ok | 0.9x (12%) |
| blas3 | syr2k | f64 | 512x512x512 | 16.443 ms | 16.33 | 0.38 | 3% | ok | 1.2x (15%) |
| blas3 | trmm | f64 | 512x512x512 | 2.102 ms | 63.87 | 2.49 | 11% | ok | 1.3x (16%) |
| blas3 | trsm | f64 | 512x512x512 | 2.121 ms | 63.29 | 2.47 | 11% | ok | 1.3x (16%) |
| blas3 | trsm | f32 | 512x512x512 | 0.974 ms | 137.85 | 2.69 | 11% | ok | 1.3x (16%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.966 ms | 0.41 | 3.28 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 6.150 ms | 14.55 | 0.68 | 2% | ok | 1.7x (21%) |
| factor | lu | f32 | 512x512 | 3.445 ms | 25.98 | 0.61 | 2% | ok | 1.0x (12%) |
| factor | cholesky | f64 | 512x512 | 6.899 ms | 6.48 | 0.61 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 6.424 ms | 6.96 | 0.33 | 1% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 7.054 ms | 6.34 | 0.59 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 44.085 ms | 1.01 | 0.10 | 0% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 126.257 ms | 1.42 | 0.03 | 0% | ok | 1.8x (23%) |
| factor | lq | f64 | 512x512 | 15.027 ms | 11.91 | 0.28 | 2% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 162.508 ms | 2.75 | 0.03 | 0% | ok | 1.6x (21%) |
| factor | householder | f64 | 512x512 | 0.426 ms | 2.47 | 9.86 | 12%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 51.962 ms | 5.17 | 0.08 | 1% | ok | 1.1x (13%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 5.504 ms | 16.26 | 0.76 | 3% | ok | 1.0x (12%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 16.806 ms | 1.33 | 0.06 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 56.196 ms | 2.69 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 738.239 ms | 0.20 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 31.162 ms | 5.38 | 0.03 | 1% | ok | 1.0x (12%) |
| eigen | schur | f64 | 256x256 | 80.053 ms | 5.24 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 92.743 ms | 3.98 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | expm | f64 | 256x256 | 10.262 ms | 24.52 | 0.10 | 4% | ok | 1.1x (14%) |
| eigen | sylvester | f64 | 256x256 | 172.552 ms | 5.83 | 0.01 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 38.867 ms | 0.22 | 3.45 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 46.005 ms | 0.18 | 2.92 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 58.765 ms | 0.14 | 2.28 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.583 ms | 0.29 | 4.70 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 201.350 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.110 ms | 23.82 | 247.77 | 302% ! | ok | 6.8x (84%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.921 ms | 2.84 | 29.58 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.377 ms | 30.41 | 63.94 | 78% | ok | 2.9x (37%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.681 ms | 0.70 | 2.46 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.387 ms | 1.45 | 12.00 | 15%\* | ok | 0.9x (12%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 71.904 ms | 0.16 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 26.928 ms | 0.42 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 9.181 ms | 1.22 | 0.08 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.650 ms | 0.56 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 55.654 ms | 0.40 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.889 ms | 1.32 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 88.733 ms | 0.02 | 0.00 | 0% | ok | 1.1x (14%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.386 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.949 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.224 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.958 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.090 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 27.998 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 20.963 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.389 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.621 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.523 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 20.074 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.532 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.381 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.489 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 14.795 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 15.828 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 18.022 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.032 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.519 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 11.891 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.060 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 1.002 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.696 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.967 ms | 2.71 | 30.33 | 37% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.288 ms | 2.03 | 22.77 | 28% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.485 ms | 1.76 | 19.75 | 24% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.804 ms | 2.60 | 19.63 | 24% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 27.840 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 39.314 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 38.633 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.499 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 1.902 ms | 8.82 | 35.29 | 43% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 0.913 ms | 294.11 | 4.60 | 49% | ok | 6.6x (82%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.306 ms | 54.85 | 58.28 | 71%\* | ok | 1.1x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.910 ms | 8.78 | 35.13 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.768 ms | 21.85 | 43.69 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.266 ms | 63.13 | 63.13 | n/a | ok | 1.1x (14%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.255 ms | 65.77 | 65.77 | n/a | ok | 1.1x (14%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 0.625 ms | 429.61 | 5.03 | n/a | ok | 6.3x (79%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 0.819 ms | 327.83 | 2.56 | n/a | ok | 6.9x (86%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 0.803 ms | 334.34 | 1.96 | n/a | ok | 7.1x (88%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.333 ms | 806.11 | 4.72 | n/a | ok | 6.4x (79%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.302 ms | 888.58 | 5.21 | n/a | ok | 6.3x (79%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

