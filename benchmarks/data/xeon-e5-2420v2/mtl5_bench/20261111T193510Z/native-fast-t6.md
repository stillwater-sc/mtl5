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
| blas1 | 10 | 98% | 108% | 0 | 18% |
| blas2 | 10 | 84% | 107% | 0 | 17% |
| blas3 | 10 | 28% | 78% | 0 | 36% |
| factor | 12 | 3% | 37% | 0 | 19% |
| eigen | 8 | 3% | 13% | 0 | 17% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 58% | 103% | 0 | 17% |
| direct | 7 | 0% | 1% | 0 | 17% |
| ordering | 4 | n/a | n/a | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 72% | 82% | 0 | 17% |
| mg | 1 | 62% | 62% | 0 | 14% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 69% | 82% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 79% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 10.905 ms | 1.54 | 12.31 | 108% | ok | 1.1x (18%) |
| blas1 | dot | f32 | n=8388608 | 5.494 ms | 3.05 | 12.21 | 107% | ok | 1.1x (18%) |
| blas1 | axpy | f64 | n=8388608 | 17.914 ms | 0.94 | 11.24 | 98% | ok | 1.1x (18%) |
| blas1 | axpy | f32 | n=8388608 | 8.469 ms | 1.98 | 11.89 | 104% | ok | 1.1x (18%) |
| blas1 | scale | f64 | n=8388608 | 12.674 ms | 0.66 | 10.59 | 93% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 5.459 ms | 3.07 | 12.29 | 108% | ok | 1.2x (20%) |
| blas1 | one_norm | f64 | n=8388608 | 9.865 ms | 0.85 | 6.80 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.055 ms | 0.64 | 5.14 | 45% | ok | 1.1x (18%) |
| blas1 | sum | f64 | n=8388608 | 9.830 ms | 0.85 | 6.83 | 60% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 273.778 ms | 0.67 | 0.49 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 10.938 ms | 3.07 | 12.28 | 107% | ok | 1.2x (20%) |
| blas2 | gemv | f32 | n=4096 | 5.483 ms | 6.12 | 12.25 | 107% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 21.167 ms | 1.59 | 6.34 | 56% | ok | 1.9x (32%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 11.157 ms | 3.01 | 6.02 | 53% | ok | 1.6x (26%) |
| blas2 | symv | f64 | n=4096 | 16.031 ms | 2.09 | 4.19 | 37% | ok | 1.0x (17%) |
| blas2 | trmv | f64 | n=4096 | 6.855 ms | 2.45 | 9.80 | 86% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 3.599 ms | 4.66 | 9.33 | 82% | ok | 1.0x (16%) |
| blas2 | trsv | f64 | n=4096 | 6.866 ms | 2.44 | 9.78 | 86% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 3.587 ms | 4.68 | 9.36 | 82% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 27.944 ms | 1.20 | 9.61 | 84% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 5.933 ms | 45.25 | 1.06 | 77% | ok | 4.7x (78%) |
| blas3 | gemm | f32 | 512x512x512 | 2.909 ms | 92.27 | 1.08 | 78% | ok | 4.6x (77%) |
| blas3 | gemmt | f64 | 512x512x512 | 5.321 ms | 25.22 | 1.18 | 43% | ok | 3.2x (53%) |
| blas3 | symm | f64 | 512x512x512 | 6.897 ms | 38.92 | 0.91 | 66% | ok | 4.2x (70%) |
| blas3 | syrk | f64 | 512x512x512 | 37.050 ms | 3.62 | 0.11 | 6% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 512x512x512 | 64.310 ms | 4.17 | 0.10 | 7% | ok | 1.0x (16%) |
| blas3 | trmm | f64 | 512x512x512 | 8.166 ms | 16.44 | 0.64 | 28% | ok | 2.2x (36%) |
| blas3 | trsm | f64 | 512x512x512 | 8.943 ms | 15.01 | 0.59 | 25% | ok | 2.0x (34%) |
| blas3 | trsm | f32 | 512x512x512 | 4.718 ms | 28.45 | 0.56 | 24% | ok | 1.9x (31%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.408 ms | 0.14 | 1.11 | 10% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 23.113 ms | 3.87 | 0.18 | 7% | ok | 1.3x (21%) |
| factor | lu | f32 | 512x512 | 13.575 ms | 6.59 | 0.15 | 6% | ok | 1.2x (20%) |
| factor | cholesky | f64 | 512x512 | 25.494 ms | 1.75 | 0.16 | 3% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 24.031 ms | 1.86 | 0.09 | 2% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 25.768 ms | 1.74 | 0.16 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 133.132 ms | 0.34 | 0.03 | 1% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 439.443 ms | 0.41 | 0.01 | 1% | ok | 1.1x (19%) |
| factor | lq | f64 | 512x512 | 50.588 ms | 3.54 | 0.08 | 6% | ok | 1.5x (24%) |
| factor | hessenberg | f64 | 512x512 | 406.816 ms | 1.10 | 0.01 | 2% | ok | 1.5x (25%) |
| factor | householder | f64 | 512x512 | 0.981 ms | 1.07 | 4.27 | 37%\* | ok | 3.2x (54%) |
| factor | inv | f64 | 512x512 | 148.554 ms | 1.81 | 0.03 | 3% | ok | 1.0x (17%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 19.093 ms | 4.69 | 0.22 | 8% | ok | 1.1x (18%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 60.932 ms | 0.37 | 0.02 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric | f64 | 256x256 | 171.823 ms | 0.88 | 0.01 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 988.132 ms | 0.15 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 105.251 ms | 1.59 | 0.01 | 3% | ok | 1.0x (16%) |
| eigen | schur | f64 | 256x256 | 259.080 ms | 1.62 | 0.00 | 3% | ok | 1.0x (16%) |
| eigen | svd | f64 | 256x256 | 260.619 ms | 1.42 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 32.297 ms | 7.79 | 0.03 | 13% | ok | 1.6x (26%) |
| eigen | sylvester | f64 | 256x256 | 564.506 ms | 1.78 | 0.00 | 3% | ok | 1.0x (17%) |
| elementwise | exp | f64 | n=8388608 | 153.676 ms | 0.05 | 0.87 | 8% | ok | 1.0x (16%) |
| elementwise | log | f64 | n=8388608 | 141.178 ms | 0.06 | 0.95 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 171.237 ms | 0.05 | 0.78 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 92.044 ms | 0.09 | 1.46 | 13% | ok | 1.0x (16%) |
| elementwise | erf | f64 | n=8388608 | 2472.458 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.312 ms | 1.13 | 11.78 | 103% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 4.104 ms | 0.64 | 6.64 | 58% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.378 ms | 3.68 | 7.74 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.364 ms | 0.20 | 0.70 | 6%\* | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.800 ms | 0.35 | 2.87 | 25% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 240.712 ms | 0.05 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 70.354 ms | 0.16 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.307 ms | 0.43 | 0.03 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 117.408 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 156.562 ms | 0.14 | 0.00 | 0% | ok | 1.0x (16%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 48.816 ms | 0.46 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 226.836 ms | 0.01 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.045 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.353 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.631 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.744 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 80.159 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 111.355 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 64.214 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 31.644 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 38.311 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 37.171 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 66.202 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 74.367 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 37.722 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 96.788 ms | 0.00 | 0.00 | n/a | ok | 0.4x (7%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 41.768 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 47.106 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 55.228 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.130 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.458 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 33.400 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 2.871 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 2.800 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | ssor | f64 | 128^2 grid | 0.074 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.186 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.588 ms | 0.73 | 8.17 | 72% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.134 ms | 0.84 | 9.36 | 82% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.879 ms | 0.67 | 7.56 | 66% | ok | 1.0x (16%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 40.859 ms | 0.94 | 7.11 | 62% | ok | 0.9x (14%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 139.520 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 180.686 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 149.311 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 16.176 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 7.170 ms | 2.34 | 9.36 | 82% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 11.774 ms | 22.80 | 0.36 | 39% | ok | 5.3x (88%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.266 ms | 7.40 | 7.87 | 69% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.024 ms | 2.79 | 11.14 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.998 ms | 5.60 | 11.19 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.073 ms | 8.09 | 8.09 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.641 ms | 10.22 | 10.22 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 3.975 ms | 67.52 | 0.79 | n/a | ok | 4.8x (79%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.553 ms | 58.95 | 0.46 | n/a | ok | 4.9x (81%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.511 ms | 59.50 | 0.35 | n/a | ok | 4.9x (81%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.342 ms | 80.32 | 0.47 | n/a | ok | 4.8x (81%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 2.590 ms | 103.63 | 0.61 | n/a | ok | 4.8x (80%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

