## xeon-e5-2420v2 -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 21.1 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 21.1 GFLOP/s |
| ... this build, SSE4 | f64 10.5, f32 21.0 GFLOP/s |
| bandwidth | read 10.5, update 10.2, triad 8.1 GB/s (ceiling 10.5) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 102% | 108% | 0 |
| blas2 | 10 | 93% | 99% | 0 |
| blas3 | 10 | 76% | 95% | 0 |
| factor | 12 | 17% | 40% | 0 |
| eigen | 8 | 15% | 47% | 0 |
| elementwise | 5 | 9% | 14% | 0 |
| sparse | 5 | 56% | 106% | 0 |
| direct | 19 | 3% | 36% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 19% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 79% | 85% | 0 |
| mg | 1 | 78% | 78% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 73% | 108% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.881 ms | 1.41 | 11.30 | 108% | ok |
| blas1 | dot | f32 | n=8388608 | 5.954 ms | 2.82 | 11.27 | 108% | ok |
| blas1 | axpy | f64 | n=8388608 | 18.748 ms | 0.89 | 10.74 | 103% | ok |
| blas1 | axpy | f32 | n=8388608 | 9.724 ms | 1.73 | 10.35 | 99% | ok |
| blas1 | scale | f64 | n=8388608 | 12.593 ms | 0.67 | 10.66 | 102% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.934 ms | 2.83 | 11.31 | 108% | ok |
| blas1 | one_norm | f64 | n=8388608 | 9.734 ms | 0.86 | 6.89 | 66% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 12.998 ms | 0.65 | 5.16 | 49% | ok |
| blas1 | sum | f64 | n=8388608 | 9.732 ms | 0.86 | 6.90 | 66% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 271.039 ms | 0.68 | 0.50 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 12.926 ms | 2.60 | 10.39 | 99% | ok |
| blas2 | gemv | f32 | n=4096 | 6.460 ms | 5.19 | 10.39 | 99% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 13.062 ms | 2.57 | 10.28 | 98% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.456 ms | 5.20 | 10.40 | 99% | ok |
| blas2 | symv | f64 | n=4096 | 12.141 ms | 2.76 | 5.53 | 53% | ok |
| blas2 | trmv | f64 | n=4096 | 6.877 ms | 2.44 | 9.77 | 93% | ok |
| blas2 | trmv | f32 | n=4096 | 3.561 ms | 4.71 | 9.43 | 90% | ok |
| blas2 | trsv | f64 | n=4096 | 6.883 ms | 2.44 | 9.76 | 93% | ok |
| blas2 | trsv | f32 | n=4096 | 3.624 ms | 4.63 | 9.27 | 88% | ok |
| blas2 | ger | f64 | n=4096 | 28.113 ms | 1.19 | 9.55 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 27.647 ms | 9.71 | 0.23 | 92% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.443 ms | 19.97 | 0.23 | 95% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 16.984 ms | 7.90 | 0.37 | 75% | ok |
| blas3 | symm | f64 | 512x512x512 | 28.876 ms | 9.30 | 0.22 | 88% | ok |
| blas3 | syrk | f64 | 512x512x512 | 16.758 ms | 8.01 | 0.25 | 76% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 33.484 ms | 8.02 | 0.19 | 76% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.618 ms | 7.62 | 0.30 | 72% | ok |
| blas3 | trsm | f64 | 512x512x512 | 18.328 ms | 7.32 | 0.29 | 70% | ok |
| blas3 | trsm | f32 | 512x512x512 | 8.825 ms | 15.21 | 0.30 | 72% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.406 ms | 0.14 | 1.11 | 11% | ok |
| factor | lu | f64 | 512x512 | 29.208 ms | 3.06 | 0.14 | 29% | ok |
| factor | lu | f32 | 512x512 | 15.522 ms | 5.76 | 0.14 | 27% | ok |
| factor | cholesky | f64 | 512x512 | 25.500 ms | 1.75 | 0.16 | 17% | ok |
| factor | cholesky | f32 | 512x512 | 23.941 ms | 1.87 | 0.09 | 9% | ok |
| factor | ldlt | f64 | 512x512 | 25.968 ms | 1.72 | 0.16 | 16% | ok |
| factor | ldlt_bk | f64 | 512x512 | 12.846 ms | 3.48 | 0.33 | 33% | ok |
| factor | qr | f64 | 512x512 | 497.670 ms | 0.36 | 0.01 | 3% | ok |
| factor | lq | f64 | 512x512 | 73.912 ms | 2.42 | 0.06 | 23% | ok |
| factor | hessenberg | f64 | 512x512 | 614.568 ms | 0.73 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.225 ms | 0.33 | 1.30 | 12%\* | ok |
| factor | inv | f64 | 512x512 | 155.320 ms | 1.73 | 0.03 | 16% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 21.135 ms | 4.23 | 0.20 | 40% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 62.956 ms | 0.36 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 172.746 ms | 0.87 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 990.613 ms | 0.15 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 106.089 ms | 1.58 | 0.01 | 15% | ok |
| eigen | schur | f64 | 256x256 | 261.304 ms | 1.61 | 0.00 | 15% | ok |
| eigen | svd | f64 | 256x256 | 261.081 ms | 1.41 | 0.00 | 13% | ok |
| eigen | expm | f64 | 256x256 | 50.650 ms | 4.97 | 0.02 | 47% | ok |
| eigen | sylvester | f64 | 256x256 | 590.504 ms | 1.70 | 0.00 | 16% | ok |
| elementwise | exp | f64 | n=8388608 | 150.444 ms | 0.06 | 0.89 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 139.358 ms | 0.06 | 0.96 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 170.085 ms | 0.05 | 0.79 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 89.720 ms | 0.09 | 1.50 | 14% | ok |
| elementwise | erf | f64 | n=8388608 | 2523.353 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.453 ms | 1.07 | 11.10 | 106% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.134 ms | 0.84 | 8.69 | 83% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.113 ms | 2.77 | 5.83 | 56% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.080 ms | 0.20 | 0.72 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.796 ms | 0.35 | 2.87 | 27% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 41.341 ms | 0.27 | 0.02 | 3% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 69.586 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 25.124 ms | 0.45 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 119.016 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 132.841 ms | 0.17 | 0.01 | 2% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 51.103 ms | 0.44 | 0.01 | 4% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 84.045 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.476 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.813 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 128.670 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.546 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.884 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.552 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.992 ms | 0.33 | 0.02 | 3% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.809 ms | 0.93 | 3.74 | 36%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.657 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.879 ms | 0.87 | 0.06 | 8% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.828 ms | 0.91 | 3.65 | 35%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.487 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 106.131 ms | 0.21 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.875 ms | 0.20 | 0.78 | 7%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.967 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 114.425 ms | 0.20 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.843 ms | 0.20 | 0.79 | 8%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 13.449 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 636.992 ms | 0.48 | 0.00 | 5% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 6.928 ms | 0.49 | 1.95 | 19%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 29.378 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 212.236 ms | 1.45 | 0.00 | 14% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 6.799 ms | 0.50 | 1.98 | 19%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.816 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1730.439 ms | 0.36 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 33.864 ms | 0.10 | 0.40 | 4%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 18.254 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1952.851 ms | 0.32 | 0.00 | 3% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.931 ms | 0.10 | 0.41 | 4%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 81.320 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 133.167 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 60.276 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 32.009 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 38.510 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 37.290 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 65.713 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 69.394 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 37.587 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 43.796 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 40.194 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 38.623 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 45.599 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.129 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.613 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 33.303 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.912 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.774 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.092 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.132 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.556 ms | 0.74 | 8.25 | 79% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.285 ms | 0.80 | 8.93 | 85% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.932 ms | 0.67 | 7.46 | 71% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.736 ms | 1.08 | 8.13 | 78% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 138.937 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 145.349 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 128.362 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 15.918 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 5.933 ms | 2.83 | 11.31 | 108% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 61.740 ms | 4.35 | 0.07 | 41% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.323 ms | 7.22 | 7.67 | 73% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.049 ms | 2.77 | 11.10 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.968 ms | 5.65 | 11.30 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.082 ms | 8.06 | 8.06 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.646 ms | 10.19 | 10.19 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 18.731 ms | 14.33 | 0.17 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 22.068 ms | 12.16 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 22.072 ms | 12.16 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 16.150 ms | 16.62 | 0.10 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 12.335 ms | 21.76 | 0.13 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

