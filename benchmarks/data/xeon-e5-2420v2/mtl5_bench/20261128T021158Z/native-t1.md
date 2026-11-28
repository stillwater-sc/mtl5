## xeon-e5-2420v2 -- native, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 21.0 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 21.0 GFLOP/s |
| ... this build, scalar | f64 0.1, f32 0.5 GFLOP/s |
| bandwidth | read 10.2, update 10.4, triad 8.1 GB/s (ceiling 10.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 23% | 104% | 0 |
| blas2 | 10 | 91% | 101% | 0 |
| blas3 | 10 | 38% | 40% | 0 |
| factor | 12 | 17% | 37% | 0 |
| eigen | 8 | 14% | 28% | 0 |
| elementwise | 5 | 9% | 14% | 0 |
| sparse | 5 | 56% | 106% | 0 |
| direct | 19 | 3% | 35% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 19% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 79% | 86% | 0 |
| mg | 1 | 78% | 78% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 22% | 72% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 183.351 ms | 0.09 | 0.73 | 7% | ok |
| blas1 | dot | f32 | n=8388608 | 27.912 ms | 0.60 | 2.40 | 23% | ok |
| blas1 | axpy | f64 | n=8388608 | 211.173 ms | 0.08 | 0.95 | 9% | ok |
| blas1 | axpy | f32 | n=8388608 | 91.312 ms | 0.18 | 1.10 | 11% | ok |
| blas1 | scale | f64 | n=8388608 | 12.411 ms | 0.68 | 10.81 | 104% | ok |
| blas1 | two_norm | f64 | n=8388608 | 185.984 ms | 0.09 | 0.36 | 3% | ok |
| blas1 | one_norm | f64 | n=8388608 | 9.712 ms | 0.86 | 6.91 | 67% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 13.103 ms | 0.64 | 5.12 | 49% | ok |
| blas1 | sum | f64 | n=8388608 | 9.709 ms | 0.86 | 6.91 | 67% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 291.821 ms | 0.63 | 0.46 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 13.916 ms | 2.41 | 9.65 | 93% | ok |
| blas2 | gemv | f32 | n=4096 | 8.322 ms | 4.03 | 8.07 | 78% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 13.335 ms | 2.52 | 10.07 | 97% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.446 ms | 5.21 | 10.42 | 101% | ok |
| blas2 | symv | f64 | n=4096 | 12.194 ms | 2.75 | 5.51 | 53% | ok |
| blas2 | trmv | f64 | n=4096 | 7.260 ms | 2.31 | 9.25 | 89% | ok |
| blas2 | trmv | f32 | n=4096 | 4.578 ms | 3.66 | 7.34 | 71% | ok |
| blas2 | trsv | f64 | n=4096 | 7.079 ms | 2.37 | 9.49 | 92% | ok |
| blas2 | trsv | f32 | n=4096 | 4.314 ms | 3.89 | 7.78 | 75% | ok |
| blas2 | ger | f64 | n=4096 | 28.502 ms | 1.18 | 9.42 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 64.544 ms | 4.16 | 0.10 | 40% | ok |
| blas3 | gemm | f32 | 512x512x512 | 35.043 ms | 7.66 | 0.09 | 36% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 36.315 ms | 3.70 | 0.17 | 35% | ok |
| blas3 | symm | f64 | 512x512x512 | 67.154 ms | 4.00 | 0.09 | 38% | ok |
| blas3 | syrk | f64 | 512x512x512 | 38.146 ms | 3.52 | 0.11 | 33% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 66.054 ms | 4.06 | 0.10 | 39% | ok |
| blas3 | trmm | f64 | 512x512x512 | 32.955 ms | 4.07 | 0.16 | 39% | ok |
| blas3 | trsm | f64 | 512x512x512 | 33.979 ms | 3.95 | 0.15 | 38% | ok |
| blas3 | trsm | f32 | 512x512x512 | 16.306 ms | 8.23 | 0.16 | 39% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.245 ms | 0.14 | 1.11 | 11% | ok |
| factor | lu | f64 | 512x512 | 31.058 ms | 2.88 | 0.14 | 27% | ok |
| factor | lu | f32 | 512x512 | 16.223 ms | 5.52 | 0.13 | 26% | ok |
| factor | cholesky | f64 | 512x512 | 24.883 ms | 1.80 | 0.17 | 17% | ok |
| factor | cholesky | f32 | 512x512 | 23.430 ms | 1.91 | 0.09 | 9% | ok |
| factor | ldlt | f64 | 512x512 | 25.223 ms | 1.77 | 0.17 | 17% | ok |
| factor | ldlt_bk | f64 | 512x512 | 13.425 ms | 3.33 | 0.31 | 32% | ok |
| factor | qr | f64 | 512x512 | 505.113 ms | 0.35 | 0.01 | 3% | ok |
| factor | lq | f64 | 512x512 | 74.217 ms | 2.41 | 0.06 | 23% | ok |
| factor | hessenberg | f64 | 512x512 | 612.600 ms | 0.73 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.188 ms | 0.33 | 1.32 | 13%\* | ok |
| factor | inv | f64 | 512x512 | 158.860 ms | 1.69 | 0.03 | 16% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 22.770 ms | 3.93 | 0.18 | 37% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 65.106 ms | 0.34 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 186.571 ms | 0.81 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 993.334 ms | 0.15 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 117.401 ms | 1.43 | 0.01 | 14% | ok |
| eigen | schur | f64 | 256x256 | 269.078 ms | 1.56 | 0.00 | 15% | ok |
| eigen | svd | f64 | 256x256 | 301.634 ms | 1.22 | 0.00 | 12% | ok |
| eigen | expm | f64 | 256x256 | 84.876 ms | 2.97 | 0.01 | 28% | ok |
| eigen | sylvester | f64 | 256x256 | 636.742 ms | 1.58 | 0.00 | 15% | ok |
| elementwise | exp | f64 | n=8388608 | 151.219 ms | 0.06 | 0.89 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 139.933 ms | 0.06 | 0.96 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 171.532 ms | 0.05 | 0.78 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 89.826 ms | 0.09 | 1.49 | 14% | ok |
| elementwise | erf | f64 | n=8388608 | 2482.071 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.489 ms | 1.05 | 10.94 | 106% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.131 ms | 0.84 | 8.70 | 84% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.142 ms | 2.77 | 5.81 | 56% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.623 ms | 0.20 | 0.69 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.829 ms | 0.34 | 2.86 | 28% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 41.021 ms | 0.27 | 0.02 | 3% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 68.920 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.748 ms | 0.45 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 115.793 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 129.648 ms | 0.17 | 0.01 | 2% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 47.865 ms | 0.47 | 0.02 | 4% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 81.662 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.025 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.223 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 127.084 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.528 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.950 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.354 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.614 ms | 0.33 | 0.02 | 3% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.834 ms | 0.91 | 3.63 | 35%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.378 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.574 ms | 0.89 | 0.06 | 8% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.852 ms | 0.89 | 3.55 | 34%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.337 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 103.093 ms | 0.22 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.855 ms | 0.20 | 0.78 | 8%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.841 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 112.042 ms | 0.20 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.879 ms | 0.19 | 0.78 | 8%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 13.006 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 638.406 ms | 0.48 | 0.00 | 5% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 6.828 ms | 0.49 | 1.97 | 19%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 29.181 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 206.047 ms | 1.50 | 0.00 | 14% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 6.868 ms | 0.49 | 1.96 | 19%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.599 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1721.050 ms | 0.36 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.542 ms | 0.10 | 0.41 | 4%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 18.085 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1534.322 ms | 0.40 | 0.00 | 4% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 33.043 ms | 0.10 | 0.41 | 4%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 232.142 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 402.014 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 128.289 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 104.109 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 122.296 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 80.167 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 549.262 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 175.443 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.952 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 203.570 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 193.305 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 212.102 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 307.749 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.171 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.354 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 33.476 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.879 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.744 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.093 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.077 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.571 ms | 0.73 | 8.21 | 79% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.291 ms | 0.80 | 8.91 | 86% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.912 ms | 0.67 | 7.50 | 72% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.957 ms | 1.07 | 8.08 | 78% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 141.155 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 148.220 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 133.848 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 43.165 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 196.861 ms | 0.09 | 0.34 | 3% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 114.070 ms | 2.35 | 0.04 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.382 ms | 7.04 | 7.48 | 72% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.459 ms | 2.60 | 10.39 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.247 ms | 5.17 | 10.33 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.986 ms | 3.37 | 3.37 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.978 ms | 3.37 | 3.37 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 65.528 ms | 4.10 | 0.05 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 45.356 ms | 5.92 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 23.688 ms | 11.33 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 23.729 ms | 11.31 | 0.07 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 71.486 ms | 3.76 | 0.02 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

