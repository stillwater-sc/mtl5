# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,6, CPUs 0,1,2,3,4,5, clock policy governor performance on every CPU, turbo on; performance 1200-2700 MHz on cpus 0,1,2,3,4,5, 2026-11-28T02:45:08Z.

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

## xeon-e5-2420v2 -- native, 6 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 58.8 GFLOP/s, f32 118.1 GFLOP/s |
| ... hardware, SSE2 | f64 58.8, f32 118.1 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.7 GFLOP/s |
| bandwidth | read 11.4, update 10.6, triad 8.4 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 53% | 106% | 0 | 84% |
| blas2 | 10 | 85% | 108% | 0 | 17% |
| blas3 | 10 | 7% | 39% | 0 | 17% |
| factor | 12 | 6% | 40% | 0 | 23% |
| eigen | 8 | 3% | 11% | 0 | 18% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 68% | 104% | 0 | 17% |
| direct | 19 | 1% | 31% | 0 | 16% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 4% | 17% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 73% | 80% | 0 | 17% |
| mg | 1 | 72% | 72% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 22% | 71% | 0 | 18% |
| integer | 9 | n/a | n/a | 0 | 91% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 32.722 ms | 0.51 | 4.10 | 36% | ok | 5.6x (93%) |
| blas1 | dot | f32 | n=8388608 | 5.536 ms | 3.03 | 12.12 | 106% | ok | 5.0x (84%) |
| blas1 | axpy | f64 | n=8388608 | 38.119 ms | 0.44 | 5.28 | 46% | ok | 5.5x (92%) |
| blas1 | axpy | f32 | n=8388608 | 16.786 ms | 1.00 | 6.00 | 53% | ok | 5.4x (91%) |
| blas1 | scale | f64 | n=8388608 | 12.086 ms | 0.69 | 11.11 | 97% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 33.035 ms | 0.51 | 2.03 | 18% | ok | 5.6x (94%) |
| blas1 | one_norm | f64 | n=8388608 | 9.771 ms | 0.86 | 6.87 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.159 ms | 0.64 | 5.10 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.745 ms | 0.86 | 6.89 | 60% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 279.997 ms | 0.66 | 0.48 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 10.929 ms | 3.07 | 12.29 | 108% | ok | 1.3x (21%) |
| blas2 | gemv | f32 | n=4096 | 5.488 ms | 6.11 | 12.23 | 107% | ok | 1.5x (25%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 10.919 ms | 3.07 | 12.30 | 108% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.456 ms | 6.15 | 12.31 | 108% | ok | 1.2x (20%) |
| blas2 | symv | f64 | n=4096 | 12.597 ms | 2.66 | 5.33 | 47% | ok | 1.0x (16%) |
| blas2 | trmv | f64 | n=4096 | 7.098 ms | 2.36 | 9.46 | 83% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 5.110 ms | 3.28 | 6.57 | 58% | ok | 0.9x (15%) |
| blas2 | trsv | f64 | n=4096 | 6.905 ms | 2.43 | 9.73 | 85% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 4.595 ms | 3.65 | 7.31 | 64% | ok | 0.9x (16%) |
| blas2 | ger | f64 | n=4096 | 28.520 ms | 1.18 | 9.41 | 83% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 11.553 ms | 23.23 | 0.54 | 39% | ok | 5.6x (93%) |
| blas3 | gemm | f32 | 512x512x512 | 6.228 ms | 43.10 | 0.51 | 36% | ok | 5.6x (94%) |
| blas3 | gemmt | f64 | 512x512x512 | 36.346 ms | 3.69 | 0.17 | 6% | ok | 1.0x (17%) |
| blas3 | symm | f64 | 512x512x512 | 65.932 ms | 4.07 | 0.10 | 7% | ok | 1.0x (17%) |
| blas3 | syrk | f64 | 512x512x512 | 38.522 ms | 3.48 | 0.11 | 6% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 512x512x512 | 65.750 ms | 4.08 | 0.10 | 7% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 512x512x512 | 32.599 ms | 4.12 | 0.16 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f64 | 512x512x512 | 32.956 ms | 4.07 | 0.16 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f32 | 512x512x512 | 17.190 ms | 7.81 | 0.15 | 7% | ok | 0.9x (16%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.671 ms | 0.14 | 1.10 | 10% | ok | 1.0x (16%) |
| factor | lu | f64 | 512x512 | 21.854 ms | 4.09 | 0.19 | 7% | ok | 1.4x (24%) |
| factor | lu | f32 | 512x512 | 11.907 ms | 7.51 | 0.18 | 6% | ok | 1.4x (23%) |
| factor | cholesky | f64 | 512x512 | 24.868 ms | 1.80 | 0.17 | 3% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 23.677 ms | 1.89 | 0.09 | 2% | ok | 1.0x (16%) |
| factor | ldlt | f64 | 512x512 | 25.585 ms | 1.75 | 0.16 | 3% | ok | 1.0x (16%) |
| factor | ldlt_bk | f64 | 512x512 | 13.487 ms | 3.32 | 0.31 | 6% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 320.889 ms | 0.56 | 0.01 | 1% | ok | 1.6x (26%) |
| factor | lq | f64 | 512x512 | 49.121 ms | 3.64 | 0.09 | 6% | ok | 1.5x (25%) |
| factor | hessenberg | f64 | 512x512 | 402.067 ms | 1.11 | 0.01 | 2% | ok | 1.5x (25%) |
| factor | householder | f64 | 512x512 | 0.927 ms | 1.13 | 4.52 | 40%\* | ok | 3.4x (57%) |
| factor | inv | f64 | 512x512 | 143.563 ms | 1.87 | 0.03 | 3% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 17.054 ms | 5.25 | 0.25 | 9% | ok | 1.3x (22%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 60.344 ms | 0.37 | 0.02 | 1% | ok | 1.1x (18%) |
| eigen | eigen_symmetric | f64 | 256x256 | 171.997 ms | 0.88 | 0.01 | 1% | ok | 1.1x (18%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1010.164 ms | 0.15 | 0.00 | 0% | ok | 1.0x (16%) |
| eigen | eigenvalue | f64 | 256x256 | 108.076 ms | 1.55 | 0.01 | 3% | ok | 1.1x (18%) |
| eigen | schur | f64 | 256x256 | 279.598 ms | 1.50 | 0.00 | 3% | ok | 1.0x (16%) |
| eigen | svd | f64 | 256x256 | 321.014 ms | 1.15 | 0.00 | 2% | ok | 0.9x (16%) |
| eigen | expm | f64 | 256x256 | 37.769 ms | 6.66 | 0.03 | 11% | ok | 2.2x (37%) |
| eigen | sylvester | f64 | 256x256 | 600.658 ms | 1.68 | 0.00 | 3% | ok | 1.1x (18%) |
| elementwise | exp | f64 | n=8388608 | 152.780 ms | 0.05 | 0.88 | 8% | ok | 1.0x (16%) |
| elementwise | log | f64 | n=8388608 | 139.526 ms | 0.06 | 0.96 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 170.996 ms | 0.05 | 0.78 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 91.155 ms | 0.09 | 1.47 | 13% | ok | 1.0x (16%) |
| elementwise | erf | f64 | n=8388608 | 2484.854 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.298 ms | 1.14 | 11.85 | 104% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.146 ms | 0.83 | 8.66 | 76% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.321 ms | 3.70 | 7.78 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.609 ms | 0.20 | 0.69 | 6%\* | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.913 ms | 0.34 | 2.82 | 25% | ok | 1.0x (16%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 41.467 ms | 0.27 | 0.02 | 0% | ok | 1.0x (16%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 73.050 ms | 0.15 | 0.01 | 0% | ok | 0.9x (16%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 25.005 ms | 0.45 | 0.03 | 1% | ok | 1.0x (16%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 118.051 ms | 0.19 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 131.251 ms | 0.17 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 48.559 ms | 0.46 | 0.02 | 1% | ok | 1.0x (16%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 82.587 ms | 0.02 | 0.00 | 0% | ok | 1.0x (16%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.879 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.280 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 126.052 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.523 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 41.012 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.216 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.431 ms | 0.33 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.853 ms | 0.89 | 3.54 | 31%\* | ok | 1.0x (16%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.372 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.767 ms | 0.88 | 0.06 | 1% | ok | 1.0x (16%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.879 ms | 0.86 | 3.44 | 30%\* | ok | 1.0x (16%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.308 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 103.970 ms | 0.22 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.818 ms | 0.20 | 0.79 | 7%\* | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.845 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 113.643 ms | 0.20 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.780 ms | 0.20 | 0.80 | 7%\* | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 12.951 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 633.732 ms | 0.49 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.072 ms | 0.48 | 1.91 | 17%\* | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.791 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 205.548 ms | 1.50 | 0.00 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.223 ms | 0.47 | 1.87 | 16%\* | ok | 1.0x (16%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.510 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1705.496 ms | 0.36 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.804 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 17.911 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1523.577 ms | 0.41 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.704 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 231.496 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 404.545 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 128.368 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 105.445 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 123.150 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 80.192 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 536.671 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 175.938 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.855 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 202.554 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 192.450 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 213.620 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 307.384 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.172 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.365 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 32.849 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 2.909 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | ildl | f64 | 128^2 grid | 2.727 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.093 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.239 ms | 0.00 | 0.00 | n/a | ok | 0.9x (15%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.528 ms | 0.74 | 8.31 | 73% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.217 ms | 0.81 | 9.12 | 80% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.953 ms | 0.66 | 7.42 | 65% | ok | 1.0x (16%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.563 ms | 1.08 | 8.17 | 72% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 142.190 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 146.248 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 128.253 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.448 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 194.190 ms | 0.09 | 0.35 | 3% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 20.546 ms | 13.07 | 0.20 | 22% | ok | 5.6x (93%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.203 ms | 7.62 | 8.09 | 71% | ok | 1.1x (18%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.502 ms | 2.58 | 10.32 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.110 ms | 5.39 | 10.79 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.982 ms | 3.37 | 3.37 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.975 ms | 3.37 | 3.37 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.536 ms | 23.27 | 0.27 | n/a | ok | 5.7x (95%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 8.214 ms | 32.68 | 0.26 | n/a | ok | 5.5x (92%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.340 ms | 61.86 | 0.36 | n/a | ok | 5.5x (91%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.333 ms | 61.95 | 0.36 | n/a | ok | 5.5x (91%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.090 ms | 20.51 | 0.12 | n/a | ok | 5.5x (91%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## xeon-e5-2420v2 -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 21.0 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 21.0 GFLOP/s |
| ... this build, SSE4 | f64 10.5, f32 21.0 GFLOP/s |
| bandwidth | read 10.4, update 10.2, triad 8.2 GB/s (ceiling 10.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 104% | 110% | 0 |
| blas2 | 10 | 91% | 99% | 0 |
| blas3 | 10 | 76% | 95% | 0 |
| factor | 12 | 17% | 41% | 0 |
| eigen | 8 | 15% | 48% | 0 |
| elementwise | 5 | 9% | 14% | 0 |
| sparse | 5 | 56% | 105% | 0 |
| direct | 19 | 3% | 35% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 19% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 78% | 84% | 0 |
| mg | 1 | 77% | 77% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 74% | 108% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 12.025 ms | 1.40 | 11.16 | 107% | ok |
| blas1 | dot | f32 | n=8388608 | 5.951 ms | 2.82 | 11.28 | 108% | ok |
| blas1 | axpy | f64 | n=8388608 | 18.823 ms | 0.89 | 10.70 | 103% | ok |
| blas1 | axpy | f32 | n=8388608 | 8.960 ms | 1.87 | 11.23 | 108% | ok |
| blas1 | scale | f64 | n=8388608 | 12.376 ms | 0.68 | 10.85 | 104% | ok |
| blas1 | two_norm | f64 | n=8388608 | 5.884 ms | 2.85 | 11.40 | 110% | ok |
| blas1 | one_norm | f64 | n=8388608 | 9.780 ms | 0.86 | 6.86 | 66% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 13.188 ms | 0.64 | 5.09 | 49% | ok |
| blas1 | sum | f64 | n=8388608 | 9.713 ms | 0.86 | 6.91 | 66% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 271.773 ms | 0.68 | 0.49 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 13.478 ms | 2.49 | 9.96 | 96% | ok |
| blas2 | gemv | f32 | n=4096 | 6.705 ms | 5.00 | 10.01 | 96% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 13.027 ms | 2.58 | 10.31 | 99% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.549 ms | 5.12 | 10.25 | 98% | ok |
| blas2 | symv | f64 | n=4096 | 12.349 ms | 2.72 | 5.44 | 52% | ok |
| blas2 | trmv | f64 | n=4096 | 7.160 ms | 2.34 | 9.38 | 90% | ok |
| blas2 | trmv | f32 | n=4096 | 3.705 ms | 4.53 | 9.07 | 87% | ok |
| blas2 | trsv | f64 | n=4096 | 7.111 ms | 2.36 | 9.45 | 91% | ok |
| blas2 | trsv | f32 | n=4096 | 3.756 ms | 4.47 | 8.94 | 86% | ok |
| blas2 | ger | f64 | n=4096 | 28.218 ms | 1.19 | 9.52 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 28.115 ms | 9.55 | 0.22 | 91% | ok |
| blas3 | gemm | f32 | 512x512x512 | 13.481 ms | 19.91 | 0.23 | 95% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 17.072 ms | 7.86 | 0.37 | 75% | ok |
| blas3 | symm | f64 | 512x512x512 | 29.262 ms | 9.17 | 0.22 | 87% | ok |
| blas3 | syrk | f64 | 512x512x512 | 16.869 ms | 7.96 | 0.25 | 76% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 33.716 ms | 7.96 | 0.19 | 76% | ok |
| blas3 | trmm | f64 | 512x512x512 | 17.758 ms | 7.56 | 0.30 | 72% | ok |
| blas3 | trsm | f64 | 512x512x512 | 18.289 ms | 7.34 | 0.29 | 70% | ok |
| blas3 | trsm | f32 | 512x512x512 | 9.378 ms | 14.31 | 0.28 | 68% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.412 ms | 0.14 | 1.11 | 11% | ok |
| factor | lu | f64 | 512x512 | 29.734 ms | 3.01 | 0.14 | 29% | ok |
| factor | lu | f32 | 512x512 | 15.633 ms | 5.72 | 0.13 | 27% | ok |
| factor | cholesky | f64 | 512x512 | 25.609 ms | 1.75 | 0.16 | 17% | ok |
| factor | cholesky | f32 | 512x512 | 24.128 ms | 1.85 | 0.09 | 9% | ok |
| factor | ldlt | f64 | 512x512 | 25.945 ms | 1.72 | 0.16 | 16% | ok |
| factor | ldlt_bk | f64 | 512x512 | 13.104 ms | 3.41 | 0.32 | 32% | ok |
| factor | qr | f64 | 512x512 | 500.732 ms | 0.36 | 0.01 | 3% | ok |
| factor | lq | f64 | 512x512 | 73.623 ms | 2.43 | 0.06 | 23% | ok |
| factor | hessenberg | f64 | 512x512 | 622.280 ms | 0.72 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.301 ms | 0.32 | 1.27 | 12%\* | ok |
| factor | inv | f64 | 512x512 | 155.344 ms | 1.73 | 0.03 | 16% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 21.003 ms | 4.26 | 0.20 | 41% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 67.177 ms | 0.33 | 0.02 | 3% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 171.196 ms | 0.88 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 967.696 ms | 0.16 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 105.778 ms | 1.59 | 0.01 | 15% | ok |
| eigen | schur | f64 | 256x256 | 270.197 ms | 1.55 | 0.00 | 15% | ok |
| eigen | svd | f64 | 256x256 | 262.334 ms | 1.41 | 0.00 | 13% | ok |
| eigen | expm | f64 | 256x256 | 50.388 ms | 4.99 | 0.02 | 48% | ok |
| eigen | sylvester | f64 | 256x256 | 600.040 ms | 1.68 | 0.00 | 16% | ok |
| elementwise | exp | f64 | n=8388608 | 150.818 ms | 0.06 | 0.89 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 140.470 ms | 0.06 | 0.96 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 170.953 ms | 0.05 | 0.79 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 91.558 ms | 0.09 | 1.47 | 14% | ok |
| elementwise | erf | f64 | n=8388608 | 2477.217 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.484 ms | 1.05 | 10.96 | 105% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.097 ms | 0.85 | 8.79 | 84% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.201 ms | 2.75 | 5.79 | 56% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.177 ms | 0.20 | 0.71 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.798 ms | 0.35 | 2.87 | 28% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 42.331 ms | 0.26 | 0.02 | 3% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 69.930 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.380 ms | 0.42 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 117.919 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 131.275 ms | 0.17 | 0.01 | 2% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 47.885 ms | 0.47 | 0.02 | 4% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 81.608 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.287 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.537 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 130.186 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.562 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 45.309 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.438 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.939 ms | 0.33 | 0.02 | 3% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.843 ms | 0.90 | 3.59 | 34%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.981 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 13.039 ms | 0.86 | 0.06 | 8% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.833 ms | 0.91 | 3.63 | 35%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.581 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 104.755 ms | 0.21 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.988 ms | 0.19 | 0.76 | 7%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 10.400 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 113.176 ms | 0.20 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.802 ms | 0.20 | 0.80 | 8%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 13.375 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 635.201 ms | 0.49 | 0.00 | 5% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 6.781 ms | 0.50 | 1.99 | 19%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 30.614 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 206.136 ms | 1.50 | 0.00 | 14% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 6.821 ms | 0.49 | 1.98 | 19%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 13.064 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1729.335 ms | 0.36 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.704 ms | 0.10 | 0.41 | 4%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 19.160 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1991.166 ms | 0.31 | 0.00 | 3% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.827 ms | 0.10 | 0.41 | 4%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 80.806 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 134.486 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 59.731 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 32.150 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 38.493 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 37.671 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 69.398 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 69.740 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 39.416 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 44.079 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 40.718 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.769 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 47.722 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.129 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.660 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 33.824 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.958 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.756 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.093 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.072 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.602 ms | 0.73 | 8.14 | 78% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.342 ms | 0.78 | 8.78 | 84% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.922 ms | 0.67 | 7.48 | 72% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 36.181 ms | 1.06 | 8.03 | 77% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 140.800 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 146.962 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 133.044 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 16.135 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 5.953 ms | 2.82 | 11.27 | 108% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 61.749 ms | 4.35 | 0.07 | 41% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.320 ms | 7.23 | 7.68 | 74% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.050 ms | 2.77 | 11.09 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.964 ms | 5.66 | 11.32 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.079 ms | 8.07 | 8.07 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.640 ms | 10.23 | 10.23 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 18.838 ms | 14.25 | 0.17 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 22.138 ms | 12.13 | 0.09 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 22.100 ms | 12.15 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 16.138 ms | 16.63 | 0.10 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 12.377 ms | 21.69 | 0.13 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## xeon-e5-2420v2 -- native-fast, 6 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 59.0 GFLOP/s, f32 118.1 GFLOP/s |
| ... hardware, SSE2 | f64 59.0, f32 118.1 GFLOP/s |
| ... this build, SSE4 | f64 58.9, f32 118.0 GFLOP/s |
| bandwidth | read 11.4, update 10.6, triad 8.5 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 99% | 108% | 0 | 18% |
| blas2 | 10 | 86% | 108% | 0 | 17% |
| blas3 | 10 | 46% | 79% | 0 | 57% |
| factor | 12 | 5% | 41% | 0 | 22% |
| eigen | 8 | 3% | 13% | 0 | 17% |
| elementwise | 5 | 7% | 13% | 0 | 17% |
| sparse | 5 | 68% | 104% | 0 | 17% |
| direct | 19 | 1% | 31% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 4% | 17% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 71% | 78% | 0 | 17% |
| mg | 1 | 71% | 71% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 69% | 94% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 80% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 10.873 ms | 1.54 | 12.34 | 108% | ok | 1.1x (18%) |
| blas1 | dot | f32 | n=8388608 | 5.444 ms | 3.08 | 12.33 | 108% | ok | 1.1x (18%) |
| blas1 | axpy | f64 | n=8388608 | 17.855 ms | 0.94 | 11.28 | 99% | ok | 1.1x (18%) |
| blas1 | axpy | f32 | n=8388608 | 8.443 ms | 1.99 | 11.92 | 104% | ok | 1.1x (18%) |
| blas1 | scale | f64 | n=8388608 | 12.078 ms | 0.69 | 11.11 | 97% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 5.428 ms | 3.09 | 12.36 | 108% | ok | 1.1x (18%) |
| blas1 | one_norm | f64 | n=8388608 | 9.720 ms | 0.86 | 6.90 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.252 ms | 0.63 | 5.06 | 44% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.703 ms | 0.86 | 6.92 | 61% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 275.128 ms | 0.67 | 0.49 | 4% | ok | 1.0x (16%) |
| blas2 | gemv | f64 | n=4096 | 10.920 ms | 3.07 | 12.30 | 108% | ok | 1.2x (21%) |
| blas2 | gemv | f32 | n=4096 | 5.638 ms | 5.95 | 11.91 | 104% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 11.066 ms | 3.03 | 12.13 | 106% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 5.812 ms | 5.77 | 11.55 | 101% | ok | 1.1x (19%) |
| blas2 | symv | f64 | n=4096 | 12.233 ms | 2.74 | 5.49 | 48% | ok | 1.0x (17%) |
| blas2 | trmv | f64 | n=4096 | 6.924 ms | 2.42 | 9.70 | 85% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=4096 | 3.601 ms | 4.66 | 9.33 | 82% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=4096 | 6.871 ms | 2.44 | 9.78 | 86% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 3.607 ms | 4.65 | 9.31 | 82% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 28.215 ms | 1.19 | 9.52 | 83% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 5.828 ms | 46.06 | 1.08 | 78% | ok | 4.8x (80%) |
| blas3 | gemm | f32 | 512x512x512 | 2.889 ms | 92.91 | 1.09 | 79% | ok | 4.7x (78%) |
| blas3 | gemmt | f64 | 512x512x512 | 4.977 ms | 26.96 | 1.26 | 46% | ok | 3.4x (57%) |
| blas3 | symm | f64 | 512x512x512 | 6.908 ms | 38.86 | 0.91 | 66% | ok | 4.2x (71%) |
| blas3 | syrk | f64 | 512x512x512 | 4.967 ms | 27.02 | 0.84 | 46% | ok | 3.4x (57%) |
| blas3 | syr2k | f64 | 512x512x512 | 9.491 ms | 28.28 | 0.66 | 48% | ok | 3.6x (59%) |
| blas3 | trmm | f64 | 512x512x512 | 7.686 ms | 17.46 | 0.68 | 30% | ok | 2.3x (39%) |
| blas3 | trsm | f64 | 512x512x512 | 8.385 ms | 16.01 | 0.63 | 27% | ok | 2.2x (36%) |
| blas3 | trsm | f32 | 512x512x512 | 4.168 ms | 32.20 | 0.63 | 27% | ok | 2.3x (38%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 37.879 ms | 0.14 | 1.12 | 10% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 20.986 ms | 4.26 | 0.20 | 7% | ok | 1.4x (24%) |
| factor | lu | f32 | 512x512 | 11.649 ms | 7.68 | 0.18 | 7% | ok | 1.3x (22%) |
| factor | cholesky | f64 | 512x512 | 25.734 ms | 1.74 | 0.16 | 3% | ok | 1.0x (17%) |
| factor | cholesky | f32 | 512x512 | 23.985 ms | 1.87 | 0.09 | 2% | ok | 1.0x (17%) |
| factor | ldlt | f64 | 512x512 | 26.033 ms | 1.72 | 0.16 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 512x512 | 12.728 ms | 3.52 | 0.33 | 6% | ok | 1.0x (17%) |
| factor | qr | f64 | 512x512 | 324.698 ms | 0.55 | 0.01 | 1% | ok | 1.5x (26%) |
| factor | lq | f64 | 512x512 | 55.191 ms | 3.24 | 0.08 | 5% | ok | 1.3x (22%) |
| factor | hessenberg | f64 | 512x512 | 404.224 ms | 1.11 | 0.01 | 2% | ok | 1.5x (26%) |
| factor | householder | f64 | 512x512 | 0.899 ms | 1.17 | 4.67 | 41%\* | ok | 3.7x (61%) |
| factor | inv | f64 | 512x512 | 149.159 ms | 1.80 | 0.03 | 3% | ok | 1.0x (17%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 16.999 ms | 5.26 | 0.25 | 9% | ok | 1.2x (21%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 60.723 ms | 0.37 | 0.02 | 1% | ok | 1.1x (18%) |
| eigen | eigen_symmetric | f64 | 256x256 | 174.084 ms | 0.87 | 0.01 | 1% | ok | 1.0x (16%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1005.526 ms | 0.15 | 0.00 | 0% | ok | 1.0x (16%) |
| eigen | eigenvalue | f64 | 256x256 | 104.378 ms | 1.61 | 0.01 | 3% | ok | 1.0x (17%) |
| eigen | schur | f64 | 256x256 | 260.930 ms | 1.61 | 0.00 | 3% | ok | 1.0x (17%) |
| eigen | svd | f64 | 256x256 | 261.107 ms | 1.41 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 256x256 | 32.359 ms | 7.78 | 0.03 | 13% | ok | 1.6x (26%) |
| eigen | sylvester | f64 | 256x256 | 580.799 ms | 1.73 | 0.00 | 3% | ok | 1.0x (17%) |
| elementwise | exp | f64 | n=8388608 | 157.394 ms | 0.05 | 0.85 | 7% | ok | 1.0x (16%) |
| elementwise | log | f64 | n=8388608 | 141.253 ms | 0.06 | 0.95 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 172.004 ms | 0.05 | 0.78 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=8388608 | 90.240 ms | 0.09 | 1.49 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 2502.576 ms | 0.00 | 0.05 | 0% | ok | 1.0x (16%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.302 ms | 1.14 | 11.83 | 104% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.119 ms | 0.84 | 8.73 | 76% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.337 ms | 3.69 | 7.77 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.307 ms | 0.20 | 0.71 | 6%\* | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.901 ms | 0.34 | 2.82 | 25% | ok | 1.0x (16%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 42.116 ms | 0.27 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 70.097 ms | 0.16 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 26.043 ms | 0.43 | 0.03 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 117.547 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 131.256 ms | 0.17 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 48.425 ms | 0.46 | 0.02 | 1% | ok | 1.0x (16%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 79.366 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 9.135 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.277 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 128.808 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.544 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.380 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.534 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.482 ms | 0.33 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.856 ms | 0.88 | 3.53 | 31%\* | ok | 1.0x (16%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.658 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.701 ms | 0.88 | 0.06 | 1% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.881 ms | 0.86 | 3.43 | 30%\* | ok | 0.9x (16%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.563 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 103.931 ms | 0.22 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.756 ms | 0.20 | 0.81 | 7%\* | ok | 1.1x (18%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 10.017 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 113.483 ms | 0.20 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.806 ms | 0.20 | 0.79 | 7%\* | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 16.726 ms | 0.00 | 0.00 | n/a | ok | 0.8x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 637.303 ms | 0.48 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.064 ms | 0.48 | 1.91 | 17%\* | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.862 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 203.983 ms | 1.51 | 0.00 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.049 ms | 0.48 | 1.91 | 17%\* | ok | 1.0x (16%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.784 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1714.061 ms | 0.36 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.740 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 18.182 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1963.990 ms | 0.31 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.658 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 80.486 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 124.228 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 60.491 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 32.095 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 38.553 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 37.448 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 66.197 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 69.526 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 37.586 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 43.802 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 40.061 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 38.686 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 45.200 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.130 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.624 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 128^2 grid | 35.827 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| pc | ic_0 | f64 | 128^2 grid | 2.870 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 2.770 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.093 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 2.393 ms | 0.00 | 0.00 | n/a | ok | 0.9x (14%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.596 ms | 0.73 | 8.15 | 71% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.314 ms | 0.79 | 8.85 | 78% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.936 ms | 0.66 | 7.45 | 65% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.740 ms | 1.08 | 8.13 | 71% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 140.222 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 144.479 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 130.901 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 16.523 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| mixed | dot | f32>f64 | n=8388608 | 6.278 ms | 2.67 | 10.69 | 94% | ok | 0.9x (16%) |
| mixed | gemm | f32>f64 | 512x512x512 | 11.766 ms | 22.81 | 0.36 | 39% | ok | 5.2x (87%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.250 ms | 7.46 | 7.92 | 69% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.048 ms | 2.77 | 11.10 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.972 ms | 5.64 | 11.29 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.078 ms | 8.07 | 8.07 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.641 ms | 10.22 | 10.22 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 3.945 ms | 68.04 | 0.80 | n/a | ok | 4.8x (80%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.535 ms | 59.20 | 0.46 | n/a | ok | 4.9x (81%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.488 ms | 59.81 | 0.35 | n/a | ok | 4.9x (82%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.318 ms | 80.91 | 0.47 | n/a | ok | 4.9x (81%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 2.568 ms | 104.54 | 0.61 | n/a | ok | 4.8x (80%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## xeon-e5-2420v2 -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 21.0 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 21.0 GFLOP/s |
| ... this build, scalar | f64 0.1, f32 0.7 GFLOP/s |
| bandwidth | read 10.2, update 10.3, triad 8.2 GB/s (ceiling 10.3) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 23% | 105% | 0 |
| blas2 | 10 | 92% | 104% | 0 |
| blas3 | 10 | 163% | 183% | 0 |
| factor | 12 | 54% | 92% | 0 |
| eigen | 8 | 51% | 120% | 0 |
| elementwise | 5 | 9% | 14% | 0 |
| sparse | 5 | 57% | 112% | 0 |
| direct | 19 | 3% | 36% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 19% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 80% | 86% | 0 |
| mg | 1 | 79% | 79% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 23% | 75% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 186.166 ms | 0.09 | 0.72 | 7% | ok |
| blas1 | dot | f32 | n=8388608 | 27.981 ms | 0.60 | 2.40 | 23% | ok |
| blas1 | axpy | f64 | n=8388608 | 211.985 ms | 0.08 | 0.95 | 9% | ok |
| blas1 | axpy | f32 | n=8388608 | 91.420 ms | 0.18 | 1.10 | 11% | ok |
| blas1 | scale | f64 | n=8388608 | 12.449 ms | 0.67 | 10.78 | 105% | ok |
| blas1 | two_norm | f64 | n=8388608 | 183.904 ms | 0.09 | 0.36 | 4% | ok |
| blas1 | one_norm | f64 | n=8388608 | 9.813 ms | 0.85 | 6.84 | 67% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 13.095 ms | 0.64 | 5.12 | 50% | ok |
| blas1 | sum | f64 | n=8388608 | 9.720 ms | 0.86 | 6.90 | 67% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 281.464 ms | 0.66 | 0.48 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 14.414 ms | 2.33 | 9.32 | 91% | ok |
| blas2 | gemv | f32 | n=4096 | 6.290 ms | 5.33 | 10.67 | 104% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 12.961 ms | 2.59 | 10.36 | 101% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.596 ms | 5.09 | 10.18 | 99% | ok |
| blas2 | symv | f64 | n=4096 | 7.062 ms | 4.75 | 9.51 | 93% | ok |
| blas2 | trmv | f64 | n=4096 | 7.540 ms | 2.23 | 8.91 | 87% | ok |
| blas2 | trmv | f32 | n=4096 | 3.691 ms | 4.55 | 9.10 | 89% | ok |
| blas2 | trsv | f64 | n=4096 | 7.479 ms | 2.24 | 8.98 | 88% | ok |
| blas2 | trsv | f32 | n=4096 | 3.567 ms | 4.70 | 9.42 | 92% | ok |
| blas2 | ger | f64 | n=4096 | 28.868 ms | 1.16 | 9.30 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 14.000 ms | 19.17 | 0.45 | 183% | ok |
| blas3 | gemm | f32 | 512x512x512 | 8.222 ms | 32.65 | 0.38 | 155% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 8.819 ms | 15.22 | 0.71 | 145% | ok |
| blas3 | symm | f64 | 512x512x512 | 14.159 ms | 18.96 | 0.44 | 181% | ok |
| blas3 | syrk | f64 | 512x512x512 | 7.839 ms | 17.12 | 0.54 | 163% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 15.234 ms | 17.62 | 0.41 | 168% | ok |
| blas3 | trmm | f64 | 512x512x512 | 7.467 ms | 17.98 | 0.70 | 171% | ok |
| blas3 | trsm | f64 | 512x512x512 | 8.183 ms | 16.40 | 0.64 | 156% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.267 ms | 25.48 | 0.50 | 121% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.486 ms | 0.14 | 1.10 | 11% | ok |
| factor | lu | f64 | 512x512 | 9.312 ms | 9.61 | 0.45 | 92% | ok |
| factor | lu | f32 | 512x512 | 6.442 ms | 13.89 | 0.33 | 66% | ok |
| factor | cholesky | f64 | 512x512 | 5.809 ms | 7.70 | 0.72 | 73% | ok |
| factor | cholesky | f32 | 512x512 | 3.972 ms | 11.26 | 0.53 | 54% | ok |
| factor | ldlt | f64 | 512x512 | 25.427 ms | 1.76 | 0.16 | 17% | ok |
| factor | ldlt_bk | f64 | 512x512 | 13.968 ms | 3.20 | 0.30 | 31% | ok |
| factor | qr | f64 | 512x512 | 19.713 ms | 9.08 | 0.21 | 87% | ok |
| factor | lq | f64 | 512x512 | 74.243 ms | 2.41 | 0.06 | 23% | ok |
| factor | hessenberg | f64 | 512x512 | 632.526 ms | 0.71 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.174 ms | 0.33 | 1.32 | 13%\* | ok |
| factor | inv | f64 | 512x512 | 128.313 ms | 2.09 | 0.03 | 20% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.722 ms | 7.03 | 0.33 | 67% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 6.443 ms | 3.47 | 0.16 | 33% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 174.020 ms | 0.87 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1012.140 ms | 0.15 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 53.534 ms | 3.13 | 0.02 | 30% | ok |
| eigen | schur | f64 | 256x256 | 72.401 ms | 5.79 | 0.01 | 55% | ok |
| eigen | svd | f64 | 256x256 | 29.450 ms | 12.53 | 0.04 | 120% | ok |
| eigen | expm | f64 | 256x256 | 35.586 ms | 7.07 | 0.03 | 67% | ok |
| eigen | sylvester | f64 | 256x256 | 189.866 ms | 5.30 | 0.01 | 51% | ok |
| elementwise | exp | f64 | n=8388608 | 151.966 ms | 0.06 | 0.88 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 140.118 ms | 0.06 | 0.96 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 170.464 ms | 0.05 | 0.79 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 91.149 ms | 0.09 | 1.47 | 14% | ok |
| elementwise | erf | f64 | n=8388608 | 2467.398 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.374 ms | 1.10 | 11.47 | 112% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.135 ms | 0.83 | 8.69 | 85% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.131 ms | 2.77 | 5.82 | 57% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.320 ms | 0.20 | 0.70 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.952 ms | 0.34 | 2.80 | 27% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 40.815 ms | 0.27 | 0.02 | 3% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 68.655 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 25.094 ms | 0.45 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 115.651 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 129.011 ms | 0.17 | 0.01 | 2% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 47.381 ms | 0.47 | 0.02 | 5% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 81.793 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.904 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.136 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 125.487 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.522 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 40.581 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.232 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.321 ms | 0.34 | 0.02 | 3% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.821 ms | 0.92 | 3.69 | 36%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.381 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.519 ms | 0.89 | 0.06 | 9% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.840 ms | 0.90 | 3.60 | 35%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.328 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 102.684 ms | 0.22 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.763 ms | 0.20 | 0.80 | 8%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.746 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 110.921 ms | 0.20 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.831 ms | 0.20 | 0.79 | 8%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 12.880 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 633.709 ms | 0.49 | 0.00 | 5% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 6.842 ms | 0.49 | 1.97 | 19%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.689 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 207.346 ms | 1.49 | 0.00 | 14% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 6.904 ms | 0.49 | 1.95 | 19%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.578 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1721.392 ms | 0.36 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.540 ms | 0.10 | 0.41 | 4%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 18.184 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1523.882 ms | 0.41 | 0.00 | 4% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.586 ms | 0.10 | 0.41 | 4%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 231.726 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 421.267 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 134.929 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 109.689 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 130.321 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 80.494 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 534.920 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 174.610 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.762 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 203.422 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 193.284 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 212.525 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 312.631 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.172 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.767 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 33.313 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.906 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.737 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.094 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 3.472 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.564 ms | 0.73 | 8.23 | 80% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.330 ms | 0.79 | 8.81 | 86% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.928 ms | 0.67 | 7.47 | 73% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.871 ms | 1.07 | 8.10 | 79% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 135.517 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 141.933 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 126.679 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.296 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 188.958 ms | 0.09 | 0.36 | 3% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 113.706 ms | 2.36 | 0.04 | 23% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.325 ms | 7.21 | 7.67 | 75% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.528 ms | 2.57 | 10.28 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.106 ms | 5.40 | 10.80 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.984 ms | 3.37 | 3.37 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.979 ms | 3.37 | 3.37 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 57.528 ms | 4.67 | 0.05 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 45.067 ms | 5.96 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 24.033 ms | 11.17 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 23.694 ms | 11.33 | 0.07 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 71.755 ms | 3.74 | 0.02 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

## xeon-e5-2420v2 -- blas, 6 threads, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 59.1 GFLOP/s, f32 118.3 GFLOP/s |
| ... hardware, SSE2 | f64 59.1, f32 118.3 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.7 GFLOP/s |
| bandwidth | read 11.4, update 10.6, triad 8.5 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 53% | 107% | 0 | 85% |
| blas2 | 10 | 101% | 108% | 0 | 19% |
| blas3 | 10 | 125% | 157% | 0 | 75% |
| factor | 12 | 17% | 40% | 0 | 25% |
| eigen | 8 | 9% | 23% | 0 | 17% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 68% | 104% | 0 | 17% |
| direct | 19 | 1% | 30% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 4% | 17% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 73% | 78% | 0 | 17% |
| mg | 1 | 72% | 72% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 16% |
| mixed | 3 | 22% | 69% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 83% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 33.406 ms | 0.50 | 4.02 | 35% | ok | 5.6x (93%) |
| blas1 | dot | f32 | n=8388608 | 5.510 ms | 3.04 | 12.18 | 107% | ok | 5.1x (85%) |
| blas1 | axpy | f64 | n=8388608 | 38.806 ms | 0.43 | 5.19 | 45% | ok | 5.5x (91%) |
| blas1 | axpy | f32 | n=8388608 | 16.768 ms | 1.00 | 6.00 | 53% | ok | 5.5x (91%) |
| blas1 | scale | f64 | n=8388608 | 12.084 ms | 0.69 | 11.11 | 97% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 33.018 ms | 0.51 | 2.03 | 18% | ok | 5.6x (93%) |
| blas1 | one_norm | f64 | n=8388608 | 9.757 ms | 0.86 | 6.88 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=8388608 | 13.072 ms | 0.64 | 5.13 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=8388608 | 9.698 ms | 0.86 | 6.92 | 61% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=8388608 | 278.755 ms | 0.66 | 0.48 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=4096 | 11.310 ms | 2.97 | 11.87 | 104% | ok | 1.3x (21%) |
| blas2 | gemv | f32 | n=4096 | 5.462 ms | 6.14 | 12.29 | 108% | ok | 1.2x (19%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 11.218 ms | 2.99 | 11.97 | 105% | ok | 1.2x (19%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 7.521 ms | 4.46 | 8.93 | 78% | ok | 0.9x (15%) |
| blas2 | symv | f64 | n=4096 | 5.796 ms | 5.79 | 11.59 | 101% | ok | 1.2x (20%) |
| blas2 | trmv | f64 | n=4096 | 5.733 ms | 2.93 | 11.72 | 103% | ok | 1.3x (22%) |
| blas2 | trmv | f32 | n=4096 | 2.954 ms | 5.68 | 11.37 | 99% | ok | 1.2x (21%) |
| blas2 | trsv | f64 | n=4096 | 7.481 ms | 2.24 | 8.98 | 79% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=4096 | 3.548 ms | 4.73 | 9.47 | 83% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=4096 | 25.917 ms | 1.29 | 10.36 | 91% | ok | 1.1x (19%) |
| blas3 | gemm | f64 | 512x512x512 | 2.897 ms | 92.66 | 2.17 | 157% | ok | 4.8x (81%) |
| blas3 | gemm | f32 | 512x512x512 | 1.818 ms | 147.61 | 1.73 | 125% | ok | 4.5x (75%) |
| blas3 | gemmt | f64 | 512x512x512 | 2.583 ms | 51.97 | 2.44 | 88% | ok | 3.4x (57%) |
| blas3 | symm | f64 | 512x512x512 | 2.929 ms | 91.66 | 2.15 | 155% | ok | 4.8x (81%) |
| blas3 | syrk | f64 | 512x512x512 | 2.530 ms | 53.04 | 1.66 | 90% | ok | 3.1x (52%) |
| blas3 | syr2k | f64 | 512x512x512 | 3.724 ms | 72.08 | 1.69 | 122% | ok | 4.1x (68%) |
| blas3 | trmm | f64 | 512x512x512 | 1.618 ms | 82.94 | 3.24 | 140% | ok | 4.6x (77%) |
| blas3 | trsm | f64 | 512x512x512 | 1.765 ms | 76.06 | 2.97 | 129% | ok | 4.6x (77%) |
| blas3 | trsm | f32 | 512x512x512 | 1.245 ms | 107.80 | 2.11 | 91% | ok | 4.2x (70%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.847 ms | 0.14 | 1.09 | 10% | ok | 1.0x (17%) |
| factor | lu | f64 | 512x512 | 5.595 ms | 15.99 | 0.75 | 27% | ok | 1.7x (28%) |
| factor | lu | f32 | 512x512 | 4.111 ms | 21.77 | 0.51 | 18% | ok | 1.6x (26%) |
| factor | cholesky | f64 | 512x512 | 4.354 ms | 10.27 | 0.96 | 17% | ok | 1.3x (22%) |
| factor | cholesky | f32 | 512x512 | 3.106 ms | 14.40 | 0.68 | 12% | ok | 1.3x (21%) |
| factor | ldlt | f64 | 512x512 | 26.513 ms | 1.69 | 0.16 | 3% | ok | 1.0x (16%) |
| factor | ldlt_bk | f64 | 512x512 | 14.209 ms | 3.15 | 0.30 | 5% | ok | 1.0x (16%) |
| factor | qr | f64 | 512x512 | 13.250 ms | 13.51 | 0.32 | 23% | ok | 1.5x (25%) |
| factor | lq | f64 | 512x512 | 48.679 ms | 3.68 | 0.09 | 6% | ok | 1.5x (25%) |
| factor | hessenberg | f64 | 512x512 | 401.665 ms | 1.11 | 0.01 | 2% | ok | 1.6x (26%) |
| factor | householder | f64 | 512x512 | 0.912 ms | 1.15 | 4.60 | 40%\* | ok | 3.5x (58%) |
| factor | inv | f64 | 512x512 | 132.722 ms | 2.02 | 0.03 | 3% | ok | 1.0x (16%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 8.703 ms | 10.28 | 0.48 | 17% | ok | 1.5x (24%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 5.890 ms | 3.80 | 0.18 | 6% | ok | 1.1x (18%) |
| eigen | eigen_symmetric | f64 | 256x256 | 171.284 ms | 0.88 | 0.01 | 1% | ok | 1.0x (17%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 982.781 ms | 0.15 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 256x256 | 56.911 ms | 2.95 | 0.02 | 5% | ok | 0.9x (16%) |
| eigen | schur | f64 | 256x256 | 76.159 ms | 5.51 | 0.01 | 9% | ok | 1.0x (16%) |
| eigen | svd | f64 | 256x256 | 26.850 ms | 13.75 | 0.04 | 23% | ok | 1.1x (18%) |
| eigen | expm | f64 | 256x256 | 27.410 ms | 9.18 | 0.04 | 16% | ok | 1.3x (22%) |
| eigen | sylvester | f64 | 256x256 | 194.627 ms | 5.17 | 0.01 | 9% | ok | 1.0x (16%) |
| elementwise | exp | f64 | n=8388608 | 152.118 ms | 0.06 | 0.88 | 8% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=8388608 | 140.269 ms | 0.06 | 0.96 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=8388608 | 173.758 ms | 0.05 | 0.77 | 7% | ok | 1.0x (16%) |
| elementwise | sqrt | f64 | n=8388608 | 90.046 ms | 0.09 | 1.49 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=8388608 | 2499.109 ms | 0.00 | 0.05 | 0% | ok | 1.0x (16%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.289 ms | 1.14 | 11.90 | 104% | ok | 1.0x (17%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.112 ms | 0.84 | 8.75 | 77% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 11.318 ms | 3.70 | 7.78 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.265 ms | 0.20 | 0.71 | 6%\* | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.934 ms | 0.34 | 2.81 | 25% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 41.171 ms | 0.27 | 0.02 | 0% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 70.138 ms | 0.16 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 25.072 ms | 0.45 | 0.03 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 121.649 ms | 0.18 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 131.078 ms | 0.17 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 47.544 ms | 0.47 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 82.030 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.875 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.131 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 125.450 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.571 ms | 0.00 | 0.00 | n/a | ok | 0.9x (15%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 42.141 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.168 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.425 ms | 0.33 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.876 ms | 0.86 | 3.45 | 30%\* | ok | 0.9x (16%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.345 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.684 ms | 0.88 | 0.06 | 1% | ok | 1.0x (16%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.895 ms | 0.85 | 3.38 | 30%\* | ok | 0.9x (16%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.312 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 106.019 ms | 0.21 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.772 ms | 0.20 | 0.80 | 7%\* | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.747 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 114.651 ms | 0.20 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.800 ms | 0.20 | 0.80 | 7%\* | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 12.878 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 636.133 ms | 0.49 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 8.131 ms | 0.41 | 1.66 | 15%\* | ok | 0.8x (14%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 29.378 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 203.709 ms | 1.51 | 0.00 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.056 ms | 0.48 | 1.91 | 17%\* | ok | 1.0x (16%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.510 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1710.838 ms | 0.36 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.652 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 17.959 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1519.827 ms | 0.41 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.889 ms | 0.10 | 0.41 | 4%\* | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 232.016 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 408.953 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 129.086 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 103.886 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 122.972 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 80.665 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 538.021 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 175.622 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 92.095 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 201.668 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 193.502 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 218.373 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 307.802 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.172 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.367 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| pc | ilut | f64 | 128^2 grid | 32.693 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 128^2 grid | 2.872 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 128^2 grid | 2.715 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 128^2 grid | 0.093 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 3.462 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.517 ms | 0.74 | 8.34 | 73% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.303 ms | 0.79 | 8.88 | 78% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.808 ms | 0.69 | 7.70 | 67% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 35.533 ms | 1.08 | 8.18 | 72% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 140.798 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 147.837 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 129.230 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.507 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=8388608 | 187.491 ms | 0.09 | 0.36 | 3% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 512x512x512 | 20.558 ms | 13.06 | 0.20 | 22% | ok | 5.5x (92%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.246 ms | 7.47 | 7.94 | 69% | ok | 1.0x (17%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 10.650 ms | 1.58 | 6.30 | n/a | ok | 0.6x (10%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.132 ms | 5.36 | 10.71 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.993 ms | 3.36 | 3.36 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.982 ms | 3.37 | 3.37 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 11.525 ms | 23.29 | 0.27 | n/a | ok | 5.0x (83%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 8.278 ms | 32.43 | 0.25 | n/a | ok | 5.4x (91%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.326 ms | 62.05 | 0.36 | n/a | ok | 5.6x (93%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.323 ms | 62.10 | 0.36 | n/a | ok | 5.5x (91%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 13.063 ms | 20.55 | 0.12 | n/a | ok | 5.5x (92%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

