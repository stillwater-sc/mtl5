## xeon-e5-2420v2 -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 20.9 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 20.9 GFLOP/s |
| ... this build, scalar | f64 0.1, f32 0.7 GFLOP/s |
| bandwidth | read 10.2, update 10.3, triad 8.2 GB/s (ceiling 10.3) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 23% | 104% | 0 |
| blas2 | 10 | 90% | 103% | 0 |
| blas3 | 10 | 164% | 182% | 0 |
| factor | 12 | 55% | 93% | 0 |
| eigen | 8 | 51% | 120% | 0 |
| elementwise | 5 | 9% | 14% | 0 |
| sparse | 5 | 56% | 108% | 0 |
| direct | 19 | 3% | 37% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 19% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 80% | 86% | 0 |
| mg | 1 | 77% | 77% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 21% | 73% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 187.237 ms | 0.09 | 0.72 | 7% | ok |
| blas1 | dot | f32 | n=8388608 | 27.846 ms | 0.60 | 2.41 | 23% | ok |
| blas1 | axpy | f64 | n=8388608 | 212.495 ms | 0.08 | 0.95 | 9% | ok |
| blas1 | axpy | f32 | n=8388608 | 91.608 ms | 0.18 | 1.10 | 11% | ok |
| blas1 | scale | f64 | n=8388608 | 12.532 ms | 0.67 | 10.71 | 104% | ok |
| blas1 | two_norm | f64 | n=8388608 | 185.889 ms | 0.09 | 0.36 | 4% | ok |
| blas1 | one_norm | f64 | n=8388608 | 9.927 ms | 0.85 | 6.76 | 66% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 13.130 ms | 0.64 | 5.11 | 50% | ok |
| blas1 | sum | f64 | n=8388608 | 9.760 ms | 0.86 | 6.88 | 67% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 278.968 ms | 0.66 | 0.48 | 6% | ok |
| blas2 | gemv | f64 | n=4096 | 14.513 ms | 2.31 | 9.25 | 90% | ok |
| blas2 | gemv | f32 | n=4096 | 6.323 ms | 5.31 | 10.62 | 103% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 13.311 ms | 2.52 | 10.09 | 98% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 6.777 ms | 4.95 | 9.91 | 96% | ok |
| blas2 | symv | f64 | n=4096 | 7.234 ms | 4.64 | 9.29 | 90% | ok |
| blas2 | trmv | f64 | n=4096 | 7.632 ms | 2.20 | 8.80 | 85% | ok |
| blas2 | trmv | f32 | n=4096 | 3.755 ms | 4.47 | 8.94 | 87% | ok |
| blas2 | trsv | f64 | n=4096 | 7.535 ms | 2.23 | 8.91 | 87% | ok |
| blas2 | trsv | f32 | n=4096 | 3.694 ms | 4.54 | 9.09 | 88% | ok |
| blas2 | ger | f64 | n=4096 | 28.796 ms | 1.17 | 9.32 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 14.058 ms | 19.09 | 0.45 | 182% | ok |
| blas3 | gemm | f32 | 512x512x512 | 9.007 ms | 29.80 | 0.35 | 142% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 8.995 ms | 14.92 | 0.70 | 143% | ok |
| blas3 | symm | f64 | 512x512x512 | 14.145 ms | 18.98 | 0.44 | 181% | ok |
| blas3 | syrk | f64 | 512x512x512 | 7.798 ms | 17.21 | 0.54 | 164% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 15.214 ms | 17.64 | 0.41 | 169% | ok |
| blas3 | trmm | f64 | 512x512x512 | 7.569 ms | 17.73 | 0.69 | 169% | ok |
| blas3 | trsm | f64 | 512x512x512 | 8.162 ms | 16.44 | 0.64 | 157% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.293 ms | 25.36 | 0.50 | 121% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 38.502 ms | 0.14 | 1.10 | 11% | ok |
| factor | lu | f64 | 512x512 | 9.234 ms | 9.69 | 0.45 | 93% | ok |
| factor | lu | f32 | 512x512 | 6.375 ms | 14.04 | 0.33 | 67% | ok |
| factor | cholesky | f64 | 512x512 | 5.758 ms | 7.77 | 0.73 | 74% | ok |
| factor | cholesky | f32 | 512x512 | 3.896 ms | 11.48 | 0.54 | 55% | ok |
| factor | ldlt | f64 | 512x512 | 25.018 ms | 1.79 | 0.17 | 17% | ok |
| factor | ldlt_bk | f64 | 512x512 | 13.592 ms | 3.29 | 0.31 | 31% | ok |
| factor | qr | f64 | 512x512 | 19.600 ms | 9.13 | 0.21 | 87% | ok |
| factor | lq | f64 | 512x512 | 73.982 ms | 2.42 | 0.06 | 23% | ok |
| factor | hessenberg | f64 | 512x512 | 621.571 ms | 0.72 | 0.01 | 7% | ok |
| factor | householder | f64 | 512x512 | 3.247 ms | 0.32 | 1.29 | 13%\* | ok |
| factor | inv | f64 | 512x512 | 129.634 ms | 2.07 | 0.03 | 20% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.862 ms | 6.96 | 0.33 | 66% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 6.475 ms | 3.46 | 0.16 | 33% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 174.060 ms | 0.87 | 0.01 | 8% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 1007.853 ms | 0.15 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 256x256 | 53.747 ms | 3.12 | 0.02 | 30% | ok |
| eigen | schur | f64 | 256x256 | 72.464 ms | 5.79 | 0.01 | 55% | ok |
| eigen | svd | f64 | 256x256 | 29.448 ms | 12.53 | 0.04 | 120% | ok |
| eigen | expm | f64 | 256x256 | 35.852 ms | 7.02 | 0.03 | 67% | ok |
| eigen | sylvester | f64 | 256x256 | 190.475 ms | 5.28 | 0.01 | 51% | ok |
| elementwise | exp | f64 | n=8388608 | 152.711 ms | 0.05 | 0.88 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 139.360 ms | 0.06 | 0.96 | 9% | ok |
| elementwise | sin | f64 | n=8388608 | 170.365 ms | 0.05 | 0.79 | 8% | ok |
| elementwise | sqrt | f64 | n=8388608 | 92.280 ms | 0.09 | 1.45 | 14% | ok |
| elementwise | erf | f64 | n=8388608 | 2504.795 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 2.451 ms | 1.07 | 11.11 | 108% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 3.126 ms | 0.84 | 8.71 | 85% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 15.219 ms | 2.75 | 5.79 | 56% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.250 ms | 0.20 | 0.71 | 7%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 5.825 ms | 0.34 | 2.86 | 28% | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 40.706 ms | 0.27 | 0.02 | 3% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 68.958 ms | 0.16 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.665 ms | 0.45 | 0.03 | 4% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 113.940 ms | 0.20 | 0.01 | 2% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 128.442 ms | 0.17 | 0.01 | 2% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 47.549 ms | 0.47 | 0.02 | 4% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 81.632 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 8.843 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.143 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 125.303 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.563 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 41.492 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 6.225 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 33.274 ms | 0.34 | 0.02 | 3% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.802 ms | 0.94 | 3.77 | 37%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.318 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 12.452 ms | 0.90 | 0.06 | 9% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.818 ms | 0.92 | 3.70 | 36%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.350 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 102.271 ms | 0.22 | 0.01 | 2% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.817 ms | 0.20 | 0.79 | 8%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.821 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 110.685 ms | 0.20 | 0.01 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 3.847 ms | 0.20 | 0.79 | 8%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 12.972 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 644.048 ms | 0.48 | 0.00 | 5% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 6.842 ms | 0.49 | 1.97 | 19%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 28.865 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 208.371 ms | 1.48 | 0.00 | 14% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 6.892 ms | 0.49 | 1.96 | 19%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 12.456 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1769.780 ms | 0.35 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.871 ms | 0.10 | 0.41 | 4%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 18.253 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1574.807 ms | 0.39 | 0.00 | 4% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 32.562 ms | 0.10 | 0.41 | 4%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 242.856 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 407.369 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 128.964 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 104.488 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 123.363 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 80.615 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 536.398 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 174.944 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 91.426 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 203.412 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 195.741 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 214.663 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 307.864 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.172 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 4.371 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 32.719 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.870 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.736 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.092 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 3.395 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 3.566 ms | 0.73 | 8.22 | 80% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 3.294 ms | 0.79 | 8.90 | 86% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.937 ms | 0.66 | 7.45 | 72% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 36.728 ms | 1.05 | 7.91 | 77% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 137.803 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 145.067 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 129.419 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 42.391 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 188.540 ms | 0.09 | 0.36 | 3% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 119.769 ms | 2.24 | 0.04 | 21% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 2.372 ms | 7.07 | 7.52 | 73% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 6.467 ms | 2.59 | 10.38 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.125 ms | 5.37 | 10.74 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 4.978 ms | 3.37 | 3.37 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 4.983 ms | 3.37 | 3.37 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 59.039 ms | 4.55 | 0.05 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 45.196 ms | 5.94 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 23.667 ms | 11.34 | 0.07 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 23.804 ms | 11.28 | 0.07 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 70.096 ms | 3.83 | 0.02 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

