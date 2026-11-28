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

