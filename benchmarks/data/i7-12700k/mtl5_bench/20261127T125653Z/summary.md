# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,8, CPUs 0,2,4,6,8,10,12,14, clock policy governor performance on the P-cores, turbo on; performance 800-4900 MHz on cpus 0,2,8,10,12,14; performance 800-5000 MHz on cpus 4,6, 2026-11-27T13:08:58Z.

## i7-12700k -- native, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.5 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.5 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.3 GFLOP/s |
| bandwidth | read 33.0, update 38.3, triad 32.5 GB/s (ceiling 38.3) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 44% | 98% | 0 |
| blas2 | 10 | 79% | 95% | 0 |
| blas3 | 10 | 17% | 18% | 0 |
| factor | 12 | 8% | 19% | 0 |
| eigen | 8 | 6% | 13% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 50% | 95% | 0 |
| direct | 19 | 1% | 31% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 24% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 63% | 78% | 0 |
| mg | 1 | 57% | 57% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 36% | 86% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.612 ms | 2.99 | 23.92 | 62% | ok |
| blas1 | dot | f32 | n=8388608 | 4.236 ms | 3.96 | 15.84 | 41% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.483 ms | 2.59 | 31.05 | 81% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.723 ms | 3.55 | 21.31 | 56% | ok |
| blas1 | scale | f64 | n=8388608 | 3.578 ms | 2.34 | 37.51 | 98% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.940 ms | 3.40 | 13.58 | 35% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.044 ms | 2.07 | 16.60 | 43% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.047 ms | 1.19 | 9.52 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 3.993 ms | 2.10 | 16.81 | 44% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 77.522 ms | 2.38 | 1.73 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.678 ms | 9.12 | 36.51 | 95% | ok |
| blas2 | gemv | f32 | n=4096 | 2.729 ms | 12.30 | 24.60 | 64% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 4.783 ms | 7.02 | 28.08 | 73% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.221 ms | 15.10 | 30.22 | 79% | ok |
| blas2 | symv | f64 | n=4096 | 3.182 ms | 10.54 | 21.11 | 55% | ok |
| blas2 | trmv | f64 | n=4096 | 2.032 ms | 8.26 | 33.06 | 86% | ok |
| blas2 | trmv | f32 | n=4096 | 1.425 ms | 11.77 | 23.56 | 61% | ok |
| blas2 | trsv | f64 | n=4096 | 1.979 ms | 8.48 | 33.94 | 89% | ok |
| blas2 | trsv | f32 | n=4096 | 1.397 ms | 12.01 | 24.05 | 63% | ok |
| blas2 | ger | f64 | n=4096 | 7.688 ms | 4.36 | 34.93 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 19.871 ms | 13.51 | 0.32 | 17% | ok |
| blas3 | gemm | f32 | 512x512x512 | 9.996 ms | 26.86 | 0.31 | 17% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 12.529 ms | 10.71 | 0.50 | 14% | ok |
| blas3 | symm | f64 | 512x512x512 | 19.745 ms | 13.60 | 0.32 | 17% | ok |
| blas3 | syrk | f64 | 512x512x512 | 13.600 ms | 9.87 | 0.31 | 13% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 24.471 ms | 10.97 | 0.26 | 14% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.551 ms | 14.05 | 0.55 | 18% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.715 ms | 13.82 | 0.54 | 18% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.215 ms | 25.74 | 0.50 | 16% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.143 ms | 0.40 | 3.23 | 8% | ok |
| factor | lu | f64 | 512x512 | 11.624 ms | 7.70 | 0.36 | 10% | ok |
| factor | lu | f32 | 512x512 | 4.489 ms | 19.93 | 0.47 | 13% | ok |
| factor | cholesky | f64 | 512x512 | 7.328 ms | 6.11 | 0.57 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.596 ms | 6.78 | 0.32 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.475 ms | 5.99 | 0.56 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 4.834 ms | 9.26 | 0.87 | 12% | ok |
| factor | qr | f64 | 512x512 | 234.563 ms | 0.76 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 25.838 ms | 6.93 | 0.16 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 275.555 ms | 1.62 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.642 ms | 0.64 | 2.55 | 7%\* | ok |
| factor | inv | f64 | 512x512 | 54.086 ms | 4.96 | 0.08 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 5.955 ms | 15.02 | 0.70 | 19% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 17.739 ms | 1.26 | 0.06 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 58.463 ms | 2.58 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 749.050 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 33.215 ms | 5.05 | 0.03 | 6% | ok |
| eigen | schur | f64 | 256x256 | 81.636 ms | 5.14 | 0.01 | 7% | ok |
| eigen | svd | f64 | 256x256 | 111.975 ms | 3.30 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 25.712 ms | 9.79 | 0.04 | 13% | ok |
| eigen | sylvester | f64 | 256x256 | 190.794 ms | 5.28 | 0.01 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 39.389 ms | 0.21 | 3.41 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 45.913 ms | 0.18 | 2.92 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 60.135 ms | 0.14 | 2.23 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 29.066 ms | 0.29 | 4.62 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 204.049 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.744 ms | 3.52 | 36.61 | 95% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.916 ms | 2.86 | 29.73 | 78% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.634 ms | 9.04 | 19.00 | 50% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 5.263 ms | 0.62 | 2.19 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.329 ms | 1.51 | 12.52 | 33%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 14.536 ms | 0.77 | 0.05 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 24.967 ms | 0.45 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.404 ms | 1.33 | 0.09 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.919 ms | 0.56 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.095 ms | 0.51 | 0.02 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.624 ms | 1.35 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 24.933 ms | 0.08 | 0.01 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.363 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.980 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 58.482 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.185 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.215 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.091 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 12.804 ms | 0.87 | 0.06 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.253 ms | 2.98 | 11.94 | 31%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.059 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.073 ms | 2.75 | 0.18 | 4% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.251 ms | 3.02 | 12.07 | 31%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.104 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.872 ms | 0.62 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.630 ms | 1.20 | 4.80 | 13%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.453 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.310 ms | 0.58 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.639 ms | 1.18 | 4.73 | 12%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.920 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 205.875 ms | 1.50 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.456 ms | 2.31 | 9.26 | 24%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.634 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 59.916 ms | 5.15 | 0.01 | 7% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.457 ms | 2.31 | 9.25 | 24%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.887 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 581.940 ms | 1.06 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.537 ms | 0.45 | 1.79 | 5%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.254 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 453.938 ms | 1.36 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.538 ms | 0.45 | 1.79 | 5%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 31.318 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 48.204 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.958 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.574 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 15.146 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 14.139 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.617 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.223 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.332 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 18.182 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.720 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 16.282 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 19.195 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.042 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.561 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 11.520 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.087 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.017 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.023 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.703 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.986 ms | 2.65 | 29.74 | 78% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.221 ms | 2.14 | 24.03 | 63% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.564 ms | 1.67 | 18.75 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.335 ms | 2.88 | 21.80 | 57% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.787 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 30.699 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.389 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 5.023 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.927 ms | 3.41 | 13.62 | 36% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 38.987 ms | 6.89 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.543 ms | 30.90 | 32.83 | 86%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.314 ms | 7.25 | 29.00 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.999 ms | 16.80 | 33.59 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.723 ms | 9.74 | 9.74 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.724 ms | 9.73 | 9.73 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.543 ms | 15.30 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.819 ms | 12.89 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.232 ms | 26.23 | 0.15 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.229 ms | 26.24 | 0.15 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 26.805 ms | 10.01 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## i7-12700k -- native, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 600.7 GFLOP/s, f32 1201.4 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.7, f32 1201.4 GFLOP/s |
| ... this build, scalar | f64 9.9, f32 10.0 GFLOP/s |
| bandwidth | read 83.7, update 78.5, triad 76.8 GB/s (ceiling 83.7) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 89% | 123% | 0 | 30% |
| blas2 | 10 | 42% | 137% | 0 | 13% |
| blas3 | 10 | 2% | 17% | 0 | 14% |
| factor | 12 | 2% | 12% | 0 | 15% |
| eigen | 8 | 1% | 4% | 0 | 13% |
| elementwise | 5 | 4% | 5% | 0 | 13% |
| sparse | 5 | 35% | 287% | 0 | 13% |
| direct | 19 | 0% | 14% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 10% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 36% | 0 | 13% |
| mg | 1 | 26% | 26% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 40% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 92% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.395 ms | 12.02 | 96.20 | 115% | ok | 4.0x (50%) |
| blas1 | dot | f32 | n=8388608 | 0.651 ms | 25.76 | 103.05 | 123% | ok | 6.5x (81%) |
| blas1 | axpy | f64 | n=8388608 | 2.746 ms | 6.11 | 73.31 | 88% | ok | 2.4x (30%) |
| blas1 | axpy | f32 | n=8388608 | 1.350 ms | 12.43 | 74.59 | 89% | ok | 3.5x (44%) |
| blas1 | scale | f64 | n=8388608 | 1.775 ms | 4.73 | 75.60 | 90% | ok | 2.0x (25%) |
| blas1 | two_norm | f64 | n=8388608 | 0.740 ms | 22.66 | 90.63 | 108% | ok | 6.7x (83%) |
| blas1 | one_norm | f64 | n=8388608 | 3.954 ms | 2.12 | 16.97 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.961 ms | 1.21 | 9.64 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.960 ms | 2.12 | 16.95 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 75.818 ms | 2.43 | 1.77 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.443 ms | 23.25 | 93.03 | 111% | ok | 2.5x (32%) |
| blas2 | gemv | f32 | n=4096 | 0.585 ms | 57.35 | 114.76 | 137% | ok | 4.7x (58%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.556 ms | 21.57 | 86.31 | 103% | ok | 3.1x (38%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.897 ms | 37.41 | 74.85 | 89% | ok | 2.5x (31%) |
| blas2 | symv | f64 | n=4096 | 3.137 ms | 10.70 | 21.41 | 26% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 2.019 ms | 8.31 | 33.27 | 40% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 1.409 ms | 11.90 | 23.83 | 28% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 1.968 ms | 8.52 | 34.13 | 41% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.384 ms | 12.12 | 24.26 | 29% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.726 ms | 4.34 | 34.75 | 42% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.937 ms | 91.40 | 2.14 | 15% | ok | 6.8x (85%) |
| blas3 | gemm | f32 | 512x512x512 | 1.352 ms | 198.50 | 2.33 | 17% | ok | 7.4x (92%) |
| blas3 | gemmt | f64 | 512x512x512 | 10.690 ms | 12.56 | 0.59 | 2% | ok | 1.2x (15%) |
| blas3 | symm | f64 | 512x512x512 | 18.503 ms | 14.51 | 0.34 | 2% | ok | 1.1x (13%) |
| blas3 | syrk | f64 | 512x512x512 | 11.762 ms | 11.41 | 0.36 | 2% | ok | 1.2x (14%) |
| blas3 | syr2k | f64 | 512x512x512 | 19.719 ms | 13.61 | 0.32 | 2% | ok | 1.2x (16%) |
| blas3 | trmm | f64 | 512x512x512 | 9.174 ms | 14.63 | 0.57 | 2% | ok | 1.0x (13%) |
| blas3 | trsm | f64 | 512x512x512 | 9.217 ms | 14.56 | 0.57 | 2% | ok | 1.1x (13%) |
| blas3 | trsm | f32 | 512x512x512 | 5.112 ms | 26.26 | 0.51 | 2% | ok | 1.0x (13%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.017 ms | 0.41 | 3.26 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 7.324 ms | 12.22 | 0.57 | 2% | ok | 1.6x (20%) |
| factor | lu | f32 | 512x512 | 3.675 ms | 24.35 | 0.57 | 2% | ok | 1.2x (15%) |
| factor | cholesky | f64 | 512x512 | 7.184 ms | 6.23 | 0.58 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 6.463 ms | 6.92 | 0.32 | 1% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 7.156 ms | 6.25 | 0.59 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 4.550 ms | 9.83 | 0.92 | 2% | ok | 1.1x (13%) |
| factor | qr | f64 | 512x512 | 130.107 ms | 1.38 | 0.03 | 0% | ok | 1.8x (23%) |
| factor | lq | f64 | 512x512 | 16.481 ms | 10.86 | 0.25 | 2% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 157.208 ms | 2.85 | 0.03 | 0% | ok | 1.8x (22%) |
| factor | householder | f64 | 512x512 | 0.415 ms | 2.53 | 10.10 | 12%\* | ok | 4.0x (49%) |
| factor | inv | f64 | 512x512 | 48.814 ms | 5.50 | 0.09 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 5.588 ms | 16.01 | 0.75 | 3% | ok | 1.1x (13%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 17.799 ms | 1.26 | 0.06 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 58.164 ms | 2.60 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 736.302 ms | 0.21 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 32.335 ms | 5.19 | 0.03 | 1% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 80.721 ms | 5.20 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 109.353 ms | 3.38 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 10.766 ms | 23.38 | 0.10 | 4% | ok | 2.4x (30%) |
| eigen | sylvester | f64 | 256x256 | 178.212 ms | 5.65 | 0.01 | 1% | ok | 1.1x (13%) |
| elementwise | exp | f64 | n=8388608 | 39.080 ms | 0.21 | 3.43 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 45.635 ms | 0.18 | 2.94 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 59.869 ms | 0.14 | 2.24 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 29.368 ms | 0.29 | 4.57 | 5% | ok | 1.0x (12%) |
| elementwise | erf | f64 | n=8388608 | 205.761 ms | 0.04 | 0.65 | 1% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.113 ms | 23.13 | 240.60 | 287% ! | ok | 6.6x (82%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.918 ms | 2.85 | 29.67 | 35% | ok | 1.0x (12%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.316 ms | 31.82 | 66.90 | 80% | ok | 3.5x (44%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 5.066 ms | 0.64 | 2.27 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.521 ms | 1.32 | 10.94 | 13%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 14.645 ms | 0.76 | 0.05 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 24.644 ms | 0.45 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.450 ms | 1.32 | 0.09 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.794 ms | 0.56 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.841 ms | 0.50 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 20.802 ms | 1.08 | 0.04 | 0% | ok | 0.8x (10%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 24.716 ms | 0.08 | 0.01 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.295 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.868 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 57.102 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.182 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.939 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.052 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 12.445 ms | 0.90 | 0.06 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.252 ms | 3.00 | 11.99 | 14%\* | ok | 1.0x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 3.976 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.037 ms | 2.77 | 0.18 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.257 ms | 2.94 | 11.77 | 14%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.065 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.829 ms | 0.62 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.630 ms | 1.20 | 4.80 | 6%\* | ok | 1.0x (12%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.402 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.765 ms | 0.58 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.623 ms | 1.21 | 4.86 | 6%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.906 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 197.101 ms | 1.57 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.646 ms | 2.05 | 8.19 | 10%\* | ok | 0.9x (11%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.467 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 59.887 ms | 5.15 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.671 ms | 2.02 | 8.07 | 10%\* | ok | 0.9x (11%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.781 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 541.245 ms | 1.14 | 0.00 | 0% | ok | 1.1x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.799 ms | 0.43 | 1.73 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.091 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 454.910 ms | 1.36 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.743 ms | 0.44 | 1.74 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.642 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 46.884 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.506 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.358 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.764 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.865 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.130 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.910 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.970 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 17.821 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.226 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 15.857 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 18.648 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.039 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.504 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 11.369 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.998 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.022 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.688 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.978 ms | 2.68 | 30.00 | 36% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.199 ms | 2.18 | 24.46 | 29% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.556 ms | 1.68 | 18.85 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.384 ms | 2.87 | 21.72 | 26% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.354 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 30.067 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 32.981 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.872 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 4.808 ms | 3.49 | 13.96 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 5.122 ms | 52.41 | 0.82 | 9% | ok | 7.6x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.532 ms | 31.52 | 33.49 | 40%\* | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.307 ms | 7.27 | 29.09 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.986 ms | 17.01 | 34.03 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.685 ms | 9.96 | 9.96 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.692 ms | 9.91 | 9.91 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.378 ms | 112.89 | 1.32 | n/a | ok | 7.4x (92%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.747 ms | 97.72 | 0.76 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.354 ms | 198.27 | 1.16 | n/a | ok | 7.6x (94%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.354 ms | 198.24 | 1.16 | n/a | ok | 7.6x (94%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.543 ms | 75.76 | 0.44 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

## i7-12700k -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.5 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.5 GFLOP/s |
| ... this build, AVX2 | f64 78.3, f32 156.5 GFLOP/s |
| bandwidth | read 37.8, update 41.5, triad 31.4 GB/s (ceiling 41.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 88% | 103% | 0 |
| blas2 | 10 | 93% | 96% | 0 |
| blas3 | 10 | 67% | 90% | 0 |
| factor | 12 | 8% | 23% | 0 |
| eigen | 8 | 7% | 28% | 0 |
| elementwise | 5 | 7% | 11% | 0 |
| sparse | 5 | 49% | 89% | 0 |
| direct | 19 | 1% | 29% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 4% | 22% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 54% | 72% | 0 |
| mg | 1 | 49% | 49% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 84% | 132% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 3.690 ms | 4.55 | 36.37 | 88% | ok |
| blas1 | dot | f32 | n=8388608 | 1.704 ms | 9.84 | 39.38 | 95% | ok |
| blas1 | axpy | f64 | n=8388608 | 4.795 ms | 3.50 | 41.99 | 101% | ok |
| blas1 | axpy | f32 | n=8388608 | 2.348 ms | 7.14 | 42.87 | 103% | ok |
| blas1 | scale | f64 | n=8388608 | 3.463 ms | 2.42 | 38.76 | 93% | ok |
| blas1 | two_norm | f64 | n=8388608 | 2.214 ms | 7.58 | 30.30 | 73% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.036 ms | 2.08 | 16.63 | 40% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.039 ms | 1.19 | 9.53 | 23% | ok |
| blas1 | sum | f64 | n=8388608 | 3.967 ms | 2.11 | 16.92 | 41% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 71.017 ms | 2.60 | 1.89 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.410 ms | 9.84 | 39.38 | 95% | ok |
| blas2 | gemv | f32 | n=4096 | 1.689 ms | 19.87 | 39.75 | 96% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.471 ms | 9.67 | 38.69 | 93% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.680 ms | 19.98 | 39.97 | 96% | ok |
| blas2 | symv | f64 | n=4096 | 2.942 ms | 11.41 | 22.84 | 55% | ok |
| blas2 | trmv | f64 | n=4096 | 1.793 ms | 9.36 | 37.47 | 90% | ok |
| blas2 | trmv | f32 | n=4096 | 0.868 ms | 19.32 | 38.68 | 93% | ok |
| blas2 | trsv | f64 | n=4096 | 1.818 ms | 9.23 | 36.95 | 89% | ok |
| blas2 | trsv | f32 | n=4096 | 0.878 ms | 19.12 | 38.28 | 92% | ok |
| blas2 | ger | f64 | n=4096 | 7.161 ms | 4.69 | 37.49 | 90% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.803 ms | 70.59 | 1.65 | 90% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.898 ms | 141.43 | 1.66 | 90% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.624 ms | 51.14 | 2.40 | 65% | ok |
| blas3 | symm | f64 | 512x512x512 | 4.042 ms | 66.41 | 1.56 | 85% | ok |
| blas3 | syrk | f64 | 512x512x512 | 2.573 ms | 52.16 | 1.63 | 67% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 5.020 ms | 53.47 | 1.25 | 68% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.693 ms | 49.84 | 1.95 | 64% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.775 ms | 48.37 | 1.89 | 62% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.411 ms | 95.15 | 1.86 | 61% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.152 ms | 0.40 | 3.23 | 8% | ok |
| factor | lu | f64 | 512x512 | 11.040 ms | 8.10 | 0.38 | 10% | ok |
| factor | lu | f32 | 512x512 | 3.288 ms | 27.22 | 0.64 | 17% | ok |
| factor | cholesky | f64 | 512x512 | 7.088 ms | 6.31 | 0.59 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.568 ms | 6.81 | 0.32 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.220 ms | 6.20 | 0.58 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 4.413 ms | 10.14 | 0.95 | 13% | ok |
| factor | qr | f64 | 512x512 | 228.000 ms | 0.78 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 23.240 ms | 7.70 | 0.18 | 10% | ok |
| factor | hessenberg | f64 | 512x512 | 261.854 ms | 1.71 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.630 ms | 0.64 | 2.57 | 6%\* | ok |
| factor | inv | f64 | 512x512 | 55.026 ms | 4.88 | 0.08 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 4.882 ms | 18.33 | 0.86 | 23% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 16.850 ms | 1.33 | 0.06 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 56.752 ms | 2.66 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 749.812 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 30.672 ms | 5.47 | 0.03 | 7% | ok |
| eigen | schur | f64 | 256x256 | 79.932 ms | 5.25 | 0.01 | 7% | ok |
| eigen | svd | f64 | 256x256 | 95.131 ms | 3.88 | 0.01 | 5% | ok |
| eigen | expm | f64 | 256x256 | 11.624 ms | 21.65 | 0.09 | 28% | ok |
| eigen | sylvester | f64 | 256x256 | 176.795 ms | 5.69 | 0.01 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 39.508 ms | 0.21 | 3.40 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 45.718 ms | 0.18 | 2.94 | 7% | ok |
| elementwise | sin | f64 | n=8388608 | 60.050 ms | 0.14 | 2.24 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 29.117 ms | 0.29 | 4.61 | 11% | ok |
| elementwise | erf | f64 | n=8388608 | 202.424 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.738 ms | 3.55 | 36.91 | 89% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.913 ms | 2.87 | 29.83 | 72% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.324 ms | 9.68 | 20.36 | 49% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.866 ms | 0.67 | 2.36 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.311 ms | 1.53 | 12.70 | 31%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 15.256 ms | 0.73 | 0.05 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 27.034 ms | 0.41 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.904 ms | 1.26 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.283 ms | 0.56 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.035 ms | 0.51 | 0.02 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.744 ms | 1.34 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 25.733 ms | 0.07 | 0.01 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.460 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.030 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 58.605 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.193 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.591 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.158 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 13.490 ms | 0.83 | 0.05 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.252 ms | 3.00 | 12.00 | 29%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.126 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.481 ms | 2.50 | 0.16 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.272 ms | 2.78 | 11.13 | 27%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.118 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.931 ms | 0.62 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.646 ms | 1.17 | 4.68 | 11%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.485 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.129 ms | 0.59 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.650 ms | 1.16 | 4.65 | 11%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.932 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 206.638 ms | 1.49 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.480 ms | 2.28 | 9.10 | 22%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.641 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 67.636 ms | 4.56 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.695 ms | 1.99 | 7.95 | 19%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.960 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 528.105 ms | 1.17 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.547 ms | 0.45 | 1.79 | 4%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.348 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 694.318 ms | 0.89 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.558 ms | 0.45 | 1.78 | 4%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.218 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 40.782 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 21.030 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.446 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.745 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.537 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 19.906 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.783 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.474 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.344 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 14.068 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 13.198 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 14.492 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.034 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.596 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.046 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.083 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.027 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.025 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.704 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.984 ms | 2.66 | 29.82 | 72% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.311 ms | 2.00 | 22.37 | 54% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.511 ms | 1.73 | 19.41 | 47% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.334 ms | 2.68 | 20.28 | 49% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 28.548 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 29.312 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.676 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.594 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 1.924 ms | 8.72 | 34.87 | 84% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 6.001 ms | 44.73 | 0.70 | 57% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.325 ms | 51.59 | 54.82 | 132%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.912 ms | 8.78 | 35.11 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.760 ms | 22.08 | 44.16 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.293 ms | 57.21 | 57.21 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.280 ms | 59.87 | 59.87 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 3.955 ms | 67.87 | 0.80 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.578 ms | 48.12 | 0.38 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.556 ms | 48.31 | 0.28 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 2.098 ms | 127.97 | 0.75 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 1.877 ms | 143.00 | 0.84 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## i7-12700k -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 600.6 GFLOP/s, f32 1201.4 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.4, f32 1201.2 GFLOP/s |
| ... this build, AVX2 | f64 600.6, f32 1201.4 GFLOP/s |
| bandwidth | read 81.6, update 78.7, triad 76.9 GB/s (ceiling 81.6) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 93% | 150% | 0 | 23% |
| blas2 | 10 | 49% | 149% | 0 | 13% |
| blas3 | 10 | 18% | 66% | 0 | 27% |
| factor | 12 | 2% | 13% | 0 | 14% |
| eigen | 8 | 1% | 4% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 304% | 0 | 13% |
| direct | 19 | 0% | 14% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 10% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 28% | 37% | 0 | 13% |
| mg | 1 | 24% | 24% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 49% | 71% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 76% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.400 ms | 11.98 | 95.88 | 117% | ok | 2.6x (33%) |
| blas1 | dot | f32 | n=8388608 | 0.547 ms | 30.69 | 122.78 | 150% ! | ok | 3.1x (39%) |
| blas1 | axpy | f64 | n=8388608 | 2.664 ms | 6.30 | 75.57 | 93% | ok | 1.8x (22%) |
| blas1 | axpy | f32 | n=8388608 | 1.250 ms | 13.42 | 80.50 | 99% | ok | 1.9x (23%) |
| blas1 | scale | f64 | n=8388608 | 1.763 ms | 4.76 | 76.13 | 93% | ok | 2.0x (25%) |
| blas1 | two_norm | f64 | n=8388608 | 0.563 ms | 29.79 | 119.17 | 146% | ok | 3.9x (49%) |
| blas1 | one_norm | f64 | n=8388608 | 3.962 ms | 2.12 | 16.94 | 21% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.951 ms | 1.21 | 9.65 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.926 ms | 2.14 | 17.09 | 21% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 69.893 ms | 2.64 | 1.92 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.403 ms | 23.92 | 95.74 | 117% | ok | 2.4x (30%) |
| blas2 | gemv | f32 | n=4096 | 0.552 ms | 60.74 | 121.54 | 149% | ok | 3.1x (38%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.441 ms | 23.28 | 93.18 | 114% | ok | 2.4x (30%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.674 ms | 49.80 | 99.66 | 122% | ok | 2.5x (31%) |
| blas2 | symv | f64 | n=4096 | 2.868 ms | 11.70 | 23.42 | 29% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 1.761 ms | 9.53 | 38.15 | 47% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 0.841 ms | 19.95 | 39.94 | 49% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 1.793 ms | 9.36 | 37.46 | 46% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.855 ms | 19.62 | 39.28 | 48% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.122 ms | 4.71 | 37.70 | 46% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 0.674 ms | 398.10 | 9.33 | 66% | ok | 5.6x (70%) |
| blas3 | gemm | f32 | 512x512x512 | 0.350 ms | 767.30 | 8.99 | 64% | ok | 5.4x (68%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.231 ms | 109.03 | 5.11 | 18% | ok | 2.1x (27%) |
| blas3 | symm | f64 | 512x512x512 | 0.917 ms | 292.73 | 6.86 | 49% | ok | 4.4x (55%) |
| blas3 | syrk | f64 | 512x512x512 | 1.277 ms | 105.12 | 3.29 | 18% | ok | 2.0x (25%) |
| blas3 | syr2k | f64 | 512x512x512 | 2.305 ms | 116.45 | 2.73 | 19% | ok | 2.2x (27%) |
| blas3 | trmm | f64 | 512x512x512 | 1.800 ms | 74.55 | 2.91 | 12% | ok | 1.5x (19%) |
| blas3 | trsm | f64 | 512x512x512 | 1.854 ms | 72.40 | 2.83 | 12% | ok | 1.5x (19%) |
| blas3 | trsm | f32 | 512x512x512 | 0.945 ms | 141.98 | 2.77 | 12% | ok | 1.5x (19%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.941 ms | 0.41 | 3.28 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 6.309 ms | 14.18 | 0.66 | 2% | ok | 1.7x (22%) |
| factor | lu | f32 | 512x512 | 2.858 ms | 31.31 | 0.73 | 3% | ok | 1.2x (14%) |
| factor | cholesky | f64 | 512x512 | 6.888 ms | 6.50 | 0.61 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 6.391 ms | 7.00 | 0.33 | 1% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 7.010 ms | 6.38 | 0.60 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 4.311 ms | 10.38 | 0.97 | 2% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 130.161 ms | 1.37 | 0.03 | 0% | ok | 1.8x (22%) |
| factor | lq | f64 | 512x512 | 14.479 ms | 12.36 | 0.29 | 2% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 149.999 ms | 2.98 | 0.03 | 0% | ok | 1.7x (22%) |
| factor | householder | f64 | 512x512 | 0.402 ms | 2.61 | 10.42 | 13%\* | ok | 4.1x (51%) |
| factor | inv | f64 | 512x512 | 49.781 ms | 5.39 | 0.08 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 4.242 ms | 21.09 | 0.99 | 4% | ok | 1.2x (14%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 17.139 ms | 1.31 | 0.06 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 56.511 ms | 2.67 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 735.961 ms | 0.21 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 30.362 ms | 5.53 | 0.03 | 1% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 77.946 ms | 5.38 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 90.615 ms | 4.07 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 9.346 ms | 26.93 | 0.11 | 4% | ok | 1.2x (16%) |
| eigen | sylvester | f64 | 256x256 | 170.888 ms | 5.89 | 0.01 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 37.301 ms | 0.22 | 3.60 | 4% | ok | 1.1x (13%) |
| elementwise | log | f64 | n=8388608 | 44.831 ms | 0.19 | 2.99 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 59.702 ms | 0.14 | 2.25 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.935 ms | 0.29 | 4.64 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 201.190 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.110 ms | 23.85 | 248.15 | 304% ! | ok | 6.7x (84%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.921 ms | 2.84 | 29.55 | 36% | ok | 1.0x (12%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.307 ms | 32.03 | 67.35 | 82% | ok | 3.3x (41%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.853 ms | 0.67 | 2.37 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.537 ms | 1.31 | 10.83 | 13%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 15.008 ms | 0.75 | 0.05 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 26.577 ms | 0.42 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.915 ms | 1.25 | 0.08 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.088 ms | 0.56 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.025 ms | 0.51 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.567 ms | 1.35 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 24.636 ms | 0.08 | 0.01 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.357 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.907 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 57.266 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.188 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.397 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.073 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 12.429 ms | 0.90 | 0.06 | 0% | ok | 1.1x (14%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.257 ms | 2.95 | 11.79 | 14%\* | ok | 1.0x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.024 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.465 ms | 2.51 | 0.16 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.279 ms | 2.71 | 10.83 | 13%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.076 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.708 ms | 0.63 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.628 ms | 1.20 | 4.82 | 6%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.391 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.359 ms | 0.58 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.623 ms | 1.21 | 4.85 | 6%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.863 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 198.022 ms | 1.56 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.627 ms | 2.07 | 8.28 | 10%\* | ok | 0.9x (11%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.434 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 65.950 ms | 4.68 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.829 ms | 1.84 | 7.37 | 9%\* | ok | 0.9x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.823 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 513.208 ms | 1.20 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.801 ms | 0.43 | 1.73 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.303 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 697.752 ms | 0.88 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.696 ms | 0.44 | 1.75 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.131 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 40.511 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 20.891 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.408 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.682 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.475 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 19.844 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.660 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.384 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.108 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 13.714 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 12.986 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 14.358 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.034 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.543 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 11.854 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.062 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 1.006 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.024 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.708 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.965 ms | 2.71 | 30.38 | 37% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.287 ms | 2.03 | 22.78 | 28% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.484 ms | 1.76 | 19.76 | 24% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.669 ms | 2.62 | 19.81 | 24% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 27.899 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 28.342 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 32.714 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.513 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 1.880 ms | 8.93 | 35.70 | 44% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 0.911 ms | 294.58 | 4.60 | 49% | ok | 6.6x (82%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.308 ms | 54.54 | 57.95 | 71%\* | ok | 1.1x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.890 ms | 8.88 | 35.50 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.743 ms | 22.58 | 45.16 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.276 ms | 60.76 | 60.76 | n/a | ok | 1.1x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.263 ms | 63.89 | 63.89 | n/a | ok | 1.1x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 0.627 ms | 428.31 | 5.02 | n/a | ok | 6.3x (79%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 0.826 ms | 324.92 | 2.54 | n/a | ok | 6.8x (84%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 0.806 ms | 332.98 | 1.95 | n/a | ok | 6.9x (86%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.337 ms | 797.37 | 4.67 | n/a | ok | 6.2x (78%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.308 ms | 870.24 | 5.10 | n/a | ok | 6.1x (76%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

## i7-12700k -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.3 GFLOP/s |
| bandwidth | read 33.3, update 38.5, triad 32.6 GB/s (ceiling 38.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 43% | 98% | 0 |
| blas2 | 10 | 100% | 109% | 0 |
| blas3 | 10 | 81% | 94% | 0 |
| factor | 12 | 24% | 39% | 0 |
| eigen | 8 | 24% | 71% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 49% | 95% | 0 |
| direct | 31 | 4% | 32% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 21 | 24% | 69% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 62% | 77% | 0 |
| mg | 1 | 57% | 57% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 36% | 84% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.705 ms | 2.94 | 23.53 | 61% | ok |
| blas1 | dot | f32 | n=8388608 | 4.275 ms | 3.92 | 15.70 | 41% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.394 ms | 2.62 | 31.49 | 82% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.704 ms | 3.57 | 21.40 | 56% | ok |
| blas1 | scale | f64 | n=8388608 | 3.571 ms | 2.35 | 37.59 | 98% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.954 ms | 3.39 | 13.55 | 35% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.069 ms | 2.06 | 16.49 | 43% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.070 ms | 1.19 | 9.49 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 4.020 ms | 2.09 | 16.69 | 43% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 76.730 ms | 2.41 | 1.75 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.508 ms | 9.57 | 38.28 | 100% | ok |
| blas2 | gemv | f32 | n=4096 | 1.602 ms | 20.95 | 41.92 | 109% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.621 ms | 9.27 | 37.08 | 96% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.650 ms | 20.33 | 40.68 | 106% | ok |
| blas2 | symv | f64 | n=4096 | 1.922 ms | 17.46 | 34.96 | 91% | ok |
| blas2 | trmv | f64 | n=4096 | 1.859 ms | 9.02 | 36.13 | 94% | ok |
| blas2 | trmv | f32 | n=4096 | 0.946 ms | 17.74 | 35.52 | 92% | ok |
| blas2 | trsv | f64 | n=4096 | 1.796 ms | 9.34 | 37.41 | 97% | ok |
| blas2 | trsv | f32 | n=4096 | 0.872 ms | 19.23 | 38.50 | 100% | ok |
| blas2 | ger | f64 | n=4096 | 6.845 ms | 4.90 | 39.22 | 102% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.705 ms | 72.46 | 1.70 | 93% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.832 ms | 146.51 | 1.72 | 94% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.577 ms | 52.09 | 2.44 | 67% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.713 ms | 72.30 | 1.69 | 92% | ok |
| blas3 | syrk | f64 | 512x512x512 | 2.548 ms | 52.68 | 1.65 | 67% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 4.718 ms | 56.90 | 1.33 | 73% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.107 ms | 63.69 | 2.49 | 81% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.131 ms | 62.99 | 2.46 | 80% | ok |
| blas3 | trsm | f32 | 512x512x512 | 0.961 ms | 139.67 | 2.73 | 89% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.146 ms | 0.40 | 3.23 | 8% | ok |
| factor | lu | f64 | 512x512 | 3.280 ms | 27.28 | 1.28 | 35% | ok |
| factor | lu | f32 | 512x512 | 1.824 ms | 49.06 | 1.15 | 31% | ok |
| factor | cholesky | f64 | 512x512 | 2.052 ms | 21.80 | 2.04 | 28% | ok |
| factor | cholesky | f32 | 512x512 | 1.169 ms | 38.27 | 1.79 | 24% | ok |
| factor | ldlt | f64 | 512x512 | 7.506 ms | 5.96 | 0.56 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 4.952 ms | 9.03 | 0.85 | 12% | ok |
| factor | qr | f64 | 512x512 | 5.891 ms | 30.38 | 0.71 | 39% | ok |
| factor | lq | f64 | 512x512 | 25.490 ms | 7.02 | 0.16 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 269.980 ms | 1.66 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.626 ms | 0.65 | 2.58 | 7%\* | ok |
| factor | inv | f64 | 512x512 | 45.955 ms | 5.84 | 0.09 | 7% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 3.810 ms | 23.49 | 1.10 | 30% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.734 ms | 12.90 | 0.60 | 16% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 59.040 ms | 2.56 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 751.796 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 16.051 ms | 10.45 | 0.07 | 13% | ok |
| eigen | schur | f64 | 256x256 | 20.926 ms | 20.04 | 0.05 | 26% | ok |
| eigen | svd | f64 | 256x256 | 6.682 ms | 55.23 | 0.16 | 71% | ok |
| eigen | expm | f64 | 256x256 | 11.150 ms | 22.57 | 0.09 | 29% | ok |
| eigen | sylvester | f64 | 256x256 | 53.314 ms | 18.88 | 0.03 | 24% | ok |
| elementwise | exp | f64 | n=8388608 | 39.559 ms | 0.21 | 3.39 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 46.283 ms | 0.18 | 2.90 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 60.476 ms | 0.14 | 2.22 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 29.196 ms | 0.29 | 4.60 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 202.323 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.749 ms | 3.50 | 36.37 | 95% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.919 ms | 2.85 | 29.63 | 77% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.657 ms | 8.99 | 18.91 | 49% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.918 ms | 0.66 | 2.34 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.329 ms | 1.51 | 12.52 | 33%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 14.921 ms | 0.75 | 0.05 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 24.972 ms | 0.45 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.391 ms | 1.33 | 0.09 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.103 ms | 0.56 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.699 ms | 0.50 | 0.02 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.556 ms | 1.35 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 24.997 ms | 0.08 | 0.01 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.362 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.985 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 58.116 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.186 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.125 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.096 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 12.079 ms | 0.93 | 0.06 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.248 ms | 3.05 | 12.18 | 32%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.071 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.058 ms | 2.76 | 0.18 | 4% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.252 ms | 3.00 | 12.00 | 31%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.088 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 37.153 ms | 0.60 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.638 ms | 1.18 | 4.74 | 12%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.453 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.563 ms | 0.58 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.634 ms | 1.19 | 4.77 | 12%\* | ok |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 5.729 ms | 1.95 | 0.13 | 2% | ok |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 1.867 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 3.480 ms | 3.21 | 0.21 | 4% | ok |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 0.328 ms | 2.31 | 9.23 | 24%\* | ok |
| direct | ref_pardiso_spd | f64 | 96^2 grid, nnz 45696 | 4.532 ms | 2.47 | 0.16 | 3% | ok |
| direct | ref_pardiso_spd.analyze | f64 | 96^2 grid, nnz 45696 | 2.110 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_pardiso_spd.factor | f64 | 96^2 grid, nnz 45696 | 1.907 ms | 5.87 | 0.38 | 7% | ok |
| direct | ref_pardiso_spd.solve | f64 | 96^2 grid, nnz 45696 | 0.457 ms | 1.65 | 6.61 | 17%\* | ok |
| direct | ref_pardiso_lu | f64 | 96^2 grid, nnz 45696 | 7.446 ms | 3.00 | 0.10 | 4% | ok |
| direct | ref_pardiso_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.561 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_pardiso_lu.factor | f64 | 96^2 grid, nnz 45696 | 3.247 ms | 6.89 | 0.23 | 9% | ok |
| direct | ref_pardiso_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.573 ms | 1.32 | 5.28 | 14%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.942 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 204.533 ms | 1.51 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.450 ms | 2.32 | 9.30 | 24%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.662 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 60.040 ms | 5.14 | 0.01 | 7% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.456 ms | 2.31 | 9.25 | 24%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.877 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 553.107 ms | 1.12 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.521 ms | 0.45 | 1.79 | 5%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.262 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 451.616 ms | 1.37 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.525 ms | 0.45 | 1.79 | 5%\* | ok |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 3.883 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 11.337 ms | 27.22 | 0.08 | 35% | ok |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 0.507 ms | 6.65 | 26.60 | 69%\* | ok |
| direct3d | ref_pardiso_spd.analyze | f64 | 20^3 grid, nnz 53600 | 7.967 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_pardiso_spd.factor | f64 | 20^3 grid, nnz 53600 | 9.703 ms | 31.81 | 0.09 | 41% | ok |
| direct3d | ref_pardiso_spd.solve | f64 | 20^3 grid, nnz 53600 | 0.870 ms | 3.87 | 15.50 | 40%\* | ok |
| direct3d | ref_pardiso_lu.analyze | f64 | 20^3 grid, nnz 53600 | 9.050 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_pardiso_lu.factor | f64 | 20^3 grid, nnz 53600 | 19.178 ms | 32.18 | 0.04 | 41% | ok |
| direct3d | ref_pardiso_lu.solve | f64 | 20^3 grid, nnz 53600 | 1.037 ms | 3.25 | 13.00 | 34%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 31.359 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 48.204 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 23.051 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.633 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 15.177 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 14.181 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.737 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.453 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.381 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 18.253 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.724 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 16.197 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 19.230 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.041 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.537 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 11.671 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.091 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.026 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.023 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.089 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.988 ms | 2.65 | 29.68 | 77% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.221 ms | 2.14 | 24.01 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.564 ms | 1.67 | 18.75 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.286 ms | 2.89 | 21.88 | 57% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 30.662 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.198 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.373 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.978 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.880 ms | 3.44 | 13.75 | 36% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 38.993 ms | 6.88 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.555 ms | 30.25 | 32.14 | 84%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.300 ms | 7.29 | 29.18 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.999 ms | 16.79 | 33.59 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.725 ms | 9.73 | 9.73 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.724 ms | 9.73 | 9.73 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.492 ms | 15.35 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.821 ms | 12.89 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.229 ms | 26.24 | 0.15 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.233 ms | 26.23 | 0.15 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 26.885 ms | 9.98 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## i7-12700k -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 600.5 GFLOP/s, f32 1201.0 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.5, f32 1201.0 GFLOP/s |
| ... this build, scalar | f64 10.0, f32 9.9 GFLOP/s |
| bandwidth | read 81.9, update 77.4, triad 75.5 GB/s (ceiling 81.9) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 91% | 126% | 0 | 29% |
| blas2 | 10 | 119% | 198% | 0 | 32% |
| blas3 | 10 | 66% | 84% | 0 | 77% |
| factor | 12 | 5% | 12% | 0 | 18% |
| eigen | 8 | 3% | 10% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 295% | 0 | 13% |
| direct | 31 | 1% | 15% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 21 | 10% | 32% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 30% | 37% | 0 | 13% |
| mg | 1 | 27% | 27% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 40% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 92% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.397 ms | 12.01 | 96.07 | 117% | ok | 4.1x (51%) |
| blas1 | dot | f32 | n=8388608 | 0.652 ms | 25.73 | 102.94 | 126% | ok | 6.6x (82%) |
| blas1 | axpy | f64 | n=8388608 | 2.743 ms | 6.12 | 73.39 | 90% | ok | 2.3x (29%) |
| blas1 | axpy | f32 | n=8388608 | 1.354 ms | 12.39 | 74.36 | 91% | ok | 3.5x (43%) |
| blas1 | scale | f64 | n=8388608 | 1.776 ms | 4.72 | 75.59 | 92% | ok | 2.0x (25%) |
| blas1 | two_norm | f64 | n=8388608 | 0.739 ms | 22.69 | 90.75 | 111% | ok | 6.7x (84%) |
| blas1 | one_norm | f64 | n=8388608 | 3.970 ms | 2.11 | 16.91 | 21% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.943 ms | 1.21 | 9.67 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.939 ms | 2.13 | 17.04 | 21% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 75.709 ms | 2.44 | 1.77 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.446 ms | 23.21 | 92.88 | 113% | ok | 2.4x (30%) |
| blas2 | gemv | f32 | n=4096 | 0.569 ms | 59.00 | 118.05 | 144% | ok | 2.8x (35%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.428 ms | 23.50 | 94.06 | 115% | ok | 2.5x (32%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.687 ms | 48.87 | 97.78 | 119% | ok | 2.4x (30%) |
| blas2 | symv | f64 | n=4096 | 0.619 ms | 54.18 | 108.47 | 132% | ok | 3.1x (39%) |
| blas2 | trmv | f64 | n=4096 | 0.600 ms | 27.94 | 111.87 | 137% | ok | 3.1x (39%) |
| blas2 | trmv | f32 | n=4096 | 0.207 ms | 81.17 | 162.50 | 198% ! | ok | 4.6x (57%) |
| blas2 | trsv | f64 | n=4096 | 1.757 ms | 9.55 | 38.23 | 47% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.853 ms | 19.67 | 39.37 | 48% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 3.637 ms | 9.23 | 73.83 | 90% | ok | 1.9x (24%) |
| blas3 | gemm | f64 | 512x512x512 | 0.544 ms | 493.49 | 11.57 | 82% | ok | 6.8x (85%) |
| blas3 | gemm | f32 | 512x512x512 | 0.266 ms | 1009.32 | 11.83 | 84% | ok | 6.9x (86%) |
| blas3 | gemmt | f64 | 512x512x512 | 0.700 ms | 191.61 | 8.98 | 32% | ok | 3.7x (46%) |
| blas3 | symm | f64 | 512x512x512 | 0.549 ms | 488.63 | 11.45 | 81% | ok | 6.8x (84%) |
| blas3 | syrk | f64 | 512x512x512 | 0.751 ms | 178.71 | 5.58 | 30% | ok | 3.4x (42%) |
| blas3 | syr2k | f64 | 512x512x512 | 1.128 ms | 237.92 | 5.58 | 40% | ok | 4.2x (52%) |
| blas3 | trmm | f64 | 512x512x512 | 0.341 ms | 393.48 | 15.37 | 66% | ok | 6.2x (77%) |
| blas3 | trsm | f64 | 512x512x512 | 0.338 ms | 396.58 | 15.49 | 66% | ok | 6.3x (79%) |
| blas3 | trsm | f32 | 512x512x512 | 0.159 ms | 842.50 | 16.46 | 70% | ok | 6.0x (75%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.959 ms | 0.41 | 3.28 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 2.085 ms | 42.92 | 2.01 | 7% | ok | 1.6x (20%) |
| factor | lu | f32 | 512x512 | 1.298 ms | 68.91 | 1.62 | 6% | ok | 1.4x (18%) |
| factor | cholesky | f64 | 512x512 | 1.526 ms | 29.31 | 2.75 | 5% | ok | 1.3x (17%) |
| factor | cholesky | f32 | 512x512 | 0.994 ms | 45.01 | 2.11 | 4% | ok | 1.2x (15%) |
| factor | ldlt | f64 | 512x512 | 7.220 ms | 6.20 | 0.58 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 4.758 ms | 9.40 | 0.88 | 2% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 4.143 ms | 43.20 | 1.01 | 7% | ok | 1.4x (18%) |
| factor | lq | f64 | 512x512 | 16.541 ms | 10.82 | 0.25 | 2% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 155.944 ms | 2.87 | 0.03 | 0% | ok | 1.7x (22%) |
| factor | householder | f64 | 512x512 | 0.411 ms | 2.55 | 10.20 | 12%\* | ok | 4.0x (49%) |
| factor | inv | f64 | 512x512 | 46.313 ms | 5.80 | 0.09 | 1% | ok | 1.0x (12%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 2.780 ms | 32.18 | 1.51 | 5% | ok | 1.4x (17%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.680 ms | 13.31 | 0.62 | 2% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 57.327 ms | 2.63 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 735.440 ms | 0.21 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 17.426 ms | 9.63 | 0.06 | 2% | ok | 0.9x (12%) |
| eigen | schur | f64 | 256x256 | 21.751 ms | 19.28 | 0.05 | 3% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 6.186 ms | 59.66 | 0.17 | 10% | ok | 1.1x (14%) |
| eigen | expm | f64 | 256x256 | 8.688 ms | 28.97 | 0.12 | 5% | ok | 1.3x (16%) |
| eigen | sylvester | f64 | 256x256 | 55.425 ms | 18.16 | 0.03 | 3% | ok | 1.0x (12%) |
| elementwise | exp | f64 | n=8388608 | 39.078 ms | 0.21 | 3.43 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 45.629 ms | 0.18 | 2.94 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 59.419 ms | 0.14 | 2.26 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.634 ms | 0.29 | 4.69 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 200.465 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.113 ms | 23.23 | 241.70 | 295% ! | ok | 6.6x (83%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.914 ms | 2.87 | 29.81 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.312 ms | 31.92 | 67.12 | 82% | ok | 3.5x (44%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.803 ms | 0.68 | 2.39 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.538 ms | 1.30 | 10.82 | 13%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 14.873 ms | 0.75 | 0.05 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 24.688 ms | 0.45 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.449 ms | 1.32 | 0.09 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.996 ms | 0.56 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.608 ms | 0.50 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 20.197 ms | 1.11 | 0.04 | 0% | ok | 0.8x (10%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 24.592 ms | 0.08 | 0.01 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.289 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.871 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 57.327 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.183 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.819 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.049 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 11.817 ms | 0.95 | 0.06 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.254 ms | 2.98 | 11.92 | 15%\* | ok | 1.0x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 3.982 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.033 ms | 2.77 | 0.18 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.255 ms | 2.97 | 11.87 | 15%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.050 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 36.103 ms | 0.62 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.605 ms | 1.25 | 5.00 | 6%\* | ok | 1.1x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.374 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.631 ms | 0.58 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.608 ms | 1.24 | 4.97 | 6%\* | ok | 1.0x (13%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 5.978 ms | 1.87 | 0.12 | 0% | ok | 1.0x (12%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 1.829 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 3.642 ms | 3.07 | 0.20 | 1% | ok | 1.0x (12%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 0.331 ms | 2.29 | 9.14 | 11%\* | ok | 1.0x (12%) |
| direct | ref_pardiso_spd | f64 | 96^2 grid, nnz 45696 | 3.203 ms | 3.49 | 0.23 | 1% | ok | 1.4x (18%) |
| direct | ref_pardiso_spd.analyze | f64 | 96^2 grid, nnz 45696 | 1.994 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct | ref_pardiso_spd.factor | f64 | 96^2 grid, nnz 45696 | 0.634 ms | 17.65 | 1.15 | 3% | ok | 3.0x (38%) |
| direct | ref_pardiso_spd.solve | f64 | 96^2 grid, nnz 45696 | 0.511 ms | 1.48 | 5.92 | 7%\* | ok | 0.9x (11%) |
| direct | ref_pardiso_lu | f64 | 96^2 grid, nnz 45696 | 5.596 ms | 4.00 | 0.13 | 1% | ok | 1.3x (17%) |
| direct | ref_pardiso_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.537 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | ref_pardiso_lu.factor | f64 | 96^2 grid, nnz 45696 | 1.240 ms | 18.04 | 0.59 | 3% | ok | 2.6x (33%) |
| direct | ref_pardiso_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.648 ms | 1.17 | 4.67 | 6%\* | ok | 0.9x (11%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.911 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 194.038 ms | 1.59 | 0.00 | 0% | ok | 1.1x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.631 ms | 2.07 | 8.26 | 10%\* | ok | 0.9x (11%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.374 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 57.597 ms | 5.36 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.652 ms | 2.04 | 8.16 | 10%\* | ok | 0.9x (11%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.771 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 543.921 ms | 1.13 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.771 ms | 0.43 | 1.73 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.091 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 456.374 ms | 1.35 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.727 ms | 0.44 | 1.74 | 2%\* | ok | 1.0x (12%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 3.831 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 8.481 ms | 36.39 | 0.10 | 6% | ok | 1.3x (17%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 0.511 ms | 6.59 | 26.36 | 32%\* | ok | 1.0x (12%) |
| direct3d | ref_pardiso_spd.analyze | f64 | 20^3 grid, nnz 53600 | 7.062 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | ref_pardiso_spd.factor | f64 | 20^3 grid, nnz 53600 | 2.392 ms | 129.00 | 0.36 | 21% | ok | 4.1x (51%) |
| direct3d | ref_pardiso_spd.solve | f64 | 20^3 grid, nnz 53600 | 0.953 ms | 3.53 | 14.14 | 17%\* | ok | 0.9x (11%) |
| direct3d | ref_pardiso_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.132 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | ref_pardiso_lu.factor | f64 | 20^3 grid, nnz 53600 | 5.562 ms | 110.96 | 0.15 | 18% | ok | 3.4x (43%) |
| direct3d | ref_pardiso_lu.solve | f64 | 20^3 grid, nnz 53600 | 1.163 ms | 2.90 | 11.59 | 14%\* | ok | 0.9x (11%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.535 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 46.387 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.356 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.255 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.732 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.761 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.068 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.665 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.033 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 17.709 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.454 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 15.829 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 18.697 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.041 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.506 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 11.362 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.058 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.995 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.022 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.089 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.974 ms | 2.69 | 30.12 | 37% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.200 ms | 2.18 | 24.44 | 30% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.536 ms | 1.70 | 19.09 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.048 ms | 2.95 | 22.28 | 27% | ok | 1.0x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.616 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 31.021 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 32.976 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.861 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 4.810 ms | 3.49 | 13.95 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 5.126 ms | 52.37 | 0.82 | 9% | ok | 7.6x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.543 ms | 30.88 | 32.81 | 40%\* | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.279 ms | 7.36 | 29.44 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.978 ms | 17.16 | 34.31 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.685 ms | 9.96 | 9.96 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.683 ms | 9.97 | 9.97 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.379 ms | 112.83 | 1.32 | n/a | ok | 7.4x (92%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.747 ms | 97.73 | 0.76 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.354 ms | 198.27 | 1.16 | n/a | ok | 7.6x (94%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.354 ms | 198.25 | 1.16 | n/a | ok | 7.6x (94%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.540 ms | 75.82 | 0.44 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

