## arduino-ventuno-q -- native-fast, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON: -march=armv8.2-a+dotprod+fp16+rcpc -mtune=cortex-a78c]`, SIMD NEON_WITHOUT_AES |
| commit | `bf315e4604cf` |
| FMA peak (ceiling) | f64 67.5 GFLOP/s, f32 135.0 GFLOP/s |
| ... hardware, NEON | f64 67.4, f32 134.9 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 67.5, f32 135.0 GFLOP/s |
| bandwidth | read 23.0, update 24.5, triad 22.4 GB/s (ceiling 24.5) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 98% | 106% | 0 | 27% |
| blas2 | 10 | 58% | 90% | 0 | 26% |
| blas3 | 10 | 28% | 55% | 0 | 59% |
| factor | 12 | 3% | 40% | 0 | 35% |
| eigen | 8 | 4% | 13% | 0 | 29% |
| elementwise | 5 | 5% | 9% | 0 | 27% |
| sparse | 5 | 56% | 92% | 0 | 28% |
| direct | 19 | 0% | 6% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 29% |
| direct3d | 12 | 2% | 7% | 0 | 28% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 53% | 62% | 0 | 28% |
| mg | 1 | 37% | 37% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 74% | 76% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 96% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.549 ms | 3.02 | 24.19 | 99% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.806 ms | 5.98 | 23.91 | 98% | ok | 1.0x (26%) |
| blas1 | axpy | f64 | n=8388608 | 8.216 ms | 2.04 | 24.50 | 100% | ok | 0.9x (24%) |
| blas1 | axpy | f32 | n=8388608 | 4.332 ms | 3.87 | 23.24 | 95% | ok | 0.9x (23%) |
| blas1 | scale | f64 | n=8388608 | 5.181 ms | 1.62 | 25.90 | 106% | ok | 1.0x (26%) |
| blas1 | two_norm | f64 | n=8388608 | 2.633 ms | 6.37 | 25.48 | 104% | ok | 1.2x (30%) |
| blas1 | one_norm | f64 | n=8388608 | 7.291 ms | 1.15 | 9.20 | 38% | ok | 1.2x (31%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.795 ms | 0.53 | 4.25 | 17% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.661 ms | 1.10 | 8.76 | 36% | ok | 1.1x (27%) |
| blas1 | dot_k3 | f64 | n=8388608 | 164.683 ms | 1.12 | 0.82 | 3% | ok | 1.2x (30%) |
| blas2 | gemv | f64 | n=4096 | 6.699 ms | 5.01 | 20.04 | 82% | ok | 1.2x (31%) |
| blas2 | gemv | f32 | n=4096 | 3.060 ms | 10.97 | 21.94 | 90% | ok | 1.3x (33%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.686 ms | 4.37 | 17.47 | 71% | ok | 1.0x (26%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.709 ms | 7.13 | 14.26 | 58% | ok | 0.8x (21%) |
| blas2 | symv | f64 | n=4096 | 15.270 ms | 2.20 | 4.40 | 18% | ok | 1.1x (26%) |
| blas2 | trmv | f64 | n=4096 | 4.693 ms | 3.57 | 14.31 | 58% | ok | 1.0x (26%) |
| blas2 | trmv | f32 | n=4096 | 2.586 ms | 6.49 | 12.99 | 53% | ok | 1.0x (26%) |
| blas2 | trsv | f64 | n=4096 | 4.831 ms | 3.47 | 13.91 | 57% | ok | 1.0x (25%) |
| blas2 | trsv | f32 | n=4096 | 2.645 ms | 6.34 | 12.70 | 52% | ok | 1.0x (26%) |
| blas2 | ger | f64 | n=4096 | 14.461 ms | 2.32 | 18.57 | 76% | ok | 1.1x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 7.430 ms | 36.13 | 0.85 | 54% | ok | 3.8x (95%) |
| blas3 | gemm | f32 | 512x512x512 | 3.589 ms | 74.79 | 0.88 | 55% | ok | 3.8x (96%) |
| blas3 | gemmt | f64 | 512x512x512 | 5.951 ms | 22.56 | 1.06 | 33% | ok | 3.0x (76%) |
| blas3 | symm | f64 | 512x512x512 | 8.135 ms | 33.00 | 0.77 | 49% | ok | 3.6x (90%) |
| blas3 | syrk | f64 | 512x512x512 | 37.168 ms | 3.61 | 0.11 | 5% | ok | 1.1x (28%) |
| blas3 | syr2k | f64 | 512x512x512 | 81.419 ms | 3.30 | 0.08 | 5% | ok | 1.0x (26%) |
| blas3 | trmm | f64 | 512x512x512 | 7.655 ms | 17.53 | 0.68 | 26% | ok | 2.3x (58%) |
| blas3 | trsm | f64 | 512x512x512 | 8.221 ms | 16.33 | 0.64 | 24% | ok | 2.2x (55%) |
| blas3 | trsm | f32 | 512x512x512 | 3.604 ms | 37.24 | 0.73 | 28% | ok | 2.4x (59%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 25.242 ms | 0.21 | 1.68 | 7% | ok | 1.0x (26%) |
| factor | lu | f64 | 512x512 | 17.170 ms | 5.21 | 0.24 | 8% | ok | 1.4x (35%) |
| factor | lu | f32 | 512x512 | 8.986 ms | 9.96 | 0.23 | 7% | ok | 1.5x (36%) |
| factor | cholesky | f64 | 512x512 | 20.102 ms | 2.23 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 18.795 ms | 2.38 | 0.11 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 18.995 ms | 2.36 | 0.22 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 61.866 ms | 0.72 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 145.931 ms | 1.23 | 0.03 | 2% | ok | 1.7x (42%) |
| factor | lq | f64 | 512x512 | 35.448 ms | 5.05 | 0.12 | 7% | ok | 1.7x (43%) |
| factor | hessenberg | f64 | 512x512 | 198.763 ms | 2.25 | 0.02 | 3% | ok | 1.7x (44%) |
| factor | householder | f64 | 512x512 | 0.430 ms | 2.44 | 9.74 | 40% | ok | 3.5x (86%) |
| factor | inv | f64 | 512x512 | 127.041 ms | 2.11 | 0.03 | 3% | ok | 1.1x (29%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.146 ms | 7.37 | 0.35 | 11% | ok | 1.4x (35%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.640 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.764 ms | 1.74 | 0.01 | 3% | ok | 1.1x (29%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 528.176 ms | 0.29 | 0.00 | 0% | ok | 1.2x (30%) |
| eigen | eigenvalue | f64 | 256x256 | 60.584 ms | 2.77 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 135.279 ms | 3.10 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 204.974 ms | 1.80 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 29.048 ms | 8.66 | 0.04 | 13% | ok | 1.7x (42%) |
| eigen | sylvester | f64 | 256x256 | 315.028 ms | 3.20 | 0.00 | 5% | ok | 1.1x (29%) |
| elementwise | exp | f64 | n=8388608 | 79.608 ms | 0.11 | 1.69 | 7% | ok | 1.1x (27%) |
| elementwise | log | f64 | n=8388608 | 101.458 ms | 0.08 | 1.32 | 5% | ok | 1.0x (26%) |
| elementwise | sin | f64 | n=8388608 | 134.417 ms | 0.06 | 1.00 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 59.836 ms | 0.14 | 2.24 | 9% | ok | 1.2x (29%) |
| elementwise | erf | f64 | n=8388608 | 105.937 ms | 0.08 | 1.27 | 5% | ok | 1.1x (27%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.207 ms | 2.17 | 22.55 | 92% | ok | 1.4x (35%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.969 ms | 1.33 | 13.83 | 56% | ok | 1.1x (27%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.133 ms | 6.83 | 14.36 | 59% | ok | 2.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.892 ms | 0.27 | 0.97 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.896 ms | 0.23 | 1.87 | 8% | ok | 1.0x (25%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 164.714 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 53.684 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.525 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 95.054 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 126.742 ms | 0.18 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.375 ms | 0.57 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 183.722 ms | 0.01 | 0.00 | 0% | ok | 1.1x (28%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.740 ms | 0.00 | 0.00 | n/a | ok | 1.1x (29%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.138 ms | 0.00 | 0.00 | n/a | ok | 1.2x (30%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.633 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 39.007 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.497 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 157.345 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.332 ms | 0.32 | 1.30 | 5% | ok | 0.8x (20%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.789 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.369 ms | 1.08 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 2.199 ms | 0.34 | 1.38 | 6% | ok | 0.8x (19%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.937 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 79.883 ms | 0.28 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.653 ms | 0.13 | 0.54 | 2% | ok | 1.1x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.075 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 108.373 ms | 0.21 | 0.01 | 0% | ok | 1.1x (29%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.688 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 19.348 ms | 0.00 | 0.00 | n/a | ok | 1.2x (29%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 14979.374 ms | 0.02 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.873 ms | 0.43 | 1.71 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 41.982 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 186.582 ms | 1.65 | 0.00 | 2% | ok | 1.2x (29%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.646 ms | 0.44 | 1.76 | 7% | ok | 1.0x (25%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.586 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1535.724 ms | 0.40 | 0.00 | 1% | ok | 1.1x (28%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 27.272 ms | 0.12 | 0.49 | 2% | ok | 1.1x (26%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 26.129 ms | 0.00 | 0.00 | n/a | ok | 1.2x (29%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2290.915 ms | 0.27 | 0.00 | 0% | ok | 1.1x (27%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.059 ms | 0.12 | 0.48 | 2% | ok | 1.0x (25%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.100 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 97.382 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 49.206 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.209 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.383 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.535 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.238 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.599 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.241 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.960 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.404 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.487 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.119 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.102 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.453 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ilut | f64 | 128^2 grid | 27.997 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.795 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ildl | f64 | 128^2 grid | 2.717 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.065 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.591 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.933 ms | 1.35 | 15.17 | 62% | ok | 1.1x (27%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.268 ms | 1.15 | 12.93 | 53% | ok | 1.1x (28%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.655 ms | 0.99 | 11.05 | 45% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 31.992 ms | 1.20 | 9.09 | 37% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 211.234 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 245.743 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 135.125 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.809 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 3.694 ms | 4.54 | 18.16 | 74% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 10.698 ms | 25.09 | 0.39 | 37% | ok | 3.9x (97%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.962 ms | 17.44 | 18.53 | 76% | ok | 0.8x (20%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.810 ms | 5.97 | 23.88 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.369 ms | 12.25 | 24.50 | n/a | ok | 1.0x (25%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.677 ms | 24.78 | 24.78 | n/a | ok | 1.0x (26%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.361 ms | 12.33 | 12.33 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 4.478 ms | 59.94 | 0.70 | n/a | ok | 3.9x (97%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.303 ms | 62.39 | 0.49 | n/a | ok | 3.9x (97%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.195 ms | 63.99 | 0.37 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.961 ms | 279.39 | 1.64 | n/a | ok | 3.8x (96%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.324 ms | 80.75 | 0.47 | n/a | ok | 4.1x (102%) |

