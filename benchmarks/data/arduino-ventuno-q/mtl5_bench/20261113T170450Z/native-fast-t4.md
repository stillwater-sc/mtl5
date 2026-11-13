## arduino-ventuno-q -- native-fast, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON: -march=armv8.2-a+dotprod+fp16+rcpc -mtune=cortex-a78c]`, SIMD NEON_WITHOUT_AES |
| commit | `bf315e4604cf` |
| FMA peak (ceiling) | f64 67.7 GFLOP/s, f32 135.3 GFLOP/s |
| ... hardware, NEON | f64 67.7, f32 135.1 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 67.5, f32 135.3 GFLOP/s |
| bandwidth | read 22.9, update 24.5, triad 22.3 GB/s (ceiling 24.5) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 98% | 116% | 0 | 27% |
| blas2 | 10 | 59% | 88% | 0 | 26% |
| blas3 | 10 | 27% | 55% | 0 | 59% |
| factor | 12 | 3% | 40% | 0 | 36% |
| eigen | 8 | 4% | 10% | 0 | 28% |
| elementwise | 5 | 6% | 8% | 0 | 27% |
| sparse | 5 | 54% | 88% | 0 | 28% |
| direct | 19 | 0% | 6% | 0 | 28% |
| ordering | 4 | n/a | n/a | 0 | 28% |
| direct3d | 12 | 2% | 7% | 0 | 28% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 53% | 62% | 0 | 28% |
| mg | 1 | 37% | 37% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 74% | 98% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 96% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.556 ms | 3.02 | 24.16 | 98% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.799 ms | 5.99 | 23.98 | 98% | ok | 1.0x (26%) |
| blas1 | axpy | f64 | n=8388608 | 7.945 ms | 2.11 | 25.34 | 103% | ok | 1.0x (25%) |
| blas1 | axpy | f32 | n=8388608 | 3.990 ms | 4.20 | 25.23 | 103% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 4.710 ms | 1.78 | 28.50 | 116% | ok | 1.1x (28%) |
| blas1 | two_norm | f64 | n=8388608 | 2.650 ms | 6.33 | 25.33 | 103% | ok | 1.2x (30%) |
| blas1 | one_norm | f64 | n=8388608 | 7.895 ms | 1.06 | 8.50 | 35% | ok | 1.1x (27%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.511 ms | 0.54 | 4.33 | 18% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.924 ms | 1.06 | 8.47 | 35% | ok | 1.0x (25%) |
| blas1 | dot_k3 | f64 | n=8388608 | 176.227 ms | 1.05 | 0.76 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 6.746 ms | 4.97 | 19.91 | 81% | ok | 1.2x (31%) |
| blas2 | gemv | f32 | n=4096 | 3.095 ms | 10.84 | 21.70 | 88% | ok | 1.3x (32%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.792 ms | 4.31 | 17.23 | 70% | ok | 1.0x (25%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.613 ms | 7.27 | 14.55 | 59% | ok | 0.9x (21%) |
| blas2 | symv | f64 | n=4096 | 15.580 ms | 2.15 | 4.31 | 18% | ok | 1.1x (27%) |
| blas2 | trmv | f64 | n=4096 | 4.833 ms | 3.47 | 13.90 | 57% | ok | 1.0x (25%) |
| blas2 | trmv | f32 | n=4096 | 2.533 ms | 6.62 | 13.26 | 54% | ok | 1.1x (27%) |
| blas2 | trsv | f64 | n=4096 | 4.880 ms | 3.44 | 13.76 | 56% | ok | 1.0x (24%) |
| blas2 | trsv | f32 | n=4096 | 2.703 ms | 6.21 | 12.43 | 51% | ok | 1.0x (26%) |
| blas2 | ger | f64 | n=4096 | 14.700 ms | 2.28 | 18.27 | 74% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 7.327 ms | 36.63 | 0.86 | 54% | ok | 3.8x (96%) |
| blas3 | gemm | f32 | 512x512x512 | 3.582 ms | 74.94 | 0.88 | 55% | ok | 3.9x (96%) |
| blas3 | gemmt | f64 | 512x512x512 | 5.899 ms | 22.75 | 1.07 | 34% | ok | 3.1x (77%) |
| blas3 | symm | f64 | 512x512x512 | 8.088 ms | 33.19 | 0.78 | 49% | ok | 3.6x (90%) |
| blas3 | syrk | f64 | 512x512x512 | 36.823 ms | 3.64 | 0.11 | 5% | ok | 1.1x (28%) |
| blas3 | syr2k | f64 | 512x512x512 | 77.421 ms | 3.47 | 0.08 | 5% | ok | 1.1x (26%) |
| blas3 | trmm | f64 | 512x512x512 | 7.674 ms | 17.49 | 0.68 | 26% | ok | 2.3x (57%) |
| blas3 | trsm | f64 | 512x512x512 | 7.935 ms | 16.91 | 0.66 | 25% | ok | 2.3x (57%) |
| blas3 | trsm | f32 | 512x512x512 | 3.616 ms | 37.12 | 0.73 | 27% | ok | 2.4x (59%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 23.543 ms | 0.23 | 1.80 | 7% | ok | 1.2x (30%) |
| factor | lu | f64 | 512x512 | 17.032 ms | 5.25 | 0.25 | 8% | ok | 1.4x (36%) |
| factor | lu | f32 | 512x512 | 8.765 ms | 10.21 | 0.24 | 8% | ok | 1.5x (37%) |
| factor | cholesky | f64 | 512x512 | 19.871 ms | 2.25 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 18.811 ms | 2.38 | 0.11 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 18.888 ms | 2.37 | 0.22 | 3% | ok | 1.1x (29%) |
| factor | ldlt_bk | f64 | 512x512 | 61.213 ms | 0.73 | 0.07 | 1% | ok | 1.1x (28%) |
| factor | qr | f64 | 512x512 | 145.794 ms | 1.23 | 0.03 | 2% | ok | 1.7x (42%) |
| factor | lq | f64 | 512x512 | 36.676 ms | 4.88 | 0.11 | 7% | ok | 1.7x (41%) |
| factor | hessenberg | f64 | 512x512 | 199.564 ms | 2.24 | 0.02 | 3% | ok | 1.7x (43%) |
| factor | householder | f64 | 512x512 | 0.425 ms | 2.47 | 9.87 | 40% | ok | 3.6x (90%) |
| factor | inv | f64 | 512x512 | 128.344 ms | 2.09 | 0.03 | 3% | ok | 1.1x (28%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.429 ms | 7.20 | 0.34 | 11% | ok | 1.4x (34%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.631 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 86.817 ms | 1.74 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 528.588 ms | 0.29 | 0.00 | 0% | ok | 1.2x (29%) |
| eigen | eigenvalue | f64 | 256x256 | 60.615 ms | 2.77 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 135.354 ms | 3.10 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.437 ms | 1.80 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 36.019 ms | 6.99 | 0.03 | 10% | ok | 1.3x (33%) |
| eigen | sylvester | f64 | 256x256 | 317.116 ms | 3.17 | 0.00 | 5% | ok | 1.1x (28%) |
| elementwise | exp | f64 | n=8388608 | 81.320 ms | 0.10 | 1.65 | 7% | ok | 1.1x (26%) |
| elementwise | log | f64 | n=8388608 | 99.079 ms | 0.08 | 1.35 | 6% | ok | 1.1x (27%) |
| elementwise | sin | f64 | n=8388608 | 134.042 ms | 0.06 | 1.00 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 65.749 ms | 0.13 | 2.04 | 8% | ok | 1.1x (28%) |
| elementwise | erf | f64 | n=8388608 | 106.840 ms | 0.08 | 1.26 | 5% | ok | 1.0x (25%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.259 ms | 2.08 | 21.62 | 88% | ok | 1.3x (33%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.060 ms | 1.27 | 13.22 | 54% | ok | 1.1x (26%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.110 ms | 6.85 | 14.41 | 59% | ok | 2.0x (50%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 11.880 ms | 0.27 | 0.97 | 4% | ok | 1.1x (28%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.874 ms | 0.23 | 1.88 | 8% | ok | 1.0x (25%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 164.612 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 53.761 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.523 ms | 0.46 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 92.935 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 126.636 ms | 0.18 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 39.108 ms | 0.57 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 181.933 ms | 0.01 | 0.00 | 0% | ok | 1.1x (28%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.725 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.252 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.628 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 38.037 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.399 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 156.535 ms | 0.07 | 0.00 | 0% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 2.073 ms | 0.36 | 1.46 | 6% | ok | 0.9x (22%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.636 ms | 0.00 | 0.00 | n/a | ok | 1.2x (29%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.387 ms | 1.08 | 0.07 | 2% | ok | 1.1x (28%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.999 ms | 0.38 | 1.51 | 6% | ok | 0.9x (21%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.945 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 79.003 ms | 0.28 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.656 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 9.087 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 108.017 ms | 0.21 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.677 ms | 0.13 | 0.53 | 2% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 19.338 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 14983.658 ms | 0.02 | 0.00 | 0% | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.871 ms | 0.43 | 1.71 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 42.122 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 186.431 ms | 1.66 | 0.00 | 2% | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.661 ms | 0.44 | 1.76 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.564 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1627.042 ms | 0.38 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.559 ms | 0.11 | 0.46 | 2% | ok | 1.0x (24%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 26.230 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 2286.520 ms | 0.27 | 0.00 | 0% | ok | 1.1x (27%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.652 ms | 0.12 | 0.47 | 2% | ok | 1.0x (25%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.135 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 96.905 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 48.914 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.190 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.358 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.648 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 56.315 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.459 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.194 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 33.955 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.403 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 35.657 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.103 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.104 ms | 0.00 | 0.00 | n/a | ok | 1.1x (26%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.511 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.951 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.807 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.696 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.064 ms | 0.00 | 0.00 | n/a | ok | 1.0x (25%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.608 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.932 ms | 1.35 | 15.18 | 62% | ok | 1.1x (28%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.252 ms | 1.16 | 13.02 | 53% | ok | 1.1x (28%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.707 ms | 0.97 | 10.83 | 44% | ok | 1.1x (27%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.120 ms | 1.20 | 9.05 | 37% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 210.879 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 244.121 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 133.082 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.717 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 3.693 ms | 4.54 | 18.17 | 74% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 10.564 ms | 25.41 | 0.40 | 38% | ok | 3.9x (99%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.742 ms | 22.61 | 24.02 | 98% | ok | 1.0x (26%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.788 ms | 6.02 | 24.07 | n/a | ok | 1.0x (25%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.373 ms | 12.22 | 24.44 | n/a | ok | 1.1x (29%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.693 ms | 24.21 | 24.21 | n/a | ok | 1.1x (27%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.374 ms | 12.21 | 12.21 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 4.495 ms | 59.72 | 0.70 | n/a | ok | 3.9x (97%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.304 ms | 62.37 | 0.49 | n/a | ok | 3.9x (99%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.192 ms | 64.04 | 0.38 | n/a | ok | 4.0x (100%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.961 ms | 279.41 | 1.64 | n/a | ok | 3.8x (96%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.324 ms | 80.75 | 0.47 | n/a | ok | 4.0x (99%) |

