## arduino-ventuno-q -- native-fast, 4 threads, class medium

| | |
|---|---|
| CPU | 0xd4b (aarch64, 8 logical cores) |
| OS | Linux Ubuntu 24.04.5 LTS (6.8.0-1089-qcom) |
| compiler | GCC 13.3.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON: -march=armv8.2-a+dotprod+fp16+rcpc -mtune=cortex-a78c]`, SIMD NEON_WITHOUT_AES |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 67.5 GFLOP/s, f32 135.7 GFLOP/s |
| ... hardware, NEON | f64 67.5, f32 134.9 GFLOP/s |
| ... this build, NEON_WITHOUT_AES | f64 67.5, f32 135.7 GFLOP/s |
| bandwidth | read 23.1, update 24.7, triad 22.7 GB/s (ceiling 24.7) |
| last-level cache | 0 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 98% | 115% | 0 | 28% |
| blas2 | 10 | 56% | 88% | 0 | 26% |
| blas3 | 10 | 37% | 55% | 0 | 79% |
| factor | 12 | 6% | 37% | 0 | 37% |
| eigen | 8 | 4% | 13% | 0 | 28% |
| elementwise | 5 | 5% | 8% | 0 | 28% |
| sparse | 5 | 55% | 92% | 0 | 27% |
| direct | 19 | 1% | 7% | 0 | 27% |
| ordering | 5 | n/a | n/a | 0 | 27% |
| direct3d | 12 | 2% | 7% | 0 | 27% |
| krylov | 13 | n/a | n/a | 0 | 28% |
| pc | 7 | n/a | n/a | 0 | 27% |
| smoother | 3 | 52% | 60% | 0 | 27% |
| mg | 1 | 37% | 37% | 0 | 25% |
| sparse_eig | 4 | n/a | n/a | 0 | 28% |
| mixed | 3 | 74% | 99% | 0 | 28% |
| integer | 9 | n/a | n/a | 0 | 55% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 5.535 ms | 3.03 | 24.25 | 98% | ok | 1.0x (26%) |
| blas1 | dot | f32 | n=8388608 | 2.794 ms | 6.00 | 24.01 | 97% | ok | 1.0x (26%) |
| blas1 | axpy | f64 | n=8388608 | 8.149 ms | 2.06 | 24.71 | 100% | ok | 1.0x (24%) |
| blas1 | axpy | f32 | n=8388608 | 3.957 ms | 4.24 | 25.44 | 103% | ok | 1.0x (25%) |
| blas1 | scale | f64 | n=8388608 | 4.740 ms | 1.77 | 28.31 | 115% | ok | 1.1x (28%) |
| blas1 | two_norm | f64 | n=8388608 | 2.619 ms | 6.41 | 25.63 | 104% | ok | 1.2x (30%) |
| blas1 | one_norm | f64 | n=8388608 | 7.460 ms | 1.12 | 9.00 | 36% | ok | 1.2x (29%) |
| blas1 | infinity_norm | f64 | n=8388608 | 15.559 ms | 0.54 | 4.31 | 17% | ok | 1.1x (28%) |
| blas1 | sum | f64 | n=8388608 | 7.470 ms | 1.12 | 8.98 | 36% | ok | 1.1x (29%) |
| blas1 | dot_k3 | f64 | n=8388608 | 179.851 ms | 1.03 | 0.75 | 3% | ok | 1.1x (28%) |
| blas2 | gemv | f64 | n=4096 | 6.728 ms | 4.99 | 19.96 | 81% | ok | 1.2x (31%) |
| blas2 | gemv | f32 | n=4096 | 3.070 ms | 10.93 | 21.87 | 88% | ok | 1.3x (33%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 7.730 ms | 4.34 | 17.37 | 70% | ok | 1.0x (25%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.844 ms | 6.93 | 13.86 | 56% | ok | 0.8x (21%) |
| blas2 | symv | f64 | n=4096 | 11.287 ms | 2.97 | 5.95 | 24% | ok | 1.0x (26%) |
| blas2 | trmv | f64 | n=4096 | 4.878 ms | 3.44 | 13.77 | 56% | ok | 1.0x (25%) |
| blas2 | trmv | f32 | n=4096 | 2.636 ms | 6.37 | 12.74 | 52% | ok | 1.0x (26%) |
| blas2 | trsv | f64 | n=4096 | 4.822 ms | 3.48 | 13.93 | 56% | ok | 1.0x (25%) |
| blas2 | trsv | f32 | n=4096 | 2.680 ms | 6.26 | 12.53 | 51% | ok | 1.0x (26%) |
| blas2 | ger | f64 | n=4096 | 14.715 ms | 2.28 | 18.25 | 74% | ok | 1.0x (26%) |
| blas3 | gemm | f64 | 512x512x512 | 7.415 ms | 36.20 | 0.85 | 54% | ok | 3.8x (96%) |
| blas3 | gemm | f32 | 512x512x512 | 3.577 ms | 75.04 | 0.88 | 55% | ok | 3.9x (96%) |
| blas3 | gemmt | f64 | 512x512x512 | 5.907 ms | 22.72 | 1.07 | 34% | ok | 3.1x (76%) |
| blas3 | symm | f64 | 512x512x512 | 8.209 ms | 32.70 | 0.77 | 48% | ok | 3.6x (89%) |
| blas3 | syrk | f64 | 512x512x512 | 5.307 ms | 25.29 | 0.79 | 37% | ok | 3.2x (79%) |
| blas3 | syr2k | f64 | 512x512x512 | 10.454 ms | 25.68 | 0.60 | 38% | ok | 3.2x (79%) |
| blas3 | trmm | f64 | 512x512x512 | 7.605 ms | 17.65 | 0.69 | 26% | ok | 2.3x (58%) |
| blas3 | trsm | f64 | 512x512x512 | 7.876 ms | 17.04 | 0.67 | 25% | ok | 2.3x (57%) |
| blas3 | trsm | f32 | 512x512x512 | 3.577 ms | 37.53 | 0.73 | 28% | ok | 2.4x (59%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 29.870 ms | 0.18 | 1.42 | 6% | ok | 1.1x (27%) |
| factor | lu | f64 | 512x512 | 17.027 ms | 5.26 | 0.25 | 8% | ok | 1.6x (39%) |
| factor | lu | f32 | 512x512 | 9.004 ms | 9.94 | 0.23 | 7% | ok | 1.5x (37%) |
| factor | cholesky | f64 | 512x512 | 19.896 ms | 2.25 | 0.21 | 3% | ok | 1.1x (28%) |
| factor | cholesky | f32 | 512x512 | 18.812 ms | 2.38 | 0.11 | 2% | ok | 1.1x (28%) |
| factor | ldlt | f64 | 512x512 | 18.989 ms | 2.36 | 0.22 | 3% | ok | 1.1x (28%) |
| factor | ldlt_bk | f64 | 512x512 | 10.302 ms | 4.34 | 0.41 | 6% | ok | 1.1x (27%) |
| factor | qr | f64 | 512x512 | 146.736 ms | 1.22 | 0.03 | 2% | ok | 1.7x (42%) |
| factor | lq | f64 | 512x512 | 36.356 ms | 4.92 | 0.12 | 7% | ok | 1.7x (42%) |
| factor | hessenberg | f64 | 512x512 | 199.533 ms | 2.24 | 0.02 | 3% | ok | 1.7x (43%) |
| factor | householder | f64 | 512x512 | 0.454 ms | 2.31 | 9.24 | 37% | ok | 3.4x (84%) |
| factor | inv | f64 | 512x512 | 130.176 ms | 2.06 | 0.03 | 3% | ok | 1.1x (28%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 12.221 ms | 7.32 | 0.34 | 11% | ok | 1.4x (35%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 35.643 ms | 0.63 | 0.03 | 1% | ok | 1.1x (28%) |
| eigen | eigen_symmetric | f64 | 256x256 | 87.667 ms | 1.72 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 533.780 ms | 0.28 | 0.00 | 0% | ok | 1.1x (28%) |
| eigen | eigenvalue | f64 | 256x256 | 60.897 ms | 2.76 | 0.02 | 4% | ok | 1.1x (28%) |
| eigen | schur | f64 | 256x256 | 135.622 ms | 3.09 | 0.01 | 5% | ok | 1.1x (28%) |
| eigen | svd | f64 | 256x256 | 205.968 ms | 1.79 | 0.01 | 3% | ok | 1.1x (28%) |
| eigen | expm | f64 | 256x256 | 28.804 ms | 8.74 | 0.04 | 13% | ok | 1.6x (41%) |
| eigen | sylvester | f64 | 256x256 | 316.511 ms | 3.18 | 0.00 | 5% | ok | 1.1x (28%) |
| elementwise | exp | f64 | n=8388608 | 81.857 ms | 0.10 | 1.64 | 7% | ok | 1.1x (28%) |
| elementwise | log | f64 | n=8388608 | 101.623 ms | 0.08 | 1.32 | 5% | ok | 1.1x (27%) |
| elementwise | sin | f64 | n=8388608 | 136.139 ms | 0.06 | 0.99 | 4% | ok | 1.1x (28%) |
| elementwise | sqrt | f64 | n=8388608 | 67.482 ms | 0.12 | 1.99 | 8% | ok | 1.1x (27%) |
| elementwise | erf | f64 | n=8388608 | 108.211 ms | 0.08 | 1.24 | 5% | ok | 1.1x (28%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 1.197 ms | 2.19 | 22.75 | 92% | ok | 1.4x (34%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.996 ms | 1.31 | 13.64 | 55% | ok | 1.1x (26%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.104 ms | 6.86 | 14.42 | 58% | ok | 2.0x (49%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 12.107 ms | 0.27 | 0.95 | 4% | ok | 1.1x (27%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 8.830 ms | 0.23 | 1.89 | 8% | ok | 1.0x (25%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 33.882 ms | 0.33 | 0.02 | 0% | ok | 1.1x (28%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 59.432 ms | 0.19 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 24.780 ms | 0.45 | 0.03 | 1% | ok | 1.1x (27%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 93.520 ms | 0.24 | 0.01 | 0% | ok | 1.1x (28%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 112.144 ms | 0.20 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 38.741 ms | 0.58 | 0.02 | 1% | ok | 1.1x (28%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 70.546 ms | 0.03 | 0.00 | 0% | ok | 1.1x (28%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 7.707 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 12.386 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 112.429 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.513 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 39.190 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 5.087 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 26.553 ms | 0.42 | 0.03 | 1% | ok | 1.1x (28%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 1.924 ms | 0.39 | 1.57 | 6% | ok | 1.0x (24%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 11.563 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 10.962 ms | 1.02 | 0.07 | 2% | ok | 1.1x (27%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 1.815 ms | 0.42 | 1.67 | 7% | ok | 1.0x (24%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.995 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 80.218 ms | 0.28 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.544 ms | 0.14 | 0.55 | 2% | ok | 1.0x (26%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 8.000 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 94.736 ms | 0.24 | 0.01 | 0% | ok | 1.1x (27%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 5.598 ms | 0.14 | 0.54 | 2% | ok | 1.0x (26%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 11.058 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 612.967 ms | 0.50 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 7.685 ms | 0.44 | 1.75 | 7% | ok | 1.0x (24%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 25.456 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 168.674 ms | 1.83 | 0.01 | 3% | ok | 1.1x (28%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 7.590 ms | 0.44 | 1.78 | 7% | ok | 1.0x (24%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.955 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1590.540 ms | 0.39 | 0.00 | 1% | ok | 1.1x (28%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 28.097 ms | 0.12 | 0.48 | 2% | ok | 1.0x (26%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 15.157 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 1133.938 ms | 0.54 | 0.00 | 1% | ok | 1.1x (27%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 29.262 ms | 0.12 | 0.46 | 2% | ok | 1.0x (26%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 64.236 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 148.136 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 49.045 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 26.508 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 31.574 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 30.653 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 55.205 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 57.657 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.399 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 35.518 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.581 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 31.672 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 37.510 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.104 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilu_0 | f64 | 128^2 grid | 4.533 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ilut | f64 | 128^2 grid | 27.696 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| pc | ic_0 | f64 | 128^2 grid | 2.841 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ildl | f64 | 128^2 grid | 2.745 ms | 0.00 | 0.00 | n/a | ok | 1.1x (27%) |
| pc | ssor | f64 | 128^2 grid | 0.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (26%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.570 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.966 ms | 1.33 | 14.92 | 60% | ok | 1.1x (26%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.280 ms | 1.15 | 12.86 | 52% | ok | 1.1x (27%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.665 ms | 0.98 | 11.00 | 45% | ok | 1.1x (28%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 32.059 ms | 1.20 | 9.07 | 37% | ok | 1.0x (25%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 208.665 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 208.767 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 129.223 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 11.762 ms | 0.00 | 0.00 | n/a | ok | 1.1x (28%) |
| mixed | dot | f32>f64 | n=8388608 | 3.667 ms | 4.58 | 18.30 | 74% | ok | 1.1x (28%) |
| mixed | gemm | f32>f64 | 512x512x512 | 10.494 ms | 25.58 | 0.40 | 38% | ok | 4.0x (99%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.729 ms | 23.00 | 24.44 | 99% | ok | 1.0x (26%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.787 ms | 6.02 | 24.08 | n/a | ok | 1.0x (26%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 1.378 ms | 12.17 | 24.35 | n/a | ok | 1.0x (26%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.683 ms | 24.55 | 24.55 | n/a | ok | 1.0x (26%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.362 ms | 12.32 | 12.32 | n/a | ok | 1.1x (28%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 4.470 ms | 60.05 | 0.70 | n/a | ok | 3.9x (97%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.250 ms | 63.16 | 0.49 | n/a | ok | 3.9x (98%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.191 ms | 64.06 | 0.38 | n/a | ok | 4.0x (99%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.961 ms | 279.30 | 1.64 | n/a | ok | 3.8x (96%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 6.004 ms | 44.71 | 0.26 | n/a | ok | 2.2x (55%) |

