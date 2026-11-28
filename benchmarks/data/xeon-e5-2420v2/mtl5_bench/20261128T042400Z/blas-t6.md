## xeon-e5-2420v2 -- blas, 6 threads, class large

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 59.1 GFLOP/s, f32 118.1 GFLOP/s |
| ... hardware, SSE2 | f64 59.1, f32 118.1 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.7 GFLOP/s |
| bandwidth | read 11.4, update 10.6, triad 8.4 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 53% | 107% | 0 | 85% |
| blas2 | 10 | 104% | 107% | 0 | 21% |
| blas3 | 10 | 125% | 163% | 0 | 71% |
| factor | 12 | 20% | 40% | 0 | 28% |
| eigen | 8 | 13% | 35% | 0 | 19% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 68% | 97% | 0 | 17% |
| direct | 19 | 1% | 15% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 3% | 14% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 69% | 75% | 0 | 17% |
| mg | 1 | 59% | 59% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 22% | 65% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 87% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 133.360 ms | 0.50 | 4.03 | 35% | ok | 5.6x (93%) |
| blas1 | dot | f32 | n=33554432 | 21.958 ms | 3.06 | 12.22 | 107% | ok | 5.1x (85%) |
| blas1 | axpy | f64 | n=33554432 | 155.196 ms | 0.43 | 5.19 | 46% | ok | 5.5x (91%) |
| blas1 | axpy | f32 | n=33554432 | 67.003 ms | 1.00 | 6.01 | 53% | ok | 5.5x (92%) |
| blas1 | scale | f64 | n=33554432 | 53.853 ms | 0.62 | 9.97 | 88% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=33554432 | 131.800 ms | 0.51 | 2.04 | 18% | ok | 5.6x (94%) |
| blas1 | one_norm | f64 | n=33554432 | 39.463 ms | 0.85 | 6.80 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=33554432 | 52.398 ms | 0.64 | 5.12 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=33554432 | 39.043 ms | 0.86 | 6.88 | 60% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=33554432 | 1118.429 ms | 0.66 | 0.48 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=8192 | 45.556 ms | 2.95 | 11.79 | 104% | ok | 1.3x (21%) |
| blas2 | gemv | f32 | n=8192 | 23.343 ms | 5.75 | 11.50 | 101% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 43.987 ms | 3.05 | 12.21 | 107% | ok | 1.3x (21%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 22.282 ms | 6.02 | 12.05 | 106% | ok | 1.3x (21%) |
| blas2 | symv | f64 | n=8192 | 22.566 ms | 5.95 | 11.90 | 105% | ok | 1.3x (22%) |
| blas2 | trmv | f64 | n=8192 | 22.609 ms | 2.97 | 11.88 | 104% | ok | 1.3x (22%) |
| blas2 | trmv | f32 | n=8192 | 11.540 ms | 5.82 | 11.64 | 102% | ok | 1.2x (20%) |
| blas2 | trsv | f64 | n=8192 | 29.890 ms | 2.25 | 8.99 | 79% | ok | 1.1x (18%) |
| blas2 | trsv | f32 | n=8192 | 13.544 ms | 4.96 | 9.91 | 87% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=8192 | 111.064 ms | 1.21 | 9.67 | 85% | ok | 1.1x (18%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 22.388 ms | 95.92 | 1.12 | 162% | ok | 4.9x (82%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 13.119 ms | 163.69 | 0.96 | 139% | ok | 5.2x (86%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 17.791 ms | 60.35 | 1.41 | 102% | ok | 3.7x (61%) |
| blas3 | symm | f64 | 1024x1024x1024 | 22.324 ms | 96.19 | 1.13 | 163% | ok | 5.0x (83%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 17.765 ms | 60.44 | 0.94 | 102% | ok | 3.4x (57%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 32.448 ms | 66.18 | 0.78 | 112% | ok | 3.6x (60%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 13.905 ms | 77.22 | 1.51 | 131% | ok | 4.1x (69%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 14.568 ms | 73.71 | 1.44 | 125% | ok | 4.2x (71%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 8.502 ms | 126.29 | 1.23 | 107% | ok | 4.6x (77%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 128.915 ms | 0.13 | 1.04 | 9% | ok | 1.0x (17%) |
| factor | lu | f64 | 1024x1024 | 33.816 ms | 21.17 | 0.50 | 36% | ok | 1.9x (32%) |
| factor | lu | f32 | 1024x1024 | 29.917 ms | 23.93 | 0.28 | 20% | ok | 1.7x (28%) |
| factor | cholesky | f64 | 1024x1024 | 29.263 ms | 12.23 | 0.57 | 21% | ok | 1.5x (25%) |
| factor | cholesky | f32 | 1024x1024 | 26.471 ms | 13.52 | 0.32 | 11% | ok | 1.4x (23%) |
| factor | ldlt | f64 | 1024x1024 | 211.083 ms | 1.70 | 0.08 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 1024x1024 | 106.009 ms | 3.38 | 0.16 | 6% | ok | 1.0x (17%) |
| factor | qr | f64 | 1024x1024 | 75.339 ms | 19.00 | 0.22 | 32% | ok | 1.8x (30%) |
| factor | lq | f64 | 1024x1024 | 148.688 ms | 9.63 | 0.11 | 16% | ok | 3.9x (65%) |
| factor | hessenberg | f64 | 1024x1024 | 1857.121 ms | 1.93 | 0.01 | 3% | ok | 4.2x (70%) |
| factor | householder | f64 | 1024x1024 | 3.709 ms | 1.13 | 4.52 | 40% | ok | 5.5x (92%) |
| factor | inv | f64 | 1024x1024 | 955.432 ms | 2.25 | 0.02 | 4% | ok | 1.0x (17%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 46.744 ms | 15.31 | 0.36 | 26% | ok | 1.6x (27%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 23.366 ms | 7.66 | 0.18 | 13% | ok | 1.5x (25%) |
| eigen | eigen_symmetric | f64 | 512x512 | 1849.128 ms | 0.65 | 0.00 | 1% | ok | 1.1x (19%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 14747.834 ms | 0.08 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 512x512 | 378.801 ms | 3.54 | 0.01 | 6% | ok | 1.1x (18%) |
| eigen | schur | f64 | 512x512 | 434.654 ms | 7.72 | 0.01 | 13% | ok | 1.1x (19%) |
| eigen | svd | f64 | 512x512 | 142.362 ms | 20.74 | 0.03 | 35% | ok | 1.2x (21%) |
| eigen | expm | f64 | 512x512 | 192.362 ms | 10.47 | 0.02 | 18% | ok | 1.4x (24%) |
| eigen | sylvester | f64 | 512x512 | 1689.886 ms | 4.77 | 0.00 | 8% | ok | 1.1x (18%) |
| elementwise | exp | f64 | n=33554432 | 621.747 ms | 0.05 | 0.86 | 8% | ok | 1.0x (16%) |
| elementwise | log | f64 | n=33554432 | 566.094 ms | 0.06 | 0.95 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=33554432 | 680.574 ms | 0.05 | 0.79 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=33554432 | 362.952 ms | 0.09 | 1.48 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=33554432 | 9885.219 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 9.855 ms | 1.06 | 11.06 | 97% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 12.759 ms | 0.82 | 8.54 | 75% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 45.521 ms | 3.68 | 7.74 | 68% | ok | 1.4x (24%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 99.759 ms | 0.13 | 0.46 | 4% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 38.466 ms | 0.26 | 2.18 | 19% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 305.098 ms | 0.35 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 554.110 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 156.410 ms | 0.68 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 905.821 ms | 0.24 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 885.243 ms | 0.24 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 384.213 ms | 0.56 | 0.01 | 1% | ok | 0.9x (15%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 1114.042 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 43.750 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 63.825 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 580.818 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 2.392 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 250.610 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 26.473 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 268.735 ms | 0.40 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 9.134 ms | 0.44 | 1.76 | 15% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 54.293 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 93.183 ms | 1.15 | 0.03 | 2% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 9.129 ms | 0.44 | 1.76 | 15% | ok | 1.0x (16%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 26.949 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 846.220 ms | 0.25 | 0.00 | 0% | ok | 1.0x (16%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.101 ms | 0.13 | 0.53 | 5% | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 40.852 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 800.958 ms | 0.27 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.204 ms | 0.13 | 0.53 | 5% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 23.596 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1951.766 ms | 0.50 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 19.426 ms | 0.39 | 1.54 | 14% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 61.600 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 607.434 ms | 1.60 | 0.00 | 3% | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 19.414 ms | 0.39 | 1.54 | 14% | ok | 1.0x (17%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 24.037 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 5722.368 ms | 0.34 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 87.688 ms | 0.09 | 0.34 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 33.959 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 4956.886 ms | 0.39 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 88.267 ms | 0.08 | 0.34 | 3% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 1747.658 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 3036.382 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 767.822 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 891.822 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 928.260 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 580.248 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 4924.591 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 1246.145 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 722.871 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 1392.768 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 1304.171 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 1503.215 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 2064.220 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.725 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| pc | ilu_0 | f64 | 256^2 grid | 17.900 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 256^2 grid | 148.838 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 256^2 grid | 12.029 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 256^2 grid | 11.232 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 256^2 grid | 0.364 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 14.078 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 14.985 ms | 0.70 | 7.83 | 69% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 13.673 ms | 0.77 | 8.58 | 75% | ok | 1.0x (16%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 15.994 ms | 0.66 | 7.34 | 65% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 173.970 ms | 0.89 | 6.72 | 59% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 565.114 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 581.296 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 1437.439 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 173.514 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=33554432 | 752.365 ms | 0.09 | 0.36 | 3% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 162.603 ms | 13.21 | 0.10 | 22% | ok | 5.6x (93%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 9.579 ms | 7.01 | 7.44 | 65% | ok | 1.0x (17%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 26.179 ms | 2.56 | 10.25 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.732 ms | 5.27 | 10.54 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 20.407 ms | 3.29 | 3.29 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 20.404 ms | 3.29 | 3.29 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 87.170 ms | 24.64 | 0.14 | n/a | ok | 5.2x (87%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 65.577 ms | 32.75 | 0.13 | n/a | ok | 5.5x (92%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 33.890 ms | 63.37 | 0.19 | n/a | ok | 5.6x (93%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 33.873 ms | 63.40 | 0.19 | n/a | ok | 5.6x (93%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 103.672 ms | 20.71 | 0.06 | n/a | ok | 5.5x (91%) |

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

