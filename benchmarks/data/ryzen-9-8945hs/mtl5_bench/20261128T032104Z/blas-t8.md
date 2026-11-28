## ryzen-9-8945hs -- blas, 8 threads, class large

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 556.1 GFLOP/s, f32 1011.6 GFLOP/s |
| ... hardware, AVX-512F | f64 556.1, f32 1011.6 GFLOP/s |
| ... this build, scalar | f64 14.6, f32 14.6 GFLOP/s |
| bandwidth | read 52.7, update 53.1, triad 40.9 GB/s (ceiling 53.1) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 82% | 110% | 0 | 51% |
| blas2 | 10 | 100% | 104% | 0 | 18% |
| blas3 | 10 | 58% | 85% | 0 | 72% |
| factor | 12 | 2% | 13% | 0 | 13% |
| eigen | 8 | 1% | 6% | 0 | 13% |
| elementwise | 5 | 6% | 10% | 0 | 13% |
| sparse | 5 | 32% | 105% | 0 | 13% |
| direct | 19 | 0% | 7% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 7% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 29% | 31% | 0 | 13% |
| mg | 1 | 24% | 24% | 0 | 14% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 10% | 18% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 97% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 9.154 ms | 7.33 | 58.65 | 110% | ok | 5.2x (64%) |
| blas1 | dot | f32 | n=33554432 | 6.148 ms | 10.92 | 43.66 | 82% | ok | 8.0x (100%) |
| blas1 | axpy | f64 | n=33554432 | 14.703 ms | 4.56 | 54.77 | 103% | ok | 4.1x (51%) |
| blas1 | axpy | f32 | n=33554432 | 7.596 ms | 8.83 | 53.01 | 100% | ok | 7.9x (98%) |
| blas1 | scale | f64 | n=33554432 | 9.390 ms | 3.57 | 57.17 | 108% | ok | 1.1x (14%) |
| blas1 | two_norm | f64 | n=33554432 | 6.148 ms | 10.92 | 43.66 | 82% | ok | 8.0x (100%) |
| blas1 | one_norm | f64 | n=33554432 | 25.700 ms | 1.31 | 10.44 | 20% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=33554432 | 25.647 ms | 1.31 | 10.47 | 20% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=33554432 | 25.597 ms | 1.31 | 10.49 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=33554432 | 995.021 ms | 0.74 | 0.54 | 1% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=8192 | 10.163 ms | 13.21 | 52.84 | 100% | ok | 1.5x (18%) |
| blas2 | gemv | f32 | n=8192 | 5.119 ms | 26.22 | 52.45 | 99% | ok | 1.5x (18%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 9.720 ms | 13.81 | 55.24 | 104% | ok | 1.7x (21%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 4.973 ms | 26.99 | 53.99 | 102% | ok | 1.4x (17%) |
| blas2 | symv | f64 | n=8192 | 5.037 ms | 26.65 | 53.32 | 100% | ok | 1.5x (19%) |
| blas2 | trmv | f64 | n=8192 | 4.964 ms | 13.52 | 54.10 | 102% | ok | 1.5x (18%) |
| blas2 | trmv | f32 | n=8192 | 2.591 ms | 25.90 | 51.83 | 98% | ok | 1.6x (20%) |
| blas2 | trsv | f64 | n=8192 | 6.671 ms | 10.06 | 40.26 | 76% | ok | 1.1x (13%) |
| blas2 | trsv | f32 | n=8192 | 3.885 ms | 17.27 | 34.56 | 65% | ok | 0.9x (11%) |
| blas2 | ger | f64 | n=8192 | 19.540 ms | 6.87 | 54.96 | 104% | ok | 1.5x (19%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 4.869 ms | 441.04 | 5.17 | 79% | ok | 7.4x (93%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 2.500 ms | 858.96 | 5.03 | 85% | ok | 7.5x (93%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 7.216 ms | 148.81 | 3.49 | 27% | ok | 3.3x (42%) |
| blas3 | symm | f64 | 1024x1024x1024 | 5.038 ms | 426.23 | 4.99 | 77% | ok | 7.4x (92%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 8.616 ms | 124.62 | 1.95 | 22% | ok | 2.8x (35%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 12.727 ms | 168.73 | 1.98 | 30% | ok | 3.4x (42%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 3.328 ms | 322.68 | 6.30 | 58% | ok | 5.5x (68%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 4.042 ms | 265.65 | 5.19 | 48% | ok | 5.8x (72%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 1.701 ms | 631.20 | 6.16 | 62% | ok | 6.7x (84%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 34.350 ms | 0.49 | 3.91 | 7% | ok | 1.0x (13%) |
| factor | lu | f64 | 1024x1024 | 29.110 ms | 24.59 | 0.58 | 4% | ok | 1.1x (14%) |
| factor | lu | f32 | 1024x1024 | 24.994 ms | 28.64 | 0.34 | 3% | ok | 1.0x (13%) |
| factor | cholesky | f64 | 1024x1024 | 26.154 ms | 13.68 | 0.64 | 2% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 1024x1024 | 21.669 ms | 16.52 | 0.39 | 2% | ok | 1.0x (12%) |
| factor | ldlt | f64 | 1024x1024 | 127.978 ms | 2.80 | 0.13 | 1% | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 1024x1024 | 92.847 ms | 3.85 | 0.18 | 1% | ok | 1.0x (12%) |
| factor | qr | f64 | 1024x1024 | 56.880 ms | 25.17 | 0.29 | 5% | ok | 1.2x (15%) |
| factor | lq | f64 | 1024x1024 | 135.916 ms | 10.53 | 0.12 | 2% | ok | 4.8x (60%) |
| factor | hessenberg | f64 | 1024x1024 | 1385.629 ms | 2.58 | 0.01 | 0% | ok | 5.0x (62%) |
| factor | householder | f64 | 1024x1024 | 2.354 ms | 1.78 | 7.13 | 13%\* | ok | 7.5x (93%) |
| factor | inv | f64 | 1024x1024 | 578.179 ms | 3.71 | 0.03 | 1% | ok | 1.0x (12%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 44.504 ms | 16.08 | 0.38 | 3% | ok | 0.8x (10%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 22.794 ms | 7.85 | 0.18 | 1% | ok | 0.6x (7%) |
| eigen | eigen_symmetric | f64 | 512x512 | 3275.852 ms | 0.37 | 0.00 | 0% | ok | 1.1x (14%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 18065.556 ms | 0.07 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 512x512 | 311.652 ms | 4.31 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | schur | f64 | 512x512 | 368.352 ms | 9.11 | 0.01 | 2% | ok | 0.9x (11%) |
| eigen | svd | f64 | 512x512 | 84.663 ms | 34.88 | 0.05 | 6% | ok | 1.0x (13%) |
| eigen | expm | f64 | 512x512 | 148.898 ms | 13.52 | 0.03 | 2% | ok | 1.2x (15%) |
| eigen | sylvester | f64 | 512x512 | 1578.997 ms | 5.10 | 0.00 | 1% | ok | 1.0x (12%) |
| elementwise | exp | f64 | n=33554432 | 133.486 ms | 0.25 | 4.02 | 8% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=33554432 | 212.330 ms | 0.16 | 2.53 | 5% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=33554432 | 191.755 ms | 0.17 | 2.80 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=33554432 | 105.031 ms | 0.32 | 5.11 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=33554432 | 169.890 ms | 0.20 | 3.16 | 6% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 1.964 ms | 5.33 | 55.48 | 105% | ok | 2.1x (27%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 6.506 ms | 1.61 | 16.75 | 32% | ok | 1.0x (12%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 8.963 ms | 18.70 | 39.30 | 74% | ok | 3.4x (43%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 97.702 ms | 0.13 | 0.47 | 1% | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 18.301 ms | 0.56 | 4.58 | 9% | ok | 1.1x (13%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 172.512 ms | 0.62 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 301.842 ms | 0.35 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 107.750 ms | 0.99 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 519.407 ms | 0.41 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 453.276 ms | 0.47 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 217.428 ms | 0.98 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 627.905 ms | 0.04 | 0.00 | 0% | ok | 1.0x (12%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 22.533 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 32.543 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 343.664 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 1.376 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 108.343 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 17.473 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 142.297 ms | 0.75 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 4.304 ms | 0.93 | 3.73 | 7%\* | ok | 1.0x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 41.270 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 49.819 ms | 2.15 | 0.06 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 4.244 ms | 0.95 | 3.78 | 7%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 18.542 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 487.956 ms | 0.44 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.322 ms | 0.26 | 1.05 | 2%\* | ok | 1.0x (12%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 27.986 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 398.826 ms | 0.54 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.335 ms | 0.26 | 1.05 | 2%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 12.805 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1058.097 ms | 0.92 | 0.00 | 0% | ok | 1.1x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 7.836 ms | 0.96 | 3.83 | 7% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 35.350 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 252.883 ms | 3.83 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 7.937 ms | 0.94 | 3.78 | 7% | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 14.077 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 3646.052 ms | 0.53 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.698 ms | 0.24 | 0.98 | 2% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 19.787 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1939.819 ms | 1.00 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.524 ms | 0.25 | 0.98 | 2% | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 360.498 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 522.315 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 235.177 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 171.202 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 194.714 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 149.967 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 595.118 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 293.833 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 181.838 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 219.529 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 201.205 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 224.594 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 281.965 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.212 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 256^2 grid | 13.914 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 256^2 grid | 126.734 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 256^2 grid | 10.652 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 256^2 grid | 9.911 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 256^2 grid | 0.150 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 7.232 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 7.039 ms | 1.49 | 16.68 | 31% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 7.725 ms | 1.36 | 15.19 | 29% | ok | 1.0x (12%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 9.083 ms | 1.15 | 12.92 | 24% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 90.685 ms | 1.72 | 12.89 | 24% | ok | 1.1x (14%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 190.547 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 191.492 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 820.581 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 38.796 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| mixed | dot | f32>f64 | n=33554432 | 49.044 ms | 1.37 | 5.47 | 10% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 38.079 ms | 56.40 | 0.44 | 10% | ok | 7.8x (98%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 7.413 ms | 9.05 | 9.62 | 18% | ok | 1.0x (12%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 16.268 ms | 4.13 | 16.50 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.992 ms | 5.17 | 10.33 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 8.539 ms | 7.86 | 7.86 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 8.523 ms | 7.87 | 7.87 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 10.173 ms | 211.09 | 1.24 | n/a | ok | 7.7x (97%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 39.025 ms | 55.03 | 0.21 | n/a | ok | 7.8x (98%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 35.057 ms | 61.26 | 0.18 | n/a | ok | 7.8x (98%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 35.045 ms | 61.28 | 0.18 | n/a | ok | 7.8x (98%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 35.278 ms | 60.87 | 0.18 | n/a | ok | 7.8x (98%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

