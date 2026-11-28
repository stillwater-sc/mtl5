## ryzen-9-8945hs -- blas, 8 threads, class large

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 555.5 GFLOP/s, f32 1011.8 GFLOP/s |
| ... hardware, AVX-512F | f64 555.5, f32 1011.8 GFLOP/s |
| ... this build, scalar | f64 14.6, f32 14.7 GFLOP/s |
| bandwidth | read 52.5, update 52.9, triad 40.9 GB/s (ceiling 52.9) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 83% | 111% | 0 | 51% |
| blas2 | 10 | 101% | 105% | 0 | 18% |
| blas3 | 10 | 59% | 85% | 0 | 71% |
| factor | 12 | 2% | 14% | 0 | 13% |
| eigen | 8 | 1% | 6% | 0 | 13% |
| elementwise | 5 | 6% | 10% | 0 | 13% |
| sparse | 5 | 28% | 103% | 0 | 13% |
| direct | 19 | 0% | 7% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 7% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 31% | 0 | 12% |
| mg | 1 | 24% | 24% | 0 | 14% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 10% | 18% | 0 | 14% |
| integer | 9 | n/a | n/a | 0 | 98% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 9.172 ms | 7.32 | 58.54 | 111% | ok | 5.1x (64%) |
| blas1 | dot | f32 | n=33554432 | 6.148 ms | 10.91 | 43.66 | 83% | ok | 8.0x (100%) |
| blas1 | axpy | f64 | n=33554432 | 14.604 ms | 4.60 | 55.14 | 104% | ok | 4.1x (51%) |
| blas1 | axpy | f32 | n=33554432 | 7.586 ms | 8.85 | 53.08 | 100% | ok | 7.9x (98%) |
| blas1 | scale | f64 | n=33554432 | 9.541 ms | 3.52 | 56.27 | 106% | ok | 1.1x (14%) |
| blas1 | two_norm | f64 | n=33554432 | 6.149 ms | 10.91 | 43.66 | 83% | ok | 8.0x (100%) |
| blas1 | one_norm | f64 | n=33554432 | 25.679 ms | 1.31 | 10.45 | 20% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=33554432 | 25.642 ms | 1.31 | 10.47 | 20% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=33554432 | 25.596 ms | 1.31 | 10.49 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=33554432 | 994.986 ms | 0.74 | 0.54 | 1% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=8192 | 10.256 ms | 13.09 | 52.36 | 99% | ok | 1.3x (16%) |
| blas2 | gemv | f32 | n=8192 | 5.145 ms | 26.09 | 52.19 | 99% | ok | 1.3x (16%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 9.672 ms | 13.88 | 55.52 | 105% | ok | 1.4x (18%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 5.005 ms | 26.82 | 53.64 | 101% | ok | 1.3x (17%) |
| blas2 | symv | f64 | n=8192 | 5.017 ms | 26.75 | 53.54 | 101% | ok | 1.5x (19%) |
| blas2 | trmv | f64 | n=8192 | 4.960 ms | 13.53 | 54.14 | 102% | ok | 1.4x (18%) |
| blas2 | trmv | f32 | n=8192 | 2.645 ms | 25.37 | 50.77 | 96% | ok | 1.5x (19%) |
| blas2 | trsv | f64 | n=8192 | 7.212 ms | 9.31 | 37.24 | 70% | ok | 0.9x (12%) |
| blas2 | trsv | f32 | n=8192 | 3.988 ms | 16.83 | 33.67 | 64% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=8192 | 19.526 ms | 6.87 | 55.00 | 104% | ok | 1.5x (18%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 4.867 ms | 441.22 | 5.17 | 79% | ok | 7.5x (93%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 2.496 ms | 860.23 | 5.04 | 85% | ok | 7.6x (96%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 7.278 ms | 147.53 | 3.46 | 27% | ok | 3.3x (42%) |
| blas3 | symm | f64 | 1024x1024x1024 | 5.012 ms | 428.51 | 5.02 | 77% | ok | 7.4x (93%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 8.690 ms | 123.56 | 1.93 | 22% | ok | 2.8x (35%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 12.702 ms | 169.07 | 1.98 | 30% | ok | 3.4x (42%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 3.255 ms | 329.91 | 6.44 | 59% | ok | 5.6x (70%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 4.063 ms | 264.27 | 5.16 | 48% | ok | 5.7x (71%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 1.700 ms | 631.72 | 6.17 | 62% | ok | 6.7x (83%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 34.525 ms | 0.49 | 3.89 | 7% | ok | 1.0x (13%) |
| factor | lu | f64 | 1024x1024 | 29.113 ms | 24.59 | 0.58 | 4% | ok | 1.1x (14%) |
| factor | lu | f32 | 1024x1024 | 24.942 ms | 28.70 | 0.34 | 3% | ok | 1.0x (13%) |
| factor | cholesky | f64 | 1024x1024 | 26.125 ms | 13.70 | 0.64 | 2% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 1024x1024 | 21.656 ms | 16.53 | 0.39 | 2% | ok | 1.0x (12%) |
| factor | ldlt | f64 | 1024x1024 | 128.103 ms | 2.79 | 0.13 | 1% | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 1024x1024 | 92.198 ms | 3.88 | 0.18 | 1% | ok | 1.0x (13%) |
| factor | qr | f64 | 1024x1024 | 56.733 ms | 25.24 | 0.30 | 5% | ok | 1.2x (15%) |
| factor | lq | f64 | 1024x1024 | 136.484 ms | 10.49 | 0.12 | 2% | ok | 4.8x (60%) |
| factor | hessenberg | f64 | 1024x1024 | 1389.311 ms | 2.58 | 0.01 | 0% | ok | 5.0x (62%) |
| factor | householder | f64 | 1024x1024 | 2.311 ms | 1.82 | 7.26 | 14%\* | ok | 7.6x (95%) |
| factor | inv | f64 | 1024x1024 | 572.245 ms | 3.75 | 0.03 | 1% | ok | 1.0x (12%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 44.458 ms | 16.10 | 0.38 | 3% | ok | 0.8x (10%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 22.848 ms | 7.83 | 0.18 | 1% | ok | 0.6x (7%) |
| eigen | eigen_symmetric | f64 | 512x512 | 3243.997 ms | 0.37 | 0.00 | 0% | ok | 1.1x (14%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 18142.822 ms | 0.07 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 512x512 | 316.456 ms | 4.24 | 0.01 | 1% | ok | 0.9x (12%) |
| eigen | schur | f64 | 512x512 | 371.692 ms | 9.03 | 0.01 | 2% | ok | 0.9x (11%) |
| eigen | svd | f64 | 512x512 | 84.784 ms | 34.83 | 0.05 | 6% | ok | 1.0x (13%) |
| eigen | expm | f64 | 512x512 | 148.372 ms | 13.57 | 0.03 | 2% | ok | 1.2x (15%) |
| eigen | sylvester | f64 | 512x512 | 1595.500 ms | 5.05 | 0.00 | 1% | ok | 1.0x (12%) |
| elementwise | exp | f64 | n=33554432 | 133.460 ms | 0.25 | 4.02 | 8% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=33554432 | 212.020 ms | 0.16 | 2.53 | 5% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=33554432 | 191.354 ms | 0.18 | 2.81 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=33554432 | 105.152 ms | 0.32 | 5.11 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=33554432 | 169.867 ms | 0.20 | 3.16 | 6% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 2.008 ms | 5.22 | 54.27 | 103% | ok | 2.2x (28%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 7.314 ms | 1.43 | 14.90 | 28% | ok | 0.9x (11%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 9.048 ms | 18.53 | 38.93 | 74% | ok | 3.5x (44%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 97.022 ms | 0.13 | 0.47 | 1% | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 18.242 ms | 0.56 | 4.59 | 9% | ok | 1.1x (13%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 170.281 ms | 0.63 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 300.172 ms | 0.36 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 107.011 ms | 1.00 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 520.880 ms | 0.41 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 456.276 ms | 0.47 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 216.816 ms | 0.99 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 621.186 ms | 0.04 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 22.151 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 32.309 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 351.300 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 1.384 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 107.647 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 17.605 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 144.878 ms | 0.74 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 4.106 ms | 0.98 | 3.91 | 7%\* | ok | 1.0x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 41.486 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 50.085 ms | 2.14 | 0.06 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 4.190 ms | 0.96 | 3.83 | 7%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 18.727 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 485.851 ms | 0.44 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.345 ms | 0.26 | 1.05 | 2%\* | ok | 1.0x (12%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 28.262 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 406.750 ms | 0.53 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.400 ms | 0.26 | 1.04 | 2%\* | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 12.916 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1065.553 ms | 0.91 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 7.909 ms | 0.95 | 3.79 | 7% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 35.192 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 252.481 ms | 3.84 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 8.139 ms | 0.92 | 3.68 | 7% | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 14.089 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 3621.579 ms | 0.54 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.538 ms | 0.25 | 0.98 | 2% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 19.415 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1968.350 ms | 0.98 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.504 ms | 0.25 | 0.98 | 2% | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 361.855 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 520.543 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 234.054 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 170.094 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 193.796 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 149.538 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 592.499 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 293.341 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 181.382 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 218.103 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 201.946 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 223.872 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 281.521 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.212 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 256^2 grid | 13.634 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 256^2 grid | 124.772 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 256^2 grid | 10.356 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 256^2 grid | 9.696 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 256^2 grid | 0.145 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 7.217 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 7.191 ms | 1.46 | 16.32 | 31% | ok | 1.0x (12%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 7.746 ms | 1.35 | 15.15 | 29% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 9.439 ms | 1.11 | 12.44 | 24% | ok | 1.0x (12%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 91.462 ms | 1.70 | 12.78 | 24% | ok | 1.1x (14%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 188.325 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 189.016 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 819.861 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 39.223 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=33554432 | 49.066 ms | 1.37 | 5.47 | 10% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 38.135 ms | 56.31 | 0.44 | 10% | ok | 7.8x (98%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 7.424 ms | 9.04 | 9.60 | 18% | ok | 1.1x (14%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 16.274 ms | 4.12 | 16.49 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 13.022 ms | 5.15 | 10.31 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 8.546 ms | 7.85 | 7.85 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 8.554 ms | 7.85 | 7.85 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 10.072 ms | 213.20 | 1.25 | n/a | ok | 7.9x (98%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 38.959 ms | 55.12 | 0.22 | n/a | ok | 7.8x (98%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 35.061 ms | 61.25 | 0.18 | n/a | ok | 7.8x (98%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 35.115 ms | 61.16 | 0.18 | n/a | ok | 7.8x (98%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 35.256 ms | 60.91 | 0.18 | n/a | ok | 7.8x (98%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

