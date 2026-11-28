## ryzen-9-8945hs -- native, 8 threads, class large

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 506.4 GFLOP/s, f32 1011.2 GFLOP/s |
| ... hardware, AVX-512F | f64 506.4, f32 1011.2 GFLOP/s |
| ... this build, scalar | f64 12.5, f32 14.5 GFLOP/s |
| bandwidth | read 52.4, update 53.2, triad 41.0 GB/s (ceiling 53.2) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 82% | 108% | 0 | 51% |
| blas2 | 10 | 67% | 109% | 0 | 14% |
| blas3 | 10 | 2% | 21% | 0 | 13% |
| factor | 12 | 1% | 14% | 0 | 56% |
| eigen | 8 | 0% | 2% | 0 | 16% |
| elementwise | 5 | 7% | 10% | 0 | 13% |
| sparse | 5 | 32% | 102% | 0 | 13% |
| direct | 19 | 0% | 7% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 7% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 28% | 33% | 0 | 13% |
| mg | 1 | 24% | 24% | 0 | 14% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 11% | 18% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 98% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 9.393 ms | 7.14 | 57.16 | 108% | ok | 6.1x (77%) |
| blas1 | dot | f32 | n=33554432 | 6.161 ms | 10.89 | 43.57 | 82% | ok | 8.0x (99%) |
| blas1 | axpy | f64 | n=33554432 | 14.627 ms | 4.59 | 55.05 | 104% | ok | 4.1x (51%) |
| blas1 | axpy | f32 | n=33554432 | 7.616 ms | 8.81 | 52.87 | 99% | ok | 7.8x (98%) |
| blas1 | scale | f64 | n=33554432 | 9.489 ms | 3.54 | 56.58 | 106% | ok | 1.1x (14%) |
| blas1 | two_norm | f64 | n=33554432 | 6.701 ms | 10.01 | 40.06 | 75% | ok | 8.0x (99%) |
| blas1 | one_norm | f64 | n=33554432 | 25.602 ms | 1.31 | 10.48 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=33554432 | 25.649 ms | 1.31 | 10.47 | 20% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=33554432 | 25.613 ms | 1.31 | 10.48 | 20% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=33554432 | 1030.171 ms | 0.72 | 0.52 | 1% | ok | 1.0x (12%) |
| blas2 | gemv | f64 | n=8192 | 9.837 ms | 13.64 | 54.59 | 103% | ok | 1.5x (19%) |
| blas2 | gemv | f32 | n=8192 | 4.759 ms | 28.20 | 56.42 | 106% | ok | 2.8x (34%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 9.272 ms | 14.48 | 57.92 | 109% | ok | 3.4x (42%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 4.802 ms | 27.95 | 55.92 | 105% | ok | 2.8x (35%) |
| blas2 | symv | f64 | n=8192 | 14.304 ms | 9.38 | 18.78 | 35% | ok | 1.1x (14%) |
| blas2 | trmv | f64 | n=8192 | 7.618 ms | 8.81 | 35.25 | 66% | ok | 1.1x (13%) |
| blas2 | trmv | f32 | n=8192 | 7.179 ms | 9.35 | 18.71 | 35% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=8192 | 7.562 ms | 8.87 | 35.51 | 67% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=8192 | 7.269 ms | 9.23 | 18.47 | 35% | ok | 1.0x (12%) |
| blas2 | ger | f64 | n=8192 | 65.214 ms | 2.06 | 16.47 | 31% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 19.945 ms | 107.67 | 1.26 | 21% | ok | 7.6x (95%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 9.986 ms | 215.05 | 1.26 | 21% | ok | 7.7x (96%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 98.771 ms | 10.87 | 0.25 | 2% | ok | 1.0x (12%) |
| blas3 | symm | f64 | 1024x1024x1024 | 152.878 ms | 14.05 | 0.16 | 3% | ok | 1.0x (13%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 109.669 ms | 9.79 | 0.15 | 2% | ok | 1.0x (13%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 254.316 ms | 8.44 | 0.10 | 2% | ok | 1.0x (13%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 242.972 ms | 4.42 | 0.09 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 277.931 ms | 3.86 | 0.08 | 1% | ok | 1.0x (13%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 257.420 ms | 4.17 | 0.04 | 0% | ok | 1.0x (12%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 34.417 ms | 0.49 | 3.90 | 7% | ok | 1.0x (13%) |
| factor | lu | f64 | 1024x1024 | 70.843 ms | 10.10 | 0.24 | 2% | ok | 4.5x (56%) |
| factor | lu | f32 | 1024x1024 | 69.848 ms | 10.25 | 0.12 | 1% | ok | 4.5x (56%) |
| factor | cholesky | f64 | 1024x1024 | 56.136 ms | 6.38 | 0.30 | 1% | ok | 2.3x (29%) |
| factor | cholesky | f32 | 1024x1024 | 54.835 ms | 6.53 | 0.15 | 1% | ok | 2.3x (28%) |
| factor | ldlt | f64 | 1024x1024 | 126.009 ms | 2.84 | 0.13 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 1024x1024 | 92.885 ms | 3.85 | 0.18 | 1% | ok | 1.0x (13%) |
| factor | qr | f64 | 1024x1024 | 1217.243 ms | 1.18 | 0.01 | 0% | ok | 4.9x (61%) |
| factor | lq | f64 | 1024x1024 | 137.205 ms | 10.43 | 0.12 | 2% | ok | 4.8x (60%) |
| factor | hessenberg | f64 | 1024x1024 | 1376.556 ms | 2.60 | 0.01 | 1% | ok | 5.0x (63%) |
| factor | householder | f64 | 1024x1024 | 2.303 ms | 1.82 | 7.29 | 14%\* | ok | 7.6x (95%) |
| factor | inv | f64 | 1024x1024 | 612.942 ms | 3.50 | 0.03 | 1% | ok | 1.4x (17%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 78.798 ms | 9.08 | 0.21 | 2% | ok | 4.2x (52%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 583.609 ms | 0.31 | 0.01 | 0% | ok | 1.6x (20%) |
| eigen | eigen_symmetric | f64 | 512x512 | 3340.158 ms | 0.36 | 0.00 | 0% | ok | 1.1x (14%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 18355.721 ms | 0.07 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 512x512 | 1270.056 ms | 1.06 | 0.00 | 0% | ok | 1.3x (16%) |
| eigen | schur | f64 | 512x512 | 2312.107 ms | 1.45 | 0.00 | 0% | ok | 1.3x (16%) |
| eigen | svd | f64 | 512x512 | 1370.140 ms | 2.16 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | expm | f64 | 512x512 | 177.488 ms | 11.34 | 0.02 | 2% | ok | 1.8x (23%) |
| eigen | sylvester | f64 | 512x512 | 5404.509 ms | 1.49 | 0.00 | 0% | ok | 1.3x (16%) |
| elementwise | exp | f64 | n=33554432 | 133.020 ms | 0.25 | 4.04 | 8% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=33554432 | 213.954 ms | 0.16 | 2.51 | 5% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=33554432 | 194.596 ms | 0.17 | 2.76 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=33554432 | 104.857 ms | 0.32 | 5.12 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=33554432 | 155.050 ms | 0.22 | 3.46 | 7% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 2.005 ms | 5.23 | 54.36 | 102% | ok | 2.0x (25%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 6.370 ms | 1.64 | 17.11 | 32% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 9.052 ms | 18.52 | 38.91 | 73% | ok | 3.4x (42%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 97.529 ms | 0.13 | 0.47 | 1% | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 18.279 ms | 0.56 | 4.58 | 9% | ok | 1.1x (13%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 165.029 ms | 0.65 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 302.910 ms | 0.35 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 107.313 ms | 1.00 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 532.775 ms | 0.40 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 450.113 ms | 0.48 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 214.932 ms | 1.00 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 616.877 ms | 0.04 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 22.133 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 32.667 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 347.086 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 1.376 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 107.737 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 17.586 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 142.417 ms | 0.75 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 4.081 ms | 0.98 | 3.93 | 7%\* | ok | 1.0x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 41.375 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 49.599 ms | 2.16 | 0.06 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 4.184 ms | 0.96 | 3.84 | 7%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 18.316 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 489.152 ms | 0.44 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.330 ms | 0.26 | 1.05 | 2%\* | ok | 1.0x (12%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 28.081 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 399.727 ms | 0.54 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.310 ms | 0.26 | 1.05 | 2%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 13.637 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1062.350 ms | 0.91 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 7.914 ms | 0.95 | 3.79 | 7% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 35.784 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 250.347 ms | 3.87 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 8.122 ms | 0.92 | 3.69 | 7% | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 14.103 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 3614.008 ms | 0.54 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.513 ms | 0.25 | 0.98 | 2% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 20.417 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1874.886 ms | 1.03 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.537 ms | 0.25 | 0.98 | 2% | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 374.340 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 546.324 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 238.537 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 177.098 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 202.381 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 152.335 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 646.007 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 299.211 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 186.383 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 229.156 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 213.251 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 237.644 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 302.066 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.225 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 256^2 grid | 13.809 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 256^2 grid | 126.541 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 256^2 grid | 10.325 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 256^2 grid | 9.795 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 256^2 grid | 0.146 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 3.476 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 6.661 ms | 1.57 | 17.62 | 33% | ok | 1.1x (13%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 7.970 ms | 1.31 | 14.73 | 28% | ok | 1.0x (12%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 9.132 ms | 1.15 | 12.85 | 24% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 92.038 ms | 1.69 | 12.70 | 24% | ok | 1.1x (14%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 189.576 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 188.563 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 823.129 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 39.653 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=33554432 | 57.570 ms | 1.17 | 4.66 | 9% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 37.700 ms | 56.96 | 0.45 | 11% | ok | 7.8x (97%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 7.378 ms | 9.10 | 9.66 | 18% | ok | 1.0x (13%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 16.272 ms | 4.12 | 16.50 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.981 ms | 5.17 | 10.34 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 6.938 ms | 9.67 | 9.67 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 6.939 ms | 9.67 | 9.67 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 9.767 ms | 219.86 | 1.29 | n/a | ok | 7.9x (98%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 38.978 ms | 55.09 | 0.22 | n/a | ok | 7.9x (98%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 35.026 ms | 61.31 | 0.18 | n/a | ok | 7.9x (98%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 35.014 ms | 61.33 | 0.18 | n/a | ok | 7.8x (98%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 35.240 ms | 60.94 | 0.18 | n/a | ok | 7.8x (98%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

