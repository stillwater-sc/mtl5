## ryzen-9-8945hs -- native, 8 threads, class large

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 505.9 GFLOP/s, f32 1012.4 GFLOP/s |
| ... hardware, AVX-512F | f64 505.9, f32 1012.4 GFLOP/s |
| ... this build, scalar | f64 12.5, f32 14.5 GFLOP/s |
| bandwidth | read 52.4, update 52.9, triad 40.9 GB/s (ceiling 52.9) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 82% | 109% | 0 | 51% |
| blas2 | 10 | 67% | 109% | 0 | 12% |
| blas3 | 10 | 2% | 21% | 0 | 13% |
| factor | 12 | 1% | 14% | 0 | 55% |
| eigen | 8 | 0% | 2% | 0 | 16% |
| elementwise | 5 | 7% | 10% | 0 | 13% |
| sparse | 5 | 29% | 102% | 0 | 13% |
| direct | 19 | 0% | 7% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 7% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 28% | 31% | 0 | 12% |
| mg | 1 | 24% | 24% | 0 | 14% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 11% | 18% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 98% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 9.273 ms | 7.24 | 57.90 | 109% | ok | 6.2x (78%) |
| blas1 | dot | f32 | n=33554432 | 6.163 ms | 10.89 | 43.56 | 82% | ok | 8.0x (100%) |
| blas1 | axpy | f64 | n=33554432 | 14.635 ms | 4.59 | 55.02 | 104% | ok | 4.1x (51%) |
| blas1 | axpy | f32 | n=33554432 | 7.610 ms | 8.82 | 52.91 | 100% | ok | 7.9x (98%) |
| blas1 | scale | f64 | n=33554432 | 9.464 ms | 3.55 | 56.73 | 107% | ok | 1.1x (14%) |
| blas1 | two_norm | f64 | n=33554432 | 6.967 ms | 9.63 | 38.53 | 73% | ok | 8.0x (100%) |
| blas1 | one_norm | f64 | n=33554432 | 25.598 ms | 1.31 | 10.49 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=33554432 | 25.656 ms | 1.31 | 10.46 | 20% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=33554432 | 25.602 ms | 1.31 | 10.49 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=33554432 | 994.795 ms | 0.74 | 0.54 | 1% | ok | 1.0x (12%) |
| blas2 | gemv | f64 | n=8192 | 9.800 ms | 13.69 | 54.79 | 104% | ok | 1.5x (19%) |
| blas2 | gemv | f32 | n=8192 | 4.756 ms | 28.22 | 56.45 | 107% | ok | 2.7x (34%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 9.274 ms | 14.47 | 57.90 | 109% | ok | 3.0x (37%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 4.661 ms | 28.80 | 57.61 | 109% | ok | 2.9x (36%) |
| blas2 | symv | f64 | n=8192 | 13.977 ms | 9.60 | 19.22 | 36% | ok | 1.0x (12%) |
| blas2 | trmv | f64 | n=8192 | 7.586 ms | 8.85 | 35.40 | 67% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=8192 | 7.179 ms | 9.35 | 18.71 | 35% | ok | 1.0x (12%) |
| blas2 | trsv | f64 | n=8192 | 7.644 ms | 8.78 | 35.13 | 66% | ok | 1.0x (12%) |
| blas2 | trsv | f32 | n=8192 | 7.201 ms | 9.32 | 18.65 | 35% | ok | 1.0x (12%) |
| blas2 | ger | f64 | n=8192 | 65.735 ms | 2.04 | 16.34 | 31% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 19.851 ms | 108.18 | 1.27 | 21% | ok | 7.6x (95%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 9.962 ms | 215.57 | 1.26 | 21% | ok | 7.7x (97%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 98.905 ms | 10.86 | 0.25 | 2% | ok | 1.0x (12%) |
| blas3 | symm | f64 | 1024x1024x1024 | 152.127 ms | 14.12 | 0.17 | 3% | ok | 1.0x (13%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 109.647 ms | 9.79 | 0.15 | 2% | ok | 1.0x (12%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 263.870 ms | 8.14 | 0.10 | 2% | ok | 1.0x (13%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 242.429 ms | 4.43 | 0.09 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 277.864 ms | 3.86 | 0.08 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 253.917 ms | 4.23 | 0.04 | 0% | ok | 1.0x (13%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 34.427 ms | 0.49 | 3.90 | 7% | ok | 1.0x (13%) |
| factor | lu | f64 | 1024x1024 | 71.781 ms | 9.97 | 0.23 | 2% | ok | 4.4x (55%) |
| factor | lu | f32 | 1024x1024 | 68.262 ms | 10.49 | 0.12 | 1% | ok | 4.5x (56%) |
| factor | cholesky | f64 | 1024x1024 | 55.862 ms | 6.41 | 0.30 | 1% | ok | 2.3x (28%) |
| factor | cholesky | f32 | 1024x1024 | 54.816 ms | 6.53 | 0.15 | 1% | ok | 2.3x (28%) |
| factor | ldlt | f64 | 1024x1024 | 126.400 ms | 2.83 | 0.13 | 1% | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 1024x1024 | 92.341 ms | 3.88 | 0.18 | 1% | ok | 1.0x (13%) |
| factor | qr | f64 | 1024x1024 | 1209.737 ms | 1.18 | 0.01 | 0% | ok | 4.9x (61%) |
| factor | lq | f64 | 1024x1024 | 137.039 ms | 10.45 | 0.12 | 2% | ok | 4.8x (60%) |
| factor | hessenberg | f64 | 1024x1024 | 1458.296 ms | 2.45 | 0.01 | 0% | ok | 4.8x (60%) |
| factor | householder | f64 | 1024x1024 | 2.302 ms | 1.82 | 7.29 | 14%\* | ok | 7.6x (95%) |
| factor | inv | f64 | 1024x1024 | 605.446 ms | 3.55 | 0.03 | 1% | ok | 1.4x (18%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 79.030 ms | 9.06 | 0.21 | 2% | ok | 4.1x (52%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 578.245 ms | 0.31 | 0.01 | 0% | ok | 1.6x (21%) |
| eigen | eigen_symmetric | f64 | 512x512 | 3342.855 ms | 0.36 | 0.00 | 0% | ok | 1.1x (14%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 18330.918 ms | 0.07 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 512x512 | 1269.731 ms | 1.06 | 0.00 | 0% | ok | 1.3x (16%) |
| eigen | schur | f64 | 512x512 | 2249.204 ms | 1.49 | 0.00 | 0% | ok | 1.3x (16%) |
| eigen | svd | f64 | 512x512 | 1370.421 ms | 2.15 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | expm | f64 | 512x512 | 177.151 ms | 11.36 | 0.02 | 2% | ok | 1.8x (23%) |
| eigen | sylvester | f64 | 512x512 | 5499.700 ms | 1.46 | 0.00 | 0% | ok | 1.2x (16%) |
| elementwise | exp | f64 | n=33554432 | 153.841 ms | 0.22 | 3.49 | 7% | ok | 0.9x (11%) |
| elementwise | log | f64 | n=33554432 | 206.553 ms | 0.16 | 2.60 | 5% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=33554432 | 194.711 ms | 0.17 | 2.76 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=33554432 | 104.593 ms | 0.32 | 5.13 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=33554432 | 154.882 ms | 0.22 | 3.47 | 7% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 2.014 ms | 5.20 | 54.10 | 102% | ok | 2.0x (25%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 7.194 ms | 1.46 | 15.15 | 29% | ok | 0.9x (12%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 9.054 ms | 18.52 | 38.91 | 74% | ok | 3.5x (44%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 96.819 ms | 0.13 | 0.48 | 1% | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 18.324 ms | 0.56 | 4.57 | 9% | ok | 1.1x (13%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 170.578 ms | 0.63 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 302.385 ms | 0.35 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 107.231 ms | 1.00 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 521.036 ms | 0.41 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 445.462 ms | 0.48 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 215.386 ms | 0.99 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 624.924 ms | 0.04 | 0.00 | 0% | ok | 1.0x (12%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 21.734 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 32.559 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 344.524 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 1.383 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 106.518 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 17.210 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 141.537 ms | 0.76 | 0.02 | 0% | ok | 1.1x (13%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 4.130 ms | 0.97 | 3.89 | 7%\* | ok | 1.0x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 40.791 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 49.769 ms | 2.15 | 0.06 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 4.204 ms | 0.95 | 3.82 | 7%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 18.203 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 484.267 ms | 0.44 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.349 ms | 0.26 | 1.05 | 2%\* | ok | 1.0x (12%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 28.004 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 398.844 ms | 0.54 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.306 ms | 0.26 | 1.05 | 2%\* | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 12.684 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1034.552 ms | 0.94 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 8.025 ms | 0.93 | 3.74 | 7% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 35.478 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 252.833 ms | 3.83 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 8.132 ms | 0.92 | 3.69 | 7% | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 14.058 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 3783.848 ms | 0.51 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.605 ms | 0.24 | 0.98 | 2% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 19.611 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1872.448 ms | 1.04 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.744 ms | 0.24 | 0.98 | 2% | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 377.900 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 557.108 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 239.036 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 178.291 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 202.583 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 152.876 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 646.812 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 301.465 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 186.273 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 231.626 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 216.281 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 239.182 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 304.408 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.224 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 256^2 grid | 13.582 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 256^2 grid | 125.426 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 256^2 grid | 10.401 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 256^2 grid | 9.732 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 256^2 grid | 0.146 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 3.594 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 7.151 ms | 1.47 | 16.41 | 31% | ok | 1.0x (12%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 7.824 ms | 1.34 | 15.00 | 28% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 9.241 ms | 1.13 | 12.70 | 24% | ok | 1.0x (12%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 91.019 ms | 1.71 | 12.84 | 24% | ok | 1.1x (14%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 189.332 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 189.007 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 821.857 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 39.919 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=33554432 | 57.562 ms | 1.17 | 4.66 | 9% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 37.635 ms | 57.06 | 0.45 | 11% | ok | 7.8x (98%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 7.368 ms | 9.11 | 9.68 | 18% | ok | 1.0x (13%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 16.256 ms | 4.13 | 16.51 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.988 ms | 5.17 | 10.33 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 6.978 ms | 9.62 | 9.62 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 6.934 ms | 9.68 | 9.68 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 9.776 ms | 219.67 | 1.29 | n/a | ok | 7.8x (98%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 39.080 ms | 54.95 | 0.21 | n/a | ok | 7.8x (98%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 35.069 ms | 61.24 | 0.18 | n/a | ok | 7.8x (98%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 35.070 ms | 61.23 | 0.18 | n/a | ok | 7.8x (98%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 35.296 ms | 60.84 | 0.18 | n/a | ok | 7.8x (98%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

