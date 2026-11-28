## ryzen-9-8945hs -- native-fast, 8 threads, class large

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 506.3 GFLOP/s, f32 1012.4 GFLOP/s |
| ... hardware, AVX-512F | f64 506.3, f32 1012.4 GFLOP/s |
| ... this build, SSE2 | f64 320.3, f32 639.3 GFLOP/s |
| bandwidth | read 52.6, update 53.3, triad 41.2 GB/s (ceiling 53.3) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 105% | 112% | 0 | 13% |
| blas2 | 10 | 68% | 110% | 0 | 15% |
| blas3 | 10 | 14% | 24% | 0 | 52% |
| factor | 12 | 1% | 14% | 0 | 56% |
| eigen | 8 | 0% | 2% | 0 | 16% |
| elementwise | 5 | 7% | 10% | 0 | 13% |
| sparse | 5 | 32% | 101% | 0 | 13% |
| direct | 19 | 0% | 7% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 7% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 14% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 28% | 31% | 0 | 13% |
| mg | 1 | 24% | 24% | 0 | 14% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 55% | 58% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 84% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 9.022 ms | 7.44 | 59.50 | 112% | ok | 1.1x (14%) |
| blas1 | dot | f32 | n=33554432 | 4.506 ms | 14.89 | 59.57 | 112% | ok | 1.1x (14%) |
| blas1 | axpy | f64 | n=33554432 | 14.571 ms | 4.61 | 55.27 | 104% | ok | 1.0x (13%) |
| blas1 | axpy | f32 | n=33554432 | 7.104 ms | 9.45 | 56.68 | 106% | ok | 1.0x (13%) |
| blas1 | scale | f64 | n=33554432 | 9.552 ms | 3.51 | 56.21 | 105% | ok | 1.0x (13%) |
| blas1 | two_norm | f64 | n=33554432 | 4.518 ms | 14.85 | 59.41 | 111% | ok | 1.0x (13%) |
| blas1 | one_norm | f64 | n=33554432 | 25.680 ms | 1.31 | 10.45 | 20% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=33554432 | 25.649 ms | 1.31 | 10.47 | 20% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=33554432 | 25.603 ms | 1.31 | 10.48 | 20% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=33554432 | 994.666 ms | 0.74 | 0.54 | 1% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=8192 | 9.179 ms | 14.62 | 58.51 | 110% | ok | 1.6x (20%) |
| blas2 | gemv | f32 | n=8192 | 4.616 ms | 29.07 | 58.16 | 109% | ok | 1.5x (19%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 9.218 ms | 14.56 | 58.26 | 109% | ok | 1.7x (21%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 4.665 ms | 28.77 | 57.56 | 108% | ok | 1.7x (22%) |
| blas2 | symv | f64 | n=8192 | 13.182 ms | 10.18 | 20.37 | 38% | ok | 1.2x (15%) |
| blas2 | trmv | f64 | n=8192 | 7.721 ms | 8.69 | 34.78 | 65% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=8192 | 3.988 ms | 16.83 | 33.68 | 63% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=8192 | 7.438 ms | 9.02 | 36.11 | 68% | ok | 1.0x (12%) |
| blas2 | trsv | f32 | n=8192 | 4.182 ms | 16.05 | 32.11 | 60% | ok | 1.0x (12%) |
| blas2 | ger | f64 | n=8192 | 65.419 ms | 2.05 | 16.42 | 31% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 18.756 ms | 114.50 | 1.34 | 23% | ok | 5.7x (72%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 8.792 ms | 244.27 | 1.43 | 24% | ok | 6.3x (79%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 16.102 ms | 66.68 | 1.56 | 13% | ok | 4.0x (50%) |
| blas3 | symm | f64 | 1024x1024x1024 | 23.505 ms | 91.36 | 1.07 | 18% | ok | 4.8x (60%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 15.121 ms | 71.01 | 1.11 | 14% | ok | 4.2x (52%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 27.466 ms | 78.19 | 0.92 | 15% | ok | 4.5x (56%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 36.954 ms | 29.06 | 0.57 | 6% | ok | 2.2x (28%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 37.768 ms | 28.43 | 0.56 | 6% | ok | 2.2x (28%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 26.531 ms | 40.47 | 0.40 | 4% | ok | 1.9x (23%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 34.888 ms | 0.48 | 3.85 | 7% | ok | 1.0x (13%) |
| factor | lu | f64 | 1024x1024 | 71.062 ms | 10.07 | 0.24 | 2% | ok | 4.5x (56%) |
| factor | lu | f32 | 1024x1024 | 67.132 ms | 10.66 | 0.12 | 1% | ok | 4.6x (58%) |
| factor | cholesky | f64 | 1024x1024 | 55.856 ms | 6.41 | 0.30 | 1% | ok | 2.3x (28%) |
| factor | cholesky | f32 | 1024x1024 | 54.811 ms | 6.53 | 0.15 | 1% | ok | 2.3x (28%) |
| factor | ldlt | f64 | 1024x1024 | 126.135 ms | 2.84 | 0.13 | 1% | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 1024x1024 | 93.583 ms | 3.82 | 0.18 | 1% | ok | 1.0x (12%) |
| factor | qr | f64 | 1024x1024 | 1216.984 ms | 1.18 | 0.01 | 0% | ok | 4.9x (61%) |
| factor | lq | f64 | 1024x1024 | 136.067 ms | 10.52 | 0.12 | 2% | ok | 4.8x (60%) |
| factor | hessenberg | f64 | 1024x1024 | 1374.780 ms | 2.60 | 0.01 | 1% | ok | 5.0x (63%) |
| factor | householder | f64 | 1024x1024 | 2.295 ms | 1.83 | 7.31 | 14%\* | ok | 7.6x (95%) |
| factor | inv | f64 | 1024x1024 | 611.930 ms | 3.51 | 0.03 | 1% | ok | 1.4x (17%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 77.041 ms | 9.29 | 0.22 | 2% | ok | 4.2x (53%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 579.213 ms | 0.31 | 0.01 | 0% | ok | 1.6x (20%) |
| eigen | eigen_symmetric | f64 | 512x512 | 3282.056 ms | 0.37 | 0.00 | 0% | ok | 1.1x (14%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 18347.210 ms | 0.07 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 512x512 | 1270.063 ms | 1.06 | 0.00 | 0% | ok | 1.3x (16%) |
| eigen | schur | f64 | 512x512 | 2434.043 ms | 1.38 | 0.00 | 0% | ok | 1.2x (16%) |
| eigen | svd | f64 | 512x512 | 1369.676 ms | 2.16 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | expm | f64 | 512x512 | 176.198 ms | 11.43 | 0.02 | 2% | ok | 1.6x (20%) |
| eigen | sylvester | f64 | 512x512 | 5498.340 ms | 1.46 | 0.00 | 0% | ok | 1.2x (15%) |
| elementwise | exp | f64 | n=33554432 | 150.045 ms | 0.22 | 3.58 | 7% | ok | 0.9x (11%) |
| elementwise | log | f64 | n=33554432 | 212.329 ms | 0.16 | 2.53 | 5% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=33554432 | 197.271 ms | 0.17 | 2.72 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=33554432 | 104.667 ms | 0.32 | 5.13 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=33554432 | 154.587 ms | 0.22 | 3.47 | 7% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 2.015 ms | 5.20 | 54.09 | 101% | ok | 2.0x (25%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 6.401 ms | 1.64 | 17.03 | 32% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 9.019 ms | 18.59 | 39.05 | 73% | ok | 3.5x (44%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 95.228 ms | 0.14 | 0.48 | 1% | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 18.352 ms | 0.55 | 4.57 | 9% | ok | 1.1x (13%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 174.981 ms | 0.61 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 272.157 ms | 0.39 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 98.391 ms | 1.09 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 522.222 ms | 0.41 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 451.156 ms | 0.47 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 215.618 ms | 0.99 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 615.682 ms | 0.04 | 0.00 | 0% | ok | 1.0x (12%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 24.127 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 32.210 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 349.323 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 1.348 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 107.689 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 17.558 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 142.983 ms | 0.75 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 4.126 ms | 0.97 | 3.89 | 7%\* | ok | 1.0x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 41.309 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 49.720 ms | 2.15 | 0.06 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 4.190 ms | 0.96 | 3.83 | 7%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 18.501 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 492.093 ms | 0.43 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.300 ms | 0.26 | 1.05 | 2%\* | ok | 1.0x (12%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 28.103 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 404.813 ms | 0.53 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 15.348 ms | 0.26 | 1.05 | 2%\* | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 12.737 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1070.711 ms | 0.91 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 7.979 ms | 0.94 | 3.76 | 7% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 35.194 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 251.651 ms | 3.85 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 8.134 ms | 0.92 | 3.69 | 7% | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 14.083 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 3597.851 ms | 0.54 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.657 ms | 0.24 | 0.98 | 2% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 19.474 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1883.145 ms | 1.03 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 30.701 ms | 0.24 | 0.98 | 2% | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 297.932 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 407.815 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 212.217 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 137.196 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 160.593 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 132.359 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 396.855 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 252.501 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 159.603 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 157.959 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 144.740 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 157.959 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 193.406 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.225 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 256^2 grid | 14.250 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| pc | ilut | f64 | 256^2 grid | 127.315 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 256^2 grid | 10.600 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 256^2 grid | 10.036 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 256^2 grid | 0.146 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 3.652 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 7.045 ms | 1.49 | 16.66 | 31% | ok | 0.9x (11%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 7.835 ms | 1.34 | 14.98 | 28% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 9.112 ms | 1.15 | 12.88 | 24% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 91.243 ms | 1.70 | 12.81 | 24% | ok | 1.1x (14%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 189.895 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 188.896 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 825.380 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 32.421 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=33554432 | 8.712 ms | 7.70 | 30.81 | 58% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 23.096 ms | 92.98 | 0.73 | 18% | ok | 6.6x (82%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 2.425 ms | 27.68 | 29.41 | 55% | ok | 1.0x (13%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 5.666 ms | 11.84 | 47.38 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 2.357 ms | 28.47 | 56.94 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 1.787 ms | 37.55 | 37.55 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 1.600 ms | 41.94 | 41.94 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 20.625 ms | 104.12 | 0.61 | n/a | ok | 7.2x (90%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 23.136 ms | 92.82 | 0.36 | n/a | ok | 7.5x (94%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 23.389 ms | 91.82 | 0.27 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 5.265 ms | 407.89 | 1.19 | n/a | ok | 6.7x (84%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 5.272 ms | 407.38 | 1.19 | n/a | ok | 6.7x (84%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

