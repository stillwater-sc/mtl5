## i7-12700k -- blas, 1 thread, class large

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.5 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.5 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.3 GFLOP/s |
| bandwidth | read 33.1, update 38.7, triad 32.5 GB/s (ceiling 38.7) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 43% | 96% | 0 |
| blas2 | 10 | 95% | 101% | 0 |
| blas3 | 10 | 88% | 96% | 0 |
| factor | 12 | 29% | 51% | 0 |
| eigen | 8 | 25% | 85% | 0 |
| elementwise | 5 | 7% | 11% | 0 |
| sparse | 5 | 47% | 79% | 0 |
| direct | 31 | 5% | 28% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 21 | 16% | 67% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 61% | 69% | 0 |
| mg | 1 | 44% | 44% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 35% | 69% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=33554432 | 23.295 ms | 2.88 | 23.05 | 60% | ok |
| blas1 | dot | f32 | n=33554432 | 17.752 ms | 3.78 | 15.12 | 39% | ok |
| blas1 | axpy | f64 | n=33554432 | 25.972 ms | 2.58 | 31.01 | 80% | ok |
| blas1 | axpy | f32 | n=33554432 | 18.855 ms | 3.56 | 21.36 | 55% | ok |
| blas1 | scale | f64 | n=33554432 | 14.457 ms | 2.32 | 37.14 | 96% | ok |
| blas1 | two_norm | f64 | n=33554432 | 20.317 ms | 3.30 | 13.21 | 34% | ok |
| blas1 | one_norm | f64 | n=33554432 | 16.476 ms | 2.04 | 16.29 | 42% | ok |
| blas1 | infinity_norm | f64 | n=33554432 | 28.288 ms | 1.19 | 9.49 | 25% | ok |
| blas1 | sum | f64 | n=33554432 | 16.327 ms | 2.06 | 16.44 | 43% | ok |
| blas1 | dot_k3 | f64 | n=33554432 | 322.552 ms | 2.29 | 1.66 | 4% | ok |
| blas2 | gemv | f64 | n=8192 | 14.362 ms | 9.35 | 37.39 | 97% | ok |
| blas2 | gemv | f32 | n=8192 | 7.195 ms | 18.65 | 37.32 | 97% | ok |
| blas2 | gemv_colmajor | f64 | n=8192 | 14.967 ms | 8.97 | 35.88 | 93% | ok |
| blas2 | gemv_colmajor | f32 | n=8192 | 7.098 ms | 18.91 | 37.83 | 98% | ok |
| blas2 | symv | f64 | n=8192 | 7.688 ms | 17.46 | 34.93 | 90% | ok |
| blas2 | trmv | f64 | n=8192 | 7.516 ms | 8.93 | 35.73 | 92% | ok |
| blas2 | trmv | f32 | n=8192 | 3.952 ms | 16.98 | 33.98 | 88% | ok |
| blas2 | trsv | f64 | n=8192 | 7.320 ms | 9.17 | 36.69 | 95% | ok |
| blas2 | trsv | f32 | n=8192 | 3.648 ms | 18.39 | 36.81 | 95% | ok |
| blas2 | ger | f64 | n=8192 | 27.436 ms | 4.89 | 39.14 | 101% | ok |
| blas3 | gemm | f64 | 1024x1024x1024 | 29.159 ms | 73.65 | 0.86 | 94% | ok |
| blas3 | gemm | f32 | 1024x1024x1024 | 14.346 ms | 149.69 | 0.88 | 96% | ok |
| blas3 | gemmt | f64 | 1024x1024x1024 | 17.777 ms | 60.40 | 1.42 | 77% | ok |
| blas3 | symm | f64 | 1024x1024x1024 | 29.303 ms | 73.29 | 0.86 | 94% | ok |
| blas3 | syrk | f64 | 1024x1024x1024 | 18.600 ms | 57.73 | 0.90 | 74% | ok |
| blas3 | syr2k | f64 | 1024x1024x1024 | 35.891 ms | 59.83 | 0.70 | 76% | ok |
| blas3 | trmm | f64 | 1024x1024x1024 | 15.510 ms | 69.23 | 1.35 | 88% | ok |
| blas3 | trsm | f64 | 1024x1024x1024 | 15.637 ms | 68.67 | 1.34 | 88% | ok |
| blas3 | trsm | f32 | 1024x1024x1024 | 7.378 ms | 145.54 | 1.42 | 93% | ok |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 44.394 ms | 0.38 | 3.02 | 8% | ok |
| factor | lu | f64 | 1024x1024 | 18.592 ms | 38.50 | 0.90 | 49% | ok |
| factor | lu | f32 | 1024x1024 | 12.264 ms | 58.37 | 0.68 | 37% | ok |
| factor | cholesky | f64 | 1024x1024 | 11.392 ms | 31.42 | 1.47 | 40% | ok |
| factor | cholesky | f32 | 1024x1024 | 7.992 ms | 44.78 | 1.05 | 29% | ok |
| factor | ldlt | f64 | 1024x1024 | 69.347 ms | 5.16 | 0.24 | 7% | ok |
| factor | ldlt_bk | f64 | 1024x1024 | 42.486 ms | 8.42 | 0.39 | 11% | ok |
| factor | qr | f64 | 1024x1024 | 36.106 ms | 39.65 | 0.46 | 51% | ok |
| factor | lq | f64 | 1024x1024 | 218.406 ms | 6.56 | 0.08 | 8% | ok |
| factor | hessenberg | f64 | 1024x1024 | 2615.883 ms | 1.37 | 0.01 | 2% | ok |
| factor | householder | f64 | 1024x1024 | 6.833 ms | 0.61 | 2.46 | 6%\* | ok |
| factor | inv | f64 | 1024x1024 | 351.581 ms | 6.11 | 0.05 | 8% | ok |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 21.477 ms | 33.33 | 0.78 | 43% | ok |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 9.093 ms | 19.68 | 0.46 | 25% | ok |
| eigen | eigen_symmetric | f64 | 512x512 | 972.787 ms | 1.24 | 0.00 | 2% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 13090.976 ms | 0.09 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 512x512 | 117.687 ms | 11.40 | 0.04 | 15% | ok |
| eigen | schur | f64 | 512x512 | 140.828 ms | 23.83 | 0.03 | 30% | ok |
| eigen | svd | f64 | 512x512 | 44.229 ms | 66.76 | 0.09 | 85% | ok |
| eigen | expm | f64 | 512x512 | 93.421 ms | 21.55 | 0.04 | 28% | ok |
| eigen | sylvester | f64 | 512x512 | 453.770 ms | 17.75 | 0.01 | 23% | ok |
| elementwise | exp | f64 | n=33554432 | 163.655 ms | 0.21 | 3.28 | 8% | ok |
| elementwise | log | f64 | n=33554432 | 190.661 ms | 0.18 | 2.82 | 7% | ok |
| elementwise | sin | f64 | n=33554432 | 248.279 ms | 0.14 | 2.16 | 6% | ok |
| elementwise | sqrt | f64 | n=33554432 | 121.927 ms | 0.28 | 4.40 | 11% | ok |
| elementwise | erf | f64 | n=33554432 | 817.840 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 3.572 ms | 2.93 | 30.51 | 79% | ok |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 4.157 ms | 2.52 | 26.22 | 68% | ok |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 19.292 ms | 8.69 | 18.26 | 47% | ok |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 29.800 ms | 0.44 | 1.55 | 4% | ok |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 14.443 ms | 0.70 | 5.80 | 15% | ok |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 103.470 ms | 1.03 | 0.03 | 1% | ok |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 188.094 ms | 0.57 | 0.02 | 1% | ok |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 50.161 ms | 2.13 | 0.06 | 3% | ok |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 318.478 ms | 0.67 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 307.096 ms | 0.70 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 130.947 ms | 1.63 | 0.02 | 2% | ok |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 369.986 ms | 0.06 | 0.00 | 0% | ok |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 17.118 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 23.969 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 265.016 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 0.913 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 91.934 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 8.516 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 93.733 ms | 1.14 | 0.03 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 2.443 ms | 1.64 | 6.57 | 17%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 18.157 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 29.103 ms | 3.68 | 0.10 | 5% | ok |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 2.294 ms | 1.75 | 7.00 | 18%\* | ok |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 8.191 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 293.405 ms | 0.73 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.782 ms | 0.46 | 1.83 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 13.381 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 273.942 ms | 0.78 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.749 ms | 0.46 | 1.84 | 5%\* | ok |
| direct | ref_cholmod | f64 | 192^2 grid, nnz 183552 | 24.236 ms | 4.41 | 0.12 | 6% | ok |
| direct | ref_cholmod.analyze | f64 | 192^2 grid, nnz 183552 | 6.873 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 192^2 grid, nnz 183552 | 15.355 ms | 6.97 | 0.19 | 9% | ok |
| direct | ref_cholmod.solve | f64 | 192^2 grid, nnz 183552 | 1.479 ms | 2.71 | 10.86 | 28%\* | ok |
| direct | ref_pardiso_spd | f64 | 192^2 grid, nnz 183552 | 18.932 ms | 5.65 | 0.16 | 7% | ok |
| direct | ref_pardiso_spd.analyze | f64 | 192^2 grid, nnz 183552 | 7.849 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_pardiso_spd.factor | f64 | 192^2 grid, nnz 183552 | 8.729 ms | 12.26 | 0.34 | 16% | ok |
| direct | ref_pardiso_spd.solve | f64 | 192^2 grid, nnz 183552 | 2.066 ms | 1.94 | 7.77 | 20%\* | ok |
| direct | ref_pardiso_lu | f64 | 192^2 grid, nnz 183552 | 44.269 ms | 4.83 | 0.07 | 6% | ok |
| direct | ref_pardiso_lu.analyze | f64 | 192^2 grid, nnz 183552 | 25.931 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_pardiso_lu.factor | f64 | 192^2 grid, nnz 183552 | 14.758 ms | 14.50 | 0.20 | 19% | ok |
| direct | ref_pardiso_lu.solve | f64 | 192^2 grid, nnz 183552 | 2.813 ms | 1.43 | 5.71 | 15%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 8.981 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 619.948 ms | 1.56 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 4.839 ms | 1.55 | 6.20 | 16% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 19.235 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 182.898 ms | 5.30 | 0.01 | 7% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 4.792 ms | 1.56 | 6.26 | 16% | ok |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 8.897 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 1937.229 ms | 1.00 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.478 ms | 0.38 | 1.54 | 4% | ok |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 13.111 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1382.638 ms | 1.40 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.461 ms | 0.39 | 1.54 | 4% | ok |
| direct3d | ref_cholmod.analyze | f64 | 24^3 grid, nnz 93312 | 6.593 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 24^3 grid, nnz 93312 | 29.117 ms | 33.29 | 0.05 | 43% | ok |
| direct3d | ref_cholmod.solve | f64 | 24^3 grid, nnz 93312 | 1.152 ms | 6.51 | 26.05 | 67% | ok |
| direct3d | ref_pardiso_spd.analyze | f64 | 24^3 grid, nnz 93312 | 14.987 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_pardiso_spd.factor | f64 | 24^3 grid, nnz 93312 | 29.246 ms | 33.14 | 0.05 | 42% | ok |
| direct3d | ref_pardiso_spd.solve | f64 | 24^3 grid, nnz 93312 | 2.060 ms | 3.64 | 14.56 | 38% | ok |
| direct3d | ref_pardiso_lu.analyze | f64 | 24^3 grid, nnz 93312 | 17.159 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_pardiso_lu.factor | f64 | 24^3 grid, nnz 93312 | 58.848 ms | 32.94 | 0.03 | 42% | ok |
| direct3d | ref_pardiso_lu.solve | f64 | 24^3 grid, nnz 93312 | 2.936 ms | 2.55 | 10.22 | 26% | ok |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 252.217 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 285.018 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 144.278 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 116.373 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 122.800 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 108.332 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 269.420 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 180.331 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 120.331 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 134.279 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 119.325 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 125.755 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 152.401 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.171 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 256^2 grid | 6.388 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 256^2 grid | 50.081 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 256^2 grid | 4.400 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 256^2 grid | 4.116 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 256^2 grid | 0.117 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 4.502 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 4.429 ms | 2.37 | 26.50 | 69% | ok |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 4.992 ms | 2.10 | 23.51 | 61% | ok |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 6.345 ms | 1.65 | 18.50 | 48% | ok |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 68.275 ms | 2.28 | 17.11 | 44% | ok |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 296.777 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 296.665 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 508.871 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 20.774 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=33554432 | 19.577 ms | 3.43 | 13.71 | 35% | ok |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 312.128 ms | 6.88 | 0.05 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 2.672 ms | 25.12 | 26.69 | 69% | ok |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 9.519 ms | 7.05 | 28.20 | n/a | ok |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 4.667 ms | 14.38 | 28.76 | n/a | ok |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 6.999 ms | 9.59 | 9.59 | n/a | ok |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 6.995 ms | 9.59 | 9.59 | n/a | ok |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 138.310 ms | 15.53 | 0.09 | n/a | ok |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 166.255 ms | 12.92 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 81.573 ms | 26.33 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 81.555 ms | 26.33 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 214.752 ms | 10.00 | 0.03 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

