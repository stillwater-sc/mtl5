## i7-12700k -- blas, 1 thread, class large

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.3 GFLOP/s |
| bandwidth | read 33.2, update 38.6, triad 32.7 GB/s (ceiling 38.6) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 43% | 96% | 0 |
| blas2 | 10 | 95% | 101% | 0 |
| blas3 | 10 | 88% | 96% | 0 |
| factor | 12 | 29% | 50% | 0 |
| eigen | 8 | 25% | 84% | 0 |
| elementwise | 5 | 7% | 11% | 0 |
| sparse | 5 | 47% | 77% | 0 |
| direct | 31 | 5% | 23% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 21 | 12% | 67% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 61% | 69% | 0 |
| mg | 1 | 44% | 44% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 35% | 69% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=33554432 | 23.026 ms | 2.91 | 23.32 | 60% | ok |
| blas1 | dot | f32 | n=33554432 | 17.353 ms | 3.87 | 15.47 | 40% | ok |
| blas1 | axpy | f64 | n=33554432 | 26.007 ms | 2.58 | 30.96 | 80% | ok |
| blas1 | axpy | f32 | n=33554432 | 18.934 ms | 3.54 | 21.27 | 55% | ok |
| blas1 | scale | f64 | n=33554432 | 14.451 ms | 2.32 | 37.15 | 96% | ok |
| blas1 | two_norm | f64 | n=33554432 | 20.042 ms | 3.35 | 13.39 | 35% | ok |
| blas1 | one_norm | f64 | n=33554432 | 16.443 ms | 2.04 | 16.33 | 42% | ok |
| blas1 | infinity_norm | f64 | n=33554432 | 28.269 ms | 1.19 | 9.50 | 25% | ok |
| blas1 | sum | f64 | n=33554432 | 16.305 ms | 2.06 | 16.46 | 43% | ok |
| blas1 | dot_k3 | f64 | n=33554432 | 321.931 ms | 2.29 | 1.67 | 4% | ok |
| blas2 | gemv | f64 | n=8192 | 14.385 ms | 9.33 | 37.33 | 97% | ok |
| blas2 | gemv | f32 | n=8192 | 7.191 ms | 18.66 | 37.34 | 97% | ok |
| blas2 | gemv_colmajor | f64 | n=8192 | 14.972 ms | 8.96 | 35.87 | 93% | ok |
| blas2 | gemv_colmajor | f32 | n=8192 | 7.090 ms | 18.93 | 37.87 | 98% | ok |
| blas2 | symv | f64 | n=8192 | 7.672 ms | 17.49 | 35.00 | 91% | ok |
| blas2 | trmv | f64 | n=8192 | 7.524 ms | 8.92 | 35.69 | 92% | ok |
| blas2 | trmv | f32 | n=8192 | 3.951 ms | 16.98 | 33.98 | 88% | ok |
| blas2 | trsv | f64 | n=8192 | 7.306 ms | 9.19 | 36.76 | 95% | ok |
| blas2 | trsv | f32 | n=8192 | 3.645 ms | 18.41 | 36.84 | 95% | ok |
| blas2 | ger | f64 | n=8192 | 27.479 ms | 4.88 | 39.08 | 101% | ok |
| blas3 | gemm | f64 | 1024x1024x1024 | 29.121 ms | 73.74 | 0.86 | 94% | ok |
| blas3 | gemm | f32 | 1024x1024x1024 | 14.337 ms | 149.79 | 0.88 | 96% | ok |
| blas3 | gemmt | f64 | 1024x1024x1024 | 17.785 ms | 60.37 | 1.41 | 77% | ok |
| blas3 | symm | f64 | 1024x1024x1024 | 29.297 ms | 73.30 | 0.86 | 94% | ok |
| blas3 | syrk | f64 | 1024x1024x1024 | 18.619 ms | 57.67 | 0.90 | 74% | ok |
| blas3 | syr2k | f64 | 1024x1024x1024 | 35.913 ms | 59.80 | 0.70 | 76% | ok |
| blas3 | trmm | f64 | 1024x1024x1024 | 15.543 ms | 69.08 | 1.35 | 88% | ok |
| blas3 | trsm | f64 | 1024x1024x1024 | 15.637 ms | 68.67 | 1.34 | 88% | ok |
| blas3 | trsm | f32 | 1024x1024x1024 | 7.378 ms | 145.54 | 1.42 | 93% | ok |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 44.274 ms | 0.38 | 3.03 | 8% | ok |
| factor | lu | f64 | 1024x1024 | 18.553 ms | 38.58 | 0.90 | 49% | ok |
| factor | lu | f32 | 1024x1024 | 12.286 ms | 58.26 | 0.68 | 37% | ok |
| factor | cholesky | f64 | 1024x1024 | 11.530 ms | 31.04 | 1.46 | 40% | ok |
| factor | cholesky | f32 | 1024x1024 | 8.017 ms | 44.65 | 1.05 | 29% | ok |
| factor | ldlt | f64 | 1024x1024 | 69.337 ms | 5.16 | 0.24 | 7% | ok |
| factor | ldlt_bk | f64 | 1024x1024 | 43.147 ms | 8.30 | 0.39 | 11% | ok |
| factor | qr | f64 | 1024x1024 | 36.231 ms | 39.51 | 0.46 | 50% | ok |
| factor | lq | f64 | 1024x1024 | 218.722 ms | 6.55 | 0.08 | 8% | ok |
| factor | hessenberg | f64 | 1024x1024 | 2653.826 ms | 1.35 | 0.01 | 2% | ok |
| factor | householder | f64 | 1024x1024 | 6.883 ms | 0.61 | 2.44 | 6%\* | ok |
| factor | inv | f64 | 1024x1024 | 356.132 ms | 6.03 | 0.05 | 8% | ok |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 21.708 ms | 32.98 | 0.77 | 42% | ok |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 9.163 ms | 19.53 | 0.46 | 25% | ok |
| eigen | eigen_symmetric | f64 | 512x512 | 971.699 ms | 1.24 | 0.00 | 2% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 12647.648 ms | 0.10 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 512x512 | 117.343 ms | 11.44 | 0.04 | 15% | ok |
| eigen | schur | f64 | 512x512 | 146.124 ms | 22.96 | 0.03 | 29% | ok |
| eigen | svd | f64 | 512x512 | 44.649 ms | 66.13 | 0.09 | 84% | ok |
| eigen | expm | f64 | 512x512 | 93.534 ms | 21.52 | 0.04 | 27% | ok |
| eigen | sylvester | f64 | 512x512 | 452.802 ms | 17.78 | 0.01 | 23% | ok |
| elementwise | exp | f64 | n=33554432 | 163.912 ms | 0.20 | 3.28 | 8% | ok |
| elementwise | log | f64 | n=33554432 | 191.336 ms | 0.18 | 2.81 | 7% | ok |
| elementwise | sin | f64 | n=33554432 | 247.415 ms | 0.14 | 2.17 | 6% | ok |
| elementwise | sqrt | f64 | n=33554432 | 122.345 ms | 0.27 | 4.39 | 11% | ok |
| elementwise | erf | f64 | n=33554432 | 816.323 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 3.666 ms | 2.86 | 29.73 | 77% | ok |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 4.313 ms | 2.43 | 25.27 | 65% | ok |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 19.394 ms | 8.64 | 18.16 | 47% | ok |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 30.221 ms | 0.43 | 1.52 | 4% | ok |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 14.500 ms | 0.70 | 5.78 | 15% | ok |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 107.678 ms | 0.99 | 0.03 | 1% | ok |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 194.713 ms | 0.55 | 0.02 | 1% | ok |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 54.009 ms | 1.98 | 0.05 | 3% | ok |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 334.337 ms | 0.64 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 343.078 ms | 0.62 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 146.733 ms | 1.46 | 0.02 | 2% | ok |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 410.764 ms | 0.05 | 0.00 | 0% | ok |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 18.559 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 27.772 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 295.846 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 0.986 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 103.798 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 9.716 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 101.596 ms | 1.05 | 0.03 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 3.656 ms | 1.10 | 4.39 | 11%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 21.676 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 32.256 ms | 3.32 | 0.09 | 4% | ok |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 2.625 ms | 1.53 | 6.12 | 16%\* | ok |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 8.623 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 300.526 ms | 0.71 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 9.140 ms | 0.44 | 1.76 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 14.433 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 283.647 ms | 0.75 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 9.265 ms | 0.43 | 1.73 | 4%\* | ok |
| direct | ref_cholmod | f64 | 192^2 grid, nnz 183552 | 25.067 ms | 4.27 | 0.12 | 5% | ok |
| direct | ref_cholmod.analyze | f64 | 192^2 grid, nnz 183552 | 6.893 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 192^2 grid, nnz 183552 | 15.823 ms | 6.76 | 0.19 | 9% | ok |
| direct | ref_cholmod.solve | f64 | 192^2 grid, nnz 183552 | 1.805 ms | 2.22 | 8.90 | 23%\* | ok |
| direct | ref_pardiso_spd | f64 | 192^2 grid, nnz 183552 | 20.746 ms | 5.16 | 0.14 | 7% | ok |
| direct | ref_pardiso_spd.analyze | f64 | 192^2 grid, nnz 183552 | 8.273 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_pardiso_spd.factor | f64 | 192^2 grid, nnz 183552 | 9.272 ms | 11.54 | 0.32 | 15% | ok |
| direct | ref_pardiso_spd.solve | f64 | 192^2 grid, nnz 183552 | 2.116 ms | 1.90 | 7.59 | 20%\* | ok |
| direct | ref_pardiso_lu | f64 | 192^2 grid, nnz 183552 | 48.149 ms | 4.44 | 0.06 | 6% | ok |
| direct | ref_pardiso_lu.analyze | f64 | 192^2 grid, nnz 183552 | 26.388 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_pardiso_lu.factor | f64 | 192^2 grid, nnz 183552 | 17.985 ms | 11.90 | 0.16 | 15% | ok |
| direct | ref_pardiso_lu.solve | f64 | 192^2 grid, nnz 183552 | 4.478 ms | 0.90 | 3.59 | 9%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 9.677 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1036.168 ms | 0.94 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 6.489 ms | 1.16 | 4.62 | 12% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 21.702 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 199.683 ms | 4.85 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 7.230 ms | 1.04 | 4.15 | 11% | ok |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 9.765 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 2105.995 ms | 0.92 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 20.453 ms | 0.37 | 1.47 | 4% | ok |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 13.365 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1395.580 ms | 1.39 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.572 ms | 0.38 | 1.53 | 4% | ok |
| direct3d | ref_cholmod.analyze | f64 | 24^3 grid, nnz 93312 | 6.627 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 24^3 grid, nnz 93312 | 29.257 ms | 33.13 | 0.05 | 42% | ok |
| direct3d | ref_cholmod.solve | f64 | 24^3 grid, nnz 93312 | 1.165 ms | 6.44 | 25.75 | 67% | ok |
| direct3d | ref_pardiso_spd.analyze | f64 | 24^3 grid, nnz 93312 | 14.995 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_pardiso_spd.factor | f64 | 24^3 grid, nnz 93312 | 29.411 ms | 32.95 | 0.05 | 42% | ok |
| direct3d | ref_pardiso_spd.solve | f64 | 24^3 grid, nnz 93312 | 2.063 ms | 3.63 | 14.54 | 38% | ok |
| direct3d | ref_pardiso_lu.analyze | f64 | 24^3 grid, nnz 93312 | 17.193 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_pardiso_lu.factor | f64 | 24^3 grid, nnz 93312 | 59.441 ms | 32.61 | 0.03 | 42% | ok |
| direct3d | ref_pardiso_lu.solve | f64 | 24^3 grid, nnz 93312 | 2.948 ms | 2.54 | 10.18 | 26% | ok |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 253.025 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 285.492 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 144.575 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 116.616 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 122.825 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 108.651 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 273.090 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 179.990 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 120.725 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 135.287 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 119.305 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 125.500 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 152.365 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.164 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 256^2 grid | 6.137 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 256^2 grid | 49.898 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 256^2 grid | 4.393 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 256^2 grid | 4.122 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 256^2 grid | 0.115 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 4.579 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 4.438 ms | 2.36 | 26.45 | 69% | ok |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 4.991 ms | 2.10 | 23.52 | 61% | ok |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 6.345 ms | 1.65 | 18.50 | 48% | ok |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 68.386 ms | 2.27 | 17.09 | 44% | ok |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 299.202 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 300.033 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 503.988 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 20.910 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=33554432 | 19.603 ms | 3.42 | 13.69 | 35% | ok |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 312.635 ms | 6.87 | 0.05 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 2.677 ms | 25.07 | 26.64 | 69% | ok |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 9.482 ms | 7.08 | 28.31 | n/a | ok |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 4.695 ms | 14.29 | 28.58 | n/a | ok |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 6.995 ms | 9.59 | 9.59 | n/a | ok |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 7.005 ms | 9.58 | 9.58 | n/a | ok |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 139.768 ms | 15.36 | 0.09 | n/a | ok |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 166.841 ms | 12.87 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 81.664 ms | 26.30 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 81.541 ms | 26.34 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 214.598 ms | 10.01 | 0.03 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

