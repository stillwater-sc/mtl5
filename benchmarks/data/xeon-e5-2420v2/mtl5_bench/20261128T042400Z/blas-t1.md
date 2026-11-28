## xeon-e5-2420v2 -- blas, 1 thread, class large

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 20.9 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 20.9 GFLOP/s |
| ... this build, scalar | f64 0.1, f32 0.7 GFLOP/s |
| bandwidth | read 10.2, update 10.2, triad 8.1 GB/s (ceiling 10.2) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 24% | 93% | 0 |
| blas2 | 10 | 93% | 97% | 0 |
| blas3 | 10 | 167% | 185% | 0 |
| factor | 12 | 47% | 103% | 0 |
| eigen | 8 | 49% | 158% | 0 |
| elementwise | 5 | 9% | 14% | 0 |
| sparse | 5 | 53% | 103% | 0 |
| direct | 19 | 4% | 17% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 16% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 77% | 86% | 0 |
| mg | 1 | 64% | 64% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 23% | 72% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=33554432 | 746.803 ms | 0.09 | 0.72 | 7% | ok |
| blas1 | dot | f32 | n=33554432 | 111.350 ms | 0.60 | 2.41 | 24% | ok |
| blas1 | axpy | f64 | n=33554432 | 850.230 ms | 0.08 | 0.95 | 9% | ok |
| blas1 | axpy | f32 | n=33554432 | 369.198 ms | 0.18 | 1.09 | 11% | ok |
| blas1 | scale | f64 | n=33554432 | 56.448 ms | 0.59 | 9.51 | 93% | ok |
| blas1 | two_norm | f64 | n=33554432 | 740.328 ms | 0.09 | 0.36 | 4% | ok |
| blas1 | one_norm | f64 | n=33554432 | 39.204 ms | 0.86 | 6.85 | 67% | ok |
| blas1 | infinity_norm | f64 | n=33554432 | 52.440 ms | 0.64 | 5.12 | 50% | ok |
| blas1 | sum | f64 | n=33554432 | 39.520 ms | 0.85 | 6.79 | 67% | ok |
| blas1 | dot_k3 | f64 | n=33554432 | 1135.027 ms | 0.65 | 0.47 | 6% | ok |
| blas2 | gemv | f64 | n=8192 | 58.649 ms | 2.29 | 9.16 | 90% | ok |
| blas2 | gemv | f32 | n=8192 | 27.394 ms | 4.90 | 9.80 | 96% | ok |
| blas2 | gemv_colmajor | f64 | n=8192 | 56.062 ms | 2.39 | 9.58 | 94% | ok |
| blas2 | gemv_colmajor | f32 | n=8192 | 28.426 ms | 4.72 | 9.45 | 93% | ok |
| blas2 | symv | f64 | n=8192 | 29.854 ms | 4.50 | 9.00 | 88% | ok |
| blas2 | trmv | f64 | n=8192 | 29.861 ms | 2.25 | 8.99 | 88% | ok |
| blas2 | trmv | f32 | n=8192 | 13.925 ms | 4.82 | 9.64 | 94% | ok |
| blas2 | trsv | f64 | n=8192 | 31.590 ms | 2.12 | 8.50 | 83% | ok |
| blas2 | trsv | f32 | n=8192 | 13.570 ms | 4.95 | 9.90 | 97% | ok |
| blas2 | ger | f64 | n=8192 | 121.028 ms | 1.11 | 8.87 | 87% | ok |
| blas3 | gemm | f64 | 1024x1024x1024 | 110.414 ms | 19.45 | 0.23 | 185% | ok |
| blas3 | gemm | f32 | 1024x1024x1024 | 67.906 ms | 31.62 | 0.19 | 151% | ok |
| blas3 | gemmt | f64 | 1024x1024x1024 | 65.051 ms | 16.51 | 0.39 | 157% | ok |
| blas3 | symm | f64 | 1024x1024x1024 | 111.567 ms | 19.25 | 0.23 | 183% | ok |
| blas3 | syrk | f64 | 1024x1024x1024 | 60.937 ms | 17.62 | 0.28 | 167% | ok |
| blas3 | syr2k | f64 | 1024x1024x1024 | 117.274 ms | 18.31 | 0.21 | 174% | ok |
| blas3 | trmm | f64 | 1024x1024x1024 | 57.315 ms | 18.73 | 0.37 | 178% | ok |
| blas3 | trsm | f64 | 1024x1024x1024 | 61.895 ms | 17.35 | 0.34 | 165% | ok |
| blas3 | trsm | f32 | 1024x1024x1024 | 39.333 ms | 27.30 | 0.27 | 130% | ok |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 130.066 ms | 0.13 | 1.03 | 10% | ok |
| factor | lu | f64 | 1024x1024 | 65.815 ms | 10.88 | 0.25 | 103% | ok |
| factor | lu | f32 | 1024x1024 | 50.992 ms | 14.04 | 0.16 | 67% | ok |
| factor | cholesky | f64 | 1024x1024 | 44.757 ms | 8.00 | 0.37 | 76% | ok |
| factor | cholesky | f32 | 1024x1024 | 36.157 ms | 9.90 | 0.23 | 47% | ok |
| factor | ldlt | f64 | 1024x1024 | 212.055 ms | 1.69 | 0.08 | 16% | ok |
| factor | ldlt_bk | f64 | 1024x1024 | 106.184 ms | 3.37 | 0.16 | 32% | ok |
| factor | qr | f64 | 1024x1024 | 133.492 ms | 10.72 | 0.13 | 102% | ok |
| factor | lq | f64 | 1024x1024 | 584.118 ms | 2.45 | 0.03 | 23% | ok |
| factor | hessenberg | f64 | 1024x1024 | 7775.902 ms | 0.46 | 0.00 | 4% | ok |
| factor | householder | f64 | 1024x1024 | 20.362 ms | 0.21 | 0.82 | 8% | ok |
| factor | inv | f64 | 1024x1024 | 993.993 ms | 2.16 | 0.02 | 21% | ok |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 75.395 ms | 9.49 | 0.22 | 90% | ok |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 34.623 ms | 5.17 | 0.12 | 49% | ok |
| eigen | eigen_symmetric | f64 | 512x512 | 2114.704 ms | 0.57 | 0.00 | 5% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 14705.132 ms | 0.08 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 512x512 | 399.752 ms | 3.36 | 0.01 | 32% | ok |
| eigen | schur | f64 | 512x512 | 496.583 ms | 6.76 | 0.01 | 64% | ok |
| eigen | svd | f64 | 512x512 | 177.160 ms | 16.67 | 0.02 | 158% | ok |
| eigen | expm | f64 | 512x512 | 274.805 ms | 7.33 | 0.02 | 70% | ok |
| eigen | sylvester | f64 | 512x512 | 1826.155 ms | 4.41 | 0.00 | 42% | ok |
| elementwise | exp | f64 | n=33554432 | 608.988 ms | 0.06 | 0.88 | 9% | ok |
| elementwise | log | f64 | n=33554432 | 561.011 ms | 0.06 | 0.96 | 9% | ok |
| elementwise | sin | f64 | n=33554432 | 684.189 ms | 0.05 | 0.78 | 8% | ok |
| elementwise | sqrt | f64 | n=33554432 | 362.839 ms | 0.09 | 1.48 | 14% | ok |
| elementwise | erf | f64 | n=33554432 | 9891.576 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 10.380 ms | 1.01 | 10.50 | 103% | ok |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 12.667 ms | 0.83 | 8.60 | 84% | ok |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 65.146 ms | 2.57 | 5.41 | 53% | ok |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 99.108 ms | 0.13 | 0.46 | 5% | ok |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 38.625 ms | 0.26 | 2.17 | 21% | ok |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 304.451 ms | 0.35 | 0.01 | 3% | ok |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 555.044 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 156.032 ms | 0.69 | 0.02 | 7% | ok |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 899.845 ms | 0.24 | 0.00 | 2% | ok |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 878.172 ms | 0.24 | 0.00 | 2% | ok |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 355.488 ms | 0.60 | 0.01 | 6% | ok |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 1113.897 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 44.128 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 63.830 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 580.487 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 2.452 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 251.339 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 26.572 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 268.683 ms | 0.40 | 0.01 | 4% | ok |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 9.175 ms | 0.44 | 1.75 | 17% | ok |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 54.516 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 92.383 ms | 1.16 | 0.03 | 11% | ok |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 8.995 ms | 0.45 | 1.79 | 17% | ok |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 26.720 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 824.300 ms | 0.26 | 0.00 | 2% | ok |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.325 ms | 0.13 | 0.53 | 5% | ok |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 40.830 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 813.055 ms | 0.26 | 0.00 | 3% | ok |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.392 ms | 0.13 | 0.53 | 5% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 23.721 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1934.754 ms | 0.50 | 0.00 | 5% | ok |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 19.675 ms | 0.38 | 1.52 | 15% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 61.886 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 589.552 ms | 1.64 | 0.00 | 16% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 19.426 ms | 0.39 | 1.54 | 15% | ok |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 24.276 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 5723.256 ms | 0.34 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 88.224 ms | 0.08 | 0.34 | 3% | ok |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 34.064 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 4935.584 ms | 0.39 | 0.00 | 4% | ok |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 87.484 ms | 0.09 | 0.34 | 3% | ok |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 1771.717 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 3113.870 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 778.800 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 897.534 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 940.197 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 584.436 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 4859.147 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 1210.390 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 724.644 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 1406.290 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 1300.515 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 1494.366 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 2040.986 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.689 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 256^2 grid | 17.989 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 256^2 grid | 149.801 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 256^2 grid | 12.053 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 256^2 grid | 11.312 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 256^2 grid | 0.378 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 13.974 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 14.890 ms | 0.70 | 7.88 | 77% | ok |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 13.422 ms | 0.78 | 8.75 | 86% | ok |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 16.005 ms | 0.65 | 7.33 | 72% | ok |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 179.338 ms | 0.87 | 6.52 | 64% | ok |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 563.150 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 590.117 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 1416.970 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 174.258 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=33554432 | 752.568 ms | 0.09 | 0.36 | 3% | ok |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 906.126 ms | 2.37 | 0.02 | 23% | ok |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 9.677 ms | 6.93 | 7.37 | 72% | ok |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 26.101 ms | 2.57 | 10.28 | n/a | ok |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.658 ms | 5.30 | 10.60 | n/a | ok |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 20.443 ms | 3.28 | 3.28 | n/a | ok |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 20.434 ms | 3.28 | 3.28 | n/a | ok |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 454.842 ms | 4.72 | 0.03 | n/a | ok |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 362.504 ms | 5.92 | 0.02 | n/a | ok |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 188.808 ms | 11.37 | 0.03 | n/a | ok |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 188.472 ms | 11.39 | 0.03 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 566.062 ms | 3.79 | 0.01 | n/a | ok |

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

