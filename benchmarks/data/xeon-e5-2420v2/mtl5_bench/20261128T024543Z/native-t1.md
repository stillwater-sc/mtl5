## xeon-e5-2420v2 -- native, 1 thread, class large

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 21.0 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 21.0 GFLOP/s |
| ... this build, scalar | f64 0.1, f32 0.7 GFLOP/s |
| bandwidth | read 10.1, update 10.2, triad 8.0 GB/s (ceiling 10.2) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 24% | 94% | 0 |
| blas2 | 10 | 90% | 98% | 0 |
| blas3 | 10 | 37% | 40% | 0 |
| factor | 12 | 17% | 46% | 0 |
| eigen | 8 | 11% | 26% | 0 |
| elementwise | 5 | 8% | 15% | 0 |
| sparse | 5 | 57% | 103% | 0 |
| direct | 19 | 4% | 17% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 15% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 78% | 86% | 0 |
| mg | 1 | 64% | 64% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 22% | 74% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=33554432 | 733.898 ms | 0.09 | 0.73 | 7% | ok |
| blas1 | dot | f32 | n=33554432 | 111.479 ms | 0.60 | 2.41 | 24% | ok |
| blas1 | axpy | f64 | n=33554432 | 850.896 ms | 0.08 | 0.95 | 9% | ok |
| blas1 | axpy | f32 | n=33554432 | 365.405 ms | 0.18 | 1.10 | 11% | ok |
| blas1 | scale | f64 | n=33554432 | 56.203 ms | 0.60 | 9.55 | 94% | ok |
| blas1 | two_norm | f64 | n=33554432 | 751.195 ms | 0.09 | 0.36 | 4% | ok |
| blas1 | one_norm | f64 | n=33554432 | 39.272 ms | 0.85 | 6.84 | 67% | ok |
| blas1 | infinity_norm | f64 | n=33554432 | 52.397 ms | 0.64 | 5.12 | 50% | ok |
| blas1 | sum | f64 | n=33554432 | 39.150 ms | 0.86 | 6.86 | 67% | ok |
| blas1 | dot_k3 | f64 | n=33554432 | 1110.614 ms | 0.66 | 0.48 | 6% | ok |
| blas2 | gemv | f64 | n=8192 | 56.244 ms | 2.39 | 9.55 | 94% | ok |
| blas2 | gemv | f32 | n=8192 | 34.115 ms | 3.93 | 7.87 | 77% | ok |
| blas2 | gemv_colmajor | f64 | n=8192 | 53.918 ms | 2.49 | 9.96 | 98% | ok |
| blas2 | gemv_colmajor | f32 | n=8192 | 27.268 ms | 4.92 | 9.85 | 97% | ok |
| blas2 | symv | f64 | n=8192 | 52.883 ms | 2.54 | 5.08 | 50% | ok |
| blas2 | trmv | f64 | n=8192 | 29.567 ms | 2.27 | 9.08 | 89% | ok |
| blas2 | trmv | f32 | n=8192 | 17.676 ms | 3.80 | 7.60 | 75% | ok |
| blas2 | trsv | f64 | n=8192 | 29.308 ms | 2.29 | 9.16 | 90% | ok |
| blas2 | trsv | f32 | n=8192 | 17.326 ms | 3.87 | 7.75 | 76% | ok |
| blas2 | ger | f64 | n=8192 | 117.410 ms | 1.14 | 9.15 | 90% | ok |
| blas3 | gemm | f64 | 1024x1024x1024 | 516.502 ms | 4.16 | 0.05 | 40% | ok |
| blas3 | gemm | f32 | 1024x1024x1024 | 289.939 ms | 7.41 | 0.04 | 35% | ok |
| blas3 | gemmt | f64 | 1024x1024x1024 | 275.131 ms | 3.90 | 0.09 | 37% | ok |
| blas3 | symm | f64 | 1024x1024x1024 | 564.430 ms | 3.80 | 0.04 | 36% | ok |
| blas3 | syrk | f64 | 1024x1024x1024 | 314.621 ms | 3.41 | 0.05 | 32% | ok |
| blas3 | syr2k | f64 | 1024x1024x1024 | 605.357 ms | 3.55 | 0.04 | 34% | ok |
| blas3 | trmm | f64 | 1024x1024x1024 | 261.567 ms | 4.11 | 0.08 | 39% | ok |
| blas3 | trsm | f64 | 1024x1024x1024 | 264.884 ms | 4.05 | 0.08 | 39% | ok |
| blas3 | trsm | f32 | 1024x1024x1024 | 128.771 ms | 8.34 | 0.08 | 40% | ok |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 131.995 ms | 0.13 | 1.02 | 10% | ok |
| factor | lu | f64 | 1024x1024 | 244.245 ms | 2.93 | 0.07 | 28% | ok |
| factor | lu | f32 | 1024x1024 | 130.580 ms | 5.48 | 0.06 | 26% | ok |
| factor | cholesky | f64 | 1024x1024 | 203.971 ms | 1.75 | 0.08 | 17% | ok |
| factor | cholesky | f32 | 1024x1024 | 199.828 ms | 1.79 | 0.04 | 9% | ok |
| factor | ldlt | f64 | 1024x1024 | 211.122 ms | 1.70 | 0.08 | 16% | ok |
| factor | ldlt_bk | f64 | 1024x1024 | 111.531 ms | 3.21 | 0.15 | 31% | ok |
| factor | qr | f64 | 1024x1024 | 6845.977 ms | 0.21 | 0.00 | 2% | ok |
| factor | lq | f64 | 1024x1024 | 580.323 ms | 2.47 | 0.03 | 23% | ok |
| factor | hessenberg | f64 | 1024x1024 | 7717.905 ms | 0.46 | 0.00 | 4% | ok |
| factor | householder | f64 | 1024x1024 | 20.222 ms | 0.21 | 0.83 | 8% | ok |
| factor | inv | f64 | 1024x1024 | 1174.497 ms | 1.83 | 0.01 | 17% | ok |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 149.228 ms | 4.80 | 0.11 | 46% | ok |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 629.608 ms | 0.28 | 0.01 | 3% | ok |
| eigen | eigen_symmetric | f64 | 512x512 | 2109.648 ms | 0.57 | 0.00 | 5% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 14760.159 ms | 0.08 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 512x512 | 1085.321 ms | 1.24 | 0.00 | 12% | ok |
| eigen | schur | f64 | 512x512 | 3007.162 ms | 1.12 | 0.00 | 11% | ok |
| eigen | svd | f64 | 512x512 | 2681.504 ms | 1.10 | 0.00 | 10% | ok |
| eigen | expm | f64 | 512x512 | 723.705 ms | 2.78 | 0.01 | 26% | ok |
| eigen | sylvester | f64 | 512x512 | 6934.218 ms | 1.16 | 0.00 | 11% | ok |
| elementwise | exp | f64 | n=33554432 | 648.607 ms | 0.05 | 0.83 | 8% | ok |
| elementwise | log | f64 | n=33554432 | 562.307 ms | 0.06 | 0.95 | 9% | ok |
| elementwise | sin | f64 | n=33554432 | 684.018 ms | 0.05 | 0.78 | 8% | ok |
| elementwise | sqrt | f64 | n=33554432 | 361.823 ms | 0.09 | 1.48 | 15% | ok |
| elementwise | erf | f64 | n=33554432 | 9954.074 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 10.456 ms | 1.00 | 10.42 | 103% | ok |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 12.971 ms | 0.81 | 8.40 | 83% | ok |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 61.075 ms | 2.74 | 5.77 | 57% | ok |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 101.959 ms | 0.13 | 0.45 | 4% | ok |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 38.685 ms | 0.26 | 2.17 | 21% | ok |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 304.773 ms | 0.35 | 0.01 | 3% | ok |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 554.219 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 155.733 ms | 0.69 | 0.02 | 7% | ok |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 898.078 ms | 0.24 | 0.00 | 2% | ok |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 880.178 ms | 0.24 | 0.00 | 2% | ok |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 360.103 ms | 0.59 | 0.01 | 6% | ok |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 1116.655 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 43.973 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 64.106 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 580.629 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 2.400 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 253.119 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 26.587 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 269.872 ms | 0.40 | 0.01 | 4% | ok |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 9.098 ms | 0.44 | 1.76 | 17% | ok |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 55.039 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 92.182 ms | 1.16 | 0.03 | 11% | ok |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 9.090 ms | 0.44 | 1.77 | 17% | ok |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 26.774 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 825.860 ms | 0.26 | 0.00 | 2% | ok |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.305 ms | 0.13 | 0.53 | 5% | ok |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 40.818 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 794.895 ms | 0.27 | 0.00 | 3% | ok |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.481 ms | 0.13 | 0.53 | 5% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 23.884 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1949.520 ms | 0.50 | 0.00 | 5% | ok |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 19.259 ms | 0.39 | 1.56 | 15% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 62.578 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 605.786 ms | 1.60 | 0.00 | 15% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 19.202 ms | 0.39 | 1.56 | 15% | ok |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 23.873 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 5708.665 ms | 0.34 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 88.560 ms | 0.08 | 0.34 | 3% | ok |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 33.983 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 4939.416 ms | 0.39 | 0.00 | 4% | ok |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 87.615 ms | 0.09 | 0.34 | 3% | ok |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 1772.250 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 3180.287 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 763.520 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 892.741 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 931.018 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 584.702 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 4848.076 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 1239.106 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 721.874 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 1399.479 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 1296.505 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 1483.551 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 2038.950 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.684 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 256^2 grid | 18.245 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 256^2 grid | 148.028 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 256^2 grid | 12.001 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 256^2 grid | 11.258 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 256^2 grid | 0.376 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 8.752 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 14.881 ms | 0.70 | 7.89 | 78% | ok |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 13.405 ms | 0.78 | 8.76 | 86% | ok |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 16.027 ms | 0.65 | 7.32 | 72% | ok |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 179.962 ms | 0.86 | 6.49 | 64% | ok |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 567.849 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 601.054 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 1435.953 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 172.714 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=33554432 | 754.635 ms | 0.09 | 0.36 | 3% | ok |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 910.227 ms | 2.36 | 0.02 | 22% | ok |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 9.533 ms | 7.04 | 7.48 | 74% | ok |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 26.214 ms | 2.56 | 10.24 | n/a | ok |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.633 ms | 5.31 | 10.62 | n/a | ok |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 20.409 ms | 3.29 | 3.29 | n/a | ok |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 20.436 ms | 3.28 | 3.28 | n/a | ok |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 455.080 ms | 4.72 | 0.03 | n/a | ok |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 364.089 ms | 5.90 | 0.02 | n/a | ok |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 189.004 ms | 11.36 | 0.03 | n/a | ok |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 188.750 ms | 11.38 | 0.03 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 566.169 ms | 3.79 | 0.01 | n/a | ok |

