# mtl5_bench platform characterization

Class `large`, variants native native-fast blas, threads 1,6, CPUs 0,1,2,3,4,5, clock policy governor performance on every CPU, turbo on; performance 1200-2700 MHz on cpus 0,1,2,3,4,5, 2026-11-28T04:23:53Z.

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

## xeon-e5-2420v2 -- native, 6 threads, class large

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 59.1 GFLOP/s, f32 118.2 GFLOP/s |
| ... hardware, SSE2 | f64 59.1, f32 118.2 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.7 GFLOP/s |
| bandwidth | read 11.4, update 10.7, triad 8.5 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 53% | 107% | 0 | 84% |
| blas2 | 10 | 79% | 107% | 0 | 16% |
| blas3 | 10 | 7% | 39% | 0 | 17% |
| factor | 12 | 6% | 40% | 0 | 53% |
| eigen | 8 | 2% | 12% | 0 | 19% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 64% | 97% | 0 | 17% |
| direct | 19 | 1% | 15% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 3% | 14% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 69% | 76% | 0 | 17% |
| mg | 1 | 59% | 59% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 22% | 65% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 87% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 131.944 ms | 0.51 | 4.07 | 36% | ok | 5.6x (93%) |
| blas1 | dot | f32 | n=33554432 | 22.067 ms | 3.04 | 12.16 | 107% | ok | 5.1x (84%) |
| blas1 | axpy | f64 | n=33554432 | 151.983 ms | 0.44 | 5.30 | 46% | ok | 5.6x (93%) |
| blas1 | axpy | f32 | n=33554432 | 66.998 ms | 1.00 | 6.01 | 53% | ok | 5.5x (91%) |
| blas1 | scale | f64 | n=33554432 | 53.956 ms | 0.62 | 9.95 | 87% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=33554432 | 132.581 ms | 0.51 | 2.02 | 18% | ok | 5.7x (94%) |
| blas1 | one_norm | f64 | n=33554432 | 39.065 ms | 0.86 | 6.87 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=33554432 | 52.538 ms | 0.64 | 5.11 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=33554432 | 39.766 ms | 0.84 | 6.75 | 59% | ok | 1.0x (16%) |
| blas1 | dot_k3 | f64 | n=33554432 | 1113.978 ms | 0.66 | 0.48 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=8192 | 44.043 ms | 3.05 | 12.19 | 107% | ok | 1.3x (21%) |
| blas2 | gemv | f32 | n=8192 | 22.116 ms | 6.07 | 12.14 | 106% | ok | 1.5x (26%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 43.881 ms | 3.06 | 12.24 | 107% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 21.909 ms | 6.13 | 12.26 | 107% | ok | 1.2x (21%) |
| blas2 | symv | f64 | n=8192 | 57.306 ms | 2.34 | 4.69 | 41% | ok | 0.9x (15%) |
| blas2 | trmv | f64 | n=8192 | 29.995 ms | 2.24 | 8.95 | 79% | ok | 1.0x (16%) |
| blas2 | trmv | f32 | n=8192 | 19.633 ms | 3.42 | 6.84 | 60% | ok | 0.9x (15%) |
| blas2 | trsv | f64 | n=8192 | 30.043 ms | 2.23 | 8.94 | 78% | ok | 1.0x (16%) |
| blas2 | trsv | f32 | n=8192 | 18.067 ms | 3.71 | 7.43 | 65% | ok | 1.0x (16%) |
| blas2 | ger | f64 | n=8192 | 120.038 ms | 1.12 | 8.95 | 78% | ok | 1.0x (16%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 92.258 ms | 23.28 | 0.27 | 39% | ok | 5.6x (93%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 51.307 ms | 41.86 | 0.25 | 35% | ok | 5.7x (94%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 276.942 ms | 3.88 | 0.09 | 7% | ok | 1.0x (17%) |
| blas3 | symm | f64 | 1024x1024x1024 | 566.905 ms | 3.79 | 0.04 | 6% | ok | 1.0x (17%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 309.704 ms | 3.47 | 0.05 | 6% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 590.272 ms | 3.64 | 0.04 | 6% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 256.762 ms | 4.18 | 0.08 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 258.863 ms | 4.15 | 0.08 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 127.966 ms | 8.39 | 0.08 | 7% | ok | 1.0x (17%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 132.239 ms | 0.13 | 1.01 | 9% | ok | 1.0x (17%) |
| factor | lu | f64 | 1024x1024 | 70.492 ms | 10.15 | 0.24 | 17% | ok | 3.5x (58%) |
| factor | lu | f32 | 1024x1024 | 41.146 ms | 17.40 | 0.20 | 15% | ok | 3.2x (53%) |
| factor | cholesky | f64 | 1024x1024 | 96.576 ms | 3.71 | 0.17 | 6% | ok | 2.1x (35%) |
| factor | cholesky | f32 | 1024x1024 | 93.417 ms | 3.83 | 0.09 | 3% | ok | 2.1x (36%) |
| factor | ldlt | f64 | 1024x1024 | 211.071 ms | 1.70 | 0.08 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 1024x1024 | 105.395 ms | 3.40 | 0.16 | 6% | ok | 1.1x (18%) |
| factor | qr | f64 | 1024x1024 | 1626.430 ms | 0.88 | 0.01 | 1% | ok | 4.2x (70%) |
| factor | lq | f64 | 1024x1024 | 149.458 ms | 9.58 | 0.11 | 16% | ok | 3.9x (65%) |
| factor | hessenberg | f64 | 1024x1024 | 1869.287 ms | 1.91 | 0.01 | 3% | ok | 4.1x (69%) |
| factor | householder | f64 | 1024x1024 | 3.720 ms | 1.13 | 4.51 | 40% | ok | 5.4x (91%) |
| factor | inv | f64 | 1024x1024 | 1004.629 ms | 2.14 | 0.02 | 4% | ok | 1.2x (19%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 56.068 ms | 12.77 | 0.30 | 22% | ok | 2.7x (44%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 415.734 ms | 0.43 | 0.01 | 1% | ok | 1.5x (25%) |
| eigen | eigen_symmetric | f64 | 512x512 | 1843.592 ms | 0.66 | 0.00 | 1% | ok | 1.1x (19%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 14764.464 ms | 0.08 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 512x512 | 859.410 ms | 1.56 | 0.00 | 3% | ok | 1.3x (21%) |
| eigen | schur | f64 | 512x512 | 2602.027 ms | 1.29 | 0.00 | 2% | ok | 1.2x (19%) |
| eigen | svd | f64 | 512x512 | 2675.193 ms | 1.10 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 512x512 | 273.603 ms | 7.36 | 0.02 | 12% | ok | 2.6x (44%) |
| eigen | sylvester | f64 | 512x512 | 5950.041 ms | 1.35 | 0.00 | 2% | ok | 1.2x (19%) |
| elementwise | exp | f64 | n=33554432 | 605.779 ms | 0.06 | 0.89 | 8% | ok | 1.1x (18%) |
| elementwise | log | f64 | n=33554432 | 561.545 ms | 0.06 | 0.96 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=33554432 | 685.244 ms | 0.05 | 0.78 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=33554432 | 363.236 ms | 0.09 | 1.48 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=33554432 | 9939.398 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 9.855 ms | 1.06 | 11.06 | 97% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 13.047 ms | 0.80 | 8.35 | 73% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 48.385 ms | 3.46 | 7.28 | 64% | ok | 1.3x (21%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 102.301 ms | 0.13 | 0.45 | 4% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 37.663 ms | 0.27 | 2.23 | 20% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 305.859 ms | 0.35 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 558.546 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 156.474 ms | 0.68 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 905.552 ms | 0.24 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 855.649 ms | 0.25 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 386.043 ms | 0.55 | 0.01 | 1% | ok | 0.9x (16%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 1154.214 ms | 0.02 | 0.00 | 0% | ok | 1.0x (16%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 43.441 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 63.247 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 580.837 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 2.396 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 252.615 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 26.663 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 270.141 ms | 0.40 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 9.172 ms | 0.44 | 1.75 | 15% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 54.583 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 93.003 ms | 1.15 | 0.03 | 2% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 9.225 ms | 0.44 | 1.74 | 15% | ok | 1.0x (16%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 27.213 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 834.543 ms | 0.26 | 0.00 | 0% | ok | 1.0x (16%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.260 ms | 0.13 | 0.53 | 5% | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 41.178 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 781.863 ms | 0.27 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.272 ms | 0.13 | 0.53 | 5% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 23.696 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1962.714 ms | 0.49 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 19.437 ms | 0.39 | 1.54 | 14% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 61.591 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 607.337 ms | 1.60 | 0.00 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 19.481 ms | 0.38 | 1.54 | 14% | ok | 1.0x (16%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 23.960 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 5766.766 ms | 0.34 | 0.00 | 1% | ok | 1.0x (16%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 88.042 ms | 0.09 | 0.34 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 33.789 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 4931.278 ms | 0.39 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 87.776 ms | 0.09 | 0.34 | 3% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 1741.883 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 2992.927 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 766.588 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 887.666 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 923.967 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 576.894 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 4859.506 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 1229.463 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 722.349 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 1427.147 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 1296.930 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 1496.126 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 2065.505 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.689 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 256^2 grid | 18.224 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 256^2 grid | 149.402 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 256^2 grid | 11.985 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 256^2 grid | 11.241 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 256^2 grid | 0.377 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 9.372 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 14.898 ms | 0.70 | 7.88 | 69% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 13.478 ms | 0.78 | 8.71 | 76% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 16.037 ms | 0.65 | 7.32 | 64% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 173.078 ms | 0.90 | 6.75 | 59% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 563.914 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 577.510 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 1423.534 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 172.500 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=33554432 | 749.453 ms | 0.09 | 0.36 | 3% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 162.890 ms | 13.18 | 0.10 | 22% | ok | 5.6x (93%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 9.634 ms | 6.97 | 7.40 | 65% | ok | 1.0x (16%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 26.074 ms | 2.57 | 10.29 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.662 ms | 5.30 | 10.60 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 20.409 ms | 3.29 | 3.29 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 20.452 ms | 3.28 | 3.28 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 87.316 ms | 24.59 | 0.14 | n/a | ok | 5.2x (87%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 65.251 ms | 32.91 | 0.13 | n/a | ok | 5.6x (93%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 33.772 ms | 63.59 | 0.19 | n/a | ok | 5.6x (93%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 33.744 ms | 63.64 | 0.19 | n/a | ok | 5.6x (93%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 103.137 ms | 20.82 | 0.06 | n/a | ok | 5.5x (91%) |

## xeon-e5-2420v2 -- native-fast, 1 thread, class large

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 21.0 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 21.0 GFLOP/s |
| ... this build, SSE4 | f64 10.5, f32 21.0 GFLOP/s |
| bandwidth | read 10.4, update 10.3, triad 8.1 GB/s (ceiling 10.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 96% | 108% | 0 |
| blas2 | 10 | 91% | 98% | 0 |
| blas3 | 10 | 82% | 95% | 0 |
| factor | 12 | 17% | 49% | 0 |
| eigen | 8 | 11% | 46% | 0 |
| elementwise | 5 | 8% | 14% | 0 |
| sparse | 5 | 55% | 99% | 0 |
| direct | 19 | 4% | 17% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 15% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 75% | 83% | 0 |
| mg | 1 | 62% | 62% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 72% | 89% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=33554432 | 47.785 ms | 1.40 | 11.24 | 108% | ok |
| blas1 | dot | f32 | n=33554432 | 23.885 ms | 2.81 | 11.24 | 108% | ok |
| blas1 | axpy | f64 | n=33554432 | 80.456 ms | 0.83 | 10.01 | 96% | ok |
| blas1 | axpy | f32 | n=33554432 | 38.287 ms | 1.75 | 10.52 | 101% | ok |
| blas1 | scale | f64 | n=33554432 | 56.327 ms | 0.60 | 9.53 | 91% | ok |
| blas1 | two_norm | f64 | n=33554432 | 24.123 ms | 2.78 | 11.13 | 107% | ok |
| blas1 | one_norm | f64 | n=33554432 | 39.199 ms | 0.86 | 6.85 | 66% | ok |
| blas1 | infinity_norm | f64 | n=33554432 | 52.227 ms | 0.64 | 5.14 | 49% | ok |
| blas1 | sum | f64 | n=33554432 | 39.135 ms | 0.86 | 6.86 | 66% | ok |
| blas1 | dot_k3 | f64 | n=33554432 | 1089.589 ms | 0.68 | 0.49 | 6% | ok |
| blas2 | gemv | f64 | n=8192 | 55.310 ms | 2.43 | 9.71 | 93% | ok |
| blas2 | gemv | f32 | n=8192 | 27.204 ms | 4.93 | 9.87 | 95% | ok |
| blas2 | gemv_colmajor | f64 | n=8192 | 52.270 ms | 2.57 | 10.27 | 98% | ok |
| blas2 | gemv_colmajor | f32 | n=8192 | 26.215 ms | 5.12 | 10.24 | 98% | ok |
| blas2 | symv | f64 | n=8192 | 53.794 ms | 2.50 | 4.99 | 48% | ok |
| blas2 | trmv | f64 | n=8192 | 28.740 ms | 2.33 | 9.34 | 90% | ok |
| blas2 | trmv | f32 | n=8192 | 15.077 ms | 4.45 | 8.91 | 85% | ok |
| blas2 | trsv | f64 | n=8192 | 29.114 ms | 2.31 | 9.22 | 88% | ok |
| blas2 | trsv | f32 | n=8192 | 14.115 ms | 4.75 | 9.51 | 91% | ok |
| blas2 | ger | f64 | n=8192 | 119.910 ms | 1.12 | 8.96 | 86% | ok |
| blas3 | gemm | f64 | 1024x1024x1024 | 225.315 ms | 9.53 | 0.11 | 91% | ok |
| blas3 | gemm | f32 | 1024x1024x1024 | 107.910 ms | 19.90 | 0.12 | 95% | ok |
| blas3 | gemmt | f64 | 1024x1024x1024 | 131.375 ms | 8.17 | 0.19 | 78% | ok |
| blas3 | symm | f64 | 1024x1024x1024 | 233.407 ms | 9.20 | 0.11 | 88% | ok |
| blas3 | syrk | f64 | 1024x1024x1024 | 124.320 ms | 8.64 | 0.13 | 82% | ok |
| blas3 | syr2k | f64 | 1024x1024x1024 | 247.051 ms | 8.69 | 0.10 | 83% | ok |
| blas3 | trmm | f64 | 1024x1024x1024 | 149.868 ms | 7.16 | 0.14 | 68% | ok |
| blas3 | trsm | f64 | 1024x1024x1024 | 153.427 ms | 7.00 | 0.14 | 67% | ok |
| blas3 | trsm | f32 | 1024x1024x1024 | 70.131 ms | 15.31 | 0.15 | 73% | ok |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 129.771 ms | 0.13 | 1.03 | 10% | ok |
| factor | lu | f64 | 1024x1024 | 234.316 ms | 3.05 | 0.07 | 29% | ok |
| factor | lu | f32 | 1024x1024 | 123.736 ms | 5.79 | 0.07 | 28% | ok |
| factor | cholesky | f64 | 1024x1024 | 208.844 ms | 1.71 | 0.08 | 16% | ok |
| factor | cholesky | f32 | 1024x1024 | 202.236 ms | 1.77 | 0.04 | 8% | ok |
| factor | ldlt | f64 | 1024x1024 | 214.714 ms | 1.67 | 0.08 | 16% | ok |
| factor | ldlt_bk | f64 | 1024x1024 | 108.226 ms | 3.31 | 0.16 | 31% | ok |
| factor | qr | f64 | 1024x1024 | 6916.868 ms | 0.21 | 0.00 | 2% | ok |
| factor | lq | f64 | 1024x1024 | 574.572 ms | 2.49 | 0.03 | 24% | ok |
| factor | hessenberg | f64 | 1024x1024 | 7781.264 ms | 0.46 | 0.00 | 4% | ok |
| factor | householder | f64 | 1024x1024 | 20.421 ms | 0.21 | 0.82 | 8% | ok |
| factor | inv | f64 | 1024x1024 | 1193.836 ms | 1.80 | 0.01 | 17% | ok |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 140.286 ms | 5.10 | 0.12 | 49% | ok |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 643.073 ms | 0.28 | 0.01 | 3% | ok |
| eigen | eigen_symmetric | f64 | 512x512 | 2105.702 ms | 0.57 | 0.00 | 5% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 14926.777 ms | 0.08 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 512x512 | 1081.943 ms | 1.24 | 0.00 | 12% | ok |
| eigen | schur | f64 | 512x512 | 2972.156 ms | 1.13 | 0.00 | 11% | ok |
| eigen | svd | f64 | 512x512 | 2346.058 ms | 1.26 | 0.00 | 12% | ok |
| eigen | expm | f64 | 512x512 | 413.426 ms | 4.87 | 0.01 | 46% | ok |
| eigen | sylvester | f64 | 512x512 | 6751.524 ms | 1.19 | 0.00 | 11% | ok |
| elementwise | exp | f64 | n=33554432 | 606.301 ms | 0.06 | 0.89 | 8% | ok |
| elementwise | log | f64 | n=33554432 | 562.224 ms | 0.06 | 0.95 | 9% | ok |
| elementwise | sin | f64 | n=33554432 | 683.887 ms | 0.05 | 0.79 | 8% | ok |
| elementwise | sqrt | f64 | n=33554432 | 362.747 ms | 0.09 | 1.48 | 14% | ok |
| elementwise | erf | f64 | n=33554432 | 9962.988 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 10.526 ms | 1.00 | 10.35 | 99% | ok |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 12.975 ms | 0.81 | 8.40 | 81% | ok |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 61.607 ms | 2.72 | 5.72 | 55% | ok |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 99.246 ms | 0.13 | 0.46 | 4% | ok |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 38.764 ms | 0.26 | 2.16 | 21% | ok |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 307.801 ms | 0.35 | 0.01 | 3% | ok |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 561.830 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 158.502 ms | 0.67 | 0.02 | 6% | ok |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 904.442 ms | 0.24 | 0.00 | 2% | ok |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 931.293 ms | 0.23 | 0.00 | 2% | ok |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 371.070 ms | 0.58 | 0.01 | 5% | ok |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 1148.416 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 44.409 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 65.931 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 592.740 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 2.523 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 251.412 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 27.597 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 269.494 ms | 0.40 | 0.01 | 4% | ok |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 9.077 ms | 0.44 | 1.77 | 17% | ok |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 55.938 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 93.460 ms | 1.14 | 0.03 | 11% | ok |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 9.087 ms | 0.44 | 1.77 | 17% | ok |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 27.392 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 830.421 ms | 0.26 | 0.00 | 2% | ok |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.316 ms | 0.13 | 0.53 | 5% | ok |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 41.932 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 848.375 ms | 0.25 | 0.00 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.348 ms | 0.13 | 0.53 | 5% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 24.267 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1947.968 ms | 0.50 | 0.00 | 5% | ok |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 19.299 ms | 0.39 | 1.55 | 15% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 62.455 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 602.435 ms | 1.61 | 0.00 | 15% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 19.345 ms | 0.39 | 1.55 | 15% | ok |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 24.518 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 5722.516 ms | 0.34 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 87.805 ms | 0.09 | 0.34 | 3% | ok |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 34.326 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 6544.543 ms | 0.30 | 0.00 | 3% | ok |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 87.622 ms | 0.09 | 0.34 | 3% | ok |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 635.484 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 983.530 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 369.403 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 288.432 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 305.520 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 280.033 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 649.440 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 509.773 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 306.991 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 329.885 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 292.630 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 301.687 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 372.637 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.529 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 256^2 grid | 19.057 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 256^2 grid | 149.982 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 256^2 grid | 12.059 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 256^2 grid | 11.538 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 256^2 grid | 0.366 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 8.667 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 14.912 ms | 0.70 | 7.87 | 75% | ok |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 13.489 ms | 0.78 | 8.70 | 83% | ok |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 16.066 ms | 0.65 | 7.31 | 70% | ok |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 180.244 ms | 0.86 | 6.48 | 62% | ok |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 562.135 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 571.854 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 1452.593 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 69.290 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=33554432 | 28.868 ms | 2.32 | 9.30 | 89% | ok |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 497.445 ms | 4.32 | 0.03 | 41% | ok |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 9.550 ms | 7.03 | 7.47 | 72% | ok |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 24.491 ms | 2.74 | 10.96 | n/a | ok |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.079 ms | 5.56 | 11.11 | n/a | ok |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 8.819 ms | 7.61 | 7.61 | n/a | ok |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 7.405 ms | 9.06 | 9.06 | n/a | ok |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 150.919 ms | 14.23 | 0.08 | n/a | ok |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 176.491 ms | 12.17 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 175.623 ms | 12.23 | 0.04 | n/a | ok |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 128.375 ms | 16.73 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 97.446 ms | 22.04 | 0.06 | n/a | ok |

## xeon-e5-2420v2 -- native-fast, 6 threads, class large

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 59.1 GFLOP/s, f32 118.2 GFLOP/s |
| ... hardware, SSE2 | f64 59.1, f32 118.2 GFLOP/s |
| ... this build, SSE4 | f64 58.9, f32 118.0 GFLOP/s |
| bandwidth | read 11.3, update 10.5, triad 8.3 GB/s (ceiling 11.3) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 95% | 110% | 0 | 17% |
| blas2 | 10 | 84% | 109% | 0 | 18% |
| blas3 | 10 | 61% | 86% | 0 | 70% |
| factor | 12 | 6% | 40% | 0 | 51% |
| eigen | 8 | 2% | 15% | 0 | 19% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 69% | 98% | 0 | 17% |
| direct | 19 | 1% | 16% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 3% | 14% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 16% |
| smoother | 3 | 70% | 77% | 0 | 17% |
| mg | 1 | 60% | 60% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 67% | 82% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 86% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 43.471 ms | 1.54 | 12.35 | 110% | ok | 1.1x (18%) |
| blas1 | dot | f32 | n=33554432 | 21.746 ms | 3.09 | 12.34 | 110% | ok | 1.1x (18%) |
| blas1 | axpy | f64 | n=33554432 | 75.218 ms | 0.89 | 10.71 | 95% | ok | 1.1x (18%) |
| blas1 | axpy | f32 | n=33554432 | 36.619 ms | 1.83 | 11.00 | 98% | ok | 1.0x (17%) |
| blas1 | scale | f64 | n=33554432 | 54.126 ms | 0.62 | 9.92 | 88% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=33554432 | 22.364 ms | 3.00 | 12.00 | 107% | ok | 1.1x (18%) |
| blas1 | one_norm | f64 | n=33554432 | 39.100 ms | 0.86 | 6.87 | 61% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=33554432 | 52.566 ms | 0.64 | 5.11 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=33554432 | 40.607 ms | 0.83 | 6.61 | 59% | ok | 1.0x (16%) |
| blas1 | dot_k3 | f64 | n=33554432 | 1095.084 ms | 0.67 | 0.49 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=8192 | 43.713 ms | 3.07 | 12.28 | 109% | ok | 1.3x (21%) |
| blas2 | gemv | f32 | n=8192 | 21.871 ms | 6.14 | 12.28 | 109% | ok | 1.2x (21%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 43.661 ms | 3.07 | 12.30 | 109% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 22.439 ms | 5.98 | 11.97 | 106% | ok | 1.2x (19%) |
| blas2 | symv | f64 | n=8192 | 55.213 ms | 2.43 | 4.86 | 43% | ok | 1.0x (16%) |
| blas2 | trmv | f64 | n=8192 | 28.876 ms | 2.32 | 9.30 | 83% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=8192 | 14.129 ms | 4.75 | 9.50 | 84% | ok | 1.1x (18%) |
| blas2 | trsv | f64 | n=8192 | 28.630 ms | 2.34 | 9.38 | 83% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=8192 | 14.136 ms | 4.75 | 9.50 | 84% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=8192 | 118.828 ms | 1.13 | 9.04 | 80% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 44.870 ms | 47.86 | 0.56 | 81% | ok | 5.0x (84%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 21.203 ms | 101.28 | 0.59 | 86% | ok | 5.1x (85%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 31.911 ms | 33.65 | 0.79 | 57% | ok | 4.1x (69%) |
| blas3 | symm | f64 | 1024x1024x1024 | 52.087 ms | 41.23 | 0.48 | 70% | ok | 4.5x (75%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 29.475 ms | 36.43 | 0.57 | 62% | ok | 4.2x (70%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 59.131 ms | 36.32 | 0.43 | 61% | ok | 4.2x (70%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 54.469 ms | 19.71 | 0.39 | 33% | ok | 2.8x (46%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 56.410 ms | 19.03 | 0.37 | 32% | ok | 2.7x (45%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 26.054 ms | 41.21 | 0.40 | 35% | ok | 2.7x (45%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 129.422 ms | 0.13 | 1.04 | 9% | ok | 1.0x (17%) |
| factor | lu | f64 | 1024x1024 | 67.998 ms | 10.53 | 0.25 | 18% | ok | 3.4x (57%) |
| factor | lu | f32 | 1024x1024 | 40.214 ms | 17.80 | 0.21 | 15% | ok | 3.1x (51%) |
| factor | cholesky | f64 | 1024x1024 | 97.321 ms | 3.68 | 0.17 | 6% | ok | 2.1x (36%) |
| factor | cholesky | f32 | 1024x1024 | 94.156 ms | 3.80 | 0.09 | 3% | ok | 2.1x (36%) |
| factor | ldlt | f64 | 1024x1024 | 213.328 ms | 1.68 | 0.08 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 1024x1024 | 100.848 ms | 3.55 | 0.17 | 6% | ok | 1.1x (18%) |
| factor | qr | f64 | 1024x1024 | 1630.123 ms | 0.88 | 0.01 | 1% | ok | 4.2x (71%) |
| factor | lq | f64 | 1024x1024 | 148.080 ms | 9.67 | 0.11 | 16% | ok | 3.9x (65%) |
| factor | hessenberg | f64 | 1024x1024 | 1868.861 ms | 1.92 | 0.01 | 3% | ok | 4.2x (69%) |
| factor | householder | f64 | 1024x1024 | 3.736 ms | 1.12 | 4.49 | 40% | ok | 5.5x (91%) |
| factor | inv | f64 | 1024x1024 | 1018.845 ms | 2.11 | 0.02 | 4% | ok | 1.2x (20%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 55.899 ms | 12.81 | 0.30 | 22% | ok | 2.5x (42%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 408.622 ms | 0.44 | 0.01 | 1% | ok | 1.6x (26%) |
| eigen | eigen_symmetric | f64 | 512x512 | 1918.594 ms | 0.63 | 0.00 | 1% | ok | 1.1x (18%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 14575.338 ms | 0.08 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 512x512 | 861.385 ms | 1.56 | 0.00 | 3% | ok | 1.3x (21%) |
| eigen | schur | f64 | 512x512 | 2574.638 ms | 1.30 | 0.00 | 2% | ok | 1.2x (19%) |
| eigen | svd | f64 | 512x512 | 2306.536 ms | 1.28 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 512x512 | 226.553 ms | 8.89 | 0.02 | 15% | ok | 1.8x (30%) |
| eigen | sylvester | f64 | 512x512 | 5854.309 ms | 1.38 | 0.00 | 2% | ok | 1.2x (19%) |
| elementwise | exp | f64 | n=33554432 | 615.379 ms | 0.05 | 0.87 | 8% | ok | 1.0x (16%) |
| elementwise | log | f64 | n=33554432 | 563.363 ms | 0.06 | 0.95 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=33554432 | 682.847 ms | 0.05 | 0.79 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=33554432 | 362.171 ms | 0.09 | 1.48 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=33554432 | 9990.773 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 9.843 ms | 1.06 | 11.07 | 98% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 12.619 ms | 0.83 | 8.64 | 77% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 45.542 ms | 3.68 | 7.73 | 69% | ok | 1.4x (23%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 96.326 ms | 0.14 | 0.48 | 4% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 37.511 ms | 0.27 | 2.23 | 20% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 305.232 ms | 0.35 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 556.160 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 156.767 ms | 0.68 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 900.838 ms | 0.24 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 933.162 ms | 0.23 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 366.652 ms | 0.58 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 1094.192 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 45.130 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 64.291 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 591.647 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 2.522 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 251.941 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 26.737 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 270.712 ms | 0.40 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 9.091 ms | 0.44 | 1.77 | 16% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 55.153 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 92.838 ms | 1.15 | 0.03 | 2% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 9.224 ms | 0.44 | 1.74 | 15% | ok | 1.0x (16%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 27.266 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 837.151 ms | 0.26 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.518 ms | 0.13 | 0.53 | 5% | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 41.974 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 838.004 ms | 0.26 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.308 ms | 0.13 | 0.53 | 5% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 24.166 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1924.249 ms | 0.50 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 19.414 ms | 0.39 | 1.54 | 14% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 62.029 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 569.319 ms | 1.70 | 0.00 | 3% | ok | 1.1x (18%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 19.454 ms | 0.39 | 1.54 | 14% | ok | 1.0x (17%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 24.484 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 5764.603 ms | 0.34 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 88.186 ms | 0.09 | 0.34 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 34.462 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 6510.583 ms | 0.30 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 87.760 ms | 0.09 | 0.34 | 3% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 615.361 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 835.602 ms | 0.00 | 0.00 | n/a | ok | 1.2x (20%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 367.998 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 283.014 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 299.383 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 275.489 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 659.158 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 528.518 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 310.444 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 349.577 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 288.148 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 312.887 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 382.171 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.518 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 256^2 grid | 19.507 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | ilut | f64 | 256^2 grid | 152.003 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | ic_0 | f64 | 256^2 grid | 13.269 ms | 0.00 | 0.00 | n/a | ok | 0.9x (15%) |
| pc | ildl | f64 | 256^2 grid | 11.427 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 256^2 grid | 0.366 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 8.986 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 14.993 ms | 0.70 | 7.83 | 70% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 13.510 ms | 0.78 | 8.69 | 77% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 16.002 ms | 0.65 | 7.33 | 65% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 174.449 ms | 0.89 | 6.70 | 60% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 563.170 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 580.499 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 1428.973 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 65.701 ms | 0.00 | 0.00 | n/a | ok | 1.1x (18%) |
| mixed | dot | f32>f64 | n=33554432 | 29.023 ms | 2.31 | 9.25 | 82% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 93.683 ms | 22.92 | 0.18 | 39% | ok | 5.3x (88%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 9.456 ms | 7.10 | 7.54 | 67% | ok | 1.0x (17%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 24.456 ms | 2.74 | 10.98 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.372 ms | 5.42 | 10.85 | n/a | ok | 1.0x (16%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 8.820 ms | 7.61 | 7.61 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 7.350 ms | 9.13 | 9.13 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 29.197 ms | 73.55 | 0.43 | n/a | ok | 5.2x (86%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 33.220 ms | 64.65 | 0.25 | n/a | ok | 5.3x (89%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 33.028 ms | 65.02 | 0.19 | n/a | ok | 5.3x (89%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 24.147 ms | 88.94 | 0.26 | n/a | ok | 5.3x (89%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 18.525 ms | 115.92 | 0.34 | n/a | ok | 5.3x (88%) |

## xeon-e5-2420v2 -- blas, 1 thread, class large

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 10.5 GFLOP/s, f32 21.0 GFLOP/s |
| ... hardware, SSE2 | f64 10.5, f32 21.0 GFLOP/s |
| ... this build, scalar | f64 0.1, f32 0.5 GFLOP/s |
| bandwidth | read 10.2, update 10.3, triad 8.0 GB/s (ceiling 10.3) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 23% | 93% | 0 |
| blas2 | 10 | 93% | 96% | 0 |
| blas3 | 10 | 170% | 184% | 0 |
| factor | 12 | 50% | 103% | 0 |
| eigen | 8 | 50% | 159% | 0 |
| elementwise | 5 | 9% | 14% | 0 |
| sparse | 5 | 54% | 102% | 0 |
| direct | 19 | 4% | 17% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 16% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 76% | 85% | 0 |
| mg | 1 | 63% | 63% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 23% | 70% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=33554432 | 747.517 ms | 0.09 | 0.72 | 7% | ok |
| blas1 | dot | f32 | n=33554432 | 111.534 ms | 0.60 | 2.41 | 23% | ok |
| blas1 | axpy | f64 | n=33554432 | 851.079 ms | 0.08 | 0.95 | 9% | ok |
| blas1 | axpy | f32 | n=33554432 | 366.560 ms | 0.18 | 1.10 | 11% | ok |
| blas1 | scale | f64 | n=33554432 | 56.231 ms | 0.60 | 9.55 | 93% | ok |
| blas1 | two_norm | f64 | n=33554432 | 736.992 ms | 0.09 | 0.36 | 4% | ok |
| blas1 | one_norm | f64 | n=33554432 | 39.194 ms | 0.86 | 6.85 | 67% | ok |
| blas1 | infinity_norm | f64 | n=33554432 | 52.536 ms | 0.64 | 5.11 | 50% | ok |
| blas1 | sum | f64 | n=33554432 | 39.184 ms | 0.86 | 6.85 | 67% | ok |
| blas1 | dot_k3 | f64 | n=33554432 | 1129.522 ms | 0.65 | 0.48 | 6% | ok |
| blas2 | gemv | f64 | n=8192 | 59.081 ms | 2.27 | 9.09 | 88% | ok |
| blas2 | gemv | f32 | n=8192 | 27.285 ms | 4.92 | 9.84 | 96% | ok |
| blas2 | gemv_colmajor | f64 | n=8192 | 56.122 ms | 2.39 | 9.57 | 93% | ok |
| blas2 | gemv_colmajor | f32 | n=8192 | 27.867 ms | 4.82 | 9.64 | 94% | ok |
| blas2 | symv | f64 | n=8192 | 29.662 ms | 4.52 | 9.05 | 88% | ok |
| blas2 | trmv | f64 | n=8192 | 30.021 ms | 2.24 | 8.95 | 87% | ok |
| blas2 | trmv | f32 | n=8192 | 13.855 ms | 4.84 | 9.69 | 94% | ok |
| blas2 | trsv | f64 | n=8192 | 29.929 ms | 2.24 | 8.97 | 87% | ok |
| blas2 | trsv | f32 | n=8192 | 13.610 ms | 4.93 | 9.87 | 96% | ok |
| blas2 | ger | f64 | n=8192 | 120.676 ms | 1.11 | 8.90 | 87% | ok |
| blas3 | gemm | f64 | 1024x1024x1024 | 111.086 ms | 19.33 | 0.23 | 184% | ok |
| blas3 | gemm | f32 | 1024x1024x1024 | 68.077 ms | 31.54 | 0.18 | 150% | ok |
| blas3 | gemmt | f64 | 1024x1024x1024 | 65.569 ms | 16.38 | 0.38 | 156% | ok |
| blas3 | symm | f64 | 1024x1024x1024 | 112.866 ms | 19.03 | 0.22 | 181% | ok |
| blas3 | syrk | f64 | 1024x1024x1024 | 61.795 ms | 17.38 | 0.27 | 166% | ok |
| blas3 | syr2k | f64 | 1024x1024x1024 | 119.319 ms | 18.00 | 0.21 | 172% | ok |
| blas3 | trmm | f64 | 1024x1024x1024 | 57.418 ms | 18.70 | 0.37 | 178% | ok |
| blas3 | trsm | f64 | 1024x1024x1024 | 60.272 ms | 17.81 | 0.35 | 170% | ok |
| blas3 | trsm | f32 | 1024x1024x1024 | 39.600 ms | 27.11 | 0.26 | 129% | ok |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 129.409 ms | 0.13 | 1.04 | 10% | ok |
| factor | lu | f64 | 1024x1024 | 67.308 ms | 10.64 | 0.25 | 101% | ok |
| factor | lu | f32 | 1024x1024 | 50.671 ms | 14.13 | 0.17 | 67% | ok |
| factor | cholesky | f64 | 1024x1024 | 43.335 ms | 8.26 | 0.39 | 79% | ok |
| factor | cholesky | f32 | 1024x1024 | 34.210 ms | 10.46 | 0.25 | 50% | ok |
| factor | ldlt | f64 | 1024x1024 | 214.333 ms | 1.67 | 0.08 | 16% | ok |
| factor | ldlt_bk | f64 | 1024x1024 | 105.382 ms | 3.40 | 0.16 | 32% | ok |
| factor | qr | f64 | 1024x1024 | 133.057 ms | 10.76 | 0.13 | 103% | ok |
| factor | lq | f64 | 1024x1024 | 580.452 ms | 2.47 | 0.03 | 24% | ok |
| factor | hessenberg | f64 | 1024x1024 | 7757.886 ms | 0.46 | 0.00 | 4% | ok |
| factor | householder | f64 | 1024x1024 | 20.343 ms | 0.21 | 0.82 | 8% | ok |
| factor | inv | f64 | 1024x1024 | 992.277 ms | 2.16 | 0.02 | 21% | ok |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 74.063 ms | 9.67 | 0.23 | 92% | ok |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 34.290 ms | 5.22 | 0.12 | 50% | ok |
| eigen | eigen_symmetric | f64 | 512x512 | 2106.146 ms | 0.57 | 0.00 | 5% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 14713.995 ms | 0.08 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 512x512 | 400.882 ms | 3.35 | 0.01 | 32% | ok |
| eigen | schur | f64 | 512x512 | 497.085 ms | 6.75 | 0.01 | 64% | ok |
| eigen | svd | f64 | 512x512 | 176.713 ms | 16.71 | 0.02 | 159% | ok |
| eigen | expm | f64 | 512x512 | 276.505 ms | 7.28 | 0.02 | 69% | ok |
| eigen | sylvester | f64 | 512x512 | 1829.989 ms | 4.40 | 0.00 | 42% | ok |
| elementwise | exp | f64 | n=33554432 | 606.652 ms | 0.06 | 0.88 | 9% | ok |
| elementwise | log | f64 | n=33554432 | 582.849 ms | 0.06 | 0.92 | 9% | ok |
| elementwise | sin | f64 | n=33554432 | 690.476 ms | 0.05 | 0.78 | 8% | ok |
| elementwise | sqrt | f64 | n=33554432 | 364.018 ms | 0.09 | 1.47 | 14% | ok |
| elementwise | erf | f64 | n=33554432 | 10011.858 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 10.411 ms | 1.01 | 10.47 | 102% | ok |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 13.080 ms | 0.80 | 8.33 | 81% | ok |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 63.655 ms | 2.63 | 5.53 | 54% | ok |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 99.602 ms | 0.13 | 0.46 | 4% | ok |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 38.652 ms | 0.26 | 2.17 | 21% | ok |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 304.031 ms | 0.35 | 0.01 | 3% | ok |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 555.473 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 156.306 ms | 0.68 | 0.02 | 7% | ok |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 899.005 ms | 0.24 | 0.00 | 2% | ok |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 874.201 ms | 0.24 | 0.00 | 2% | ok |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 354.207 ms | 0.60 | 0.01 | 6% | ok |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 1114.118 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 44.142 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 63.874 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 582.842 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 2.465 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 251.919 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 26.710 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 268.121 ms | 0.40 | 0.01 | 4% | ok |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 9.195 ms | 0.44 | 1.75 | 17% | ok |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 54.562 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 92.155 ms | 1.16 | 0.03 | 11% | ok |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 9.077 ms | 0.44 | 1.77 | 17% | ok |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 26.933 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 821.201 ms | 0.26 | 0.00 | 2% | ok |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.419 ms | 0.13 | 0.53 | 5% | ok |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 40.871 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 796.420 ms | 0.27 | 0.00 | 3% | ok |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.305 ms | 0.13 | 0.53 | 5% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 23.793 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1924.329 ms | 0.50 | 0.00 | 5% | ok |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 19.462 ms | 0.39 | 1.54 | 15% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 61.931 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 582.663 ms | 1.66 | 0.00 | 16% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 19.619 ms | 0.38 | 1.53 | 15% | ok |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 24.180 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 5716.337 ms | 0.34 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 88.223 ms | 0.08 | 0.34 | 3% | ok |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 34.171 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 4909.173 ms | 0.39 | 0.00 | 4% | ok |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 88.629 ms | 0.08 | 0.34 | 3% | ok |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 1764.962 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 3107.119 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 764.823 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 897.538 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 935.455 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 585.406 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 4876.029 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 1225.340 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 724.576 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 1410.623 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 1302.092 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 1512.457 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 2032.860 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.688 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 256^2 grid | 18.139 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 256^2 grid | 149.756 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 256^2 grid | 12.062 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 256^2 grid | 11.250 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 256^2 grid | 0.377 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 14.045 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 14.990 ms | 0.70 | 7.83 | 76% | ok |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 13.477 ms | 0.78 | 8.71 | 85% | ok |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 16.179 ms | 0.65 | 7.25 | 71% | ok |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 178.981 ms | 0.87 | 6.53 | 63% | ok |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 562.686 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 580.896 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 1418.578 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 172.051 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=33554432 | 755.030 ms | 0.09 | 0.36 | 3% | ok |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 907.870 ms | 2.37 | 0.02 | 23% | ok |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 9.858 ms | 6.81 | 7.23 | 70% | ok |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 26.272 ms | 2.55 | 10.22 | n/a | ok |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.717 ms | 5.28 | 10.55 | n/a | ok |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 21.136 ms | 3.18 | 3.18 | n/a | ok |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 21.520 ms | 3.12 | 3.12 | n/a | ok |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 566.923 ms | 3.79 | 0.02 | n/a | ok |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 364.994 ms | 5.88 | 0.02 | n/a | ok |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 189.780 ms | 11.32 | 0.03 | n/a | ok |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 193.213 ms | 11.11 | 0.03 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 580.807 ms | 3.70 | 0.01 | n/a | ok |

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

## xeon-e5-2420v2 -- blas, 6 threads, class large

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 59.0 GFLOP/s, f32 118.1 GFLOP/s |
| ... hardware, SSE2 | f64 59.0, f32 118.1 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.7 GFLOP/s |
| bandwidth | read 11.3, update 10.6, triad 8.4 GB/s (ceiling 11.3) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 53% | 108% | 0 | 84% |
| blas2 | 10 | 104% | 108% | 0 | 21% |
| blas3 | 10 | 127% | 163% | 0 | 69% |
| factor | 12 | 20% | 40% | 0 | 28% |
| eigen | 8 | 13% | 35% | 0 | 19% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 68% | 93% | 0 | 17% |
| direct | 19 | 1% | 16% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 3% | 14% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 69% | 76% | 0 | 17% |
| mg | 1 | 60% | 60% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 22% | 65% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 93% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 133.984 ms | 0.50 | 4.01 | 35% | ok | 5.6x (93%) |
| blas1 | dot | f32 | n=33554432 | 22.014 ms | 3.05 | 12.19 | 108% | ok | 5.1x (84%) |
| blas1 | axpy | f64 | n=33554432 | 156.253 ms | 0.43 | 5.15 | 45% | ok | 5.4x (91%) |
| blas1 | axpy | f32 | n=33554432 | 67.162 ms | 1.00 | 6.00 | 53% | ok | 5.5x (91%) |
| blas1 | scale | f64 | n=33554432 | 53.840 ms | 0.62 | 9.97 | 88% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=33554432 | 132.647 ms | 0.51 | 2.02 | 18% | ok | 5.6x (93%) |
| blas1 | one_norm | f64 | n=33554432 | 39.391 ms | 0.85 | 6.81 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=33554432 | 52.521 ms | 0.64 | 5.11 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=33554432 | 39.645 ms | 0.85 | 6.77 | 60% | ok | 1.0x (16%) |
| blas1 | dot_k3 | f64 | n=33554432 | 1119.651 ms | 0.66 | 0.48 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=8192 | 45.546 ms | 2.95 | 11.79 | 104% | ok | 1.3x (22%) |
| blas2 | gemv | f32 | n=8192 | 23.398 ms | 5.74 | 11.48 | 101% | ok | 1.2x (19%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 43.930 ms | 3.06 | 12.22 | 108% | ok | 1.3x (21%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 22.420 ms | 5.99 | 11.98 | 106% | ok | 1.2x (21%) |
| blas2 | symv | f64 | n=8192 | 22.740 ms | 5.90 | 11.81 | 104% | ok | 1.3x (22%) |
| blas2 | trmv | f64 | n=8192 | 22.572 ms | 2.97 | 11.90 | 105% | ok | 1.3x (22%) |
| blas2 | trmv | f32 | n=8192 | 11.562 ms | 5.80 | 11.61 | 102% | ok | 1.2x (20%) |
| blas2 | trsv | f64 | n=8192 | 29.924 ms | 2.24 | 8.97 | 79% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=8192 | 13.597 ms | 4.94 | 9.88 | 87% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=8192 | 111.264 ms | 1.21 | 9.65 | 85% | ok | 1.1x (18%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 22.277 ms | 96.40 | 1.13 | 163% | ok | 5.0x (83%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 13.007 ms | 165.10 | 0.97 | 140% | ok | 5.2x (87%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 17.729 ms | 60.56 | 1.42 | 103% | ok | 3.7x (62%) |
| blas3 | symm | f64 | 1024x1024x1024 | 22.369 ms | 96.00 | 1.13 | 163% | ok | 5.0x (84%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 17.752 ms | 60.49 | 0.95 | 103% | ok | 3.5x (58%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 28.699 ms | 74.83 | 0.88 | 127% | ok | 4.2x (69%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 13.856 ms | 77.50 | 1.51 | 131% | ok | 4.1x (69%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 14.567 ms | 73.71 | 1.44 | 125% | ok | 4.1x (69%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 8.611 ms | 124.69 | 1.22 | 106% | ok | 4.6x (77%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 127.771 ms | 0.13 | 1.05 | 9% | ok | 1.0x (17%) |
| factor | lu | f64 | 1024x1024 | 33.809 ms | 21.17 | 0.50 | 36% | ok | 2.0x (33%) |
| factor | lu | f32 | 1024x1024 | 29.826 ms | 24.00 | 0.28 | 20% | ok | 1.7x (28%) |
| factor | cholesky | f64 | 1024x1024 | 29.156 ms | 12.28 | 0.58 | 21% | ok | 1.5x (25%) |
| factor | cholesky | f32 | 1024x1024 | 26.513 ms | 13.50 | 0.32 | 11% | ok | 1.3x (22%) |
| factor | ldlt | f64 | 1024x1024 | 210.472 ms | 1.70 | 0.08 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 1024x1024 | 104.631 ms | 3.42 | 0.16 | 6% | ok | 1.0x (17%) |
| factor | qr | f64 | 1024x1024 | 75.556 ms | 18.95 | 0.22 | 32% | ok | 1.8x (29%) |
| factor | lq | f64 | 1024x1024 | 148.298 ms | 9.65 | 0.11 | 16% | ok | 3.9x (65%) |
| factor | hessenberg | f64 | 1024x1024 | 1862.151 ms | 1.92 | 0.01 | 3% | ok | 4.2x (69%) |
| factor | householder | f64 | 1024x1024 | 3.706 ms | 1.13 | 4.53 | 40% | ok | 5.5x (91%) |
| factor | inv | f64 | 1024x1024 | 955.412 ms | 2.25 | 0.02 | 4% | ok | 1.0x (17%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 47.012 ms | 15.23 | 0.36 | 26% | ok | 1.6x (26%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 23.546 ms | 7.60 | 0.18 | 13% | ok | 1.5x (24%) |
| eigen | eigen_symmetric | f64 | 512x512 | 1840.995 ms | 0.66 | 0.00 | 1% | ok | 1.1x (19%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 14427.954 ms | 0.08 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 512x512 | 380.699 ms | 3.53 | 0.01 | 6% | ok | 1.1x (18%) |
| eigen | schur | f64 | 512x512 | 443.712 ms | 7.56 | 0.01 | 13% | ok | 1.1x (19%) |
| eigen | svd | f64 | 512x512 | 142.478 ms | 20.72 | 0.03 | 35% | ok | 1.2x (21%) |
| eigen | expm | f64 | 512x512 | 192.246 ms | 10.47 | 0.02 | 18% | ok | 1.4x (24%) |
| eigen | sylvester | f64 | 512x512 | 1689.791 ms | 4.77 | 0.00 | 8% | ok | 1.1x (18%) |
| elementwise | exp | f64 | n=33554432 | 611.983 ms | 0.05 | 0.88 | 8% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=33554432 | 559.546 ms | 0.06 | 0.96 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=33554432 | 689.588 ms | 0.05 | 0.78 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=33554432 | 362.471 ms | 0.09 | 1.48 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=33554432 | 9934.926 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 10.304 ms | 1.02 | 10.58 | 93% | ok | 1.0x (17%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 13.541 ms | 0.77 | 8.05 | 71% | ok | 1.0x (16%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 45.571 ms | 3.68 | 7.73 | 68% | ok | 1.4x (23%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 99.722 ms | 0.13 | 0.46 | 4% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 37.608 ms | 0.27 | 2.23 | 20% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 304.499 ms | 0.35 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 574.394 ms | 0.19 | 0.01 | 0% | ok | 1.0x (16%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 157.488 ms | 0.68 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 905.766 ms | 0.24 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 887.520 ms | 0.24 | 0.00 | 0% | ok | 1.0x (16%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 389.369 ms | 0.55 | 0.01 | 1% | ok | 0.9x (15%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 1115.773 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 43.441 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 63.433 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 593.993 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 2.460 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 251.262 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 26.275 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 267.113 ms | 0.40 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 9.057 ms | 0.44 | 1.77 | 16% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 54.945 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 93.305 ms | 1.15 | 0.03 | 2% | ok | 1.0x (16%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 9.101 ms | 0.44 | 1.76 | 16% | ok | 1.0x (17%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 26.897 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 832.362 ms | 0.26 | 0.00 | 0% | ok | 1.0x (16%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.620 ms | 0.13 | 0.52 | 5% | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 40.869 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 800.282 ms | 0.27 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.211 ms | 0.13 | 0.53 | 5% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 23.737 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1949.529 ms | 0.50 | 0.00 | 1% | ok | 1.0x (16%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 19.423 ms | 0.39 | 1.54 | 14% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 61.634 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 609.737 ms | 1.59 | 0.00 | 3% | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 19.525 ms | 0.38 | 1.54 | 14% | ok | 1.0x (17%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 23.892 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 5840.012 ms | 0.33 | 0.00 | 1% | ok | 1.0x (16%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 87.801 ms | 0.09 | 0.34 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 33.757 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 4893.015 ms | 0.40 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 88.363 ms | 0.08 | 0.34 | 3% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 1747.579 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 3007.590 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 769.420 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 890.426 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 923.830 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 576.781 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 4871.759 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 1222.342 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 721.769 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 1392.815 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 1294.968 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 1492.036 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 2053.255 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.687 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 256^2 grid | 17.971 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 256^2 grid | 148.457 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 256^2 grid | 12.966 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| pc | ildl | f64 | 256^2 grid | 11.243 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 256^2 grid | 0.366 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 14.413 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 14.975 ms | 0.70 | 7.84 | 69% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 13.690 ms | 0.77 | 8.57 | 76% | ok | 1.0x (16%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 16.012 ms | 0.65 | 7.33 | 65% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 173.082 ms | 0.90 | 6.75 | 60% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 566.523 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 586.637 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 1428.957 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 171.994 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=33554432 | 752.031 ms | 0.09 | 0.36 | 3% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 162.858 ms | 13.19 | 0.10 | 22% | ok | 5.6x (93%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 9.708 ms | 6.91 | 7.34 | 65% | ok | 1.0x (17%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 26.445 ms | 2.54 | 10.15 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.752 ms | 5.26 | 10.52 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 20.419 ms | 3.29 | 3.29 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 20.426 ms | 3.29 | 3.29 | n/a | ok | 1.1x (18%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 81.347 ms | 26.40 | 0.15 | n/a | ok | 7.0x (116%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 65.417 ms | 32.83 | 0.13 | n/a | ok | 5.6x (93%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 33.774 ms | 63.58 | 0.19 | n/a | ok | 5.6x (94%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 33.754 ms | 63.62 | 0.19 | n/a | ok | 5.7x (95%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 103.366 ms | 20.78 | 0.06 | n/a | ok | 5.6x (94%) |

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

