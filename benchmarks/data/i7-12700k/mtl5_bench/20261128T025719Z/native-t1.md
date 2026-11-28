## i7-12700k -- native, 1 thread, class large

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
| bandwidth | read 32.7, update 38.3, triad 32.4 GB/s (ceiling 38.3) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 43% | 97% | 0 |
| blas2 | 10 | 71% | 94% | 0 |
| blas3 | 10 | 17% | 17% | 0 |
| factor | 12 | 7% | 15% | 0 |
| eigen | 8 | 3% | 11% | 0 |
| elementwise | 5 | 7% | 12% | 0 |
| sparse | 5 | 48% | 80% | 0 |
| direct | 19 | 1% | 18% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 4% | 16% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 61% | 69% | 0 |
| mg | 1 | 45% | 45% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 36% | 70% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=33554432 | 22.574 ms | 2.97 | 23.78 | 62% | ok |
| blas1 | dot | f32 | n=33554432 | 17.064 ms | 3.93 | 15.73 | 41% | ok |
| blas1 | axpy | f64 | n=33554432 | 25.869 ms | 2.59 | 31.13 | 81% | ok |
| blas1 | axpy | f32 | n=33554432 | 18.940 ms | 3.54 | 21.26 | 56% | ok |
| blas1 | scale | f64 | n=33554432 | 14.442 ms | 2.32 | 37.17 | 97% | ok |
| blas1 | two_norm | f64 | n=33554432 | 19.852 ms | 3.38 | 13.52 | 35% | ok |
| blas1 | one_norm | f64 | n=33554432 | 16.447 ms | 2.04 | 16.32 | 43% | ok |
| blas1 | infinity_norm | f64 | n=33554432 | 28.272 ms | 1.19 | 9.49 | 25% | ok |
| blas1 | sum | f64 | n=33554432 | 16.330 ms | 2.05 | 16.44 | 43% | ok |
| blas1 | dot_k3 | f64 | n=33554432 | 322.234 ms | 2.29 | 1.67 | 4% | ok |
| blas2 | gemv | f64 | n=8192 | 15.009 ms | 8.94 | 35.78 | 94% | ok |
| blas2 | gemv | f32 | n=8192 | 10.971 ms | 12.23 | 24.47 | 64% | ok |
| blas2 | gemv_colmajor | f64 | n=8192 | 20.004 ms | 6.71 | 26.84 | 70% | ok |
| blas2 | gemv_colmajor | f32 | n=8192 | 9.879 ms | 13.59 | 27.18 | 71% | ok |
| blas2 | symv | f64 | n=8192 | 13.208 ms | 10.16 | 20.33 | 53% | ok |
| blas2 | trmv | f64 | n=8192 | 8.137 ms | 8.25 | 33.01 | 86% | ok |
| blas2 | trmv | f32 | n=8192 | 5.721 ms | 11.73 | 23.47 | 61% | ok |
| blas2 | trsv | f64 | n=8192 | 8.059 ms | 8.33 | 33.32 | 87% | ok |
| blas2 | trsv | f32 | n=8192 | 5.651 ms | 11.88 | 23.76 | 62% | ok |
| blas2 | ger | f64 | n=8192 | 31.068 ms | 4.32 | 34.57 | 90% | ok |
| blas3 | gemm | f64 | 1024x1024x1024 | 171.251 ms | 12.54 | 0.15 | 16% | ok |
| blas3 | gemm | f32 | 1024x1024x1024 | 81.715 ms | 26.28 | 0.15 | 17% | ok |
| blas3 | gemmt | f64 | 1024x1024x1024 | 99.401 ms | 10.80 | 0.25 | 14% | ok |
| blas3 | symm | f64 | 1024x1024x1024 | 164.900 ms | 13.02 | 0.15 | 17% | ok |
| blas3 | syrk | f64 | 1024x1024x1024 | 103.032 ms | 10.42 | 0.16 | 13% | ok |
| blas3 | syr2k | f64 | 1024x1024x1024 | 191.370 ms | 11.22 | 0.13 | 14% | ok |
| blas3 | trmm | f64 | 1024x1024x1024 | 79.431 ms | 13.52 | 0.26 | 17% | ok |
| blas3 | trsm | f64 | 1024x1024x1024 | 80.071 ms | 13.41 | 0.26 | 17% | ok |
| blas3 | trsm | f32 | 1024x1024x1024 | 39.644 ms | 27.08 | 0.26 | 17% | ok |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 44.504 ms | 0.38 | 3.02 | 8% | ok |
| factor | lu | f64 | 1024x1024 | 105.872 ms | 6.76 | 0.16 | 9% | ok |
| factor | lu | f32 | 1024x1024 | 56.750 ms | 12.61 | 0.15 | 8% | ok |
| factor | cholesky | f64 | 1024x1024 | 69.211 ms | 5.17 | 0.24 | 7% | ok |
| factor | cholesky | f32 | 1024x1024 | 62.627 ms | 5.72 | 0.13 | 4% | ok |
| factor | ldlt | f64 | 1024x1024 | 69.796 ms | 5.13 | 0.24 | 7% | ok |
| factor | ldlt_bk | f64 | 1024x1024 | 42.561 ms | 8.41 | 0.39 | 11% | ok |
| factor | qr | f64 | 1024x1024 | 2271.395 ms | 0.63 | 0.01 | 1% | ok |
| factor | lq | f64 | 1024x1024 | 220.387 ms | 6.50 | 0.08 | 8% | ok |
| factor | hessenberg | f64 | 1024x1024 | 2606.751 ms | 1.37 | 0.01 | 2% | ok |
| factor | householder | f64 | 1024x1024 | 6.875 ms | 0.61 | 2.44 | 6%\* | ok |
| factor | inv | f64 | 1024x1024 | 438.438 ms | 4.90 | 0.04 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 62.338 ms | 11.48 | 0.27 | 15% | ok |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 272.851 ms | 0.66 | 0.02 | 1% | ok |
| eigen | eigen_symmetric | f64 | 512x512 | 970.663 ms | 1.24 | 0.00 | 2% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 12765.492 ms | 0.09 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 512x512 | 452.726 ms | 2.96 | 0.01 | 4% | ok |
| eigen | schur | f64 | 512x512 | 1439.771 ms | 2.33 | 0.00 | 3% | ok |
| eigen | svd | f64 | 512x512 | 1290.856 ms | 2.29 | 0.00 | 3% | ok |
| eigen | expm | f64 | 512x512 | 233.377 ms | 8.63 | 0.02 | 11% | ok |
| eigen | sylvester | f64 | 512x512 | 3015.909 ms | 2.67 | 0.00 | 3% | ok |
| elementwise | exp | f64 | n=33554432 | 163.860 ms | 0.20 | 3.28 | 9% | ok |
| elementwise | log | f64 | n=33554432 | 189.902 ms | 0.18 | 2.83 | 7% | ok |
| elementwise | sin | f64 | n=33554432 | 247.704 ms | 0.14 | 2.17 | 6% | ok |
| elementwise | sqrt | f64 | n=33554432 | 122.014 ms | 0.28 | 4.40 | 12% | ok |
| elementwise | erf | f64 | n=33554432 | 818.495 ms | 0.04 | 0.66 | 2% | ok |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 3.561 ms | 2.94 | 30.61 | 80% | ok |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 4.136 ms | 2.53 | 26.35 | 69% | ok |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 19.160 ms | 8.75 | 18.39 | 48% | ok |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 31.562 ms | 0.41 | 1.46 | 4% | ok |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 14.401 ms | 0.71 | 5.82 | 15% | ok |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 105.361 ms | 1.02 | 0.03 | 1% | ok |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 187.648 ms | 0.57 | 0.02 | 1% | ok |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 50.007 ms | 2.14 | 0.06 | 3% | ok |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 321.875 ms | 0.66 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 305.189 ms | 0.70 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 129.286 ms | 1.66 | 0.02 | 2% | ok |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 371.803 ms | 0.06 | 0.00 | 0% | ok |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 17.133 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 23.849 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 264.945 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 0.911 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 94.332 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 8.509 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 94.897 ms | 1.13 | 0.03 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 2.426 ms | 1.65 | 6.62 | 17%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 18.077 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 29.037 ms | 3.68 | 0.10 | 5% | ok |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 2.277 ms | 1.76 | 7.05 | 18%\* | ok |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 8.170 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 293.172 ms | 0.73 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.774 ms | 0.46 | 1.83 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 13.324 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 273.852 ms | 0.78 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.758 ms | 0.46 | 1.83 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 8.874 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 620.377 ms | 1.56 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 4.834 ms | 1.55 | 6.21 | 16% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 19.209 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 183.100 ms | 5.29 | 0.01 | 7% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 4.822 ms | 1.55 | 6.22 | 16% | ok |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 8.885 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 1922.692 ms | 1.01 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.458 ms | 0.39 | 1.54 | 4% | ok |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 13.063 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1385.372 ms | 1.40 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.457 ms | 0.39 | 1.54 | 4% | ok |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 253.395 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 287.276 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 143.617 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 116.495 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 122.974 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 108.133 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 269.409 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 179.708 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 120.159 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 135.049 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 119.984 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 126.058 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 153.411 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.164 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 256^2 grid | 6.156 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 256^2 grid | 49.648 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 256^2 grid | 4.366 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 256^2 grid | 4.111 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 256^2 grid | 0.117 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 2.851 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 4.433 ms | 2.36 | 26.48 | 69% | ok |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 4.990 ms | 2.10 | 23.52 | 61% | ok |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 6.345 ms | 1.65 | 18.50 | 48% | ok |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 68.264 ms | 2.28 | 17.12 | 45% | ok |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 297.944 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 297.193 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 500.722 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 20.795 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=33554432 | 19.666 ms | 3.41 | 13.65 | 36% | ok |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 311.986 ms | 6.88 | 0.05 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 2.674 ms | 25.09 | 26.66 | 70% | ok |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 9.483 ms | 7.08 | 28.31 | n/a | ok |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 4.703 ms | 14.27 | 28.54 | n/a | ok |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 6.998 ms | 9.59 | 9.59 | n/a | ok |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 6.999 ms | 9.59 | 9.59 | n/a | ok |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 137.137 ms | 15.66 | 0.09 | n/a | ok |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 166.316 ms | 12.91 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 81.536 ms | 26.34 | 0.08 | n/a | ok |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 81.535 ms | 26.34 | 0.08 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 214.263 ms | 10.02 | 0.03 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

