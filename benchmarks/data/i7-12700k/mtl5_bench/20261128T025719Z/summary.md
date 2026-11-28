# mtl5_bench platform characterization

Class `large`, variants native native-fast blas, threads 1,8, CPUs 0,2,4,6,8,10,12,14, clock policy governor performance on the P-cores, turbo on; performance 800-4900 MHz on cpus 0,2,8,10,12,14; performance 800-5000 MHz on cpus 4,6, 2026-11-28T03:45:25Z.

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

## i7-12700k -- native, 8 threads, class large

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 600.8 GFLOP/s, f32 1201.4 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.8, f32 1201.4 GFLOP/s |
| ... this build, scalar | f64 9.9, f32 10.0 GFLOP/s |
| bandwidth | read 83.1, update 78.1, triad 76.7 GB/s (ceiling 83.1) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 86% | 105% | 0 | 29% |
| blas2 | 10 | 42% | 103% | 0 | 13% |
| blas3 | 10 | 2% | 16% | 0 | 14% |
| factor | 12 | 2% | 22% | 0 | 55% |
| eigen | 8 | 0% | 4% | 0 | 15% |
| elementwise | 5 | 3% | 5% | 0 | 13% |
| sparse | 5 | 31% | 110% | 0 | 14% |
| direct | 19 | 0% | 8% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 7% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 14% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 28% | 32% | 0 | 13% |
| mg | 1 | 22% | 22% | 0 | 14% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 33% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 95% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 6.280 ms | 10.69 | 85.49 | 103% | ok | 3.6x (45%) |
| blas1 | dot | f32 | n=33554432 | 3.067 ms | 21.88 | 87.52 | 105% | ok | 5.6x (70%) |
| blas1 | axpy | f64 | n=33554432 | 11.190 ms | 6.00 | 71.96 | 87% | ok | 2.3x (29%) |
| blas1 | axpy | f32 | n=33554432 | 5.611 ms | 11.96 | 71.76 | 86% | ok | 3.4x (42%) |
| blas1 | scale | f64 | n=33554432 | 7.468 ms | 4.49 | 71.89 | 86% | ok | 1.9x (24%) |
| blas1 | two_norm | f64 | n=33554432 | 3.344 ms | 20.07 | 80.27 | 97% | ok | 5.9x (74%) |
| blas1 | one_norm | f64 | n=33554432 | 16.308 ms | 2.06 | 16.46 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=33554432 | 27.871 ms | 1.20 | 9.63 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=33554432 | 16.161 ms | 2.08 | 16.61 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=33554432 | 317.075 ms | 2.33 | 1.69 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=8192 | 6.533 ms | 20.54 | 82.20 | 99% | ok | 2.3x (29%) |
| blas2 | gemv | f32 | n=8192 | 3.135 ms | 42.81 | 85.65 | 103% | ok | 3.5x (44%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 6.490 ms | 20.68 | 82.75 | 100% | ok | 3.1x (39%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 3.278 ms | 40.95 | 81.91 | 99% | ok | 3.0x (38%) |
| blas2 | symv | f64 | n=8192 | 12.946 ms | 10.37 | 20.75 | 25% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=8192 | 8.014 ms | 8.37 | 33.51 | 40% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=8192 | 5.641 ms | 11.90 | 23.80 | 29% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=8192 | 7.898 ms | 8.50 | 34.00 | 41% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=8192 | 5.540 ms | 12.11 | 24.24 | 29% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=8192 | 30.700 ms | 4.37 | 34.98 | 42% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 22.744 ms | 94.42 | 1.11 | 16% | ok | 7.5x (94%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 11.304 ms | 189.98 | 1.11 | 16% | ok | 7.2x (90%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 84.568 ms | 12.70 | 0.30 | 2% | ok | 1.2x (15%) |
| blas3 | symm | f64 | 1024x1024x1024 | 150.881 ms | 14.23 | 0.17 | 2% | ok | 1.1x (14%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 89.852 ms | 11.95 | 0.19 | 2% | ok | 1.1x (14%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 174.682 ms | 12.29 | 0.14 | 2% | ok | 1.1x (14%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 73.977 ms | 14.51 | 0.28 | 2% | ok | 1.1x (13%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 74.268 ms | 14.46 | 0.28 | 2% | ok | 1.1x (13%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 37.853 ms | 28.37 | 0.28 | 2% | ok | 1.0x (13%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 43.491 ms | 0.39 | 3.09 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 1024x1024 | 19.554 ms | 36.61 | 0.86 | 6% | ok | 5.4x (68%) |
| factor | lu | f32 | 1024x1024 | 12.805 ms | 55.90 | 0.66 | 5% | ok | 4.4x (55%) |
| factor | cholesky | f64 | 1024x1024 | 31.296 ms | 11.44 | 0.54 | 2% | ok | 2.2x (28%) |
| factor | cholesky | f32 | 1024x1024 | 29.601 ms | 12.09 | 0.28 | 1% | ok | 2.1x (26%) |
| factor | ldlt | f64 | 1024x1024 | 66.931 ms | 5.35 | 0.25 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 1024x1024 | 40.371 ms | 8.87 | 0.42 | 1% | ok | 1.1x (13%) |
| factor | qr | f64 | 1024x1024 | 433.548 ms | 3.30 | 0.04 | 1% | ok | 5.2x (65%) |
| factor | lq | f64 | 1024x1024 | 45.543 ms | 31.44 | 0.37 | 5% | ok | 4.8x (60%) |
| factor | hessenberg | f64 | 1024x1024 | 506.850 ms | 7.06 | 0.03 | 1% | ok | 5.1x (64%) |
| factor | householder | f64 | 1024x1024 | 0.899 ms | 4.67 | 18.67 | 22%\* | ok | 7.6x (96%) |
| factor | inv | f64 | 1024x1024 | 345.314 ms | 6.22 | 0.05 | 1% | ok | 1.3x (16%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 18.741 ms | 38.19 | 0.90 | 6% | ok | 3.3x (42%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 161.780 ms | 1.11 | 0.03 | 0% | ok | 1.7x (21%) |
| eigen | eigen_symmetric | f64 | 512x512 | 811.280 ms | 1.49 | 0.01 | 0% | ok | 1.2x (15%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 11983.344 ms | 0.10 | 0.00 | 0% | ok | 1.1x (13%) |
| eigen | eigenvalue | f64 | 512x512 | 332.613 ms | 4.04 | 0.01 | 1% | ok | 1.4x (17%) |
| eigen | schur | f64 | 512x512 | 1163.383 ms | 2.88 | 0.00 | 0% | ok | 1.2x (15%) |
| eigen | svd | f64 | 512x512 | 1231.619 ms | 2.40 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | expm | f64 | 512x512 | 87.935 ms | 22.90 | 0.05 | 4% | ok | 2.7x (33%) |
| eigen | sylvester | f64 | 512x512 | 2463.256 ms | 3.27 | 0.00 | 1% | ok | 1.2x (15%) |
| elementwise | exp | f64 | n=33554432 | 162.522 ms | 0.21 | 3.30 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=33554432 | 187.112 ms | 0.18 | 2.87 | 3% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=33554432 | 242.893 ms | 0.14 | 2.21 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=33554432 | 121.512 ms | 0.28 | 4.42 | 5% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=33554432 | 813.483 ms | 0.04 | 0.66 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 1.187 ms | 8.83 | 91.84 | 110% | ok | 3.0x (38%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 4.209 ms | 2.49 | 25.89 | 31% | ok | 1.0x (12%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 6.863 ms | 24.43 | 51.32 | 62% | ok | 2.8x (35%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 44.089 ms | 0.30 | 1.04 | 1% | ok | 0.7x (9%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 13.318 ms | 0.76 | 6.29 | 8% | ok | 1.1x (14%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 103.898 ms | 1.03 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 184.678 ms | 0.58 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 49.509 ms | 2.16 | 0.06 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 313.910 ms | 0.68 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 299.801 ms | 0.71 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 129.093 ms | 1.66 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 362.315 ms | 0.06 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 16.383 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 23.046 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 263.856 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 0.899 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 91.695 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 8.349 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 93.128 ms | 1.15 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 2.641 ms | 1.52 | 6.08 | 7%\* | ok | 0.9x (11%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 17.942 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 28.675 ms | 3.73 | 0.10 | 1% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 2.523 ms | 1.59 | 6.36 | 8%\* | ok | 0.9x (11%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 7.924 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 285.860 ms | 0.75 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.791 ms | 0.46 | 1.83 | 2%\* | ok | 1.0x (12%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 12.888 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 273.150 ms | 0.78 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.856 ms | 0.45 | 1.81 | 2%\* | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 8.801 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 604.652 ms | 1.60 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 5.122 ms | 1.46 | 5.86 | 7% | ok | 0.9x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 19.323 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 180.005 ms | 5.38 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 5.097 ms | 1.47 | 5.88 | 7% | ok | 0.9x (12%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 8.787 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 1956.739 ms | 0.99 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.983 ms | 0.38 | 1.50 | 2% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 12.919 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1370.897 ms | 1.41 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.799 ms | 0.38 | 1.51 | 2% | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 238.068 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 227.584 ms | 0.00 | 0.00 | n/a | ok | 1.3x (16%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 149.843 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 106.960 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 113.741 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 99.898 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 279.403 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 186.396 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 110.275 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 124.037 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 110.727 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 116.626 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 155.757 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.159 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 256^2 grid | 6.059 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 256^2 grid | 48.945 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 256^2 grid | 4.311 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 256^2 grid | 4.024 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 256^2 grid | 0.108 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 2.816 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 4.451 ms | 2.35 | 26.37 | 32% | ok | 1.0x (12%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 4.986 ms | 2.10 | 23.54 | 28% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 6.315 ms | 1.66 | 18.59 | 22% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 62.907 ms | 2.47 | 18.58 | 22% | ok | 1.1x (14%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 300.281 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 299.030 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 492.572 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 20.483 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=33554432 | 19.301 ms | 3.48 | 13.91 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 41.085 ms | 52.27 | 0.41 | 9% | ok | 7.6x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 2.636 ms | 25.46 | 27.05 | 33% | ok | 1.0x (13%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 9.399 ms | 7.14 | 28.56 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 4.681 ms | 14.34 | 28.67 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 6.899 ms | 9.73 | 9.73 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 6.960 ms | 9.64 | 9.64 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 18.086 ms | 118.74 | 0.70 | n/a | ok | 7.6x (95%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 21.894 ms | 98.08 | 0.38 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 10.742 ms | 199.92 | 0.59 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 10.751 ms | 199.75 | 0.59 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 28.139 ms | 76.32 | 0.22 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## i7-12700k -- native-fast, 1 thread, class large

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.5 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.5 GFLOP/s |
| ... this build, AVX2 | f64 78.2, f32 156.4 GFLOP/s |
| bandwidth | read 37.6, update 41.1, triad 31.4 GB/s (ceiling 41.1) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 86% | 101% | 0 |
| blas2 | 10 | 91% | 94% | 0 |
| blas3 | 10 | 76% | 92% | 0 |
| factor | 12 | 7% | 17% | 0 |
| eigen | 8 | 3% | 18% | 0 |
| elementwise | 5 | 5% | 8% | 0 |
| sparse | 5 | 30% | 37% | 0 |
| direct | 19 | 1% | 15% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 4% | 11% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 48% | 55% | 0 |
| mg | 1 | 40% | 40% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 78% | 79% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=33554432 | 15.250 ms | 4.40 | 35.20 | 86% | ok |
| blas1 | dot | f32 | n=33554432 | 7.581 ms | 8.85 | 35.41 | 86% | ok |
| blas1 | axpy | f64 | n=33554432 | 19.536 ms | 3.44 | 41.22 | 100% | ok |
| blas1 | axpy | f32 | n=33554432 | 9.725 ms | 6.90 | 41.40 | 101% | ok |
| blas1 | scale | f64 | n=33554432 | 13.999 ms | 2.40 | 38.35 | 93% | ok |
| blas1 | two_norm | f64 | n=33554432 | 9.713 ms | 6.91 | 27.64 | 67% | ok |
| blas1 | one_norm | f64 | n=33554432 | 16.555 ms | 2.03 | 16.21 | 39% | ok |
| blas1 | infinity_norm | f64 | n=33554432 | 28.326 ms | 1.18 | 9.48 | 23% | ok |
| blas1 | sum | f64 | n=33554432 | 16.289 ms | 2.06 | 16.48 | 40% | ok |
| blas1 | dot_k3 | f64 | n=33554432 | 302.506 ms | 2.44 | 1.77 | 4% | ok |
| blas2 | gemv | f64 | n=8192 | 13.892 ms | 9.66 | 38.66 | 94% | ok |
| blas2 | gemv | f32 | n=8192 | 6.925 ms | 19.38 | 38.77 | 94% | ok |
| blas2 | gemv_colmajor | f64 | n=8192 | 14.316 ms | 9.38 | 37.51 | 91% | ok |
| blas2 | gemv_colmajor | f32 | n=8192 | 7.109 ms | 18.88 | 37.77 | 92% | ok |
| blas2 | symv | f64 | n=8192 | 12.387 ms | 10.84 | 21.68 | 53% | ok |
| blas2 | trmv | f64 | n=8192 | 7.259 ms | 9.25 | 37.00 | 90% | ok |
| blas2 | trmv | f32 | n=8192 | 3.688 ms | 18.20 | 36.41 | 89% | ok |
| blas2 | trsv | f64 | n=8192 | 7.387 ms | 9.09 | 36.36 | 89% | ok |
| blas2 | trsv | f32 | n=8192 | 3.727 ms | 18.01 | 36.03 | 88% | ok |
| blas2 | ger | f64 | n=8192 | 28.817 ms | 4.66 | 37.27 | 91% | ok |
| blas3 | gemm | f64 | 1024x1024x1024 | 30.792 ms | 69.74 | 0.82 | 89% | ok |
| blas3 | gemm | f32 | 1024x1024x1024 | 14.963 ms | 143.52 | 0.84 | 92% | ok |
| blas3 | gemmt | f64 | 1024x1024x1024 | 19.359 ms | 55.47 | 1.30 | 71% | ok |
| blas3 | symm | f64 | 1024x1024x1024 | 33.496 ms | 64.11 | 0.75 | 82% | ok |
| blas3 | syrk | f64 | 1024x1024x1024 | 17.705 ms | 60.65 | 0.95 | 77% | ok |
| blas3 | syr2k | f64 | 1024x1024x1024 | 36.086 ms | 59.51 | 0.70 | 76% | ok |
| blas3 | trmm | f64 | 1024x1024x1024 | 22.225 ms | 48.31 | 0.94 | 62% | ok |
| blas3 | trsm | f64 | 1024x1024x1024 | 22.879 ms | 46.93 | 0.92 | 60% | ok |
| blas3 | trsm | f32 | 1024x1024x1024 | 10.538 ms | 101.89 | 1.00 | 65% | ok |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 44.912 ms | 0.37 | 2.99 | 7% | ok |
| factor | lu | f64 | 1024x1024 | 101.362 ms | 7.06 | 0.17 | 9% | ok |
| factor | lu | f32 | 1024x1024 | 55.916 ms | 12.80 | 0.15 | 8% | ok |
| factor | cholesky | f64 | 1024x1024 | 68.867 ms | 5.20 | 0.24 | 7% | ok |
| factor | cholesky | f32 | 1024x1024 | 62.368 ms | 5.74 | 0.13 | 4% | ok |
| factor | ldlt | f64 | 1024x1024 | 69.250 ms | 5.17 | 0.24 | 7% | ok |
| factor | ldlt_bk | f64 | 1024x1024 | 40.301 ms | 8.88 | 0.42 | 11% | ok |
| factor | qr | f64 | 1024x1024 | 2270.336 ms | 0.63 | 0.01 | 1% | ok |
| factor | lq | f64 | 1024x1024 | 213.061 ms | 6.72 | 0.08 | 9% | ok |
| factor | hessenberg | f64 | 1024x1024 | 2584.049 ms | 1.39 | 0.01 | 2% | ok |
| factor | householder | f64 | 1024x1024 | 7.991 ms | 0.53 | 2.10 | 5%\* | ok |
| factor | inv | f64 | 1024x1024 | 463.873 ms | 4.63 | 0.04 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 52.549 ms | 13.62 | 0.32 | 17% | ok |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 256.205 ms | 0.70 | 0.02 | 1% | ok |
| eigen | eigen_symmetric | f64 | 512x512 | 886.876 ms | 1.36 | 0.00 | 2% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 11928.339 ms | 0.10 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 512x512 | 462.052 ms | 2.90 | 0.01 | 4% | ok |
| eigen | schur | f64 | 512x512 | 1411.903 ms | 2.38 | 0.00 | 3% | ok |
| eigen | svd | f64 | 512x512 | 1022.945 ms | 2.89 | 0.00 | 4% | ok |
| eigen | expm | f64 | 512x512 | 145.395 ms | 13.85 | 0.03 | 18% | ok |
| eigen | sylvester | f64 | 512x512 | 3095.851 ms | 2.60 | 0.00 | 3% | ok |
| elementwise | exp | f64 | n=33554432 | 273.473 ms | 0.12 | 1.96 | 5% | ok |
| elementwise | log | f64 | n=33554432 | 286.450 ms | 0.12 | 1.87 | 5% | ok |
| elementwise | sin | f64 | n=33554432 | 378.494 ms | 0.09 | 1.42 | 3% | ok |
| elementwise | sqrt | f64 | n=33554432 | 154.165 ms | 0.22 | 3.48 | 8% | ok |
| elementwise | erf | f64 | n=33554432 | 1083.069 ms | 0.03 | 0.50 | 1% | ok |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 7.522 ms | 1.39 | 14.49 | 35% | ok |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 7.257 ms | 1.44 | 15.02 | 37% | ok |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 28.996 ms | 5.78 | 12.15 | 30% | ok |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 44.638 ms | 0.29 | 1.03 | 3% | ok |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 22.656 ms | 0.45 | 3.70 | 9% | ok |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 165.637 ms | 0.65 | 0.02 | 1% | ok |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 206.622 ms | 0.52 | 0.01 | 1% | ok |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 56.631 ms | 1.89 | 0.05 | 2% | ok |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 311.663 ms | 0.69 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 330.467 ms | 0.65 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 129.718 ms | 1.65 | 0.02 | 2% | ok |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 385.105 ms | 0.06 | 0.00 | 0% | ok |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 17.515 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 25.175 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 268.570 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 0.948 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 94.502 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 8.876 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 97.606 ms | 1.10 | 0.03 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 2.688 ms | 1.49 | 5.97 | 15%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 18.721 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 32.660 ms | 3.28 | 0.09 | 4% | ok |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 2.668 ms | 1.50 | 6.02 | 15%\* | ok |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 8.461 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 280.380 ms | 0.76 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 9.070 ms | 0.44 | 1.77 | 4%\* | ok |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 13.709 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 297.360 ms | 0.72 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 9.091 ms | 0.44 | 1.77 | 4%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 9.077 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 727.844 ms | 1.33 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 6.897 ms | 1.09 | 4.35 | 11% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 22.678 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 222.881 ms | 4.35 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 7.145 ms | 1.05 | 4.20 | 10% | ok |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 9.835 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 2410.790 ms | 0.80 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 20.502 ms | 0.37 | 1.46 | 4% | ok |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 13.491 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 2294.596 ms | 0.84 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 21.940 ms | 0.34 | 1.37 | 3% | ok |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 280.855 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 405.971 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 174.036 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 142.344 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 184.582 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 162.165 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 354.436 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 250.091 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 155.031 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 151.261 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 136.983 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 176.245 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 208.809 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.267 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 256^2 grid | 11.230 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 256^2 grid | 85.718 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 256^2 grid | 7.475 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 256^2 grid | 7.619 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 256^2 grid | 0.271 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 5.091 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 6.736 ms | 1.56 | 17.42 | 42% | ok |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 5.241 ms | 2.00 | 22.39 | 55% | ok |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 6.009 ms | 1.74 | 19.53 | 48% | ok |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 71.716 ms | 2.17 | 16.29 | 40% | ok |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 333.285 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 305.222 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 503.649 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 19.177 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=33554432 | 8.408 ms | 7.98 | 31.93 | 78% | ok |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 48.388 ms | 44.38 | 0.35 | 57% | ok |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 2.199 ms | 30.51 | 32.42 | 79% | ok |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 8.147 ms | 8.24 | 32.95 | n/a | ok |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 3.979 ms | 16.86 | 33.73 | n/a | ok |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 2.176 ms | 30.83 | 30.83 | n/a | ok |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 2.121 ms | 31.64 | 31.64 | n/a | ok |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 35.641 ms | 60.25 | 0.35 | n/a | ok |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 48.279 ms | 44.48 | 0.17 | n/a | ok |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 48.281 ms | 44.48 | 0.13 | n/a | ok |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 18.240 ms | 117.73 | 0.34 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 15.950 ms | 134.64 | 0.39 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## i7-12700k -- native-fast, 8 threads, class large

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 425.6 GFLOP/s, f32 852.2 GFLOP/s |
| ... hardware, AVX2+FMA | f64 400.4, f32 852.2 GFLOP/s |
| ... this build, AVX2 | f64 425.6, f32 772.7 GFLOP/s |
| bandwidth | read 42.3, update 53.9, triad 47.9 GB/s (ceiling 53.9) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 106% | 153% | 0 | 19% |
| blas2 | 10 | 70% | 156% | 0 | 13% |
| blas3 | 10 | 48% | 112% | 0 | 46% |
| factor | 12 | 3% | 35% | 0 | 60% |
| eigen | 8 | 1% | 7% | 0 | 17% |
| elementwise | 5 | 5% | 8% | 0 | 19% |
| sparse | 5 | 49% | 174% | 0 | 22% |
| direct | 19 | 0% | 11% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 3% | 11% | 0 | 14% |
| krylov | 13 | n/a | n/a | 0 | 19% |
| pc | 7 | n/a | n/a | 0 | 23% |
| smoother | 3 | 44% | 50% | 0 | 13% |
| mg | 1 | 35% | 35% | 0 | 15% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 62% | 76% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 97% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 9.438 ms | 7.11 | 56.89 | 106% | ok | 1.6x (20%) |
| blas1 | dot | f32 | n=33554432 | 3.882 ms | 17.29 | 69.14 | 128% | ok | 2.0x (24%) |
| blas1 | axpy | f64 | n=33554432 | 12.838 ms | 5.23 | 62.73 | 116% | ok | 1.5x (19%) |
| blas1 | axpy | f32 | n=33554432 | 7.653 ms | 8.77 | 52.61 | 98% | ok | 1.3x (16%) |
| blas1 | scale | f64 | n=33554432 | 8.692 ms | 3.86 | 61.77 | 115% | ok | 1.6x (20%) |
| blas1 | two_norm | f64 | n=33554432 | 3.247 ms | 20.66 | 82.66 | 153% ! | ok | 3.0x (37%) |
| blas1 | one_norm | f64 | n=33554432 | 17.081 ms | 1.96 | 15.72 | 29% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=33554432 | 28.861 ms | 1.16 | 9.30 | 17% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=33554432 | 16.844 ms | 1.99 | 15.94 | 30% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=33554432 | 309.184 ms | 2.39 | 1.74 | 3% | ok | 1.0x (12%) |
| blas2 | gemv | f64 | n=8192 | 7.332 ms | 18.31 | 73.24 | 136% | ok | 1.9x (24%) |
| blas2 | gemv | f32 | n=8192 | 3.194 ms | 42.02 | 84.06 | 156% ! | ok | 2.2x (27%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 6.655 ms | 20.17 | 80.69 | 150% | ok | 2.2x (27%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 3.461 ms | 38.78 | 77.58 | 144% | ok | 2.1x (26%) |
| blas2 | symv | f64 | n=8192 | 12.846 ms | 10.45 | 20.91 | 39% | ok | 1.0x (12%) |
| blas2 | trmv | f64 | n=8192 | 7.161 ms | 9.37 | 37.50 | 70% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=8192 | 3.642 ms | 18.43 | 36.87 | 68% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=8192 | 7.153 ms | 9.38 | 37.55 | 70% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=8192 | 3.659 ms | 18.34 | 36.70 | 68% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=8192 | 28.525 ms | 4.71 | 37.65 | 70% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 4.520 ms | 475.12 | 5.57 | 112% | ok | 6.8x (85%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 2.279 ms | 942.26 | 5.52 | 111% | ok | 6.6x (82%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 5.228 ms | 205.39 | 4.81 | 48% | ok | 3.7x (46%) |
| blas3 | symm | f64 | 1024x1024x1024 | 6.738 ms | 318.70 | 3.73 | 75% | ok | 5.0x (62%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 5.306 ms | 202.35 | 3.16 | 48% | ok | 3.3x (42%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 9.821 ms | 218.65 | 2.56 | 51% | ok | 3.7x (46%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 9.542 ms | 112.52 | 2.20 | 26% | ok | 2.3x (29%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 10.237 ms | 104.89 | 2.05 | 25% | ok | 2.2x (28%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 5.155 ms | 208.29 | 2.03 | 24% | ok | 2.0x (26%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 43.532 ms | 0.39 | 3.08 | 6% | ok | 1.0x (13%) |
| factor | lu | f64 | 1024x1024 | 16.331 ms | 43.83 | 1.03 | 10% | ok | 6.2x (78%) |
| factor | lu | f32 | 1024x1024 | 11.617 ms | 61.62 | 0.72 | 7% | ok | 4.8x (60%) |
| factor | cholesky | f64 | 1024x1024 | 31.052 ms | 11.53 | 0.54 | 3% | ok | 2.2x (28%) |
| factor | cholesky | f32 | 1024x1024 | 29.450 ms | 12.15 | 0.28 | 1% | ok | 2.1x (26%) |
| factor | ldlt | f64 | 1024x1024 | 66.163 ms | 5.41 | 0.25 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 1024x1024 | 38.422 ms | 9.32 | 0.44 | 2% | ok | 1.0x (13%) |
| factor | qr | f64 | 1024x1024 | 429.834 ms | 3.33 | 0.04 | 1% | ok | 5.3x (66%) |
| factor | lq | f64 | 1024x1024 | 40.838 ms | 35.06 | 0.41 | 8% | ok | 5.2x (65%) |
| factor | hessenberg | f64 | 1024x1024 | 491.465 ms | 7.28 | 0.03 | 2% | ok | 5.3x (66%) |
| factor | householder | f64 | 1024x1024 | 0.889 ms | 4.72 | 18.86 | 35%\* | ok | 9.0x (112%) |
| factor | inv | f64 | 1024x1024 | 353.480 ms | 6.08 | 0.05 | 1% | ok | 1.3x (16%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 17.775 ms | 40.27 | 0.94 | 9% | ok | 3.0x (37%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 153.016 ms | 1.17 | 0.03 | 0% | ok | 1.7x (21%) |
| eigen | eigen_symmetric | f64 | 512x512 | 777.605 ms | 1.55 | 0.01 | 0% | ok | 1.1x (14%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 11624.628 ms | 0.10 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 512x512 | 323.318 ms | 4.15 | 0.01 | 1% | ok | 1.4x (18%) |
| eigen | schur | f64 | 512x512 | 1119.792 ms | 3.00 | 0.00 | 1% | ok | 1.3x (16%) |
| eigen | svd | f64 | 512x512 | 1151.943 ms | 2.56 | 0.00 | 1% | ok | 0.9x (11%) |
| eigen | expm | f64 | 512x512 | 71.745 ms | 28.06 | 0.06 | 7% | ok | 2.0x (25%) |
| eigen | sylvester | f64 | 512x512 | 2342.988 ms | 3.44 | 0.00 | 1% | ok | 1.3x (17%) |
| elementwise | exp | f64 | n=33554432 | 161.202 ms | 0.21 | 3.33 | 6% | ok | 1.7x (21%) |
| elementwise | log | f64 | n=33554432 | 185.168 ms | 0.18 | 2.90 | 5% | ok | 1.5x (19%) |
| elementwise | sin | f64 | n=33554432 | 242.319 ms | 0.14 | 2.22 | 4% | ok | 1.6x (20%) |
| elementwise | sqrt | f64 | n=33554432 | 120.192 ms | 0.28 | 4.47 | 8% | ok | 1.3x (16%) |
| elementwise | erf | f64 | n=33554432 | 807.811 ms | 0.04 | 0.66 | 1% | ok | 1.3x (17%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 1.165 ms | 8.99 | 93.54 | 174% ! | ok | 6.5x (81%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 4.146 ms | 2.53 | 26.29 | 49% | ok | 1.8x (22%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 6.838 ms | 24.51 | 51.51 | 96% | ok | 4.2x (53%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 41.987 ms | 0.31 | 1.10 | 2% | ok | 1.1x (13%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 13.221 ms | 0.77 | 6.34 | 12% | ok | 1.7x (21%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 102.716 ms | 1.04 | 0.03 | 0% | ok | 1.6x (20%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 192.822 ms | 0.55 | 0.02 | 0% | ok | 1.1x (13%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 53.242 ms | 2.01 | 0.06 | 0% | ok | 1.1x (13%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 296.381 ms | 0.72 | 0.01 | 0% | ok | 1.1x (13%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 326.398 ms | 0.66 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 130.236 ms | 1.64 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 374.645 ms | 0.06 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 16.319 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 23.029 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 260.594 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 0.903 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 92.815 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 8.389 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 94.360 ms | 1.13 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 2.660 ms | 1.51 | 6.04 | 11%\* | ok | 1.0x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 17.908 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 32.183 ms | 3.32 | 0.09 | 1% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 2.672 ms | 1.50 | 6.01 | 11%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 8.030 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 274.576 ms | 0.78 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.747 ms | 0.46 | 1.84 | 3%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 13.064 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 295.596 ms | 0.72 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.778 ms | 0.46 | 1.83 | 3%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 8.707 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 585.527 ms | 1.66 | 0.00 | 0% | ok | 1.2x (16%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 5.003 ms | 1.50 | 6.00 | 11% | ok | 1.4x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 18.938 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 205.557 ms | 4.71 | 0.01 | 1% | ok | 1.1x (14%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 5.417 ms | 1.38 | 5.54 | 10% | ok | 1.3x (16%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 8.756 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 1753.100 ms | 1.11 | 0.00 | 0% | ok | 1.4x (17%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.609 ms | 0.38 | 1.53 | 3% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 12.812 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 2197.485 ms | 0.88 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.619 ms | 0.38 | 1.53 | 3% | ok | 1.1x (14%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 213.357 ms | 0.00 | 0.00 | n/a | ok | 1.3x (16%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 195.777 ms | 0.00 | 0.00 | n/a | ok | 2.1x (26%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 134.378 ms | 0.00 | 0.00 | n/a | ok | 1.3x (16%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 96.235 ms | 0.00 | 0.00 | n/a | ok | 1.5x (18%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 102.013 ms | 0.00 | 0.00 | n/a | ok | 1.8x (23%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 92.864 ms | 0.00 | 0.00 | n/a | ok | 1.7x (22%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 195.844 ms | 0.00 | 0.00 | n/a | ok | 1.8x (23%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 167.117 ms | 0.00 | 0.00 | n/a | ok | 1.5x (19%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 101.999 ms | 0.00 | 0.00 | n/a | ok | 1.5x (19%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 103.243 ms | 0.00 | 0.00 | n/a | ok | 1.5x (18%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 92.681 ms | 0.00 | 0.00 | n/a | ok | 1.5x (18%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 97.152 ms | 0.00 | 0.00 | n/a | ok | 1.8x (23%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 113.966 ms | 0.00 | 0.00 | n/a | ok | 1.8x (23%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.138 ms | 0.00 | 0.00 | n/a | ok | 1.9x (24%) |
| pc | ilu_0 | f64 | 256^2 grid | 6.165 ms | 0.00 | 0.00 | n/a | ok | 1.8x (23%) |
| pc | ilut | f64 | 256^2 grid | 50.420 ms | 0.00 | 0.00 | n/a | ok | 1.7x (21%) |
| pc | ic_0 | f64 | 256^2 grid | 4.308 ms | 0.00 | 0.00 | n/a | ok | 1.7x (22%) |
| pc | ildl | f64 | 256^2 grid | 4.016 ms | 0.00 | 0.00 | n/a | ok | 1.9x (24%) |
| pc | ssor | f64 | 256^2 grid | 0.109 ms | 0.00 | 0.00 | n/a | ok | 2.5x (31%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 2.825 ms | 0.00 | 0.00 | n/a | ok | 1.8x (23%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 4.368 ms | 2.40 | 26.87 | 50% | ok | 1.5x (19%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 4.951 ms | 2.12 | 23.71 | 44% | ok | 1.1x (13%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 5.755 ms | 1.82 | 20.39 | 38% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 61.502 ms | 2.53 | 19.00 | 35% | ok | 1.2x (15%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 288.350 ms | 0.00 | 0.00 | n/a | ok | 1.2x (14%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 287.552 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 488.614 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 18.732 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=33554432 | 8.145 ms | 8.24 | 32.96 | 61% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 6.652 ms | 322.84 | 2.52 | 76% | ok | 7.3x (91%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 2.118 ms | 31.68 | 33.66 | 62% | ok | 1.0x (13%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 7.904 ms | 8.49 | 33.96 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 3.835 ms | 17.50 | 35.00 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 2.087 ms | 32.16 | 32.16 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 2.075 ms | 32.34 | 32.34 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 4.425 ms | 485.35 | 2.84 | n/a | ok | 8.1x (101%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 6.155 ms | 348.90 | 1.36 | n/a | ok | 7.8x (98%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 6.120 ms | 350.91 | 1.03 | n/a | ok | 7.9x (99%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 2.319 ms | 926.00 | 2.71 | n/a | ok | 7.9x (98%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 2.060 ms | 1042.52 | 3.05 | n/a | ok | 7.7x (97%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

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

## i7-12700k -- blas, 8 threads, class large

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 600.6 GFLOP/s, f32 1201.4 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.6, f32 1201.4 GFLOP/s |
| ... this build, scalar | f64 10.0, f32 9.9 GFLOP/s |
| bandwidth | read 82.4, update 77.5, triad 75.5 GB/s (ceiling 82.4) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 88% | 106% | 0 | 29% |
| blas2 | 10 | 102% | 106% | 0 | 28% |
| blas3 | 10 | 75% | 92% | 0 | 82% |
| factor | 12 | 8% | 23% | 0 | 22% |
| eigen | 8 | 4% | 15% | 0 | 15% |
| elementwise | 5 | 3% | 5% | 0 | 13% |
| sparse | 5 | 32% | 114% | 0 | 14% |
| direct | 31 | 1% | 13% | 0 | 14% |
| ordering | 5 | n/a | n/a | 0 | 14% |
| direct3d | 21 | 7% | 31% | 0 | 14% |
| krylov | 13 | n/a | n/a | 0 | 14% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 33% | 0 | 13% |
| mg | 1 | 23% | 23% | 0 | 14% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 33% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 90% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 6.306 ms | 10.64 | 85.13 | 103% | ok | 3.7x (46%) |
| blas1 | dot | f32 | n=33554432 | 3.086 ms | 21.74 | 86.97 | 106% | ok | 5.6x (70%) |
| blas1 | axpy | f64 | n=33554432 | 11.151 ms | 6.02 | 72.22 | 88% | ok | 2.3x (29%) |
| blas1 | axpy | f32 | n=33554432 | 5.567 ms | 12.05 | 72.33 | 88% | ok | 3.4x (43%) |
| blas1 | scale | f64 | n=33554432 | 7.460 ms | 4.50 | 71.97 | 87% | ok | 1.9x (24%) |
| blas1 | two_norm | f64 | n=33554432 | 3.273 ms | 20.50 | 82.02 | 99% | ok | 6.1x (77%) |
| blas1 | one_norm | f64 | n=33554432 | 16.314 ms | 2.06 | 16.45 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=33554432 | 27.872 ms | 1.20 | 9.63 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=33554432 | 16.159 ms | 2.08 | 16.61 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=33554432 | 316.006 ms | 2.34 | 1.70 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=8192 | 6.556 ms | 20.47 | 81.92 | 99% | ok | 2.2x (27%) |
| blas2 | gemv | f32 | n=8192 | 3.196 ms | 41.99 | 84.00 | 102% | ok | 2.2x (28%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 6.404 ms | 20.96 | 83.85 | 102% | ok | 2.3x (29%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 3.138 ms | 42.78 | 85.57 | 104% | ok | 2.3x (28%) |
| blas2 | symv | f64 | n=8192 | 3.186 ms | 42.12 | 84.29 | 102% | ok | 2.4x (30%) |
| blas2 | trmv | f64 | n=8192 | 3.177 ms | 21.13 | 84.55 | 103% | ok | 2.4x (30%) |
| blas2 | trmv | f32 | n=8192 | 1.537 ms | 43.67 | 87.37 | 106% | ok | 2.6x (32%) |
| blas2 | trsv | f64 | n=8192 | 7.190 ms | 9.33 | 37.35 | 45% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=8192 | 3.571 ms | 18.79 | 37.61 | 46% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=8192 | 15.173 ms | 8.85 | 70.78 | 86% | ok | 1.8x (23%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 4.057 ms | 529.37 | 6.20 | 88% | ok | 7.2x (90%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 1.951 ms | 1100.85 | 6.45 | 92% | ok | 7.3x (92%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 3.629 ms | 295.86 | 6.93 | 49% | ok | 4.9x (61%) |
| blas3 | symm | f64 | 1024x1024x1024 | 4.078 ms | 526.55 | 6.17 | 88% | ok | 7.2x (90%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 4.290 ms | 250.30 | 3.91 | 42% | ok | 4.3x (54%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 6.713 ms | 319.91 | 3.75 | 53% | ok | 5.4x (67%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 2.398 ms | 447.82 | 8.75 | 75% | ok | 6.5x (81%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 2.386 ms | 450.06 | 8.79 | 75% | ok | 6.6x (82%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 1.088 ms | 986.72 | 9.64 | 82% | ok | 6.8x (85%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 44.200 ms | 0.38 | 3.04 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 1024x1024 | 8.514 ms | 84.07 | 1.97 | 14% | ok | 2.2x (27%) |
| factor | lu | f32 | 1024x1024 | 6.991 ms | 102.39 | 1.20 | 9% | ok | 1.8x (22%) |
| factor | cholesky | f64 | 1024x1024 | 7.727 ms | 46.32 | 2.17 | 8% | ok | 1.5x (19%) |
| factor | cholesky | f32 | 1024x1024 | 6.205 ms | 57.68 | 1.35 | 5% | ok | 1.3x (16%) |
| factor | ldlt | f64 | 1024x1024 | 67.050 ms | 5.34 | 0.25 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 1024x1024 | 39.531 ms | 9.05 | 0.42 | 2% | ok | 1.1x (14%) |
| factor | qr | f64 | 1024x1024 | 20.223 ms | 70.79 | 0.83 | 12% | ok | 1.8x (22%) |
| factor | lq | f64 | 1024x1024 | 45.853 ms | 31.22 | 0.37 | 5% | ok | 4.8x (60%) |
| factor | hessenberg | f64 | 1024x1024 | 513.014 ms | 6.98 | 0.03 | 1% | ok | 5.2x (65%) |
| factor | householder | f64 | 1024x1024 | 0.898 ms | 4.68 | 18.69 | 23%\* | ok | 7.7x (96%) |
| factor | inv | f64 | 1024x1024 | 335.909 ms | 6.39 | 0.05 | 1% | ok | 1.1x (13%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 13.487 ms | 53.08 | 1.24 | 9% | ok | 1.6x (20%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 6.709 ms | 26.67 | 0.63 | 4% | ok | 1.4x (17%) |
| eigen | eigen_symmetric | f64 | 512x512 | 808.955 ms | 1.49 | 0.01 | 0% | ok | 1.2x (15%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 12547.254 ms | 0.10 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 512x512 | 115.830 ms | 11.59 | 0.04 | 2% | ok | 1.0x (13%) |
| eigen | schur | f64 | 512x512 | 128.305 ms | 26.15 | 0.03 | 4% | ok | 1.1x (14%) |
| eigen | svd | f64 | 512x512 | 32.682 ms | 90.35 | 0.13 | 15% | ok | 1.4x (17%) |
| eigen | expm | f64 | 512x512 | 68.228 ms | 29.51 | 0.06 | 5% | ok | 1.4x (17%) |
| eigen | sylvester | f64 | 512x512 | 422.132 ms | 19.08 | 0.01 | 3% | ok | 1.1x (13%) |
| elementwise | exp | f64 | n=33554432 | 161.257 ms | 0.21 | 3.33 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=33554432 | 188.217 ms | 0.18 | 2.85 | 3% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=33554432 | 242.842 ms | 0.14 | 2.21 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=33554432 | 120.099 ms | 0.28 | 4.47 | 5% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=33554432 | 807.370 ms | 0.04 | 0.66 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 1.164 ms | 9.00 | 93.65 | 114% | ok | 3.2x (39%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 4.150 ms | 2.52 | 26.26 | 32% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 6.840 ms | 24.51 | 51.50 | 62% | ok | 2.8x (35%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 29.262 ms | 0.45 | 1.57 | 2% | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 13.253 ms | 0.77 | 6.32 | 8% | ok | 1.1x (14%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 100.691 ms | 1.06 | 0.03 | 0% | ok | 1.1x (13%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 184.568 ms | 0.58 | 0.02 | 0% | ok | 1.1x (13%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 49.503 ms | 2.16 | 0.06 | 0% | ok | 1.1x (14%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 310.186 ms | 0.69 | 0.01 | 0% | ok | 1.1x (13%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 287.356 ms | 0.74 | 0.01 | 0% | ok | 1.2x (15%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 134.989 ms | 1.59 | 0.02 | 0% | ok | 1.1x (14%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 366.144 ms | 0.06 | 0.00 | 0% | ok | 1.1x (14%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 16.301 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 22.962 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 259.743 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 0.888 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 89.591 ms | 0.00 | 0.00 | n/a | ok | 1.2x (14%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 8.312 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 89.421 ms | 1.20 | 0.03 | 0% | ok | 1.1x (14%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 2.587 ms | 1.55 | 6.21 | 8%\* | ok | 1.4x (18%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 17.863 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 30.150 ms | 3.55 | 0.10 | 1% | ok | 1.1x (13%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 2.505 ms | 1.60 | 6.41 | 8%\* | ok | 1.0x (13%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 7.880 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 289.682 ms | 0.74 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.704 ms | 0.46 | 1.84 | 2%\* | ok | 1.1x (13%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 12.836 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 268.758 ms | 0.80 | 0.01 | 0% | ok | 1.1x (13%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.746 ms | 0.46 | 1.84 | 2%\* | ok | 1.1x (13%) |
| direct | ref_cholmod | f64 | 192^2 grid, nnz 183552 | 24.949 ms | 4.29 | 0.12 | 1% | ok | 1.0x (13%) |
| direct | ref_cholmod.analyze | f64 | 192^2 grid, nnz 183552 | 6.668 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | ref_cholmod.factor | f64 | 192^2 grid, nnz 183552 | 15.826 ms | 6.76 | 0.19 | 1% | ok | 1.0x (12%) |
| direct | ref_cholmod.solve | f64 | 192^2 grid, nnz 183552 | 1.462 ms | 2.75 | 10.98 | 13%\* | ok | 1.2x (15%) |
| direct | ref_pardiso_spd | f64 | 192^2 grid, nnz 183552 | 11.708 ms | 9.14 | 0.25 | 2% | ok | 1.8x (22%) |
| direct | ref_pardiso_spd.analyze | f64 | 192^2 grid, nnz 183552 | 6.499 ms | 0.00 | 0.00 | n/a | ok | 1.3x (16%) |
| direct | ref_pardiso_spd.factor | f64 | 192^2 grid, nnz 183552 | 2.752 ms | 38.87 | 1.07 | 6% | ok | 3.4x (42%) |
| direct | ref_pardiso_spd.solve | f64 | 192^2 grid, nnz 183552 | 2.142 ms | 1.87 | 7.50 | 9%\* | ok | 1.0x (12%) |
| direct | ref_pardiso_lu | f64 | 192^2 grid, nnz 183552 | 32.696 ms | 6.54 | 0.09 | 1% | ok | 1.5x (18%) |
| direct | ref_pardiso_lu.analyze | f64 | 192^2 grid, nnz 183552 | 24.665 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct | ref_pardiso_lu.factor | f64 | 192^2 grid, nnz 183552 | 4.737 ms | 45.17 | 0.62 | 8% | ok | 3.8x (47%) |
| direct | ref_pardiso_lu.solve | f64 | 192^2 grid, nnz 183552 | 2.845 ms | 1.41 | 5.64 | 7%\* | ok | 1.6x (20%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 8.678 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 595.191 ms | 1.63 | 0.00 | 0% | ok | 1.7x (22%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 5.002 ms | 1.50 | 6.00 | 7% | ok | 1.3x (16%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 18.748 ms | 0.00 | 0.00 | n/a | ok | 1.2x (14%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 178.169 ms | 5.44 | 0.01 | 1% | ok | 1.1x (14%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 4.984 ms | 1.50 | 6.02 | 7% | ok | 1.5x (18%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 8.608 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 1878.950 ms | 1.03 | 0.00 | 0% | ok | 1.1x (14%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.628 ms | 0.38 | 1.53 | 2% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 12.714 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 1353.463 ms | 1.43 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.592 ms | 0.38 | 1.53 | 2% | ok | 1.0x (12%) |
| direct3d | ref_cholmod.analyze | f64 | 24^3 grid, nnz 93312 | 6.462 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | ref_cholmod.factor | f64 | 24^3 grid, nnz 93312 | 18.439 ms | 52.56 | 0.08 | 9% | ok | 1.6x (20%) |
| direct3d | ref_cholmod.solve | f64 | 24^3 grid, nnz 93312 | 1.166 ms | 6.43 | 25.73 | 31% | ok | 1.0x (12%) |
| direct3d | ref_pardiso_spd.analyze | f64 | 24^3 grid, nnz 93312 | 12.607 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| direct3d | ref_pardiso_spd.factor | f64 | 24^3 grid, nnz 93312 | 5.928 ms | 163.50 | 0.25 | 27% | ok | 5.0x (62%) |
| direct3d | ref_pardiso_spd.solve | f64 | 24^3 grid, nnz 93312 | 2.152 ms | 3.48 | 13.94 | 17% | ok | 1.0x (12%) |
| direct3d | ref_pardiso_lu.analyze | f64 | 24^3 grid, nnz 93312 | 14.366 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| direct3d | ref_pardiso_lu.factor | f64 | 24^3 grid, nnz 93312 | 12.148 ms | 159.56 | 0.12 | 27% | ok | 4.9x (61%) |
| direct3d | ref_pardiso_lu.solve | f64 | 24^3 grid, nnz 93312 | 3.055 ms | 2.45 | 9.82 | 12% | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 237.072 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 225.284 ms | 0.00 | 0.00 | n/a | ok | 1.3x (16%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 148.778 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 106.325 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 113.024 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 98.419 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 271.639 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 184.908 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 109.118 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 123.476 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 110.414 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 115.511 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 154.259 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.159 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 256^2 grid | 5.982 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 256^2 grid | 49.316 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 256^2 grid | 4.280 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 256^2 grid | 4.012 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 256^2 grid | 0.105 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 4.524 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 4.371 ms | 2.40 | 26.85 | 33% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 4.940 ms | 2.12 | 23.76 | 29% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 6.264 ms | 1.67 | 18.74 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 61.840 ms | 2.52 | 18.90 | 23% | ok | 1.1x (14%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 289.041 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 291.080 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 483.527 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 20.350 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=33554432 | 19.272 ms | 3.48 | 13.93 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 41.035 ms | 52.33 | 0.41 | 9% | ok | 7.6x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 2.624 ms | 25.57 | 27.17 | 33% | ok | 1.0x (13%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 9.360 ms | 7.17 | 28.68 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 4.618 ms | 14.53 | 29.07 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 6.852 ms | 9.79 | 9.79 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 6.859 ms | 9.78 | 9.78 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 19.454 ms | 110.39 | 0.65 | n/a | ok | 7.2x (90%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 21.854 ms | 98.27 | 0.38 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 10.738 ms | 199.99 | 0.59 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 10.741 ms | 199.93 | 0.59 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 28.128 ms | 76.35 | 0.22 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

