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

