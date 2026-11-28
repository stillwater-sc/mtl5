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
| bandwidth | read 10.2, update 10.2, triad 8.1 GB/s (ceiling 10.2) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 23% | 93% | 0 |
| blas2 | 10 | 88% | 97% | 0 |
| blas3 | 10 | 38% | 40% | 0 |
| factor | 12 | 17% | 47% | 0 |
| eigen | 8 | 11% | 26% | 0 |
| elementwise | 5 | 9% | 15% | 0 |
| sparse | 5 | 56% | 102% | 0 |
| direct | 19 | 4% | 17% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 15% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 77% | 85% | 0 |
| mg | 1 | 64% | 64% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 23% | 72% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=33554432 | 733.142 ms | 0.09 | 0.73 | 7% | ok |
| blas1 | dot | f32 | n=33554432 | 111.621 ms | 0.60 | 2.40 | 23% | ok |
| blas1 | axpy | f64 | n=33554432 | 847.084 ms | 0.08 | 0.95 | 9% | ok |
| blas1 | axpy | f32 | n=33554432 | 366.854 ms | 0.18 | 1.10 | 11% | ok |
| blas1 | scale | f64 | n=33554432 | 56.335 ms | 0.60 | 9.53 | 93% | ok |
| blas1 | two_norm | f64 | n=33554432 | 738.997 ms | 0.09 | 0.36 | 4% | ok |
| blas1 | one_norm | f64 | n=33554432 | 39.152 ms | 0.86 | 6.86 | 67% | ok |
| blas1 | infinity_norm | f64 | n=33554432 | 52.494 ms | 0.64 | 5.11 | 50% | ok |
| blas1 | sum | f64 | n=33554432 | 39.165 ms | 0.86 | 6.85 | 67% | ok |
| blas1 | dot_k3 | f64 | n=33554432 | 1103.757 ms | 0.67 | 0.49 | 6% | ok |
| blas2 | gemv | f64 | n=8192 | 57.079 ms | 2.35 | 9.41 | 92% | ok |
| blas2 | gemv | f32 | n=8192 | 33.851 ms | 3.96 | 7.93 | 77% | ok |
| blas2 | gemv_colmajor | f64 | n=8192 | 54.303 ms | 2.47 | 9.89 | 97% | ok |
| blas2 | gemv_colmajor | f32 | n=8192 | 27.273 ms | 4.92 | 9.84 | 96% | ok |
| blas2 | symv | f64 | n=8192 | 53.696 ms | 2.50 | 5.00 | 49% | ok |
| blas2 | trmv | f64 | n=8192 | 29.819 ms | 2.25 | 9.01 | 88% | ok |
| blas2 | trmv | f32 | n=8192 | 18.198 ms | 3.69 | 7.38 | 72% | ok |
| blas2 | trsv | f64 | n=8192 | 29.767 ms | 2.25 | 9.02 | 88% | ok |
| blas2 | trsv | f32 | n=8192 | 17.879 ms | 3.75 | 7.51 | 73% | ok |
| blas2 | ger | f64 | n=8192 | 118.346 ms | 1.13 | 9.07 | 89% | ok |
| blas3 | gemm | f64 | 1024x1024x1024 | 513.362 ms | 4.18 | 0.05 | 40% | ok |
| blas3 | gemm | f32 | 1024x1024x1024 | 291.198 ms | 7.37 | 0.04 | 35% | ok |
| blas3 | gemmt | f64 | 1024x1024x1024 | 274.635 ms | 3.91 | 0.09 | 37% | ok |
| blas3 | symm | f64 | 1024x1024x1024 | 536.659 ms | 4.00 | 0.05 | 38% | ok |
| blas3 | syrk | f64 | 1024x1024x1024 | 288.687 ms | 3.72 | 0.06 | 35% | ok |
| blas3 | syr2k | f64 | 1024x1024x1024 | 600.313 ms | 3.58 | 0.04 | 34% | ok |
| blas3 | trmm | f64 | 1024x1024x1024 | 261.194 ms | 4.11 | 0.08 | 39% | ok |
| blas3 | trsm | f64 | 1024x1024x1024 | 261.068 ms | 4.11 | 0.08 | 39% | ok |
| blas3 | trsm | f32 | 1024x1024x1024 | 128.641 ms | 8.35 | 0.08 | 40% | ok |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 131.444 ms | 0.13 | 1.02 | 10% | ok |
| factor | lu | f64 | 1024x1024 | 243.456 ms | 2.94 | 0.07 | 28% | ok |
| factor | lu | f32 | 1024x1024 | 129.888 ms | 5.51 | 0.06 | 26% | ok |
| factor | cholesky | f64 | 1024x1024 | 204.963 ms | 1.75 | 0.08 | 17% | ok |
| factor | cholesky | f32 | 1024x1024 | 200.259 ms | 1.79 | 0.04 | 9% | ok |
| factor | ldlt | f64 | 1024x1024 | 210.476 ms | 1.70 | 0.08 | 16% | ok |
| factor | ldlt_bk | f64 | 1024x1024 | 110.168 ms | 3.25 | 0.15 | 31% | ok |
| factor | qr | f64 | 1024x1024 | 6848.212 ms | 0.21 | 0.00 | 2% | ok |
| factor | lq | f64 | 1024x1024 | 580.419 ms | 2.47 | 0.03 | 24% | ok |
| factor | hessenberg | f64 | 1024x1024 | 7751.286 ms | 0.46 | 0.00 | 4% | ok |
| factor | householder | f64 | 1024x1024 | 20.368 ms | 0.21 | 0.82 | 8% | ok |
| factor | inv | f64 | 1024x1024 | 1174.318 ms | 1.83 | 0.01 | 17% | ok |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 145.767 ms | 4.91 | 0.12 | 47% | ok |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 626.606 ms | 0.29 | 0.01 | 3% | ok |
| eigen | eigen_symmetric | f64 | 512x512 | 2099.999 ms | 0.58 | 0.00 | 5% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 14039.804 ms | 0.09 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 512x512 | 1083.218 ms | 1.24 | 0.00 | 12% | ok |
| eigen | schur | f64 | 512x512 | 2987.230 ms | 1.12 | 0.00 | 11% | ok |
| eigen | svd | f64 | 512x512 | 2675.730 ms | 1.10 | 0.00 | 11% | ok |
| eigen | expm | f64 | 512x512 | 735.929 ms | 2.74 | 0.01 | 26% | ok |
| eigen | sylvester | f64 | 512x512 | 6930.849 ms | 1.16 | 0.00 | 11% | ok |
| elementwise | exp | f64 | n=33554432 | 605.050 ms | 0.06 | 0.89 | 9% | ok |
| elementwise | log | f64 | n=33554432 | 559.611 ms | 0.06 | 0.96 | 9% | ok |
| elementwise | sin | f64 | n=33554432 | 704.091 ms | 0.05 | 0.76 | 7% | ok |
| elementwise | sqrt | f64 | n=33554432 | 361.511 ms | 0.09 | 1.49 | 15% | ok |
| elementwise | erf | f64 | n=33554432 | 9896.180 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 10.401 ms | 1.01 | 10.48 | 102% | ok |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 12.765 ms | 0.82 | 8.54 | 83% | ok |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 61.046 ms | 2.75 | 5.77 | 56% | ok |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 101.141 ms | 0.13 | 0.46 | 4% | ok |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 38.529 ms | 0.26 | 2.18 | 21% | ok |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 303.571 ms | 0.35 | 0.01 | 3% | ok |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 558.456 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 155.778 ms | 0.69 | 0.02 | 7% | ok |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 896.792 ms | 0.24 | 0.00 | 2% | ok |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 882.290 ms | 0.24 | 0.00 | 2% | ok |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 361.080 ms | 0.59 | 0.01 | 6% | ok |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 1120.501 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 43.631 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 63.654 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 581.220 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 2.399 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 253.961 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 26.653 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 268.558 ms | 0.40 | 0.01 | 4% | ok |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 9.124 ms | 0.44 | 1.76 | 17% | ok |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 54.572 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 92.268 ms | 1.16 | 0.03 | 11% | ok |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 9.078 ms | 0.44 | 1.77 | 17% | ok |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 26.834 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 839.127 ms | 0.25 | 0.00 | 2% | ok |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.274 ms | 0.13 | 0.53 | 5% | ok |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 41.041 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 795.739 ms | 0.27 | 0.00 | 3% | ok |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.201 ms | 0.13 | 0.53 | 5% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 23.718 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1944.875 ms | 0.50 | 0.00 | 5% | ok |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 19.334 ms | 0.39 | 1.55 | 15% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 61.362 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 604.691 ms | 1.60 | 0.00 | 15% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 20.083 ms | 0.37 | 1.49 | 15% | ok |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 23.994 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 5813.650 ms | 0.33 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 88.054 ms | 0.09 | 0.34 | 3% | ok |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 33.995 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 4943.681 ms | 0.39 | 0.00 | 4% | ok |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 87.106 ms | 0.09 | 0.34 | 3% | ok |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 1759.342 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 3063.144 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 762.399 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 908.563 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 929.020 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 589.260 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 4848.106 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 1206.121 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 718.967 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 1394.143 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 1290.658 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 1489.256 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 2032.987 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.684 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 256^2 grid | 18.034 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 256^2 grid | 148.445 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 256^2 grid | 11.955 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 256^2 grid | 11.456 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 256^2 grid | 0.364 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 8.873 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 14.799 ms | 0.71 | 7.93 | 77% | ok |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 13.431 ms | 0.78 | 8.74 | 85% | ok |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 15.996 ms | 0.65 | 7.34 | 72% | ok |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 179.251 ms | 0.87 | 6.52 | 64% | ok |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 563.776 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 581.284 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 1437.874 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 171.976 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=33554432 | 752.811 ms | 0.09 | 0.36 | 3% | ok |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 905.074 ms | 2.37 | 0.02 | 23% | ok |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 9.668 ms | 6.94 | 7.38 | 72% | ok |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 26.175 ms | 2.56 | 10.26 | n/a | ok |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 13.014 ms | 5.16 | 10.31 | n/a | ok |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 20.395 ms | 3.29 | 3.29 | n/a | ok |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 20.402 ms | 3.29 | 3.29 | n/a | ok |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 456.758 ms | 4.70 | 0.03 | n/a | ok |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 360.437 ms | 5.96 | 0.02 | n/a | ok |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 189.011 ms | 11.36 | 0.03 | n/a | ok |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 188.661 ms | 11.38 | 0.03 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 564.194 ms | 3.81 | 0.01 | n/a | ok |

