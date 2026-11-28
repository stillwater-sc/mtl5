# mtl5_bench platform characterization

Class `large`, variants native native-fast blas, threads 1,6, CPUs 0,1,2,3,4,5, clock policy governor performance on every CPU, turbo on; performance 1200-2700 MHz on cpus 0,1,2,3,4,5, 2026-11-28T06:11:10Z.

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

## xeon-e5-2420v2 -- native, 6 threads, class large

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 59.0 GFLOP/s, f32 118.0 GFLOP/s |
| ... hardware, SSE2 | f64 59.0, f32 118.0 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.7 GFLOP/s |
| bandwidth | read 11.4, update 10.7, triad 8.5 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 53% | 107% | 0 | 85% |
| blas2 | 10 | 78% | 108% | 0 | 16% |
| blas3 | 10 | 7% | 40% | 0 | 17% |
| factor | 12 | 6% | 40% | 0 | 53% |
| eigen | 8 | 2% | 12% | 0 | 20% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 68% | 97% | 0 | 17% |
| direct | 19 | 1% | 15% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 3% | 14% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 69% | 76% | 0 | 17% |
| mg | 1 | 59% | 59% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 17% | 64% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 76% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 131.339 ms | 0.51 | 4.09 | 36% | ok | 5.6x (93%) |
| blas1 | dot | f32 | n=33554432 | 21.979 ms | 3.05 | 12.21 | 107% | ok | 5.1x (85%) |
| blas1 | axpy | f64 | n=33554432 | 152.110 ms | 0.44 | 5.29 | 46% | ok | 5.6x (93%) |
| blas1 | axpy | f32 | n=33554432 | 67.037 ms | 1.00 | 6.01 | 53% | ok | 5.5x (91%) |
| blas1 | scale | f64 | n=33554432 | 53.903 ms | 0.62 | 9.96 | 87% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=33554432 | 132.236 ms | 0.51 | 2.03 | 18% | ok | 5.6x (93%) |
| blas1 | one_norm | f64 | n=33554432 | 39.016 ms | 0.86 | 6.88 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=33554432 | 52.308 ms | 0.64 | 5.13 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=33554432 | 39.110 ms | 0.86 | 6.86 | 60% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=33554432 | 1114.926 ms | 0.66 | 0.48 | 4% | ok | 1.0x (16%) |
| blas2 | gemv | f64 | n=8192 | 44.127 ms | 3.04 | 12.17 | 106% | ok | 1.3x (22%) |
| blas2 | gemv | f32 | n=8192 | 22.156 ms | 6.06 | 12.12 | 106% | ok | 1.5x (25%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 46.237 ms | 2.90 | 11.61 | 102% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 21.844 ms | 6.14 | 12.29 | 108% | ok | 1.2x (21%) |
| blas2 | symv | f64 | n=8192 | 59.962 ms | 2.24 | 4.48 | 39% | ok | 0.9x (15%) |
| blas2 | trmv | f64 | n=8192 | 31.004 ms | 2.16 | 8.66 | 76% | ok | 1.0x (16%) |
| blas2 | trmv | f32 | n=8192 | 21.192 ms | 3.17 | 6.34 | 55% | ok | 0.9x (14%) |
| blas2 | trsv | f64 | n=8192 | 30.244 ms | 2.22 | 8.88 | 78% | ok | 1.0x (16%) |
| blas2 | trsv | f32 | n=8192 | 19.347 ms | 3.47 | 6.94 | 61% | ok | 0.9x (15%) |
| blas2 | ger | f64 | n=8192 | 121.637 ms | 1.10 | 8.83 | 77% | ok | 1.0x (16%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 91.836 ms | 23.38 | 0.27 | 40% | ok | 5.6x (93%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 51.184 ms | 41.96 | 0.25 | 36% | ok | 5.7x (95%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 275.061 ms | 3.90 | 0.09 | 7% | ok | 1.0x (17%) |
| blas3 | symm | f64 | 1024x1024x1024 | 524.059 ms | 4.10 | 0.05 | 7% | ok | 1.0x (17%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 287.800 ms | 3.73 | 0.06 | 6% | ok | 1.0x (17%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 576.935 ms | 3.72 | 0.04 | 6% | ok | 1.0x (17%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 256.369 ms | 4.19 | 0.08 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 258.302 ms | 4.16 | 0.08 | 7% | ok | 1.0x (17%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 127.454 ms | 8.42 | 0.08 | 7% | ok | 1.0x (17%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 131.413 ms | 0.13 | 1.02 | 9% | ok | 1.0x (17%) |
| factor | lu | f64 | 1024x1024 | 70.374 ms | 10.17 | 0.24 | 17% | ok | 3.5x (58%) |
| factor | lu | f32 | 1024x1024 | 41.005 ms | 17.46 | 0.20 | 15% | ok | 3.2x (53%) |
| factor | cholesky | f64 | 1024x1024 | 94.932 ms | 3.77 | 0.18 | 6% | ok | 2.2x (36%) |
| factor | cholesky | f32 | 1024x1024 | 92.406 ms | 3.87 | 0.09 | 3% | ok | 2.2x (36%) |
| factor | ldlt | f64 | 1024x1024 | 213.927 ms | 1.67 | 0.08 | 3% | ok | 1.0x (16%) |
| factor | ldlt_bk | f64 | 1024x1024 | 105.566 ms | 3.39 | 0.16 | 6% | ok | 1.0x (17%) |
| factor | qr | f64 | 1024x1024 | 1625.886 ms | 0.88 | 0.01 | 1% | ok | 4.2x (70%) |
| factor | lq | f64 | 1024x1024 | 149.453 ms | 9.58 | 0.11 | 16% | ok | 3.9x (65%) |
| factor | hessenberg | f64 | 1024x1024 | 1868.339 ms | 1.92 | 0.01 | 3% | ok | 4.1x (69%) |
| factor | householder | f64 | 1024x1024 | 3.709 ms | 1.13 | 4.52 | 40% | ok | 5.5x (92%) |
| factor | inv | f64 | 1024x1024 | 999.110 ms | 2.15 | 0.02 | 4% | ok | 1.2x (20%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 55.810 ms | 12.83 | 0.30 | 22% | ok | 2.6x (44%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 415.554 ms | 0.43 | 0.01 | 1% | ok | 1.5x (25%) |
| eigen | eigen_symmetric | f64 | 512x512 | 1849.656 ms | 0.65 | 0.00 | 1% | ok | 1.1x (19%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 14618.110 ms | 0.08 | 0.00 | 0% | ok | 1.0x (16%) |
| eigen | eigenvalue | f64 | 512x512 | 867.405 ms | 1.55 | 0.00 | 3% | ok | 1.2x (21%) |
| eigen | schur | f64 | 512x512 | 2609.530 ms | 1.29 | 0.00 | 2% | ok | 1.1x (19%) |
| eigen | svd | f64 | 512x512 | 2691.159 ms | 1.10 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 512x512 | 279.563 ms | 7.20 | 0.02 | 12% | ok | 2.6x (44%) |
| eigen | sylvester | f64 | 512x512 | 5914.523 ms | 1.36 | 0.00 | 2% | ok | 1.2x (20%) |
| elementwise | exp | f64 | n=33554432 | 603.338 ms | 0.06 | 0.89 | 8% | ok | 1.0x (17%) |
| elementwise | log | f64 | n=33554432 | 562.534 ms | 0.06 | 0.95 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=33554432 | 682.965 ms | 0.05 | 0.79 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=33554432 | 362.306 ms | 0.09 | 1.48 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=33554432 | 9909.443 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 9.851 ms | 1.06 | 11.06 | 97% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 12.717 ms | 0.82 | 8.57 | 75% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 45.532 ms | 3.68 | 7.74 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 101.113 ms | 0.13 | 0.46 | 4% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 37.528 ms | 0.27 | 2.23 | 20% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 305.572 ms | 0.35 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 554.638 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 156.001 ms | 0.69 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 893.036 ms | 0.24 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 886.451 ms | 0.24 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 379.959 ms | 0.56 | 0.01 | 1% | ok | 1.0x (16%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 1122.399 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 43.879 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 63.125 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 581.725 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 2.393 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 253.894 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 26.485 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 271.118 ms | 0.39 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 9.098 ms | 0.44 | 1.76 | 15% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 54.559 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 92.644 ms | 1.15 | 0.03 | 2% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 9.150 ms | 0.44 | 1.75 | 15% | ok | 1.0x (17%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 27.446 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 828.009 ms | 0.26 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.306 ms | 0.13 | 0.53 | 5% | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 40.951 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 802.785 ms | 0.27 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.552 ms | 0.13 | 0.53 | 5% | ok | 1.0x (16%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 23.665 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1951.215 ms | 0.50 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 19.378 ms | 0.39 | 1.55 | 14% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 61.987 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 606.383 ms | 1.60 | 0.00 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 19.431 ms | 0.39 | 1.54 | 14% | ok | 1.0x (17%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 24.041 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 5763.888 ms | 0.34 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 87.394 ms | 0.09 | 0.34 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 33.856 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 4954.223 ms | 0.39 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 88.076 ms | 0.09 | 0.34 | 3% | ok | 1.0x (16%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 1745.140 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 2985.520 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 768.787 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 888.924 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 920.712 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 576.145 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 4875.976 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 1221.857 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 718.851 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 1390.252 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 1294.590 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 1494.922 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 2047.635 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.676 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilu_0 | f64 | 256^2 grid | 18.100 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 256^2 grid | 148.465 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 256^2 grid | 12.140 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | ildl | f64 | 256^2 grid | 11.411 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 256^2 grid | 0.370 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 8.982 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 14.932 ms | 0.70 | 7.86 | 69% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 13.512 ms | 0.78 | 8.69 | 76% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 16.023 ms | 0.65 | 7.33 | 64% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 173.540 ms | 0.90 | 6.73 | 59% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 563.830 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 576.668 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 1419.158 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 172.257 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=33554432 | 754.142 ms | 0.09 | 0.36 | 3% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 219.074 ms | 9.80 | 0.08 | 17% | ok | 4.1x (69%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 9.732 ms | 6.90 | 7.33 | 64% | ok | 1.0x (17%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 26.321 ms | 2.55 | 10.20 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.737 ms | 5.27 | 10.54 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 20.379 ms | 3.29 | 3.29 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 20.464 ms | 3.28 | 3.28 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 100.213 ms | 21.43 | 0.13 | n/a | ok | 4.6x (76%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 65.393 ms | 32.84 | 0.13 | n/a | ok | 5.5x (92%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 33.821 ms | 63.50 | 0.19 | n/a | ok | 5.6x (93%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 33.740 ms | 63.65 | 0.19 | n/a | ok | 5.6x (93%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 103.507 ms | 20.75 | 0.06 | n/a | ok | 5.5x (91%) |

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
| bandwidth | read 10.5, update 10.4, triad 8.2 GB/s (ceiling 10.5) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 98% | 108% | 0 |
| blas2 | 10 | 89% | 97% | 0 |
| blas3 | 10 | 82% | 95% | 0 |
| factor | 12 | 17% | 48% | 0 |
| eigen | 8 | 11% | 47% | 0 |
| elementwise | 5 | 8% | 14% | 0 |
| sparse | 5 | 55% | 100% | 0 |
| direct | 19 | 4% | 17% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 15% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 75% | 83% | 0 |
| mg | 1 | 62% | 62% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 72% | 90% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=33554432 | 47.934 ms | 1.40 | 11.20 | 107% | ok |
| blas1 | dot | f32 | n=33554432 | 23.950 ms | 2.80 | 11.21 | 107% | ok |
| blas1 | axpy | f64 | n=33554432 | 78.283 ms | 0.86 | 10.29 | 98% | ok |
| blas1 | axpy | f32 | n=33554432 | 38.593 ms | 1.74 | 10.43 | 100% | ok |
| blas1 | scale | f64 | n=33554432 | 56.530 ms | 0.59 | 9.50 | 91% | ok |
| blas1 | two_norm | f64 | n=33554432 | 23.649 ms | 2.84 | 11.35 | 108% | ok |
| blas1 | one_norm | f64 | n=33554432 | 39.366 ms | 0.85 | 6.82 | 65% | ok |
| blas1 | infinity_norm | f64 | n=33554432 | 52.704 ms | 0.64 | 5.09 | 49% | ok |
| blas1 | sum | f64 | n=33554432 | 39.079 ms | 0.86 | 6.87 | 66% | ok |
| blas1 | dot_k3 | f64 | n=33554432 | 1085.469 ms | 0.68 | 0.49 | 6% | ok |
| blas2 | gemv | f64 | n=8192 | 57.092 ms | 2.35 | 9.41 | 90% | ok |
| blas2 | gemv | f32 | n=8192 | 28.850 ms | 4.65 | 9.31 | 89% | ok |
| blas2 | gemv_colmajor | f64 | n=8192 | 52.857 ms | 2.54 | 10.16 | 97% | ok |
| blas2 | gemv_colmajor | f32 | n=8192 | 26.598 ms | 5.05 | 10.09 | 96% | ok |
| blas2 | symv | f64 | n=8192 | 54.634 ms | 2.46 | 4.92 | 47% | ok |
| blas2 | trmv | f64 | n=8192 | 29.212 ms | 2.30 | 9.19 | 88% | ok |
| blas2 | trmv | f32 | n=8192 | 14.615 ms | 4.59 | 9.19 | 88% | ok |
| blas2 | trsv | f64 | n=8192 | 29.312 ms | 2.29 | 9.16 | 87% | ok |
| blas2 | trsv | f32 | n=8192 | 14.391 ms | 4.66 | 9.33 | 89% | ok |
| blas2 | ger | f64 | n=8192 | 119.334 ms | 1.12 | 9.00 | 86% | ok |
| blas3 | gemm | f64 | 1024x1024x1024 | 224.993 ms | 9.54 | 0.11 | 91% | ok |
| blas3 | gemm | f32 | 1024x1024x1024 | 107.785 ms | 19.92 | 0.12 | 95% | ok |
| blas3 | gemmt | f64 | 1024x1024x1024 | 131.433 ms | 8.17 | 0.19 | 78% | ok |
| blas3 | symm | f64 | 1024x1024x1024 | 231.160 ms | 9.29 | 0.11 | 88% | ok |
| blas3 | syrk | f64 | 1024x1024x1024 | 122.881 ms | 8.74 | 0.14 | 83% | ok |
| blas3 | syr2k | f64 | 1024x1024x1024 | 247.870 ms | 8.66 | 0.10 | 82% | ok |
| blas3 | trmm | f64 | 1024x1024x1024 | 148.680 ms | 7.22 | 0.14 | 69% | ok |
| blas3 | trsm | f64 | 1024x1024x1024 | 151.748 ms | 7.08 | 0.14 | 67% | ok |
| blas3 | trsm | f32 | 1024x1024x1024 | 69.297 ms | 15.49 | 0.15 | 74% | ok |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 132.156 ms | 0.13 | 1.02 | 10% | ok |
| factor | lu | f64 | 1024x1024 | 232.048 ms | 3.08 | 0.07 | 29% | ok |
| factor | lu | f32 | 1024x1024 | 123.399 ms | 5.80 | 0.07 | 28% | ok |
| factor | cholesky | f64 | 1024x1024 | 206.755 ms | 1.73 | 0.08 | 16% | ok |
| factor | cholesky | f32 | 1024x1024 | 200.110 ms | 1.79 | 0.04 | 9% | ok |
| factor | ldlt | f64 | 1024x1024 | 213.468 ms | 1.68 | 0.08 | 16% | ok |
| factor | ldlt_bk | f64 | 1024x1024 | 107.846 ms | 3.32 | 0.16 | 32% | ok |
| factor | qr | f64 | 1024x1024 | 6918.841 ms | 0.21 | 0.00 | 2% | ok |
| factor | lq | f64 | 1024x1024 | 575.271 ms | 2.49 | 0.03 | 24% | ok |
| factor | hessenberg | f64 | 1024x1024 | 7787.259 ms | 0.46 | 0.00 | 4% | ok |
| factor | householder | f64 | 1024x1024 | 20.478 ms | 0.20 | 0.82 | 8% | ok |
| factor | inv | f64 | 1024x1024 | 1179.453 ms | 1.82 | 0.01 | 17% | ok |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 140.426 ms | 5.10 | 0.12 | 48% | ok |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 635.961 ms | 0.28 | 0.01 | 3% | ok |
| eigen | eigen_symmetric | f64 | 512x512 | 2114.544 ms | 0.57 | 0.00 | 5% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 14729.421 ms | 0.08 | 0.00 | 1% | ok |
| eigen | eigenvalue | f64 | 512x512 | 1077.245 ms | 1.25 | 0.00 | 12% | ok |
| eigen | schur | f64 | 512x512 | 2976.590 ms | 1.13 | 0.00 | 11% | ok |
| eigen | svd | f64 | 512x512 | 2335.613 ms | 1.26 | 0.00 | 12% | ok |
| eigen | expm | f64 | 512x512 | 409.950 ms | 4.91 | 0.01 | 47% | ok |
| eigen | sylvester | f64 | 512x512 | 6736.834 ms | 1.20 | 0.00 | 11% | ok |
| elementwise | exp | f64 | n=33554432 | 605.205 ms | 0.06 | 0.89 | 8% | ok |
| elementwise | log | f64 | n=33554432 | 561.804 ms | 0.06 | 0.96 | 9% | ok |
| elementwise | sin | f64 | n=33554432 | 688.823 ms | 0.05 | 0.78 | 7% | ok |
| elementwise | sqrt | f64 | n=33554432 | 361.719 ms | 0.09 | 1.48 | 14% | ok |
| elementwise | erf | f64 | n=33554432 | 9973.321 ms | 0.00 | 0.05 | 1% | ok |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 10.452 ms | 1.00 | 10.43 | 100% | ok |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 12.996 ms | 0.81 | 8.39 | 80% | ok |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 61.505 ms | 2.73 | 5.73 | 55% | ok |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 99.073 ms | 0.13 | 0.47 | 4% | ok |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 38.700 ms | 0.26 | 2.17 | 21% | ok |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 306.151 ms | 0.35 | 0.01 | 3% | ok |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 558.863 ms | 0.19 | 0.01 | 2% | ok |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 158.958 ms | 0.67 | 0.02 | 6% | ok |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 903.709 ms | 0.24 | 0.00 | 2% | ok |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 926.489 ms | 0.23 | 0.00 | 2% | ok |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 372.646 ms | 0.57 | 0.01 | 5% | ok |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 1090.627 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 44.510 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 65.617 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 597.738 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 2.532 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 251.798 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 27.445 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 270.540 ms | 0.40 | 0.01 | 4% | ok |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 9.093 ms | 0.44 | 1.77 | 17% | ok |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 56.179 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 93.276 ms | 1.15 | 0.03 | 11% | ok |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 9.090 ms | 0.44 | 1.77 | 17% | ok |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 27.434 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 833.960 ms | 0.26 | 0.00 | 2% | ok |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.277 ms | 0.13 | 0.53 | 5% | ok |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 41.675 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 850.439 ms | 0.25 | 0.00 | 2% | ok |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.249 ms | 0.13 | 0.53 | 5% | ok |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 24.215 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1952.941 ms | 0.50 | 0.00 | 5% | ok |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 19.327 ms | 0.39 | 1.55 | 15% | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 62.206 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 609.497 ms | 1.59 | 0.00 | 15% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 19.287 ms | 0.39 | 1.56 | 15% | ok |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 24.682 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 5856.982 ms | 0.33 | 0.00 | 3% | ok |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 88.052 ms | 0.09 | 0.34 | 3% | ok |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 34.749 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 6526.814 ms | 0.30 | 0.00 | 3% | ok |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 87.361 ms | 0.09 | 0.34 | 3% | ok |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 637.325 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 1006.862 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 365.863 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 291.238 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 306.373 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 283.365 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 655.415 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 513.758 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 309.686 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 331.967 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 288.955 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 304.799 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 371.483 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.513 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 256^2 grid | 19.061 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 256^2 grid | 149.929 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 256^2 grid | 12.001 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 256^2 grid | 11.513 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 256^2 grid | 0.365 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 8.536 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 14.887 ms | 0.70 | 7.88 | 75% | ok |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 13.471 ms | 0.78 | 8.71 | 83% | ok |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 16.037 ms | 0.65 | 7.32 | 70% | ok |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 179.728 ms | 0.87 | 6.50 | 62% | ok |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 559.322 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 592.827 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 1431.146 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 64.686 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=33554432 | 28.604 ms | 2.35 | 9.38 | 90% | ok |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 496.767 ms | 4.32 | 0.03 | 41% | ok |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 9.453 ms | 7.10 | 7.54 | 72% | ok |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 24.368 ms | 2.75 | 11.02 | n/a | ok |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.046 ms | 5.57 | 11.14 | n/a | ok |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 8.770 ms | 7.65 | 7.65 | n/a | ok |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 7.407 ms | 9.06 | 9.06 | n/a | ok |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 150.613 ms | 14.26 | 0.08 | n/a | ok |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 175.863 ms | 12.21 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 175.042 ms | 12.27 | 0.04 | n/a | ok |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 127.558 ms | 16.84 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 96.720 ms | 22.20 | 0.07 | n/a | ok |

## xeon-e5-2420v2 -- native-fast, 6 threads, class large

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD SSE4 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 59.0 GFLOP/s, f32 118.1 GFLOP/s |
| ... hardware, SSE2 | f64 59.0, f32 118.1 GFLOP/s |
| ... this build, SSE4 | f64 59.0, f32 117.9 GFLOP/s |
| bandwidth | read 11.4, update 10.7, triad 8.5 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 94% | 108% | 0 | 17% |
| blas2 | 10 | 84% | 108% | 0 | 17% |
| blas3 | 10 | 62% | 85% | 0 | 70% |
| factor | 12 | 6% | 39% | 0 | 51% |
| eigen | 8 | 2% | 15% | 0 | 19% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 68% | 97% | 0 | 17% |
| direct | 19 | 1% | 15% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 3% | 13% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 16% |
| smoother | 3 | 69% | 77% | 0 | 17% |
| mg | 1 | 60% | 60% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 66% | 82% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 86% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 43.563 ms | 1.54 | 12.32 | 108% | ok | 1.1x (18%) |
| blas1 | dot | f32 | n=33554432 | 21.767 ms | 3.08 | 12.33 | 108% | ok | 1.1x (18%) |
| blas1 | axpy | f64 | n=33554432 | 75.128 ms | 0.89 | 10.72 | 94% | ok | 1.0x (17%) |
| blas1 | axpy | f32 | n=33554432 | 36.902 ms | 1.82 | 10.91 | 96% | ok | 1.0x (17%) |
| blas1 | scale | f64 | n=33554432 | 53.820 ms | 0.62 | 9.98 | 87% | ok | 1.1x (18%) |
| blas1 | two_norm | f64 | n=33554432 | 21.694 ms | 3.09 | 12.37 | 108% | ok | 1.1x (18%) |
| blas1 | one_norm | f64 | n=33554432 | 39.070 ms | 0.86 | 6.87 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=33554432 | 52.426 ms | 0.64 | 5.12 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=33554432 | 39.072 ms | 0.86 | 6.87 | 60% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=33554432 | 1101.283 ms | 0.67 | 0.49 | 4% | ok | 1.0x (16%) |
| blas2 | gemv | f64 | n=8192 | 43.729 ms | 3.07 | 12.28 | 108% | ok | 1.3x (22%) |
| blas2 | gemv | f32 | n=8192 | 21.945 ms | 6.12 | 12.24 | 107% | ok | 1.3x (22%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 43.804 ms | 3.06 | 12.26 | 107% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 22.446 ms | 5.98 | 11.96 | 105% | ok | 1.2x (20%) |
| blas2 | symv | f64 | n=8192 | 55.322 ms | 2.43 | 4.85 | 43% | ok | 1.0x (16%) |
| blas2 | trmv | f64 | n=8192 | 28.634 ms | 2.34 | 9.38 | 82% | ok | 1.0x (17%) |
| blas2 | trmv | f32 | n=8192 | 14.042 ms | 4.78 | 9.56 | 84% | ok | 1.0x (17%) |
| blas2 | trsv | f64 | n=8192 | 28.737 ms | 2.34 | 9.35 | 82% | ok | 1.0x (17%) |
| blas2 | trsv | f32 | n=8192 | 14.097 ms | 4.76 | 9.53 | 83% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=8192 | 119.428 ms | 1.12 | 8.99 | 79% | ok | 1.0x (17%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 44.664 ms | 48.08 | 0.56 | 81% | ok | 5.0x (84%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 21.326 ms | 100.70 | 0.59 | 85% | ok | 5.1x (84%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 31.597 ms | 33.98 | 0.80 | 58% | ok | 4.2x (69%) |
| blas3 | symm | f64 | 1024x1024x1024 | 52.416 ms | 40.97 | 0.48 | 69% | ok | 4.4x (74%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 29.392 ms | 36.53 | 0.57 | 62% | ok | 4.2x (70%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 59.108 ms | 36.33 | 0.43 | 62% | ok | 4.2x (70%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 54.697 ms | 19.63 | 0.38 | 33% | ok | 2.7x (45%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 56.343 ms | 19.06 | 0.37 | 32% | ok | 2.7x (45%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 26.119 ms | 41.11 | 0.40 | 35% | ok | 2.7x (44%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 131.100 ms | 0.13 | 1.02 | 9% | ok | 1.0x (17%) |
| factor | lu | f64 | 1024x1024 | 68.508 ms | 10.45 | 0.24 | 18% | ok | 3.4x (56%) |
| factor | lu | f32 | 1024x1024 | 40.112 ms | 17.85 | 0.21 | 15% | ok | 3.1x (51%) |
| factor | cholesky | f64 | 1024x1024 | 102.076 ms | 3.51 | 0.16 | 6% | ok | 2.0x (34%) |
| factor | cholesky | f32 | 1024x1024 | 100.104 ms | 3.58 | 0.08 | 3% | ok | 2.0x (33%) |
| factor | ldlt | f64 | 1024x1024 | 213.559 ms | 1.68 | 0.08 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 1024x1024 | 100.652 ms | 3.56 | 0.17 | 6% | ok | 1.1x (18%) |
| factor | qr | f64 | 1024x1024 | 1633.923 ms | 0.88 | 0.01 | 1% | ok | 4.2x (71%) |
| factor | lq | f64 | 1024x1024 | 148.231 ms | 9.66 | 0.11 | 16% | ok | 3.9x (65%) |
| factor | hessenberg | f64 | 1024x1024 | 1897.444 ms | 1.89 | 0.01 | 3% | ok | 4.1x (68%) |
| factor | householder | f64 | 1024x1024 | 3.768 ms | 1.11 | 4.45 | 39% | ok | 5.4x (91%) |
| factor | inv | f64 | 1024x1024 | 1094.530 ms | 1.96 | 0.02 | 3% | ok | 1.1x (18%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 56.051 ms | 12.77 | 0.30 | 22% | ok | 2.5x (42%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 413.229 ms | 0.43 | 0.01 | 1% | ok | 1.5x (26%) |
| eigen | eigen_symmetric | f64 | 512x512 | 1853.981 ms | 0.65 | 0.00 | 1% | ok | 1.1x (19%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 14428.361 ms | 0.08 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 512x512 | 863.384 ms | 1.55 | 0.00 | 3% | ok | 1.2x (21%) |
| eigen | schur | f64 | 512x512 | 2570.615 ms | 1.31 | 0.00 | 2% | ok | 1.2x (19%) |
| eigen | svd | f64 | 512x512 | 2313.357 ms | 1.28 | 0.00 | 2% | ok | 1.0x (17%) |
| eigen | expm | f64 | 512x512 | 226.471 ms | 8.89 | 0.02 | 15% | ok | 1.8x (30%) |
| eigen | sylvester | f64 | 512x512 | 5828.320 ms | 1.38 | 0.00 | 2% | ok | 1.2x (19%) |
| elementwise | exp | f64 | n=33554432 | 618.150 ms | 0.05 | 0.87 | 8% | ok | 1.0x (16%) |
| elementwise | log | f64 | n=33554432 | 559.886 ms | 0.06 | 0.96 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=33554432 | 685.153 ms | 0.05 | 0.78 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=33554432 | 360.958 ms | 0.09 | 1.49 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=33554432 | 9896.111 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 9.853 ms | 1.06 | 11.06 | 97% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 12.740 ms | 0.82 | 8.55 | 75% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 45.663 ms | 3.67 | 7.71 | 68% | ok | 1.3x (22%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 96.058 ms | 0.14 | 0.48 | 4% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 37.595 ms | 0.27 | 2.23 | 20% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 312.440 ms | 0.34 | 0.01 | 1% | ok | 1.0x (16%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 559.291 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 158.249 ms | 0.68 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 893.080 ms | 0.24 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 923.644 ms | 0.23 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 375.998 ms | 0.57 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 1101.398 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 44.648 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 65.344 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 595.150 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 2.521 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 250.465 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 26.850 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 269.403 ms | 0.40 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 9.180 ms | 0.44 | 1.75 | 15% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 55.277 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 93.107 ms | 1.15 | 0.03 | 2% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 9.265 ms | 0.43 | 1.73 | 15% | ok | 1.0x (16%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 28.272 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 840.816 ms | 0.25 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 31.024 ms | 0.13 | 0.52 | 5% | ok | 1.0x (16%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 42.439 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 836.860 ms | 0.26 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.444 ms | 0.13 | 0.53 | 5% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 24.053 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1917.779 ms | 0.51 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 19.523 ms | 0.38 | 1.54 | 13% | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 67.593 ms | 0.00 | 0.00 | n/a | ok | 0.9x (15%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 573.167 ms | 1.69 | 0.00 | 3% | ok | 1.1x (18%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 19.531 ms | 0.38 | 1.54 | 13% | ok | 1.0x (16%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 24.573 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 5791.476 ms | 0.33 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 87.781 ms | 0.09 | 0.34 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 34.565 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 6583.403 ms | 0.29 | 0.00 | 0% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 87.996 ms | 0.09 | 0.34 | 3% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 612.950 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 828.839 ms | 0.00 | 0.00 | n/a | ok | 1.2x (20%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 385.037 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 280.505 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 297.328 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 273.859 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 654.813 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 527.953 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 306.818 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 318.484 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 285.844 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 310.849 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 381.223 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.521 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | ilu_0 | f64 | 256^2 grid | 19.316 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | ilut | f64 | 256^2 grid | 148.588 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 256^2 grid | 11.986 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 256^2 grid | 11.409 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 256^2 grid | 0.374 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 9.021 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 14.916 ms | 0.70 | 7.87 | 69% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 13.443 ms | 0.78 | 8.73 | 77% | ok | 1.0x (17%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 15.939 ms | 0.66 | 7.36 | 65% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 171.535 ms | 0.91 | 6.81 | 60% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 553.879 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 573.111 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 1422.123 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 64.749 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=33554432 | 28.557 ms | 2.35 | 9.40 | 82% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 93.895 ms | 22.87 | 0.18 | 39% | ok | 5.3x (88%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 9.483 ms | 7.08 | 7.52 | 66% | ok | 1.0x (17%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 25.252 ms | 2.66 | 10.63 | n/a | ok | 1.0x (16%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.166 ms | 5.52 | 11.03 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 8.811 ms | 7.62 | 7.62 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 7.363 ms | 9.11 | 9.11 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 29.233 ms | 73.46 | 0.43 | n/a | ok | 5.2x (86%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 33.287 ms | 64.51 | 0.25 | n/a | ok | 5.3x (88%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 33.139 ms | 64.80 | 0.19 | n/a | ok | 5.3x (88%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 24.161 ms | 88.88 | 0.26 | n/a | ok | 5.3x (88%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 18.547 ms | 115.79 | 0.34 | n/a | ok | 5.2x (87%) |

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

## xeon-e5-2420v2 -- blas, 6 threads, class large

| | |
|---|---|
| CPU | Intel(R) Xeon(R) CPU E5-2420 v2 @ 2.20GHz (x86_64, 12 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 59.1 GFLOP/s, f32 118.1 GFLOP/s |
| ... hardware, SSE2 | f64 59.1, f32 118.1 GFLOP/s |
| ... this build, scalar | f64 0.5, f32 3.7 GFLOP/s |
| bandwidth | read 11.4, update 10.6, triad 8.4 GB/s (ceiling 11.4) |
| last-level cache | 15 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 53% | 107% | 0 | 85% |
| blas2 | 10 | 104% | 107% | 0 | 21% |
| blas3 | 10 | 125% | 163% | 0 | 71% |
| factor | 12 | 20% | 40% | 0 | 28% |
| eigen | 8 | 13% | 35% | 0 | 19% |
| elementwise | 5 | 8% | 13% | 0 | 17% |
| sparse | 5 | 68% | 97% | 0 | 17% |
| direct | 19 | 1% | 15% | 0 | 17% |
| ordering | 5 | n/a | n/a | 0 | 17% |
| direct3d | 12 | 3% | 14% | 0 | 17% |
| krylov | 13 | n/a | n/a | 0 | 17% |
| pc | 7 | n/a | n/a | 0 | 17% |
| smoother | 3 | 69% | 75% | 0 | 17% |
| mg | 1 | 59% | 59% | 0 | 17% |
| sparse_eig | 4 | n/a | n/a | 0 | 17% |
| mixed | 3 | 22% | 65% | 0 | 17% |
| integer | 9 | n/a | n/a | 0 | 87% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 133.360 ms | 0.50 | 4.03 | 35% | ok | 5.6x (93%) |
| blas1 | dot | f32 | n=33554432 | 21.958 ms | 3.06 | 12.22 | 107% | ok | 5.1x (85%) |
| blas1 | axpy | f64 | n=33554432 | 155.196 ms | 0.43 | 5.19 | 46% | ok | 5.5x (91%) |
| blas1 | axpy | f32 | n=33554432 | 67.003 ms | 1.00 | 6.01 | 53% | ok | 5.5x (92%) |
| blas1 | scale | f64 | n=33554432 | 53.853 ms | 0.62 | 9.97 | 88% | ok | 1.0x (17%) |
| blas1 | two_norm | f64 | n=33554432 | 131.800 ms | 0.51 | 2.04 | 18% | ok | 5.6x (94%) |
| blas1 | one_norm | f64 | n=33554432 | 39.463 ms | 0.85 | 6.80 | 60% | ok | 1.0x (17%) |
| blas1 | infinity_norm | f64 | n=33554432 | 52.398 ms | 0.64 | 5.12 | 45% | ok | 1.0x (17%) |
| blas1 | sum | f64 | n=33554432 | 39.043 ms | 0.86 | 6.88 | 60% | ok | 1.0x (17%) |
| blas1 | dot_k3 | f64 | n=33554432 | 1118.429 ms | 0.66 | 0.48 | 4% | ok | 1.0x (17%) |
| blas2 | gemv | f64 | n=8192 | 45.556 ms | 2.95 | 11.79 | 104% | ok | 1.3x (21%) |
| blas2 | gemv | f32 | n=8192 | 23.343 ms | 5.75 | 11.50 | 101% | ok | 1.2x (20%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 43.987 ms | 3.05 | 12.21 | 107% | ok | 1.3x (21%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 22.282 ms | 6.02 | 12.05 | 106% | ok | 1.3x (21%) |
| blas2 | symv | f64 | n=8192 | 22.566 ms | 5.95 | 11.90 | 105% | ok | 1.3x (22%) |
| blas2 | trmv | f64 | n=8192 | 22.609 ms | 2.97 | 11.88 | 104% | ok | 1.3x (22%) |
| blas2 | trmv | f32 | n=8192 | 11.540 ms | 5.82 | 11.64 | 102% | ok | 1.2x (20%) |
| blas2 | trsv | f64 | n=8192 | 29.890 ms | 2.25 | 8.99 | 79% | ok | 1.1x (18%) |
| blas2 | trsv | f32 | n=8192 | 13.544 ms | 4.96 | 9.91 | 87% | ok | 1.0x (17%) |
| blas2 | ger | f64 | n=8192 | 111.064 ms | 1.21 | 9.67 | 85% | ok | 1.1x (18%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 22.388 ms | 95.92 | 1.12 | 162% | ok | 4.9x (82%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 13.119 ms | 163.69 | 0.96 | 139% | ok | 5.2x (86%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 17.791 ms | 60.35 | 1.41 | 102% | ok | 3.7x (61%) |
| blas3 | symm | f64 | 1024x1024x1024 | 22.324 ms | 96.19 | 1.13 | 163% | ok | 5.0x (83%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 17.765 ms | 60.44 | 0.94 | 102% | ok | 3.4x (57%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 32.448 ms | 66.18 | 0.78 | 112% | ok | 3.6x (60%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 13.905 ms | 77.22 | 1.51 | 131% | ok | 4.1x (69%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 14.568 ms | 73.71 | 1.44 | 125% | ok | 4.2x (71%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 8.502 ms | 126.29 | 1.23 | 107% | ok | 4.6x (77%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 128.915 ms | 0.13 | 1.04 | 9% | ok | 1.0x (17%) |
| factor | lu | f64 | 1024x1024 | 33.816 ms | 21.17 | 0.50 | 36% | ok | 1.9x (32%) |
| factor | lu | f32 | 1024x1024 | 29.917 ms | 23.93 | 0.28 | 20% | ok | 1.7x (28%) |
| factor | cholesky | f64 | 1024x1024 | 29.263 ms | 12.23 | 0.57 | 21% | ok | 1.5x (25%) |
| factor | cholesky | f32 | 1024x1024 | 26.471 ms | 13.52 | 0.32 | 11% | ok | 1.4x (23%) |
| factor | ldlt | f64 | 1024x1024 | 211.083 ms | 1.70 | 0.08 | 3% | ok | 1.0x (17%) |
| factor | ldlt_bk | f64 | 1024x1024 | 106.009 ms | 3.38 | 0.16 | 6% | ok | 1.0x (17%) |
| factor | qr | f64 | 1024x1024 | 75.339 ms | 19.00 | 0.22 | 32% | ok | 1.8x (30%) |
| factor | lq | f64 | 1024x1024 | 148.688 ms | 9.63 | 0.11 | 16% | ok | 3.9x (65%) |
| factor | hessenberg | f64 | 1024x1024 | 1857.121 ms | 1.93 | 0.01 | 3% | ok | 4.2x (70%) |
| factor | householder | f64 | 1024x1024 | 3.709 ms | 1.13 | 4.52 | 40% | ok | 5.5x (92%) |
| factor | inv | f64 | 1024x1024 | 955.432 ms | 2.25 | 0.02 | 4% | ok | 1.0x (17%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 46.744 ms | 15.31 | 0.36 | 26% | ok | 1.6x (27%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 23.366 ms | 7.66 | 0.18 | 13% | ok | 1.5x (25%) |
| eigen | eigen_symmetric | f64 | 512x512 | 1849.128 ms | 0.65 | 0.00 | 1% | ok | 1.1x (19%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 14747.834 ms | 0.08 | 0.00 | 0% | ok | 1.0x (17%) |
| eigen | eigenvalue | f64 | 512x512 | 378.801 ms | 3.54 | 0.01 | 6% | ok | 1.1x (18%) |
| eigen | schur | f64 | 512x512 | 434.654 ms | 7.72 | 0.01 | 13% | ok | 1.1x (19%) |
| eigen | svd | f64 | 512x512 | 142.362 ms | 20.74 | 0.03 | 35% | ok | 1.2x (21%) |
| eigen | expm | f64 | 512x512 | 192.362 ms | 10.47 | 0.02 | 18% | ok | 1.4x (24%) |
| eigen | sylvester | f64 | 512x512 | 1689.886 ms | 4.77 | 0.00 | 8% | ok | 1.1x (18%) |
| elementwise | exp | f64 | n=33554432 | 621.747 ms | 0.05 | 0.86 | 8% | ok | 1.0x (16%) |
| elementwise | log | f64 | n=33554432 | 566.094 ms | 0.06 | 0.95 | 8% | ok | 1.0x (17%) |
| elementwise | sin | f64 | n=33554432 | 680.574 ms | 0.05 | 0.79 | 7% | ok | 1.0x (17%) |
| elementwise | sqrt | f64 | n=33554432 | 362.952 ms | 0.09 | 1.48 | 13% | ok | 1.0x (17%) |
| elementwise | erf | f64 | n=33554432 | 9885.219 ms | 0.00 | 0.05 | 0% | ok | 1.0x (17%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 9.855 ms | 1.06 | 11.06 | 97% | ok | 1.1x (18%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 12.759 ms | 0.82 | 8.54 | 75% | ok | 1.0x (17%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 45.521 ms | 3.68 | 7.74 | 68% | ok | 1.4x (24%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 99.759 ms | 0.13 | 0.46 | 4% | ok | 1.0x (17%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 38.466 ms | 0.26 | 2.18 | 19% | ok | 1.0x (17%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 305.098 ms | 0.35 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 554.110 ms | 0.19 | 0.01 | 0% | ok | 1.0x (17%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 156.410 ms | 0.68 | 0.02 | 1% | ok | 1.0x (17%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 905.821 ms | 0.24 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 885.243 ms | 0.24 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 384.213 ms | 0.56 | 0.01 | 1% | ok | 0.9x (15%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 1114.042 ms | 0.02 | 0.00 | 0% | ok | 1.0x (17%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 43.750 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 63.825 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 580.818 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 2.392 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 250.610 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 26.473 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 268.735 ms | 0.40 | 0.01 | 1% | ok | 1.0x (17%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 9.134 ms | 0.44 | 1.76 | 15% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 54.293 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 93.183 ms | 1.15 | 0.03 | 2% | ok | 1.0x (17%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 9.129 ms | 0.44 | 1.76 | 15% | ok | 1.0x (16%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 26.949 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 846.220 ms | 0.25 | 0.00 | 0% | ok | 1.0x (16%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.101 ms | 0.13 | 0.53 | 5% | ok | 1.0x (17%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 40.852 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 800.958 ms | 0.27 | 0.00 | 0% | ok | 1.0x (17%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 30.204 ms | 0.13 | 0.53 | 5% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 23.596 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 1951.766 ms | 0.50 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 19.426 ms | 0.39 | 1.54 | 14% | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 61.600 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 607.434 ms | 1.60 | 0.00 | 3% | ok | 1.0x (16%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 19.414 ms | 0.39 | 1.54 | 14% | ok | 1.0x (17%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 24.037 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 5722.368 ms | 0.34 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 87.688 ms | 0.09 | 0.34 | 3% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 33.959 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 4956.886 ms | 0.39 | 0.00 | 1% | ok | 1.0x (17%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 88.267 ms | 0.08 | 0.34 | 3% | ok | 1.0x (17%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 1747.658 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 3036.382 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 767.822 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 891.822 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 928.260 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 580.248 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 4924.591 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 1246.145 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 722.871 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 1392.768 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 1304.171 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 1503.215 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 2064.220 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.725 ms | 0.00 | 0.00 | n/a | ok | 0.9x (16%) |
| pc | ilu_0 | f64 | 256^2 grid | 17.900 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ilut | f64 | 256^2 grid | 148.838 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ic_0 | f64 | 256^2 grid | 12.029 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ildl | f64 | 256^2 grid | 11.232 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | ssor | f64 | 256^2 grid | 0.364 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 14.078 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 14.985 ms | 0.70 | 7.83 | 69% | ok | 1.0x (17%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 13.673 ms | 0.77 | 8.58 | 75% | ok | 1.0x (16%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 15.994 ms | 0.66 | 7.34 | 65% | ok | 1.0x (17%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 173.970 ms | 0.89 | 6.72 | 59% | ok | 1.0x (17%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 565.114 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 581.296 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 1437.439 ms | 0.00 | 0.00 | n/a | ok | 1.0x (16%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 173.514 ms | 0.00 | 0.00 | n/a | ok | 1.0x (17%) |
| mixed | dot | f32>f64 | n=33554432 | 752.365 ms | 0.09 | 0.36 | 3% | ok | 1.0x (17%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 162.603 ms | 13.21 | 0.10 | 22% | ok | 5.6x (93%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 9.579 ms | 7.01 | 7.44 | 65% | ok | 1.0x (17%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 26.179 ms | 2.56 | 10.25 | n/a | ok | 1.0x (17%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 12.732 ms | 5.27 | 10.54 | n/a | ok | 1.0x (17%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 20.407 ms | 3.29 | 3.29 | n/a | ok | 1.0x (17%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 20.404 ms | 3.29 | 3.29 | n/a | ok | 1.0x (17%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 87.170 ms | 24.64 | 0.14 | n/a | ok | 5.2x (87%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 65.577 ms | 32.75 | 0.13 | n/a | ok | 5.5x (92%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 33.890 ms | 63.37 | 0.19 | n/a | ok | 5.6x (93%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 33.873 ms | 63.40 | 0.19 | n/a | ok | 5.6x (93%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 103.672 ms | 20.71 | 0.06 | n/a | ok | 5.5x (91%) |

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- libmkl_rt not found: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

