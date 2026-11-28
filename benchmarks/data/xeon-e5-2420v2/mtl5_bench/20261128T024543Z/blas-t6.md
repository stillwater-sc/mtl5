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

