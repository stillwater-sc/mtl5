## i7-12700k -- native, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 600.7 GFLOP/s, f32 1201.4 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.7, f32 1201.4 GFLOP/s |
| ... this build, scalar | f64 9.9, f32 10.0 GFLOP/s |
| bandwidth | read 83.7, update 78.5, triad 76.8 GB/s (ceiling 83.7) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 89% | 123% | 0 | 30% |
| blas2 | 10 | 42% | 137% | 0 | 13% |
| blas3 | 10 | 2% | 17% | 0 | 14% |
| factor | 12 | 2% | 12% | 0 | 15% |
| eigen | 8 | 1% | 4% | 0 | 13% |
| elementwise | 5 | 4% | 5% | 0 | 13% |
| sparse | 5 | 35% | 287% | 0 | 13% |
| direct | 19 | 0% | 14% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 10% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 36% | 0 | 13% |
| mg | 1 | 26% | 26% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 40% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 92% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.395 ms | 12.02 | 96.20 | 115% | ok | 4.0x (50%) |
| blas1 | dot | f32 | n=8388608 | 0.651 ms | 25.76 | 103.05 | 123% | ok | 6.5x (81%) |
| blas1 | axpy | f64 | n=8388608 | 2.746 ms | 6.11 | 73.31 | 88% | ok | 2.4x (30%) |
| blas1 | axpy | f32 | n=8388608 | 1.350 ms | 12.43 | 74.59 | 89% | ok | 3.5x (44%) |
| blas1 | scale | f64 | n=8388608 | 1.775 ms | 4.73 | 75.60 | 90% | ok | 2.0x (25%) |
| blas1 | two_norm | f64 | n=8388608 | 0.740 ms | 22.66 | 90.63 | 108% | ok | 6.7x (83%) |
| blas1 | one_norm | f64 | n=8388608 | 3.954 ms | 2.12 | 16.97 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.961 ms | 1.21 | 9.64 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.960 ms | 2.12 | 16.95 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 75.818 ms | 2.43 | 1.77 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.443 ms | 23.25 | 93.03 | 111% | ok | 2.5x (32%) |
| blas2 | gemv | f32 | n=4096 | 0.585 ms | 57.35 | 114.76 | 137% | ok | 4.7x (58%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.556 ms | 21.57 | 86.31 | 103% | ok | 3.1x (38%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.897 ms | 37.41 | 74.85 | 89% | ok | 2.5x (31%) |
| blas2 | symv | f64 | n=4096 | 3.137 ms | 10.70 | 21.41 | 26% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 2.019 ms | 8.31 | 33.27 | 40% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 1.409 ms | 11.90 | 23.83 | 28% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 1.968 ms | 8.52 | 34.13 | 41% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.384 ms | 12.12 | 24.26 | 29% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.726 ms | 4.34 | 34.75 | 42% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.937 ms | 91.40 | 2.14 | 15% | ok | 6.8x (85%) |
| blas3 | gemm | f32 | 512x512x512 | 1.352 ms | 198.50 | 2.33 | 17% | ok | 7.4x (92%) |
| blas3 | gemmt | f64 | 512x512x512 | 10.690 ms | 12.56 | 0.59 | 2% | ok | 1.2x (15%) |
| blas3 | symm | f64 | 512x512x512 | 18.503 ms | 14.51 | 0.34 | 2% | ok | 1.1x (13%) |
| blas3 | syrk | f64 | 512x512x512 | 11.762 ms | 11.41 | 0.36 | 2% | ok | 1.2x (14%) |
| blas3 | syr2k | f64 | 512x512x512 | 19.719 ms | 13.61 | 0.32 | 2% | ok | 1.2x (16%) |
| blas3 | trmm | f64 | 512x512x512 | 9.174 ms | 14.63 | 0.57 | 2% | ok | 1.0x (13%) |
| blas3 | trsm | f64 | 512x512x512 | 9.217 ms | 14.56 | 0.57 | 2% | ok | 1.1x (13%) |
| blas3 | trsm | f32 | 512x512x512 | 5.112 ms | 26.26 | 0.51 | 2% | ok | 1.0x (13%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.017 ms | 0.41 | 3.26 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 7.324 ms | 12.22 | 0.57 | 2% | ok | 1.6x (20%) |
| factor | lu | f32 | 512x512 | 3.675 ms | 24.35 | 0.57 | 2% | ok | 1.2x (15%) |
| factor | cholesky | f64 | 512x512 | 7.184 ms | 6.23 | 0.58 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 6.463 ms | 6.92 | 0.32 | 1% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 7.156 ms | 6.25 | 0.59 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 4.550 ms | 9.83 | 0.92 | 2% | ok | 1.1x (13%) |
| factor | qr | f64 | 512x512 | 130.107 ms | 1.38 | 0.03 | 0% | ok | 1.8x (23%) |
| factor | lq | f64 | 512x512 | 16.481 ms | 10.86 | 0.25 | 2% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 157.208 ms | 2.85 | 0.03 | 0% | ok | 1.8x (22%) |
| factor | householder | f64 | 512x512 | 0.415 ms | 2.53 | 10.10 | 12%\* | ok | 4.0x (49%) |
| factor | inv | f64 | 512x512 | 48.814 ms | 5.50 | 0.09 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 5.588 ms | 16.01 | 0.75 | 3% | ok | 1.1x (13%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 17.799 ms | 1.26 | 0.06 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 58.164 ms | 2.60 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 736.302 ms | 0.21 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 32.335 ms | 5.19 | 0.03 | 1% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 80.721 ms | 5.20 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 109.353 ms | 3.38 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 10.766 ms | 23.38 | 0.10 | 4% | ok | 2.4x (30%) |
| eigen | sylvester | f64 | 256x256 | 178.212 ms | 5.65 | 0.01 | 1% | ok | 1.1x (13%) |
| elementwise | exp | f64 | n=8388608 | 39.080 ms | 0.21 | 3.43 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 45.635 ms | 0.18 | 2.94 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 59.869 ms | 0.14 | 2.24 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 29.368 ms | 0.29 | 4.57 | 5% | ok | 1.0x (12%) |
| elementwise | erf | f64 | n=8388608 | 205.761 ms | 0.04 | 0.65 | 1% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.113 ms | 23.13 | 240.60 | 287% ! | ok | 6.6x (82%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.918 ms | 2.85 | 29.67 | 35% | ok | 1.0x (12%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.316 ms | 31.82 | 66.90 | 80% | ok | 3.5x (44%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 5.066 ms | 0.64 | 2.27 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.521 ms | 1.32 | 10.94 | 13%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 14.645 ms | 0.76 | 0.05 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 24.644 ms | 0.45 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.450 ms | 1.32 | 0.09 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.794 ms | 0.56 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.841 ms | 0.50 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 20.802 ms | 1.08 | 0.04 | 0% | ok | 0.8x (10%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 24.716 ms | 0.08 | 0.01 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.295 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.868 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 57.102 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.182 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.939 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.052 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 12.445 ms | 0.90 | 0.06 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.252 ms | 3.00 | 11.99 | 14%\* | ok | 1.0x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 3.976 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.037 ms | 2.77 | 0.18 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.257 ms | 2.94 | 11.77 | 14%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.065 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.829 ms | 0.62 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.630 ms | 1.20 | 4.80 | 6%\* | ok | 1.0x (12%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.402 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.765 ms | 0.58 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.623 ms | 1.21 | 4.86 | 6%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.906 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 197.101 ms | 1.57 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.646 ms | 2.05 | 8.19 | 10%\* | ok | 0.9x (11%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.467 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 59.887 ms | 5.15 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.671 ms | 2.02 | 8.07 | 10%\* | ok | 0.9x (11%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.781 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 541.245 ms | 1.14 | 0.00 | 0% | ok | 1.1x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.799 ms | 0.43 | 1.73 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.091 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 454.910 ms | 1.36 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.743 ms | 0.44 | 1.74 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.642 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 46.884 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.506 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.358 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.764 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.865 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.130 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.910 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.970 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 17.821 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.226 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 15.857 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 18.648 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.039 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.504 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 11.369 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.998 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.022 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.688 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.978 ms | 2.68 | 30.00 | 36% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.199 ms | 2.18 | 24.46 | 29% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.556 ms | 1.68 | 18.85 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.384 ms | 2.87 | 21.72 | 26% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.354 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 30.067 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 32.981 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.872 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 4.808 ms | 3.49 | 13.96 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 5.122 ms | 52.41 | 0.82 | 9% | ok | 7.6x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.532 ms | 31.52 | 33.49 | 40%\* | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.307 ms | 7.27 | 29.09 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.986 ms | 17.01 | 34.03 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.685 ms | 9.96 | 9.96 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.692 ms | 9.91 | 9.91 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.378 ms | 112.89 | 1.32 | n/a | ok | 7.4x (92%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.747 ms | 97.72 | 0.76 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.354 ms | 198.27 | 1.16 | n/a | ok | 7.6x (94%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.354 ms | 198.24 | 1.16 | n/a | ok | 7.6x (94%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.543 ms | 75.76 | 0.44 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

