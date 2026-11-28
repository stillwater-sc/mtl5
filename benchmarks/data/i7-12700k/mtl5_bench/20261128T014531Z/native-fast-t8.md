## i7-12700k -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 600.8 GFLOP/s, f32 1201.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.8, f32 1201.6 GFLOP/s |
| ... this build, AVX2 | f64 600.6, f32 1201.1 GFLOP/s |
| bandwidth | read 82.0, update 79.0, triad 77.0 GB/s (ceiling 82.0) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 93% | 149% | 0 | 23% |
| blas2 | 10 | 49% | 147% | 0 | 13% |
| blas3 | 10 | 18% | 66% | 0 | 27% |
| factor | 12 | 2% | 12% | 0 | 14% |
| eigen | 8 | 1% | 4% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 300% | 0 | 13% |
| direct | 19 | 0% | 14% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 10% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 30% | 37% | 0 | 13% |
| mg | 1 | 26% | 26% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 49% | 71% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 76% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.403 ms | 11.96 | 95.66 | 117% | ok | 2.7x (33%) |
| blas1 | dot | f32 | n=8388608 | 0.549 ms | 30.58 | 122.30 | 149% | ok | 3.1x (39%) |
| blas1 | axpy | f64 | n=8388608 | 2.662 ms | 6.30 | 75.63 | 92% | ok | 1.8x (22%) |
| blas1 | axpy | f32 | n=8388608 | 1.252 ms | 13.40 | 80.38 | 98% | ok | 1.9x (23%) |
| blas1 | scale | f64 | n=8388608 | 1.760 ms | 4.77 | 76.25 | 93% | ok | 2.0x (25%) |
| blas1 | two_norm | f64 | n=8388608 | 0.568 ms | 29.56 | 118.22 | 144% | ok | 3.9x (49%) |
| blas1 | one_norm | f64 | n=8388608 | 3.956 ms | 2.12 | 16.96 | 21% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.917 ms | 1.21 | 9.70 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.909 ms | 2.15 | 17.17 | 21% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 70.023 ms | 2.64 | 1.92 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.406 ms | 23.87 | 95.53 | 117% | ok | 2.4x (30%) |
| blas2 | gemv | f32 | n=4096 | 0.556 ms | 60.38 | 120.82 | 147% | ok | 3.0x (38%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.425 ms | 23.54 | 94.22 | 115% | ok | 2.4x (30%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.676 ms | 49.67 | 99.39 | 121% | ok | 2.5x (31%) |
| blas2 | symv | f64 | n=4096 | 2.862 ms | 11.72 | 23.47 | 29% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 1.757 ms | 9.55 | 38.22 | 47% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 0.840 ms | 19.96 | 39.96 | 49% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 1.795 ms | 9.35 | 37.43 | 46% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.848 ms | 19.79 | 39.61 | 48% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.121 ms | 4.71 | 37.71 | 46% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 0.674 ms | 398.10 | 9.33 | 66% | ok | 5.6x (70%) |
| blas3 | gemm | f32 | 512x512x512 | 0.354 ms | 758.85 | 8.89 | 63% | ok | 5.4x (67%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.220 ms | 110.01 | 5.16 | 18% | ok | 2.2x (27%) |
| blas3 | symm | f64 | 512x512x512 | 0.914 ms | 293.55 | 6.88 | 49% | ok | 4.4x (55%) |
| blas3 | syrk | f64 | 512x512x512 | 1.269 ms | 105.74 | 3.30 | 18% | ok | 2.0x (25%) |
| blas3 | syr2k | f64 | 512x512x512 | 2.304 ms | 116.53 | 2.73 | 19% | ok | 2.2x (27%) |
| blas3 | trmm | f64 | 512x512x512 | 1.670 ms | 80.36 | 3.14 | 13% | ok | 1.8x (23%) |
| blas3 | trsm | f64 | 512x512x512 | 1.728 ms | 77.68 | 3.03 | 13% | ok | 1.8x (22%) |
| blas3 | trsm | f32 | 512x512x512 | 0.947 ms | 141.79 | 2.77 | 12% | ok | 1.5x (19%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.925 ms | 0.41 | 3.29 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 6.185 ms | 14.47 | 0.68 | 2% | ok | 1.8x (22%) |
| factor | lu | f32 | 512x512 | 2.808 ms | 31.87 | 0.75 | 3% | ok | 1.1x (14%) |
| factor | cholesky | f64 | 512x512 | 6.899 ms | 6.49 | 0.61 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 6.392 ms | 7.00 | 0.33 | 1% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 7.057 ms | 6.34 | 0.59 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 4.481 ms | 9.99 | 0.94 | 2% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 127.944 ms | 1.40 | 0.03 | 0% | ok | 1.8x (22%) |
| factor | lq | f64 | 512x512 | 14.376 ms | 12.45 | 0.29 | 2% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 149.733 ms | 2.99 | 0.03 | 0% | ok | 1.8x (22%) |
| factor | householder | f64 | 512x512 | 0.433 ms | 2.43 | 9.69 | 12%\* | ok | 3.7x (47%) |
| factor | inv | f64 | 512x512 | 49.272 ms | 5.45 | 0.09 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 4.293 ms | 20.84 | 0.98 | 3% | ok | 1.1x (13%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 16.330 ms | 1.37 | 0.06 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 56.193 ms | 2.69 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 742.754 ms | 0.20 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 30.267 ms | 5.54 | 0.03 | 1% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 79.057 ms | 5.31 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 88.584 ms | 4.17 | 0.01 | 1% | ok | 1.1x (13%) |
| eigen | expm | f64 | 256x256 | 9.398 ms | 26.78 | 0.11 | 4% | ok | 1.2x (15%) |
| eigen | sylvester | f64 | 256x256 | 173.001 ms | 5.82 | 0.01 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 38.744 ms | 0.22 | 3.46 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 44.672 ms | 0.19 | 3.00 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 58.933 ms | 0.14 | 2.28 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.486 ms | 0.29 | 4.71 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 201.451 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.111 ms | 23.62 | 245.78 | 300% ! | ok | 6.6x (83%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.911 ms | 2.87 | 29.88 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.304 ms | 32.12 | 67.54 | 82% | ok | 3.1x (39%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.839 ms | 0.67 | 2.38 | 3%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.521 ms | 1.32 | 10.95 | 13%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 14.977 ms | 0.75 | 0.05 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 26.614 ms | 0.42 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.987 ms | 1.24 | 0.08 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.112 ms | 0.56 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.507 ms | 0.50 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 20.579 ms | 1.09 | 0.04 | 0% | ok | 0.8x (10%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 24.951 ms | 0.08 | 0.01 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.388 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.886 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 57.312 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.188 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.156 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.093 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 12.448 ms | 0.90 | 0.06 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.258 ms | 2.94 | 11.74 | 14%\* | ok | 1.1x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.044 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.476 ms | 2.50 | 0.16 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.279 ms | 2.71 | 10.84 | 13%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.075 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.946 ms | 0.62 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.622 ms | 1.22 | 4.86 | 6%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.399 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.737 ms | 0.58 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.624 ms | 1.21 | 4.85 | 6%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.848 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 197.584 ms | 1.56 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.638 ms | 2.06 | 8.23 | 10%\* | ok | 0.9x (11%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.392 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 67.601 ms | 4.56 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.854 ms | 1.82 | 7.27 | 9%\* | ok | 0.9x (11%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.863 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 516.051 ms | 1.20 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.740 ms | 0.44 | 1.74 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.176 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 690.476 ms | 0.89 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.725 ms | 0.44 | 1.74 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.089 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 29.902 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 21.162 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.453 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.697 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.484 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 19.821 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.560 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.386 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.088 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 13.678 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 12.992 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 14.353 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.034 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.559 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 11.749 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.065 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 1.011 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.027 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.707 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.964 ms | 2.72 | 30.43 | 37% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.198 ms | 2.19 | 24.49 | 30% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.396 ms | 1.88 | 21.01 | 26% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.515 ms | 2.85 | 21.51 | 26% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 27.705 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 28.215 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 32.780 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.518 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 1.852 ms | 9.06 | 36.24 | 44% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 0.905 ms | 296.65 | 4.64 | 49% | ok | 6.6x (83%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.307 ms | 54.63 | 58.05 | 71%\* | ok | 1.1x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.889 ms | 8.88 | 35.53 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.747 ms | 22.45 | 44.90 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.269 ms | 62.26 | 62.26 | n/a | ok | 1.1x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.253 ms | 66.27 | 66.27 | n/a | ok | 1.1x (14%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 0.622 ms | 431.44 | 5.06 | n/a | ok | 6.4x (80%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 0.824 ms | 325.71 | 2.54 | n/a | ok | 6.8x (85%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 0.810 ms | 331.28 | 1.94 | n/a | ok | 6.9x (86%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.330 ms | 814.51 | 4.77 | n/a | ok | 6.4x (80%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.306 ms | 878.15 | 5.15 | n/a | ok | 6.1x (76%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

