## i7-12700k -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 600.2 GFLOP/s, f32 1201.3 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.0, f32 1201.0 GFLOP/s |
| ... this build, AVX2 | f64 600.2, f32 1201.3 GFLOP/s |
| bandwidth | read 81.8, update 78.6, triad 76.6 GB/s (ceiling 81.8) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 93% | 147% | 0 | 24% |
| blas2 | 10 | 48% | 149% | 0 | 13% |
| blas3 | 10 | 18% | 67% | 0 | 26% |
| factor | 12 | 2% | 13% | 0 | 15% |
| eigen | 8 | 1% | 4% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 37% | 303% | 0 | 13% |
| direct | 19 | 0% | 14% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 10% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 30% | 37% | 0 | 13% |
| mg | 1 | 26% | 26% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 49% | 72% | 0 | 14% |
| integer | 9 | n/a | n/a | 0 | 77% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.397 ms | 12.01 | 96.06 | 117% | ok | 2.6x (33%) |
| blas1 | dot | f32 | n=8388608 | 0.557 ms | 30.13 | 120.50 | 147% | ok | 3.1x (38%) |
| blas1 | axpy | f64 | n=8388608 | 2.689 ms | 6.24 | 74.87 | 92% | ok | 1.8x (22%) |
| blas1 | axpy | f32 | n=8388608 | 1.254 ms | 13.38 | 80.30 | 98% | ok | 1.9x (24%) |
| blas1 | scale | f64 | n=8388608 | 1.764 ms | 4.76 | 76.09 | 93% | ok | 2.0x (25%) |
| blas1 | two_norm | f64 | n=8388608 | 0.581 ms | 28.86 | 115.43 | 141% | ok | 3.8x (48%) |
| blas1 | one_norm | f64 | n=8388608 | 4.053 ms | 2.07 | 16.56 | 20% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.951 ms | 1.21 | 9.66 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.940 ms | 2.13 | 17.03 | 21% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 70.847 ms | 2.60 | 1.89 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.395 ms | 24.05 | 96.25 | 118% | ok | 2.4x (31%) |
| blas2 | gemv | f32 | n=4096 | 0.551 ms | 60.89 | 121.85 | 149% | ok | 3.1x (38%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.435 ms | 23.39 | 93.60 | 114% | ok | 2.4x (30%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.681 ms | 49.26 | 98.57 | 121% | ok | 2.5x (31%) |
| blas2 | symv | f64 | n=4096 | 2.884 ms | 11.64 | 23.30 | 28% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 1.768 ms | 9.49 | 38.00 | 46% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 0.848 ms | 19.77 | 39.59 | 48% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 1.793 ms | 9.35 | 37.46 | 46% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.848 ms | 19.77 | 39.59 | 48% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.107 ms | 4.72 | 37.78 | 46% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 0.672 ms | 399.29 | 9.36 | 67% | ok | 5.6x (71%) |
| blas3 | gemm | f32 | 512x512x512 | 0.352 ms | 762.57 | 8.94 | 63% | ok | 5.4x (67%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.218 ms | 110.24 | 5.17 | 18% | ok | 2.2x (27%) |
| blas3 | symm | f64 | 512x512x512 | 0.921 ms | 291.43 | 6.83 | 49% | ok | 4.4x (55%) |
| blas3 | syrk | f64 | 512x512x512 | 1.274 ms | 105.38 | 3.29 | 18% | ok | 2.0x (25%) |
| blas3 | syr2k | f64 | 512x512x512 | 2.372 ms | 113.17 | 2.65 | 19% | ok | 2.1x (26%) |
| blas3 | trmm | f64 | 512x512x512 | 1.551 ms | 86.53 | 3.38 | 14% | ok | 2.0x (25%) |
| blas3 | trsm | f64 | 512x512x512 | 1.760 ms | 76.28 | 2.98 | 13% | ok | 1.8x (22%) |
| blas3 | trsm | f32 | 512x512x512 | 0.950 ms | 141.31 | 2.76 | 12% | ok | 1.5x (19%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.993 ms | 0.41 | 3.27 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 6.307 ms | 14.19 | 0.67 | 2% | ok | 1.8x (23%) |
| factor | lu | f32 | 512x512 | 2.799 ms | 31.97 | 0.75 | 3% | ok | 1.2x (15%) |
| factor | cholesky | f64 | 512x512 | 6.931 ms | 6.46 | 0.61 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 6.412 ms | 6.98 | 0.33 | 1% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 7.058 ms | 6.34 | 0.59 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 4.451 ms | 10.05 | 0.94 | 2% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 126.058 ms | 1.42 | 0.03 | 0% | ok | 1.8x (23%) |
| factor | lq | f64 | 512x512 | 14.404 ms | 12.42 | 0.29 | 2% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 150.218 ms | 2.98 | 0.03 | 0% | ok | 1.8x (22%) |
| factor | householder | f64 | 512x512 | 0.395 ms | 2.66 | 10.62 | 13%\* | ok | 4.1x (52%) |
| factor | inv | f64 | 512x512 | 49.219 ms | 5.45 | 0.09 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 4.325 ms | 20.69 | 0.97 | 3% | ok | 1.1x (13%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 16.773 ms | 1.33 | 0.06 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 55.968 ms | 2.70 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 741.996 ms | 0.20 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 30.073 ms | 5.58 | 0.03 | 1% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 79.202 ms | 5.30 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 91.234 ms | 4.05 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 9.360 ms | 26.89 | 0.11 | 4% | ok | 1.2x (16%) |
| eigen | sylvester | f64 | 256x256 | 170.781 ms | 5.89 | 0.01 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 39.299 ms | 0.21 | 3.42 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 45.695 ms | 0.18 | 2.94 | 4% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 59.301 ms | 0.14 | 2.26 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.559 ms | 0.29 | 4.70 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 201.655 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.110 ms | 23.80 | 247.64 | 303% ! | ok | 6.7x (84%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.912 ms | 2.87 | 29.86 | 37% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.307 ms | 32.05 | 67.39 | 82% | ok | 3.1x (39%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.808 ms | 0.68 | 2.39 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.535 ms | 1.31 | 10.84 | 13%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 15.037 ms | 0.74 | 0.05 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 26.662 ms | 0.42 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.977 ms | 1.25 | 0.08 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.598 ms | 0.57 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.115 ms | 0.51 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 20.483 ms | 1.09 | 0.04 | 0% | ok | 0.8x (10%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 24.561 ms | 0.08 | 0.01 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.361 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.945 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 57.768 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.189 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.213 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.108 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 12.460 ms | 0.90 | 0.06 | 0% | ok | 1.1x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.257 ms | 2.95 | 11.79 | 14%\* | ok | 1.0x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.038 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.464 ms | 2.51 | 0.16 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.279 ms | 2.71 | 10.85 | 13%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.069 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.878 ms | 0.62 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.632 ms | 1.20 | 4.78 | 6%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.388 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.669 ms | 0.58 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.615 ms | 1.23 | 4.92 | 6%\* | ok | 1.1x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.890 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 194.867 ms | 1.58 | 0.00 | 0% | ok | 1.1x (14%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.646 ms | 2.05 | 8.19 | 10%\* | ok | 0.9x (11%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.418 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 67.453 ms | 4.57 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.856 ms | 1.82 | 7.26 | 9%\* | ok | 1.0x (13%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.828 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 512.958 ms | 1.20 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.705 ms | 0.44 | 1.75 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.171 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 692.225 ms | 0.89 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.722 ms | 0.44 | 1.75 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.092 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 29.759 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 20.819 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.435 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.684 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.443 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 19.783 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.564 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.372 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 14.998 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 13.644 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 12.938 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 14.231 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.035 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.541 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 11.745 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.059 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 1.000 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.027 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.710 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.968 ms | 2.71 | 30.31 | 37% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.200 ms | 2.18 | 24.44 | 30% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.396 ms | 1.88 | 21.01 | 26% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.455 ms | 2.86 | 21.60 | 26% | ok | 1.0x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 27.816 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 28.372 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 32.613 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.531 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 1.977 ms | 8.49 | 33.94 | 42% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 0.913 ms | 294.14 | 4.60 | 49% | ok | 6.6x (82%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.301 ms | 55.69 | 59.17 | 72%\* | ok | 1.1x (14%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.908 ms | 8.79 | 35.17 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.746 ms | 22.50 | 45.00 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.262 ms | 64.03 | 64.03 | n/a | ok | 1.1x (14%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.247 ms | 68.05 | 68.05 | n/a | ok | 1.1x (14%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 0.635 ms | 422.45 | 4.95 | n/a | ok | 6.2x (78%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 0.826 ms | 324.97 | 2.54 | n/a | ok | 6.8x (85%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 0.814 ms | 329.78 | 1.93 | n/a | ok | 6.8x (85%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.336 ms | 799.45 | 4.68 | n/a | ok | 6.3x (78%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.305 ms | 881.07 | 5.16 | n/a | ok | 6.2x (77%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

