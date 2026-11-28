## i7-12700k -- native, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 600.4 GFLOP/s, f32 1199.3 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.4, f32 1199.3 GFLOP/s |
| ... this build, scalar | f64 9.9, f32 10.0 GFLOP/s |
| bandwidth | read 82.9, update 77.6, triad 76.1 GB/s (ceiling 82.9) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 89% | 125% | 0 | 29% |
| blas2 | 10 | 42% | 138% | 0 | 13% |
| blas3 | 10 | 2% | 15% | 0 | 14% |
| factor | 12 | 2% | 12% | 0 | 16% |
| eigen | 8 | 1% | 4% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 292% | 0 | 13% |
| direct | 19 | 0% | 14% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 10% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 36% | 0 | 13% |
| mg | 1 | 26% | 26% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 39% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 93% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.409 ms | 11.90 | 95.24 | 115% | ok | 4.2x (52%) |
| blas1 | dot | f32 | n=8388608 | 0.650 ms | 25.80 | 103.21 | 125% | ok | 6.5x (81%) |
| blas1 | axpy | f64 | n=8388608 | 2.742 ms | 6.12 | 73.42 | 89% | ok | 2.3x (29%) |
| blas1 | axpy | f32 | n=8388608 | 1.367 ms | 12.28 | 73.66 | 89% | ok | 3.5x (43%) |
| blas1 | scale | f64 | n=8388608 | 1.773 ms | 4.73 | 75.71 | 91% | ok | 2.0x (25%) |
| blas1 | two_norm | f64 | n=8388608 | 0.767 ms | 21.87 | 87.47 | 106% | ok | 6.4x (80%) |
| blas1 | one_norm | f64 | n=8388608 | 4.048 ms | 2.07 | 16.58 | 20% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.974 ms | 1.20 | 9.62 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.987 ms | 2.10 | 16.83 | 20% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 76.078 ms | 2.43 | 1.76 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.444 ms | 23.24 | 92.99 | 112% | ok | 2.5x (32%) |
| blas2 | gemv | f32 | n=4096 | 0.587 ms | 57.21 | 114.47 | 138% | ok | 4.7x (58%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.584 ms | 21.18 | 84.77 | 102% | ok | 3.0x (38%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.860 ms | 39.03 | 78.10 | 94% | ok | 2.6x (32%) |
| blas2 | symv | f64 | n=4096 | 3.153 ms | 10.64 | 21.31 | 26% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 2.001 ms | 8.38 | 33.57 | 41% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 1.408 ms | 11.92 | 23.86 | 29% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 1.957 ms | 8.57 | 34.32 | 41% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.385 ms | 12.11 | 24.25 | 29% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.669 ms | 4.38 | 35.01 | 42% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 2.972 ms | 90.33 | 2.12 | 15% | ok | 6.7x (84%) |
| blas3 | gemm | f32 | 512x512x512 | 1.503 ms | 178.66 | 2.09 | 15% | ok | 6.9x (86%) |
| blas3 | gemmt | f64 | 512x512x512 | 10.839 ms | 12.38 | 0.58 | 2% | ok | 1.2x (14%) |
| blas3 | symm | f64 | 512x512x512 | 18.995 ms | 14.13 | 0.33 | 2% | ok | 1.1x (13%) |
| blas3 | syrk | f64 | 512x512x512 | 11.726 ms | 11.45 | 0.36 | 2% | ok | 1.2x (15%) |
| blas3 | syr2k | f64 | 512x512x512 | 18.845 ms | 14.24 | 0.33 | 2% | ok | 1.3x (16%) |
| blas3 | trmm | f64 | 512x512x512 | 9.248 ms | 14.51 | 0.57 | 2% | ok | 1.0x (13%) |
| blas3 | trsm | f64 | 512x512x512 | 9.332 ms | 14.38 | 0.56 | 2% | ok | 1.0x (13%) |
| blas3 | trsm | f32 | 512x512x512 | 5.119 ms | 26.22 | 0.51 | 2% | ok | 1.0x (13%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.977 ms | 0.41 | 3.27 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 6.825 ms | 13.11 | 0.61 | 2% | ok | 1.7x (21%) |
| factor | lu | f32 | 512x512 | 3.611 ms | 24.78 | 0.58 | 2% | ok | 1.3x (16%) |
| factor | cholesky | f64 | 512x512 | 7.123 ms | 6.28 | 0.59 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 6.454 ms | 6.93 | 0.32 | 1% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 7.215 ms | 6.20 | 0.58 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 4.686 ms | 9.55 | 0.90 | 2% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 127.989 ms | 1.40 | 0.03 | 0% | ok | 1.8x (23%) |
| factor | lq | f64 | 512x512 | 16.616 ms | 10.77 | 0.25 | 2% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 156.488 ms | 2.86 | 0.03 | 0% | ok | 1.7x (22%) |
| factor | householder | f64 | 512x512 | 0.409 ms | 2.57 | 10.26 | 12%\* | ok | 4.1x (51%) |
| factor | inv | f64 | 512x512 | 48.207 ms | 5.57 | 0.09 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 5.494 ms | 16.29 | 0.76 | 3% | ok | 1.1x (14%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 18.001 ms | 1.24 | 0.06 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 57.977 ms | 2.60 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 742.614 ms | 0.20 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 32.078 ms | 5.23 | 0.03 | 1% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 79.941 ms | 5.25 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 108.846 ms | 3.39 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 10.776 ms | 23.35 | 0.10 | 4% | ok | 2.4x (30%) |
| eigen | sylvester | f64 | 256x256 | 179.456 ms | 5.61 | 0.01 | 1% | ok | 1.1x (13%) |
| elementwise | exp | f64 | n=8388608 | 39.454 ms | 0.21 | 3.40 | 4% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 45.385 ms | 0.18 | 2.96 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 59.272 ms | 0.14 | 2.26 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.685 ms | 0.29 | 4.68 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 201.271 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.112 ms | 23.30 | 242.36 | 292% ! | ok | 6.7x (84%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.902 ms | 2.90 | 30.19 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.313 ms | 31.90 | 67.06 | 81% | ok | 3.5x (44%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 5.063 ms | 0.64 | 2.27 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.538 ms | 1.31 | 10.82 | 13%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 14.451 ms | 0.77 | 0.05 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 24.682 ms | 0.45 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.406 ms | 1.33 | 0.09 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.532 ms | 0.57 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.056 ms | 0.51 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.525 ms | 1.35 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 24.632 ms | 0.08 | 0.01 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.287 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.916 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 57.056 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.184 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.958 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.067 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 11.849 ms | 0.94 | 0.06 | 0% | ok | 1.1x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.254 ms | 2.98 | 11.91 | 14%\* | ok | 1.0x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 3.998 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 3.986 ms | 2.81 | 0.18 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.258 ms | 2.93 | 11.74 | 14%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.032 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.378 ms | 0.63 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.614 ms | 1.23 | 4.93 | 6%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.350 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.292 ms | 0.58 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.619 ms | 1.22 | 4.89 | 6%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.818 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 197.888 ms | 1.56 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.634 ms | 2.06 | 8.25 | 10%\* | ok | 0.9x (11%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.418 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 58.555 ms | 5.27 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.653 ms | 2.04 | 8.15 | 10%\* | ok | 0.9x (11%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.783 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 542.762 ms | 1.14 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.708 ms | 0.44 | 1.75 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.112 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 455.075 ms | 1.36 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.714 ms | 0.44 | 1.75 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.651 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 33.794 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.495 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.318 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.796 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.894 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.080 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.767 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.037 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 17.749 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.267 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 15.830 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 18.647 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.041 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.509 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 11.203 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.067 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.998 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.026 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.708 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.980 ms | 2.67 | 29.93 | 36% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.200 ms | 2.18 | 24.43 | 29% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.536 ms | 1.70 | 19.10 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.520 ms | 2.84 | 21.50 | 26% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.604 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 30.362 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 32.933 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.886 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 4.804 ms | 3.49 | 13.97 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 5.122 ms | 52.41 | 0.82 | 9% | ok | 7.6x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.546 ms | 30.71 | 32.63 | 39%\* | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.258 ms | 7.43 | 29.72 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.975 ms | 17.21 | 34.42 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.684 ms | 9.96 | 9.96 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.685 ms | 9.96 | 9.96 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.362 ms | 113.67 | 1.33 | n/a | ok | 7.4x (93%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.743 ms | 97.85 | 0.76 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.354 ms | 198.26 | 1.16 | n/a | ok | 7.6x (94%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.354 ms | 198.22 | 1.16 | n/a | ok | 7.6x (94%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.542 ms | 75.78 | 0.44 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

