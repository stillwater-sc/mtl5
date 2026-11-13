## i7-12700k -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `8867b44b3f52` |
| FMA peak (ceiling) | f64 601.0 GFLOP/s, f32 1201.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 601.0, f32 1201.6 GFLOP/s |
| ... this build, scalar | f64 10.0, f32 10.0 GFLOP/s |
| bandwidth | read 82.6, update 77.0, triad 75.8 GB/s (ceiling 82.6) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 85% | 124% | 0 | 28% |
| blas2 | 10 | 119% | 198% | 0 | 32% |
| blas3 | 10 | 66% | 85% | 0 | 77% |
| factor | 12 | 5% | 12% | 0 | 17% |
| eigen | 8 | 3% | 10% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 37% | 289% | 0 | 13% |
| direct | 31 | 1% | 14% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| direct3d | 21 | 11% | 32% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 30% | 37% | 0 | 13% |
| mg | 1 | 26% | 26% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 40% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 84% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.601 ms | 10.48 | 83.81 | 101% | ok | 3.5x (44%) |
| blas1 | dot | f32 | n=8388608 | 0.655 ms | 25.63 | 102.53 | 124% | ok | 6.4x (80%) |
| blas1 | axpy | f64 | n=8388608 | 2.851 ms | 5.88 | 70.61 | 85% | ok | 2.3x (28%) |
| blas1 | axpy | f32 | n=8388608 | 1.439 ms | 11.66 | 69.95 | 85% | ok | 3.4x (43%) |
| blas1 | scale | f64 | n=8388608 | 1.915 ms | 4.38 | 70.08 | 85% | ok | 1.9x (23%) |
| blas1 | two_norm | f64 | n=8388608 | 0.737 ms | 22.76 | 91.06 | 110% | ok | 6.5x (82%) |
| blas1 | one_norm | f64 | n=8388608 | 3.923 ms | 2.14 | 17.11 | 21% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.902 ms | 1.22 | 9.72 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.895 ms | 2.15 | 17.23 | 21% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 75.114 ms | 2.46 | 1.79 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.429 ms | 23.48 | 93.98 | 114% | ok | 2.4x (31%) |
| blas2 | gemv | f32 | n=4096 | 0.544 ms | 61.73 | 123.52 | 149% | ok | 3.0x (38%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.422 ms | 23.60 | 94.46 | 114% | ok | 2.5x (32%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.683 ms | 49.14 | 98.34 | 119% | ok | 2.4x (30%) |
| blas2 | symv | f64 | n=4096 | 0.630 ms | 53.29 | 106.69 | 129% | ok | 3.0x (38%) |
| blas2 | trmv | f64 | n=4096 | 0.599 ms | 28.00 | 112.12 | 136% | ok | 3.1x (39%) |
| blas2 | trmv | f32 | n=4096 | 0.205 ms | 81.81 | 163.77 | 198% ! | ok | 4.6x (57%) |
| blas2 | trsv | f64 | n=4096 | 1.752 ms | 9.58 | 38.35 | 46% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.852 ms | 19.69 | 39.43 | 48% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.707 ms | 4.35 | 34.84 | 42% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 0.544 ms | 493.50 | 11.57 | 82% | ok | 6.8x (85%) |
| blas3 | gemm | f32 | 512x512x512 | 0.264 ms | 1016.35 | 11.91 | 85% | ok | 6.9x (87%) |
| blas3 | gemmt | f64 | 512x512x512 | 0.700 ms | 191.78 | 8.99 | 32% | ok | 3.7x (46%) |
| blas3 | symm | f64 | 512x512x512 | 0.544 ms | 493.72 | 11.57 | 82% | ok | 6.8x (85%) |
| blas3 | syrk | f64 | 512x512x512 | 0.748 ms | 179.34 | 5.60 | 30% | ok | 3.4x (43%) |
| blas3 | syr2k | f64 | 512x512x512 | 1.117 ms | 240.27 | 5.63 | 40% | ok | 4.2x (53%) |
| blas3 | trmm | f64 | 512x512x512 | 0.341 ms | 393.22 | 15.36 | 65% | ok | 6.2x (77%) |
| blas3 | trsm | f64 | 512x512x512 | 0.339 ms | 395.69 | 15.46 | 66% | ok | 6.3x (79%) |
| blas3 | trsm | f32 | 512x512x512 | 0.159 ms | 843.82 | 16.48 | 70% | ok | 6.1x (76%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.901 ms | 0.41 | 3.29 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 2.095 ms | 42.71 | 2.00 | 7% | ok | 1.6x (20%) |
| factor | lu | f32 | 512x512 | 1.314 ms | 68.10 | 1.60 | 6% | ok | 1.4x (17%) |
| factor | cholesky | f64 | 512x512 | 1.523 ms | 29.38 | 2.75 | 5% | ok | 1.3x (17%) |
| factor | cholesky | f32 | 512x512 | 1.003 ms | 44.62 | 2.09 | 4% | ok | 1.3x (16%) |
| factor | ldlt | f64 | 512x512 | 7.200 ms | 6.21 | 0.58 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 43.946 ms | 1.02 | 0.10 | 0% | ok | 1.1x (13%) |
| factor | qr | f64 | 512x512 | 4.098 ms | 43.67 | 1.02 | 7% | ok | 1.4x (18%) |
| factor | lq | f64 | 512x512 | 17.570 ms | 10.19 | 0.24 | 2% | ok | 1.5x (18%) |
| factor | hessenberg | f64 | 512x512 | 166.554 ms | 2.69 | 0.03 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 0.426 ms | 2.47 | 9.85 | 12%\* | ok | 3.8x (48%) |
| factor | inv | f64 | 512x512 | 45.830 ms | 5.86 | 0.09 | 1% | ok | 1.0x (12%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 2.779 ms | 32.20 | 1.51 | 5% | ok | 1.4x (17%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.642 ms | 13.62 | 0.64 | 2% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 57.566 ms | 2.62 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 731.209 ms | 0.21 | 0.00 | 0% | ok | 1.1x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 17.436 ms | 9.62 | 0.06 | 2% | ok | 0.9x (11%) |
| eigen | schur | f64 | 256x256 | 21.723 ms | 19.31 | 0.05 | 3% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 6.123 ms | 60.28 | 0.17 | 10% | ok | 1.1x (14%) |
| eigen | expm | f64 | 256x256 | 8.698 ms | 28.93 | 0.12 | 5% | ok | 1.3x (16%) |
| eigen | sylvester | f64 | 256x256 | 57.295 ms | 17.57 | 0.03 | 3% | ok | 0.9x (12%) |
| elementwise | exp | f64 | n=8388608 | 39.539 ms | 0.21 | 3.39 | 4% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 45.051 ms | 0.19 | 2.98 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 59.543 ms | 0.14 | 2.25 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.787 ms | 0.29 | 4.66 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 201.101 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.114 ms | 22.94 | 238.69 | 289% ! | ok | 6.5x (81%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.903 ms | 2.90 | 30.16 | 37% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.405 ms | 29.80 | 62.65 | 76% | ok | 3.4x (42%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.841 ms | 0.67 | 2.38 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.379 ms | 1.46 | 12.07 | 15%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 89.809 ms | 0.12 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.812 ms | 0.43 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.880 ms | 1.26 | 0.08 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.963 ms | 0.55 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 53.166 ms | 0.42 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 17.008 ms | 1.32 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 85.593 ms | 0.02 | 0.00 | 0% | ok | 1.2x (15%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.280 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.874 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.215 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.992 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.287 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 87.254 ms | 0.13 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.268 ms | 2.82 | 11.29 | 14%\* | ok | 0.9x (11%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.389 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.123 ms | 2.71 | 0.18 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.262 ms | 2.88 | 11.53 | 14%\* | ok | 0.9x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.041 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 36.359 ms | 0.62 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.598 ms | 1.26 | 5.06 | 6%\* | ok | 1.1x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.809 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.240 ms | 0.47 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.603 ms | 1.25 | 5.02 | 6%\* | ok | 1.1x (14%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 6.040 ms | 1.85 | 0.12 | 0% | ok | 0.9x (12%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 1.851 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 3.686 ms | 3.04 | 0.20 | 1% | ok | 0.9x (11%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 0.328 ms | 2.30 | 9.21 | 11%\* | ok | 1.0x (12%) |
| direct | ref_pardiso_spd | f64 | 96^2 grid, nnz 45696 | 3.301 ms | 3.39 | 0.22 | 1% | ok | 1.4x (17%) |
| direct | ref_pardiso_spd.analyze | f64 | 96^2 grid, nnz 45696 | 1.994 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct | ref_pardiso_spd.factor | f64 | 96^2 grid, nnz 45696 | 0.630 ms | 17.76 | 1.16 | 3% | ok | 3.0x (38%) |
| direct | ref_pardiso_spd.solve | f64 | 96^2 grid, nnz 45696 | 0.499 ms | 1.52 | 6.06 | 7%\* | ok | 0.9x (12%) |
| direct | ref_pardiso_lu | f64 | 96^2 grid, nnz 45696 | 5.611 ms | 3.99 | 0.13 | 1% | ok | 1.3x (17%) |
| direct | ref_pardiso_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.525 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | ref_pardiso_lu.factor | f64 | 96^2 grid, nnz 45696 | 1.264 ms | 17.69 | 0.58 | 3% | ok | 2.5x (32%) |
| direct | ref_pardiso_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.644 ms | 1.17 | 4.70 | 6%\* | ok | 0.9x (11%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 9.307 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 7283.432 ms | 0.04 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.532 ms | 2.20 | 8.80 | 11%\* | ok | 0.9x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 18.276 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 65.265 ms | 4.73 | 0.01 | 1% | ok | 1.1x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.550 ms | 2.17 | 8.69 | 11%\* | ok | 0.9x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.775 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 540.304 ms | 1.14 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 8.052 ms | 0.42 | 1.67 | 2%\* | ok | 0.9x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.365 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 617.762 ms | 1.00 | 0.00 | 0% | ok | 1.1x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 8.045 ms | 0.42 | 1.68 | 2%\* | ok | 0.9x (12%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 3.804 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 8.388 ms | 36.79 | 0.10 | 6% | ok | 1.4x (17%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 0.512 ms | 6.58 | 26.31 | 32%\* | ok | 1.0x (12%) |
| direct3d | ref_pardiso_spd.analyze | f64 | 20^3 grid, nnz 53600 | 7.277 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | ref_pardiso_spd.factor | f64 | 20^3 grid, nnz 53600 | 2.379 ms | 129.70 | 0.36 | 22% | ok | 4.1x (51%) |
| direct3d | ref_pardiso_spd.solve | f64 | 20^3 grid, nnz 53600 | 0.945 ms | 3.57 | 14.26 | 17%\* | ok | 0.9x (12%) |
| direct3d | ref_pardiso_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.136 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | ref_pardiso_lu.factor | f64 | 20^3 grid, nnz 53600 | 5.572 ms | 110.77 | 0.15 | 18% | ok | 3.4x (43%) |
| direct3d | ref_pardiso_lu.solve | f64 | 20^3 grid, nnz 53600 | 1.145 ms | 2.94 | 11.77 | 14%\* | ok | 0.9x (11%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.447 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 32.039 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.389 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.219 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.623 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.705 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.375 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.747 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.934 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 16.832 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.052 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.537 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 20.877 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.034 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.532 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 12.008 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.049 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.984 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.163 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.959 ms | 2.73 | 30.59 | 37% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.196 ms | 2.19 | 24.51 | 30% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.534 ms | 1.71 | 19.12 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.764 ms | 2.79 | 21.12 | 26% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.662 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 31.895 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.199 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.889 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 4.793 ms | 3.50 | 14.00 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 5.162 ms | 52.00 | 0.81 | 9% | ok | 7.5x (94%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.543 ms | 30.91 | 32.84 | 40%\* | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.260 ms | 7.42 | 29.70 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.976 ms | 17.19 | 34.39 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.683 ms | 9.97 | 9.97 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.683 ms | 9.97 | 9.97 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.599 ms | 103.29 | 1.21 | n/a | ok | 6.7x (84%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.747 ms | 97.73 | 0.76 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.355 ms | 198.06 | 1.16 | n/a | ok | 7.5x (94%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.356 ms | 197.97 | 1.16 | n/a | ok | 7.5x (94%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.549 ms | 75.64 | 0.44 | n/a | ok | 7.6x (94%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

