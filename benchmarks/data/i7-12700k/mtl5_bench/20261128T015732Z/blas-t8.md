## i7-12700k -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 600.5 GFLOP/s, f32 1201.3 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.5, f32 1201.3 GFLOP/s |
| ... this build, scalar | f64 10.0, f32 9.9 GFLOP/s |
| bandwidth | read 81.5, update 77.0, triad 75.7 GB/s (ceiling 81.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 92% | 126% | 0 | 30% |
| blas2 | 10 | 121% | 199% | 0 | 32% |
| blas3 | 10 | 66% | 84% | 0 | 78% |
| factor | 12 | 5% | 12% | 0 | 17% |
| eigen | 8 | 3% | 10% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 37% | 300% | 0 | 13% |
| direct | 31 | 1% | 15% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 21 | 10% | 32% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 30% | 37% | 0 | 13% |
| mg | 1 | 27% | 27% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 40% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 94% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.392 ms | 12.05 | 96.42 | 118% | ok | 4.1x (52%) |
| blas1 | dot | f32 | n=8388608 | 0.653 ms | 25.70 | 102.82 | 126% | ok | 6.6x (82%) |
| blas1 | axpy | f64 | n=8388608 | 2.714 ms | 6.18 | 74.17 | 91% | ok | 2.4x (30%) |
| blas1 | axpy | f32 | n=8388608 | 1.349 ms | 12.44 | 74.65 | 92% | ok | 3.5x (44%) |
| blas1 | scale | f64 | n=8388608 | 1.773 ms | 4.73 | 75.71 | 93% | ok | 2.0x (25%) |
| blas1 | two_norm | f64 | n=8388608 | 0.745 ms | 22.52 | 90.08 | 111% | ok | 6.6x (82%) |
| blas1 | one_norm | f64 | n=8388608 | 3.945 ms | 2.13 | 17.01 | 21% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.952 ms | 1.21 | 9.65 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.928 ms | 2.14 | 17.09 | 21% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 75.971 ms | 2.43 | 1.77 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.416 ms | 23.69 | 94.80 | 116% | ok | 2.5x (31%) |
| blas2 | gemv | f32 | n=4096 | 0.548 ms | 61.24 | 122.55 | 150% ! | ok | 3.0x (38%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.422 ms | 23.59 | 94.42 | 116% | ok | 2.6x (32%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.682 ms | 49.18 | 98.41 | 121% | ok | 2.4x (30%) |
| blas2 | symv | f64 | n=4096 | 0.622 ms | 53.94 | 107.99 | 133% | ok | 3.1x (39%) |
| blas2 | trmv | f64 | n=4096 | 0.601 ms | 27.90 | 111.70 | 137% | ok | 3.1x (39%) |
| blas2 | trmv | f32 | n=4096 | 0.207 ms | 80.95 | 162.07 | 199% ! | ok | 4.6x (57%) |
| blas2 | trsv | f64 | n=4096 | 1.761 ms | 9.53 | 38.15 | 47% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.855 ms | 19.63 | 39.31 | 48% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 3.633 ms | 9.24 | 73.91 | 91% | ok | 1.9x (23%) |
| blas3 | gemm | f64 | 512x512x512 | 0.537 ms | 500.04 | 11.72 | 83% | ok | 6.9x (86%) |
| blas3 | gemm | f32 | 512x512x512 | 0.265 ms | 1012.43 | 11.86 | 84% | ok | 6.9x (86%) |
| blas3 | gemmt | f64 | 512x512x512 | 0.702 ms | 191.26 | 8.97 | 32% | ok | 3.7x (46%) |
| blas3 | symm | f64 | 512x512x512 | 0.545 ms | 492.61 | 11.55 | 82% | ok | 6.8x (85%) |
| blas3 | syrk | f64 | 512x512x512 | 0.750 ms | 178.86 | 5.59 | 30% | ok | 3.4x (42%) |
| blas3 | syr2k | f64 | 512x512x512 | 1.114 ms | 240.91 | 5.65 | 40% | ok | 4.2x (53%) |
| blas3 | trmm | f64 | 512x512x512 | 0.340 ms | 395.22 | 15.44 | 66% | ok | 6.2x (78%) |
| blas3 | trsm | f64 | 512x512x512 | 0.337 ms | 398.12 | 15.55 | 66% | ok | 6.3x (79%) |
| blas3 | trsm | f32 | 512x512x512 | 0.159 ms | 842.15 | 16.45 | 70% | ok | 6.0x (75%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.119 ms | 0.40 | 3.24 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 2.101 ms | 42.59 | 2.00 | 7% | ok | 1.6x (20%) |
| factor | lu | f32 | 512x512 | 1.359 ms | 65.85 | 1.54 | 5% | ok | 1.4x (17%) |
| factor | cholesky | f64 | 512x512 | 1.526 ms | 29.33 | 2.75 | 5% | ok | 1.3x (17%) |
| factor | cholesky | f32 | 512x512 | 1.039 ms | 43.04 | 2.02 | 4% | ok | 1.2x (14%) |
| factor | ldlt | f64 | 512x512 | 7.173 ms | 6.24 | 0.58 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 4.551 ms | 9.83 | 0.92 | 2% | ok | 1.1x (13%) |
| factor | qr | f64 | 512x512 | 4.117 ms | 43.46 | 1.02 | 7% | ok | 1.4x (17%) |
| factor | lq | f64 | 512x512 | 16.436 ms | 10.89 | 0.26 | 2% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 157.321 ms | 2.84 | 0.03 | 0% | ok | 1.7x (21%) |
| factor | householder | f64 | 512x512 | 0.427 ms | 2.46 | 9.83 | 12%\* | ok | 3.8x (47%) |
| factor | inv | f64 | 512x512 | 46.157 ms | 5.82 | 0.09 | 1% | ok | 1.0x (12%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 2.798 ms | 31.98 | 1.50 | 5% | ok | 1.4x (17%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.681 ms | 13.31 | 0.62 | 2% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 57.755 ms | 2.61 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 737.767 ms | 0.20 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 17.565 ms | 9.55 | 0.06 | 2% | ok | 0.9x (11%) |
| eigen | schur | f64 | 256x256 | 21.797 ms | 19.24 | 0.05 | 3% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 6.203 ms | 59.50 | 0.17 | 10% | ok | 1.1x (13%) |
| eigen | expm | f64 | 256x256 | 8.645 ms | 29.11 | 0.12 | 5% | ok | 1.3x (16%) |
| eigen | sylvester | f64 | 256x256 | 55.399 ms | 18.17 | 0.03 | 3% | ok | 1.0x (12%) |
| elementwise | exp | f64 | n=8388608 | 39.020 ms | 0.21 | 3.44 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 45.349 ms | 0.18 | 2.96 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 58.949 ms | 0.14 | 2.28 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.696 ms | 0.29 | 4.68 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 201.167 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.111 ms | 23.51 | 244.60 | 300% ! | ok | 6.7x (84%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.907 ms | 2.88 | 30.01 | 37% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.309 ms | 32.00 | 67.28 | 83% | ok | 3.6x (44%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.851 ms | 0.67 | 2.37 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.521 ms | 1.32 | 10.95 | 13%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 14.402 ms | 0.78 | 0.05 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 24.655 ms | 0.45 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.450 ms | 1.32 | 0.09 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.111 ms | 0.56 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.095 ms | 0.51 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 20.369 ms | 1.10 | 0.04 | 0% | ok | 0.8x (10%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 24.437 ms | 0.08 | 0.01 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.325 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.882 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 57.138 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.182 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.900 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.057 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 11.918 ms | 0.94 | 0.06 | 0% | ok | 1.1x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.251 ms | 3.01 | 12.05 | 15%\* | ok | 1.0x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 3.973 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.001 ms | 2.80 | 0.18 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.256 ms | 2.96 | 11.82 | 15%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.048 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 36.434 ms | 0.61 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.599 ms | 1.26 | 5.05 | 6%\* | ok | 1.1x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.396 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.314 ms | 0.58 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.612 ms | 1.23 | 4.94 | 6%\* | ok | 1.0x (13%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 5.905 ms | 1.89 | 0.12 | 0% | ok | 1.0x (12%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 1.836 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 3.579 ms | 3.13 | 0.20 | 1% | ok | 1.0x (12%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 0.329 ms | 2.30 | 9.18 | 11%\* | ok | 1.0x (12%) |
| direct | ref_pardiso_spd | f64 | 96^2 grid, nnz 45696 | 3.213 ms | 3.48 | 0.23 | 1% | ok | 1.4x (18%) |
| direct | ref_pardiso_spd.analyze | f64 | 96^2 grid, nnz 45696 | 1.995 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct | ref_pardiso_spd.factor | f64 | 96^2 grid, nnz 45696 | 0.605 ms | 18.49 | 1.21 | 3% | ok | 3.2x (40%) |
| direct | ref_pardiso_spd.solve | f64 | 96^2 grid, nnz 45696 | 0.501 ms | 1.51 | 6.04 | 7%\* | ok | 0.9x (11%) |
| direct | ref_pardiso_lu | f64 | 96^2 grid, nnz 45696 | 5.529 ms | 4.05 | 0.13 | 1% | ok | 1.3x (17%) |
| direct | ref_pardiso_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.506 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | ref_pardiso_lu.factor | f64 | 96^2 grid, nnz 45696 | 1.243 ms | 17.99 | 0.59 | 3% | ok | 2.6x (33%) |
| direct | ref_pardiso_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.644 ms | 1.17 | 4.70 | 6%\* | ok | 0.9x (11%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.822 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 197.014 ms | 1.57 | 0.00 | 0% | ok | 1.1x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.631 ms | 2.07 | 8.26 | 10%\* | ok | 0.9x (11%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.506 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 57.678 ms | 5.35 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.625 ms | 2.07 | 8.29 | 10%\* | ok | 0.9x (11%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.781 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 538.927 ms | 1.15 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.703 ms | 0.44 | 1.75 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.078 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 440.507 ms | 1.40 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.710 ms | 0.44 | 1.75 | 2%\* | ok | 1.0x (12%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 3.838 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 8.290 ms | 37.23 | 0.10 | 6% | ok | 1.4x (17%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 0.515 ms | 6.54 | 26.18 | 32%\* | ok | 1.0x (12%) |
| direct3d | ref_pardiso_spd.analyze | f64 | 20^3 grid, nnz 53600 | 7.044 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | ref_pardiso_spd.factor | f64 | 20^3 grid, nnz 53600 | 2.379 ms | 129.69 | 0.36 | 22% | ok | 4.1x (51%) |
| direct3d | ref_pardiso_spd.solve | f64 | 20^3 grid, nnz 53600 | 0.954 ms | 3.53 | 14.13 | 17%\* | ok | 0.9x (11%) |
| direct3d | ref_pardiso_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.162 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | ref_pardiso_lu.factor | f64 | 20^3 grid, nnz 53600 | 5.538 ms | 111.45 | 0.15 | 19% | ok | 3.5x (43%) |
| direct3d | ref_pardiso_lu.solve | f64 | 20^3 grid, nnz 53600 | 1.167 ms | 2.89 | 11.55 | 14%\* | ok | 0.9x (11%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.703 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 34.350 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.434 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.365 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.752 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.848 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.195 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.801 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.013 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 17.816 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.290 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 15.832 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 18.618 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.039 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.509 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 11.373 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.057 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 1.001 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.026 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.142 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.969 ms | 2.70 | 30.25 | 37% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.199 ms | 2.18 | 24.46 | 30% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.537 ms | 1.70 | 19.08 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.257 ms | 2.90 | 21.92 | 27% | ok | 1.0x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 30.323 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 31.818 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.340 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.882 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 4.788 ms | 3.50 | 14.02 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 5.121 ms | 52.42 | 0.82 | 9% | ok | 7.6x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.543 ms | 30.91 | 32.84 | 40%\* | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.311 ms | 7.26 | 29.04 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.981 ms | 17.10 | 34.20 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.689 ms | 9.93 | 9.93 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.683 ms | 9.97 | 9.97 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.289 ms | 117.29 | 1.37 | n/a | ok | 7.6x (95%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.743 ms | 97.87 | 0.76 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.354 ms | 198.30 | 1.16 | n/a | ok | 7.6x (94%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.354 ms | 198.33 | 1.16 | n/a | ok | 7.6x (94%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.540 ms | 75.83 | 0.44 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

