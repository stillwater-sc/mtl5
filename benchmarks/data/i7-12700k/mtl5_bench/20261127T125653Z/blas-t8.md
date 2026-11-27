## i7-12700k -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `f98777e6c6f7` |
| FMA peak (ceiling) | f64 600.5 GFLOP/s, f32 1201.0 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.5, f32 1201.0 GFLOP/s |
| ... this build, scalar | f64 10.0, f32 9.9 GFLOP/s |
| bandwidth | read 81.9, update 77.4, triad 75.5 GB/s (ceiling 81.9) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 91% | 126% | 0 | 29% |
| blas2 | 10 | 119% | 198% | 0 | 32% |
| blas3 | 10 | 66% | 84% | 0 | 77% |
| factor | 12 | 5% | 12% | 0 | 18% |
| eigen | 8 | 3% | 10% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 295% | 0 | 13% |
| direct | 31 | 1% | 15% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 21 | 10% | 32% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 30% | 37% | 0 | 13% |
| mg | 1 | 27% | 27% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 40% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 92% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.397 ms | 12.01 | 96.07 | 117% | ok | 4.1x (51%) |
| blas1 | dot | f32 | n=8388608 | 0.652 ms | 25.73 | 102.94 | 126% | ok | 6.6x (82%) |
| blas1 | axpy | f64 | n=8388608 | 2.743 ms | 6.12 | 73.39 | 90% | ok | 2.3x (29%) |
| blas1 | axpy | f32 | n=8388608 | 1.354 ms | 12.39 | 74.36 | 91% | ok | 3.5x (43%) |
| blas1 | scale | f64 | n=8388608 | 1.776 ms | 4.72 | 75.59 | 92% | ok | 2.0x (25%) |
| blas1 | two_norm | f64 | n=8388608 | 0.739 ms | 22.69 | 90.75 | 111% | ok | 6.7x (84%) |
| blas1 | one_norm | f64 | n=8388608 | 3.970 ms | 2.11 | 16.91 | 21% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.943 ms | 1.21 | 9.67 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.939 ms | 2.13 | 17.04 | 21% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 75.709 ms | 2.44 | 1.77 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.446 ms | 23.21 | 92.88 | 113% | ok | 2.4x (30%) |
| blas2 | gemv | f32 | n=4096 | 0.569 ms | 59.00 | 118.05 | 144% | ok | 2.8x (35%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.428 ms | 23.50 | 94.06 | 115% | ok | 2.5x (32%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.687 ms | 48.87 | 97.78 | 119% | ok | 2.4x (30%) |
| blas2 | symv | f64 | n=4096 | 0.619 ms | 54.18 | 108.47 | 132% | ok | 3.1x (39%) |
| blas2 | trmv | f64 | n=4096 | 0.600 ms | 27.94 | 111.87 | 137% | ok | 3.1x (39%) |
| blas2 | trmv | f32 | n=4096 | 0.207 ms | 81.17 | 162.50 | 198% ! | ok | 4.6x (57%) |
| blas2 | trsv | f64 | n=4096 | 1.757 ms | 9.55 | 38.23 | 47% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.853 ms | 19.67 | 39.37 | 48% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 3.637 ms | 9.23 | 73.83 | 90% | ok | 1.9x (24%) |
| blas3 | gemm | f64 | 512x512x512 | 0.544 ms | 493.49 | 11.57 | 82% | ok | 6.8x (85%) |
| blas3 | gemm | f32 | 512x512x512 | 0.266 ms | 1009.32 | 11.83 | 84% | ok | 6.9x (86%) |
| blas3 | gemmt | f64 | 512x512x512 | 0.700 ms | 191.61 | 8.98 | 32% | ok | 3.7x (46%) |
| blas3 | symm | f64 | 512x512x512 | 0.549 ms | 488.63 | 11.45 | 81% | ok | 6.8x (84%) |
| blas3 | syrk | f64 | 512x512x512 | 0.751 ms | 178.71 | 5.58 | 30% | ok | 3.4x (42%) |
| blas3 | syr2k | f64 | 512x512x512 | 1.128 ms | 237.92 | 5.58 | 40% | ok | 4.2x (52%) |
| blas3 | trmm | f64 | 512x512x512 | 0.341 ms | 393.48 | 15.37 | 66% | ok | 6.2x (77%) |
| blas3 | trsm | f64 | 512x512x512 | 0.338 ms | 396.58 | 15.49 | 66% | ok | 6.3x (79%) |
| blas3 | trsm | f32 | 512x512x512 | 0.159 ms | 842.50 | 16.46 | 70% | ok | 6.0x (75%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.959 ms | 0.41 | 3.28 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 2.085 ms | 42.92 | 2.01 | 7% | ok | 1.6x (20%) |
| factor | lu | f32 | 512x512 | 1.298 ms | 68.91 | 1.62 | 6% | ok | 1.4x (18%) |
| factor | cholesky | f64 | 512x512 | 1.526 ms | 29.31 | 2.75 | 5% | ok | 1.3x (17%) |
| factor | cholesky | f32 | 512x512 | 0.994 ms | 45.01 | 2.11 | 4% | ok | 1.2x (15%) |
| factor | ldlt | f64 | 512x512 | 7.220 ms | 6.20 | 0.58 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 4.758 ms | 9.40 | 0.88 | 2% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 4.143 ms | 43.20 | 1.01 | 7% | ok | 1.4x (18%) |
| factor | lq | f64 | 512x512 | 16.541 ms | 10.82 | 0.25 | 2% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 155.944 ms | 2.87 | 0.03 | 0% | ok | 1.7x (22%) |
| factor | householder | f64 | 512x512 | 0.411 ms | 2.55 | 10.20 | 12%\* | ok | 4.0x (49%) |
| factor | inv | f64 | 512x512 | 46.313 ms | 5.80 | 0.09 | 1% | ok | 1.0x (12%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 2.780 ms | 32.18 | 1.51 | 5% | ok | 1.4x (17%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.680 ms | 13.31 | 0.62 | 2% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 57.327 ms | 2.63 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 735.440 ms | 0.21 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 17.426 ms | 9.63 | 0.06 | 2% | ok | 0.9x (12%) |
| eigen | schur | f64 | 256x256 | 21.751 ms | 19.28 | 0.05 | 3% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 6.186 ms | 59.66 | 0.17 | 10% | ok | 1.1x (14%) |
| eigen | expm | f64 | 256x256 | 8.688 ms | 28.97 | 0.12 | 5% | ok | 1.3x (16%) |
| eigen | sylvester | f64 | 256x256 | 55.425 ms | 18.16 | 0.03 | 3% | ok | 1.0x (12%) |
| elementwise | exp | f64 | n=8388608 | 39.078 ms | 0.21 | 3.43 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 45.629 ms | 0.18 | 2.94 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 59.419 ms | 0.14 | 2.26 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.634 ms | 0.29 | 4.69 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 200.465 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.113 ms | 23.23 | 241.70 | 295% ! | ok | 6.6x (83%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.914 ms | 2.87 | 29.81 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.312 ms | 31.92 | 67.12 | 82% | ok | 3.5x (44%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.803 ms | 0.68 | 2.39 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.538 ms | 1.30 | 10.82 | 13%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 14.873 ms | 0.75 | 0.05 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 24.688 ms | 0.45 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.449 ms | 1.32 | 0.09 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.996 ms | 0.56 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 44.608 ms | 0.50 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 20.197 ms | 1.11 | 0.04 | 0% | ok | 0.8x (10%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 24.592 ms | 0.08 | 0.01 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.289 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.871 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 57.327 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.183 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.819 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.049 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 11.817 ms | 0.95 | 0.06 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.254 ms | 2.98 | 11.92 | 15%\* | ok | 1.0x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 3.982 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.033 ms | 2.77 | 0.18 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.255 ms | 2.97 | 11.87 | 15%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.050 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 36.103 ms | 0.62 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.605 ms | 1.25 | 5.00 | 6%\* | ok | 1.1x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.374 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 38.631 ms | 0.58 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.608 ms | 1.24 | 4.97 | 6%\* | ok | 1.0x (13%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 5.978 ms | 1.87 | 0.12 | 0% | ok | 1.0x (12%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 1.829 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 3.642 ms | 3.07 | 0.20 | 1% | ok | 1.0x (12%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 0.331 ms | 2.29 | 9.14 | 11%\* | ok | 1.0x (12%) |
| direct | ref_pardiso_spd | f64 | 96^2 grid, nnz 45696 | 3.203 ms | 3.49 | 0.23 | 1% | ok | 1.4x (18%) |
| direct | ref_pardiso_spd.analyze | f64 | 96^2 grid, nnz 45696 | 1.994 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct | ref_pardiso_spd.factor | f64 | 96^2 grid, nnz 45696 | 0.634 ms | 17.65 | 1.15 | 3% | ok | 3.0x (38%) |
| direct | ref_pardiso_spd.solve | f64 | 96^2 grid, nnz 45696 | 0.511 ms | 1.48 | 5.92 | 7%\* | ok | 0.9x (11%) |
| direct | ref_pardiso_lu | f64 | 96^2 grid, nnz 45696 | 5.596 ms | 4.00 | 0.13 | 1% | ok | 1.3x (17%) |
| direct | ref_pardiso_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.537 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | ref_pardiso_lu.factor | f64 | 96^2 grid, nnz 45696 | 1.240 ms | 18.04 | 0.59 | 3% | ok | 2.6x (33%) |
| direct | ref_pardiso_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.648 ms | 1.17 | 4.67 | 6%\* | ok | 0.9x (11%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 4.911 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 194.038 ms | 1.59 | 0.00 | 0% | ok | 1.1x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.631 ms | 2.07 | 8.26 | 10%\* | ok | 0.9x (11%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 9.374 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 57.597 ms | 5.36 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.652 ms | 2.04 | 8.16 | 10%\* | ok | 0.9x (11%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.771 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 543.921 ms | 1.13 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.771 ms | 0.43 | 1.73 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.091 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 456.374 ms | 1.35 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.727 ms | 0.44 | 1.74 | 2%\* | ok | 1.0x (12%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 3.831 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 8.481 ms | 36.39 | 0.10 | 6% | ok | 1.3x (17%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 0.511 ms | 6.59 | 26.36 | 32%\* | ok | 1.0x (12%) |
| direct3d | ref_pardiso_spd.analyze | f64 | 20^3 grid, nnz 53600 | 7.062 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | ref_pardiso_spd.factor | f64 | 20^3 grid, nnz 53600 | 2.392 ms | 129.00 | 0.36 | 21% | ok | 4.1x (51%) |
| direct3d | ref_pardiso_spd.solve | f64 | 20^3 grid, nnz 53600 | 0.953 ms | 3.53 | 14.14 | 17%\* | ok | 0.9x (11%) |
| direct3d | ref_pardiso_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.132 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | ref_pardiso_lu.factor | f64 | 20^3 grid, nnz 53600 | 5.562 ms | 110.96 | 0.15 | 18% | ok | 3.4x (43%) |
| direct3d | ref_pardiso_lu.solve | f64 | 20^3 grid, nnz 53600 | 1.163 ms | 2.90 | 11.59 | 14%\* | ok | 0.9x (11%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.535 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 46.387 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.356 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.255 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.732 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.761 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.068 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.665 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.033 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 17.709 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.454 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 15.829 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 18.697 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.041 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.506 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 11.362 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.058 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.995 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.022 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.089 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.974 ms | 2.69 | 30.12 | 37% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.200 ms | 2.18 | 24.44 | 30% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.536 ms | 1.70 | 19.09 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.048 ms | 2.95 | 22.28 | 27% | ok | 1.0x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.616 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 31.021 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 32.976 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.861 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 4.810 ms | 3.49 | 13.95 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 5.126 ms | 52.37 | 0.82 | 9% | ok | 7.6x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.543 ms | 30.88 | 32.81 | 40%\* | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.279 ms | 7.36 | 29.44 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.978 ms | 17.16 | 34.31 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.685 ms | 9.96 | 9.96 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.683 ms | 9.97 | 9.97 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.379 ms | 112.83 | 1.32 | n/a | ok | 7.4x (92%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.747 ms | 97.73 | 0.76 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.354 ms | 198.27 | 1.16 | n/a | ok | 7.6x (94%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.354 ms | 198.25 | 1.16 | n/a | ok | 7.6x (94%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.540 ms | 75.82 | 0.44 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

