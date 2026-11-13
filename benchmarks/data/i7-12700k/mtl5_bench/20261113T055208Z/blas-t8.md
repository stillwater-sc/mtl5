## i7-12700k -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `8867b44b3f52` |
| FMA peak (ceiling) | f64 600.9 GFLOP/s, f32 1201.8 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.9, f32 1201.8 GFLOP/s |
| ... this build, scalar | f64 10.0, f32 10.0 GFLOP/s |
| bandwidth | read 82.5, update 77.7, triad 75.9 GB/s (ceiling 82.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 85% | 125% | 0 | 28% |
| blas2 | 10 | 120% | 198% | 0 | 32% |
| blas3 | 10 | 66% | 85% | 0 | 77% |
| factor | 12 | 5% | 13% | 0 | 18% |
| eigen | 8 | 3% | 10% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 292% | 0 | 13% |
| direct | 31 | 1% | 13% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| direct3d | 21 | 10% | 32% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 30% | 37% | 0 | 13% |
| mg | 1 | 25% | 25% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 40% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 90% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.496 ms | 11.22 | 89.73 | 109% | ok | 3.8x (48%) |
| blas1 | dot | f32 | n=8388608 | 0.650 ms | 25.82 | 103.27 | 125% | ok | 6.6x (82%) |
| blas1 | axpy | f64 | n=8388608 | 2.842 ms | 5.90 | 70.83 | 86% | ok | 2.3x (28%) |
| blas1 | axpy | f32 | n=8388608 | 1.438 ms | 11.66 | 69.98 | 85% | ok | 3.4x (43%) |
| blas1 | scale | f64 | n=8388608 | 1.915 ms | 4.38 | 70.08 | 85% | ok | 1.9x (23%) |
| blas1 | two_norm | f64 | n=8388608 | 0.739 ms | 22.70 | 90.79 | 110% | ok | 6.6x (82%) |
| blas1 | one_norm | f64 | n=8388608 | 3.942 ms | 2.13 | 17.03 | 21% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.907 ms | 1.21 | 9.72 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.912 ms | 2.14 | 17.15 | 21% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 75.066 ms | 2.46 | 1.79 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.441 ms | 23.28 | 93.17 | 113% | ok | 2.4x (30%) |
| blas2 | gemv | f32 | n=4096 | 0.547 ms | 61.35 | 122.76 | 149% | ok | 3.0x (38%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.421 ms | 23.61 | 94.50 | 114% | ok | 2.5x (32%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.678 ms | 49.49 | 99.03 | 120% | ok | 2.4x (30%) |
| blas2 | symv | f64 | n=4096 | 0.630 ms | 53.29 | 106.69 | 129% | ok | 3.0x (38%) |
| blas2 | trmv | f64 | n=4096 | 0.601 ms | 27.93 | 111.83 | 136% | ok | 3.1x (39%) |
| blas2 | trmv | f32 | n=4096 | 0.205 ms | 81.72 | 163.60 | 198% ! | ok | 4.6x (57%) |
| blas2 | trsv | f64 | n=4096 | 1.754 ms | 9.57 | 38.30 | 46% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.851 ms | 19.72 | 39.49 | 48% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.637 ms | 4.39 | 35.16 | 43% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 0.537 ms | 499.49 | 11.71 | 83% | ok | 6.9x (86%) |
| blas3 | gemm | f32 | 512x512x512 | 0.263 ms | 1020.04 | 11.95 | 85% | ok | 7.0x (87%) |
| blas3 | gemmt | f64 | 512x512x512 | 0.699 ms | 192.09 | 9.00 | 32% | ok | 3.7x (46%) |
| blas3 | symm | f64 | 512x512x512 | 0.536 ms | 501.22 | 11.75 | 83% | ok | 6.9x (87%) |
| blas3 | syrk | f64 | 512x512x512 | 0.748 ms | 179.51 | 5.61 | 30% | ok | 3.4x (43%) |
| blas3 | syr2k | f64 | 512x512x512 | 1.122 ms | 239.22 | 5.61 | 40% | ok | 4.2x (53%) |
| blas3 | trmm | f64 | 512x512x512 | 0.341 ms | 394.11 | 15.39 | 66% | ok | 6.2x (77%) |
| blas3 | trsm | f64 | 512x512x512 | 0.340 ms | 394.50 | 15.41 | 66% | ok | 6.3x (78%) |
| blas3 | trsm | f32 | 512x512x512 | 0.159 ms | 843.44 | 16.47 | 70% | ok | 6.0x (76%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.856 ms | 0.41 | 3.30 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 2.101 ms | 42.59 | 2.00 | 7% | ok | 1.6x (20%) |
| factor | lu | f32 | 512x512 | 1.313 ms | 68.17 | 1.60 | 6% | ok | 1.4x (18%) |
| factor | cholesky | f64 | 512x512 | 1.515 ms | 29.54 | 2.77 | 5% | ok | 1.3x (17%) |
| factor | cholesky | f32 | 512x512 | 0.961 ms | 46.54 | 2.18 | 4% | ok | 1.2x (15%) |
| factor | ldlt | f64 | 512x512 | 7.213 ms | 6.20 | 0.58 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 44.642 ms | 1.00 | 0.09 | 0% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 4.067 ms | 44.01 | 1.03 | 7% | ok | 1.4x (18%) |
| factor | lq | f64 | 512x512 | 17.382 ms | 10.30 | 0.24 | 2% | ok | 1.5x (18%) |
| factor | hessenberg | f64 | 512x512 | 166.121 ms | 2.69 | 0.03 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 0.406 ms | 2.59 | 10.33 | 13%\* | ok | 4.0x (50%) |
| factor | inv | f64 | 512x512 | 46.123 ms | 5.82 | 0.09 | 1% | ok | 1.0x (12%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 2.796 ms | 32.01 | 1.50 | 5% | ok | 1.4x (17%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.639 ms | 13.65 | 0.64 | 2% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 58.238 ms | 2.59 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 729.505 ms | 0.21 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 17.438 ms | 9.62 | 0.06 | 2% | ok | 0.9x (12%) |
| eigen | schur | f64 | 256x256 | 21.677 ms | 19.35 | 0.05 | 3% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 6.143 ms | 60.09 | 0.17 | 10% | ok | 1.1x (14%) |
| eigen | expm | f64 | 256x256 | 8.724 ms | 28.85 | 0.12 | 5% | ok | 1.3x (16%) |
| eigen | sylvester | f64 | 256x256 | 55.135 ms | 18.26 | 0.03 | 3% | ok | 1.0x (12%) |
| elementwise | exp | f64 | n=8388608 | 38.553 ms | 0.22 | 3.48 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 44.724 ms | 0.19 | 3.00 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 58.872 ms | 0.14 | 2.28 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.404 ms | 0.30 | 4.73 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 199.780 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.113 ms | 23.15 | 240.84 | 292% ! | ok | 6.5x (82%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.911 ms | 2.87 | 29.88 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.319 ms | 31.75 | 66.75 | 81% | ok | 3.6x (45%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.803 ms | 0.68 | 2.39 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.369 ms | 1.47 | 12.16 | 15%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 89.216 ms | 0.13 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.839 ms | 0.43 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.921 ms | 1.25 | 0.08 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.172 ms | 0.57 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 53.092 ms | 0.42 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.720 ms | 1.34 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 85.929 ms | 0.02 | 0.00 | 0% | ok | 1.2x (15%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.268 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.877 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.216 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.839 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.287 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 86.587 ms | 0.13 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.276 ms | 2.74 | 10.95 | 13%\* | ok | 0.9x (11%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.385 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.078 ms | 2.74 | 0.18 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.275 ms | 2.75 | 11.01 | 13%\* | ok | 0.9x (11%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.025 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 34.987 ms | 0.64 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.609 ms | 1.24 | 4.97 | 6%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.788 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.068 ms | 0.48 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.594 ms | 1.27 | 5.10 | 6%\* | ok | 1.1x (13%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 5.991 ms | 1.87 | 0.12 | 0% | ok | 0.9x (12%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 1.824 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 3.670 ms | 3.05 | 0.20 | 1% | ok | 0.9x (12%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 0.328 ms | 2.31 | 9.23 | 11%\* | ok | 1.0x (12%) |
| direct | ref_pardiso_spd | f64 | 96^2 grid, nnz 45696 | 3.210 ms | 3.48 | 0.23 | 1% | ok | 1.4x (18%) |
| direct | ref_pardiso_spd.analyze | f64 | 96^2 grid, nnz 45696 | 1.992 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct | ref_pardiso_spd.factor | f64 | 96^2 grid, nnz 45696 | 0.618 ms | 18.09 | 1.18 | 3% | ok | 3.1x (39%) |
| direct | ref_pardiso_spd.solve | f64 | 96^2 grid, nnz 45696 | 0.500 ms | 1.51 | 6.05 | 7%\* | ok | 0.9x (11%) |
| direct | ref_pardiso_lu | f64 | 96^2 grid, nnz 45696 | 5.608 ms | 3.99 | 0.13 | 1% | ok | 1.3x (17%) |
| direct | ref_pardiso_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.512 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | ref_pardiso_lu.factor | f64 | 96^2 grid, nnz 45696 | 1.190 ms | 18.80 | 0.61 | 3% | ok | 2.7x (34%) |
| direct | ref_pardiso_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.629 ms | 1.20 | 4.81 | 6%\* | ok | 0.9x (11%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 9.278 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 7286.650 ms | 0.04 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.549 ms | 2.18 | 8.70 | 11%\* | ok | 0.9x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 18.241 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 66.197 ms | 4.66 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.569 ms | 2.15 | 8.59 | 10%\* | ok | 0.9x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.756 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 541.169 ms | 1.14 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 8.057 ms | 0.42 | 1.67 | 2%\* | ok | 0.9x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.373 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 644.934 ms | 0.96 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.791 ms | 0.43 | 1.73 | 2%\* | ok | 1.0x (12%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 3.804 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 8.272 ms | 37.31 | 0.10 | 6% | ok | 1.4x (17%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 0.510 ms | 6.61 | 26.44 | 32%\* | ok | 1.0x (12%) |
| direct3d | ref_pardiso_spd.analyze | f64 | 20^3 grid, nnz 53600 | 7.117 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | ref_pardiso_spd.factor | f64 | 20^3 grid, nnz 53600 | 2.347 ms | 131.48 | 0.37 | 22% | ok | 4.1x (52%) |
| direct3d | ref_pardiso_spd.solve | f64 | 20^3 grid, nnz 53600 | 0.943 ms | 3.57 | 14.29 | 17%\* | ok | 0.9x (12%) |
| direct3d | ref_pardiso_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.140 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | ref_pardiso_lu.factor | f64 | 20^3 grid, nnz 53600 | 5.529 ms | 111.63 | 0.16 | 19% | ok | 3.5x (43%) |
| direct3d | ref_pardiso_lu.solve | f64 | 20^3 grid, nnz 53600 | 1.154 ms | 2.92 | 11.68 | 14%\* | ok | 0.9x (11%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.471 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 32.130 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.355 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.217 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.618 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.706 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.123 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.653 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.926 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 16.787 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.054 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.454 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 20.686 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.035 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.527 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | ilut | f64 | 128^2 grid | 12.003 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.047 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.981 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.020 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.155 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.965 ms | 2.71 | 30.39 | 37% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.198 ms | 2.19 | 24.49 | 30% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.534 ms | 1.71 | 19.11 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.863 ms | 2.77 | 20.97 | 25% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.661 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.086 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.766 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.926 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 4.796 ms | 3.50 | 13.99 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 5.187 ms | 51.75 | 0.81 | 9% | ok | 7.5x (94%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.543 ms | 30.91 | 32.84 | 40%\* | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.267 ms | 7.40 | 29.60 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.981 ms | 17.10 | 34.20 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.683 ms | 9.97 | 9.97 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.683 ms | 9.97 | 9.97 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.410 ms | 111.37 | 1.31 | n/a | ok | 7.2x (90%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.827 ms | 94.96 | 0.74 | n/a | ok | 7.4x (92%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.357 ms | 197.86 | 1.16 | n/a | ok | 7.5x (94%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.356 ms | 197.93 | 1.16 | n/a | ok | 7.5x (94%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.547 ms | 75.67 | 0.44 | n/a | ok | 7.6x (94%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

