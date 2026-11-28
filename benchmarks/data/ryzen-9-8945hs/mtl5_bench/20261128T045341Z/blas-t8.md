## ryzen-9-8945hs -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 554.5 GFLOP/s, f32 1011.9 GFLOP/s |
| ... hardware, AVX-512F | f64 554.5, f32 1011.9 GFLOP/s |
| ... this build, scalar | f64 14.6, f32 14.7 GFLOP/s |
| bandwidth | read 52.5, update 53.1, triad 40.8 GB/s (ceiling 53.1) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 82% | 124% | 0 | 53% |
| blas2 | 10 | 103% | 112% | 0 | 18% |
| blas3 | 10 | 44% | 71% | 0 | 68% |
| factor | 12 | 1% | 6% | 0 | 12% |
| eigen | 8 | 2% | 5% | 0 | 11% |
| elementwise | 5 | 6% | 10% | 0 | 13% |
| sparse | 5 | 30% | 283% | 0 | 15% |
| direct | 19 | 0% | 16% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 11% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 28% | 31% | 0 | 13% |
| mg | 1 | 30% | 30% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 10% | 18% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 95% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.347 ms | 7.15 | 57.19 | 108% | ok | 5.0x (63%) |
| blas1 | dot | f32 | n=8388608 | 1.593 ms | 10.53 | 42.13 | 79% | ok | 7.7x (96%) |
| blas1 | axpy | f64 | n=8388608 | 3.503 ms | 4.79 | 57.48 | 108% | ok | 4.3x (53%) |
| blas1 | axpy | f32 | n=8388608 | 1.926 ms | 8.71 | 52.28 | 98% | ok | 7.7x (97%) |
| blas1 | scale | f64 | n=8388608 | 2.032 ms | 4.13 | 66.05 | 124% | ok | 1.3x (16%) |
| blas1 | two_norm | f64 | n=8388608 | 1.541 ms | 10.89 | 43.54 | 82% | ok | 8.0x (99%) |
| blas1 | one_norm | f64 | n=8388608 | 6.397 ms | 1.31 | 10.49 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.406 ms | 1.31 | 10.48 | 20% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 6.394 ms | 1.31 | 10.50 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 113.531 ms | 1.63 | 1.18 | 2% | ok | 0.9x (11%) |
| blas2 | gemv | f64 | n=4096 | 2.598 ms | 12.92 | 51.69 | 97% | ok | 1.3x (16%) |
| blas2 | gemv | f32 | n=4096 | 1.131 ms | 29.66 | 59.34 | 112% | ok | 1.4x (17%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.588 ms | 12.97 | 51.89 | 98% | ok | 1.5x (19%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.249 ms | 26.87 | 53.77 | 101% | ok | 1.2x (15%) |
| blas2 | symv | f64 | n=4096 | 1.229 ms | 27.29 | 54.64 | 103% | ok | 1.6x (19%) |
| blas2 | trmv | f64 | n=4096 | 1.218 ms | 13.77 | 55.14 | 104% | ok | 1.5x (18%) |
| blas2 | trmv | f32 | n=4096 | 0.579 ms | 29.00 | 58.06 | 109% | ok | 1.6x (20%) |
| blas2 | trsv | f64 | n=4096 | 1.813 ms | 9.25 | 37.05 | 70% | ok | 0.9x (12%) |
| blas2 | trsv | f32 | n=4096 | 0.947 ms | 17.71 | 35.46 | 67% | ok | 1.1x (14%) |
| blas2 | ger | f64 | n=4096 | 4.687 ms | 7.16 | 57.29 | 108% | ok | 1.4x (18%) |
| blas3 | gemm | f64 | 512x512x512 | 0.680 ms | 394.93 | 9.26 | 71% | ok | 6.7x (84%) |
| blas3 | gemm | f32 | 512x512x512 | 0.382 ms | 701.79 | 8.22 | 69% | ok | 6.2x (78%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.644 ms | 81.62 | 3.83 | 15% | ok | 2.1x (26%) |
| blas3 | symm | f64 | 512x512x512 | 0.680 ms | 394.76 | 9.25 | 71% | ok | 7.0x (87%) |
| blas3 | syrk | f64 | 512x512x512 | 2.006 ms | 66.91 | 2.09 | 12% | ok | 1.9x (23%) |
| blas3 | syr2k | f64 | 512x512x512 | 2.725 ms | 98.49 | 2.31 | 18% | ok | 2.2x (28%) |
| blas3 | trmm | f64 | 512x512x512 | 0.451 ms | 297.53 | 11.62 | 54% | ok | 5.2x (65%) |
| blas3 | trsm | f64 | 512x512x512 | 0.620 ms | 216.55 | 8.46 | 39% | ok | 5.6x (71%) |
| blas3 | trsm | f32 | 512x512x512 | 0.301 ms | 446.50 | 8.72 | 44% | ok | 5.5x (68%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.987 ms | 0.48 | 3.87 | 7% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 11.084 ms | 8.07 | 0.38 | 1% | ok | 0.6x (7%) |
| factor | lu | f32 | 512x512 | 5.122 ms | 17.47 | 0.41 | 2% | ok | 0.6x (7%) |
| factor | cholesky | f64 | 512x512 | 6.920 ms | 6.47 | 0.61 | 1% | ok | 0.8x (10%) |
| factor | cholesky | f32 | 512x512 | 2.487 ms | 17.99 | 0.84 | 2% | ok | 0.8x (11%) |
| factor | ldlt | f64 | 512x512 | 14.800 ms | 3.02 | 0.28 | 1% | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 12.986 ms | 3.45 | 0.32 | 1% | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 14.138 ms | 12.66 | 0.30 | 2% | ok | 0.9x (11%) |
| factor | lq | f64 | 512x512 | 50.995 ms | 3.51 | 0.08 | 1% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 573.402 ms | 0.78 | 0.01 | 0% | ok | 1.6x (21%) |
| factor | householder | f64 | 512x512 | 1.244 ms | 0.84 | 3.37 | 6%\* | ok | 4.0x (50%) |
| factor | inv | f64 | 512x512 | 81.484 ms | 3.29 | 0.05 | 1% | ok | 0.9x (12%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 8.761 ms | 10.21 | 0.48 | 2% | ok | 0.8x (10%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 4.281 ms | 5.23 | 0.24 | 1% | ok | 0.5x (6%) |
| eigen | eigen_symmetric | f64 | 256x256 | 160.889 ms | 0.94 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 656.853 ms | 0.23 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 256x256 | 25.581 ms | 6.56 | 0.04 | 1% | ok | 0.8x (10%) |
| eigen | schur | f64 | 256x256 | 31.100 ms | 13.49 | 0.03 | 2% | ok | 0.9x (11%) |
| eigen | svd | f64 | 256x256 | 13.598 ms | 27.14 | 0.08 | 5% | ok | 0.8x (10%) |
| eigen | expm | f64 | 256x256 | 20.964 ms | 12.00 | 0.05 | 2% | ok | 0.9x (12%) |
| eigen | sylvester | f64 | 256x256 | 99.435 ms | 10.12 | 0.02 | 2% | ok | 0.9x (11%) |
| elementwise | exp | f64 | n=8388608 | 33.419 ms | 0.25 | 4.02 | 8% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 52.993 ms | 0.16 | 2.53 | 5% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 47.809 ms | 0.18 | 2.81 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 26.458 ms | 0.32 | 5.07 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 42.467 ms | 0.20 | 3.16 | 6% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.181 ms | 14.48 | 150.61 | 283% ! | ok | 5.1x (64%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.687 ms | 1.55 | 16.15 | 30% | ok | 1.2x (15%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.258 ms | 18.55 | 39.00 | 73% | ok | 3.5x (44%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 22.010 ms | 0.15 | 0.52 | 1%\* | ok | 1.1x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.853 ms | 1.08 | 8.98 | 17%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 24.192 ms | 0.46 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 39.066 ms | 0.29 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 19.551 ms | 0.57 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 66.231 ms | 0.34 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 69.692 ms | 0.32 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 31.200 ms | 0.72 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 42.987 ms | 0.04 | 0.00 | 0% | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.632 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.935 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 75.054 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.237 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.979 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.711 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.805 ms | 0.63 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.353 ms | 2.14 | 8.56 | 16%\* | ok | 1.0x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.162 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.301 ms | 1.53 | 0.10 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.348 ms | 2.18 | 8.70 | 16%\* | ok | 1.1x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.074 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 57.356 ms | 0.39 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.301 ms | 0.58 | 2.33 | 4%\* | ok | 1.0x (12%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.894 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 61.395 ms | 0.36 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.180 ms | 0.64 | 2.56 | 5%\* | ok | 1.1x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 6.966 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 340.608 ms | 0.91 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.220 ms | 1.52 | 6.07 | 11%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.164 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 84.125 ms | 3.67 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.414 ms | 1.40 | 5.58 | 11%\* | ok | 1.0x (13%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.614 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1179.228 ms | 0.52 | 0.00 | 0% | ok | 0.9x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.184 ms | 0.30 | 1.21 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.530 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 625.811 ms | 0.99 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.199 ms | 0.30 | 1.20 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 49.450 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 74.103 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 39.806 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.121 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 27.383 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.325 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 67.953 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 43.374 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.051 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 32.997 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 31.101 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 32.261 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.110 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.055 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 3.026 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 28.276 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 2.229 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 2.176 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.036 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.641 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.809 ms | 1.45 | 16.21 | 31% | ok | 0.9x (11%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.942 ms | 1.35 | 15.10 | 28% | ok | 1.3x (16%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.221 ms | 1.18 | 13.21 | 25% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.531 ms | 2.07 | 15.69 | 30% | ok | 1.1x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.564 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.249 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 69.543 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.747 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 12.259 ms | 1.37 | 5.47 | 10% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.853 ms | 55.31 | 0.86 | 10% | ok | 7.7x (96%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.835 ms | 9.14 | 9.72 | 18% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.041 ms | 4.15 | 16.61 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.245 ms | 5.17 | 10.34 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.122 ms | 7.91 | 7.91 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.121 ms | 7.91 | 7.91 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.292 ms | 207.75 | 2.43 | n/a | ok | 7.9x (98%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.017 ms | 53.50 | 0.42 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.550 ms | 59.00 | 0.35 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.528 ms | 59.28 | 0.35 | n/a | ok | 7.6x (96%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.516 ms | 59.44 | 0.35 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

