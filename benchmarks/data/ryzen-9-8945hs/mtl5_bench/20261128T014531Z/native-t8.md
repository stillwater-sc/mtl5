## ryzen-9-8945hs -- native, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 504.0 GFLOP/s, f32 1010.5 GFLOP/s |
| ... hardware, AVX-512F | f64 504.0, f32 1010.5 GFLOP/s |
| ... this build, scalar | f64 12.5, f32 14.5 GFLOP/s |
| bandwidth | read 52.5, update 53.1, triad 40.8 GB/s (ceiling 53.1) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 82% | 124% | 0 | 53% |
| blas2 | 10 | 61% | 132% | 0 | 13% |
| blas3 | 10 | 2% | 21% | 0 | 13% |
| factor | 12 | 1% | 6% | 0 | 20% |
| eigen | 8 | 0% | 2% | 0 | 13% |
| elementwise | 5 | 7% | 10% | 0 | 13% |
| sparse | 5 | 28% | 103% | 0 | 25% |
| direct | 19 | 0% | 16% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 10% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 35% | 0 | 13% |
| mg | 1 | 29% | 29% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 11% | 18% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 95% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.305 ms | 7.28 | 58.22 | 110% | ok | 6.2x (78%) |
| blas1 | dot | f32 | n=8388608 | 1.545 ms | 10.86 | 43.45 | 82% | ok | 8.1x (101%) |
| blas1 | axpy | f64 | n=8388608 | 3.546 ms | 4.73 | 56.78 | 107% | ok | 4.2x (53%) |
| blas1 | axpy | f32 | n=8388608 | 1.928 ms | 8.70 | 52.21 | 98% | ok | 7.7x (97%) |
| blas1 | scale | f64 | n=8388608 | 2.043 ms | 4.11 | 65.69 | 124% | ok | 1.3x (16%) |
| blas1 | two_norm | f64 | n=8388608 | 1.748 ms | 9.60 | 38.38 | 72% | ok | 7.9x (99%) |
| blas1 | one_norm | f64 | n=8388608 | 6.448 ms | 1.30 | 10.41 | 20% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.461 ms | 1.30 | 10.39 | 20% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 6.444 ms | 1.30 | 10.41 | 20% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 104.095 ms | 1.77 | 1.29 | 2% | ok | 1.1x (13%) |
| blas2 | gemv | f64 | n=4096 | 2.426 ms | 13.83 | 55.35 | 104% | ok | 1.4x (18%) |
| blas2 | gemv | f32 | n=4096 | 0.959 ms | 34.98 | 69.99 | 132% | ok | 3.5x (44%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.342 ms | 14.33 | 57.33 | 108% | ok | 3.3x (42%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.133 ms | 29.61 | 59.25 | 112% | ok | 2.1x (26%) |
| blas2 | symv | f64 | n=4096 | 3.799 ms | 8.83 | 17.68 | 33% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 2.108 ms | 7.96 | 31.87 | 60% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=4096 | 1.893 ms | 8.86 | 17.74 | 33% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 2.071 ms | 8.10 | 32.44 | 61% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.896 ms | 8.85 | 17.71 | 33% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 16.397 ms | 2.05 | 16.38 | 31% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.524 ms | 106.37 | 2.49 | 21% | ok | 7.7x (96%) |
| blas3 | gemm | f32 | 512x512x512 | 1.260 ms | 213.08 | 2.50 | 21% | ok | 7.9x (99%) |
| blas3 | gemmt | f64 | 512x512x512 | 14.203 ms | 9.45 | 0.44 | 2% | ok | 2.8x (35%) |
| blas3 | symm | f64 | 512x512x512 | 20.244 ms | 13.26 | 0.31 | 3% | ok | 1.0x (13%) |
| blas3 | syrk | f64 | 512x512x512 | 17.038 ms | 7.88 | 0.25 | 2% | ok | 1.0x (12%) |
| blas3 | syr2k | f64 | 512x512x512 | 28.918 ms | 9.28 | 0.22 | 2% | ok | 1.0x (12%) |
| blas3 | trmm | f64 | 512x512x512 | 30.836 ms | 4.35 | 0.17 | 1% | ok | 1.0x (13%) |
| blas3 | trsm | f64 | 512x512x512 | 35.426 ms | 3.79 | 0.15 | 1% | ok | 1.0x (13%) |
| blas3 | trsm | f32 | 512x512x512 | 32.233 ms | 4.16 | 0.08 | 0% | ok | 1.0x (12%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.317 ms | 0.51 | 4.12 | 8% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 25.944 ms | 3.45 | 0.16 | 1% | ok | 1.6x (20%) |
| factor | lu | f32 | 512x512 | 24.427 ms | 3.66 | 0.09 | 0% | ok | 1.6x (20%) |
| factor | cholesky | f64 | 512x512 | 14.342 ms | 3.12 | 0.29 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 13.762 ms | 3.25 | 0.15 | 0% | ok | 1.0x (12%) |
| factor | ldlt | f64 | 512x512 | 14.466 ms | 3.09 | 0.29 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 12.478 ms | 3.59 | 0.34 | 1% | ok | 1.1x (14%) |
| factor | qr | f64 | 512x512 | 499.604 ms | 0.36 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 51.137 ms | 3.50 | 0.08 | 1% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 570.661 ms | 0.78 | 0.01 | 0% | ok | 1.7x (21%) |
| factor | householder | f64 | 512x512 | 1.251 ms | 0.84 | 3.35 | 6%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 96.983 ms | 2.77 | 0.04 | 1% | ok | 1.2x (15%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 27.640 ms | 3.24 | 0.15 | 1% | ok | 1.5x (19%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 48.730 ms | 0.46 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 166.325 ms | 0.91 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 622.937 ms | 0.24 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 75.904 ms | 2.21 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | schur | f64 | 256x256 | 154.491 ms | 2.71 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 150.758 ms | 2.45 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 22.382 ms | 11.24 | 0.05 | 2% | ok | 1.7x (21%) |
| eigen | sylvester | f64 | 256x256 | 332.534 ms | 3.03 | 0.00 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 32.784 ms | 0.26 | 4.09 | 8% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 50.838 ms | 0.17 | 2.64 | 5% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 47.996 ms | 0.17 | 2.80 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 25.582 ms | 0.33 | 5.25 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 38.104 ms | 0.22 | 3.52 | 7% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.501 ms | 5.23 | 54.40 | 103% | ok | 2.0x (25%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.866 ms | 1.40 | 14.59 | 28% | ok | 4.1x (51%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.281 ms | 18.36 | 38.60 | 73% | ok | 3.5x (43%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 19.693 ms | 0.17 | 0.58 | 1%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.933 ms | 1.04 | 8.61 | 16%\* | ok | 1.0x (12%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 23.862 ms | 0.47 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 38.906 ms | 0.29 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 19.096 ms | 0.59 | 0.04 | 0% | ok | 1.1x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 66.214 ms | 0.34 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 69.308 ms | 0.32 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 30.576 ms | 0.73 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 43.304 ms | 0.04 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.705 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 7.034 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 75.754 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.242 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.927 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.692 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.461 ms | 0.64 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.348 ms | 2.17 | 8.70 | 16%\* | ok | 1.1x (14%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.181 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 6.987 ms | 1.60 | 0.10 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.371 ms | 2.04 | 8.16 | 15%\* | ok | 1.0x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.067 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 57.715 ms | 0.39 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.391 ms | 0.54 | 2.17 | 4%\* | ok | 1.2x (14%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.943 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.981 ms | 0.37 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.646 ms | 0.46 | 1.84 | 3%\* | ok | 0.6x (8%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 7.009 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 362.872 ms | 0.85 | 0.00 | 0% | ok | 1.1x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.551 ms | 1.32 | 5.28 | 10%\* | ok | 0.9x (11%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.354 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 85.560 ms | 3.61 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.634 ms | 1.28 | 5.12 | 10%\* | ok | 0.8x (10%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.535 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1120.050 ms | 0.55 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.588 ms | 0.29 | 1.16 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.650 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 709.485 ms | 0.87 | 0.00 | 0% | ok | 0.9x (11%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.447 ms | 0.29 | 1.18 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 51.605 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 78.981 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 40.939 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 22.150 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 28.604 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 23.009 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 76.210 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 45.032 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.806 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 34.945 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 33.050 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 34.613 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 45.655 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.054 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.972 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 28.315 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 2.207 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 2.138 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.036 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.803 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.589 ms | 1.65 | 18.46 | 35% | ok | 1.1x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.930 ms | 1.36 | 15.20 | 29% | ok | 1.0x (12%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.252 ms | 1.16 | 13.02 | 25% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.994 ms | 2.02 | 15.30 | 29% | ok | 1.1x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 41.041 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.483 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 70.446 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 10.046 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 14.471 ms | 1.16 | 4.64 | 9% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.808 ms | 55.83 | 0.87 | 11% | ok | 7.7x (96%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.834 ms | 9.15 | 9.72 | 18% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.055 ms | 4.14 | 16.55 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.238 ms | 5.18 | 10.36 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.728 ms | 9.71 | 9.71 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.727 ms | 9.72 | 9.72 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.319 ms | 203.53 | 2.39 | n/a | ok | 7.6x (95%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.033 ms | 53.34 | 0.42 | n/a | ok | 7.6x (96%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.522 ms | 59.37 | 0.35 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.522 ms | 59.36 | 0.35 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.528 ms | 59.28 | 0.35 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

