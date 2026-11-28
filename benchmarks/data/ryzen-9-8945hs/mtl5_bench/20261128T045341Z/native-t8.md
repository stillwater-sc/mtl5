## ryzen-9-8945hs -- native, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 505.8 GFLOP/s, f32 1011.4 GFLOP/s |
| ... hardware, AVX-512F | f64 505.8, f32 1011.4 GFLOP/s |
| ... this build, scalar | f64 12.5, f32 14.4 GFLOP/s |
| bandwidth | read 52.8, update 52.9, triad 40.9 GB/s (ceiling 52.9) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 82% | 128% | 0 | 54% |
| blas2 | 10 | 67% | 109% | 0 | 14% |
| blas3 | 10 | 2% | 21% | 0 | 13% |
| factor | 12 | 1% | 6% | 0 | 19% |
| eigen | 8 | 0% | 2% | 0 | 13% |
| elementwise | 5 | 7% | 10% | 0 | 13% |
| sparse | 5 | 33% | 276% | 0 | 14% |
| direct | 19 | 0% | 17% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 11% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 29% | 34% | 0 | 13% |
| mg | 1 | 29% | 29% | 0 | 15% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 11% | 18% | 0 | 15% |
| integer | 9 | n/a | n/a | 0 | 95% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.296 ms | 7.31 | 58.45 | 111% | ok | 6.3x (79%) |
| blas1 | dot | f32 | n=8388608 | 1.541 ms | 10.89 | 43.56 | 82% | ok | 8.0x (100%) |
| blas1 | axpy | f64 | n=8388608 | 3.499 ms | 4.79 | 57.54 | 109% | ok | 4.3x (54%) |
| blas1 | axpy | f32 | n=8388608 | 1.918 ms | 8.75 | 52.49 | 99% | ok | 7.8x (97%) |
| blas1 | scale | f64 | n=8388608 | 1.982 ms | 4.23 | 67.71 | 128% | ok | 1.3x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 1.676 ms | 10.01 | 40.05 | 76% | ok | 7.9x (99%) |
| blas1 | one_norm | f64 | n=8388608 | 6.417 ms | 1.31 | 10.46 | 20% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.405 ms | 1.31 | 10.48 | 20% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 6.393 ms | 1.31 | 10.50 | 20% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 111.436 ms | 1.66 | 1.20 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 2.401 ms | 13.98 | 55.94 | 106% | ok | 1.4x (17%) |
| blas2 | gemv | f32 | n=4096 | 1.186 ms | 28.30 | 56.62 | 107% | ok | 2.4x (30%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.331 ms | 14.40 | 57.61 | 109% | ok | 3.3x (42%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.162 ms | 28.89 | 57.80 | 109% | ok | 2.1x (27%) |
| blas2 | symv | f64 | n=4096 | 3.367 ms | 9.97 | 19.95 | 38% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 1.894 ms | 8.86 | 35.47 | 67% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=4096 | 1.759 ms | 9.54 | 19.09 | 36% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 1.898 ms | 8.84 | 35.39 | 67% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.791 ms | 9.37 | 18.75 | 35% | ok | 1.1x (14%) |
| blas2 | ger | f64 | n=4096 | 16.194 ms | 2.07 | 16.58 | 31% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 2.515 ms | 106.74 | 2.50 | 21% | ok | 8.0x (99%) |
| blas3 | gemm | f32 | 512x512x512 | 1.249 ms | 214.95 | 2.52 | 21% | ok | 8.0x (100%) |
| blas3 | gemmt | f64 | 512x512x512 | 13.986 ms | 9.60 | 0.45 | 2% | ok | 1.0x (12%) |
| blas3 | symm | f64 | 512x512x512 | 19.803 ms | 13.56 | 0.32 | 3% | ok | 1.0x (13%) |
| blas3 | syrk | f64 | 512x512x512 | 16.845 ms | 7.97 | 0.25 | 2% | ok | 1.0x (12%) |
| blas3 | syr2k | f64 | 512x512x512 | 28.124 ms | 9.54 | 0.22 | 2% | ok | 1.0x (12%) |
| blas3 | trmm | f64 | 512x512x512 | 33.115 ms | 4.05 | 0.16 | 1% | ok | 0.9x (12%) |
| blas3 | trsm | f64 | 512x512x512 | 35.262 ms | 3.81 | 0.15 | 1% | ok | 1.0x (13%) |
| blas3 | trsm | f32 | 512x512x512 | 32.139 ms | 4.18 | 0.08 | 0% | ok | 1.0x (12%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 11.233 ms | 0.47 | 3.78 | 7% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 25.932 ms | 3.45 | 0.16 | 1% | ok | 1.5x (19%) |
| factor | lu | f32 | 512x512 | 24.215 ms | 3.70 | 0.09 | 0% | ok | 1.6x (20%) |
| factor | cholesky | f64 | 512x512 | 14.281 ms | 3.13 | 0.29 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 13.709 ms | 3.26 | 0.15 | 0% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 14.398 ms | 3.11 | 0.29 | 1% | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 13.020 ms | 3.44 | 0.32 | 1% | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 502.958 ms | 0.36 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 50.327 ms | 3.56 | 0.08 | 1% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 568.181 ms | 0.79 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.250 ms | 0.84 | 3.36 | 6%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 96.566 ms | 2.78 | 0.04 | 1% | ok | 1.2x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 27.336 ms | 3.27 | 0.15 | 1% | ok | 1.5x (19%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 48.699 ms | 0.46 | 0.02 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 167.665 ms | 0.90 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 653.946 ms | 0.23 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 256x256 | 75.205 ms | 2.23 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 141.217 ms | 2.97 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 149.388 ms | 2.47 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | expm | f64 | 256x256 | 22.340 ms | 11.26 | 0.05 | 2% | ok | 1.7x (21%) |
| eigen | sylvester | f64 | 256x256 | 324.845 ms | 3.10 | 0.00 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 33.293 ms | 0.25 | 4.03 | 8% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 51.649 ms | 0.16 | 2.60 | 5% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 48.592 ms | 0.17 | 2.76 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 26.397 ms | 0.32 | 5.08 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 38.500 ms | 0.22 | 3.49 | 7% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.186 ms | 14.05 | 146.16 | 276% ! | ok | 4.9x (61%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.543 ms | 1.70 | 17.65 | 33% | ok | 1.1x (14%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.261 ms | 18.52 | 38.94 | 74% | ok | 3.4x (43%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 22.818 ms | 0.14 | 0.50 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.919 ms | 1.05 | 8.68 | 16%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 32.289 ms | 0.35 | 0.02 | 0% | ok | 0.7x (9%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 59.977 ms | 0.19 | 0.01 | 0% | ok | 0.6x (8%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 20.280 ms | 0.55 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 66.656 ms | 0.34 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 71.896 ms | 0.31 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 37.166 ms | 0.60 | 0.02 | 0% | ok | 0.8x (10%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 44.427 ms | 0.04 | 0.00 | 0% | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 5.471 ms | 0.00 | 0.00 | n/a | ok | 0.8x (10%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 7.030 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 76.873 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.271 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 18.264 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.691 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.674 ms | 0.63 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.344 ms | 2.20 | 8.81 | 17%\* | ok | 1.1x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.143 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.268 ms | 1.54 | 0.10 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.344 ms | 2.20 | 8.78 | 17%\* | ok | 1.0x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.099 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 57.981 ms | 0.39 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.255 ms | 0.60 | 2.41 | 5%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.004 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 63.390 ms | 0.35 | 0.01 | 0% | ok | 0.9x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.277 ms | 0.59 | 2.37 | 4%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 6.976 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 342.792 ms | 0.90 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.274 ms | 1.48 | 5.93 | 11%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.192 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 84.877 ms | 3.64 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.307 ms | 1.46 | 5.84 | 11%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.603 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1103.797 ms | 0.56 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.250 ms | 0.30 | 1.20 | 2%\* | ok | 1.0x (13%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.629 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 615.382 ms | 1.00 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.220 ms | 0.30 | 1.20 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 51.143 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 77.483 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 40.621 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.926 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 28.379 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.797 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 74.040 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 44.197 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.599 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 34.372 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.716 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 34.205 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 45.331 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.055 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 3.053 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 28.203 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 2.268 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 128^2 grid | 2.169 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.036 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.800 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.635 ms | 1.60 | 17.94 | 34% | ok | 1.1x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.908 ms | 1.37 | 15.37 | 29% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 3.003 ms | 0.87 | 9.76 | 18% | ok | 0.7x (9%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.706 ms | 2.06 | 15.54 | 29% | ok | 1.2x (15%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.421 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.131 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 65.477 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.935 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 14.381 ms | 1.17 | 4.67 | 9% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.790 ms | 56.05 | 0.88 | 11% | ok | 7.7x (96%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.838 ms | 9.13 | 9.70 | 18% | ok | 1.2x (15%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.023 ms | 4.17 | 16.68 | n/a | ok | 1.5x (19%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.243 ms | 5.17 | 10.35 | n/a | ok | 1.1x (14%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.726 ms | 9.72 | 9.72 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.725 ms | 9.72 | 9.72 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.283 ms | 209.24 | 2.45 | n/a | ok | 7.7x (96%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.995 ms | 53.74 | 0.42 | n/a | ok | 7.7x (96%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.510 ms | 59.51 | 0.35 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.523 ms | 59.34 | 0.35 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.512 ms | 59.50 | 0.35 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

