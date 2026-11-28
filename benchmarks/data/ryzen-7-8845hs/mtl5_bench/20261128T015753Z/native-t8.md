## ryzen-7-8845hs -- native, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 627.5 GFLOP/s, f32 1254.8 GFLOP/s |
| ... hardware, AVX-512F | f64 627.5, f32 1254.8 GFLOP/s |
| ... this build, scalar | f64 12.3, f32 14.3 GFLOP/s |
| bandwidth | read 53.1, update 53.3, triad 41.3 GB/s (ceiling 53.3) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 94% | 128% | 0 | 43% |
| blas2 | 10 | 74% | 110% | 0 | 13% |
| blas3 | 10 | 2% | 18% | 0 | 12% |
| factor | 12 | 1% | 8% | 0 | 18% |
| eigen | 8 | 0% | 2% | 0 | 12% |
| elementwise | 5 | 8% | 12% | 0 | 12% |
| sparse | 5 | 37% | 298% | 0 | 13% |
| direct | 19 | 0% | 21% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 12% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 35% | 36% | 0 | 12% |
| mg | 1 | 36% | 36% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 11% | 22% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 88% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.273 ms | 7.38 | 59.04 | 111% | ok | 5.1x (63%) |
| blas1 | dot | f32 | n=8388608 | 1.336 ms | 12.56 | 50.22 | 94% | ok | 7.4x (92%) |
| blas1 | axpy | f64 | n=8388608 | 3.520 ms | 4.77 | 57.19 | 107% | ok | 3.4x (43%) |
| blas1 | axpy | f32 | n=8388608 | 1.750 ms | 9.59 | 57.54 | 108% | ok | 6.8x (86%) |
| blas1 | scale | f64 | n=8388608 | 1.970 ms | 4.26 | 68.13 | 128% | ok | 1.2x (15%) |
| blas1 | two_norm | f64 | n=8388608 | 1.497 ms | 11.20 | 44.82 | 84% | ok | 7.4x (92%) |
| blas1 | one_norm | f64 | n=8388608 | 5.061 ms | 1.66 | 13.26 | 25% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.100 ms | 1.64 | 13.16 | 25% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 5.062 ms | 1.66 | 13.26 | 25% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 84.509 ms | 2.18 | 1.59 | 3% | ok | 1.1x (13%) |
| blas2 | gemv | f64 | n=4096 | 2.377 ms | 14.11 | 56.48 | 106% | ok | 1.3x (16%) |
| blas2 | gemv | f32 | n=4096 | 1.184 ms | 28.35 | 56.73 | 107% | ok | 1.9x (24%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.285 ms | 14.68 | 58.76 | 110% | ok | 3.1x (39%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.150 ms | 29.18 | 58.39 | 110% | ok | 2.0x (25%) |
| blas2 | symv | f64 | n=4096 | 2.924 ms | 11.48 | 22.98 | 43% | ok | 1.0x (12%) |
| blas2 | trmv | f64 | n=4096 | 1.712 ms | 9.80 | 39.25 | 74% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=4096 | 1.482 ms | 11.32 | 22.66 | 43% | ok | 1.0x (12%) |
| blas2 | trsv | f64 | n=4096 | 1.753 ms | 9.57 | 38.32 | 72% | ok | 1.0x (12%) |
| blas2 | trsv | f32 | n=4096 | 1.520 ms | 11.04 | 22.09 | 41% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 13.675 ms | 2.45 | 19.63 | 37% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.455 ms | 109.34 | 2.56 | 17% | ok | 6.5x (82%) |
| blas3 | gemm | f32 | 512x512x512 | 1.184 ms | 226.78 | 2.66 | 18% | ok | 7.0x (88%) |
| blas3 | gemmt | f64 | 512x512x512 | 11.581 ms | 11.59 | 0.54 | 2% | ok | 1.0x (12%) |
| blas3 | symm | f64 | 512x512x512 | 17.277 ms | 15.54 | 0.36 | 2% | ok | 1.0x (12%) |
| blas3 | syrk | f64 | 512x512x512 | 13.717 ms | 9.78 | 0.31 | 2% | ok | 1.0x (12%) |
| blas3 | syr2k | f64 | 512x512x512 | 23.113 ms | 11.61 | 0.27 | 2% | ok | 1.0x (12%) |
| blas3 | trmm | f64 | 512x512x512 | 25.359 ms | 5.29 | 0.21 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f64 | 512x512x512 | 26.965 ms | 4.98 | 0.19 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f32 | 512x512x512 | 26.296 ms | 5.10 | 0.10 | 0% | ok | 1.0x (12%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.446 ms | 0.56 | 4.50 | 8% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 22.345 ms | 4.00 | 0.19 | 1% | ok | 1.5x (18%) |
| factor | lu | f32 | 512x512 | 21.193 ms | 4.22 | 0.10 | 0% | ok | 1.5x (18%) |
| factor | cholesky | f64 | 512x512 | 11.369 ms | 3.94 | 0.37 | 1%\* | ok | 1.0x (12%) |
| factor | cholesky | f32 | 512x512 | 10.890 ms | 4.11 | 0.19 | 0%\* | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 11.424 ms | 3.92 | 0.37 | 1%\* | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 10.383 ms | 4.31 | 0.40 | 1%\* | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 396.484 ms | 0.45 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 41.870 ms | 4.27 | 0.10 | 1% | ok | 1.6x (19%) |
| factor | hessenberg | f64 | 512x512 | 457.529 ms | 0.98 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.012 ms | 1.04 | 4.14 | 8%\* | ok | 3.8x (48%) |
| factor | inv | f64 | 512x512 | 78.307 ms | 3.43 | 0.05 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 23.827 ms | 3.76 | 0.18 | 1% | ok | 1.4x (18%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 38.797 ms | 0.58 | 0.03 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 129.465 ms | 1.17 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 520.204 ms | 0.29 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 256x256 | 61.755 ms | 2.72 | 0.02 | 0% | ok | 1.0x (12%) |
| eigen | schur | f64 | 256x256 | 116.456 ms | 3.60 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 119.498 ms | 3.09 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | expm | f64 | 256x256 | 18.604 ms | 13.53 | 0.06 | 2% | ok | 1.6x (20%) |
| eigen | sylvester | f64 | 256x256 | 262.427 ms | 3.84 | 0.01 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 28.050 ms | 0.30 | 4.79 | 9% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 42.386 ms | 0.20 | 3.17 | 6% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 40.267 ms | 0.21 | 3.33 | 6% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 21.457 ms | 0.39 | 6.26 | 12% | ok | 1.0x (12%) |
| elementwise | erf | f64 | n=8388608 | 31.690 ms | 0.26 | 4.24 | 8% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.171 ms | 15.26 | 158.78 | 298% ! | ok | 5.0x (63%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.398 ms | 1.87 | 19.48 | 37% | ok | 1.0x (12%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.235 ms | 18.74 | 39.39 | 74% | ok | 2.9x (36%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 18.562 ms | 0.18 | 0.62 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.660 ms | 1.21 | 10.03 | 19%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 19.765 ms | 0.57 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 31.585 ms | 0.35 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 15.435 ms | 0.72 | 0.05 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 54.244 ms | 0.41 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 58.308 ms | 0.38 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 25.623 ms | 0.87 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 38.663 ms | 0.05 | 0.00 | 0%\* | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.739 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.721 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 61.729 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.227 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.309 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.986 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.245 ms | 0.79 | 0.05 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.281 ms | 2.69 | 10.77 | 20%\* | ok | 1.0x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.706 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 6.032 ms | 1.85 | 0.12 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.276 ms | 2.74 | 10.94 | 21%\* | ok | 1.0x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.376 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.518 ms | 0.46 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.137 ms | 0.67 | 2.66 | 5%\* | ok | 1.0x (12%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.852 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 49.140 ms | 0.46 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.102 ms | 0.69 | 2.74 | 5%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.639 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 257.469 ms | 1.20 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.025 ms | 1.66 | 6.66 | 12%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 14.020 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.556 ms | 4.44 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.065 ms | 1.63 | 6.53 | 12%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.206 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 896.055 ms | 0.69 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.576 ms | 0.32 | 1.27 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.609 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 499.133 ms | 1.24 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.535 ms | 0.32 | 1.28 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 40.683 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 61.271 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 32.392 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.356 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.497 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 18.015 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 58.760 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 35.196 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 20.218 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 27.320 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 25.934 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 27.164 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 36.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.044 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.519 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 25.270 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| pc | ic_0 | f64 | 128^2 grid | 1.807 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 128^2 grid | 1.873 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.030 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.659 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.519 ms | 1.72 | 19.31 | 36% | ok | 1.0x (12%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.560 ms | 1.68 | 18.79 | 35% | ok | 1.0x (12%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.183 ms | 1.20 | 13.43 | 25% | ok | 0.8x (10%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.967 ms | 2.57 | 19.42 | 36% | ok | 1.0x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.652 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 33.057 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 53.580 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.018 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 11.437 ms | 1.47 | 5.87 | 11% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.021 ms | 66.75 | 1.04 | 11% | ok | 7.2x (90%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.497 ms | 11.21 | 11.91 | 22% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.669 ms | 4.57 | 18.29 | n/a | ok | 0.9x (11%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.895 ms | 5.80 | 11.59 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.409 ms | 11.91 | 11.91 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.408 ms | 11.91 | 11.91 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.164 ms | 230.67 | 2.70 | n/a | ok | 7.1x (88%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.159 ms | 64.54 | 0.50 | n/a | ok | 7.3x (91%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 3.770 ms | 71.20 | 0.42 | n/a | ok | 7.3x (91%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.778 ms | 71.06 | 0.42 | n/a | ok | 7.2x (90%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.797 ms | 70.69 | 0.41 | n/a | ok | 7.4x (92%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

