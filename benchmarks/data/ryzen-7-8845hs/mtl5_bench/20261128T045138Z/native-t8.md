## ryzen-7-8845hs -- native, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 626.3 GFLOP/s, f32 1255.2 GFLOP/s |
| ... hardware, AVX-512F | f64 626.3, f32 1255.2 GFLOP/s |
| ... this build, scalar | f64 12.3, f32 14.3 GFLOP/s |
| bandwidth | read 53.0, update 53.0, triad 41.2 GB/s (ceiling 53.0) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 95% | 129% | 0 | 42% |
| blas2 | 10 | 75% | 111% | 0 | 13% |
| blas3 | 10 | 2% | 18% | 0 | 12% |
| factor | 12 | 1% | 8% | 0 | 19% |
| eigen | 8 | 0% | 2% | 0 | 12% |
| elementwise | 5 | 8% | 12% | 0 | 12% |
| sparse | 5 | 37% | 249% | 0 | 13% |
| direct | 19 | 0% | 20% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 13% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 35% | 38% | 0 | 13% |
| mg | 1 | 36% | 36% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 11% | 23% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 86% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.289 ms | 7.33 | 58.63 | 111% | ok | 5.0x (63%) |
| blas1 | dot | f32 | n=8388608 | 1.336 ms | 12.56 | 50.24 | 95% | ok | 7.3x (92%) |
| blas1 | axpy | f64 | n=8388608 | 3.525 ms | 4.76 | 57.11 | 108% | ok | 3.4x (42%) |
| blas1 | axpy | f32 | n=8388608 | 1.770 ms | 9.48 | 56.87 | 107% | ok | 6.7x (84%) |
| blas1 | scale | f64 | n=8388608 | 1.966 ms | 4.27 | 68.26 | 129% | ok | 1.2x (15%) |
| blas1 | two_norm | f64 | n=8388608 | 1.442 ms | 11.64 | 46.54 | 88% | ok | 7.3x (92%) |
| blas1 | one_norm | f64 | n=8388608 | 5.062 ms | 1.66 | 13.26 | 25% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.093 ms | 1.65 | 13.18 | 25% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 5.063 ms | 1.66 | 13.26 | 25% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 85.932 ms | 2.15 | 1.56 | 3% | ok | 1.1x (13%) |
| blas2 | gemv | f64 | n=4096 | 2.378 ms | 14.11 | 56.46 | 107% | ok | 1.3x (16%) |
| blas2 | gemv | f32 | n=4096 | 1.161 ms | 28.90 | 57.82 | 109% | ok | 2.0x (25%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.288 ms | 14.66 | 58.68 | 111% | ok | 3.1x (39%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.151 ms | 29.16 | 58.35 | 110% | ok | 1.9x (24%) |
| blas2 | symv | f64 | n=4096 | 2.871 ms | 11.69 | 23.40 | 44% | ok | 1.0x (12%) |
| blas2 | trmv | f64 | n=4096 | 1.698 ms | 9.88 | 39.56 | 75% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=4096 | 1.523 ms | 11.02 | 22.06 | 42% | ok | 1.0x (12%) |
| blas2 | trsv | f64 | n=4096 | 1.704 ms | 9.85 | 39.43 | 74% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.501 ms | 11.17 | 22.37 | 42% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 13.606 ms | 2.47 | 19.73 | 37% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.467 ms | 108.83 | 2.55 | 17% | ok | 6.7x (84%) |
| blas3 | gemm | f32 | 512x512x512 | 1.199 ms | 223.96 | 2.62 | 18% | ok | 6.9x (86%) |
| blas3 | gemmt | f64 | 512x512x512 | 11.511 ms | 11.66 | 0.55 | 2% | ok | 1.0x (12%) |
| blas3 | symm | f64 | 512x512x512 | 17.334 ms | 15.49 | 0.36 | 2% | ok | 1.0x (12%) |
| blas3 | syrk | f64 | 512x512x512 | 13.735 ms | 9.77 | 0.31 | 2% | ok | 1.0x (12%) |
| blas3 | syr2k | f64 | 512x512x512 | 23.265 ms | 11.54 | 0.27 | 2% | ok | 1.0x (12%) |
| blas3 | trmm | f64 | 512x512x512 | 25.580 ms | 5.25 | 0.20 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f64 | 512x512x512 | 26.927 ms | 4.98 | 0.19 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f32 | 512x512x512 | 26.285 ms | 5.11 | 0.10 | 0% | ok | 1.0x (12%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.453 ms | 0.56 | 4.49 | 8% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 21.733 ms | 4.12 | 0.19 | 1% | ok | 1.5x (19%) |
| factor | lu | f32 | 512x512 | 20.490 ms | 4.37 | 0.10 | 0% | ok | 1.5x (19%) |
| factor | cholesky | f64 | 512x512 | 11.364 ms | 3.94 | 0.37 | 1%\* | ok | 1.0x (12%) |
| factor | cholesky | f32 | 512x512 | 10.905 ms | 4.10 | 0.19 | 0%\* | ok | 1.0x (12%) |
| factor | ldlt | f64 | 512x512 | 11.442 ms | 3.91 | 0.37 | 1%\* | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 10.482 ms | 4.27 | 0.40 | 1%\* | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 397.787 ms | 0.45 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 41.206 ms | 4.34 | 0.10 | 1% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 454.120 ms | 0.99 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.006 ms | 1.04 | 4.17 | 8%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 77.559 ms | 3.46 | 0.05 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 23.263 ms | 3.85 | 0.18 | 1% | ok | 1.5x (18%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 40.138 ms | 0.56 | 0.03 | 0% | ok | 0.9x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 132.422 ms | 1.14 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 515.030 ms | 0.29 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 61.470 ms | 2.73 | 0.02 | 0% | ok | 1.0x (12%) |
| eigen | schur | f64 | 256x256 | 114.178 ms | 3.67 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 118.635 ms | 3.11 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | expm | f64 | 256x256 | 18.513 ms | 13.59 | 0.06 | 2% | ok | 1.6x (20%) |
| eigen | sylvester | f64 | 256x256 | 251.992 ms | 3.99 | 0.01 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 27.914 ms | 0.30 | 4.81 | 9% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 42.403 ms | 0.20 | 3.17 | 6% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 39.821 ms | 0.21 | 3.37 | 6% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 21.267 ms | 0.39 | 6.31 | 12% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 31.818 ms | 0.26 | 4.22 | 8% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.207 ms | 12.67 | 131.80 | 249% ! | ok | 4.3x (54%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.380 ms | 1.90 | 19.74 | 37% | ok | 1.0x (12%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.263 ms | 18.51 | 38.91 | 73% | ok | 2.9x (36%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 17.608 ms | 0.19 | 0.65 | 1%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.620 ms | 1.24 | 10.28 | 19%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 19.235 ms | 0.58 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 31.400 ms | 0.36 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 15.666 ms | 0.71 | 0.05 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 54.052 ms | 0.41 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 58.239 ms | 0.38 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 25.671 ms | 0.87 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 38.419 ms | 0.05 | 0.00 | 0%\* | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.730 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.679 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 60.463 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.228 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.256 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.028 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.239 ms | 0.79 | 0.05 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.278 ms | 2.72 | 10.86 | 20%\* | ok | 1.0x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.698 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 6.009 ms | 1.86 | 0.12 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.278 ms | 2.72 | 10.86 | 20%\* | ok | 1.0x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.366 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 50.354 ms | 0.44 | 0.01 | 0% | ok | 0.9x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.155 ms | 0.65 | 2.62 | 5%\* | ok | 1.0x (12%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.834 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 49.100 ms | 0.46 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.129 ms | 0.67 | 2.68 | 5%\* | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.614 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 257.362 ms | 1.20 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.011 ms | 1.68 | 6.70 | 13%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 13.873 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.327 ms | 4.45 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.090 ms | 1.61 | 6.45 | 12%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.168 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 956.072 ms | 0.65 | 0.00 | 0% | ok | 0.9x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.553 ms | 0.32 | 1.28 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.552 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 496.745 ms | 1.24 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.543 ms | 0.32 | 1.28 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 40.397 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 60.780 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 32.112 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.347 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.583 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 18.143 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 61.249 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 38.421 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 20.326 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 28.144 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 26.341 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 27.108 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 36.011 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.063 ms | 0.00 | 0.00 | n/a | ok | 0.7x (9%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.403 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 23.206 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 1.818 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 128^2 grid | 1.746 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.030 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.660 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.456 ms | 1.80 | 20.15 | 38% | ok | 1.1x (14%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.578 ms | 1.66 | 18.59 | 35% | ok | 1.0x (12%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.799 ms | 1.46 | 16.30 | 31% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.235 ms | 2.52 | 19.08 | 36% | ok | 1.1x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.376 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.896 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 52.782 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.955 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 11.431 ms | 1.47 | 5.87 | 11% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.020 ms | 66.78 | 1.04 | 11% | ok | 7.2x (90%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.490 ms | 11.26 | 11.96 | 23% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.275 ms | 5.12 | 20.49 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.887 ms | 5.81 | 11.62 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.402 ms | 11.97 | 11.97 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.402 ms | 11.97 | 11.97 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.170 ms | 229.41 | 2.69 | n/a | ok | 6.9x (86%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.164 ms | 64.47 | 0.50 | n/a | ok | 7.3x (91%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 3.767 ms | 71.27 | 0.42 | n/a | ok | 7.2x (91%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.775 ms | 71.11 | 0.42 | n/a | ok | 7.3x (91%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.788 ms | 70.86 | 0.42 | n/a | ok | 7.2x (90%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

