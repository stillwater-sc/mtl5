## ryzen-7-8845hs -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 625.8 GFLOP/s, f32 1249.7 GFLOP/s |
| ... hardware, AVX-512F | f64 625.8, f32 1249.7 GFLOP/s |
| ... this build, scalar | f64 14.4, f32 14.4 GFLOP/s |
| bandwidth | read 53.1, update 53.6, triad 41.4 GB/s (ceiling 53.6) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 93% | 125% | 0 | 43% |
| blas2 | 10 | 102% | 111% | 0 | 16% |
| blas3 | 10 | 41% | 71% | 0 | 64% |
| factor | 12 | 2% | 8% | 0 | 11% |
| eigen | 8 | 2% | 5% | 0 | 11% |
| elementwise | 5 | 7% | 12% | 0 | 12% |
| sparse | 5 | 32% | 116% | 0 | 13% |
| direct | 19 | 0% | 20% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 13% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 34% | 35% | 0 | 14% |
| mg | 1 | 35% | 35% | 0 | 15% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 13% | 22% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 87% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.243 ms | 7.48 | 59.84 | 112% | ok | 4.1x (51%) |
| blas1 | dot | f32 | n=8388608 | 1.345 ms | 12.48 | 49.90 | 93% | ok | 7.3x (92%) |
| blas1 | axpy | f64 | n=8388608 | 3.513 ms | 4.78 | 57.31 | 107% | ok | 3.4x (43%) |
| blas1 | axpy | f32 | n=8388608 | 1.738 ms | 9.65 | 57.92 | 108% | ok | 6.9x (86%) |
| blas1 | scale | f64 | n=8388608 | 1.997 ms | 4.20 | 67.20 | 125% | ok | 1.1x (14%) |
| blas1 | two_norm | f64 | n=8388608 | 1.348 ms | 12.44 | 49.78 | 93% | ok | 7.3x (91%) |
| blas1 | one_norm | f64 | n=8388608 | 5.090 ms | 1.65 | 13.18 | 25% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.099 ms | 1.65 | 13.16 | 25% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 5.079 ms | 1.65 | 13.21 | 25% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 89.991 ms | 2.05 | 1.49 | 3% | ok | 0.9x (12%) |
| blas2 | gemv | f64 | n=4096 | 2.569 ms | 13.06 | 52.28 | 98% | ok | 1.3x (16%) |
| blas2 | gemv | f32 | n=4096 | 1.159 ms | 28.94 | 57.91 | 108% | ok | 1.2x (15%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.686 ms | 12.49 | 49.99 | 93% | ok | 1.3x (16%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.226 ms | 27.38 | 54.78 | 102% | ok | 1.2x (15%) |
| blas2 | symv | f64 | n=4096 | 1.238 ms | 27.11 | 54.26 | 101% | ok | 1.5x (18%) |
| blas2 | trmv | f64 | n=4096 | 1.201 ms | 13.97 | 55.93 | 104% | ok | 1.3x (17%) |
| blas2 | trmv | f32 | n=4096 | 0.563 ms | 29.79 | 59.64 | 111% | ok | 1.6x (20%) |
| blas2 | trsv | f64 | n=4096 | 1.678 ms | 10.00 | 40.03 | 75% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.874 ms | 19.19 | 38.42 | 72% | ok | 1.0x (12%) |
| blas2 | ger | f64 | n=4096 | 4.662 ms | 7.20 | 57.59 | 107% | ok | 1.4x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 0.607 ms | 442.16 | 10.36 | 71% | ok | 6.0x (75%) |
| blas3 | gemm | f32 | 512x512x512 | 0.338 ms | 793.48 | 9.30 | 63% | ok | 5.6x (70%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.367 ms | 98.21 | 4.60 | 16% | ok | 2.0x (25%) |
| blas3 | symm | f64 | 512x512x512 | 0.611 ms | 439.70 | 10.31 | 70% | ok | 6.2x (78%) |
| blas3 | syrk | f64 | 512x512x512 | 1.723 ms | 77.91 | 2.43 | 12% | ok | 1.7x (21%) |
| blas3 | syr2k | f64 | 512x512x512 | 2.395 ms | 112.10 | 2.63 | 18% | ok | 2.0x (25%) |
| blas3 | trmm | f64 | 512x512x512 | 0.400 ms | 335.96 | 13.12 | 54% | ok | 4.7x (59%) |
| blas3 | trsm | f64 | 512x512x512 | 0.544 ms | 246.77 | 9.64 | 39% | ok | 5.2x (65%) |
| blas3 | trsm | f32 | 512x512x512 | 0.263 ms | 510.33 | 9.97 | 41% | ok | 5.1x (64%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.553 ms | 0.56 | 4.45 | 8% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 9.180 ms | 9.75 | 0.46 | 2% | ok | 0.5x (7%) |
| factor | lu | f32 | 512x512 | 4.384 ms | 20.41 | 0.48 | 2% | ok | 0.6x (7%) |
| factor | cholesky | f64 | 512x512 | 6.019 ms | 7.43 | 0.70 | 1%\* | ok | 0.7x (9%) |
| factor | cholesky | f32 | 512x512 | 2.305 ms | 19.41 | 0.91 | 2%\* | ok | 0.7x (9%) |
| factor | ldlt | f64 | 512x512 | 11.832 ms | 3.78 | 0.35 | 1%\* | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 10.591 ms | 4.22 | 0.40 | 1%\* | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 11.835 ms | 15.12 | 0.35 | 2% | ok | 0.9x (11%) |
| factor | lq | f64 | 512x512 | 41.901 ms | 4.27 | 0.10 | 1% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 456.350 ms | 0.98 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.016 ms | 1.03 | 4.13 | 8%\* | ok | 3.8x (48%) |
| factor | inv | f64 | 512x512 | 66.375 ms | 4.04 | 0.06 | 1% | ok | 0.9x (11%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.289 ms | 12.28 | 0.58 | 2% | ok | 0.8x (9%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 3.612 ms | 6.19 | 0.29 | 1% | ok | 0.5x (6%) |
| eigen | eigen_symmetric | f64 | 256x256 | 131.576 ms | 1.15 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 507.990 ms | 0.30 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 21.027 ms | 7.98 | 0.05 | 1% | ok | 0.8x (10%) |
| eigen | schur | f64 | 256x256 | 26.607 ms | 15.76 | 0.04 | 3% | ok | 0.8x (10%) |
| eigen | svd | f64 | 256x256 | 12.055 ms | 30.62 | 0.09 | 5% | ok | 0.7x (9%) |
| eigen | expm | f64 | 256x256 | 17.627 ms | 14.28 | 0.06 | 2% | ok | 0.9x (11%) |
| eigen | sylvester | f64 | 256x256 | 81.857 ms | 12.30 | 0.02 | 2% | ok | 0.9x (11%) |
| elementwise | exp | f64 | n=8388608 | 28.449 ms | 0.29 | 4.72 | 9% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 43.808 ms | 0.19 | 3.06 | 6% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 39.507 ms | 0.21 | 3.40 | 6% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 21.545 ms | 0.39 | 6.23 | 12% | ok | 1.0x (12%) |
| elementwise | erf | f64 | n=8388608 | 35.303 ms | 0.24 | 3.80 | 7% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.439 ms | 5.96 | 62.00 | 116% | ok | 1.9x (24%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.612 ms | 1.62 | 16.89 | 32% | ok | 0.9x (11%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.248 ms | 18.63 | 39.17 | 73% | ok | 2.9x (36%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 18.538 ms | 0.18 | 0.62 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.607 ms | 1.25 | 10.36 | 19%\* | ok | 1.1x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 19.952 ms | 0.56 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 31.758 ms | 0.35 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 15.961 ms | 0.70 | 0.05 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 55.333 ms | 0.40 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 58.235 ms | 0.38 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 27.787 ms | 0.81 | 0.03 | 0% | ok | 0.9x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 38.532 ms | 0.05 | 0.00 | 0%\* | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.759 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.703 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 61.220 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.233 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.578 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.979 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.505 ms | 0.77 | 0.05 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.282 ms | 2.68 | 10.71 | 20%\* | ok | 1.0x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.709 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 6.049 ms | 1.85 | 0.12 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.286 ms | 2.65 | 10.59 | 20%\* | ok | 1.0x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.356 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.525 ms | 0.46 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.043 ms | 0.73 | 2.90 | 5%\* | ok | 1.1x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.853 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 50.193 ms | 0.45 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.093 ms | 0.69 | 2.77 | 5%\* | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.651 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 260.230 ms | 1.19 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.009 ms | 1.68 | 6.71 | 13%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 14.403 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 70.628 ms | 4.37 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.045 ms | 1.65 | 6.59 | 12%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.235 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 902.118 ms | 0.68 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.520 ms | 0.32 | 1.28 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.593 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 510.662 ms | 1.21 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.568 ms | 0.32 | 1.28 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 38.988 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 57.954 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 31.584 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 16.644 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 21.651 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.652 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 53.697 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.374 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 19.768 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 26.052 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 24.549 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 25.374 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 33.087 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.043 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.401 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 23.177 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 1.838 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 128^2 grid | 1.779 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.029 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.357 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.598 ms | 1.64 | 18.36 | 34% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.572 ms | 1.66 | 18.65 | 35% | ok | 1.1x (14%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.083 ms | 1.26 | 14.08 | 26% | ok | 1.4x (17%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.310 ms | 2.51 | 18.99 | 35% | ok | 1.2x (15%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.702 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 33.154 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 55.775 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.824 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 9.878 ms | 1.70 | 6.79 | 13% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.072 ms | 65.92 | 1.03 | 11% | ok | 7.2x (91%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.502 ms | 11.17 | 11.86 | 22% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.267 ms | 5.14 | 20.54 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.871 ms | 5.84 | 11.69 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.396 ms | 12.01 | 12.01 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.396 ms | 12.02 | 12.02 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.208 ms | 222.21 | 2.60 | n/a | ok | 6.9x (87%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.184 ms | 64.15 | 0.50 | n/a | ok | 7.3x (91%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 3.785 ms | 70.91 | 0.42 | n/a | ok | 7.2x (90%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.790 ms | 70.83 | 0.42 | n/a | ok | 7.2x (90%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.809 ms | 70.47 | 0.41 | n/a | ok | 7.2x (90%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

