## ryzen-7-8845hs -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 623.9 GFLOP/s, f32 1248.4 GFLOP/s |
| ... hardware, AVX-512F | f64 623.9, f32 1248.4 GFLOP/s |
| ... this build, scalar | f64 14.4, f32 14.4 GFLOP/s |
| bandwidth | read 52.3, update 52.7, triad 40.4 GB/s (ceiling 52.7) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 95% | 128% | 0 | 43% |
| blas2 | 10 | 102% | 112% | 0 | 17% |
| blas3 | 10 | 40% | 71% | 0 | 63% |
| factor | 12 | 2% | 8% | 0 | 12% |
| eigen | 8 | 2% | 5% | 0 | 11% |
| elementwise | 5 | 7% | 12% | 0 | 12% |
| sparse | 5 | 23% | 295% | 0 | 13% |
| direct | 19 | 0% | 20% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 13% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 31% | 40% | 0 | 12% |
| mg | 1 | 36% | 36% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 13% | 22% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 88% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.335 ms | 7.18 | 57.48 | 109% | ok | 4.1x (51%) |
| blas1 | dot | f32 | n=8388608 | 1.347 ms | 12.45 | 49.81 | 95% | ok | 8.0x (100%) |
| blas1 | axpy | f64 | n=8388608 | 3.514 ms | 4.77 | 57.29 | 109% | ok | 3.4x (43%) |
| blas1 | axpy | f32 | n=8388608 | 1.774 ms | 9.46 | 56.75 | 108% | ok | 6.8x (85%) |
| blas1 | scale | f64 | n=8388608 | 1.984 ms | 4.23 | 67.64 | 128% | ok | 1.2x (14%) |
| blas1 | two_norm | f64 | n=8388608 | 1.356 ms | 12.37 | 49.49 | 94% | ok | 7.3x (91%) |
| blas1 | one_norm | f64 | n=8388608 | 5.068 ms | 1.66 | 13.24 | 25% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.114 ms | 1.64 | 13.12 | 25% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 5.062 ms | 1.66 | 13.26 | 25% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 88.120 ms | 2.09 | 1.52 | 3% | ok | 0.9x (12%) |
| blas2 | gemv | f64 | n=4096 | 2.551 ms | 13.15 | 52.64 | 100% | ok | 1.4x (17%) |
| blas2 | gemv | f32 | n=4096 | 1.170 ms | 28.69 | 57.40 | 109% | ok | 1.3x (17%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.625 ms | 12.78 | 51.15 | 97% | ok | 1.4x (18%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.266 ms | 26.50 | 53.02 | 101% | ok | 1.1x (14%) |
| blas2 | symv | f64 | n=4096 | 1.250 ms | 26.85 | 53.76 | 102% | ok | 1.5x (18%) |
| blas2 | trmv | f64 | n=4096 | 1.232 ms | 13.62 | 54.52 | 104% | ok | 1.4x (18%) |
| blas2 | trmv | f32 | n=4096 | 0.571 ms | 29.40 | 58.86 | 112% | ok | 1.6x (20%) |
| blas2 | trsv | f64 | n=4096 | 1.780 ms | 9.42 | 37.73 | 72% | ok | 0.9x (11%) |
| blas2 | trsv | f32 | n=4096 | 0.882 ms | 19.03 | 38.09 | 72% | ok | 1.1x (14%) |
| blas2 | ger | f64 | n=4096 | 4.754 ms | 7.06 | 56.48 | 107% | ok | 1.3x (17%) |
| blas3 | gemm | f64 | 512x512x512 | 0.607 ms | 441.87 | 10.36 | 71% | ok | 6.0x (75%) |
| blas3 | gemm | f32 | 512x512x512 | 0.339 ms | 792.31 | 9.28 | 63% | ok | 5.6x (70%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.348 ms | 99.55 | 4.67 | 16% | ok | 2.0x (25%) |
| blas3 | symm | f64 | 512x512x512 | 0.611 ms | 439.70 | 10.31 | 70% | ok | 6.2x (78%) |
| blas3 | syrk | f64 | 512x512x512 | 1.724 ms | 77.85 | 2.43 | 12% | ok | 1.7x (21%) |
| blas3 | syr2k | f64 | 512x512x512 | 2.388 ms | 112.41 | 2.63 | 18% | ok | 2.0x (25%) |
| blas3 | trmm | f64 | 512x512x512 | 0.390 ms | 344.15 | 13.44 | 55% | ok | 4.9x (61%) |
| blas3 | trsm | f64 | 512x512x512 | 0.538 ms | 249.24 | 9.74 | 40% | ok | 5.3x (66%) |
| blas3 | trsm | f32 | 512x512x512 | 0.268 ms | 501.37 | 9.79 | 40% | ok | 5.0x (63%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 8.699 ms | 0.61 | 4.88 | 9% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 9.088 ms | 9.85 | 0.46 | 2% | ok | 0.5x (7%) |
| factor | lu | f32 | 512x512 | 4.300 ms | 20.81 | 0.49 | 2% | ok | 0.6x (7%) |
| factor | cholesky | f64 | 512x512 | 5.979 ms | 7.48 | 0.70 | 1%\* | ok | 0.7x (9%) |
| factor | cholesky | f32 | 512x512 | 2.259 ms | 19.81 | 0.93 | 2%\* | ok | 0.8x (9%) |
| factor | ldlt | f64 | 512x512 | 11.874 ms | 3.77 | 0.35 | 1%\* | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 10.709 ms | 4.18 | 0.39 | 1%\* | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 11.780 ms | 15.19 | 0.36 | 2% | ok | 0.9x (11%) |
| factor | lq | f64 | 512x512 | 42.052 ms | 4.26 | 0.10 | 1% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 459.745 ms | 0.97 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.014 ms | 1.04 | 4.14 | 8%\* | ok | 3.7x (47%) |
| factor | inv | f64 | 512x512 | 65.489 ms | 4.10 | 0.06 | 1% | ok | 0.9x (12%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 7.228 ms | 12.38 | 0.58 | 2% | ok | 0.8x (10%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 3.597 ms | 6.22 | 0.29 | 1% | ok | 0.5x (6%) |
| eigen | eigen_symmetric | f64 | 256x256 | 134.195 ms | 1.13 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 506.985 ms | 0.30 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 21.383 ms | 7.85 | 0.05 | 1% | ok | 0.8x (10%) |
| eigen | schur | f64 | 256x256 | 26.235 ms | 15.99 | 0.04 | 3% | ok | 0.8x (11%) |
| eigen | svd | f64 | 256x256 | 12.021 ms | 30.71 | 0.09 | 5% | ok | 0.7x (9%) |
| eigen | expm | f64 | 256x256 | 17.730 ms | 14.19 | 0.06 | 2% | ok | 0.9x (11%) |
| eigen | sylvester | f64 | 256x256 | 81.335 ms | 12.38 | 0.02 | 2% | ok | 0.9x (11%) |
| elementwise | exp | f64 | n=8388608 | 27.825 ms | 0.30 | 4.82 | 9% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 43.398 ms | 0.19 | 3.09 | 6% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 39.050 ms | 0.21 | 3.44 | 7% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 20.668 ms | 0.41 | 6.49 | 12% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 34.476 ms | 0.24 | 3.89 | 7% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.175 ms | 14.96 | 155.60 | 295% ! | ok | 5.1x (64%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.218 ms | 1.18 | 12.28 | 23% | ok | 0.6x (7%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.263 ms | 18.50 | 38.90 | 74% | ok | 2.9x (37%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.717 ms | 0.19 | 0.69 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.639 ms | 1.22 | 10.15 | 19%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 19.496 ms | 0.57 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 31.396 ms | 0.36 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 15.454 ms | 0.72 | 0.05 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 54.681 ms | 0.41 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 57.340 ms | 0.39 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 25.246 ms | 0.89 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 38.253 ms | 0.05 | 0.00 | 0%\* | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.757 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.681 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 61.252 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.230 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.478 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.985 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.664 ms | 0.76 | 0.05 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.283 ms | 2.67 | 10.68 | 20%\* | ok | 1.0x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.681 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 5.715 ms | 1.96 | 0.13 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.284 ms | 2.66 | 10.66 | 20%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.325 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.032 ms | 0.47 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.124 ms | 0.67 | 2.69 | 5%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.808 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 49.053 ms | 0.46 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.152 ms | 0.66 | 2.63 | 5%\* | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.732 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 259.842 ms | 1.19 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.021 ms | 1.67 | 6.67 | 13%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 13.742 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 70.064 ms | 4.40 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.075 ms | 1.62 | 6.49 | 12%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.147 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 906.269 ms | 0.68 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.556 ms | 0.32 | 1.28 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.500 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 510.693 ms | 1.21 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.586 ms | 0.32 | 1.27 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 39.205 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 58.877 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 31.837 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 16.772 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 21.767 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.765 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 54.630 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 35.723 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 19.870 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 26.272 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 24.709 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 25.729 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 33.624 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.044 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.407 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 22.754 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.809 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 1.782 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.030 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.357 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.406 ms | 1.86 | 20.86 | 40% | ok | 1.1x (14%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.768 ms | 1.48 | 16.59 | 31% | ok | 0.9x (11%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.828 ms | 1.43 | 16.04 | 30% | ok | 1.0x (12%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.381 ms | 2.50 | 18.90 | 36% | ok | 1.1x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.668 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 33.159 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 51.113 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.872 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 9.917 ms | 1.69 | 6.77 | 13% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.083 ms | 65.75 | 1.03 | 11% | ok | 7.2x (90%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.505 ms | 11.15 | 11.84 | 22% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.268 ms | 5.13 | 20.53 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.876 ms | 5.83 | 11.67 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.404 ms | 11.95 | 11.95 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.403 ms | 11.96 | 11.96 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.204 ms | 222.99 | 2.61 | n/a | ok | 7.0x (88%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.199 ms | 63.92 | 0.50 | n/a | ok | 7.3x (91%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 3.787 ms | 70.89 | 0.42 | n/a | ok | 7.2x (90%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.790 ms | 70.83 | 0.42 | n/a | ok | 7.2x (90%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.812 ms | 70.42 | 0.41 | n/a | ok | 7.2x (90%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

