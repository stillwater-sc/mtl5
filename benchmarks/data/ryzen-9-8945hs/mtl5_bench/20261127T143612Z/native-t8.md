## ryzen-9-8945hs -- native, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `57512adcb8fe` |
| FMA peak (ceiling) | f64 501.0 GFLOP/s, f32 1004.2 GFLOP/s |
| ... hardware, AVX-512F | f64 501.0, f32 1004.2 GFLOP/s |
| ... this build, scalar | f64 14.7, f32 13.3 GFLOP/s |
| bandwidth | read 52.3, update 52.9, triad 41.2 GB/s (ceiling 52.9) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 90% | 126% | 0 | 47% |
| blas2 | 10 | 63% | 113% | 0 | 13% |
| blas3 | 10 | 2% | 21% | 0 | 12% |
| factor | 12 | 1% | 6% | 0 | 19% |
| eigen | 8 | 0% | 2% | 0 | 13% |
| elementwise | 5 | 7% | 10% | 0 | 13% |
| sparse | 5 | 23% | 284% | 0 | 13% |
| direct | 19 | 0% | 17% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 11% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 26% | 30% | 0 | 12% |
| mg | 1 | 30% | 30% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 11% | 18% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 95% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.293 ms | 7.32 | 58.53 | 111% | ok | 5.3x (67%) |
| blas1 | dot | f32 | n=8388608 | 1.541 ms | 10.89 | 43.55 | 82% | ok | 8.0x (99%) |
| blas1 | axpy | f64 | n=8388608 | 3.544 ms | 4.73 | 56.82 | 107% | ok | 3.8x (47%) |
| blas1 | axpy | f32 | n=8388608 | 1.769 ms | 9.49 | 56.92 | 108% | ok | 7.2x (90%) |
| blas1 | scale | f64 | n=8388608 | 2.018 ms | 4.16 | 66.52 | 126% | ok | 1.3x (16%) |
| blas1 | two_norm | f64 | n=8388608 | 1.409 ms | 11.90 | 47.61 | 90% | ok | 7.9x (99%) |
| blas1 | one_norm | f64 | n=8388608 | 6.400 ms | 1.31 | 10.49 | 20% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.415 ms | 1.31 | 10.46 | 20% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 6.399 ms | 1.31 | 10.49 | 20% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 110.232 ms | 1.67 | 1.22 | 2% | ok | 1.0x (12%) |
| blas2 | gemv | f64 | n=4096 | 2.426 ms | 13.83 | 55.36 | 105% | ok | 1.4x (17%) |
| blas2 | gemv | f32 | n=4096 | 1.123 ms | 29.88 | 59.79 | 113% | ok | 2.5x (31%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.332 ms | 14.39 | 57.58 | 109% | ok | 3.0x (38%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.153 ms | 29.11 | 58.25 | 110% | ok | 2.0x (25%) |
| blas2 | symv | f64 | n=4096 | 3.570 ms | 9.40 | 18.81 | 36% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 2.054 ms | 8.17 | 32.71 | 62% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=4096 | 1.761 ms | 9.53 | 19.07 | 36% | ok | 1.1x (13%) |
| blas2 | trsv | f64 | n=4096 | 2.028 ms | 8.27 | 33.12 | 63% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.800 ms | 9.32 | 18.66 | 35% | ok | 1.1x (13%) |
| blas2 | ger | f64 | n=4096 | 16.322 ms | 2.06 | 16.45 | 31% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 2.523 ms | 106.40 | 2.49 | 21% | ok | 7.7x (97%) |
| blas3 | gemm | f32 | 512x512x512 | 1.260 ms | 212.98 | 2.50 | 21% | ok | 7.9x (99%) |
| blas3 | gemmt | f64 | 512x512x512 | 13.959 ms | 9.62 | 0.45 | 2% | ok | 1.0x (12%) |
| blas3 | symm | f64 | 512x512x512 | 20.229 ms | 13.27 | 0.31 | 3% | ok | 1.0x (12%) |
| blas3 | syrk | f64 | 512x512x512 | 16.819 ms | 7.98 | 0.25 | 2% | ok | 1.0x (12%) |
| blas3 | syr2k | f64 | 512x512x512 | 28.464 ms | 9.43 | 0.22 | 2% | ok | 1.0x (12%) |
| blas3 | trmm | f64 | 512x512x512 | 37.080 ms | 3.62 | 0.14 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f64 | 512x512x512 | 33.922 ms | 3.96 | 0.15 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f32 | 512x512x512 | 35.199 ms | 3.81 | 0.07 | 0% | ok | 1.0x (13%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.565 ms | 0.50 | 4.02 | 8% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 25.914 ms | 3.45 | 0.16 | 1% | ok | 1.6x (20%) |
| factor | lu | f32 | 512x512 | 24.548 ms | 3.65 | 0.09 | 0% | ok | 1.6x (19%) |
| factor | cholesky | f64 | 512x512 | 14.382 ms | 3.11 | 0.29 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 13.724 ms | 3.26 | 0.15 | 0% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 14.472 ms | 3.09 | 0.29 | 1% | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 12.680 ms | 3.53 | 0.33 | 1% | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 501.112 ms | 0.36 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 51.927 ms | 3.45 | 0.08 | 1% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 576.186 ms | 0.78 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.251 ms | 0.84 | 3.35 | 6%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 97.073 ms | 2.77 | 0.04 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 28.162 ms | 3.18 | 0.15 | 1% | ok | 1.5x (19%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 47.703 ms | 0.47 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 161.536 ms | 0.93 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 620.388 ms | 0.24 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 70.726 ms | 2.37 | 0.01 | 0% | ok | 1.1x (13%) |
| eigen | schur | f64 | 256x256 | 146.965 ms | 2.85 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 152.128 ms | 2.43 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | expm | f64 | 256x256 | 22.227 ms | 11.32 | 0.05 | 2% | ok | 1.7x (21%) |
| eigen | sylvester | f64 | 256x256 | 325.296 ms | 3.09 | 0.00 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 32.590 ms | 0.26 | 4.12 | 8% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 50.756 ms | 0.17 | 2.64 | 5% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 50.257 ms | 0.17 | 2.67 | 5% | ok | 0.9x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 25.457 ms | 0.33 | 5.27 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 38.384 ms | 0.22 | 3.50 | 7% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.181 ms | 14.43 | 150.11 | 284% ! | ok | 5.0x (63%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.266 ms | 1.16 | 12.02 | 23% | ok | 0.7x (9%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.302 ms | 18.20 | 38.26 | 72% | ok | 3.2x (40%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 20.401 ms | 0.16 | 0.56 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.852 ms | 1.08 | 8.99 | 17%\* | ok | 1.1x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 23.846 ms | 0.47 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 38.835 ms | 0.29 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 19.430 ms | 0.58 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 64.922 ms | 0.34 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 82.637 ms | 0.27 | 0.01 | 0% | ok | 0.8x (10%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 34.286 ms | 0.65 | 0.02 | 0% | ok | 0.9x (11%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 43.250 ms | 0.04 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.608 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.913 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 76.117 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.242 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.949 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.691 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.713 ms | 0.63 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.344 ms | 2.20 | 8.79 | 17%\* | ok | 1.1x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.173 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.027 ms | 1.59 | 0.10 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.347 ms | 2.18 | 8.71 | 16%\* | ok | 1.0x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.039 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 60.428 ms | 0.37 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.331 ms | 0.57 | 2.27 | 4%\* | ok | 1.0x (12%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.900 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 60.038 ms | 0.37 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.309 ms | 0.58 | 2.31 | 4%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 6.972 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 342.741 ms | 0.90 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.259 ms | 1.49 | 5.97 | 11%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 16.913 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 84.388 ms | 3.66 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.319 ms | 1.45 | 5.81 | 11%\* | ok | 1.0x (13%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.534 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1138.679 ms | 0.54 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.295 ms | 0.30 | 1.19 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.482 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 623.518 ms | 0.99 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.480 ms | 0.29 | 1.17 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 49.766 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 102.939 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 39.746 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.163 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 27.497 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.350 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 69.018 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 43.169 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.997 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 32.826 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 30.986 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 32.264 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.230 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.054 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.962 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 27.813 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 2.201 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 2.153 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.030 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.837 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.092 ms | 1.25 | 14.02 | 26% | ok | 0.8x (10%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.849 ms | 1.42 | 15.86 | 30% | ok | 1.0x (12%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.238 ms | 1.17 | 13.11 | 25% | ok | 1.0x (12%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.364 ms | 2.09 | 15.83 | 30% | ok | 1.1x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.839 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.343 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 68.275 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.617 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 12.245 ms | 1.37 | 5.48 | 10% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.811 ms | 55.80 | 0.87 | 11% | ok | 7.7x (96%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.835 ms | 9.14 | 9.71 | 18% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.960 ms | 3.38 | 13.53 | n/a | ok | 0.8x (10%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.246 ms | 5.17 | 10.34 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.727 ms | 9.72 | 9.72 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.726 ms | 9.72 | 9.72 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.246 ms | 215.52 | 2.53 | n/a | ok | 7.9x (99%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.029 ms | 53.38 | 0.42 | n/a | ok | 7.7x (96%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.541 ms | 59.12 | 0.35 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.555 ms | 58.93 | 0.35 | n/a | ok | 7.6x (95%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.553 ms | 58.95 | 0.35 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

