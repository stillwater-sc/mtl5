## ryzen-9-8945hs -- native, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 506.3 GFLOP/s, f32 1011.7 GFLOP/s |
| ... hardware, AVX-512F | f64 506.3, f32 1011.7 GFLOP/s |
| ... this build, scalar | f64 12.5, f32 14.5 GFLOP/s |
| bandwidth | read 52.7, update 53.4, triad 41.0 GB/s (ceiling 53.4) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 82% | 126% | 0 | 53% |
| blas2 | 10 | 61% | 108% | 0 | 12% |
| blas3 | 10 | 2% | 21% | 0 | 13% |
| factor | 12 | 1% | 6% | 0 | 19% |
| eigen | 8 | 0% | 2% | 0 | 13% |
| elementwise | 5 | 6% | 10% | 0 | 13% |
| sparse | 5 | 25% | 290% | 0 | 14% |
| direct | 19 | 0% | 16% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 11% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 25% | 31% | 0 | 12% |
| mg | 1 | 29% | 29% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 11% | 18% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 95% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.297 ms | 7.30 | 58.43 | 109% | ok | 6.3x (78%) |
| blas1 | dot | f32 | n=8388608 | 1.541 ms | 10.88 | 43.53 | 82% | ok | 8.0x (99%) |
| blas1 | axpy | f64 | n=8388608 | 3.532 ms | 4.75 | 57.00 | 107% | ok | 4.2x (53%) |
| blas1 | axpy | f32 | n=8388608 | 1.916 ms | 8.76 | 52.54 | 98% | ok | 7.8x (97%) |
| blas1 | scale | f64 | n=8388608 | 1.990 ms | 4.21 | 67.44 | 126% | ok | 1.3x (17%) |
| blas1 | two_norm | f64 | n=8388608 | 1.740 ms | 9.64 | 38.57 | 72% | ok | 8.0x (100%) |
| blas1 | one_norm | f64 | n=8388608 | 6.416 ms | 1.31 | 10.46 | 20% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.443 ms | 1.30 | 10.42 | 20% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 6.394 ms | 1.31 | 10.49 | 20% | ok | 1.1x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 108.433 ms | 1.70 | 1.24 | 2% | ok | 1.0x (12%) |
| blas2 | gemv | f64 | n=4096 | 2.414 ms | 13.90 | 55.64 | 104% | ok | 1.3x (17%) |
| blas2 | gemv | f32 | n=4096 | 1.183 ms | 28.36 | 56.74 | 106% | ok | 2.4x (30%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.330 ms | 14.40 | 57.64 | 108% | ok | 2.9x (36%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.163 ms | 28.84 | 57.72 | 108% | ok | 2.0x (25%) |
| blas2 | symv | f64 | n=4096 | 3.879 ms | 8.65 | 17.32 | 32% | ok | 0.8x (11%) |
| blas2 | trmv | f64 | n=4096 | 2.129 ms | 7.88 | 31.56 | 59% | ok | 0.9x (11%) |
| blas2 | trmv | f32 | n=4096 | 1.911 ms | 8.78 | 17.57 | 33% | ok | 0.9x (12%) |
| blas2 | trsv | f64 | n=4096 | 2.067 ms | 8.12 | 32.49 | 61% | ok | 0.9x (12%) |
| blas2 | trsv | f32 | n=4096 | 1.913 ms | 8.77 | 17.55 | 33% | ok | 0.9x (12%) |
| blas2 | ger | f64 | n=4096 | 16.457 ms | 2.04 | 16.32 | 31% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.521 ms | 106.47 | 2.50 | 21% | ok | 7.9x (99%) |
| blas3 | gemm | f32 | 512x512x512 | 1.244 ms | 215.70 | 2.53 | 21% | ok | 8.0x (100%) |
| blas3 | gemmt | f64 | 512x512x512 | 13.956 ms | 9.62 | 0.45 | 2% | ok | 1.0x (12%) |
| blas3 | symm | f64 | 512x512x512 | 20.648 ms | 13.00 | 0.30 | 3% | ok | 1.0x (13%) |
| blas3 | syrk | f64 | 512x512x512 | 16.772 ms | 8.00 | 0.25 | 2% | ok | 1.0x (13%) |
| blas3 | syr2k | f64 | 512x512x512 | 28.433 ms | 9.44 | 0.22 | 2% | ok | 1.0x (12%) |
| blas3 | trmm | f64 | 512x512x512 | 30.685 ms | 4.37 | 0.17 | 1% | ok | 1.0x (13%) |
| blas3 | trsm | f64 | 512x512x512 | 35.242 ms | 3.81 | 0.15 | 1% | ok | 1.0x (13%) |
| blas3 | trsm | f32 | 512x512x512 | 32.146 ms | 4.18 | 0.08 | 0% | ok | 1.0x (13%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 11.203 ms | 0.47 | 3.79 | 7% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 25.940 ms | 3.45 | 0.16 | 1% | ok | 1.6x (19%) |
| factor | lu | f32 | 512x512 | 23.963 ms | 3.73 | 0.09 | 0% | ok | 1.6x (20%) |
| factor | cholesky | f64 | 512x512 | 14.268 ms | 3.14 | 0.29 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 13.691 ms | 3.27 | 0.15 | 0% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 14.384 ms | 3.11 | 0.29 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 13.072 ms | 3.42 | 0.32 | 1% | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 501.841 ms | 0.36 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 50.862 ms | 3.52 | 0.08 | 1% | ok | 1.6x (20%) |
| factor | hessenberg | f64 | 512x512 | 568.612 ms | 0.79 | 0.01 | 0% | ok | 1.7x (21%) |
| factor | householder | f64 | 512x512 | 1.250 ms | 0.84 | 3.36 | 6%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 96.925 ms | 2.77 | 0.04 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 27.412 ms | 3.26 | 0.15 | 1% | ok | 1.5x (19%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 49.422 ms | 0.45 | 0.02 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 163.895 ms | 0.92 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 643.626 ms | 0.23 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 76.657 ms | 2.19 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | schur | f64 | 256x256 | 142.029 ms | 2.95 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 181.679 ms | 2.03 | 0.01 | 0% | ok | 0.8x (10%) |
| eigen | expm | f64 | 256x256 | 24.666 ms | 10.20 | 0.04 | 2% | ok | 1.5x (19%) |
| eigen | sylvester | f64 | 256x256 | 324.617 ms | 3.10 | 0.00 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 39.706 ms | 0.21 | 3.38 | 6% | ok | 0.8x (11%) |
| elementwise | log | f64 | n=8388608 | 53.108 ms | 0.16 | 2.53 | 5% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 48.494 ms | 0.17 | 2.77 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 26.401 ms | 0.32 | 5.08 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 38.566 ms | 0.22 | 3.48 | 7% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.176 ms | 14.87 | 154.72 | 290% ! | ok | 5.3x (66%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 2.063 ms | 1.27 | 13.20 | 25% | ok | 0.8x (10%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.259 ms | 18.54 | 38.98 | 73% | ok | 3.5x (44%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 21.124 ms | 0.15 | 0.54 | 1%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.883 ms | 1.07 | 8.84 | 17%\* | ok | 1.1x (14%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 23.311 ms | 0.48 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 39.368 ms | 0.28 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 19.500 ms | 0.57 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 65.227 ms | 0.34 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 69.882 ms | 0.32 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 29.839 ms | 0.75 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 42.445 ms | 0.04 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.628 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 7.059 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 75.790 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.235 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.707 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.684 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.064 ms | 0.66 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.344 ms | 2.20 | 8.80 | 16%\* | ok | 1.1x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.182 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.283 ms | 1.54 | 0.10 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.345 ms | 2.19 | 8.76 | 16%\* | ok | 1.1x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.137 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.304 ms | 0.38 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.286 ms | 0.59 | 2.35 | 4%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.936 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.817 ms | 0.37 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.258 ms | 0.60 | 2.41 | 5%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 7.182 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 346.734 ms | 0.89 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.248 ms | 1.50 | 5.99 | 11%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.625 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 92.537 ms | 3.33 | 0.01 | 1% | ok | 0.9x (11%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.338 ms | 1.44 | 5.76 | 11%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.640 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1112.142 ms | 0.55 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.230 ms | 0.30 | 1.20 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.641 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 608.671 ms | 1.01 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.251 ms | 0.30 | 1.20 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 51.263 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 77.373 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 40.648 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.984 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 28.459 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.806 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 73.999 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 44.450 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.622 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 34.687 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.894 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 34.390 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 45.391 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.055 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 3.092 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| pc | ilut | f64 | 128^2 grid | 27.502 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 2.244 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 128^2 grid | 2.169 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.036 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.805 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.751 ms | 1.50 | 16.75 | 31% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.164 ms | 1.21 | 13.55 | 25% | ok | 0.9x (11%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.235 ms | 1.17 | 13.12 | 25% | ok | 1.0x (12%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.551 ms | 2.07 | 15.67 | 29% | ok | 1.1x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.490 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.175 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 67.978 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.999 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 14.382 ms | 1.17 | 4.67 | 9% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.785 ms | 56.10 | 0.88 | 11% | ok | 7.7x (96%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.837 ms | 9.13 | 9.70 | 18% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.999 ms | 4.20 | 16.78 | n/a | ok | 1.1x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.232 ms | 5.19 | 10.38 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.727 ms | 9.72 | 9.72 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.726 ms | 9.72 | 9.72 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.265 ms | 212.27 | 2.49 | n/a | ok | 7.9x (99%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.995 ms | 53.74 | 0.42 | n/a | ok | 7.7x (96%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 4.509 ms | 59.54 | 0.35 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 4.502 ms | 59.62 | 0.35 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.530 ms | 59.26 | 0.35 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

