## ryzen-7-8845hs -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `57512adcb8fe` |
| FMA peak (ceiling) | f64 625.9 GFLOP/s, f32 1254.7 GFLOP/s |
| ... hardware, AVX-512F | f64 625.9, f32 1254.7 GFLOP/s |
| ... this build, SSE2 | f64 313.7, f32 624.8 GFLOP/s |
| bandwidth | read 53.0, update 53.5, triad 41.2 GB/s (ceiling 53.5) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 112% | 129% | 0 | 13% |
| blas2 | 10 | 70% | 111% | 0 | 14% |
| blas3 | 10 | 9% | 19% | 0 | 38% |
| factor | 12 | 1% | 8% | 0 | 18% |
| eigen | 8 | 0% | 2% | 0 | 13% |
| elementwise | 5 | 8% | 12% | 0 | 12% |
| sparse | 5 | 39% | 286% | 0 | 14% |
| direct | 19 | 0% | 20% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 12% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 33% | 37% | 0 | 12% |
| mg | 1 | 36% | 36% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 69% | 70% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 70% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.237 ms | 7.50 | 60.00 | 112% | ok | 1.1x (13%) |
| blas1 | dot | f32 | n=8388608 | 0.971 ms | 17.28 | 69.11 | 129% | ok | 1.2x (16%) |
| blas1 | axpy | f64 | n=8388608 | 3.460 ms | 4.85 | 58.19 | 109% | ok | 1.0x (13%) |
| blas1 | axpy | f32 | n=8388608 | 1.623 ms | 10.34 | 62.03 | 116% | ok | 1.0x (13%) |
| blas1 | scale | f64 | n=8388608 | 1.965 ms | 4.27 | 68.31 | 128% | ok | 1.0x (12%) |
| blas1 | two_norm | f64 | n=8388608 | 1.119 ms | 14.99 | 59.98 | 112% | ok | 1.0x (13%) |
| blas1 | one_norm | f64 | n=8388608 | 5.066 ms | 1.66 | 13.25 | 25% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.098 ms | 1.65 | 13.16 | 25% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 5.063 ms | 1.66 | 13.25 | 25% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 87.996 ms | 2.10 | 1.53 | 3% | ok | 1.0x (12%) |
| blas2 | gemv | f64 | n=4096 | 2.291 ms | 14.65 | 58.63 | 110% | ok | 1.5x (18%) |
| blas2 | gemv | f32 | n=4096 | 1.134 ms | 29.59 | 59.22 | 111% | ok | 1.6x (19%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.288 ms | 14.67 | 58.70 | 110% | ok | 1.7x (22%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.193 ms | 28.13 | 56.29 | 105% | ok | 1.8x (22%) |
| blas2 | symv | f64 | n=4096 | 3.209 ms | 10.46 | 20.94 | 39% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 1.886 ms | 8.89 | 35.61 | 67% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=4096 | 1.100 ms | 15.25 | 30.54 | 57% | ok | 0.9x (11%) |
| blas2 | trsv | f64 | n=4096 | 1.795 ms | 9.35 | 37.42 | 70% | ok | 1.1x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.955 ms | 17.58 | 35.18 | 66% | ok | 1.1x (14%) |
| blas2 | ger | f64 | n=4096 | 13.705 ms | 2.45 | 19.59 | 37% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.228 ms | 120.49 | 2.82 | 19% | ok | 4.9x (62%) |
| blas3 | gemm | f32 | 512x512x512 | 1.216 ms | 220.75 | 2.59 | 18% | ok | 4.6x (57%) |
| blas3 | gemmt | f64 | 512x512x512 | 3.091 ms | 43.42 | 2.04 | 7% | ok | 2.4x (30%) |
| blas3 | symm | f64 | 512x512x512 | 2.950 ms | 90.99 | 2.13 | 15% | ok | 4.1x (51%) |
| blas3 | syrk | f64 | 512x512x512 | 2.401 ms | 55.91 | 1.75 | 9% | ok | 3.1x (38%) |
| blas3 | syr2k | f64 | 512x512x512 | 4.616 ms | 58.15 | 1.36 | 9% | ok | 3.1x (38%) |
| blas3 | trmm | f64 | 512x512x512 | 5.545 ms | 24.21 | 0.95 | 4% | ok | 1.7x (21%) |
| blas3 | trsm | f64 | 512x512x512 | 5.748 ms | 23.35 | 0.91 | 4% | ok | 1.7x (21%) |
| blas3 | trsm | f32 | 512x512x512 | 4.460 ms | 30.09 | 0.59 | 2% | ok | 1.4x (18%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.000 ms | 0.59 | 4.72 | 9% | ok | 0.9x (12%) |
| factor | lu | f64 | 512x512 | 21.979 ms | 4.07 | 0.19 | 1% | ok | 1.5x (19%) |
| factor | lu | f32 | 512x512 | 21.101 ms | 4.24 | 0.10 | 0% | ok | 1.5x (18%) |
| factor | cholesky | f64 | 512x512 | 11.453 ms | 3.91 | 0.37 | 1%\* | ok | 1.0x (12%) |
| factor | cholesky | f32 | 512x512 | 10.933 ms | 4.09 | 0.19 | 0%\* | ok | 1.0x (12%) |
| factor | ldlt | f64 | 512x512 | 11.520 ms | 3.88 | 0.36 | 1%\* | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 10.443 ms | 4.28 | 0.40 | 1%\* | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 396.684 ms | 0.45 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 42.355 ms | 4.23 | 0.10 | 1% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 456.711 ms | 0.98 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.014 ms | 1.04 | 4.14 | 8%\* | ok | 3.9x (48%) |
| factor | inv | f64 | 512x512 | 78.644 ms | 3.41 | 0.05 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 24.020 ms | 3.73 | 0.17 | 1% | ok | 1.4x (18%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.368 ms | 0.57 | 0.03 | 0% | ok | 0.9x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 126.698 ms | 1.19 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 506.464 ms | 0.30 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 256x256 | 60.667 ms | 2.77 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 114.870 ms | 3.65 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 120.048 ms | 3.07 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | expm | f64 | 256x256 | 18.316 ms | 13.74 | 0.06 | 2% | ok | 1.4x (18%) |
| eigen | sylvester | f64 | 256x256 | 264.603 ms | 3.80 | 0.01 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 27.600 ms | 0.30 | 4.86 | 9% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 41.859 ms | 0.20 | 3.21 | 6% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 38.732 ms | 0.22 | 3.47 | 6% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 20.542 ms | 0.41 | 6.53 | 12% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 31.462 ms | 0.27 | 4.27 | 8% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.178 ms | 14.69 | 152.81 | 286% ! | ok | 4.9x (61%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.291 ms | 2.03 | 21.08 | 39% | ok | 1.2x (14%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.246 ms | 18.64 | 39.20 | 73% | ok | 2.9x (36%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.866 ms | 0.19 | 0.68 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.624 ms | 1.24 | 10.25 | 19%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 18.524 ms | 0.60 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 28.050 ms | 0.40 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 13.412 ms | 0.83 | 0.05 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 54.467 ms | 0.41 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 56.525 ms | 0.40 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 25.282 ms | 0.88 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 38.133 ms | 0.05 | 0.00 | 0%\* | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.752 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.639 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 60.563 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.228 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.100 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.995 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.595 ms | 0.77 | 0.05 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.279 ms | 2.71 | 10.83 | 20%\* | ok | 1.0x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.667 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 5.750 ms | 1.95 | 0.13 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.282 ms | 2.69 | 10.74 | 20%\* | ok | 1.0x (13%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.317 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.367 ms | 0.47 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.128 ms | 0.67 | 2.68 | 5%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.801 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.497 ms | 0.46 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.136 ms | 0.67 | 2.66 | 5%\* | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.634 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 260.146 ms | 1.19 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.019 ms | 1.67 | 6.68 | 12%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 13.996 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.300 ms | 4.45 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.064 ms | 1.63 | 6.53 | 12%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.080 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 907.634 ms | 0.68 | 0.00 | 0% | ok | 1.1x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.576 ms | 0.32 | 1.27 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.488 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 494.820 ms | 1.25 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.595 ms | 0.32 | 1.27 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 32.654 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 61.684 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 28.390 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 13.512 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 17.988 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 15.749 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 34.601 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 29.399 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 17.367 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 18.645 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 17.595 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.691 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.649 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.044 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.404 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 22.679 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 1.804 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 1.758 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.024 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.666 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.478 ms | 1.77 | 19.84 | 37% | ok | 0.9x (12%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.642 ms | 1.59 | 17.86 | 33% | ok | 0.9x (11%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.761 ms | 1.49 | 16.65 | 31% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.983 ms | 2.57 | 19.40 | 36% | ok | 1.1x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.328 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.803 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 51.087 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 6.477 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 1.786 ms | 9.39 | 37.57 | 70% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 2.558 ms | 104.94 | 1.64 | 17% | ok | 6.0x (75%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.484 ms | 34.68 | 36.85 | 69% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.247 ms | 13.46 | 53.83 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.469 ms | 35.77 | 71.54 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.351 ms | 47.85 | 47.85 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.321 ms | 52.35 | 52.35 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.346 ms | 114.41 | 1.34 | n/a | ok | 6.4x (80%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.519 ms | 106.57 | 0.83 | n/a | ok | 6.9x (86%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 2.543 ms | 105.58 | 0.62 | n/a | ok | 7.0x (87%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.661 ms | 406.35 | 2.38 | n/a | ok | 5.6x (70%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.659 ms | 407.21 | 2.39 | n/a | ok | 5.6x (70%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

