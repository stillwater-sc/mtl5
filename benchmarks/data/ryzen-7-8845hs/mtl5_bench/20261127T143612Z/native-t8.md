## ryzen-7-8845hs -- native, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `57512adcb8fe` |
| FMA peak (ceiling) | f64 628.0 GFLOP/s, f32 1254.7 GFLOP/s |
| ... hardware, AVX-512F | f64 628.0, f32 1254.7 GFLOP/s |
| ... this build, scalar | f64 14.6, f32 14.4 GFLOP/s |
| bandwidth | read 53.1, update 53.6, triad 41.4 GB/s (ceiling 53.6) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 98% | 124% | 0 | 42% |
| blas2 | 10 | 74% | 109% | 0 | 13% |
| blas3 | 10 | 2% | 18% | 0 | 12% |
| factor | 12 | 1% | 8% | 0 | 18% |
| eigen | 8 | 0% | 2% | 0 | 13% |
| elementwise | 5 | 8% | 12% | 0 | 12% |
| sparse | 5 | 35% | 291% | 0 | 15% |
| direct | 19 | 0% | 20% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 13% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 37% | 37% | 0 | 12% |
| mg | 1 | 36% | 36% | 0 | 15% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 13% | 22% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 88% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.381 ms | 7.05 | 56.37 | 105% | ok | 4.1x (52%) |
| blas1 | dot | f32 | n=8388608 | 1.335 ms | 12.57 | 50.26 | 94% | ok | 7.6x (95%) |
| blas1 | axpy | f64 | n=8388608 | 3.524 ms | 4.76 | 57.13 | 106% | ok | 3.4x (42%) |
| blas1 | axpy | f32 | n=8388608 | 1.734 ms | 9.68 | 58.06 | 108% | ok | 5.9x (74%) |
| blas1 | scale | f64 | n=8388608 | 2.025 ms | 4.14 | 66.27 | 124% | ok | 1.1x (14%) |
| blas1 | two_norm | f64 | n=8388608 | 1.282 ms | 13.08 | 52.33 | 98% | ok | 7.4x (92%) |
| blas1 | one_norm | f64 | n=8388608 | 5.085 ms | 1.65 | 13.20 | 25% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.127 ms | 1.64 | 13.09 | 24% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 5.064 ms | 1.66 | 13.25 | 25% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 81.506 ms | 2.26 | 1.65 | 3% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 2.378 ms | 14.11 | 56.46 | 105% | ok | 1.3x (16%) |
| blas2 | gemv | f32 | n=4096 | 1.170 ms | 28.69 | 57.41 | 107% | ok | 1.9x (24%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.288 ms | 14.66 | 58.69 | 109% | ok | 2.8x (35%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.150 ms | 29.19 | 58.40 | 109% | ok | 1.8x (23%) |
| blas2 | symv | f64 | n=4096 | 2.970 ms | 11.30 | 22.62 | 42% | ok | 1.0x (12%) |
| blas2 | trmv | f64 | n=4096 | 1.713 ms | 9.79 | 39.22 | 73% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=4096 | 1.489 ms | 11.27 | 22.56 | 42% | ok | 1.0x (12%) |
| blas2 | trsv | f64 | n=4096 | 1.683 ms | 9.97 | 39.91 | 74% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.504 ms | 11.15 | 22.33 | 42% | ok | 1.0x (12%) |
| blas2 | ger | f64 | n=4096 | 13.555 ms | 2.48 | 19.81 | 37% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.482 ms | 108.17 | 2.54 | 17% | ok | 6.4x (81%) |
| blas3 | gemm | f32 | 512x512x512 | 1.200 ms | 223.70 | 2.62 | 18% | ok | 6.8x (85%) |
| blas3 | gemmt | f64 | 512x512x512 | 11.402 ms | 11.77 | 0.55 | 2% | ok | 1.0x (12%) |
| blas3 | symm | f64 | 512x512x512 | 17.316 ms | 15.50 | 0.36 | 2% | ok | 1.0x (12%) |
| blas3 | syrk | f64 | 512x512x512 | 13.689 ms | 9.80 | 0.31 | 2% | ok | 1.0x (12%) |
| blas3 | syr2k | f64 | 512x512x512 | 22.923 ms | 11.71 | 0.27 | 2% | ok | 1.0x (12%) |
| blas3 | trmm | f64 | 512x512x512 | 25.951 ms | 5.17 | 0.20 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f64 | 512x512x512 | 26.629 ms | 5.04 | 0.20 | 1% | ok | 1.0x (12%) |
| blas3 | trsm | f32 | 512x512x512 | 26.623 ms | 5.04 | 0.10 | 0% | ok | 1.0x (12%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 8.592 ms | 0.62 | 4.94 | 9% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 22.124 ms | 4.04 | 0.19 | 1% | ok | 1.5x (18%) |
| factor | lu | f32 | 512x512 | 20.711 ms | 4.32 | 0.10 | 0% | ok | 1.5x (19%) |
| factor | cholesky | f64 | 512x512 | 11.601 ms | 3.86 | 0.36 | 1%\* | ok | 1.0x (12%) |
| factor | cholesky | f32 | 512x512 | 10.948 ms | 4.09 | 0.19 | 0%\* | ok | 1.0x (12%) |
| factor | ldlt | f64 | 512x512 | 11.456 ms | 3.91 | 0.37 | 1%\* | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 10.546 ms | 4.24 | 0.40 | 1%\* | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 395.898 ms | 0.45 | 0.01 | 0% | ok | 1.6x (21%) |
| factor | lq | f64 | 512x512 | 41.727 ms | 4.29 | 0.10 | 1% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 461.723 ms | 0.97 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.004 ms | 1.05 | 4.18 | 8%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 77.955 ms | 3.44 | 0.05 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 23.298 ms | 3.84 | 0.18 | 1% | ok | 1.5x (18%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 36.805 ms | 0.61 | 0.03 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 127.568 ms | 1.18 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 498.172 ms | 0.30 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 58.349 ms | 2.88 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 108.802 ms | 3.85 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 119.595 ms | 3.09 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | expm | f64 | 256x256 | 18.392 ms | 13.68 | 0.06 | 2% | ok | 1.6x (20%) |
| eigen | sylvester | f64 | 256x256 | 260.721 ms | 3.86 | 0.01 | 1% | ok | 1.0x (12%) |
| elementwise | exp | f64 | n=8388608 | 27.144 ms | 0.31 | 4.94 | 9% | ok | 1.2x (14%) |
| elementwise | log | f64 | n=8388608 | 41.687 ms | 0.20 | 3.22 | 6% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 38.425 ms | 0.22 | 3.49 | 7% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 20.554 ms | 0.41 | 6.53 | 12% | ok | 1.0x (12%) |
| elementwise | erf | f64 | n=8388608 | 31.278 ms | 0.27 | 4.29 | 8% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.174 ms | 15.01 | 156.14 | 291% ! | ok | 5.1x (63%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.435 ms | 1.82 | 18.97 | 35% | ok | 0.9x (11%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.253 ms | 18.59 | 39.08 | 73% | ok | 2.9x (36%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.595 ms | 0.20 | 0.69 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.612 ms | 1.25 | 10.33 | 19%\* | ok | 1.2x (15%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 19.182 ms | 0.58 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 31.295 ms | 0.36 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 15.094 ms | 0.74 | 0.05 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 53.785 ms | 0.42 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 57.994 ms | 0.39 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 25.457 ms | 0.88 | 0.03 | 0% | ok | 0.9x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 38.188 ms | 0.05 | 0.00 | 0%\* | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.738 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.645 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 60.395 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.227 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.360 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.997 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.364 ms | 0.78 | 0.05 | 0% | ok | 1.0x (12%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.278 ms | 2.72 | 10.88 | 20%\* | ok | 1.0x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.655 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 5.667 ms | 1.97 | 0.13 | 0% | ok | 1.2x (15%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.282 ms | 2.68 | 10.73 | 20%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.336 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.463 ms | 0.47 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.128 ms | 0.67 | 2.68 | 5%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.821 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.399 ms | 0.46 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.132 ms | 0.67 | 2.67 | 5%\* | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.728 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 258.520 ms | 1.19 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.009 ms | 1.68 | 6.71 | 13%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 13.933 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.054 ms | 4.47 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.052 ms | 1.64 | 6.57 | 12%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.085 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 906.412 ms | 0.68 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.599 ms | 0.32 | 1.27 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.469 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 499.461 ms | 1.24 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.504 ms | 0.32 | 1.28 | 2%\* | ok | 1.0x (13%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 39.142 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 80.251 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 31.525 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 16.750 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 21.766 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.708 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 54.976 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.419 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 19.806 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 25.943 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 24.584 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 25.616 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 33.573 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.044 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.457 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 25.598 ms | 0.00 | 0.00 | n/a | ok | 0.9x (11%) |
| pc | ic_0 | f64 | 128^2 grid | 1.794 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 1.739 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.024 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.692 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.480 ms | 1.77 | 19.82 | 37% | ok | 0.9x (12%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.493 ms | 1.75 | 19.64 | 37% | ok | 1.0x (12%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.921 ms | 1.36 | 15.27 | 28% | ok | 0.9x (12%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.953 ms | 2.57 | 19.44 | 36% | ok | 1.2x (15%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.324 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.660 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 52.726 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.758 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 9.812 ms | 1.71 | 6.84 | 13% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.004 ms | 67.04 | 1.05 | 11% | ok | 7.3x (91%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.495 ms | 11.22 | 11.92 | 22% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.223 ms | 5.21 | 20.82 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.894 ms | 5.80 | 11.59 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.402 ms | 11.96 | 11.96 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.403 ms | 11.96 | 11.96 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.158 ms | 231.85 | 2.72 | n/a | ok | 7.0x (88%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 4.136 ms | 64.90 | 0.51 | n/a | ok | 7.4x (92%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 3.751 ms | 71.56 | 0.42 | n/a | ok | 7.3x (91%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 3.761 ms | 71.38 | 0.42 | n/a | ok | 7.3x (91%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.766 ms | 71.28 | 0.42 | n/a | ok | 7.2x (91%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

