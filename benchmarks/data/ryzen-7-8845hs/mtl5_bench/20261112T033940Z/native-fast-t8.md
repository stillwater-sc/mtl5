## ryzen-7-8845hs -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 626.5 GFLOP/s, f32 1254.5 GFLOP/s |
| ... hardware, AVX-512F | f64 626.5, f32 1254.5 GFLOP/s |
| ... this build, SSE2 | f64 313.7, f32 627.5 GFLOP/s |
| bandwidth | read 53.0, update 53.4, triad 41.0 GB/s (ceiling 53.4) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 111% | 126% | 0 | 13% |
| blas2 | 10 | 67% | 111% | 0 | 13% |
| blas3 | 10 | 6% | 19% | 0 | 21% |
| factor | 12 | 1% | 8% | 0 | 18% |
| eigen | 8 | 0% | 2% | 0 | 13% |
| elementwise | 5 | 8% | 11% | 0 | 12% |
| sparse | 5 | 40% | 250% | 0 | 34% |
| direct | 7 | 0% | 0% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 32% | 34% | 0 | 12% |
| mg | 1 | 35% | 35% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 70% | 71% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 68% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.259 ms | 7.43 | 59.41 | 111% | ok | 1.1x (13%) |
| blas1 | dot | f32 | n=8388608 | 1.000 ms | 16.77 | 67.10 | 126% | ok | 1.2x (15%) |
| blas1 | axpy | f64 | n=8388608 | 3.486 ms | 4.81 | 57.75 | 108% | ok | 1.0x (13%) |
| blas1 | axpy | f32 | n=8388608 | 1.667 ms | 10.07 | 60.40 | 113% | ok | 1.1x (14%) |
| blas1 | scale | f64 | n=8388608 | 2.116 ms | 3.96 | 63.42 | 119% | ok | 0.9x (11%) |
| blas1 | two_norm | f64 | n=8388608 | 1.054 ms | 15.92 | 63.69 | 119% | ok | 1.1x (14%) |
| blas1 | one_norm | f64 | n=8388608 | 5.064 ms | 1.66 | 13.25 | 25% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.076 ms | 1.65 | 13.22 | 25% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 5.063 ms | 1.66 | 13.25 | 25% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 91.399 ms | 2.02 | 1.47 | 3% | ok | 0.9x (12%) |
| blas2 | gemv | f64 | n=4096 | 2.339 ms | 14.35 | 57.42 | 107% | ok | 1.4x (18%) |
| blas2 | gemv | f32 | n=4096 | 1.136 ms | 29.53 | 59.10 | 111% | ok | 1.6x (19%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 9.678 ms | 3.47 | 13.88 | 26% | ok | 2.1x (27%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.733 ms | 7.09 | 14.19 | 27% | ok | 2.1x (26%) |
| blas2 | symv | f64 | n=4096 | 8.715 ms | 3.85 | 7.71 | 14% | ok | 1.0x (12%) |
| blas2 | trmv | f64 | n=4096 | 1.888 ms | 8.89 | 35.59 | 67% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=4096 | 0.940 ms | 17.86 | 35.75 | 67% | ok | 1.1x (13%) |
| blas2 | trsv | f64 | n=4096 | 1.839 ms | 9.12 | 36.53 | 68% | ok | 1.0x (12%) |
| blas2 | trsv | f32 | n=4096 | 1.076 ms | 15.60 | 31.23 | 58% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 13.438 ms | 2.50 | 19.98 | 37% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.248 ms | 119.43 | 2.80 | 19% | ok | 4.9x (61%) |
| blas3 | gemm | f32 | 512x512x512 | 1.206 ms | 222.56 | 2.61 | 18% | ok | 4.6x (57%) |
| blas3 | gemmt | f64 | 512x512x512 | 3.498 ms | 38.37 | 1.80 | 6% | ok | 2.1x (26%) |
| blas3 | symm | f64 | 512x512x512 | 3.481 ms | 77.11 | 1.81 | 12% | ok | 3.5x (43%) |
| blas3 | syrk | f64 | 512x512x512 | 13.822 ms | 9.71 | 0.30 | 2% | ok | 1.0x (12%) |
| blas3 | syr2k | f64 | 512x512x512 | 24.502 ms | 10.96 | 0.26 | 2% | ok | 0.9x (12%) |
| blas3 | trmm | f64 | 512x512x512 | 5.673 ms | 23.66 | 0.92 | 4% | ok | 1.7x (21%) |
| blas3 | trsm | f64 | 512x512x512 | 6.040 ms | 22.22 | 0.87 | 4% | ok | 1.6x (20%) |
| blas3 | trsm | f32 | 512x512x512 | 4.533 ms | 29.61 | 0.58 | 2% | ok | 1.4x (18%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.619 ms | 0.55 | 4.42 | 8% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 23.230 ms | 3.85 | 0.18 | 1% | ok | 1.4x (18%) |
| factor | lu | f32 | 512x512 | 22.367 ms | 4.00 | 0.09 | 0% | ok | 1.4x (18%) |
| factor | cholesky | f64 | 512x512 | 11.388 ms | 3.93 | 0.37 | 1%\* | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 10.905 ms | 4.10 | 0.19 | 0%\* | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 11.454 ms | 3.91 | 0.37 | 1%\* | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 166.835 ms | 0.27 | 0.03 | 0%\* | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 396.951 ms | 0.45 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 44.755 ms | 4.00 | 0.09 | 1% | ok | 1.5x (18%) |
| factor | hessenberg | f64 | 512x512 | 467.536 ms | 0.96 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.026 ms | 1.02 | 4.09 | 8%\* | ok | 3.8x (47%) |
| factor | inv | f64 | 512x512 | 78.505 ms | 3.42 | 0.05 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 24.907 ms | 3.59 | 0.17 | 1% | ok | 1.4x (17%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 39.219 ms | 0.57 | 0.03 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 129.294 ms | 1.17 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 504.947 ms | 0.30 | 0.00 | 0% | ok | 1.0x (12%) |
| eigen | eigenvalue | f64 | 256x256 | 60.029 ms | 2.79 | 0.02 | 0% | ok | 1.0x (12%) |
| eigen | schur | f64 | 256x256 | 111.431 ms | 3.76 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 117.845 ms | 3.13 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 19.518 ms | 12.89 | 0.05 | 2% | ok | 1.3x (17%) |
| eigen | sylvester | f64 | 256x256 | 258.496 ms | 3.89 | 0.01 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 28.558 ms | 0.29 | 4.70 | 9% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 43.320 ms | 0.19 | 3.10 | 6% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 40.207 ms | 0.21 | 3.34 | 6% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 22.524 ms | 0.37 | 5.96 | 11% | ok | 1.0x (12%) |
| elementwise | erf | f64 | n=8388608 | 32.893 ms | 0.26 | 4.08 | 8% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.204 ms | 12.86 | 133.74 | 250% ! | ok | 4.2x (53%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.271 ms | 2.06 | 21.42 | 40% | ok | 4.3x (54%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.271 ms | 18.44 | 38.76 | 73% | ok | 2.7x (34%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.131 ms | 0.20 | 0.71 | 1%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.709 ms | 1.17 | 9.74 | 18%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 98.366 ms | 0.11 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 28.695 ms | 0.39 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 15.257 ms | 0.73 | 0.05 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 55.586 ms | 0.40 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 71.695 ms | 0.31 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 26.142 ms | 0.86 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 83.696 ms | 0.02 | 0.00 | 0%\* | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.720 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.580 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.421 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.129 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 32.706 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 31.364 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 27.588 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 13.428 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 17.881 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 15.729 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 30.895 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 29.024 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 17.306 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 19.305 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 18.364 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 19.536 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 22.079 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.036 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.411 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 23.668 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | ic_0 | f64 | 128^2 grid | 1.826 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 128^2 grid | 1.791 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.658 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.635 ms | 1.60 | 17.93 | 34% | ok | 1.0x (12%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.700 ms | 1.54 | 17.25 | 32% | ok | 0.9x (11%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.779 ms | 1.47 | 16.49 | 31% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.404 ms | 2.50 | 18.87 | 35% | ok | 1.0x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.136 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 46.781 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 54.484 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 6.437 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 1.778 ms | 9.43 | 37.73 | 71% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 2.570 ms | 104.46 | 1.63 | 17% | ok | 5.9x (74%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.479 ms | 35.05 | 37.24 | 70% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.204 ms | 13.93 | 55.72 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.473 ms | 35.50 | 71.00 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.348 ms | 48.25 | 48.25 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.318 ms | 52.76 | 52.76 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.376 ms | 112.98 | 1.32 | n/a | ok | 6.3x (79%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.538 ms | 105.78 | 0.83 | n/a | ok | 6.8x (85%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 2.561 ms | 104.80 | 0.61 | n/a | ok | 6.9x (87%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.670 ms | 400.71 | 2.35 | n/a | ok | 5.5x (68%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.670 ms | 400.77 | 2.35 | n/a | ok | 5.4x (68%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

