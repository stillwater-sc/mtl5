## ryzen-7-8845hs -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 80.1 GFLOP/s, f32 160.0 GFLOP/s |
| ... hardware, AVX-512F | f64 80.1, f32 160.0 GFLOP/s |
| ... this build, scalar | f64 1.9, f32 1.9 GFLOP/s |
| bandwidth | read 43.1, update 53.3, triad 39.0 GB/s (ceiling 53.3) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 25% | 110% | 0 |
| blas2 | 10 | 73% | 89% | 0 |
| blas3 | 10 | 70% | 92% | 0 |
| factor | 12 | 13% | 23% | 0 |
| eigen | 8 | 17% | 54% | 0 |
| elementwise | 5 | 7% | 12% | 0 |
| sparse | 5 | 25% | 57% | 0 |
| direct | 19 | 1% | 20% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 13% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 35% | 36% | 0 |
| mg | 1 | 33% | 33% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 13% | 22% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 9.546 ms | 1.76 | 14.06 | 26% | ok |
| blas1 | dot | f32 | n=8388608 | 10.727 ms | 1.56 | 6.26 | 12% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.060 ms | 1.39 | 16.69 | 31% | ok |
| blas1 | axpy | f32 | n=8388608 | 12.027 ms | 1.39 | 8.37 | 16% | ok |
| blas1 | scale | f64 | n=8388608 | 2.296 ms | 3.65 | 58.45 | 110% | ok |
| blas1 | two_norm | f64 | n=8388608 | 9.852 ms | 1.70 | 6.81 | 13% | ok |
| blas1 | one_norm | f64 | n=8388608 | 5.065 ms | 1.66 | 13.25 | 25% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 5.089 ms | 1.65 | 13.19 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 5.065 ms | 1.66 | 13.25 | 25% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 81.623 ms | 2.26 | 1.64 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.464 ms | 9.69 | 38.76 | 73% | ok |
| blas2 | gemv | f32 | n=4096 | 1.577 ms | 21.27 | 42.56 | 80% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.721 ms | 9.02 | 36.08 | 68% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.409 ms | 23.81 | 47.65 | 89% | ok |
| blas2 | symv | f64 | n=4096 | 1.827 ms | 18.36 | 36.76 | 69% | ok |
| blas2 | trmv | f64 | n=4096 | 1.785 ms | 9.40 | 37.63 | 71% | ok |
| blas2 | trmv | f32 | n=4096 | 0.898 ms | 18.69 | 37.42 | 70% | ok |
| blas2 | trsv | f64 | n=4096 | 1.592 ms | 10.54 | 42.19 | 79% | ok |
| blas2 | trsv | f32 | n=4096 | 1.013 ms | 16.56 | 33.14 | 62% | ok |
| blas2 | ger | f64 | n=4096 | 6.366 ms | 5.27 | 42.18 | 79% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.657 ms | 73.40 | 1.72 | 92% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.906 ms | 140.83 | 1.65 | 88% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.740 ms | 48.98 | 2.30 | 61% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.795 ms | 70.72 | 1.66 | 88% | ok |
| blas3 | syrk | f64 | 512x512x512 | 2.950 ms | 45.49 | 1.42 | 57% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 4.770 ms | 56.28 | 1.32 | 70% | ok |
| blas3 | trmm | f64 | 512x512x512 | 1.898 ms | 70.72 | 2.76 | 88% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.845 ms | 47.18 | 1.84 | 59% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.344 ms | 99.88 | 1.95 | 62% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 8.849 ms | 0.60 | 4.80 | 9% | ok |
| factor | lu | f64 | 512x512 | 4.954 ms | 18.06 | 0.85 | 23% | ok |
| factor | lu | f32 | 512x512 | 2.456 ms | 36.43 | 0.85 | 23% | ok |
| factor | cholesky | f64 | 512x512 | 4.338 ms | 10.31 | 0.97 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 1.710 ms | 26.16 | 1.23 | 16% | ok |
| factor | ldlt | f64 | 512x512 | 11.742 ms | 3.81 | 0.36 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 10.602 ms | 4.22 | 0.40 | 5% | ok |
| factor | qr | f64 | 512x512 | 10.323 ms | 17.33 | 0.41 | 22% | ok |
| factor | lq | f64 | 512x512 | 64.713 ms | 2.77 | 0.06 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 742.697 ms | 0.60 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 3.774 ms | 0.28 | 1.11 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 61.261 ms | 4.38 | 0.07 | 5% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 5.518 ms | 16.22 | 0.76 | 20% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.697 ms | 13.18 | 0.62 | 16% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 130.519 ms | 1.16 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 513.407 ms | 0.29 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 16.889 ms | 9.93 | 0.06 | 12% | ok |
| eigen | schur | f64 | 256x256 | 22.149 ms | 18.94 | 0.05 | 24% | ok |
| eigen | svd | f64 | 256x256 | 8.602 ms | 42.91 | 0.12 | 54% | ok |
| eigen | expm | f64 | 256x256 | 15.544 ms | 16.19 | 0.07 | 20% | ok |
| eigen | sylvester | f64 | 256x256 | 73.573 ms | 13.68 | 0.02 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 27.086 ms | 0.31 | 4.96 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 42.518 ms | 0.20 | 3.16 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 38.474 ms | 0.22 | 3.49 | 7% | ok |
| elementwise | sqrt | f64 | n=8388608 | 20.682 ms | 0.41 | 6.49 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 33.999 ms | 0.25 | 3.95 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.897 ms | 2.92 | 30.37 | 57% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.296 ms | 2.02 | 21.00 | 39% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.636 ms | 6.31 | 13.27 | 25% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.142 ms | 0.20 | 0.71 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.695 ms | 1.18 | 9.82 | 18%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 18.973 ms | 0.59 | 0.04 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 30.757 ms | 0.36 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 14.853 ms | 0.75 | 0.05 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 53.152 ms | 0.42 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 57.277 ms | 0.39 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 24.544 ms | 0.91 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 37.369 ms | 0.05 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.668 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.610 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 60.829 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.230 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.433 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.942 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.349 ms | 0.78 | 0.05 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.292 ms | 2.59 | 10.36 | 19%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.591 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 5.663 ms | 1.98 | 0.13 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.277 ms | 2.73 | 10.90 | 20%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.302 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 46.960 ms | 0.48 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.146 ms | 0.66 | 2.64 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.747 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.628 ms | 0.46 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.128 ms | 0.67 | 2.68 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.582 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 255.858 ms | 1.21 | 0.00 | 2% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.986 ms | 1.70 | 6.79 | 13%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 13.622 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.690 ms | 4.43 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.022 ms | 1.67 | 6.66 | 13%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.063 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 901.880 ms | 0.68 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.496 ms | 0.32 | 1.28 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.438 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 509.710 ms | 1.21 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.489 ms | 0.32 | 1.28 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 39.038 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 58.322 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 31.899 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 16.821 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 21.911 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.798 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 54.276 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.471 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 19.897 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 26.209 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 24.795 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 25.593 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 33.284 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.044 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.395 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 23.106 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.816 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.751 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.030 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.341 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.566 ms | 1.67 | 18.73 | 35% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.545 ms | 1.69 | 18.98 | 36% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.797 ms | 1.46 | 16.32 | 31% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 16.330 ms | 2.35 | 17.80 | 33% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.721 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.891 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 52.104 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.819 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 9.818 ms | 1.71 | 6.84 | 13% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 29.556 ms | 9.08 | 0.14 | 11% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.487 ms | 11.28 | 11.99 | 22% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.311 ms | 5.07 | 20.27 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.876 ms | 5.83 | 11.67 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.391 ms | 12.06 | 12.06 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.392 ms | 12.06 | 12.06 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 8.442 ms | 31.80 | 0.37 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 30.468 ms | 8.81 | 0.07 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 27.401 ms | 9.80 | 0.06 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 27.415 ms | 9.79 | 0.06 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 27.405 ms | 9.80 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

