## ryzen-9-8945hs -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 79.0 GFLOP/s, f32 126.4 GFLOP/s |
| ... hardware, AVX-512F | f64 79.0, f32 126.4 GFLOP/s |
| ... this build, scalar | f64 1.9, f32 1.9 GFLOP/s |
| bandwidth | read 36.9, update 51.8, triad 37.8 GB/s (ceiling 51.8) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 20% | 98% | 0 |
| blas2 | 10 | 71% | 85% | 0 |
| blas3 | 10 | 65% | 89% | 0 |
| factor | 12 | 10% | 23% | 0 |
| eigen | 8 | 14% | 43% | 0 |
| elementwise | 5 | 6% | 10% | 0 |
| sparse | 5 | 18% | 57% | 0 |
| direct | 19 | 1% | 17% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 12% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 29% | 31% | 0 |
| mg | 1 | 28% | 28% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 11% | 19% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.232 ms | 1.49 | 11.95 | 23% | ok |
| blas1 | dot | f32 | n=8388608 | 12.256 ms | 1.37 | 5.48 | 11% | ok |
| blas1 | axpy | f64 | n=8388608 | 14.984 ms | 1.12 | 13.44 | 26% | ok |
| blas1 | axpy | f32 | n=8388608 | 14.940 ms | 1.12 | 6.74 | 13% | ok |
| blas1 | scale | f64 | n=8388608 | 2.638 ms | 3.18 | 50.87 | 98% | ok |
| blas1 | two_norm | f64 | n=8388608 | 12.260 ms | 1.37 | 5.47 | 11% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.394 ms | 1.31 | 10.49 | 20% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.413 ms | 1.31 | 10.47 | 20% | ok |
| blas1 | sum | f64 | n=8388608 | 6.397 ms | 1.31 | 10.49 | 20% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 111.558 ms | 1.65 | 1.20 | 2% | ok |
| blas2 | gemv | f64 | n=4096 | 3.645 ms | 9.21 | 36.84 | 71% | ok |
| blas2 | gemv | f32 | n=4096 | 1.520 ms | 22.08 | 44.18 | 85% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 4.216 ms | 7.96 | 31.85 | 61% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.558 ms | 21.54 | 43.11 | 83% | ok |
| blas2 | symv | f64 | n=4096 | 1.925 ms | 17.43 | 34.90 | 67% | ok |
| blas2 | trmv | f64 | n=4096 | 1.825 ms | 9.19 | 36.81 | 71% | ok |
| blas2 | trmv | f32 | n=4096 | 1.083 ms | 15.50 | 31.03 | 60% | ok |
| blas2 | trsv | f64 | n=4096 | 1.773 ms | 9.46 | 37.89 | 73% | ok |
| blas2 | trsv | f32 | n=4096 | 0.983 ms | 17.07 | 34.18 | 66% | ok |
| blas2 | ger | f64 | n=4096 | 6.639 ms | 5.05 | 40.44 | 78% | ok |
| blas3 | gemm | f64 | 512x512x512 | 4.573 ms | 58.70 | 1.38 | 74% | ok |
| blas3 | gemm | f32 | 512x512x512 | 2.386 ms | 112.52 | 1.32 | 89% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 3.421 ms | 39.24 | 1.84 | 50% | ok |
| blas3 | symm | f64 | 512x512x512 | 4.751 ms | 56.50 | 1.32 | 72% | ok |
| blas3 | syrk | f64 | 512x512x512 | 3.702 ms | 36.26 | 1.13 | 46% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 5.996 ms | 44.77 | 1.05 | 57% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.362 ms | 56.83 | 2.22 | 72% | ok |
| blas3 | trsm | f64 | 512x512x512 | 3.501 ms | 38.34 | 1.50 | 49% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.639 ms | 81.92 | 1.60 | 65% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.867 ms | 0.49 | 3.91 | 8% | ok |
| factor | lu | f64 | 512x512 | 6.332 ms | 14.13 | 0.66 | 18% | ok |
| factor | lu | f32 | 512x512 | 3.091 ms | 28.95 | 0.68 | 23% | ok |
| factor | cholesky | f64 | 512x512 | 5.469 ms | 8.18 | 0.77 | 10% | ok |
| factor | cholesky | f32 | 512x512 | 2.162 ms | 20.70 | 0.97 | 16% | ok |
| factor | ldlt | f64 | 512x512 | 14.783 ms | 3.03 | 0.28 | 4% | ok |
| factor | ldlt_bk | f64 | 512x512 | 12.812 ms | 3.49 | 0.33 | 4% | ok |
| factor | qr | f64 | 512x512 | 12.955 ms | 13.81 | 0.32 | 17% | ok |
| factor | lq | f64 | 512x512 | 80.638 ms | 2.22 | 0.05 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 935.701 ms | 0.48 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.933 ms | 0.21 | 0.85 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 76.131 ms | 3.53 | 0.06 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.937 ms | 12.90 | 0.60 | 16% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 2.178 ms | 10.27 | 0.48 | 13% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 160.457 ms | 0.94 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 647.134 ms | 0.23 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 21.106 ms | 7.95 | 0.05 | 10% | ok |
| eigen | schur | f64 | 256x256 | 27.364 ms | 15.33 | 0.04 | 19% | ok |
| eigen | svd | f64 | 256x256 | 10.875 ms | 33.94 | 0.10 | 43% | ok |
| eigen | expm | f64 | 256x256 | 19.522 ms | 12.89 | 0.05 | 16% | ok |
| eigen | sylvester | f64 | 256x256 | 90.189 ms | 11.16 | 0.02 | 14% | ok |
| elementwise | exp | f64 | n=8388608 | 33.413 ms | 0.25 | 4.02 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 53.140 ms | 0.16 | 2.53 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 47.901 ms | 0.18 | 2.80 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 26.476 ms | 0.32 | 5.07 | 10% | ok |
| elementwise | erf | f64 | n=8388608 | 45.035 ms | 0.19 | 2.98 | 6% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.927 ms | 2.82 | 29.36 | 57% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.536 ms | 1.70 | 17.72 | 34% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 9.686 ms | 4.32 | 9.09 | 18% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 21.385 ms | 0.15 | 0.54 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.914 ms | 1.05 | 8.70 | 17%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 23.761 ms | 0.47 | 0.03 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 38.876 ms | 0.29 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 19.606 ms | 0.57 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 70.147 ms | 0.32 | 0.01 | 0% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 69.874 ms | 0.32 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 29.809 ms | 0.75 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 42.799 ms | 0.04 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.571 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.933 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 75.003 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.245 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.715 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.604 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.602 ms | 0.64 | 0.04 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.348 ms | 2.18 | 8.70 | 17%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.055 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.203 ms | 1.55 | 0.10 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.366 ms | 2.06 | 8.25 | 16%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.097 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.023 ms | 0.38 | 0.01 | 0% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.229 ms | 0.61 | 2.46 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 5.891 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.536 ms | 0.38 | 0.01 | 0% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.210 ms | 0.63 | 2.50 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 6.963 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 337.330 ms | 0.91 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.216 ms | 1.52 | 6.08 | 12%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 17.161 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 85.624 ms | 3.60 | 0.01 | 5% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.278 ms | 1.48 | 5.92 | 11%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.591 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1109.012 ms | 0.56 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.267 ms | 0.30 | 1.20 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.522 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 626.061 ms | 0.99 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.072 ms | 0.30 | 1.22 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 49.123 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 73.650 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 39.768 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.025 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 27.378 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.402 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 66.670 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 43.524 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.004 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 32.893 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 31.010 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 31.962 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 42.048 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.055 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.933 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 27.100 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.241 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.159 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.036 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.661 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.842 ms | 1.42 | 15.92 | 31% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.962 ms | 1.33 | 14.95 | 29% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.237 ms | 1.17 | 13.11 | 25% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 19.807 ms | 1.94 | 14.67 | 28% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 40.547 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.322 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 67.663 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.813 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 12.258 ms | 1.37 | 5.47 | 11% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 37.330 ms | 7.19 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.835 ms | 9.14 | 9.72 | 19% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.999 ms | 4.20 | 16.78 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.247 ms | 5.17 | 10.33 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.122 ms | 7.91 | 7.91 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.122 ms | 7.91 | 7.91 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 10.216 ms | 26.28 | 0.31 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 39.455 ms | 6.80 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 34.554 ms | 7.77 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 34.618 ms | 7.75 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 34.482 ms | 7.78 | 0.05 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

