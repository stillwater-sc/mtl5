## ryzen-7-8845hs -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 80.1 GFLOP/s, f32 160.1 GFLOP/s |
| ... hardware, AVX-512F | f64 80.1, f32 160.1 GFLOP/s |
| ... this build, scalar | f64 1.9, f32 1.9 GFLOP/s |
| bandwidth | read 43.1, update 53.0, triad 38.8 GB/s (ceiling 53.0) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 25% | 111% | 0 |
| blas2 | 10 | 79% | 90% | 0 |
| blas3 | 10 | 69% | 92% | 0 |
| factor | 12 | 13% | 23% | 0 |
| eigen | 8 | 17% | 53% | 0 |
| elementwise | 5 | 7% | 12% | 0 |
| sparse | 5 | 25% | 60% | 0 |
| direct | 19 | 1% | 20% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 13% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 31% | 37% | 0 |
| mg | 1 | 34% | 34% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 13% | 23% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 9.517 ms | 1.76 | 14.10 | 27% | ok |
| blas1 | dot | f32 | n=8388608 | 9.822 ms | 1.71 | 6.83 | 13% | ok |
| blas1 | axpy | f64 | n=8388608 | 11.989 ms | 1.40 | 16.79 | 32% | ok |
| blas1 | axpy | f32 | n=8388608 | 11.951 ms | 1.40 | 8.42 | 16% | ok |
| blas1 | scale | f64 | n=8388608 | 2.287 ms | 3.67 | 58.68 | 111% | ok |
| blas1 | two_norm | f64 | n=8388608 | 9.797 ms | 1.71 | 6.85 | 13% | ok |
| blas1 | one_norm | f64 | n=8388608 | 5.063 ms | 1.66 | 13.25 | 25% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 5.073 ms | 1.65 | 13.23 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 5.062 ms | 1.66 | 13.26 | 25% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 89.533 ms | 2.06 | 1.50 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.478 ms | 9.65 | 38.61 | 73% | ok |
| blas2 | gemv | f32 | n=4096 | 1.424 ms | 23.56 | 47.14 | 89% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.213 ms | 10.44 | 41.79 | 79% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.400 ms | 23.96 | 47.95 | 90% | ok |
| blas2 | symv | f64 | n=4096 | 1.818 ms | 18.46 | 36.95 | 70% | ok |
| blas2 | trmv | f64 | n=4096 | 1.765 ms | 9.50 | 38.06 | 72% | ok |
| blas2 | trmv | f32 | n=4096 | 0.989 ms | 16.97 | 33.98 | 64% | ok |
| blas2 | trsv | f64 | n=4096 | 1.583 ms | 10.60 | 42.45 | 80% | ok |
| blas2 | trsv | f32 | n=4096 | 0.865 ms | 19.41 | 38.85 | 73% | ok |
| blas2 | ger | f64 | n=4096 | 6.316 ms | 5.31 | 42.51 | 80% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.640 ms | 73.75 | 1.73 | 92% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.900 ms | 141.31 | 1.66 | 88% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.731 ms | 49.15 | 2.30 | 61% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.782 ms | 70.97 | 1.66 | 89% | ok |
| blas3 | syrk | f64 | 512x512x512 | 2.936 ms | 45.71 | 1.43 | 57% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 4.829 ms | 55.58 | 1.30 | 69% | ok |
| blas3 | trmm | f64 | 512x512x512 | 1.889 ms | 71.06 | 2.78 | 89% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.833 ms | 47.37 | 1.85 | 59% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.341 ms | 100.10 | 1.96 | 63% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.114 ms | 0.58 | 4.66 | 9% | ok |
| factor | lu | f64 | 512x512 | 4.980 ms | 17.97 | 0.84 | 22% | ok |
| factor | lu | f32 | 512x512 | 2.473 ms | 36.18 | 0.85 | 23% | ok |
| factor | cholesky | f64 | 512x512 | 4.362 ms | 10.26 | 0.96 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 1.720 ms | 26.01 | 1.22 | 16% | ok |
| factor | ldlt | f64 | 512x512 | 11.767 ms | 3.80 | 0.36 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 10.351 ms | 4.32 | 0.41 | 5% | ok |
| factor | qr | f64 | 512x512 | 10.194 ms | 17.55 | 0.41 | 22% | ok |
| factor | lq | f64 | 512x512 | 64.495 ms | 2.77 | 0.07 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 747.227 ms | 0.60 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 3.870 ms | 0.27 | 1.08 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 60.075 ms | 4.47 | 0.07 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 5.511 ms | 16.24 | 0.76 | 20% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.702 ms | 13.15 | 0.62 | 16% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 131.100 ms | 1.15 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 494.349 ms | 0.31 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 16.977 ms | 9.88 | 0.06 | 12% | ok |
| eigen | schur | f64 | 256x256 | 22.001 ms | 19.06 | 0.05 | 24% | ok |
| eigen | svd | f64 | 256x256 | 8.646 ms | 42.69 | 0.12 | 53% | ok |
| eigen | expm | f64 | 256x256 | 15.568 ms | 16.16 | 0.07 | 20% | ok |
| eigen | sylvester | f64 | 256x256 | 73.102 ms | 13.77 | 0.02 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 27.668 ms | 0.30 | 4.85 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 43.064 ms | 0.19 | 3.12 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 38.982 ms | 0.22 | 3.44 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 21.355 ms | 0.39 | 6.28 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 34.487 ms | 0.24 | 3.89 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.856 ms | 3.06 | 31.81 | 60% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.298 ms | 2.02 | 20.98 | 40% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.617 ms | 6.33 | 13.31 | 25% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 17.916 ms | 0.18 | 0.64 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.701 ms | 1.18 | 9.79 | 18%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 19.374 ms | 0.58 | 0.04 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 30.979 ms | 0.36 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 15.469 ms | 0.72 | 0.05 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 53.705 ms | 0.42 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 57.279 ms | 0.39 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 24.662 ms | 0.91 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 36.040 ms | 0.05 | 0.01 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.638 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.633 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 60.805 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.228 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.527 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.932 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.106 ms | 0.79 | 0.05 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.292 ms | 2.59 | 10.36 | 20%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.852 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 6.013 ms | 1.86 | 0.12 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.290 ms | 2.60 | 10.42 | 20%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.355 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.025 ms | 0.47 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.148 ms | 0.66 | 2.63 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.788 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 49.118 ms | 0.46 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.115 ms | 0.68 | 2.71 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.590 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 263.897 ms | 1.17 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.984 ms | 1.70 | 6.79 | 13%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 13.918 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.158 ms | 4.46 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.032 ms | 1.66 | 6.63 | 13%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.168 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 893.966 ms | 0.69 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.405 ms | 0.32 | 1.30 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.505 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 511.891 ms | 1.21 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.518 ms | 0.32 | 1.28 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 38.978 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 58.235 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 31.838 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 16.678 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 21.747 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.749 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 53.903 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 34.510 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 19.817 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 26.095 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 24.558 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 25.504 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 33.311 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.043 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.379 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 21.845 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.809 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.784 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.029 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.327 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.515 ms | 1.73 | 19.36 | 37% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.775 ms | 1.47 | 16.53 | 31% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.796 ms | 1.46 | 16.33 | 31% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 16.106 ms | 2.39 | 18.05 | 34% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.367 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 33.099 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 52.271 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.772 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 9.765 ms | 1.72 | 6.87 | 13% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 29.601 ms | 9.07 | 0.14 | 11% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.477 ms | 11.36 | 12.07 | 23% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.260 ms | 5.15 | 20.59 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.859 ms | 5.87 | 11.74 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.392 ms | 12.06 | 12.06 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.390 ms | 12.07 | 12.07 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 8.464 ms | 31.71 | 0.37 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 30.281 ms | 8.86 | 0.07 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 27.280 ms | 9.84 | 0.06 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 27.279 ms | 9.84 | 0.06 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 27.269 ms | 9.84 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

