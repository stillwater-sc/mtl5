## ryzen-9-8945hs -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `57512adcb8fe` |
| FMA peak (ceiling) | f64 64.2 GFLOP/s, f32 136.0 GFLOP/s |
| ... hardware, AVX-512F | f64 64.2, f32 136.0 GFLOP/s |
| ... this build, scalar | f64 1.6, f32 1.9 GFLOP/s |
| bandwidth | read 36.5, update 52.0, triad 37.7 GB/s (ceiling 52.0) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 20% | 57% | 0 |
| blas2 | 10 | 75% | 88% | 0 |
| blas3 | 10 | 68% | 91% | 0 |
| factor | 12 | 13% | 22% | 0 |
| eigen | 8 | 17% | 53% | 0 |
| elementwise | 5 | 6% | 10% | 0 |
| sparse | 5 | 21% | 56% | 0 |
| direct | 19 | 1% | 17% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 12% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 27% | 33% | 0 |
| mg | 1 | 25% | 25% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 11% | 19% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 13.737 ms | 1.22 | 9.77 | 19% | ok |
| blas1 | dot | f32 | n=8388608 | 12.235 ms | 1.37 | 5.49 | 11% | ok |
| blas1 | axpy | f64 | n=8388608 | 14.982 ms | 1.12 | 13.44 | 26% | ok |
| blas1 | axpy | f32 | n=8388608 | 12.800 ms | 1.31 | 7.86 | 15% | ok |
| blas1 | scale | f64 | n=8388608 | 4.498 ms | 1.87 | 29.84 | 57% | ok |
| blas1 | two_norm | f64 | n=8388608 | 14.550 ms | 1.15 | 4.61 | 9% | ok |
| blas1 | one_norm | f64 | n=8388608 | 6.402 ms | 1.31 | 10.48 | 20% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 6.417 ms | 1.31 | 10.46 | 20% | ok |
| blas1 | sum | f64 | n=8388608 | 6.404 ms | 1.31 | 10.48 | 20% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 112.552 ms | 1.64 | 1.19 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.263 ms | 10.28 | 41.15 | 79% | ok |
| blas2 | gemv | f32 | n=4096 | 1.570 ms | 21.37 | 42.75 | 82% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.435 ms | 9.77 | 39.09 | 75% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.464 ms | 22.92 | 45.86 | 88% | ok |
| blas2 | symv | f64 | n=4096 | 2.112 ms | 15.88 | 31.80 | 61% | ok |
| blas2 | trmv | f64 | n=4096 | 1.900 ms | 8.83 | 35.36 | 68% | ok |
| blas2 | trmv | f32 | n=4096 | 0.999 ms | 16.80 | 33.63 | 65% | ok |
| blas2 | trsv | f64 | n=4096 | 1.921 ms | 8.73 | 34.97 | 67% | ok |
| blas2 | trsv | f32 | n=4096 | 1.083 ms | 15.49 | 31.01 | 60% | ok |
| blas2 | ger | f64 | n=4096 | 6.630 ms | 5.06 | 40.50 | 78% | ok |
| blas3 | gemm | f64 | 512x512x512 | 4.577 ms | 58.65 | 1.37 | 91% | ok |
| blas3 | gemm | f32 | 512x512x512 | 2.377 ms | 112.94 | 1.32 | 83% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 3.423 ms | 39.22 | 1.84 | 61% | ok |
| blas3 | symm | f64 | 512x512x512 | 4.756 ms | 56.44 | 1.32 | 88% | ok |
| blas3 | syrk | f64 | 512x512x512 | 3.718 ms | 36.10 | 1.13 | 56% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 6.107 ms | 43.96 | 1.03 | 68% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.361 ms | 56.85 | 2.22 | 89% | ok |
| blas3 | trsm | f64 | 512x512x512 | 3.501 ms | 38.34 | 1.50 | 60% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.644 ms | 81.66 | 1.59 | 60% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.495 ms | 0.51 | 4.05 | 8% | ok |
| factor | lu | f64 | 512x512 | 6.330 ms | 14.14 | 0.66 | 22% | ok |
| factor | lu | f32 | 512x512 | 3.039 ms | 29.44 | 0.69 | 22% | ok |
| factor | cholesky | f64 | 512x512 | 5.460 ms | 8.19 | 0.77 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 2.087 ms | 21.44 | 1.01 | 16% | ok |
| factor | ldlt | f64 | 512x512 | 14.818 ms | 3.02 | 0.28 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 12.944 ms | 3.46 | 0.32 | 5% | ok |
| factor | qr | f64 | 512x512 | 12.907 ms | 13.87 | 0.32 | 22% | ok |
| factor | lq | f64 | 512x512 | 81.412 ms | 2.20 | 0.05 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 936.525 ms | 0.48 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.910 ms | 0.21 | 0.85 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 77.050 ms | 3.48 | 0.05 | 5% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.954 ms | 12.87 | 0.60 | 20% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 2.165 ms | 10.33 | 0.48 | 16% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 165.385 ms | 0.91 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 619.665 ms | 0.24 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 20.874 ms | 8.04 | 0.05 | 13% | ok |
| eigen | schur | f64 | 256x256 | 27.233 ms | 15.40 | 0.04 | 24% | ok |
| eigen | svd | f64 | 256x256 | 10.871 ms | 33.95 | 0.10 | 53% | ok |
| eigen | expm | f64 | 256x256 | 19.572 ms | 12.86 | 0.05 | 20% | ok |
| eigen | sylvester | f64 | 256x256 | 93.066 ms | 10.82 | 0.02 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 32.660 ms | 0.26 | 4.11 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 52.815 ms | 0.16 | 2.54 | 5% | ok |
| elementwise | sin | f64 | n=8388608 | 48.000 ms | 0.17 | 2.80 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 26.007 ms | 0.32 | 5.16 | 10% | ok |
| elementwise | erf | f64 | n=8388608 | 42.014 ms | 0.20 | 3.19 | 6% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.938 ms | 2.79 | 29.02 | 56% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.534 ms | 1.71 | 17.75 | 34% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 8.086 ms | 5.18 | 10.89 | 21% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 19.651 ms | 0.17 | 0.59 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.921 ms | 1.04 | 8.67 | 17%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 23.781 ms | 0.47 | 0.03 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 38.983 ms | 0.29 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 19.219 ms | 0.58 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 64.649 ms | 0.35 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 69.652 ms | 0.32 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 29.999 ms | 0.75 | 0.02 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 42.887 ms | 0.04 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.542 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.977 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 75.249 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.241 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.767 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 3.590 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 17.551 ms | 0.64 | 0.04 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.349 ms | 2.17 | 8.66 | 17%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 8.040 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 7.168 ms | 1.56 | 0.10 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.351 ms | 2.16 | 8.62 | 17%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.056 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 57.298 ms | 0.39 | 0.01 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.302 ms | 0.58 | 2.32 | 4%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 6.121 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 59.704 ms | 0.37 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.249 ms | 0.61 | 2.42 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 6.960 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 346.051 ms | 0.89 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 2.222 ms | 1.52 | 6.07 | 12%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 16.816 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 84.295 ms | 3.66 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.278 ms | 1.48 | 5.92 | 11%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 7.597 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 1119.007 ms | 0.55 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.229 ms | 0.30 | 1.20 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 10.415 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 635.899 ms | 0.97 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 11.180 ms | 0.30 | 1.21 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 51.016 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 105.781 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 40.390 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.841 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 28.439 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.889 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 72.016 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 44.605 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.629 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 34.788 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 32.972 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 34.118 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 44.471 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.057 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.925 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 28.710 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 2.224 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 2.145 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.029 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.695 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.696 ms | 1.54 | 17.29 | 33% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.126 ms | 1.23 | 13.80 | 27% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.300 ms | 1.14 | 12.75 | 25% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 22.634 ms | 1.70 | 12.84 | 25% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 41.174 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.332 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 67.942 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 10.106 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 14.381 ms | 1.17 | 4.67 | 9% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 37.737 ms | 7.11 | 0.11 | 11% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.836 ms | 9.14 | 9.71 | 19% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.041 ms | 4.15 | 16.61 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.314 ms | 5.06 | 10.13 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 2.122 ms | 7.91 | 7.91 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 2.122 ms | 7.90 | 7.90 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 10.411 ms | 25.78 | 0.30 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 38.426 ms | 6.99 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 34.775 ms | 7.72 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 34.777 ms | 7.72 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 34.653 ms | 7.75 | 0.05 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

