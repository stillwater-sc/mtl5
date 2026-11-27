## ryzen-7-8845hs -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `57512adcb8fe` |
| FMA peak (ceiling) | f64 80.0 GFLOP/s, f32 160.1 GFLOP/s |
| ... hardware, AVX-512F | f64 80.0, f32 160.1 GFLOP/s |
| ... this build, scalar | f64 1.6, f32 1.9 GFLOP/s |
| bandwidth | read 42.5, update 53.3, triad 38.7 GB/s (ceiling 53.3) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 25% | 108% | 0 |
| blas2 | 10 | 72% | 85% | 0 |
| blas3 | 10 | 71% | 92% | 0 |
| factor | 12 | 13% | 23% | 0 |
| eigen | 8 | 17% | 53% | 0 |
| elementwise | 5 | 7% | 12% | 0 |
| sparse | 5 | 18% | 59% | 0 |
| direct | 19 | 1% | 19% | 0 |
| ordering | 5 | n/a | n/a | 0 |
| direct3d | 12 | 2% | 13% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 35% | 37% | 0 |
| mg | 1 | 34% | 34% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 11% | 22% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 11.211 ms | 1.50 | 11.97 | 22% | ok |
| blas1 | dot | f32 | n=8388608 | 9.859 ms | 1.70 | 6.81 | 13% | ok |
| blas1 | axpy | f64 | n=8388608 | 12.043 ms | 1.39 | 16.72 | 31% | ok |
| blas1 | axpy | f32 | n=8388608 | 10.366 ms | 1.62 | 9.71 | 18% | ok |
| blas1 | scale | f64 | n=8388608 | 2.322 ms | 3.61 | 57.79 | 108% | ok |
| blas1 | two_norm | f64 | n=8388608 | 11.483 ms | 1.46 | 5.84 | 11% | ok |
| blas1 | one_norm | f64 | n=8388608 | 5.066 ms | 1.66 | 13.25 | 25% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 5.074 ms | 1.65 | 13.23 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 5.065 ms | 1.66 | 13.25 | 25% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 89.415 ms | 2.06 | 1.50 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.490 ms | 9.61 | 38.48 | 72% | ok |
| blas2 | gemv | f32 | n=4096 | 1.563 ms | 21.47 | 42.97 | 81% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.765 ms | 8.91 | 35.67 | 67% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.486 ms | 22.59 | 45.20 | 85% | ok |
| blas2 | symv | f64 | n=4096 | 1.936 ms | 17.33 | 34.70 | 65% | ok |
| blas2 | trmv | f64 | n=4096 | 1.781 ms | 9.42 | 37.72 | 71% | ok |
| blas2 | trmv | f32 | n=4096 | 0.943 ms | 17.79 | 35.62 | 67% | ok |
| blas2 | trsv | f64 | n=4096 | 1.754 ms | 9.56 | 38.29 | 72% | ok |
| blas2 | trsv | f32 | n=4096 | 0.851 ms | 19.72 | 39.48 | 74% | ok |
| blas2 | ger | f64 | n=4096 | 6.380 ms | 5.26 | 42.09 | 79% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.650 ms | 73.54 | 1.72 | 92% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.901 ms | 141.22 | 1.65 | 88% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.743 ms | 48.92 | 2.29 | 61% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.796 ms | 70.71 | 1.66 | 88% | ok |
| blas3 | syrk | f64 | 512x512x512 | 3.013 ms | 44.55 | 1.39 | 56% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 4.757 ms | 56.42 | 1.32 | 71% | ok |
| blas3 | trmm | f64 | 512x512x512 | 1.898 ms | 70.71 | 2.76 | 88% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.853 ms | 47.05 | 1.84 | 59% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.345 ms | 99.79 | 1.95 | 62% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 8.813 ms | 0.60 | 4.82 | 9% | ok |
| factor | lu | f64 | 512x512 | 4.970 ms | 18.01 | 0.84 | 23% | ok |
| factor | lu | f32 | 512x512 | 2.422 ms | 36.95 | 0.87 | 23% | ok |
| factor | cholesky | f64 | 512x512 | 4.336 ms | 10.32 | 0.97 | 13% | ok |
| factor | cholesky | f32 | 512x512 | 1.663 ms | 26.91 | 1.26 | 17% | ok |
| factor | ldlt | f64 | 512x512 | 11.748 ms | 3.81 | 0.36 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 10.597 ms | 4.22 | 0.40 | 5% | ok |
| factor | qr | f64 | 512x512 | 10.249 ms | 17.46 | 0.41 | 22% | ok |
| factor | lq | f64 | 512x512 | 65.219 ms | 2.74 | 0.06 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 741.440 ms | 0.60 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 3.880 ms | 0.27 | 1.08 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 60.082 ms | 4.47 | 0.07 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 5.512 ms | 16.23 | 0.76 | 20% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.694 ms | 13.21 | 0.62 | 17% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 129.651 ms | 1.16 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 521.773 ms | 0.29 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 17.033 ms | 9.85 | 0.06 | 12% | ok |
| eigen | schur | f64 | 256x256 | 22.193 ms | 18.90 | 0.05 | 24% | ok |
| eigen | svd | f64 | 256x256 | 8.665 ms | 42.60 | 0.12 | 53% | ok |
| eigen | expm | f64 | 256x256 | 15.504 ms | 16.23 | 0.07 | 20% | ok |
| eigen | sylvester | f64 | 256x256 | 74.783 ms | 13.46 | 0.02 | 17% | ok |
| elementwise | exp | f64 | n=8388608 | 27.020 ms | 0.31 | 4.97 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 42.417 ms | 0.20 | 3.16 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 38.297 ms | 0.22 | 3.50 | 7% | ok |
| elementwise | sqrt | f64 | n=8388608 | 20.729 ms | 0.40 | 6.47 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 34.156 ms | 0.25 | 3.93 | 7% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.862 ms | 3.04 | 31.60 | 59% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 5.928 ms | 0.44 | 4.59 | 9% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.574 ms | 6.37 | 13.39 | 25% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 16.852 ms | 0.19 | 0.68 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.691 ms | 1.19 | 9.84 | 18%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 18.983 ms | 0.59 | 0.04 | 1% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 30.617 ms | 0.37 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 14.223 ms | 0.79 | 0.05 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 53.170 ms | 0.42 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 56.183 ms | 0.40 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 24.888 ms | 0.90 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 37.943 ms | 0.05 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.681 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.621 ms | 0.00 | 0.00 | n/a | ok |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 60.272 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.226 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.234 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.933 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.234 ms | 0.79 | 0.05 | 1% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.294 ms | 2.57 | 10.28 | 19%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.588 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 5.721 ms | 1.96 | 0.13 | 2% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.291 ms | 2.60 | 10.38 | 19%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.293 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 46.818 ms | 0.48 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.131 ms | 0.67 | 2.67 | 5%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.764 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.765 ms | 0.46 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.098 ms | 0.69 | 2.75 | 5%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.608 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 261.770 ms | 1.18 | 0.00 | 1% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.966 ms | 1.71 | 6.85 | 13%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 14.012 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.454 ms | 4.44 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.019 ms | 1.67 | 6.68 | 13%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.072 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 905.149 ms | 0.68 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.541 ms | 0.32 | 1.28 | 2%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.427 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 512.274 ms | 1.20 | 0.00 | 2% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.565 ms | 0.32 | 1.28 | 2%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 40.275 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 83.924 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 32.133 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.339 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.426 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 18.096 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 58.085 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 35.334 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 20.225 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 27.475 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 25.988 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 27.098 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 35.688 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.044 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.436 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 22.168 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.823 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.759 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.024 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.343 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.571 ms | 1.67 | 18.67 | 35% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.493 ms | 1.75 | 19.64 | 37% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.891 ms | 1.38 | 15.51 | 29% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.938 ms | 2.41 | 18.24 | 34% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.347 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.940 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 52.860 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.197 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 11.467 ms | 1.46 | 5.85 | 11% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 29.556 ms | 9.08 | 0.14 | 11% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.489 ms | 11.27 | 11.97 | 22% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.204 ms | 5.24 | 20.95 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.864 ms | 5.86 | 11.72 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.391 ms | 12.06 | 12.06 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.390 ms | 12.07 | 12.07 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 8.397 ms | 31.97 | 0.37 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 30.447 ms | 8.82 | 0.07 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 27.407 ms | 9.79 | 0.06 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 27.408 ms | 9.79 | 0.06 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 27.388 ms | 9.80 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

Skipped in this build:

- CHOLMOD not found when this build was configured: `direct/ref_cholmod/f64`, `direct/ref_cholmod.analyze/f64`, `direct/ref_cholmod.factor/f64`, `direct/ref_cholmod.solve/f64`, `direct3d/ref_cholmod.analyze/f64`, `direct3d/ref_cholmod.factor/f64`, `direct3d/ref_cholmod.solve/f64`
- MKL PARDISO is loaded only on Linux and macOS: `direct/ref_pardiso_spd/f64`, `direct/ref_pardiso_spd.analyze/f64`, `direct/ref_pardiso_spd.factor/f64`, `direct/ref_pardiso_spd.solve/f64`, `direct/ref_pardiso_lu/f64`, `direct/ref_pardiso_lu.analyze/f64`, `direct/ref_pardiso_lu.factor/f64`, `direct/ref_pardiso_lu.solve/f64`, `direct3d/ref_pardiso_spd.analyze/f64`, `direct3d/ref_pardiso_spd.factor/f64`, `direct3d/ref_pardiso_spd.solve/f64`, `direct3d/ref_pardiso_lu.analyze/f64`, `direct3d/ref_pardiso_lu.factor/f64`, `direct3d/ref_pardiso_lu.solve/f64`

