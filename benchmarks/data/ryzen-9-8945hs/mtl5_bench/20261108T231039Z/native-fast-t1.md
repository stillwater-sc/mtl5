## AMD Ryzen 9 8945HS w/ Radeon 780M Graphics -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.51.36247 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `685e2711d5ff` |
| FMA peak (ceiling) | f64 81.9 GFLOP/s, f32 163.5 GFLOP/s |
| ... hardware, AVX-512F | f64 81.9, f32 163.5 GFLOP/s |
| ... this build, SSE2 | f64 40.8, f32 81.7 GFLOP/s |
| bandwidth | read 45.6, update 52.6, triad 36.9 GB/s (ceiling 52.6) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 107% | 114% | 0 |
| blas2 | 6 | 67% | 84% | 0 |
| blas3 | 10 | 16% | 28% | 0 |
| factor | 12 | 3% | 4% | 0 |
| eigen | 8 | 4% | 11% | 0 |
| elementwise | 5 | 8% | 13% | 0 |
| sparse | 5 | 27% | 58% | 0 |
| direct | 7 | 0% | 1% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 33% | 35% | 0 |
| mg | 1 | 30% | 30% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 62% | 65% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 2.731 ms | 6.14 | 49.14 | 93% | ok |
| blas1 | dot | f32 | n=8388608 | 1.194 ms | 14.05 | 56.21 | 107% | ok |
| blas1 | axpy | f64 | n=8388608 | 3.578 ms | 4.69 | 56.26 | 107% | ok |
| blas1 | axpy | f32 | n=8388608 | 1.732 ms | 9.68 | 58.11 | 110% | ok |
| blas1 | scale | f64 | n=8388608 | 2.244 ms | 3.74 | 59.82 | 114% | ok |
| blas1 | two_norm | f64 | n=8388608 | 1.172 ms | 14.31 | 57.26 | 109% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.960 ms | 1.69 | 13.53 | 26% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 5.075 ms | 1.65 | 13.22 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 4.956 ms | 1.69 | 13.54 | 26% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 79.152 ms | 2.33 | 1.70 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.456 ms | 9.71 | 38.86 | 74% | ok |
| blas2 | gemv | f32 | n=4096 | 1.515 ms | 22.15 | 44.31 | 84% | ok |
| blas2 | symv | f64 | n=4096 | 8.353 ms | 4.02 | 8.04 | 15% | ok |
| blas2 | trmv | f64 | n=4096 | 1.908 ms | 8.79 | 35.21 | 67% | ok |
| blas2 | trsv | f64 | n=4096 | 2.018 ms | 8.31 | 33.28 | 63% | ok |
| blas2 | ger | f64 | n=4096 | 13.860 ms | 2.42 | 19.37 | 37% | ok |
| blas3 | gemm | f64 | 512x512x512 | 12.314 ms | 21.80 | 0.51 | 27% | ok |
| blas3 | gemm | f32 | 512x512x512 | 5.885 ms | 45.61 | 0.53 | 28% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 8.087 ms | 16.60 | 0.78 | 20% | ok |
| blas3 | symm | f64 | 512x512x512 | 13.318 ms | 20.16 | 0.47 | 25% | ok |
| blas3 | syrk | f64 | 512x512x512 | 25.299 ms | 5.31 | 0.17 | 6% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 74.490 ms | 3.60 | 0.08 | 4% | ok |
| blas3 | trmm | f64 | 512x512x512 | 10.540 ms | 12.73 | 0.50 | 16% | ok |
| blas3 | trsm | f64 | 512x512x512 | 11.988 ms | 11.20 | 0.44 | 14% | ok |
| blas3 | trsm | f32 | 512x512x512 | 7.170 ms | 18.72 | 0.37 | 11% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.017 ms | 0.59 | 4.71 | 9% | ok |
| factor | lu | f64 | 512x512 | 36.922 ms | 2.42 | 0.11 | 3% | ok |
| factor | lu | f32 | 512x512 | 33.045 ms | 2.71 | 0.06 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 14.106 ms | 3.17 | 0.30 | 4% | ok |
| factor | cholesky | f32 | 512x512 | 10.999 ms | 4.07 | 0.19 | 2% | ok |
| factor | ldlt | f64 | 512x512 | 16.054 ms | 2.79 | 0.26 | 3% | ok |
| factor | ldlt_bk | f64 | 512x512 | 175.339 ms | 0.26 | 0.02 | 0% | ok |
| factor | qr | f64 | 512x512 | 870.002 ms | 0.21 | 0.00 | 0% | ok |
| factor | lq | f64 | 512x512 | 64.553 ms | 2.77 | 0.06 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 796.354 ms | 0.56 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 4.572 ms | 0.23 | 0.92 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 114.951 ms | 2.34 | 0.04 | 3% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 36.150 ms | 2.48 | 0.12 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 41.643 ms | 0.54 | 0.03 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 135.185 ms | 1.12 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 482.850 ms | 0.31 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 60.066 ms | 2.79 | 0.02 | 3% | ok |
| eigen | schur | f64 | 256x256 | 123.523 ms | 3.40 | 0.01 | 4% | ok |
| eigen | svd | f64 | 256x256 | 117.976 ms | 3.13 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 28.847 ms | 8.72 | 0.04 | 11% | ok |
| eigen | sylvester | f64 | 256x256 | 281.932 ms | 3.57 | 0.01 | 4% | ok |
| elementwise | exp | f64 | n=8388608 | 27.447 ms | 0.31 | 4.89 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 42.661 ms | 0.20 | 3.15 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 38.897 ms | 0.22 | 3.45 | 7% | ok |
| elementwise | sqrt | f64 | n=8388608 | 20.155 ms | 0.42 | 6.66 | 13% | ok |
| elementwise | erf | f64 | n=8388608 | 31.318 ms | 0.27 | 4.29 | 8% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.890 ms | 2.94 | 30.59 | 58% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.926 ms | 1.36 | 14.14 | 27% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.109 ms | 6.86 | 14.41 | 27% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 12.709 ms | 0.26 | 0.90 | 2%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 2.008 ms | 1.00 | 8.29 | 16%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 102.020 ms | 0.11 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 30.399 ms | 0.37 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 16.788 ms | 0.67 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 53.441 ms | 0.42 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 72.118 ms | 0.31 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 26.326 ms | 0.85 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 290.906 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.734 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.676 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.431 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 19.386 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 33.277 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 32.890 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 27.915 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 13.513 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 18.082 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 15.885 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 31.469 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 29.437 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 17.492 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 23.743 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 25.454 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 36.894 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 29.101 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.054 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.419 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 26.230 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.874 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.778 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.024 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.670 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.698 ms | 1.54 | 17.27 | 33% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.571 ms | 1.67 | 18.67 | 35% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.915 ms | 1.37 | 15.31 | 29% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.149 ms | 2.12 | 16.01 | 30% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 35.652 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 52.339 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 88.010 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 6.593 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 1.950 ms | 8.60 | 34.42 | 65% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 16.444 ms | 16.32 | 0.26 | 20% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.545 ms | 30.77 | 32.70 | 62% | ok |
| integer | dot | i32 | n=8388608, |v| <= 15 | 1.496 ms | 11.22 | 44.86 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, |v| <= 15 | 0.568 ms | 29.56 | 59.13 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, |v| <= 15 | 0.387 ms | 43.37 | 43.37 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, |v| <= 15 | 0.334 ms | 50.19 | 50.19 | n/a | ok |
| integer | gemm | i32 | 512^3, |v| <= 2047 | 15.386 ms | 17.45 | 0.20 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, |v| <= 2047 | 17.468 ms | 15.37 | 0.12 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, |v| <= 127 | 17.783 ms | 15.09 | 0.09 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, |v| <= 127 | 3.680 ms | 72.94 | 0.43 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, |v| <= 127 | 3.490 ms | 76.90 | 0.45 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

