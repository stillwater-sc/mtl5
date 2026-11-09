## AMD Ryzen 9 8945HS w/ Radeon 780M Graphics -- native, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.51.36247 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `685e2711d5ff` |
| FMA peak (ceiling) | f64 79.9 GFLOP/s, f32 161.9 GFLOP/s |
| ... hardware, AVX-512F | f64 79.9, f32 161.9 GFLOP/s |
| ... this build, scalar | f64 1.9, f32 1.8 GFLOP/s |
| bandwidth | read 42.7, update 53.9, triad 38.9 GB/s (ceiling 53.9) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 25% | 106% | 0 |
| blas2 | 6 | 22% | 36% | 0 |
| blas3 | 10 | 9% | 20% | 0 |
| factor | 12 | 3% | 5% | 0 |
| eigen | 8 | 4% | 9% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 26% | 53% | 0 |
| direct | 7 | 0% | 1% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 34% | 36% | 0 |
| mg | 1 | 30% | 30% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 13% | 26% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 9.463 ms | 1.77 | 14.18 | 26% | ok |
| blas1 | dot | f32 | n=8388608 | 10.456 ms | 1.60 | 6.42 | 12% | ok |
| blas1 | axpy | f64 | n=8388608 | 10.404 ms | 1.61 | 19.35 | 36% | ok |
| blas1 | axpy | f32 | n=8388608 | 10.344 ms | 1.62 | 9.73 | 18% | ok |
| blas1 | scale | f64 | n=8388608 | 2.337 ms | 3.59 | 57.44 | 106% | ok |
| blas1 | two_norm | f64 | n=8388608 | 9.811 ms | 1.71 | 6.84 | 13% | ok |
| blas1 | one_norm | f64 | n=8388608 | 4.966 ms | 1.69 | 13.51 | 25% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 5.061 ms | 1.66 | 13.26 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 4.955 ms | 1.69 | 13.54 | 25% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 79.878 ms | 2.31 | 1.68 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 10.603 ms | 3.16 | 12.66 | 23% | ok |
| blas2 | gemv | f32 | n=4096 | 9.943 ms | 3.37 | 6.75 | 13% | ok |
| blas2 | symv | f64 | n=4096 | 8.450 ms | 3.97 | 7.95 | 15% | ok |
| blas2 | trmv | f64 | n=4096 | 5.737 ms | 2.92 | 11.71 | 22% | ok |
| blas2 | trsv | f64 | n=4096 | 5.775 ms | 2.91 | 11.63 | 22% | ok |
| blas2 | ger | f64 | n=4096 | 13.891 ms | 2.42 | 19.33 | 36% | ok |
| blas3 | gemm | f64 | 512x512x512 | 17.007 ms | 15.78 | 0.37 | 20% | ok |
| blas3 | gemm | f32 | 512x512x512 | 8.783 ms | 30.56 | 0.36 | 19% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 13.561 ms | 9.90 | 0.46 | 12% | ok |
| blas3 | symm | f64 | 512x512x512 | 24.107 ms | 11.14 | 0.26 | 14% | ok |
| blas3 | syrk | f64 | 512x512x512 | 42.817 ms | 3.13 | 0.10 | 4% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 74.039 ms | 3.63 | 0.08 | 5% | ok |
| blas3 | trmm | f64 | 512x512x512 | 25.907 ms | 5.18 | 0.20 | 6% | ok |
| blas3 | trsm | f64 | 512x512x512 | 29.704 ms | 4.52 | 0.18 | 6% | ok |
| blas3 | trsm | f32 | 512x512x512 | 29.201 ms | 4.60 | 0.09 | 3% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 8.790 ms | 0.60 | 4.83 | 9% | ok |
| factor | lu | f64 | 512x512 | 34.798 ms | 2.57 | 0.12 | 3% | ok |
| factor | lu | f32 | 512x512 | 32.659 ms | 2.74 | 0.06 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 14.670 ms | 3.05 | 0.29 | 4% | ok |
| factor | cholesky | f32 | 512x512 | 11.013 ms | 4.06 | 0.19 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 11.663 ms | 3.84 | 0.36 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 172.351 ms | 0.26 | 0.02 | 0% | ok |
| factor | qr | f64 | 512x512 | 823.015 ms | 0.22 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 65.450 ms | 2.73 | 0.06 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 931.446 ms | 0.48 | 0.00 | 1% | ok |
| factor | householder | f64 | 512x512 | 3.910 ms | 0.27 | 1.07 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 144.220 ms | 1.86 | 0.03 | 2% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 35.968 ms | 2.49 | 0.12 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 38.286 ms | 0.58 | 0.03 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 129.725 ms | 1.16 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 497.231 ms | 0.30 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 65.824 ms | 2.55 | 0.02 | 3% | ok |
| eigen | schur | f64 | 256x256 | 127.225 ms | 3.30 | 0.01 | 4% | ok |
| eigen | svd | f64 | 256x256 | 116.775 ms | 3.16 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 36.384 ms | 6.92 | 0.03 | 9% | ok |
| eigen | sylvester | f64 | 256x256 | 291.256 ms | 3.46 | 0.01 | 4% | ok |
| elementwise | exp | f64 | n=8388608 | 27.611 ms | 0.30 | 4.86 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 41.320 ms | 0.20 | 3.25 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 37.955 ms | 0.22 | 3.54 | 7% | ok |
| elementwise | sqrt | f64 | n=8388608 | 20.305 ms | 0.41 | 6.61 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 31.059 ms | 0.27 | 4.32 | 8% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.960 ms | 2.73 | 28.35 | 53% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.361 ms | 1.92 | 20.01 | 37% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.304 ms | 6.64 | 13.97 | 26% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 12.958 ms | 0.25 | 0.89 | 2%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 2.413 ms | 0.83 | 6.90 | 13%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 95.898 ms | 0.12 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 31.913 ms | 0.35 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 21.349 ms | 0.52 | 0.03 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 58.089 ms | 0.39 | 0.01 | 0% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 75.543 ms | 0.30 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 27.273 ms | 0.82 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 276.931 ms | 0.01 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.797 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.340 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.465 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 22.719 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 42.841 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 56.874 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 33.077 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 17.755 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 22.959 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 19.076 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 52.540 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 36.981 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 21.860 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 32.006 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 36.000 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 56.349 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 47.502 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.044 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.448 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 26.941 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.994 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.796 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.026 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.780 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.604 ms | 1.63 | 18.29 | 34% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.514 ms | 1.73 | 19.36 | 36% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.433 ms | 1.08 | 12.05 | 22% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 18.182 ms | 2.11 | 15.99 | 30% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 45.610 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 59.618 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 91.352 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.787 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 9.829 ms | 1.71 | 6.83 | 13% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 29.233 ms | 9.18 | 0.14 | 11% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.263 ms | 13.28 | 14.11 | 26% | ok |
| integer | dot | i32 | n=8388608, |v| <= 15 | 3.210 ms | 5.23 | 20.91 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, |v| <= 15 | 2.597 ms | 6.46 | 12.92 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, |v| <= 15 | 1.366 ms | 12.28 | 12.28 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, |v| <= 15 | 1.728 ms | 9.71 | 9.71 | n/a | ok |
| integer | gemm | i32 | 512^3, |v| <= 2047 | 9.184 ms | 29.23 | 0.34 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, |v| <= 2047 | 38.412 ms | 6.99 | 0.05 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, |v| <= 127 | 30.708 ms | 8.74 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, |v| <= 127 | 30.613 ms | 8.77 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, |v| <= 127 | 27.631 ms | 9.71 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

