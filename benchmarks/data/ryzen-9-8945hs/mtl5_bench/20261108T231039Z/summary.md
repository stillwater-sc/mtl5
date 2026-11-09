# mtl5_bench platform characterization

Class `medium`, variants native native-fast, threads 1,16, 2026-11-08T23:19:00Z.

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

## AMD Ryzen 9 8945HS w/ Radeon 780M Graphics -- native, 16 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.51.36247 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `685e2711d5ff` |
| FMA peak (ceiling) | f64 602.3 GFLOP/s, f32 1196.6 GFLOP/s |
| ... hardware, AVX-512F | f64 602.3, f32 1196.6 GFLOP/s |
| ... this build, scalar | f64 20.2, f32 19.8 GFLOP/s |
| bandwidth | read 53.2, update 55.0, triad 41.0 GB/s (ceiling 55.0) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 82% | 98% | 0 | 17% |
| blas2 | 6 | 34% | 93% | 0 | 6% |
| blas3 | 10 | 2% | 12% | 0 | 7% |
| factor | 12 | 0% | 2% | 0 | 8% |
| eigen | 8 | 1% | 2% | 0 | 6% |
| elementwise | 5 | 8% | 12% | 0 | 6% |
| sparse | 5 | 37% | 130% | 0 | 6% |
| direct | 7 | 0% | 0% | 0 | 6% |
| ordering | 4 | n/a | n/a | 0 | 6% |
| krylov | 13 | n/a | n/a | 0 | 7% |
| pc | 7 | n/a | n/a | 0 | 6% |
| smoother | 3 | 35% | 36% | 0 | 7% |
| mg | 1 | 31% | 31% | 0 | 7% |
| sparse_eig | 4 | n/a | n/a | 0 | 7% |
| mixed | 3 | 12% | 25% | 0 | 6% |
| integer | 9 | n/a | n/a | 0 | 20% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.531 ms | 6.63 | 53.04 | 96% | ok | 3.7x (23%) |
| blas1 | dot | f32 | n=8388608 | 1.649 ms | 10.18 | 40.71 | 74% | ok | 6.3x (40%) |
| blas1 | axpy | f64 | n=8388608 | 3.911 ms | 4.29 | 51.48 | 94% | ok | 2.7x (17%) |
| blas1 | axpy | f32 | n=8388608 | 2.227 ms | 7.53 | 45.20 | 82% | ok | 4.6x (29%) |
| blas1 | scale | f64 | n=8388608 | 2.487 ms | 3.37 | 53.98 | 98% | ok | 0.9x (6%) |
| blas1 | two_norm | f64 | n=8388608 | 1.282 ms | 13.08 | 52.34 | 95% | ok | 7.7x (48%) |
| blas1 | one_norm | f64 | n=8388608 | 4.965 ms | 1.69 | 13.52 | 25% | ok | 1.0x (6%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.158 ms | 1.63 | 13.01 | 24% | ok | 1.0x (6%) |
| blas1 | sum | f64 | n=8388608 | 4.971 ms | 1.69 | 13.50 | 25% | ok | 1.0x (6%) |
| blas1 | dot_k3 | f64 | n=8388608 | 85.340 ms | 2.16 | 1.57 | 3% | ok | 0.9x (6%) |
| blas2 | gemv | f64 | n=4096 | 2.639 ms | 12.71 | 50.88 | 93% | ok | 4.0x (25%) |
| blas2 | gemv | f32 | n=4096 | 1.540 ms | 21.79 | 43.61 | 79% | ok | 6.5x (40%) |
| blas2 | symv | f64 | n=4096 | 8.666 ms | 3.87 | 7.75 | 14% | ok | 1.0x (6%) |
| blas2 | trmv | f64 | n=4096 | 5.853 ms | 2.87 | 11.48 | 21% | ok | 1.0x (6%) |
| blas2 | trsv | f64 | n=4096 | 5.861 ms | 2.86 | 11.46 | 21% | ok | 1.0x (6%) |
| blas2 | ger | f64 | n=4096 | 14.179 ms | 2.37 | 18.94 | 34% | ok | 1.0x (6%) |
| blas3 | gemm | f64 | 512x512x512 | 3.586 ms | 74.86 | 1.75 | 12% | ok | 4.7x (30%) |
| blas3 | gemm | f32 | 512x512x512 | 1.969 ms | 136.37 | 1.60 | 11% | ok | 4.5x (28%) |
| blas3 | gemmt | f64 | 512x512x512 | 12.409 ms | 10.82 | 0.51 | 2% | ok | 1.1x (7%) |
| blas3 | symm | f64 | 512x512x512 | 22.864 ms | 11.74 | 0.28 | 2% | ok | 1.1x (7%) |
| blas3 | syrk | f64 | 512x512x512 | 39.358 ms | 3.41 | 0.11 | 1% | ok | 1.1x (7%) |
| blas3 | syr2k | f64 | 512x512x512 | 75.624 ms | 3.55 | 0.08 | 1% | ok | 1.0x (6%) |
| blas3 | trmm | f64 | 512x512x512 | 26.578 ms | 5.05 | 0.20 | 1% | ok | 1.0x (6%) |
| blas3 | trsm | f64 | 512x512x512 | 31.219 ms | 4.30 | 0.17 | 1% | ok | 1.0x (6%) |
| blas3 | trsm | f32 | 512x512x512 | 29.604 ms | 4.53 | 0.09 | 0% | ok | 1.0x (6%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 8.991 ms | 0.59 | 4.72 | 9% | ok | 1.0x (6%) |
| factor | lu | f64 | 512x512 | 26.537 ms | 3.37 | 0.16 | 1% | ok | 1.3x (8%) |
| factor | lu | f32 | 512x512 | 24.738 ms | 3.62 | 0.08 | 0% | ok | 1.3x (8%) |
| factor | cholesky | f64 | 512x512 | 17.935 ms | 2.49 | 0.23 | 0%\* | ok | 0.8x (5%) |
| factor | cholesky | f32 | 512x512 | 11.277 ms | 3.97 | 0.19 | 0%\* | ok | 1.0x (6%) |
| factor | ldlt | f64 | 512x512 | 18.289 ms | 2.45 | 0.23 | 0%\* | ok | 0.6x (4%) |
| factor | ldlt_bk | f64 | 512x512 | 170.746 ms | 0.26 | 0.02 | 0%\* | ok | 1.0x (6%) |
| factor | qr | f64 | 512x512 | 473.817 ms | 0.38 | 0.01 | 0% | ok | 1.7x (11%) |
| factor | lq | f64 | 512x512 | 46.030 ms | 3.89 | 0.09 | 1% | ok | 1.4x (9%) |
| factor | hessenberg | f64 | 512x512 | 624.079 ms | 0.72 | 0.01 | 0% | ok | 1.5x (9%) |
| factor | householder | f64 | 512x512 | 3.351 ms | 0.31 | 1.25 | 2%\* | ok | 1.2x (7%) |
| factor | inv | f64 | 512x512 | 148.570 ms | 1.81 | 0.03 | 0% | ok | 1.0x (6%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 28.886 ms | 3.10 | 0.15 | 1% | ok | 1.2x (8%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 43.858 ms | 0.51 | 0.02 | 0% | ok | 0.9x (5%) |
| eigen | eigen_symmetric | f64 | 256x256 | 133.385 ms | 1.13 | 0.01 | 0% | ok | 1.0x (6%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 474.234 ms | 0.32 | 0.00 | 0% | ok | 1.0x (7%) |
| eigen | eigenvalue | f64 | 256x256 | 72.463 ms | 2.32 | 0.01 | 0% | ok | 0.9x (6%) |
| eigen | schur | f64 | 256x256 | 129.447 ms | 3.24 | 0.01 | 1% | ok | 1.0x (6%) |
| eigen | svd | f64 | 256x256 | 118.899 ms | 3.10 | 0.01 | 1% | ok | 1.0x (6%) |
| eigen | expm | f64 | 256x256 | 25.738 ms | 9.78 | 0.04 | 2% | ok | 1.4x (9%) |
| eigen | sylvester | f64 | 256x256 | 305.155 ms | 3.30 | 0.01 | 1% | ok | 1.0x (6%) |
| elementwise | exp | f64 | n=8388608 | 27.454 ms | 0.31 | 4.89 | 9% | ok | 1.0x (6%) |
| elementwise | log | f64 | n=8388608 | 42.511 ms | 0.20 | 3.16 | 6% | ok | 1.0x (6%) |
| elementwise | sin | f64 | n=8388608 | 38.620 ms | 0.22 | 3.48 | 6% | ok | 1.0x (6%) |
| elementwise | sqrt | f64 | n=8388608 | 20.344 ms | 0.41 | 6.60 | 12% | ok | 1.0x (6%) |
| elementwise | erf | f64 | n=8388608 | 31.714 ms | 0.26 | 4.23 | 8% | ok | 1.0x (6%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.380 ms | 6.89 | 71.72 | 130% | ok | 2.5x (16%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.355 ms | 1.93 | 20.09 | 37% | ok | 1.0x (6%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.257 ms | 18.55 | 39.01 | 71% | ok | 2.8x (17%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 12.916 ms | 0.25 | 0.89 | 2%\* | ok | 1.0x (6%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 2.951 ms | 0.68 | 5.64 | 10%\* | ok | 0.8x (5%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 99.854 ms | 0.11 | 0.01 | 0% | ok | 1.0x (6%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 32.900 ms | 0.34 | 0.02 | 0% | ok | 1.0x (6%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 20.229 ms | 0.55 | 0.04 | 0% | ok | 1.1x (7%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 55.482 ms | 0.40 | 0.01 | 0% | ok | 1.0x (7%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 75.041 ms | 0.30 | 0.01 | 0% | ok | 1.0x (6%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 27.339 ms | 0.82 | 0.03 | 0% | ok | 1.0x (6%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 264.092 ms | 0.01 | 0.00 | 0%\* | ok | 1.0x (7%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.210 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 7.860 ms | 0.00 | 0.00 | n/a | ok | 0.8x (5%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.459 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 20.208 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 41.442 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 46.531 ms | 0.00 | 0.00 | n/a | ok | 1.2x (8%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 30.886 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 16.868 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 21.895 ms | 0.00 | 0.00 | n/a | ok | 1.0x (7%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.890 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 52.375 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 35.076 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 20.439 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 32.171 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 34.510 ms | 0.00 | 0.00 | n/a | ok | 1.0x (7%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 57.643 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 47.504 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.047 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.622 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| pc | ilut | f64 | 128^2 grid | 24.963 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| pc | ic_0 | f64 | 128^2 grid | 2.070 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| pc | ildl | f64 | 128^2 grid | 1.970 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| pc | ssor | f64 | 128^2 grid | 0.028 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.831 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.500 ms | 1.75 | 19.55 | 36% | ok | 1.1x (7%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.505 ms | 1.74 | 19.48 | 35% | ok | 1.0x (6%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.798 ms | 1.46 | 16.31 | 30% | ok | 1.4x (8%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 17.098 ms | 2.25 | 17.00 | 31% | ok | 1.1x (7%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 35.952 ms | 0.00 | 0.00 | n/a | ok | 1.3x (8%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 50.493 ms | 0.00 | 0.00 | n/a | ok | 1.2x (7%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 88.719 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 8.754 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| mixed | dot | f32>f64 | n=8388608 | 9.810 ms | 1.71 | 6.84 | 12% | ok | 1.0x (6%) |
| mixed | gemm | f32>f64 | 512x512x512 | 6.503 ms | 41.28 | 0.64 | 7% | ok | 4.5x (28%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.290 ms | 13.00 | 13.82 | 25% | ok | 1.0x (6%) |
| integer | dot | i32 | n=8388608, |v| <= 15 | 3.214 ms | 5.22 | 20.88 | n/a | ok | 1.0x (6%) |
| integer | dot | i16>i32 | n=8388608, |v| <= 15 | 2.615 ms | 6.42 | 12.83 | n/a | ok | 1.0x (6%) |
| integer | dot | i8>i32 | n=8388608, |v| <= 15 | 1.370 ms | 12.25 | 12.25 | n/a | ok | 1.0x (6%) |
| integer | dot | u8i8>i32 | n=8388608, |v| <= 15 | 1.734 ms | 9.68 | 9.68 | n/a | ok | 1.0x (6%) |
| integer | gemm | i32 | 512^3, |v| <= 2047 | 1.420 ms | 189.00 | 2.21 | n/a | ok | 6.5x (40%) |
| integer | gemm | i16>i32 | 512^3, |v| <= 2047 | 5.879 ms | 45.66 | 0.36 | n/a | ok | 6.5x (41%) |
| integer | gemm | i8>i32 | 512^3, |v| <= 127 | 6.590 ms | 40.73 | 0.24 | n/a | ok | 4.7x (29%) |
| integer | gemm_quad | i8>i32 | 512^3, |v| <= 127 | 6.035 ms | 44.48 | 0.26 | n/a | ok | 5.1x (32%) |
| integer | gemm_quad | u8i8>i32 | 512^3, |v| <= 127 | 8.431 ms | 31.84 | 0.19 | n/a | ok | 3.3x (20%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

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

## AMD Ryzen 9 8945HS w/ Radeon 780M Graphics -- native-fast, 16 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.51.36247 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `685e2711d5ff` |
| FMA peak (ceiling) | f64 604.8 GFLOP/s, f32 1213.5 GFLOP/s |
| ... hardware, AVX-512F | f64 604.8, f32 1213.5 GFLOP/s |
| ... this build, SSE2 | f64 299.8, f32 599.3 GFLOP/s |
| bandwidth | read 52.0, update 52.9, triad 39.8 GB/s (ceiling 52.9) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 106% | 120% | 0 | 6% |
| blas2 | 6 | 67% | 108% | 0 | 6% |
| blas3 | 10 | 4% | 15% | 0 | 10% |
| factor | 12 | 0% | 7% | 0 | 8% |
| eigen | 8 | 0% | 2% | 0 | 6% |
| elementwise | 5 | 7% | 10% | 0 | 5% |
| sparse | 5 | 37% | 159% | 0 | 9% |
| direct | 7 | 0% | 0% | 0 | 6% |
| ordering | 4 | n/a | n/a | 0 | 6% |
| krylov | 13 | n/a | n/a | 0 | 6% |
| pc | 7 | n/a | n/a | 0 | 6% |
| smoother | 3 | 27% | 31% | 0 | 5% |
| mg | 1 | 31% | 31% | 0 | 6% |
| sparse_eig | 4 | n/a | n/a | 0 | 6% |
| mixed | 3 | 64% | 65% | 0 | 6% |
| integer | 9 | n/a | n/a | 0 | 17% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.387 ms | 7.03 | 56.22 | 106% | ok | 1.1x (7%) |
| blas1 | dot | f32 | n=8388608 | 1.064 ms | 15.76 | 63.05 | 119% | ok | 1.1x (7%) |
| blas1 | axpy | f64 | n=8388608 | 3.669 ms | 4.57 | 54.87 | 104% | ok | 1.0x (6%) |
| blas1 | axpy | f32 | n=8388608 | 1.718 ms | 9.76 | 58.58 | 111% | ok | 1.0x (6%) |
| blas1 | scale | f64 | n=8388608 | 2.225 ms | 3.77 | 60.33 | 114% | ok | 1.0x (6%) |
| blas1 | two_norm | f64 | n=8388608 | 1.062 ms | 15.80 | 63.21 | 120% | ok | 1.1x (7%) |
| blas1 | one_norm | f64 | n=8388608 | 5.008 ms | 1.68 | 13.40 | 25% | ok | 1.0x (6%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.234 ms | 1.60 | 12.82 | 24% | ok | 1.0x (6%) |
| blas1 | sum | f64 | n=8388608 | 4.999 ms | 1.68 | 13.42 | 25% | ok | 1.0x (6%) |
| blas1 | dot_k3 | f64 | n=8388608 | 87.385 ms | 2.11 | 1.54 | 3% | ok | 0.9x (6%) |
| blas2 | gemv | f64 | n=4096 | 2.419 ms | 13.87 | 55.51 | 105% | ok | 1.4x (9%) |
| blas2 | gemv | f32 | n=4096 | 1.174 ms | 28.59 | 57.21 | 108% | ok | 1.3x (8%) |
| blas2 | symv | f64 | n=4096 | 10.979 ms | 3.06 | 6.12 | 12% | ok | 0.8x (5%) |
| blas2 | trmv | f64 | n=4096 | 1.907 ms | 8.80 | 35.23 | 67% | ok | 1.0x (6%) |
| blas2 | trsv | f64 | n=4096 | 1.960 ms | 8.56 | 34.28 | 65% | ok | 1.0x (6%) |
| blas2 | ger | f64 | n=4096 | 14.160 ms | 2.37 | 18.96 | 36% | ok | 1.0x (6%) |
| blas3 | gemm | f64 | 512x512x512 | 2.960 ms | 90.68 | 2.13 | 15% | ok | 4.2x (26%) |
| blas3 | gemm | f32 | 512x512x512 | 1.809 ms | 148.38 | 1.74 | 12% | ok | 3.3x (20%) |
| blas3 | gemmt | f64 | 512x512x512 | 5.240 ms | 25.62 | 1.20 | 4% | ok | 1.5x (10%) |
| blas3 | symm | f64 | 512x512x512 | 3.670 ms | 73.14 | 1.71 | 12% | ok | 3.6x (23%) |
| blas3 | syrk | f64 | 512x512x512 | 22.893 ms | 5.86 | 0.18 | 1% | ok | 1.1x (7%) |
| blas3 | syr2k | f64 | 512x512x512 | 32.505 ms | 8.26 | 0.19 | 1% | ok | 2.3x (14%) |
| blas3 | trmm | f64 | 512x512x512 | 7.636 ms | 17.58 | 0.69 | 3% | ok | 1.4x (9%) |
| blas3 | trsm | f64 | 512x512x512 | 7.226 ms | 18.57 | 0.73 | 3% | ok | 1.7x (10%) |
| blas3 | trsm | f32 | 512x512x512 | 5.617 ms | 23.90 | 0.47 | 2% | ok | 1.3x (8%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.500 ms | 0.51 | 4.04 | 8% | ok | 0.9x (5%) |
| factor | lu | f64 | 512x512 | 27.398 ms | 3.27 | 0.15 | 1% | ok | 1.3x (8%) |
| factor | lu | f32 | 512x512 | 26.398 ms | 3.39 | 0.08 | 0% | ok | 1.3x (8%) |
| factor | cholesky | f64 | 512x512 | 12.307 ms | 3.64 | 0.34 | 1%\* | ok | 1.1x (7%) |
| factor | cholesky | f32 | 512x512 | 11.908 ms | 3.76 | 0.18 | 0%\* | ok | 0.9x (6%) |
| factor | ldlt | f64 | 512x512 | 12.439 ms | 3.60 | 0.34 | 1%\* | ok | 1.3x (8%) |
| factor | ldlt_bk | f64 | 512x512 | 177.705 ms | 0.25 | 0.02 | 0%\* | ok | 1.0x (6%) |
| factor | qr | f64 | 512x512 | 435.826 ms | 0.41 | 0.01 | 0% | ok | 2.0x (12%) |
| factor | lq | f64 | 512x512 | 48.609 ms | 3.68 | 0.09 | 1% | ok | 1.3x (8%) |
| factor | hessenberg | f64 | 512x512 | 517.322 ms | 0.86 | 0.01 | 0% | ok | 1.5x (10%) |
| factor | householder | f64 | 512x512 | 1.179 ms | 0.89 | 3.56 | 7%\* | ok | 3.9x (24%) |
| factor | inv | f64 | 512x512 | 115.532 ms | 2.32 | 0.04 | 0% | ok | 1.0x (6%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 31.650 ms | 2.83 | 0.13 | 0% | ok | 1.1x (7%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 38.367 ms | 0.58 | 0.03 | 0% | ok | 1.1x (7%) |
| eigen | eigen_symmetric | f64 | 256x256 | 130.422 ms | 1.16 | 0.01 | 0% | ok | 1.0x (6%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 538.470 ms | 0.28 | 0.00 | 0% | ok | 0.9x (6%) |
| eigen | eigenvalue | f64 | 256x256 | 58.864 ms | 2.85 | 0.02 | 0% | ok | 1.0x (6%) |
| eigen | schur | f64 | 256x256 | 120.082 ms | 3.49 | 0.01 | 1% | ok | 1.0x (6%) |
| eigen | svd | f64 | 256x256 | 124.708 ms | 2.96 | 0.01 | 0% | ok | 0.9x (6%) |
| eigen | expm | f64 | 256x256 | 24.788 ms | 10.15 | 0.04 | 2% | ok | 1.2x (7%) |
| eigen | sylvester | f64 | 256x256 | 277.213 ms | 3.63 | 0.01 | 1% | ok | 1.0x (6%) |
| elementwise | exp | f64 | n=8388608 | 36.584 ms | 0.23 | 3.67 | 7% | ok | 0.8x (5%) |
| elementwise | log | f64 | n=8388608 | 45.084 ms | 0.19 | 2.98 | 6% | ok | 0.9x (6%) |
| elementwise | sin | f64 | n=8388608 | 45.019 ms | 0.19 | 2.98 | 6% | ok | 0.9x (5%) |
| elementwise | sqrt | f64 | n=8388608 | 26.082 ms | 0.32 | 5.15 | 10% | ok | 0.8x (5%) |
| elementwise | erf | f64 | n=8388608 | 37.959 ms | 0.22 | 3.54 | 7% | ok | 0.8x (5%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.323 ms | 8.10 | 84.23 | 159% ! | ok | 2.8x (17%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.398 ms | 1.87 | 19.48 | 37% | ok | 1.4x (9%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.537 ms | 16.51 | 34.71 | 66% | ok | 2.4x (15%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 12.242 ms | 0.27 | 0.94 | 2%\* | ok | 1.0x (6%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 2.091 ms | 0.96 | 7.96 | 15%\* | ok | 1.0x (6%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 111.857 ms | 0.10 | 0.01 | 0% | ok | 0.9x (6%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 30.459 ms | 0.37 | 0.02 | 0% | ok | 1.0x (6%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 16.430 ms | 0.68 | 0.04 | 0% | ok | 1.0x (6%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 58.357 ms | 0.38 | 0.01 | 0% | ok | 0.9x (6%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 78.016 ms | 0.29 | 0.01 | 0% | ok | 0.9x (6%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 28.265 ms | 0.79 | 0.03 | 0% | ok | 0.9x (6%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 292.591 ms | 0.01 | 0.00 | 0%\* | ok | 1.0x (6%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.913 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.886 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.427 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 19.194 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 36.706 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 42.137 ms | 0.00 | 0.00 | n/a | ok | 0.8x (5%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 29.757 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 14.705 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 19.121 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.236 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 35.837 ms | 0.00 | 0.00 | n/a | ok | 0.9x (5%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 31.840 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 19.699 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 25.590 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 26.631 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 39.973 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 33.418 ms | 0.00 | 0.00 | n/a | ok | 0.9x (5%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.047 ms | 0.00 | 0.00 | n/a | ok | 1.1x (7%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.626 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| pc | ilut | f64 | 128^2 grid | 26.354 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| pc | ic_0 | f64 | 128^2 grid | 2.019 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| pc | ildl | f64 | 128^2 grid | 1.878 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| pc | ssor | f64 | 128^2 grid | 0.023 ms | 0.00 | 0.00 | n/a | ok | 1.0x (7%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.710 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 2.080 ms | 1.26 | 14.10 | 27% | ok | 0.8x (5%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.765 ms | 1.48 | 16.62 | 31% | ok | 0.9x (6%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.186 ms | 1.20 | 13.41 | 25% | ok | 0.9x (5%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 17.712 ms | 2.17 | 16.41 | 31% | ok | 1.0x (6%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 36.531 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 50.737 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 93.215 ms | 0.00 | 0.00 | n/a | ok | 0.9x (6%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 6.839 ms | 0.00 | 0.00 | n/a | ok | 1.0x (6%) |
| mixed | dot | f32>f64 | n=8388608 | 1.956 ms | 8.58 | 34.30 | 65% | ok | 1.0x (6%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.725 ms | 56.81 | 0.89 | 9% | ok | 3.5x (22%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.529 ms | 31.73 | 33.72 | 64% | ok | 1.0x (6%) |
| integer | dot | i32 | n=8388608, |v| <= 15 | 1.509 ms | 11.12 | 44.47 | n/a | ok | 1.0x (6%) |
| integer | dot | i16>i32 | n=8388608, |v| <= 15 | 0.582 ms | 28.81 | 57.61 | n/a | ok | 1.0x (6%) |
| integer | dot | i8>i32 | n=8388608, |v| <= 15 | 0.401 ms | 41.88 | 41.88 | n/a | ok | 1.0x (6%) |
| integer | dot | u8i8>i32 | n=8388608, |v| <= 15 | 0.364 ms | 46.05 | 46.05 | n/a | ok | 0.9x (6%) |
| integer | gemm | i32 | 512^3, |v| <= 2047 | 4.619 ms | 58.11 | 0.68 | n/a | ok | 3.3x (21%) |
| integer | gemm | i16>i32 | 512^3, |v| <= 2047 | 5.194 ms | 51.68 | 0.40 | n/a | ok | 3.4x (21%) |
| integer | gemm | i8>i32 | 512^3, |v| <= 127 | 5.171 ms | 51.92 | 0.30 | n/a | ok | 3.4x (21%) |
| integer | gemm_quad | i8>i32 | 512^3, |v| <= 127 | 1.322 ms | 203.11 | 1.19 | n/a | ok | 2.8x (17%) |
| integer | gemm_quad | u8i8>i32 | 512^3, |v| <= 127 | 1.271 ms | 211.20 | 1.24 | n/a | ok | 2.7x (17%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

