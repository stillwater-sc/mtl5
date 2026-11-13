# mtl5_bench platform characterization

Class `medium`, variants native native-fast blas, threads 1,8, CPUs 0,2,4,6,8,10,12,14, clock policy governor performance on the P-cores, turbo on; performance 800-4900 MHz on cpus 0,2,8,10,12,14; performance 800-5000 MHz on cpus 4,6, 2026-11-13T06:25:05Z.

## i7-12700k -- native, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `8867b44b3f52` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.4 GFLOP/s |
| bandwidth | read 33.4, update 38.6, triad 32.6 GB/s (ceiling 38.6) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 44% | 97% | 0 |
| blas2 | 10 | 80% | 96% | 0 |
| blas3 | 10 | 17% | 18% | 0 |
| factor | 12 | 8% | 18% | 0 |
| eigen | 8 | 7% | 12% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 49% | 95% | 0 |
| direct | 19 | 1% | 32% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 12 | 5% | 24% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 62% | 78% | 0 |
| mg | 1 | 57% | 57% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 36% | 84% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.578 ms | 3.01 | 24.06 | 62% | ok |
| blas1 | dot | f32 | n=8388608 | 4.203 ms | 3.99 | 15.97 | 41% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.301 ms | 2.66 | 31.95 | 83% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.686 ms | 3.58 | 21.48 | 56% | ok |
| blas1 | scale | f64 | n=8388608 | 3.568 ms | 2.35 | 37.61 | 97% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.827 ms | 3.48 | 13.90 | 36% | ok |
| blas1 | one_norm | f64 | n=8388608 | 3.987 ms | 2.10 | 16.83 | 44% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.029 ms | 1.19 | 9.55 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 3.953 ms | 2.12 | 16.98 | 44% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 76.300 ms | 2.42 | 1.76 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.631 ms | 9.24 | 36.98 | 96% | ok |
| blas2 | gemv | f32 | n=4096 | 2.746 ms | 12.22 | 24.45 | 63% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 4.695 ms | 7.15 | 28.60 | 74% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.187 ms | 15.34 | 30.69 | 80% | ok |
| blas2 | symv | f64 | n=4096 | 6.577 ms | 5.10 | 10.21 | 26% | ok |
| blas2 | trmv | f64 | n=4096 | 2.006 ms | 8.36 | 33.48 | 87% | ok |
| blas2 | trmv | f32 | n=4096 | 1.417 ms | 11.84 | 23.70 | 61% | ok |
| blas2 | trsv | f64 | n=4096 | 1.969 ms | 8.52 | 34.12 | 88% | ok |
| blas2 | trsv | f32 | n=4096 | 1.383 ms | 12.13 | 24.29 | 63% | ok |
| blas2 | ger | f64 | n=4096 | 7.681 ms | 4.37 | 34.96 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 19.950 ms | 13.46 | 0.32 | 17% | ok |
| blas3 | gemm | f32 | 512x512x512 | 10.616 ms | 25.29 | 0.30 | 16% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 10.754 ms | 12.48 | 0.59 | 16% | ok |
| blas3 | symm | f64 | 512x512x512 | 18.723 ms | 14.34 | 0.34 | 18% | ok |
| blas3 | syrk | f64 | 512x512x512 | 13.637 ms | 9.84 | 0.31 | 13% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 24.459 ms | 10.98 | 0.26 | 14% | ok |
| blas3 | trmm | f64 | 512x512x512 | 9.876 ms | 13.59 | 0.53 | 17% | ok |
| blas3 | trsm | f64 | 512x512x512 | 9.767 ms | 13.74 | 0.54 | 18% | ok |
| blas3 | trsm | f32 | 512x512x512 | 5.176 ms | 25.93 | 0.51 | 17% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.080 ms | 0.41 | 3.25 | 8% | ok |
| factor | lu | f64 | 512x512 | 11.536 ms | 7.76 | 0.36 | 10% | ok |
| factor | lu | f32 | 512x512 | 6.314 ms | 14.17 | 0.33 | 9% | ok |
| factor | cholesky | f64 | 512x512 | 7.301 ms | 6.13 | 0.57 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.574 ms | 6.81 | 0.32 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.416 ms | 6.03 | 0.57 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 45.877 ms | 0.98 | 0.09 | 1% | ok |
| factor | qr | f64 | 512x512 | 229.889 ms | 0.78 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 25.827 ms | 6.93 | 0.16 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 269.801 ms | 1.66 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.650 ms | 0.64 | 2.54 | 7%\* | ok |
| factor | inv | f64 | 512x512 | 54.323 ms | 4.94 | 0.08 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.213 ms | 14.40 | 0.68 | 18% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 17.872 ms | 1.25 | 0.06 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 58.694 ms | 2.57 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 746.021 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 32.512 ms | 5.16 | 0.03 | 7% | ok |
| eigen | schur | f64 | 256x256 | 82.255 ms | 5.10 | 0.01 | 7% | ok |
| eigen | svd | f64 | 256x256 | 111.286 ms | 3.32 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 26.338 ms | 9.55 | 0.04 | 12% | ok |
| eigen | sylvester | f64 | 256x256 | 188.447 ms | 5.34 | 0.01 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 39.278 ms | 0.21 | 3.42 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 45.669 ms | 0.18 | 2.94 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 60.009 ms | 0.14 | 2.24 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 28.876 ms | 0.29 | 4.65 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 201.442 ms | 0.04 | 0.67 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.746 ms | 3.51 | 36.52 | 95% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.928 ms | 2.82 | 29.33 | 76% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.636 ms | 9.03 | 18.99 | 49% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.843 ms | 0.67 | 2.37 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.383 ms | 1.45 | 12.04 | 31%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 91.135 ms | 0.12 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.488 ms | 0.44 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.897 ms | 1.26 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.069 ms | 0.56 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 53.724 ms | 0.42 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.852 ms | 1.33 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 99.545 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.434 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.078 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.224 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.121 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.355 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 88.624 ms | 0.13 | 0.01 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.247 ms | 3.06 | 12.24 | 32%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.493 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.171 ms | 2.68 | 0.18 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.251 ms | 3.01 | 12.05 | 31%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.091 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 36.452 ms | 0.61 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.625 ms | 1.21 | 4.84 | 13%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.932 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.663 ms | 0.47 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.623 ms | 1.21 | 4.86 | 13%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 9.480 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 7431.880 ms | 0.04 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.428 ms | 2.36 | 9.44 | 24%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 18.787 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 67.422 ms | 4.58 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.446 ms | 2.33 | 9.32 | 24%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.887 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 554.834 ms | 1.11 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.497 ms | 0.45 | 1.80 | 5%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.648 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 644.323 ms | 0.96 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.543 ms | 0.45 | 1.79 | 5%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 31.249 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 33.274 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.893 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.567 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 15.091 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 14.107 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.769 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.389 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.397 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 17.333 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.539 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.976 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.258 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.036 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.595 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.413 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.076 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.010 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.023 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.715 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.979 ms | 2.67 | 29.97 | 78% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.217 ms | 2.15 | 24.09 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.562 ms | 1.68 | 18.77 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.274 ms | 2.90 | 21.90 | 57% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.793 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 44.903 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 35.809 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 5.039 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.898 ms | 3.43 | 13.70 | 36% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 39.306 ms | 6.83 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.549 ms | 30.56 | 32.47 | 84%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.285 ms | 7.34 | 29.36 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.987 ms | 16.99 | 33.98 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.721 ms | 9.75 | 9.75 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.721 ms | 9.75 | 9.75 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.455 ms | 15.38 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.803 ms | 12.90 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.224 ms | 26.26 | 0.15 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.230 ms | 26.24 | 0.15 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 27.017 ms | 9.94 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## i7-12700k -- native, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `8867b44b3f52` |
| FMA peak (ceiling) | f64 600.8 GFLOP/s, f32 1201.8 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.8, f32 1201.8 GFLOP/s |
| ... this build, scalar | f64 9.8, f32 10.5 GFLOP/s |
| bandwidth | read 83.8, update 78.7, triad 76.9 GB/s (ceiling 83.8) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 84% | 123% | 0 | 28% |
| blas2 | 10 | 42% | 137% | 0 | 13% |
| blas3 | 10 | 2% | 16% | 0 | 15% |
| factor | 12 | 1% | 12% | 0 | 19% |
| eigen | 8 | 1% | 4% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 280% | 0 | 13% |
| direct | 19 | 0% | 13% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 10% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 29% | 36% | 0 | 13% |
| mg | 1 | 25% | 25% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 40% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 92% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.618 ms | 10.37 | 82.95 | 99% | ok | 3.4x (43%) |
| blas1 | dot | f32 | n=8388608 | 0.653 ms | 25.71 | 102.85 | 123% | ok | 6.4x (81%) |
| blas1 | axpy | f64 | n=8388608 | 2.853 ms | 5.88 | 70.57 | 84% | ok | 2.2x (28%) |
| blas1 | axpy | f32 | n=8388608 | 1.452 ms | 11.56 | 69.34 | 83% | ok | 3.2x (40%) |
| blas1 | scale | f64 | n=8388608 | 1.912 ms | 4.39 | 70.18 | 84% | ok | 1.9x (23%) |
| blas1 | two_norm | f64 | n=8388608 | 0.740 ms | 22.66 | 90.64 | 108% | ok | 6.5x (81%) |
| blas1 | one_norm | f64 | n=8388608 | 3.920 ms | 2.14 | 17.12 | 20% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.903 ms | 1.22 | 9.72 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.891 ms | 2.16 | 17.25 | 21% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 75.065 ms | 2.46 | 1.79 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.450 ms | 23.13 | 92.58 | 110% | ok | 2.5x (31%) |
| blas2 | gemv | f32 | n=4096 | 0.583 ms | 57.57 | 115.20 | 137% | ok | 4.7x (59%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.573 ms | 21.33 | 85.35 | 102% | ok | 3.0x (37%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.868 ms | 38.66 | 77.36 | 92% | ok | 2.5x (32%) |
| blas2 | symv | f64 | n=4096 | 6.465 ms | 5.19 | 10.39 | 12% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 1.992 ms | 8.42 | 33.72 | 40% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 1.397 ms | 12.01 | 24.04 | 29% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 1.953 ms | 8.59 | 34.39 | 41% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 1.361 ms | 12.32 | 24.67 | 29% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.623 ms | 4.40 | 35.22 | 42% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 2.733 ms | 98.23 | 2.30 | 16% | ok | 7.3x (91%) |
| blas3 | gemm | f32 | 512x512x512 | 1.529 ms | 175.53 | 2.06 | 15% | ok | 6.9x (87%) |
| blas3 | gemmt | f64 | 512x512x512 | 9.014 ms | 14.89 | 0.70 | 2% | ok | 1.2x (15%) |
| blas3 | symm | f64 | 512x512x512 | 16.644 ms | 16.13 | 0.38 | 3% | ok | 1.1x (14%) |
| blas3 | syrk | f64 | 512x512x512 | 11.603 ms | 11.57 | 0.36 | 2% | ok | 1.2x (15%) |
| blas3 | syr2k | f64 | 512x512x512 | 19.509 ms | 13.76 | 0.32 | 2% | ok | 1.3x (16%) |
| blas3 | trmm | f64 | 512x512x512 | 9.117 ms | 14.72 | 0.58 | 2% | ok | 1.1x (14%) |
| blas3 | trsm | f64 | 512x512x512 | 9.270 ms | 14.48 | 0.57 | 2% | ok | 1.1x (13%) |
| blas3 | trsm | f32 | 512x512x512 | 5.050 ms | 26.58 | 0.52 | 2% | ok | 1.0x (13%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.825 ms | 0.41 | 3.31 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 7.199 ms | 12.43 | 0.58 | 2% | ok | 1.6x (20%) |
| factor | lu | f32 | 512x512 | 3.998 ms | 22.38 | 0.52 | 2% | ok | 1.6x (20%) |
| factor | cholesky | f64 | 512x512 | 7.058 ms | 6.34 | 0.59 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 6.446 ms | 6.94 | 0.33 | 1% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 7.180 ms | 6.23 | 0.58 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 45.293 ms | 0.99 | 0.09 | 0% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 128.122 ms | 1.40 | 0.03 | 0% | ok | 1.8x (22%) |
| factor | lq | f64 | 512x512 | 17.425 ms | 10.27 | 0.24 | 2% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 165.708 ms | 2.70 | 0.03 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 0.434 ms | 2.42 | 9.66 | 12%\* | ok | 3.8x (47%) |
| factor | inv | f64 | 512x512 | 51.055 ms | 5.26 | 0.08 | 1% | ok | 1.1x (13%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.165 ms | 14.51 | 0.68 | 2% | ok | 1.0x (13%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 17.932 ms | 1.25 | 0.06 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 57.476 ms | 2.63 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 743.840 ms | 0.20 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 31.619 ms | 5.31 | 0.03 | 1% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 81.222 ms | 5.16 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 109.493 ms | 3.37 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 11.631 ms | 21.64 | 0.09 | 4% | ok | 2.3x (28%) |
| eigen | sylvester | f64 | 256x256 | 175.079 ms | 5.75 | 0.01 | 1% | ok | 1.1x (13%) |
| elementwise | exp | f64 | n=8388608 | 38.436 ms | 0.22 | 3.49 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 44.713 ms | 0.19 | 3.00 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 58.771 ms | 0.14 | 2.28 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.255 ms | 0.30 | 4.75 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 199.387 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.116 ms | 22.58 | 234.90 | 280% ! | ok | 6.4x (80%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.898 ms | 2.92 | 30.34 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.316 ms | 31.82 | 66.90 | 80% | ok | 3.5x (44%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.768 ms | 0.68 | 2.41 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.380 ms | 1.45 | 12.06 | 14%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 89.346 ms | 0.13 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.028 ms | 0.45 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.880 ms | 1.26 | 0.08 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.399 ms | 0.57 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 52.301 ms | 0.43 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.766 ms | 1.33 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 85.428 ms | 0.02 | 0.00 | 0% | ok | 1.2x (15%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.262 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.863 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.220 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.714 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.279 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 86.680 ms | 0.13 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.268 ms | 2.82 | 11.29 | 13%\* | ok | 0.9x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.360 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.137 ms | 2.70 | 0.18 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.272 ms | 2.78 | 11.13 | 13%\* | ok | 0.9x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.034 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.123 ms | 0.64 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.599 ms | 1.26 | 5.05 | 6%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.783 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 46.658 ms | 0.48 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.605 ms | 1.25 | 5.00 | 6%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 9.248 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 7281.000 ms | 0.04 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.533 ms | 2.20 | 8.79 | 10%\* | ok | 0.9x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 18.283 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 66.073 ms | 4.67 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.555 ms | 2.17 | 8.67 | 10%\* | ok | 0.9x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.754 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 542.204 ms | 1.14 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.970 ms | 0.42 | 1.69 | 2%\* | ok | 0.9x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.332 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 655.625 ms | 0.94 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 8.013 ms | 0.42 | 1.68 | 2%\* | ok | 0.9x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.454 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 32.020 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.392 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.201 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.605 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.677 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.129 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.683 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.909 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 16.766 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.037 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.457 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 20.696 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.032 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.530 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 12.087 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.044 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.987 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.020 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.699 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.963 ms | 2.72 | 30.44 | 36% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.197 ms | 2.19 | 24.50 | 29% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.533 ms | 1.71 | 19.13 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.759 ms | 2.79 | 21.12 | 25% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.497 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 44.271 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.634 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.901 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 4.786 ms | 3.51 | 14.02 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 5.187 ms | 51.75 | 0.81 | 9% | ok | 7.6x (95%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.534 ms | 31.40 | 33.37 | 40%\* | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.252 ms | 7.45 | 29.80 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.977 ms | 17.17 | 34.35 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.683 ms | 9.97 | 9.97 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.684 ms | 9.96 | 9.96 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.379 ms | 112.85 | 1.32 | n/a | ok | 7.3x (92%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.749 ms | 97.65 | 0.76 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.356 ms | 197.95 | 1.16 | n/a | ok | 7.5x (94%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.356 ms | 197.96 | 1.16 | n/a | ok | 7.5x (94%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.607 ms | 74.41 | 0.44 | n/a | ok | 7.5x (94%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

## i7-12700k -- native-fast, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `8867b44b3f52` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, AVX2 | f64 78.3, f32 156.6 GFLOP/s |
| bandwidth | read 37.8, update 41.5, triad 31.5 GB/s (ceiling 41.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 88% | 104% | 0 |
| blas2 | 10 | 94% | 97% | 0 |
| blas3 | 10 | 65% | 91% | 0 |
| factor | 12 | 8% | 24% | 0 |
| eigen | 8 | 7% | 27% | 0 |
| elementwise | 5 | 7% | 11% | 0 |
| sparse | 5 | 53% | 89% | 0 |
| direct | 19 | 1% | 29% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 12 | 4% | 23% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 54% | 72% | 0 |
| mg | 1 | 49% | 49% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 84% | 135% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 3.671 ms | 4.57 | 36.56 | 88% | ok |
| blas1 | dot | f32 | n=8388608 | 1.686 ms | 9.95 | 39.80 | 96% | ok |
| blas1 | axpy | f64 | n=8388608 | 4.756 ms | 3.53 | 42.33 | 102% | ok |
| blas1 | axpy | f32 | n=8388608 | 2.342 ms | 7.16 | 42.98 | 104% | ok |
| blas1 | scale | f64 | n=8388608 | 3.454 ms | 2.43 | 38.85 | 94% | ok |
| blas1 | two_norm | f64 | n=8388608 | 2.186 ms | 7.68 | 30.70 | 74% | ok |
| blas1 | one_norm | f64 | n=8388608 | 3.986 ms | 2.10 | 16.83 | 41% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.029 ms | 1.19 | 9.55 | 23% | ok |
| blas1 | sum | f64 | n=8388608 | 3.935 ms | 2.13 | 17.06 | 41% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 70.596 ms | 2.61 | 1.90 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.393 ms | 9.89 | 39.58 | 95% | ok |
| blas2 | gemv | f32 | n=4096 | 1.689 ms | 19.87 | 39.76 | 96% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.451 ms | 9.72 | 38.91 | 94% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.675 ms | 20.03 | 40.08 | 97% | ok |
| blas2 | symv | f64 | n=4096 | 6.176 ms | 5.43 | 10.88 | 26% | ok |
| blas2 | trmv | f64 | n=4096 | 1.782 ms | 9.42 | 37.70 | 91% | ok |
| blas2 | trmv | f32 | n=4096 | 0.857 ms | 19.57 | 39.19 | 94% | ok |
| blas2 | trsv | f64 | n=4096 | 1.818 ms | 9.23 | 36.96 | 89% | ok |
| blas2 | trsv | f32 | n=4096 | 0.865 ms | 19.39 | 38.82 | 94% | ok |
| blas2 | ger | f64 | n=4096 | 7.128 ms | 4.71 | 37.67 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.795 ms | 70.74 | 1.66 | 90% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.892 ms | 141.89 | 1.66 | 91% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.628 ms | 51.07 | 2.39 | 65% | ok |
| blas3 | symm | f64 | 512x512x512 | 4.044 ms | 66.38 | 1.56 | 85% | ok |
| blas3 | syrk | f64 | 512x512x512 | 9.491 ms | 14.14 | 0.44 | 18% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 20.260 ms | 13.25 | 0.31 | 17% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.699 ms | 49.72 | 1.94 | 63% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.754 ms | 48.73 | 1.90 | 62% | ok |
| blas3 | trsm | f32 | 512x512x512 | 1.293 ms | 103.79 | 2.03 | 66% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.101 ms | 0.41 | 3.24 | 8% | ok |
| factor | lu | f64 | 512x512 | 10.840 ms | 8.25 | 0.39 | 11% | ok |
| factor | lu | f32 | 512x512 | 3.643 ms | 24.56 | 0.58 | 16% | ok |
| factor | cholesky | f64 | 512x512 | 7.107 ms | 6.30 | 0.59 | 8% | ok |
| factor | cholesky | f32 | 512x512 | 6.578 ms | 6.80 | 0.32 | 4% | ok |
| factor | ldlt | f64 | 512x512 | 7.174 ms | 6.24 | 0.58 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 46.374 ms | 0.96 | 0.09 | 1% | ok |
| factor | qr | f64 | 512x512 | 221.941 ms | 0.81 | 0.02 | 1% | ok |
| factor | lq | f64 | 512x512 | 23.179 ms | 7.72 | 0.18 | 10% | ok |
| factor | hessenberg | f64 | 512x512 | 257.501 ms | 1.74 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.619 ms | 0.65 | 2.59 | 6%\* | ok |
| factor | inv | f64 | 512x512 | 55.052 ms | 4.88 | 0.08 | 6% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 4.798 ms | 18.65 | 0.87 | 24% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 17.083 ms | 1.31 | 0.06 | 2% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 57.581 ms | 2.62 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 752.584 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 30.872 ms | 5.43 | 0.03 | 7% | ok |
| eigen | schur | f64 | 256x256 | 80.182 ms | 5.23 | 0.01 | 7% | ok |
| eigen | svd | f64 | 256x256 | 103.577 ms | 3.56 | 0.01 | 5% | ok |
| eigen | expm | f64 | 256x256 | 11.709 ms | 21.49 | 0.09 | 27% | ok |
| eigen | sylvester | f64 | 256x256 | 176.365 ms | 5.71 | 0.01 | 7% | ok |
| elementwise | exp | f64 | n=8388608 | 39.343 ms | 0.21 | 3.41 | 8% | ok |
| elementwise | log | f64 | n=8388608 | 46.811 ms | 0.18 | 2.87 | 7% | ok |
| elementwise | sin | f64 | n=8388608 | 59.864 ms | 0.14 | 2.24 | 5% | ok |
| elementwise | sqrt | f64 | n=8388608 | 28.846 ms | 0.29 | 4.65 | 11% | ok |
| elementwise | erf | f64 | n=8388608 | 201.097 ms | 0.04 | 0.67 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.735 ms | 3.56 | 37.03 | 89% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.924 ms | 2.83 | 29.46 | 71% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.034 ms | 10.38 | 21.83 | 53% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.824 ms | 0.68 | 2.38 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.250 ms | 1.61 | 13.31 | 32%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 72.975 ms | 0.15 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 27.441 ms | 0.41 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 9.035 ms | 1.24 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.502 ms | 0.57 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 55.178 ms | 0.41 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.903 ms | 1.32 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 99.941 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.501 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.028 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.225 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.157 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.343 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 70.124 ms | 0.16 | 0.01 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.253 ms | 2.98 | 11.93 | 29%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.491 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.284 ms | 2.61 | 0.17 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.272 ms | 2.78 | 11.12 | 27%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.134 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.378 ms | 0.63 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.642 ms | 1.18 | 4.71 | 11%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.921 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 49.278 ms | 0.45 | 0.01 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.639 ms | 1.18 | 4.74 | 11%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 9.594 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 5323.295 ms | 0.06 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.440 ms | 2.34 | 9.36 | 23%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 18.739 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 67.439 ms | 4.58 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.657 ms | 2.03 | 8.14 | 20%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.936 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 525.953 ms | 1.17 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.494 ms | 0.45 | 1.80 | 4%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.668 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 657.303 ms | 0.94 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.499 ms | 0.45 | 1.80 | 4%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.369 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 28.621 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 20.869 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.474 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.721 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.613 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 20.247 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.835 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.494 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.625 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 14.926 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 15.984 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 18.205 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.034 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.575 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.019 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.097 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.029 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.023 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.733 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.976 ms | 2.68 | 30.03 | 72% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.308 ms | 2.00 | 22.42 | 54% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.509 ms | 1.73 | 19.44 | 47% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.294 ms | 2.69 | 20.33 | 49% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 28.513 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 39.547 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 34.610 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.589 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 1.928 ms | 8.70 | 34.80 | 84% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 5.996 ms | 44.77 | 0.70 | 57% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.318 ms | 52.68 | 55.97 | 135%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.909 ms | 8.79 | 35.15 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.759 ms | 22.09 | 44.19 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.291 ms | 57.66 | 57.66 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.279 ms | 60.10 | 60.10 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 3.980 ms | 67.45 | 0.79 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.621 ms | 47.75 | 0.37 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.699 ms | 47.11 | 0.28 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 2.126 ms | 126.27 | 0.74 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 1.908 ms | 140.65 | 0.82 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## i7-12700k -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `8867b44b3f52` |
| FMA peak (ceiling) | f64 600.9 GFLOP/s, f32 1201.9 GFLOP/s |
| ... hardware, AVX2+FMA | f64 600.9, f32 1201.9 GFLOP/s |
| ... this build, AVX2 | f64 600.8, f32 1201.7 GFLOP/s |
| bandwidth | read 83.0, update 78.9, triad 77.5 GB/s (ceiling 83.0) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 87% | 146% | 0 | 22% |
| blas2 | 10 | 48% | 148% | 0 | 13% |
| blas3 | 10 | 12% | 67% | 0 | 18% |
| factor | 12 | 1% | 12% | 0 | 13% |
| eigen | 8 | 1% | 4% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 36% | 291% | 0 | 13% |
| direct | 19 | 0% | 13% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 11% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 28% | 37% | 0 | 13% |
| mg | 1 | 24% | 24% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 49% | 72% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 78% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.394 ms | 12.04 | 96.30 | 116% | ok | 2.6x (33%) |
| blas1 | dot | f32 | n=8388608 | 0.554 ms | 30.29 | 121.18 | 146% | ok | 3.0x (38%) |
| blas1 | axpy | f64 | n=8388608 | 2.774 ms | 6.05 | 72.58 | 87% | ok | 1.7x (21%) |
| blas1 | axpy | f32 | n=8388608 | 1.362 ms | 12.32 | 73.93 | 89% | ok | 1.7x (22%) |
| blas1 | scale | f64 | n=8388608 | 1.903 ms | 4.41 | 70.53 | 85% | ok | 1.8x (23%) |
| blas1 | two_norm | f64 | n=8388608 | 0.563 ms | 29.78 | 119.13 | 144% | ok | 3.9x (49%) |
| blas1 | one_norm | f64 | n=8388608 | 3.927 ms | 2.14 | 17.09 | 21% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.904 ms | 1.22 | 9.72 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.878 ms | 2.16 | 17.30 | 21% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 69.276 ms | 2.66 | 1.94 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.438 ms | 23.34 | 93.39 | 113% | ok | 2.4x (29%) |
| blas2 | gemv | f32 | n=4096 | 0.546 ms | 61.45 | 122.97 | 148% | ok | 3.1x (39%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.449 ms | 23.15 | 92.64 | 112% | ok | 2.4x (30%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.675 ms | 49.75 | 99.54 | 120% | ok | 2.5x (31%) |
| blas2 | symv | f64 | n=4096 | 6.192 ms | 5.42 | 10.85 | 13% | ok | 1.0x (12%) |
| blas2 | trmv | f64 | n=4096 | 1.750 ms | 9.59 | 38.38 | 46% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 0.836 ms | 20.07 | 40.18 | 48% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 1.780 ms | 9.42 | 37.73 | 45% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.850 ms | 19.74 | 39.51 | 48% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.099 ms | 4.73 | 37.82 | 46% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 0.671 ms | 399.90 | 9.37 | 67% | ok | 5.7x (71%) |
| blas3 | gemm | f32 | 512x512x512 | 0.354 ms | 757.34 | 8.88 | 63% | ok | 5.3x (67%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.248 ms | 107.57 | 5.04 | 18% | ok | 2.1x (26%) |
| blas3 | symm | f64 | 512x512x512 | 1.120 ms | 239.66 | 5.62 | 40% | ok | 3.6x (45%) |
| blas3 | syrk | f64 | 512x512x512 | 10.766 ms | 12.47 | 0.39 | 2% | ok | 0.9x (11%) |
| blas3 | syr2k | f64 | 512x512x512 | 16.674 ms | 16.10 | 0.38 | 3% | ok | 1.2x (15%) |
| blas3 | trmm | f64 | 512x512x512 | 1.884 ms | 71.22 | 2.78 | 12% | ok | 1.4x (18%) |
| blas3 | trsm | f64 | 512x512x512 | 2.142 ms | 62.67 | 2.45 | 10% | ok | 1.3x (16%) |
| blas3 | trsm | f32 | 512x512x512 | 1.105 ms | 121.43 | 2.37 | 10% | ok | 1.2x (15%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.845 ms | 0.41 | 3.31 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 6.129 ms | 14.60 | 0.68 | 2% | ok | 1.8x (22%) |
| factor | lu | f32 | 512x512 | 3.570 ms | 25.07 | 0.59 | 2% | ok | 1.0x (13%) |
| factor | cholesky | f64 | 512x512 | 6.930 ms | 6.46 | 0.61 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 6.407 ms | 6.98 | 0.33 | 1% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 7.042 ms | 6.35 | 0.60 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 43.169 ms | 1.04 | 0.10 | 0% | ok | 1.1x (13%) |
| factor | qr | f64 | 512x512 | 126.396 ms | 1.42 | 0.03 | 0% | ok | 1.8x (22%) |
| factor | lq | f64 | 512x512 | 15.133 ms | 11.83 | 0.28 | 2% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 158.763 ms | 2.82 | 0.03 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 0.412 ms | 2.55 | 10.19 | 12%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 51.663 ms | 5.20 | 0.08 | 1% | ok | 1.1x (13%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 4.780 ms | 18.72 | 0.88 | 3% | ok | 1.0x (13%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 16.493 ms | 1.36 | 0.06 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 55.973 ms | 2.70 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 735.554 ms | 0.21 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 30.866 ms | 5.44 | 0.03 | 1% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 78.618 ms | 5.34 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 91.303 ms | 4.04 | 0.01 | 1% | ok | 1.1x (14%) |
| eigen | expm | f64 | 256x256 | 10.026 ms | 25.10 | 0.10 | 4% | ok | 1.2x (15%) |
| eigen | sylvester | f64 | 256x256 | 169.264 ms | 5.95 | 0.01 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 38.474 ms | 0.22 | 3.49 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 45.719 ms | 0.18 | 2.94 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 58.512 ms | 0.14 | 2.29 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.293 ms | 0.30 | 4.74 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 198.832 ms | 0.04 | 0.68 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.113 ms | 23.23 | 241.72 | 291% ! | ok | 6.5x (82%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.911 ms | 2.87 | 29.89 | 36% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.362 ms | 30.75 | 64.66 | 78% | ok | 3.0x (37%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.784 ms | 0.68 | 2.40 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.386 ms | 1.45 | 12.01 | 14%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 71.877 ms | 0.16 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 26.997 ms | 0.41 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 9.113 ms | 1.23 | 0.08 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.264 ms | 0.57 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 55.389 ms | 0.40 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.803 ms | 1.33 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 85.698 ms | 0.02 | 0.00 | 0% | ok | 1.2x (15%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.358 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.877 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.222 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.865 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.279 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 68.926 ms | 0.16 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.275 ms | 2.75 | 11.00 | 13%\* | ok | 0.9x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.372 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.280 ms | 2.61 | 0.17 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.289 ms | 2.62 | 10.48 | 13%\* | ok | 0.9x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.056 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.433 ms | 0.63 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.611 ms | 1.24 | 4.95 | 6%\* | ok | 1.1x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.795 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 49.245 ms | 0.45 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.613 ms | 1.23 | 4.94 | 6%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 9.262 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 5212.481 ms | 0.06 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.537 ms | 2.19 | 8.77 | 11%\* | ok | 0.9x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 18.259 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 64.492 ms | 4.79 | 0.01 | 1% | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.776 ms | 1.90 | 7.59 | 9%\* | ok | 0.9x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.830 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 511.821 ms | 1.21 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 8.021 ms | 0.42 | 1.68 | 2%\* | ok | 0.9x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.364 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 643.745 ms | 0.96 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 8.028 ms | 0.42 | 1.68 | 2%\* | ok | 0.9x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.138 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 28.367 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 20.665 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.390 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.646 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.521 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 20.188 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.490 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.426 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.519 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 14.765 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 15.848 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 17.909 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.035 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.527 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 11.809 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.068 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.996 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.721 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.954 ms | 2.74 | 30.74 | 37% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.284 ms | 2.04 | 22.85 | 28% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.480 ms | 1.77 | 19.81 | 24% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.888 ms | 2.58 | 19.52 | 24% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 28.577 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 39.682 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.695 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.505 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 1.855 ms | 9.04 | 36.18 | 44% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 0.911 ms | 294.56 | 4.60 | 49% | ok | 6.6x (82%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.297 ms | 56.45 | 59.98 | 72%\* | ok | 1.1x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.885 ms | 8.90 | 35.61 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.739 ms | 22.72 | 45.44 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.266 ms | 63.04 | 63.04 | n/a | ok | 1.1x (14%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.251 ms | 66.81 | 66.81 | n/a | ok | 1.1x (14%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 0.624 ms | 430.46 | 5.04 | n/a | ok | 6.4x (80%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 0.819 ms | 327.93 | 2.56 | n/a | ok | 6.9x (86%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 0.820 ms | 327.41 | 1.92 | n/a | ok | 7.0x (87%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.336 ms | 798.49 | 4.68 | n/a | ok | 6.3x (79%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.307 ms | 874.79 | 5.13 | n/a | ok | 6.2x (78%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

## i7-12700k -- blas, 1 thread, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `8867b44b3f52` |
| FMA peak (ceiling) | f64 78.3 GFLOP/s, f32 156.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 78.3, f32 156.6 GFLOP/s |
| ... this build, scalar | f64 1.3, f32 1.3 GFLOP/s |
| bandwidth | read 33.4, update 38.5, triad 32.7 GB/s (ceiling 38.5) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 44% | 98% | 0 |
| blas2 | 10 | 98% | 107% | 0 |
| blas3 | 10 | 81% | 94% | 0 |
| factor | 12 | 22% | 40% | 0 |
| eigen | 8 | 24% | 71% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 48% | 95% | 0 |
| direct | 31 | 3% | 32% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| direct3d | 21 | 25% | 70% | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 62% | 78% | 0 |
| mg | 1 | 57% | 57% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 36% | 83% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 5.580 ms | 3.01 | 24.06 | 62% | ok |
| blas1 | dot | f32 | n=8388608 | 4.214 ms | 3.98 | 15.93 | 41% | ok |
| blas1 | axpy | f64 | n=8388608 | 6.438 ms | 2.61 | 31.27 | 81% | ok |
| blas1 | axpy | f32 | n=8388608 | 4.930 ms | 3.40 | 20.42 | 53% | ok |
| blas1 | scale | f64 | n=8388608 | 3.570 ms | 2.35 | 37.59 | 98% | ok |
| blas1 | two_norm | f64 | n=8388608 | 4.827 ms | 3.48 | 13.90 | 36% | ok |
| blas1 | one_norm | f64 | n=8388608 | 3.978 ms | 2.11 | 16.87 | 44% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 7.029 ms | 1.19 | 9.55 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 3.947 ms | 2.13 | 17.00 | 44% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 76.460 ms | 2.41 | 1.76 | 5% | ok |
| blas2 | gemv | f64 | n=4096 | 3.489 ms | 9.62 | 38.49 | 100% | ok |
| blas2 | gemv | f32 | n=4096 | 1.649 ms | 20.35 | 40.72 | 106% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 3.610 ms | 9.30 | 37.20 | 97% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.635 ms | 20.52 | 41.06 | 107% | ok |
| blas2 | symv | f64 | n=4096 | 1.916 ms | 17.52 | 35.07 | 91% | ok |
| blas2 | trmv | f64 | n=4096 | 1.851 ms | 9.06 | 36.29 | 94% | ok |
| blas2 | trmv | f32 | n=4096 | 0.943 ms | 17.80 | 35.63 | 92% | ok |
| blas2 | trsv | f64 | n=4096 | 1.785 ms | 9.40 | 37.63 | 98% | ok |
| blas2 | trsv | f32 | n=4096 | 0.872 ms | 19.24 | 38.52 | 100% | ok |
| blas2 | ger | f64 | n=4096 | 7.664 ms | 4.38 | 35.03 | 91% | ok |
| blas3 | gemm | f64 | 512x512x512 | 3.704 ms | 72.47 | 1.70 | 93% | ok |
| blas3 | gemm | f32 | 512x512x512 | 1.832 ms | 146.56 | 1.72 | 94% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 2.578 ms | 52.07 | 2.44 | 66% | ok |
| blas3 | symm | f64 | 512x512x512 | 3.711 ms | 72.33 | 1.70 | 92% | ok |
| blas3 | syrk | f64 | 512x512x512 | 2.546 ms | 52.72 | 1.65 | 67% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 4.744 ms | 56.58 | 1.33 | 72% | ok |
| blas3 | trmm | f64 | 512x512x512 | 2.113 ms | 63.52 | 2.48 | 81% | ok |
| blas3 | trsm | f64 | 512x512x512 | 2.134 ms | 62.91 | 2.46 | 80% | ok |
| blas3 | trsm | f32 | 512x512x512 | 0.963 ms | 139.32 | 2.72 | 89% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 13.096 ms | 0.41 | 3.24 | 8% | ok |
| factor | lu | f64 | 512x512 | 3.339 ms | 26.80 | 1.26 | 34% | ok |
| factor | lu | f32 | 512x512 | 1.781 ms | 50.25 | 1.18 | 32% | ok |
| factor | cholesky | f64 | 512x512 | 2.050 ms | 21.83 | 2.05 | 28% | ok |
| factor | cholesky | f32 | 512x512 | 1.304 ms | 34.30 | 1.61 | 22% | ok |
| factor | ldlt | f64 | 512x512 | 7.465 ms | 5.99 | 0.56 | 8% | ok |
| factor | ldlt_bk | f64 | 512x512 | 46.657 ms | 0.96 | 0.09 | 1% | ok |
| factor | qr | f64 | 512x512 | 5.750 ms | 31.12 | 0.73 | 40% | ok |
| factor | lq | f64 | 512x512 | 25.573 ms | 7.00 | 0.16 | 9% | ok |
| factor | hessenberg | f64 | 512x512 | 268.325 ms | 1.67 | 0.02 | 2% | ok |
| factor | householder | f64 | 512x512 | 1.633 ms | 0.64 | 2.57 | 7%\* | ok |
| factor | inv | f64 | 512x512 | 45.725 ms | 5.87 | 0.09 | 7% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 3.786 ms | 23.63 | 1.11 | 30% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.699 ms | 13.16 | 0.62 | 17% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 58.924 ms | 2.56 | 0.02 | 3% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 769.140 ms | 0.20 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 16.021 ms | 10.47 | 0.07 | 13% | ok |
| eigen | schur | f64 | 256x256 | 21.208 ms | 19.78 | 0.05 | 25% | ok |
| eigen | svd | f64 | 256x256 | 6.675 ms | 55.29 | 0.16 | 71% | ok |
| eigen | expm | f64 | 256x256 | 11.222 ms | 22.43 | 0.09 | 29% | ok |
| eigen | sylvester | f64 | 256x256 | 53.574 ms | 18.79 | 0.03 | 24% | ok |
| elementwise | exp | f64 | n=8388608 | 39.227 ms | 0.21 | 3.42 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 45.541 ms | 0.18 | 2.95 | 8% | ok |
| elementwise | sin | f64 | n=8388608 | 59.963 ms | 0.14 | 2.24 | 6% | ok |
| elementwise | sqrt | f64 | n=8388608 | 28.845 ms | 0.29 | 4.65 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 201.570 ms | 0.04 | 0.67 | 2% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.741 ms | 3.53 | 36.75 | 95% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.930 ms | 2.81 | 29.27 | 76% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 4.769 ms | 8.78 | 18.46 | 48% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.878 ms | 0.67 | 2.36 | 6%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.244 ms | 1.61 | 13.38 | 35%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 91.045 ms | 0.12 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.539 ms | 0.44 | 0.03 | 1% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.916 ms | 1.25 | 0.08 | 2% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.490 ms | 0.57 | 0.02 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 53.607 ms | 0.42 | 0.01 | 1% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.784 ms | 1.33 | 0.04 | 2% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 99.843 ms | 0.02 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.384 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.968 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.221 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 15.133 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.361 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 88.448 ms | 0.13 | 0.01 | 0% | ok |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.246 ms | 3.07 | 12.30 | 32%\* | ok |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.526 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.183 ms | 2.67 | 0.17 | 3% | ok |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.248 ms | 3.05 | 12.19 | 32%\* | ok |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.090 ms | 0.00 | 0.00 | n/a | ok |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.413 ms | 0.63 | 0.02 | 1% | ok |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.643 ms | 1.18 | 4.71 | 12%\* | ok |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.929 ms | 0.00 | 0.00 | n/a | ok |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.630 ms | 0.47 | 0.02 | 1% | ok |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.666 ms | 1.14 | 4.54 | 12%\* | ok |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 5.637 ms | 1.98 | 0.13 | 3% | ok |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 1.882 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 3.385 ms | 3.30 | 0.22 | 4% | ok |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 0.327 ms | 2.31 | 9.25 | 24%\* | ok |
| direct | ref_pardiso_spd | f64 | 96^2 grid, nnz 45696 | 4.524 ms | 2.47 | 0.16 | 3% | ok |
| direct | ref_pardiso_spd.analyze | f64 | 96^2 grid, nnz 45696 | 2.110 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_pardiso_spd.factor | f64 | 96^2 grid, nnz 45696 | 1.901 ms | 5.88 | 0.38 | 8% | ok |
| direct | ref_pardiso_spd.solve | f64 | 96^2 grid, nnz 45696 | 0.462 ms | 1.64 | 6.55 | 17%\* | ok |
| direct | ref_pardiso_lu | f64 | 96^2 grid, nnz 45696 | 7.448 ms | 3.00 | 0.10 | 4% | ok |
| direct | ref_pardiso_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.591 ms | 0.00 | 0.00 | n/a | ok |
| direct | ref_pardiso_lu.factor | f64 | 96^2 grid, nnz 45696 | 3.223 ms | 6.94 | 0.23 | 9% | ok |
| direct | ref_pardiso_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.575 ms | 1.32 | 5.26 | 14%\* | ok |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 9.497 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 7433.298 ms | 0.04 | 0.00 | 0% | ok |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.416 ms | 2.38 | 9.52 | 25%\* | ok |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 18.719 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.644 ms | 4.43 | 0.01 | 6% | ok |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.444 ms | 2.33 | 9.33 | 24%\* | ok |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.878 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 551.574 ms | 1.12 | 0.00 | 1% | ok |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.505 ms | 0.45 | 1.80 | 5%\* | ok |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.629 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 666.424 ms | 0.93 | 0.00 | 1% | ok |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 7.509 ms | 0.45 | 1.79 | 5%\* | ok |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 3.912 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 11.397 ms | 27.08 | 0.08 | 35% | ok |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 0.501 ms | 6.72 | 26.88 | 70%\* | ok |
| direct3d | ref_pardiso_spd.analyze | f64 | 20^3 grid, nnz 53600 | 7.942 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_pardiso_spd.factor | f64 | 20^3 grid, nnz 53600 | 9.680 ms | 31.88 | 0.09 | 41% | ok |
| direct3d | ref_pardiso_spd.solve | f64 | 20^3 grid, nnz 53600 | 0.870 ms | 3.87 | 15.49 | 40%\* | ok |
| direct3d | ref_pardiso_lu.analyze | f64 | 20^3 grid, nnz 53600 | 9.045 ms | 0.00 | 0.00 | n/a | ok |
| direct3d | ref_pardiso_lu.factor | f64 | 20^3 grid, nnz 53600 | 19.119 ms | 32.28 | 0.04 | 41% | ok |
| direct3d | ref_pardiso_lu.solve | f64 | 20^3 grid, nnz 53600 | 1.052 ms | 3.20 | 12.81 | 33%\* | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 31.158 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 33.001 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.926 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.557 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 15.047 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 14.058 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.718 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.410 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 14.360 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 17.264 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.550 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.946 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.225 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.036 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 1.565 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 12.248 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.073 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.015 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.022 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.170 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.976 ms | 2.68 | 30.04 | 78% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.217 ms | 2.15 | 24.09 | 62% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.562 ms | 1.68 | 18.78 | 49% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.238 ms | 2.90 | 21.96 | 57% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.869 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.238 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 34.205 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 5.054 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 4.850 ms | 3.46 | 13.84 | 36% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 38.939 ms | 6.89 | 0.11 | 9% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.554 ms | 30.26 | 32.15 | 83%\* | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.294 ms | 7.31 | 29.26 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.996 ms | 16.84 | 33.69 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.721 ms | 9.75 | 9.75 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.720 ms | 9.75 | 9.75 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 17.442 ms | 15.39 | 0.18 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 20.845 ms | 12.88 | 0.10 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 10.220 ms | 26.27 | 0.15 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 10.219 ms | 26.27 | 0.15 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 26.814 ms | 10.01 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

## i7-12700k -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG`, SIMD scalar |
| commit | `8867b44b3f52` |
| FMA peak (ceiling) | f64 601.0 GFLOP/s, f32 1201.6 GFLOP/s |
| ... hardware, AVX2+FMA | f64 601.0, f32 1201.6 GFLOP/s |
| ... this build, scalar | f64 10.0, f32 10.0 GFLOP/s |
| bandwidth | read 82.6, update 77.0, triad 75.8 GB/s (ceiling 82.6) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 85% | 124% | 0 | 28% |
| blas2 | 10 | 119% | 198% | 0 | 32% |
| blas3 | 10 | 66% | 85% | 0 | 77% |
| factor | 12 | 5% | 12% | 0 | 17% |
| eigen | 8 | 3% | 10% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 37% | 289% | 0 | 13% |
| direct | 31 | 1% | 14% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| direct3d | 21 | 11% | 32% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 30% | 37% | 0 | 13% |
| mg | 1 | 26% | 26% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 17% | 40% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 84% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.601 ms | 10.48 | 83.81 | 101% | ok | 3.5x (44%) |
| blas1 | dot | f32 | n=8388608 | 0.655 ms | 25.63 | 102.53 | 124% | ok | 6.4x (80%) |
| blas1 | axpy | f64 | n=8388608 | 2.851 ms | 5.88 | 70.61 | 85% | ok | 2.3x (28%) |
| blas1 | axpy | f32 | n=8388608 | 1.439 ms | 11.66 | 69.95 | 85% | ok | 3.4x (43%) |
| blas1 | scale | f64 | n=8388608 | 1.915 ms | 4.38 | 70.08 | 85% | ok | 1.9x (23%) |
| blas1 | two_norm | f64 | n=8388608 | 0.737 ms | 22.76 | 91.06 | 110% | ok | 6.5x (82%) |
| blas1 | one_norm | f64 | n=8388608 | 3.923 ms | 2.14 | 17.11 | 21% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.902 ms | 1.22 | 9.72 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.895 ms | 2.15 | 17.23 | 21% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 75.114 ms | 2.46 | 1.79 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.429 ms | 23.48 | 93.98 | 114% | ok | 2.4x (31%) |
| blas2 | gemv | f32 | n=4096 | 0.544 ms | 61.73 | 123.52 | 149% | ok | 3.0x (38%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.422 ms | 23.60 | 94.46 | 114% | ok | 2.5x (32%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.683 ms | 49.14 | 98.34 | 119% | ok | 2.4x (30%) |
| blas2 | symv | f64 | n=4096 | 0.630 ms | 53.29 | 106.69 | 129% | ok | 3.0x (38%) |
| blas2 | trmv | f64 | n=4096 | 0.599 ms | 28.00 | 112.12 | 136% | ok | 3.1x (39%) |
| blas2 | trmv | f32 | n=4096 | 0.205 ms | 81.81 | 163.77 | 198% ! | ok | 4.6x (57%) |
| blas2 | trsv | f64 | n=4096 | 1.752 ms | 9.58 | 38.35 | 46% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.852 ms | 19.69 | 39.43 | 48% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.707 ms | 4.35 | 34.84 | 42% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 0.544 ms | 493.50 | 11.57 | 82% | ok | 6.8x (85%) |
| blas3 | gemm | f32 | 512x512x512 | 0.264 ms | 1016.35 | 11.91 | 85% | ok | 6.9x (87%) |
| blas3 | gemmt | f64 | 512x512x512 | 0.700 ms | 191.78 | 8.99 | 32% | ok | 3.7x (46%) |
| blas3 | symm | f64 | 512x512x512 | 0.544 ms | 493.72 | 11.57 | 82% | ok | 6.8x (85%) |
| blas3 | syrk | f64 | 512x512x512 | 0.748 ms | 179.34 | 5.60 | 30% | ok | 3.4x (43%) |
| blas3 | syr2k | f64 | 512x512x512 | 1.117 ms | 240.27 | 5.63 | 40% | ok | 4.2x (53%) |
| blas3 | trmm | f64 | 512x512x512 | 0.341 ms | 393.22 | 15.36 | 65% | ok | 6.2x (77%) |
| blas3 | trsm | f64 | 512x512x512 | 0.339 ms | 395.69 | 15.46 | 66% | ok | 6.3x (79%) |
| blas3 | trsm | f32 | 512x512x512 | 0.159 ms | 843.82 | 16.48 | 70% | ok | 6.1x (76%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.901 ms | 0.41 | 3.29 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 2.095 ms | 42.71 | 2.00 | 7% | ok | 1.6x (20%) |
| factor | lu | f32 | 512x512 | 1.314 ms | 68.10 | 1.60 | 6% | ok | 1.4x (17%) |
| factor | cholesky | f64 | 512x512 | 1.523 ms | 29.38 | 2.75 | 5% | ok | 1.3x (17%) |
| factor | cholesky | f32 | 512x512 | 1.003 ms | 44.62 | 2.09 | 4% | ok | 1.3x (16%) |
| factor | ldlt | f64 | 512x512 | 7.200 ms | 6.21 | 0.58 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 43.946 ms | 1.02 | 0.10 | 0% | ok | 1.1x (13%) |
| factor | qr | f64 | 512x512 | 4.098 ms | 43.67 | 1.02 | 7% | ok | 1.4x (18%) |
| factor | lq | f64 | 512x512 | 17.570 ms | 10.19 | 0.24 | 2% | ok | 1.5x (18%) |
| factor | hessenberg | f64 | 512x512 | 166.554 ms | 2.69 | 0.03 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 0.426 ms | 2.47 | 9.85 | 12%\* | ok | 3.8x (48%) |
| factor | inv | f64 | 512x512 | 45.830 ms | 5.86 | 0.09 | 1% | ok | 1.0x (12%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 2.779 ms | 32.20 | 1.51 | 5% | ok | 1.4x (17%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 1.642 ms | 13.62 | 0.64 | 2% | ok | 1.0x (13%) |
| eigen | eigen_symmetric | f64 | 256x256 | 57.566 ms | 2.62 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 731.209 ms | 0.21 | 0.00 | 0% | ok | 1.1x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 17.436 ms | 9.62 | 0.06 | 2% | ok | 0.9x (11%) |
| eigen | schur | f64 | 256x256 | 21.723 ms | 19.31 | 0.05 | 3% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 6.123 ms | 60.28 | 0.17 | 10% | ok | 1.1x (14%) |
| eigen | expm | f64 | 256x256 | 8.698 ms | 28.93 | 0.12 | 5% | ok | 1.3x (16%) |
| eigen | sylvester | f64 | 256x256 | 57.295 ms | 17.57 | 0.03 | 3% | ok | 0.9x (12%) |
| elementwise | exp | f64 | n=8388608 | 39.539 ms | 0.21 | 3.39 | 4% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 45.051 ms | 0.19 | 2.98 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 59.543 ms | 0.14 | 2.25 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.787 ms | 0.29 | 4.66 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 201.101 ms | 0.04 | 0.67 | 1% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.114 ms | 22.94 | 238.69 | 289% ! | ok | 6.5x (81%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.903 ms | 2.90 | 30.16 | 37% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.405 ms | 29.80 | 62.65 | 76% | ok | 3.4x (42%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.841 ms | 0.67 | 2.38 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.379 ms | 1.46 | 12.07 | 15%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 89.809 ms | 0.12 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 25.812 ms | 0.43 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 8.880 ms | 1.26 | 0.08 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.963 ms | 0.55 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 53.166 ms | 0.42 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 17.008 ms | 1.32 | 0.04 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 85.593 ms | 0.02 | 0.00 | 0% | ok | 1.2x (15%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.280 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.874 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.215 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.992 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.287 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 87.254 ms | 0.13 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.268 ms | 2.82 | 11.29 | 14%\* | ok | 0.9x (11%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.389 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.123 ms | 2.71 | 0.18 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.262 ms | 2.88 | 11.53 | 14%\* | ok | 0.9x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.041 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 36.359 ms | 0.62 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.598 ms | 1.26 | 5.06 | 6%\* | ok | 1.1x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.809 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 47.240 ms | 0.47 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.603 ms | 1.25 | 5.02 | 6%\* | ok | 1.1x (14%) |
| direct | ref_cholmod | f64 | 96^2 grid, nnz 45696 | 6.040 ms | 1.85 | 0.12 | 0% | ok | 0.9x (12%) |
| direct | ref_cholmod.analyze | f64 | 96^2 grid, nnz 45696 | 1.851 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | ref_cholmod.factor | f64 | 96^2 grid, nnz 45696 | 3.686 ms | 3.04 | 0.20 | 1% | ok | 0.9x (11%) |
| direct | ref_cholmod.solve | f64 | 96^2 grid, nnz 45696 | 0.328 ms | 2.30 | 9.21 | 11%\* | ok | 1.0x (12%) |
| direct | ref_pardiso_spd | f64 | 96^2 grid, nnz 45696 | 3.301 ms | 3.39 | 0.22 | 1% | ok | 1.4x (17%) |
| direct | ref_pardiso_spd.analyze | f64 | 96^2 grid, nnz 45696 | 1.994 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct | ref_pardiso_spd.factor | f64 | 96^2 grid, nnz 45696 | 0.630 ms | 17.76 | 1.16 | 3% | ok | 3.0x (38%) |
| direct | ref_pardiso_spd.solve | f64 | 96^2 grid, nnz 45696 | 0.499 ms | 1.52 | 6.06 | 7%\* | ok | 0.9x (12%) |
| direct | ref_pardiso_lu | f64 | 96^2 grid, nnz 45696 | 5.611 ms | 3.99 | 0.13 | 1% | ok | 1.3x (17%) |
| direct | ref_pardiso_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.525 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | ref_pardiso_lu.factor | f64 | 96^2 grid, nnz 45696 | 1.264 ms | 17.69 | 0.58 | 3% | ok | 2.5x (32%) |
| direct | ref_pardiso_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.644 ms | 1.17 | 4.70 | 6%\* | ok | 0.9x (11%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 9.307 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 7283.432 ms | 0.04 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.532 ms | 2.20 | 8.80 | 11%\* | ok | 0.9x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 18.276 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 65.265 ms | 4.73 | 0.01 | 1% | ok | 1.1x (13%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.550 ms | 2.17 | 8.69 | 11%\* | ok | 0.9x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.775 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 540.304 ms | 1.14 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 8.052 ms | 0.42 | 1.67 | 2%\* | ok | 0.9x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.365 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 617.762 ms | 1.00 | 0.00 | 0% | ok | 1.1x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 8.045 ms | 0.42 | 1.68 | 2%\* | ok | 0.9x (12%) |
| direct3d | ref_cholmod.analyze | f64 | 20^3 grid, nnz 53600 | 3.804 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | ref_cholmod.factor | f64 | 20^3 grid, nnz 53600 | 8.388 ms | 36.79 | 0.10 | 6% | ok | 1.4x (17%) |
| direct3d | ref_cholmod.solve | f64 | 20^3 grid, nnz 53600 | 0.512 ms | 6.58 | 26.31 | 32%\* | ok | 1.0x (12%) |
| direct3d | ref_pardiso_spd.analyze | f64 | 20^3 grid, nnz 53600 | 7.277 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | ref_pardiso_spd.factor | f64 | 20^3 grid, nnz 53600 | 2.379 ms | 129.70 | 0.36 | 22% | ok | 4.1x (51%) |
| direct3d | ref_pardiso_spd.solve | f64 | 20^3 grid, nnz 53600 | 0.945 ms | 3.57 | 14.26 | 17%\* | ok | 0.9x (12%) |
| direct3d | ref_pardiso_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.136 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | ref_pardiso_lu.factor | f64 | 20^3 grid, nnz 53600 | 5.572 ms | 110.77 | 0.15 | 18% | ok | 3.4x (43%) |
| direct3d | ref_pardiso_lu.solve | f64 | 20^3 grid, nnz 53600 | 1.145 ms | 2.94 | 11.77 | 14%\* | ok | 0.9x (11%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 30.447 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 32.039 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 22.389 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 12.219 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 14.623 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.705 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 27.375 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 24.747 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.934 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 16.832 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 16.052 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.537 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 20.877 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.034 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.532 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 12.008 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.049 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.984 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.163 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.959 ms | 2.73 | 30.59 | 37% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.196 ms | 2.19 | 24.51 | 30% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.534 ms | 1.71 | 19.12 | 23% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 13.764 ms | 2.79 | 21.12 | 26% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 29.662 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 31.895 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.199 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.889 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 4.793 ms | 3.50 | 14.00 | 17% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 5.162 ms | 52.00 | 0.81 | 9% | ok | 7.5x (94%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.543 ms | 30.91 | 32.84 | 40%\* | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 2.260 ms | 7.42 | 29.70 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.976 ms | 17.19 | 34.39 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.683 ms | 9.97 | 9.97 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.683 ms | 9.97 | 9.97 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.599 ms | 103.29 | 1.21 | n/a | ok | 6.7x (84%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.747 ms | 97.73 | 0.76 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.355 ms | 198.06 | 1.16 | n/a | ok | 7.5x (94%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 1.356 ms | 197.97 | 1.16 | n/a | ok | 7.5x (94%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 3.549 ms | 75.64 | 0.44 | n/a | ok | 7.6x (94%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

