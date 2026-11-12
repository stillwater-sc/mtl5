## ryzen-7-8845hs -- native, 1 thread, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 80.1 GFLOP/s, f32 160.2 GFLOP/s |
| ... hardware, AVX-512F | f64 80.1, f32 160.2 GFLOP/s |
| ... this build, scalar | f64 1.9, f32 1.9 GFLOP/s |
| bandwidth | read 43.3, update 53.3, triad 39.0 GB/s (ceiling 53.3) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks |
|---|---:|---:|---:|---:|
| blas1 | 10 | 25% | 110% | 0 |
| blas2 | 10 | 52% | 79% | 0 |
| blas3 | 10 | 14% | 20% | 0 |
| factor | 12 | 3% | 5% | 0 |
| eigen | 8 | 4% | 10% | 0 |
| elementwise | 5 | 8% | 12% | 0 |
| sparse | 5 | 25% | 61% | 0 |
| direct | 7 | 0% | 1% | 0 |
| ordering | 4 | n/a | n/a | 0 |
| krylov | 13 | n/a | n/a | 0 |
| pc | 7 | n/a | n/a | 0 |
| smoother | 3 | 35% | 37% | 0 |
| mg | 1 | 34% | 34% | 0 |
| sparse_eig | 4 | n/a | n/a | 0 |
| mixed | 3 | 14% | 24% | 0 |
| integer | 9 | n/a | n/a | 0 |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check |
|---|---|---|---|---:|---:|---:|---:|---|
| blas1 | dot | f64 | n=8388608 | 9.796 ms | 1.71 | 13.70 | 26% | ok |
| blas1 | dot | f32 | n=8388608 | 9.748 ms | 1.72 | 6.88 | 13% | ok |
| blas1 | axpy | f64 | n=8388608 | 11.898 ms | 1.41 | 16.92 | 32% | ok |
| blas1 | axpy | f32 | n=8388608 | 11.855 ms | 1.42 | 8.49 | 16% | ok |
| blas1 | scale | f64 | n=8388608 | 2.292 ms | 3.66 | 58.56 | 110% | ok |
| blas1 | two_norm | f64 | n=8388608 | 9.326 ms | 1.80 | 7.20 | 14% | ok |
| blas1 | one_norm | f64 | n=8388608 | 5.063 ms | 1.66 | 13.25 | 25% | ok |
| blas1 | infinity_norm | f64 | n=8388608 | 5.074 ms | 1.65 | 13.23 | 25% | ok |
| blas1 | sum | f64 | n=8388608 | 5.064 ms | 1.66 | 13.25 | 25% | ok |
| blas1 | dot_k3 | f64 | n=8388608 | 87.618 ms | 2.11 | 1.53 | 3% | ok |
| blas2 | gemv | f64 | n=4096 | 3.184 ms | 10.54 | 42.17 | 79% | ok |
| blas2 | gemv | f32 | n=4096 | 2.445 ms | 13.73 | 27.47 | 52% | ok |
| blas2 | gemv_colmajor | f64 | n=4096 | 6.646 ms | 5.05 | 20.21 | 38% | ok |
| blas2 | gemv_colmajor | f32 | n=4096 | 2.256 ms | 14.87 | 29.76 | 56% | ok |
| blas2 | symv | f64 | n=4096 | 8.555 ms | 3.92 | 7.85 | 15% | ok |
| blas2 | trmv | f64 | n=4096 | 1.855 ms | 9.04 | 36.21 | 68% | ok |
| blas2 | trmv | f32 | n=4096 | 1.543 ms | 10.88 | 21.77 | 41% | ok |
| blas2 | trsv | f64 | n=4096 | 1.772 ms | 9.47 | 37.90 | 71% | ok |
| blas2 | trsv | f32 | n=4096 | 1.573 ms | 10.67 | 21.36 | 40% | ok |
| blas2 | ger | f64 | n=4096 | 13.543 ms | 2.48 | 19.83 | 37% | ok |
| blas3 | gemm | f64 | 512x512x512 | 16.376 ms | 16.39 | 0.38 | 20% | ok |
| blas3 | gemm | f32 | 512x512x512 | 8.239 ms | 32.58 | 0.38 | 20% | ok |
| blas3 | gemmt | f64 | 512x512x512 | 11.224 ms | 11.96 | 0.56 | 15% | ok |
| blas3 | symm | f64 | 512x512x512 | 17.706 ms | 15.16 | 0.36 | 19% | ok |
| blas3 | syrk | f64 | 512x512x512 | 13.574 ms | 9.89 | 0.31 | 12% | ok |
| blas3 | syr2k | f64 | 512x512x512 | 23.489 ms | 11.43 | 0.27 | 14% | ok |
| blas3 | trmm | f64 | 512x512x512 | 25.087 ms | 5.35 | 0.21 | 7% | ok |
| blas3 | trsm | f64 | 512x512x512 | 26.572 ms | 5.05 | 0.20 | 6% | ok |
| blas3 | trsm | f32 | 512x512x512 | 25.995 ms | 5.16 | 0.10 | 3% | ok |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.015 ms | 0.59 | 4.71 | 9% | ok |
| factor | lu | f64 | 512x512 | 32.730 ms | 2.73 | 0.13 | 3% | ok |
| factor | lu | f32 | 512x512 | 30.913 ms | 2.89 | 0.07 | 2% | ok |
| factor | cholesky | f64 | 512x512 | 11.394 ms | 3.93 | 0.37 | 5% | ok |
| factor | cholesky | f32 | 512x512 | 10.910 ms | 4.10 | 0.19 | 3% | ok |
| factor | ldlt | f64 | 512x512 | 11.450 ms | 3.91 | 0.37 | 5% | ok |
| factor | ldlt_bk | f64 | 512x512 | 166.463 ms | 0.27 | 0.03 | 0% | ok |
| factor | qr | f64 | 512x512 | 645.572 ms | 0.28 | 0.01 | 0% | ok |
| factor | lq | f64 | 512x512 | 64.390 ms | 2.78 | 0.07 | 3% | ok |
| factor | hessenberg | f64 | 512x512 | 740.793 ms | 0.60 | 0.01 | 1% | ok |
| factor | householder | f64 | 512x512 | 3.918 ms | 0.27 | 1.07 | 2%\* | ok |
| factor | inv | f64 | 512x512 | 88.836 ms | 3.02 | 0.05 | 4% | ok |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 33.800 ms | 2.65 | 0.12 | 3% | ok |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 38.616 ms | 0.58 | 0.03 | 1% | ok |
| eigen | eigen_symmetric | f64 | 256x256 | 126.776 ms | 1.19 | 0.01 | 1% | ok |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 520.252 ms | 0.29 | 0.00 | 0% | ok |
| eigen | eigenvalue | f64 | 256x256 | 57.420 ms | 2.92 | 0.02 | 4% | ok |
| eigen | schur | f64 | 256x256 | 110.628 ms | 3.79 | 0.01 | 5% | ok |
| eigen | svd | f64 | 256x256 | 117.948 ms | 3.13 | 0.01 | 4% | ok |
| eigen | expm | f64 | 256x256 | 30.289 ms | 8.31 | 0.03 | 10% | ok |
| eigen | sylvester | f64 | 256x256 | 261.271 ms | 3.85 | 0.01 | 5% | ok |
| elementwise | exp | f64 | n=8388608 | 27.455 ms | 0.31 | 4.89 | 9% | ok |
| elementwise | log | f64 | n=8388608 | 41.556 ms | 0.20 | 3.23 | 6% | ok |
| elementwise | sin | f64 | n=8388608 | 38.614 ms | 0.22 | 3.48 | 7% | ok |
| elementwise | sqrt | f64 | n=8388608 | 21.302 ms | 0.39 | 6.30 | 12% | ok |
| elementwise | erf | f64 | n=8388608 | 32.112 ms | 0.26 | 4.18 | 8% | ok |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.832 ms | 3.15 | 32.74 | 61% | ok |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.381 ms | 1.90 | 19.72 | 37% | ok |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 6.517 ms | 6.43 | 13.51 | 25% | ok |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 18.294 ms | 0.18 | 0.63 | 1%\* | ok |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.786 ms | 1.12 | 9.32 | 17%\* | ok |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 104.848 ms | 0.11 | 0.01 | 0% | ok |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 31.221 ms | 0.36 | 0.02 | 0% | ok |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 18.305 ms | 0.61 | 0.04 | 1% | ok |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 54.321 ms | 0.41 | 0.01 | 1% | ok |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 73.608 ms | 0.30 | 0.01 | 0% | ok |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 25.380 ms | 0.88 | 0.03 | 1% | ok |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 75.715 ms | 0.03 | 0.00 | 0% | ok |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.599 ms | 0.00 | 0.00 | n/a | ok |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.528 ms | 0.00 | 0.00 | n/a | ok |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.419 ms | 0.00 | 0.00 | n/a | ok |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.074 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 39.256 ms | 0.00 | 0.00 | n/a | ok |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 43.754 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 30.510 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 16.668 ms | 0.00 | 0.00 | n/a | ok |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 21.684 ms | 0.00 | 0.00 | n/a | ok |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 17.593 ms | 0.00 | 0.00 | n/a | ok |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 50.784 ms | 0.00 | 0.00 | n/a | ok |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 33.841 ms | 0.00 | 0.00 | n/a | ok |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 19.777 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 23.595 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 22.770 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 25.309 ms | 0.00 | 0.00 | n/a | ok |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 30.987 ms | 0.00 | 0.00 | n/a | ok |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.036 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilu_0 | f64 | 128^2 grid | 2.383 ms | 0.00 | 0.00 | n/a | ok |
| pc | ilut | f64 | 128^2 grid | 23.169 ms | 0.00 | 0.00 | n/a | ok |
| pc | ic_0 | f64 | 128^2 grid | 1.814 ms | 0.00 | 0.00 | n/a | ok |
| pc | ildl | f64 | 128^2 grid | 1.734 ms | 0.00 | 0.00 | n/a | ok |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.663 ms | 0.00 | 0.00 | n/a | ok |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.585 ms | 1.65 | 18.50 | 35% | ok |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.493 ms | 1.75 | 19.64 | 37% | ok |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.780 ms | 1.47 | 16.47 | 31% | ok |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.830 ms | 2.43 | 18.36 | 34% | ok |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.230 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 46.726 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 56.665 ms | 0.00 | 0.00 | n/a | ok |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 7.733 ms | 0.00 | 0.00 | n/a | ok |
| mixed | dot | f32>f64 | n=8388608 | 9.321 ms | 1.80 | 7.20 | 14% | ok |
| mixed | gemm | f32>f64 | 512x512x512 | 29.049 ms | 9.24 | 0.14 | 12% | ok |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.377 ms | 12.19 | 12.95 | 24% | ok |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 3.123 ms | 5.37 | 21.49 | n/a | ok |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 2.879 ms | 5.83 | 11.66 | n/a | ok |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.382 ms | 12.14 | 12.14 | n/a | ok |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.381 ms | 12.15 | 12.15 | n/a | ok |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 8.087 ms | 33.19 | 0.39 | n/a | ok |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 30.291 ms | 8.86 | 0.07 | n/a | ok |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 30.562 ms | 8.78 | 0.05 | n/a | ok |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 30.551 ms | 8.79 | 0.05 | n/a | ok |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 27.274 ms | 9.84 | 0.06 | n/a | ok |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

