## ryzen-9-8945hs -- blas, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 9 8945HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD scalar |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 504.9 GFLOP/s, f32 1000.8 GFLOP/s |
| ... hardware, AVX-512F | f64 504.9, f32 1000.8 GFLOP/s |
| ... this build, scalar | f64 12.3, f32 14.4 GFLOP/s |
| bandwidth | read 52.5, update 53.1, triad 40.7 GB/s (ceiling 53.1) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 81% | 118% | 0 | 53% |
| blas2 | 10 | 100% | 116% | 0 | 16% |
| blas3 | 10 | 45% | 78% | 0 | 68% |
| factor | 12 | 2% | 6% | 0 | 12% |
| eigen | 8 | 2% | 5% | 0 | 12% |
| elementwise | 5 | 6% | 10% | 0 | 13% |
| sparse | 5 | 28% | 98% | 0 | 13% |
| direct | 7 | 0% | 0% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 27% | 30% | 0 | 12% |
| mg | 1 | 26% | 26% | 0 | 11% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 11% | 18% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 95% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.352 ms | 7.13 | 57.06 | 107% | ok | 6.1x (77%) |
| blas1 | dot | f32 | n=8388608 | 1.566 ms | 10.71 | 42.86 | 81% | ok | 8.0x (100%) |
| blas1 | axpy | f64 | n=8388608 | 3.571 ms | 4.70 | 56.37 | 106% | ok | 4.2x (53%) |
| blas1 | axpy | f32 | n=8388608 | 1.813 ms | 9.25 | 55.52 | 104% | ok | 7.1x (88%) |
| blas1 | scale | f64 | n=8388608 | 2.146 ms | 3.91 | 62.54 | 118% | ok | 1.2x (15%) |
| blas1 | two_norm | f64 | n=8388608 | 1.701 ms | 9.86 | 39.46 | 74% | ok | 7.9x (98%) |
| blas1 | one_norm | f64 | n=8388608 | 6.397 ms | 1.31 | 10.49 | 20% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.423 ms | 1.31 | 10.45 | 20% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 6.394 ms | 1.31 | 10.50 | 20% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 111.408 ms | 1.66 | 1.20 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 2.538 ms | 13.22 | 52.92 | 100% | ok | 1.5x (18%) |
| blas2 | gemv | f32 | n=4096 | 1.177 ms | 28.50 | 57.02 | 107% | ok | 1.3x (16%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.709 ms | 12.39 | 49.57 | 93% | ok | 1.3x (16%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.319 ms | 25.43 | 50.88 | 96% | ok | 1.2x (15%) |
| blas2 | symv | f64 | n=4096 | 1.255 ms | 26.73 | 53.52 | 101% | ok | 1.7x (21%) |
| blas2 | trmv | f64 | n=4096 | 1.256 ms | 13.36 | 53.48 | 101% | ok | 1.5x (19%) |
| blas2 | trmv | f32 | n=4096 | 0.543 ms | 30.91 | 61.89 | 116% | ok | 1.8x (22%) |
| blas2 | trsv | f64 | n=4096 | 1.738 ms | 9.65 | 38.65 | 73% | ok | 1.1x (14%) |
| blas2 | trsv | f32 | n=4096 | 0.944 ms | 17.78 | 35.59 | 67% | ok | 1.0x (12%) |
| blas2 | ger | f64 | n=4096 | 16.304 ms | 2.06 | 16.47 | 31% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 0.679 ms | 395.11 | 9.26 | 78% | ok | 6.7x (84%) |
| blas3 | gemm | f32 | 512x512x512 | 0.386 ms | 695.25 | 8.15 | 69% | ok | 6.2x (77%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.665 ms | 80.59 | 3.78 | 16% | ok | 2.1x (26%) |
| blas3 | symm | f64 | 512x512x512 | 0.680 ms | 394.87 | 9.25 | 78% | ok | 7.0x (88%) |
| blas3 | syrk | f64 | 512x512x512 | 2.002 ms | 67.03 | 2.09 | 13% | ok | 1.9x (23%) |
| blas3 | syr2k | f64 | 512x512x512 | 2.704 ms | 99.29 | 2.33 | 20% | ok | 2.3x (28%) |
| blas3 | trmm | f64 | 512x512x512 | 0.455 ms | 295.05 | 11.53 | 58% | ok | 5.2x (65%) |
| blas3 | trsm | f64 | 512x512x512 | 0.619 ms | 216.86 | 8.47 | 43% | ok | 5.7x (71%) |
| blas3 | trsm | f32 | 512x512x512 | 0.301 ms | 445.91 | 8.71 | 45% | ok | 5.4x (68%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 10.934 ms | 0.49 | 3.88 | 7% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 11.301 ms | 7.92 | 0.37 | 2% | ok | 0.6x (7%) |
| factor | lu | f32 | 512x512 | 5.086 ms | 17.59 | 0.41 | 2% | ok | 0.6x (8%) |
| factor | cholesky | f64 | 512x512 | 6.892 ms | 6.49 | 0.61 | 1% | ok | 0.8x (10%) |
| factor | cholesky | f32 | 512x512 | 2.448 ms | 18.28 | 0.86 | 2% | ok | 0.9x (11%) |
| factor | ldlt | f64 | 512x512 | 14.861 ms | 3.01 | 0.28 | 1% | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 208.001 ms | 0.22 | 0.02 | 0% | ok | 1.0x (12%) |
| factor | qr | f64 | 512x512 | 14.186 ms | 12.62 | 0.30 | 2% | ok | 0.9x (11%) |
| factor | lq | f64 | 512x512 | 54.589 ms | 3.28 | 0.08 | 1% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 581.926 ms | 0.77 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.262 ms | 0.83 | 3.32 | 6%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 82.227 ms | 3.26 | 0.05 | 1% | ok | 0.9x (12%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 8.890 ms | 10.06 | 0.47 | 2% | ok | 0.8x (10%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 4.354 ms | 5.14 | 0.24 | 1% | ok | 0.5x (6%) |
| eigen | eigen_symmetric | f64 | 256x256 | 164.264 ms | 0.92 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 632.834 ms | 0.24 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 25.176 ms | 6.66 | 0.04 | 1% | ok | 0.8x (10%) |
| eigen | schur | f64 | 256x256 | 31.292 ms | 13.40 | 0.03 | 3% | ok | 0.9x (11%) |
| eigen | svd | f64 | 256x256 | 13.562 ms | 27.22 | 0.08 | 5% | ok | 0.8x (10%) |
| eigen | expm | f64 | 256x256 | 21.416 ms | 11.75 | 0.05 | 2% | ok | 0.9x (12%) |
| eigen | sylvester | f64 | 256x256 | 99.564 ms | 10.11 | 0.02 | 2% | ok | 0.9x (12%) |
| elementwise | exp | f64 | n=8388608 | 33.478 ms | 0.25 | 4.01 | 8% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 53.352 ms | 0.16 | 2.52 | 5% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 47.937 ms | 0.17 | 2.80 | 5% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 26.436 ms | 0.32 | 5.08 | 10% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 42.412 ms | 0.20 | 3.16 | 6% | ok | 1.0x (13%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.523 ms | 5.00 | 52.07 | 98% | ok | 1.8x (22%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.817 ms | 1.44 | 14.98 | 28% | ok | 1.1x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.344 ms | 17.86 | 37.56 | 71% | ok | 3.3x (42%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 21.551 ms | 0.15 | 0.53 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.926 ms | 1.04 | 8.65 | 16%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 126.269 ms | 0.09 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 38.507 ms | 0.29 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 23.558 ms | 0.47 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 67.254 ms | 0.33 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 90.984 ms | 0.25 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 31.943 ms | 0.70 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 101.647 ms | 0.02 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 4.604 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 6.924 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.502 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 17.587 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 50.965 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 58.304 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 39.520 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 21.854 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 28.312 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 22.720 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 69.200 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 44.074 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 25.624 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 30.761 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 29.804 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 33.434 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 41.410 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.048 ms | 0.00 | 0.00 | n/a | ok | 0.9x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.974 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 28.810 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 2.288 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ildl | f64 | 128^2 grid | 2.148 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.028 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 1.698 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.839 ms | 1.42 | 15.95 | 30% | ok | 1.0x (12%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 2.072 ms | 1.26 | 14.16 | 27% | ok | 1.0x (12%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 2.593 ms | 1.01 | 11.31 | 21% | ok | 0.9x (11%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 21.346 ms | 1.80 | 13.62 | 26% | ok | 0.9x (11%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 42.203 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 41.724 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 69.550 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 9.971 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 14.381 ms | 1.17 | 4.67 | 9% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 4.886 ms | 54.94 | 0.86 | 11% | ok | 7.7x (96%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 1.835 ms | 9.14 | 9.71 | 18% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 4.405 ms | 3.81 | 15.24 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 3.234 ms | 5.19 | 10.38 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 1.726 ms | 9.72 | 9.72 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 1.727 ms | 9.72 | 9.72 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.303 ms | 206.08 | 2.41 | n/a | ok | 7.8x (98%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 5.034 ms | 53.32 | 0.42 | n/a | ok | 7.6x (95%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 5.056 ms | 53.10 | 0.31 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 5.064 ms | 53.01 | 0.31 | n/a | ok | 7.7x (96%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 4.545 ms | 59.06 | 0.35 | n/a | ok | 7.6x (95%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

