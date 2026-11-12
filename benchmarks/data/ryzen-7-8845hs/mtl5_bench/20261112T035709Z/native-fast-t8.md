## ryzen-7-8845hs -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `cd6762a6e989` |
| FMA peak (ceiling) | f64 625.1 GFLOP/s, f32 1250.1 GFLOP/s |
| ... hardware, AVX-512F | f64 625.1, f32 1250.1 GFLOP/s |
| ... this build, SSE2 | f64 312.8, f32 624.6 GFLOP/s |
| bandwidth | read 53.2, update 53.3, triad 41.3 GB/s (ceiling 53.3) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 111% | 127% | 0 | 12% |
| blas2 | 10 | 57% | 115% | 0 | 12% |
| blas3 | 10 | 4% | 13% | 0 | 20% |
| factor | 12 | 1% | 8% | 0 | 18% |
| eigen | 8 | 0% | 2% | 0 | 13% |
| elementwise | 5 | 8% | 12% | 0 | 12% |
| sparse | 5 | 40% | 249% | 0 | 13% |
| direct | 7 | 0% | 0% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 37% | 37% | 0 | 13% |
| mg | 1 | 35% | 35% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 70% | 70% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 69% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.266 ms | 7.40 | 59.24 | 111% | ok | 1.1x (13%) |
| blas1 | dot | f32 | n=8388608 | 0.991 ms | 16.94 | 67.75 | 127% | ok | 1.2x (15%) |
| blas1 | axpy | f64 | n=8388608 | 3.548 ms | 4.73 | 56.75 | 106% | ok | 1.0x (12%) |
| blas1 | axpy | f32 | n=8388608 | 1.678 ms | 10.00 | 59.98 | 112% | ok | 1.0x (12%) |
| blas1 | scale | f64 | n=8388608 | 2.134 ms | 3.93 | 62.89 | 118% | ok | 0.9x (11%) |
| blas1 | two_norm | f64 | n=8388608 | 1.046 ms | 16.04 | 64.18 | 120% | ok | 1.1x (14%) |
| blas1 | one_norm | f64 | n=8388608 | 5.292 ms | 1.59 | 12.68 | 24% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.098 ms | 1.65 | 13.16 | 25% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 5.143 ms | 1.63 | 13.05 | 24% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 97.716 ms | 1.89 | 1.37 | 3% | ok | 0.9x (12%) |
| blas2 | gemv | f64 | n=4096 | 2.347 ms | 14.29 | 57.21 | 107% | ok | 1.4x (18%) |
| blas2 | gemv | f32 | n=4096 | 1.095 ms | 30.66 | 61.34 | 115% | ok | 1.6x (19%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 9.513 ms | 3.53 | 14.12 | 26% | ok | 2.3x (28%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 4.853 ms | 6.91 | 13.83 | 26% | ok | 1.6x (20%) |
| blas2 | symv | f64 | n=4096 | 9.195 ms | 3.65 | 7.31 | 14% | ok | 0.9x (12%) |
| blas2 | trmv | f64 | n=4096 | 2.167 ms | 7.74 | 31.00 | 58% | ok | 0.8x (11%) |
| blas2 | trmv | f32 | n=4096 | 1.121 ms | 14.96 | 29.96 | 56% | ok | 0.8x (11%) |
| blas2 | trsv | f64 | n=4096 | 2.149 ms | 7.81 | 31.25 | 59% | ok | 0.9x (11%) |
| blas2 | trsv | f32 | n=4096 | 1.101 ms | 15.24 | 30.50 | 57% | ok | 0.8x (10%) |
| blas2 | ger | f64 | n=4096 | 17.212 ms | 1.95 | 15.60 | 29% | ok | 0.8x (10%) |
| blas3 | gemm | f64 | 512x512x512 | 3.236 ms | 82.95 | 1.94 | 13% | ok | 3.4x (42%) |
| blas3 | gemm | f32 | 512x512x512 | 1.822 ms | 147.31 | 1.73 | 12% | ok | 3.1x (38%) |
| blas3 | gemmt | f64 | 512x512x512 | 5.034 ms | 26.66 | 1.25 | 4% | ok | 1.5x (18%) |
| blas3 | symm | f64 | 512x512x512 | 3.734 ms | 71.89 | 1.68 | 12% | ok | 3.2x (40%) |
| blas3 | syrk | f64 | 512x512x512 | 14.165 ms | 9.48 | 0.30 | 2% | ok | 1.0x (12%) |
| blas3 | syr2k | f64 | 512x512x512 | 24.116 ms | 11.13 | 0.26 | 2% | ok | 1.0x (12%) |
| blas3 | trmm | f64 | 512x512x512 | 5.917 ms | 22.68 | 0.89 | 4% | ok | 1.6x (21%) |
| blas3 | trsm | f64 | 512x512x512 | 6.195 ms | 21.66 | 0.85 | 3% | ok | 1.6x (20%) |
| blas3 | trsm | f32 | 512x512x512 | 4.605 ms | 29.15 | 0.57 | 2% | ok | 1.4x (18%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.264 ms | 0.57 | 4.58 | 9% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 23.200 ms | 3.86 | 0.18 | 1% | ok | 1.4x (18%) |
| factor | lu | f32 | 512x512 | 22.390 ms | 4.00 | 0.09 | 0% | ok | 1.4x (18%) |
| factor | cholesky | f64 | 512x512 | 11.397 ms | 3.93 | 0.37 | 1%\* | ok | 1.0x (12%) |
| factor | cholesky | f32 | 512x512 | 10.906 ms | 4.10 | 0.19 | 0%\* | ok | 1.0x (12%) |
| factor | ldlt | f64 | 512x512 | 11.437 ms | 3.91 | 0.37 | 1%\* | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 165.245 ms | 0.27 | 0.03 | 0%\* | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 395.043 ms | 0.45 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 44.608 ms | 4.01 | 0.09 | 1% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 470.760 ms | 0.95 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.021 ms | 1.03 | 4.11 | 8%\* | ok | 3.9x (49%) |
| factor | inv | f64 | 512x512 | 79.181 ms | 3.39 | 0.05 | 1% | ok | 1.2x (15%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 25.187 ms | 3.55 | 0.17 | 1% | ok | 1.4x (17%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 33.577 ms | 0.67 | 0.03 | 0% | ok | 1.1x (14%) |
| eigen | eigen_symmetric | f64 | 256x256 | 123.975 ms | 1.22 | 0.01 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 496.530 ms | 0.30 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 60.808 ms | 2.76 | 0.02 | 0% | ok | 0.9x (11%) |
| eigen | schur | f64 | 256x256 | 112.942 ms | 3.71 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 118.691 ms | 3.11 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | expm | f64 | 256x256 | 19.380 ms | 12.99 | 0.05 | 2% | ok | 1.4x (17%) |
| eigen | sylvester | f64 | 256x256 | 259.498 ms | 3.88 | 0.01 | 1% | ok | 1.0x (13%) |
| elementwise | exp | f64 | n=8388608 | 27.896 ms | 0.30 | 4.81 | 9% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 42.556 ms | 0.20 | 3.15 | 6% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 39.823 ms | 0.21 | 3.37 | 6% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 21.289 ms | 0.39 | 6.30 | 12% | ok | 1.0x (12%) |
| elementwise | erf | f64 | n=8388608 | 31.967 ms | 0.26 | 4.20 | 8% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.205 ms | 12.78 | 132.96 | 249% ! | ok | 4.2x (52%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.289 ms | 2.03 | 21.12 | 40% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.288 ms | 18.30 | 38.48 | 72% | ok | 2.7x (34%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 18.861 ms | 0.17 | 0.61 | 1%\* | ok | 1.0x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.753 ms | 1.15 | 9.50 | 18%\* | ok | 1.0x (13%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 98.618 ms | 0.11 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 29.054 ms | 0.39 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 15.885 ms | 0.70 | 0.05 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 55.213 ms | 0.41 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 72.275 ms | 0.31 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 25.865 ms | 0.86 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 83.988 ms | 0.02 | 0.00 | 0%\* | ok | 1.0x (13%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.703 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.668 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.444 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.117 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 32.635 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 32.034 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 27.434 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 13.466 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 17.877 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 15.724 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 31.046 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 28.980 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 17.304 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 19.322 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 18.431 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 19.587 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 22.431 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.036 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.433 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 23.180 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ic_0 | f64 | 128^2 grid | 1.828 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 1.758 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.022 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.664 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.467 ms | 1.78 | 19.99 | 37% | ok | 1.2x (15%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.495 ms | 1.75 | 19.62 | 37% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.783 ms | 1.47 | 16.45 | 31% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.564 ms | 2.47 | 18.67 | 35% | ok | 1.0x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.454 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 46.887 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 54.576 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 6.519 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 1.786 ms | 9.39 | 37.56 | 70% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 2.635 ms | 101.86 | 1.59 | 16% | ok | 5.8x (73%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.475 ms | 35.35 | 37.56 | 70% | ok | 1.0x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.229 ms | 13.65 | 54.59 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.473 ms | 35.44 | 70.88 | n/a | ok | 1.0x (12%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.346 ms | 48.49 | 48.49 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.318 ms | 52.84 | 52.84 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.438 ms | 110.09 | 1.29 | n/a | ok | 6.1x (77%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.544 ms | 105.51 | 0.82 | n/a | ok | 6.8x (85%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 2.569 ms | 104.51 | 0.61 | n/a | ok | 6.9x (86%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.674 ms | 398.45 | 2.33 | n/a | ok | 5.5x (69%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.672 ms | 399.16 | 2.34 | n/a | ok | 5.5x (69%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

