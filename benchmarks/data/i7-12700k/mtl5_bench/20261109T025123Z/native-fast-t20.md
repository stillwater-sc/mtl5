## 12th Gen Intel(R) Core(TM) i7-12700K -- native-fast, 20 threads, class medium

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-43-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `e7497e6e4c47` |
| FMA peak (ceiling) | f64 668.5 GFLOP/s, f32 1260.4 GFLOP/s |
| ... hardware, AVX2+FMA | f64 629.7, f32 1259.4 GFLOP/s |
| ... this build, AVX2 | f64 668.5, f32 1260.4 GFLOP/s |
| bandwidth | read 72.6, update 71.5, triad 64.5 GB/s (ceiling 72.6) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 106% | 166% | 0 | 9% |
| blas2 | 6 | 52% | 164% | 0 | 5% |
| blas3 | 10 | 10% | 42% | 0 | 7% |
| factor | 12 | 1% | 12% | 0 | 5% |
| eigen | 8 | 1% | 3% | 0 | 5% |
| elementwise | 5 | 4% | 6% | 0 | 5% |
| sparse | 5 | 41% | 351% | 0 | 5% |
| direct | 7 | 0% | 0% | 0 | 5% |
| ordering | 4 | n/a | n/a | 0 | 5% |
| krylov | 13 | n/a | n/a | 0 | 5% |
| pc | 7 | n/a | n/a | 0 | 5% |
| smoother | 3 | 31% | 42% | 0 | 5% |
| mg | 1 | 26% | 26% | 0 | 5% |
| sparse_eig | 4 | n/a | n/a | 0 | 5% |
| mixed | 3 | 49% | 81% | 0 | 5% |
| integer | 9 | n/a | n/a | 0 | 15% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.644 ms | 10.20 | 81.62 | 112% | ok | 2.2x (11%) |
| blas1 | dot | f32 | n=8388608 | 0.567 ms | 29.57 | 118.27 | 163% ! | ok | 3.0x (15%) |
| blas1 | axpy | f64 | n=8388608 | 2.711 ms | 6.19 | 74.26 | 102% | ok | 1.8x (9%) |
| blas1 | axpy | f32 | n=8388608 | 1.259 ms | 13.32 | 79.93 | 110% | ok | 1.9x (9%) |
| blas1 | scale | f64 | n=8388608 | 1.746 ms | 4.81 | 76.89 | 106% | ok | 2.0x (10%) |
| blas1 | two_norm | f64 | n=8388608 | 0.557 ms | 30.11 | 120.45 | 166% ! | ok | 3.9x (20%) |
| blas1 | one_norm | f64 | n=8388608 | 3.985 ms | 2.11 | 16.84 | 23% | ok | 1.0x (5%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.939 ms | 1.21 | 9.67 | 13% | ok | 1.0x (5%) |
| blas1 | sum | f64 | n=8388608 | 3.954 ms | 2.12 | 16.97 | 23% | ok | 1.0x (5%) |
| blas1 | dot_k3 | f64 | n=8388608 | 70.123 ms | 2.63 | 1.91 | 3% | ok | 1.0x (5%) |
| blas2 | gemv | f64 | n=4096 | 1.598 ms | 21.00 | 84.03 | 116% | ok | 2.1x (11%) |
| blas2 | gemv | f32 | n=4096 | 0.564 ms | 59.53 | 119.13 | 164% ! | ok | 2.9x (15%) |
| blas2 | symv | f64 | n=4096 | 6.216 ms | 5.40 | 10.81 | 15% | ok | 1.0x (5%) |
| blas2 | trmv | f64 | n=4096 | 1.764 ms | 9.51 | 38.09 | 52% | ok | 1.0x (5%) |
| blas2 | trsv | f64 | n=4096 | 1.810 ms | 9.27 | 37.12 | 51% | ok | 1.0x (5%) |
| blas2 | ger | f64 | n=4096 | 7.142 ms | 4.70 | 37.59 | 52% | ok | 1.0x (5%) |
| blas3 | gemm | f64 | 512x512x512 | 1.569 ms | 171.08 | 4.01 | 26% | ok | 2.4x (12%) |
| blas3 | gemm | f32 | 512x512x512 | 0.502 ms | 534.26 | 6.26 | 42% | ok | 3.7x (19%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.832 ms | 73.27 | 3.43 | 11% | ok | 1.4x (7%) |
| blas3 | symm | f64 | 512x512x512 | 1.478 ms | 181.68 | 4.26 | 27% | ok | 2.7x (14%) |
| blas3 | syrk | f64 | 512x512x512 | 11.426 ms | 11.75 | 0.37 | 2% | ok | 0.9x (4%) |
| blas3 | syr2k | f64 | 512x512x512 | 17.816 ms | 15.07 | 0.35 | 2% | ok | 1.0x (5%) |
| blas3 | trmm | f64 | 512x512x512 | 2.416 ms | 55.56 | 2.17 | 8% | ok | 1.2x (6%) |
| blas3 | trsm | f64 | 512x512x512 | 2.144 ms | 62.59 | 2.44 | 9% | ok | 1.4x (7%) |
| blas3 | trsm | f32 | 512x512x512 | 1.065 ms | 126.01 | 2.46 | 10% | ok | 1.4x (7%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.955 ms | 0.41 | 3.28 | 5% | ok | 1.0x (5%) |
| factor | lu | f64 | 512x512 | 8.639 ms | 10.36 | 0.49 | 2% | ok | 1.2x (6%) |
| factor | lu | f32 | 512x512 | 4.264 ms | 20.98 | 0.49 | 2% | ok | 0.9x (4%) |
| factor | cholesky | f64 | 512x512 | 6.975 ms | 6.41 | 0.60 | 1% | ok | 1.0x (5%) |
| factor | cholesky | f32 | 512x512 | 6.415 ms | 6.97 | 0.33 | 1% | ok | 1.0x (5%) |
| factor | ldlt | f64 | 512x512 | 7.077 ms | 6.32 | 0.59 | 1% | ok | 1.0x (5%) |
| factor | ldlt_bk | f64 | 512x512 | 48.645 ms | 0.92 | 0.09 | 0% | ok | 0.9x (5%) |
| factor | qr | f64 | 512x512 | 153.122 ms | 1.17 | 0.03 | 0% | ok | 1.4x (7%) |
| factor | lq | f64 | 512x512 | 16.694 ms | 10.72 | 0.25 | 2% | ok | 1.4x (7%) |
| factor | hessenberg | f64 | 512x512 | 187.219 ms | 2.39 | 0.02 | 0% | ok | 1.3x (7%) |
| factor | householder | f64 | 512x512 | 0.483 ms | 2.17 | 8.68 | 12%\* | ok | 3.1x (16%) |
| factor | inv | f64 | 512x512 | 62.205 ms | 4.32 | 0.07 | 1% | ok | 1.0x (5%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 6.940 ms | 12.89 | 0.60 | 2% | ok | 0.9x (4%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 16.544 ms | 1.35 | 0.06 | 0% | ok | 1.0x (5%) |
| eigen | eigen_symmetric | f64 | 256x256 | 58.728 ms | 2.57 | 0.02 | 0% | ok | 1.0x (5%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 760.714 ms | 0.20 | 0.00 | 0% | ok | 1.0x (5%) |
| eigen | eigenvalue | f64 | 256x256 | 30.422 ms | 5.51 | 0.03 | 1% | ok | 1.0x (5%) |
| eigen | schur | f64 | 256x256 | 79.354 ms | 5.29 | 0.01 | 1% | ok | 1.0x (5%) |
| eigen | svd | f64 | 256x256 | 93.669 ms | 3.94 | 0.01 | 1% | ok | 1.0x (5%) |
| eigen | expm | f64 | 256x256 | 11.902 ms | 21.14 | 0.09 | 3% | ok | 1.0x (5%) |
| eigen | sylvester | f64 | 256x256 | 171.339 ms | 5.88 | 0.01 | 1% | ok | 1.0x (5%) |
| elementwise | exp | f64 | n=8388608 | 38.796 ms | 0.22 | 3.46 | 5% | ok | 1.0x (5%) |
| elementwise | log | f64 | n=8388608 | 45.838 ms | 0.18 | 2.93 | 4% | ok | 1.0x (5%) |
| elementwise | sin | f64 | n=8388608 | 59.093 ms | 0.14 | 2.27 | 3% | ok | 1.0x (5%) |
| elementwise | sqrt | f64 | n=8388608 | 28.571 ms | 0.29 | 4.70 | 6% | ok | 1.0x (5%) |
| elementwise | erf | f64 | n=8388608 | 201.236 ms | 0.04 | 0.67 | 1% | ok | 1.0x (5%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.107 ms | 24.49 | 254.84 | 351% ! | ok | 6.9x (34%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.911 ms | 2.87 | 29.90 | 41% | ok | 1.0x (5%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.567 ms | 26.73 | 56.19 | 77% | ok | 2.6x (13%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.684 ms | 0.70 | 2.46 | 3%\* | ok | 1.0x (5%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.442 ms | 1.39 | 11.55 | 16%\* | ok | 0.8x (4%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 71.656 ms | 0.16 | 0.01 | 0% | ok | 1.0x (5%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 27.790 ms | 0.40 | 0.03 | 0% | ok | 1.0x (5%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 9.391 ms | 1.19 | 0.08 | 0% | ok | 0.9x (5%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 40.139 ms | 0.56 | 0.02 | 0% | ok | 1.0x (5%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 56.232 ms | 0.40 | 0.01 | 0% | ok | 1.0x (5%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 21.086 ms | 1.06 | 0.03 | 0% | ok | 0.8x (4%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 86.034 ms | 0.02 | 0.00 | 0% | ok | 1.0x (5%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.382 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.881 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.223 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.884 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.245 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 28.448 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 20.812 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.352 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.665 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.520 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 20.186 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.557 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.489 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.515 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 14.731 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 15.768 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 17.903 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.032 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.516 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| pc | ilut | f64 | 128^2 grid | 11.823 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| pc | ic_0 | f64 | 128^2 grid | 1.056 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| pc | ildl | f64 | 128^2 grid | 1.043 ms | 0.00 | 0.00 | n/a | ok | 0.9x (5%) |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.692 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.973 ms | 2.69 | 30.15 | 42% | ok | 1.0x (5%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.300 ms | 2.01 | 22.56 | 31% | ok | 1.0x (5%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.484 ms | 1.76 | 19.76 | 27% | ok | 1.0x (5%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.128 ms | 2.54 | 19.21 | 26% | ok | 0.9x (5%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 27.871 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 38.711 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 39.177 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.567 ms | 0.00 | 0.00 | n/a | ok | 1.0x (5%) |
| mixed | dot | f32>f64 | n=8388608 | 1.889 ms | 8.88 | 35.52 | 49% | ok | 1.0x (5%) |
| mixed | gemm | f32>f64 | 512x512x512 | 1.396 ms | 192.30 | 3.00 | 29% | ok | 4.3x (21%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.303 ms | 55.33 | 58.79 | 81%\* | ok | 1.0x (5%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.915 ms | 8.76 | 35.05 | n/a | ok | 1.0x (5%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.746 ms | 22.48 | 44.95 | n/a | ok | 1.0x (5%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.268 ms | 62.55 | 62.55 | n/a | ok | 1.0x (5%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.253 ms | 66.24 | 66.24 | n/a | ok | 1.0x (5%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 1.280 ms | 209.76 | 2.46 | n/a | ok | 3.1x (15%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 1.411 ms | 190.30 | 1.49 | n/a | ok | 3.9x (20%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 1.386 ms | 193.67 | 1.13 | n/a | ok | 4.0x (20%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.499 ms | 537.68 | 3.15 | n/a | ok | 4.2x (21%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.457 ms | 587.70 | 3.44 | n/a | ok | 4.1x (20%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

