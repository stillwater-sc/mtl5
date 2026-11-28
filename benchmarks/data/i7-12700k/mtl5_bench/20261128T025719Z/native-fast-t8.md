## i7-12700k -- native-fast, 8 threads, class large

| | |
|---|---|
| CPU | 12th Gen Intel(R) Core(TM) i7-12700K (x86_64, 20 logical cores) |
| OS | Linux Ubuntu 26.04.1 LTS (7.0.0-44-generic) |
| compiler | GCC 15.2.0 |
| build | ` -O3 -DNDEBUG [MTL5_NATIVE_ARCH=ON]`, SIMD AVX2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 425.6 GFLOP/s, f32 852.2 GFLOP/s |
| ... hardware, AVX2+FMA | f64 400.4, f32 852.2 GFLOP/s |
| ... this build, AVX2 | f64 425.6, f32 772.7 GFLOP/s |
| bandwidth | read 42.3, update 53.9, triad 47.9 GB/s (ceiling 53.9) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 106% | 153% | 0 | 19% |
| blas2 | 10 | 70% | 156% | 0 | 13% |
| blas3 | 10 | 48% | 112% | 0 | 46% |
| factor | 12 | 3% | 35% | 0 | 60% |
| eigen | 8 | 1% | 7% | 0 | 17% |
| elementwise | 5 | 5% | 8% | 0 | 19% |
| sparse | 5 | 49% | 174% | 0 | 22% |
| direct | 19 | 0% | 11% | 0 | 13% |
| ordering | 5 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 3% | 11% | 0 | 14% |
| krylov | 13 | n/a | n/a | 0 | 19% |
| pc | 7 | n/a | n/a | 0 | 23% |
| smoother | 3 | 44% | 50% | 0 | 13% |
| mg | 1 | 35% | 35% | 0 | 15% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 62% | 76% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 97% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=33554432 | 9.438 ms | 7.11 | 56.89 | 106% | ok | 1.6x (20%) |
| blas1 | dot | f32 | n=33554432 | 3.882 ms | 17.29 | 69.14 | 128% | ok | 2.0x (24%) |
| blas1 | axpy | f64 | n=33554432 | 12.838 ms | 5.23 | 62.73 | 116% | ok | 1.5x (19%) |
| blas1 | axpy | f32 | n=33554432 | 7.653 ms | 8.77 | 52.61 | 98% | ok | 1.3x (16%) |
| blas1 | scale | f64 | n=33554432 | 8.692 ms | 3.86 | 61.77 | 115% | ok | 1.6x (20%) |
| blas1 | two_norm | f64 | n=33554432 | 3.247 ms | 20.66 | 82.66 | 153% ! | ok | 3.0x (37%) |
| blas1 | one_norm | f64 | n=33554432 | 17.081 ms | 1.96 | 15.72 | 29% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=33554432 | 28.861 ms | 1.16 | 9.30 | 17% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=33554432 | 16.844 ms | 1.99 | 15.94 | 30% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=33554432 | 309.184 ms | 2.39 | 1.74 | 3% | ok | 1.0x (12%) |
| blas2 | gemv | f64 | n=8192 | 7.332 ms | 18.31 | 73.24 | 136% | ok | 1.9x (24%) |
| blas2 | gemv | f32 | n=8192 | 3.194 ms | 42.02 | 84.06 | 156% ! | ok | 2.2x (27%) |
| blas2 | gemv_colmajor | f64 | n=8192 | 6.655 ms | 20.17 | 80.69 | 150% | ok | 2.2x (27%) |
| blas2 | gemv_colmajor | f32 | n=8192 | 3.461 ms | 38.78 | 77.58 | 144% | ok | 2.1x (26%) |
| blas2 | symv | f64 | n=8192 | 12.846 ms | 10.45 | 20.91 | 39% | ok | 1.0x (12%) |
| blas2 | trmv | f64 | n=8192 | 7.161 ms | 9.37 | 37.50 | 70% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=8192 | 3.642 ms | 18.43 | 36.87 | 68% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=8192 | 7.153 ms | 9.38 | 37.55 | 70% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=8192 | 3.659 ms | 18.34 | 36.70 | 68% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=8192 | 28.525 ms | 4.71 | 37.65 | 70% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 1024x1024x1024 | 4.520 ms | 475.12 | 5.57 | 112% | ok | 6.8x (85%) |
| blas3 | gemm | f32 | 1024x1024x1024 | 2.279 ms | 942.26 | 5.52 | 111% | ok | 6.6x (82%) |
| blas3 | gemmt | f64 | 1024x1024x1024 | 5.228 ms | 205.39 | 4.81 | 48% | ok | 3.7x (46%) |
| blas3 | symm | f64 | 1024x1024x1024 | 6.738 ms | 318.70 | 3.73 | 75% | ok | 5.0x (62%) |
| blas3 | syrk | f64 | 1024x1024x1024 | 5.306 ms | 202.35 | 3.16 | 48% | ok | 3.3x (42%) |
| blas3 | syr2k | f64 | 1024x1024x1024 | 9.821 ms | 218.65 | 2.56 | 51% | ok | 3.7x (46%) |
| blas3 | trmm | f64 | 1024x1024x1024 | 9.542 ms | 112.52 | 2.20 | 26% | ok | 2.3x (29%) |
| blas3 | trsm | f64 | 1024x1024x1024 | 10.237 ms | 104.89 | 2.05 | 25% | ok | 2.2x (28%) |
| blas3 | trsm | f32 | 1024x1024x1024 | 5.155 ms | 208.29 | 2.03 | 24% | ok | 2.0x (26%) |
| blas3 | kron | f64 | 64x64 (x) 64x64 | 43.532 ms | 0.39 | 3.08 | 6% | ok | 1.0x (13%) |
| factor | lu | f64 | 1024x1024 | 16.331 ms | 43.83 | 1.03 | 10% | ok | 6.2x (78%) |
| factor | lu | f32 | 1024x1024 | 11.617 ms | 61.62 | 0.72 | 7% | ok | 4.8x (60%) |
| factor | cholesky | f64 | 1024x1024 | 31.052 ms | 11.53 | 0.54 | 3% | ok | 2.2x (28%) |
| factor | cholesky | f32 | 1024x1024 | 29.450 ms | 12.15 | 0.28 | 1% | ok | 2.1x (26%) |
| factor | ldlt | f64 | 1024x1024 | 66.163 ms | 5.41 | 0.25 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 1024x1024 | 38.422 ms | 9.32 | 0.44 | 2% | ok | 1.0x (13%) |
| factor | qr | f64 | 1024x1024 | 429.834 ms | 3.33 | 0.04 | 1% | ok | 5.3x (66%) |
| factor | lq | f64 | 1024x1024 | 40.838 ms | 35.06 | 0.41 | 8% | ok | 5.2x (65%) |
| factor | hessenberg | f64 | 1024x1024 | 491.465 ms | 7.28 | 0.03 | 2% | ok | 5.3x (66%) |
| factor | householder | f64 | 1024x1024 | 0.889 ms | 4.72 | 18.86 | 35%\* | ok | 9.0x (112%) |
| factor | inv | f64 | 1024x1024 | 353.480 ms | 6.08 | 0.05 | 1% | ok | 1.3x (16%) |
| factor | lu_iterative_refine | f32>f64 | 1024x1024 | 17.775 ms | 40.27 | 0.94 | 9% | ok | 3.0x (37%) |
| eigen | eigenvalue_symmetric | f64 | 512x512 | 153.016 ms | 1.17 | 0.03 | 0% | ok | 1.7x (21%) |
| eigen | eigen_symmetric | f64 | 512x512 | 777.605 ms | 1.55 | 0.01 | 0% | ok | 1.1x (14%) |
| eigen | eigen_symmetric_jacobi | f64 | 512x512 | 11624.628 ms | 0.10 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 512x512 | 323.318 ms | 4.15 | 0.01 | 1% | ok | 1.4x (18%) |
| eigen | schur | f64 | 512x512 | 1119.792 ms | 3.00 | 0.00 | 1% | ok | 1.3x (16%) |
| eigen | svd | f64 | 512x512 | 1151.943 ms | 2.56 | 0.00 | 1% | ok | 0.9x (11%) |
| eigen | expm | f64 | 512x512 | 71.745 ms | 28.06 | 0.06 | 7% | ok | 2.0x (25%) |
| eigen | sylvester | f64 | 512x512 | 2342.988 ms | 3.44 | 0.00 | 1% | ok | 1.3x (17%) |
| elementwise | exp | f64 | n=33554432 | 161.202 ms | 0.21 | 3.33 | 6% | ok | 1.7x (21%) |
| elementwise | log | f64 | n=33554432 | 185.168 ms | 0.18 | 2.90 | 5% | ok | 1.5x (19%) |
| elementwise | sin | f64 | n=33554432 | 242.319 ms | 0.14 | 2.22 | 4% | ok | 1.6x (20%) |
| elementwise | sqrt | f64 | n=33554432 | 120.192 ms | 0.28 | 4.47 | 8% | ok | 1.3x (16%) |
| elementwise | erf | f64 | n=33554432 | 807.811 ms | 0.04 | 0.66 | 1% | ok | 1.3x (17%) |
| sparse | spmv | f64 | 1024^2 grid, nnz 5238784 | 1.165 ms | 8.99 | 93.54 | 174% ! | ok | 6.5x (81%) |
| sparse | spmv_trans | f64 | 1024^2 grid, nnz 5238784 | 4.146 ms | 2.53 | 26.29 | 49% | ok | 1.8x (22%) |
| sparse | crs_dense | f64 | 1024^2 grid, nnz 5238784 x 16 | 6.838 ms | 24.51 | 51.51 | 96% | ok | 4.2x (53%) |
| sparse | spgemm | f64 | 512^2 grid, nnz 1308672 | 41.987 ms | 0.31 | 1.10 | 2% | ok | 1.1x (13%) |
| sparse | lower_solve_level | f64 | 384^2 grid, nnz(L) 5090475 | 13.221 ms | 0.77 | 6.34 | 12% | ok | 1.7x (21%) |
| direct | sparse_cholesky | f64 | 192^2 grid, nnz 183552 | 102.716 ms | 1.04 | 0.03 | 0% | ok | 1.6x (20%) |
| direct | sparse_ldlt | f64 | 192^2 grid, nnz 183552 | 192.822 ms | 0.55 | 0.02 | 0% | ok | 1.1x (13%) |
| direct | supernodal_ldlt | f64 | 192^2 grid, nnz 183552 | 53.242 ms | 2.01 | 0.06 | 0% | ok | 1.1x (13%) |
| direct | sparse_lu | f64 | 192^2 grid, nnz 183552 | 296.381 ms | 0.72 | 0.01 | 0% | ok | 1.1x (13%) |
| direct | supernodal_lu | f64 | 192^2 grid, nnz 183552 | 326.398 ms | 0.66 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 192^2 grid, nnz 183552 | 130.236 ms | 1.64 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 96^2 grid, nnz 45696 | 374.645 ms | 0.06 | 0.00 | 0% | ok | 1.0x (13%) |
| ordering | amd | f64 | 192^2 grid, nnz 183552, scrambled | 16.319 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| ordering | colamd | f64 | 192^2 grid, nnz 183552, columns scrambled | 23.029 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| ordering | nested_dissection | f64 | 192^2 grid, nnz 183552, scrambled | 260.594 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 192^2 grid, nnz 183552, scrambled | 0.903 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| ordering | block_triangular_form | f64 | 64 planted blocks, n 16384 | 92.815 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.analyze | f64 | 192^2 grid, nnz 183552 | 8.389 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct | sparse_cholesky.factor | f64 | 192^2 grid, nnz 183552 | 94.360 ms | 1.13 | 0.03 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 192^2 grid, nnz 183552 | 2.660 ms | 1.51 | 6.04 | 11%\* | ok | 1.0x (13%) |
| direct | supernodal_ldlt.analyze | f64 | 192^2 grid, nnz 183552 | 17.908 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 192^2 grid, nnz 183552 | 32.183 ms | 3.32 | 0.09 | 1% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 192^2 grid, nnz 183552 | 2.672 ms | 1.50 | 6.01 | 11%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 192^2 grid, nnz 183552 | 8.030 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct | sparse_lu.factor | f64 | 192^2 grid, nnz 183552 | 274.576 ms | 0.78 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.747 ms | 0.46 | 1.84 | 3%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 192^2 grid, nnz 183552 | 13.064 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 192^2 grid, nnz 183552 | 295.596 ms | 0.72 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 192^2 grid, nnz 183552 | 8.778 ms | 0.46 | 1.83 | 3%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 24^3 grid, nnz 93312 | 8.707 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 24^3 grid, nnz 93312 | 585.527 ms | 1.66 | 0.00 | 0% | ok | 1.2x (16%) |
| direct3d | sparse_cholesky.solve | f64 | 24^3 grid, nnz 93312 | 5.003 ms | 1.50 | 6.00 | 11% | ok | 1.4x (17%) |
| direct3d | supernodal_ldlt.analyze | f64 | 24^3 grid, nnz 93312 | 18.938 ms | 0.00 | 0.00 | n/a | ok | 1.2x (15%) |
| direct3d | supernodal_ldlt.factor | f64 | 24^3 grid, nnz 93312 | 205.557 ms | 4.71 | 0.01 | 1% | ok | 1.1x (14%) |
| direct3d | supernodal_ldlt.solve | f64 | 24^3 grid, nnz 93312 | 5.417 ms | 1.38 | 5.54 | 10% | ok | 1.3x (16%) |
| direct3d | sparse_lu.analyze | f64 | 24^3 grid, nnz 93312 | 8.756 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct3d | sparse_lu.factor | f64 | 24^3 grid, nnz 93312 | 1753.100 ms | 1.11 | 0.00 | 0% | ok | 1.4x (17%) |
| direct3d | sparse_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.609 ms | 0.38 | 1.53 | 3% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.analyze | f64 | 24^3 grid, nnz 93312 | 12.812 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| direct3d | supernodal_lu.factor | f64 | 24^3 grid, nnz 93312 | 2197.485 ms | 0.88 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 24^3 grid, nnz 93312 | 19.619 ms | 0.38 | 1.53 | 3% | ok | 1.1x (14%) |
| krylov | cg+ic0 | f64 | 256^2 grid, n 65536 | 213.357 ms | 0.00 | 0.00 | n/a | ok | 1.3x (16%) |
| krylov | minres+diag | f64 | 256^2 grid, n 65536 | 195.777 ms | 0.00 | 0.00 | n/a | ok | 2.1x (26%) |
| krylov | bicg+ilu0 | f64 | 256^2 grid, n 65536 | 134.378 ms | 0.00 | 0.00 | n/a | ok | 1.3x (16%) |
| krylov | bicgstab+ilu0 | f64 | 256^2 grid, n 65536 | 96.235 ms | 0.00 | 0.00 | n/a | ok | 1.5x (18%) |
| krylov | bicgstab_ell+ilu0 | f64 | 256^2 grid, n 65536 | 102.013 ms | 0.00 | 0.00 | n/a | ok | 1.8x (23%) |
| krylov | cgs+ilu0 | f64 | 256^2 grid, n 65536 | 92.864 ms | 0.00 | 0.00 | n/a | ok | 1.7x (22%) |
| krylov | gmres30+ilu0 | f64 | 256^2 grid, n 65536 | 195.844 ms | 0.00 | 0.00 | n/a | ok | 1.8x (23%) |
| krylov | qmr+ilu0 | f64 | 256^2 grid, n 65536 | 167.117 ms | 0.00 | 0.00 | n/a | ok | 1.5x (19%) |
| krylov | tfqmr+ilu0 | f64 | 256^2 grid, n 65536 | 101.999 ms | 0.00 | 0.00 | n/a | ok | 1.5x (19%) |
| krylov | idr_s1+ilu0 | f64 | 256^2 grid, n 65536 | 103.243 ms | 0.00 | 0.00 | n/a | ok | 1.5x (18%) |
| krylov | idr_s2+ilu0 | f64 | 256^2 grid, n 65536 | 92.681 ms | 0.00 | 0.00 | n/a | ok | 1.5x (18%) |
| krylov | idr_s4+ilu0 | f64 | 256^2 grid, n 65536 | 97.152 ms | 0.00 | 0.00 | n/a | ok | 1.8x (23%) |
| krylov | idr_s8+ilu0 | f64 | 256^2 grid, n 65536 | 113.966 ms | 0.00 | 0.00 | n/a | ok | 1.8x (23%) |
| pc | diagonal | f64 | 256^2 grid, rows scaled | 0.138 ms | 0.00 | 0.00 | n/a | ok | 1.9x (24%) |
| pc | ilu_0 | f64 | 256^2 grid | 6.165 ms | 0.00 | 0.00 | n/a | ok | 1.8x (23%) |
| pc | ilut | f64 | 256^2 grid | 50.420 ms | 0.00 | 0.00 | n/a | ok | 1.7x (21%) |
| pc | ic_0 | f64 | 256^2 grid | 4.308 ms | 0.00 | 0.00 | n/a | ok | 1.7x (22%) |
| pc | ildl | f64 | 256^2 grid | 4.016 ms | 0.00 | 0.00 | n/a | ok | 1.9x (24%) |
| pc | ssor | f64 | 256^2 grid | 0.109 ms | 0.00 | 0.00 | n/a | ok | 2.5x (31%) |
| pc | block_diagonal4 | f64 | 256^2 grid, rows scaled | 2.825 ms | 0.00 | 0.00 | n/a | ok | 1.8x (23%) |
| smoother | jacobi | f64 | 1024^2 grid, nnz 5238784 | 4.368 ms | 2.40 | 26.87 | 50% | ok | 1.5x (19%) |
| smoother | gauss_seidel | f64 | 1024^2 grid, nnz 5238784 | 4.951 ms | 2.12 | 23.71 | 44% | ok | 1.1x (13%) |
| smoother | sor1.5 | f64 | 1024^2 grid, nnz 5238784 | 5.755 ms | 1.82 | 20.39 | 38% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 511^2 grid, 7 levels, 10 V(1,1) | 61.502 ms | 2.53 | 19.00 | 35% | ok | 1.2x (15%) |
| sparse_eig | lanczos | f64 | 128x129 Laplacian, k=4, m=80 | 288.350 ms | 0.00 | 0.00 | n/a | ok | 1.2x (14%) |
| sparse_eig | arnoldi | f64 | 128x129 convection-diffusion, k=4, m=80 | 287.552 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| sparse_eig | shift_invert | f64 | 128x129 Laplacian, k=4 nearest 0, m=40 | 488.614 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 128x129 Laplacian, 300 steps | 18.732 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=33554432 | 8.145 ms | 8.24 | 32.96 | 61% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 1024x1024x1024 | 6.652 ms | 322.84 | 2.52 | 76% | ok | 7.3x (91%) |
| mixed | block_scaled_dot | i8+f32 | n=33554432, block 32 | 2.118 ms | 31.68 | 33.66 | 62% | ok | 1.0x (13%) |
| integer | dot | i32 | n=33554432, \|v\| <= 7 | 7.904 ms | 8.49 | 33.96 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=33554432, \|v\| <= 7 | 3.835 ms | 17.50 | 35.00 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=33554432, \|v\| <= 7 | 2.087 ms | 32.16 | 32.16 | n/a | ok | 1.0x (13%) |
| integer | dot | u8i8>i32 | n=33554432, \|v\| <= 7 | 2.075 ms | 32.34 | 32.34 | n/a | ok | 1.0x (13%) |
| integer | gemm | i32 | 1024^3, \|v\| <= 1447 | 4.425 ms | 485.35 | 2.84 | n/a | ok | 8.1x (101%) |
| integer | gemm | i16>i32 | 1024^3, \|v\| <= 1447 | 6.155 ms | 348.90 | 1.36 | n/a | ok | 7.8x (98%) |
| integer | gemm | i8>i32 | 1024^3, \|v\| <= 127 | 6.120 ms | 350.91 | 1.03 | n/a | ok | 7.9x (99%) |
| integer | gemm_quad | i8>i32 | 1024^3, \|v\| <= 127 | 2.319 ms | 926.00 | 2.71 | n/a | ok | 7.9x (98%) |
| integer | gemm_quad | u8i8>i32 | 1024^3, \|v\| <= 127 | 2.060 ms | 1042.52 | 3.05 | n/a | ok | 7.7x (97%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

