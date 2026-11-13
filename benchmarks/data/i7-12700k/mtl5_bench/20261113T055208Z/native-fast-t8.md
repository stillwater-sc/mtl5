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
| ... this build, AVX2 | f64 600.9, f32 1201.7 GFLOP/s |
| bandwidth | read 81.0, update 77.9, triad 76.2 GB/s (ceiling 81.0) |
| last-level cache | 25 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 90% | 149% | 0 | 22% |
| blas2 | 10 | 49% | 151% | 0 | 13% |
| blas3 | 10 | 12% | 67% | 0 | 19% |
| factor | 12 | 1% | 12% | 0 | 13% |
| eigen | 8 | 1% | 4% | 0 | 13% |
| elementwise | 5 | 4% | 6% | 0 | 13% |
| sparse | 5 | 37% | 302% | 0 | 13% |
| direct | 19 | 0% | 14% | 0 | 13% |
| ordering | 4 | n/a | n/a | 0 | 13% |
| direct3d | 12 | 2% | 11% | 0 | 13% |
| krylov | 13 | n/a | n/a | 0 | 13% |
| pc | 7 | n/a | n/a | 0 | 13% |
| smoother | 3 | 28% | 37% | 0 | 13% |
| mg | 1 | 24% | 24% | 0 | 12% |
| sparse_eig | 4 | n/a | n/a | 0 | 13% |
| mixed | 3 | 49% | 73% | 0 | 13% |
| integer | 9 | n/a | n/a | 0 | 78% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 1.408 ms | 11.91 | 95.32 | 118% | ok | 2.6x (33%) |
| blas1 | dot | f32 | n=8388608 | 0.554 ms | 30.27 | 121.06 | 149% | ok | 3.1x (38%) |
| blas1 | axpy | f64 | n=8388608 | 2.773 ms | 6.05 | 72.60 | 90% | ok | 1.7x (21%) |
| blas1 | axpy | f32 | n=8388608 | 1.358 ms | 12.35 | 74.12 | 91% | ok | 1.7x (22%) |
| blas1 | scale | f64 | n=8388608 | 1.899 ms | 4.42 | 70.67 | 87% | ok | 1.8x (23%) |
| blas1 | two_norm | f64 | n=8388608 | 0.563 ms | 29.79 | 119.18 | 147% | ok | 3.9x (49%) |
| blas1 | one_norm | f64 | n=8388608 | 3.950 ms | 2.12 | 16.99 | 21% | ok | 1.0x (13%) |
| blas1 | infinity_norm | f64 | n=8388608 | 6.908 ms | 1.21 | 9.71 | 12% | ok | 1.0x (13%) |
| blas1 | sum | f64 | n=8388608 | 3.897 ms | 2.15 | 17.22 | 21% | ok | 1.0x (13%) |
| blas1 | dot_k3 | f64 | n=8388608 | 69.444 ms | 2.66 | 1.93 | 2% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 1.444 ms | 23.24 | 93.00 | 115% | ok | 2.4x (29%) |
| blas2 | gemv | f32 | n=4096 | 0.547 ms | 61.30 | 122.66 | 151% ! | ok | 3.1x (38%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 1.436 ms | 23.36 | 93.49 | 115% | ok | 2.4x (30%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 0.678 ms | 49.50 | 99.04 | 122% | ok | 2.5x (31%) |
| blas2 | symv | f64 | n=4096 | 6.117 ms | 5.49 | 10.98 | 14% | ok | 1.0x (13%) |
| blas2 | trmv | f64 | n=4096 | 1.755 ms | 9.56 | 38.27 | 47% | ok | 1.0x (13%) |
| blas2 | trmv | f32 | n=4096 | 0.839 ms | 19.99 | 40.02 | 49% | ok | 1.0x (13%) |
| blas2 | trsv | f64 | n=4096 | 1.790 ms | 9.37 | 37.52 | 46% | ok | 1.0x (13%) |
| blas2 | trsv | f32 | n=4096 | 0.848 ms | 19.79 | 39.62 | 49% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 7.109 ms | 4.72 | 37.77 | 47% | ok | 1.0x (13%) |
| blas3 | gemm | f64 | 512x512x512 | 0.670 ms | 400.67 | 9.39 | 67% | ok | 5.7x (71%) |
| blas3 | gemm | f32 | 512x512x512 | 0.356 ms | 753.02 | 8.82 | 63% | ok | 5.3x (67%) |
| blas3 | gemmt | f64 | 512x512x512 | 1.250 ms | 107.36 | 5.03 | 18% | ok | 2.1x (26%) |
| blas3 | symm | f64 | 512x512x512 | 1.121 ms | 239.55 | 5.61 | 40% | ok | 3.6x (45%) |
| blas3 | syrk | f64 | 512x512x512 | 10.781 ms | 12.45 | 0.39 | 2% | ok | 0.9x (11%) |
| blas3 | syr2k | f64 | 512x512x512 | 16.423 ms | 16.34 | 0.38 | 3% | ok | 1.2x (15%) |
| blas3 | trmm | f64 | 512x512x512 | 1.856 ms | 72.30 | 2.82 | 12% | ok | 1.5x (19%) |
| blas3 | trsm | f64 | 512x512x512 | 2.183 ms | 61.50 | 2.40 | 10% | ok | 1.3x (17%) |
| blas3 | trsm | f32 | 512x512x512 | 1.108 ms | 121.14 | 2.37 | 10% | ok | 1.2x (16%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 12.843 ms | 0.41 | 3.31 | 4% | ok | 1.0x (13%) |
| factor | lu | f64 | 512x512 | 6.458 ms | 13.86 | 0.65 | 2% | ok | 1.7x (21%) |
| factor | lu | f32 | 512x512 | 3.483 ms | 25.69 | 0.60 | 2% | ok | 1.0x (12%) |
| factor | cholesky | f64 | 512x512 | 6.961 ms | 6.43 | 0.60 | 1% | ok | 1.0x (13%) |
| factor | cholesky | f32 | 512x512 | 6.403 ms | 6.99 | 0.33 | 1% | ok | 1.0x (13%) |
| factor | ldlt | f64 | 512x512 | 7.076 ms | 6.32 | 0.59 | 1% | ok | 1.0x (13%) |
| factor | ldlt_bk | f64 | 512x512 | 45.628 ms | 0.98 | 0.09 | 0% | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 129.577 ms | 1.38 | 0.03 | 0% | ok | 1.7x (22%) |
| factor | lq | f64 | 512x512 | 15.067 ms | 11.88 | 0.28 | 2% | ok | 1.5x (19%) |
| factor | hessenberg | f64 | 512x512 | 166.367 ms | 2.69 | 0.03 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 0.429 ms | 2.45 | 9.78 | 12%\* | ok | 3.8x (47%) |
| factor | inv | f64 | 512x512 | 51.556 ms | 5.21 | 0.08 | 1% | ok | 1.1x (13%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 4.864 ms | 18.40 | 0.86 | 3% | ok | 1.0x (12%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 16.931 ms | 1.32 | 0.06 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 56.146 ms | 2.69 | 0.02 | 0% | ok | 1.0x (13%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 744.017 ms | 0.20 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 30.527 ms | 5.50 | 0.03 | 1% | ok | 1.0x (13%) |
| eigen | schur | f64 | 256x256 | 77.856 ms | 5.39 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | svd | f64 | 256x256 | 91.533 ms | 4.03 | 0.01 | 1% | ok | 1.0x (13%) |
| eigen | expm | f64 | 256x256 | 10.101 ms | 24.92 | 0.10 | 4% | ok | 1.2x (15%) |
| eigen | sylvester | f64 | 256x256 | 168.846 ms | 5.96 | 0.01 | 1% | ok | 1.2x (15%) |
| elementwise | exp | f64 | n=8388608 | 38.461 ms | 0.22 | 3.49 | 4% | ok | 1.0x (13%) |
| elementwise | log | f64 | n=8388608 | 45.664 ms | 0.18 | 2.94 | 4% | ok | 1.0x (13%) |
| elementwise | sin | f64 | n=8388608 | 58.403 ms | 0.14 | 2.30 | 3% | ok | 1.0x (13%) |
| elementwise | sqrt | f64 | n=8388608 | 28.264 ms | 0.30 | 4.75 | 6% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 201.206 ms | 0.04 | 0.67 | 1% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.111 ms | 23.50 | 244.52 | 302% ! | ok | 6.6x (82%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 0.900 ms | 2.91 | 30.25 | 37% | ok | 1.0x (13%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 1.366 ms | 30.65 | 64.44 | 80% | ok | 3.0x (38%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 4.770 ms | 0.68 | 2.41 | 3%\* | ok | 1.0x (13%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.379 ms | 1.46 | 12.08 | 15%\* | ok | 0.9x (11%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 71.892 ms | 0.16 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 27.695 ms | 0.40 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 9.136 ms | 1.22 | 0.08 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 39.352 ms | 0.57 | 0.02 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 55.079 ms | 0.41 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 16.654 ms | 1.34 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 85.710 ms | 0.02 | 0.00 | 0% | ok | 1.2x (14%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.378 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 4.902 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.225 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.846 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.300 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 68.729 ms | 0.16 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.272 ms | 2.78 | 11.12 | 14%\* | ok | 0.9x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 4.388 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 4.291 ms | 2.61 | 0.17 | 0% | ok | 1.0x (13%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.300 ms | 2.52 | 10.08 | 12%\* | ok | 0.9x (11%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 2.073 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 35.447 ms | 0.63 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.625 ms | 1.21 | 4.84 | 6%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.808 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 49.248 ms | 0.45 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 0.622 ms | 1.22 | 4.86 | 6%\* | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 9.284 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 5226.596 ms | 0.06 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.533 ms | 2.20 | 8.79 | 11%\* | ok | 0.9x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 18.265 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 64.898 ms | 4.76 | 0.01 | 1% | ok | 1.1x (14%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 1.763 ms | 1.91 | 7.65 | 9%\* | ok | 0.9x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 4.849 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 513.234 ms | 1.20 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 8.057 ms | 0.42 | 1.67 | 2%\* | ok | 0.9x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 11.372 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 640.538 ms | 0.96 | 0.00 | 0% | ok | 1.0x (13%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 8.081 ms | 0.42 | 1.67 | 2%\* | ok | 0.9x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 28.128 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 28.120 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 20.655 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 11.393 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 13.665 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 13.501 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 20.135 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 22.575 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 13.400 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 15.510 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 14.761 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 15.812 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 17.945 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.032 ms | 0.00 | 0.00 | n/a | ok | 1.1x (14%) |
| pc | ilu_0 | f64 | 128^2 grid | 1.529 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ilut | f64 | 128^2 grid | 11.780 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.063 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 0.996 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ssor | f64 | 128^2 grid | 0.021 ms | 0.00 | 0.00 | n/a | ok | 1.1x (13%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.722 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 0.969 ms | 2.70 | 30.27 | 37% | ok | 1.0x (13%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.285 ms | 2.04 | 22.83 | 28% | ok | 1.0x (13%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.483 ms | 1.77 | 19.78 | 24% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 14.857 ms | 2.59 | 19.56 | 24% | ok | 1.0x (12%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 27.731 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 38.516 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 33.976 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 4.493 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| mixed | dot | f32>f64 | n=8388608 | 1.913 ms | 8.77 | 35.08 | 43% | ok | 1.0x (13%) |
| mixed | gemm | f32>f64 | 512x512x512 | 0.912 ms | 294.18 | 4.60 | 49% | ok | 6.6x (82%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.302 ms | 55.56 | 59.04 | 73%\* | ok | 1.1x (13%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.898 ms | 8.84 | 35.35 | n/a | ok | 1.0x (13%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.739 ms | 22.70 | 45.39 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.263 ms | 63.74 | 63.74 | n/a | ok | 1.1x (14%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.246 ms | 68.16 | 68.16 | n/a | ok | 1.1x (14%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 0.773 ms | 347.37 | 4.07 | n/a | ok | 6.7x (83%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 0.818 ms | 328.03 | 2.56 | n/a | ok | 6.9x (86%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 0.801 ms | 335.00 | 1.96 | n/a | ok | 7.0x (88%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.332 ms | 808.89 | 4.74 | n/a | ok | 6.3x (79%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.303 ms | 884.67 | 5.18 | n/a | ok | 6.3x (78%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

