## ryzen-7-8845hs -- native-fast, 8 threads, class medium

| | |
|---|---|
| CPU | AMD Ryzen 7 8845HS w/ Radeon 780M Graphics (x86_64, 16 logical cores) |
| OS | Windows 10.0.26300 (26H2) |
| compiler | MSVC 19.52.37029 |
| build | `/DWIN32 /D_WINDOWS /EHsc /O2 /Ob2 /DNDEBUG`, SIMD SSE2 |
| commit | `b2d022726b3d` |
| FMA peak (ceiling) | f64 625.1 GFLOP/s, f32 1248.8 GFLOP/s |
| ... hardware, AVX-512F | f64 625.1, f32 1248.8 GFLOP/s |
| ... this build, SSE2 | f64 312.3, f32 624.4 GFLOP/s |
| bandwidth | read 53.2, update 53.3, triad 41.2 GB/s (ceiling 53.3) |
| last-level cache | 16 MiB |

| area | entries | median %roof | best %roof | failed checks | median parallel efficiency |
|---|---:|---:|---:|---:|---:|
| blas1 | 10 | 113% | 126% | 0 | 13% |
| blas2 | 10 | 69% | 113% | 0 | 14% |
| blas3 | 10 | 9% | 18% | 0 | 38% |
| factor | 12 | 1% | 8% | 0 | 19% |
| eigen | 8 | 0% | 2% | 0 | 12% |
| elementwise | 5 | 8% | 12% | 0 | 12% |
| sparse | 5 | 41% | 292% | 0 | 15% |
| direct | 19 | 0% | 20% | 0 | 12% |
| ordering | 5 | n/a | n/a | 0 | 12% |
| direct3d | 12 | 2% | 13% | 0 | 12% |
| krylov | 13 | n/a | n/a | 0 | 12% |
| pc | 7 | n/a | n/a | 0 | 12% |
| smoother | 3 | 31% | 35% | 0 | 12% |
| mg | 1 | 36% | 36% | 0 | 13% |
| sparse_eig | 4 | n/a | n/a | 0 | 12% |
| mixed | 3 | 68% | 71% | 0 | 12% |
| integer | 9 | n/a | n/a | 0 | 69% |

| area | op | type | shape | median | GFLOP/s | GB/s | %roof | check | speedup (efficiency) |
|---|---|---|---|---:|---:|---:|---:|---|---:|
| blas1 | dot | f64 | n=8388608 | 2.229 ms | 7.53 | 60.22 | 113% | ok | 1.1x (14%) |
| blas1 | dot | f32 | n=8388608 | 1.008 ms | 16.65 | 66.59 | 125% | ok | 1.2x (15%) |
| blas1 | axpy | f64 | n=8388608 | 3.476 ms | 4.83 | 57.92 | 109% | ok | 1.0x (13%) |
| blas1 | axpy | f32 | n=8388608 | 1.623 ms | 10.34 | 62.02 | 116% | ok | 1.0x (13%) |
| blas1 | scale | f64 | n=8388608 | 1.997 ms | 4.20 | 67.20 | 126% | ok | 1.0x (12%) |
| blas1 | two_norm | f64 | n=8388608 | 1.077 ms | 15.57 | 62.30 | 117% | ok | 1.1x (13%) |
| blas1 | one_norm | f64 | n=8388608 | 5.064 ms | 1.66 | 13.25 | 25% | ok | 1.0x (12%) |
| blas1 | infinity_norm | f64 | n=8388608 | 5.090 ms | 1.65 | 13.18 | 25% | ok | 1.0x (12%) |
| blas1 | sum | f64 | n=8388608 | 5.062 ms | 1.66 | 13.26 | 25% | ok | 1.0x (12%) |
| blas1 | dot_k3 | f64 | n=8388608 | 84.562 ms | 2.18 | 1.59 | 3% | ok | 1.0x (13%) |
| blas2 | gemv | f64 | n=4096 | 2.284 ms | 14.69 | 58.79 | 110% | ok | 1.5x (18%) |
| blas2 | gemv | f32 | n=4096 | 1.118 ms | 30.00 | 60.03 | 113% | ok | 1.4x (18%) |
| blas2 | gemv_colmajor | f64 | n=4096 | 2.296 ms | 14.62 | 58.49 | 110% | ok | 1.7x (22%) |
| blas2 | gemv_colmajor | f32 | n=4096 | 1.189 ms | 28.22 | 56.47 | 106% | ok | 1.8x (23%) |
| blas2 | symv | f64 | n=4096 | 3.176 ms | 10.56 | 21.15 | 40% | ok | 0.9x (11%) |
| blas2 | trmv | f64 | n=4096 | 1.834 ms | 9.15 | 36.63 | 69% | ok | 1.0x (12%) |
| blas2 | trmv | f32 | n=4096 | 0.990 ms | 16.95 | 33.93 | 64% | ok | 1.1x (14%) |
| blas2 | trsv | f64 | n=4096 | 1.829 ms | 9.17 | 36.72 | 69% | ok | 1.0x (12%) |
| blas2 | trsv | f32 | n=4096 | 1.077 ms | 15.57 | 31.17 | 58% | ok | 1.0x (13%) |
| blas2 | ger | f64 | n=4096 | 13.795 ms | 2.43 | 19.46 | 37% | ok | 1.0x (12%) |
| blas3 | gemm | f64 | 512x512x512 | 2.377 ms | 112.92 | 2.65 | 18% | ok | 4.6x (58%) |
| blas3 | gemm | f32 | 512x512x512 | 1.268 ms | 211.68 | 2.48 | 17% | ok | 4.4x (55%) |
| blas3 | gemmt | f64 | 512x512x512 | 2.867 ms | 46.81 | 2.19 | 7% | ok | 2.6x (33%) |
| blas3 | symm | f64 | 512x512x512 | 3.077 ms | 87.24 | 2.04 | 14% | ok | 3.9x (49%) |
| blas3 | syrk | f64 | 512x512x512 | 2.430 ms | 55.23 | 1.73 | 9% | ok | 3.0x (38%) |
| blas3 | syr2k | f64 | 512x512x512 | 4.693 ms | 57.20 | 1.34 | 9% | ok | 3.0x (38%) |
| blas3 | trmm | f64 | 512x512x512 | 5.568 ms | 24.11 | 0.94 | 4% | ok | 1.7x (22%) |
| blas3 | trsm | f64 | 512x512x512 | 5.839 ms | 22.99 | 0.90 | 4% | ok | 1.7x (21%) |
| blas3 | trsm | f32 | 512x512x512 | 4.430 ms | 30.30 | 0.59 | 2% | ok | 1.5x (18%) |
| blas3 | kron | f64 | 48x48 (x) 48x48 | 9.553 ms | 0.56 | 4.45 | 8% | ok | 1.0x (12%) |
| factor | lu | f64 | 512x512 | 21.946 ms | 4.08 | 0.19 | 1% | ok | 1.5x (19%) |
| factor | lu | f32 | 512x512 | 20.818 ms | 4.30 | 0.10 | 0% | ok | 1.5x (19%) |
| factor | cholesky | f64 | 512x512 | 11.380 ms | 3.93 | 0.37 | 1%\* | ok | 1.0x (12%) |
| factor | cholesky | f32 | 512x512 | 10.897 ms | 4.11 | 0.19 | 0%\* | ok | 1.0x (12%) |
| factor | ldlt | f64 | 512x512 | 11.447 ms | 3.91 | 0.37 | 1%\* | ok | 1.0x (12%) |
| factor | ldlt_bk | f64 | 512x512 | 10.243 ms | 4.37 | 0.41 | 1%\* | ok | 1.0x (13%) |
| factor | qr | f64 | 512x512 | 399.855 ms | 0.45 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | lq | f64 | 512x512 | 41.864 ms | 4.27 | 0.10 | 1% | ok | 1.6x (19%) |
| factor | hessenberg | f64 | 512x512 | 461.996 ms | 0.97 | 0.01 | 0% | ok | 1.6x (20%) |
| factor | householder | f64 | 512x512 | 1.023 ms | 1.03 | 4.10 | 8%\* | ok | 3.8x (48%) |
| factor | inv | f64 | 512x512 | 77.904 ms | 3.45 | 0.05 | 1% | ok | 1.1x (14%) |
| factor | lu_iterative_refine | f32>f64 | 512x512 | 23.875 ms | 3.75 | 0.18 | 1% | ok | 1.4x (18%) |
| eigen | eigenvalue_symmetric | f64 | 256x256 | 38.150 ms | 0.59 | 0.03 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric | f64 | 256x256 | 127.871 ms | 1.18 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | eigen_symmetric_jacobi | f64 | 256x256 | 506.029 ms | 0.30 | 0.00 | 0% | ok | 1.0x (13%) |
| eigen | eigenvalue | f64 | 256x256 | 56.803 ms | 2.95 | 0.02 | 0% | ok | 1.0x (12%) |
| eigen | schur | f64 | 256x256 | 115.081 ms | 3.64 | 0.01 | 1% | ok | 1.0x (12%) |
| eigen | svd | f64 | 256x256 | 119.649 ms | 3.08 | 0.01 | 0% | ok | 1.0x (12%) |
| eigen | expm | f64 | 256x256 | 18.071 ms | 13.93 | 0.06 | 2% | ok | 1.4x (18%) |
| eigen | sylvester | f64 | 256x256 | 261.320 ms | 3.85 | 0.01 | 1% | ok | 1.0x (12%) |
| elementwise | exp | f64 | n=8388608 | 28.115 ms | 0.30 | 4.77 | 9% | ok | 1.0x (12%) |
| elementwise | log | f64 | n=8388608 | 42.594 ms | 0.20 | 3.15 | 6% | ok | 1.0x (12%) |
| elementwise | sin | f64 | n=8388608 | 40.003 ms | 0.21 | 3.36 | 6% | ok | 1.0x (12%) |
| elementwise | sqrt | f64 | n=8388608 | 21.301 ms | 0.39 | 6.30 | 12% | ok | 1.0x (13%) |
| elementwise | erf | f64 | n=8388608 | 31.774 ms | 0.26 | 4.22 | 8% | ok | 1.0x (12%) |
| sparse | spmv | f64 | 512^2 grid, nnz 1308672 | 0.175 ms | 14.96 | 155.60 | 292% ! | ok | 5.0x (63%) |
| sparse | spmv_trans | f64 | 512^2 grid, nnz 1308672 | 1.247 ms | 2.10 | 21.83 | 41% | ok | 1.1x (14%) |
| sparse | crs_dense | f64 | 512^2 grid, nnz 1308672 x 16 | 2.244 ms | 18.66 | 39.23 | 74% | ok | 3.0x (37%) |
| sparse | spgemm | f64 | 256^2 grid, nnz 326656 | 18.725 ms | 0.17 | 0.61 | 1%\* | ok | 0.9x (12%) |
| sparse | lower_solve_level | f64 | 192^2 grid, nnz(L) 1003564 | 1.617 ms | 1.24 | 10.29 | 19%\* | ok | 1.2x (15%) |
| direct | sparse_cholesky | f64 | 96^2 grid, nnz 45696 | 18.110 ms | 0.62 | 0.04 | 0% | ok | 1.0x (13%) |
| direct | sparse_ldlt | f64 | 96^2 grid, nnz 45696 | 28.149 ms | 0.40 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt | f64 | 96^2 grid, nnz 45696 | 13.270 ms | 0.84 | 0.06 | 0% | ok | 1.0x (13%) |
| direct | sparse_lu | f64 | 96^2 grid, nnz 45696 | 54.383 ms | 0.41 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu | f64 | 96^2 grid, nnz 45696 | 57.391 ms | 0.39 | 0.01 | 0% | ok | 1.0x (12%) |
| direct | native_klu | f64 | 96^2 grid, nnz 45696 | 25.677 ms | 0.87 | 0.03 | 0% | ok | 1.0x (12%) |
| direct | sparse_qr | f64 | 48^2 grid, nnz 11328 | 38.576 ms | 0.05 | 0.00 | 0%\* | ok | 1.0x (12%) |
| ordering | amd | f64 | 96^2 grid, nnz 45696, scrambled | 3.719 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | colamd | f64 | 96^2 grid, nnz 45696, columns scrambled | 5.666 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | nested_dissection | f64 | 96^2 grid, nnz 45696, scrambled | 60.803 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | rcm | f64 | 96^2 grid, nnz 45696, scrambled | 0.226 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| ordering | block_triangular_form | f64 | 32 planted blocks, n 8192 | 14.173 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.analyze | f64 | 96^2 grid, nnz 45696 | 2.979 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_cholesky.factor | f64 | 96^2 grid, nnz 45696 | 14.459 ms | 0.77 | 0.05 | 0% | ok | 1.0x (13%) |
| direct | sparse_cholesky.solve | f64 | 96^2 grid, nnz 45696 | 0.278 ms | 2.72 | 10.90 | 20%\* | ok | 1.0x (12%) |
| direct | supernodal_ldlt.analyze | f64 | 96^2 grid, nnz 45696 | 6.657 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_ldlt.factor | f64 | 96^2 grid, nnz 45696 | 6.045 ms | 1.85 | 0.12 | 0% | ok | 1.0x (12%) |
| direct | supernodal_ldlt.solve | f64 | 96^2 grid, nnz 45696 | 0.280 ms | 2.70 | 10.80 | 20%\* | ok | 1.0x (12%) |
| direct | sparse_lu.analyze | f64 | 96^2 grid, nnz 45696 | 3.361 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | sparse_lu.factor | f64 | 96^2 grid, nnz 45696 | 48.058 ms | 0.47 | 0.02 | 0% | ok | 1.0x (12%) |
| direct | sparse_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.143 ms | 0.66 | 2.65 | 5%\* | ok | 1.0x (13%) |
| direct | supernodal_lu.analyze | f64 | 96^2 grid, nnz 45696 | 4.832 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct | supernodal_lu.factor | f64 | 96^2 grid, nnz 45696 | 49.158 ms | 0.46 | 0.01 | 0% | ok | 1.0x (13%) |
| direct | supernodal_lu.solve | f64 | 96^2 grid, nnz 45696 | 1.095 ms | 0.69 | 2.76 | 5%\* | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.analyze | f64 | 20^3 grid, nnz 53600 | 5.567 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | sparse_cholesky.factor | f64 | 20^3 grid, nnz 53600 | 259.807 ms | 1.19 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_cholesky.solve | f64 | 20^3 grid, nnz 53600 | 1.998 ms | 1.69 | 6.74 | 13%\* | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.analyze | f64 | 20^3 grid, nnz 53600 | 13.857 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| direct3d | supernodal_ldlt.factor | f64 | 20^3 grid, nnz 53600 | 69.934 ms | 4.41 | 0.01 | 1% | ok | 1.0x (12%) |
| direct3d | supernodal_ldlt.solve | f64 | 20^3 grid, nnz 53600 | 2.061 ms | 1.63 | 6.54 | 12%\* | ok | 1.0x (12%) |
| direct3d | sparse_lu.analyze | f64 | 20^3 grid, nnz 53600 | 6.203 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | sparse_lu.factor | f64 | 20^3 grid, nnz 53600 | 905.916 ms | 0.68 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | sparse_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.648 ms | 0.32 | 1.27 | 2%\* | ok | 1.0x (12%) |
| direct3d | supernodal_lu.analyze | f64 | 20^3 grid, nnz 53600 | 8.556 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| direct3d | supernodal_lu.factor | f64 | 20^3 grid, nnz 53600 | 497.633 ms | 1.24 | 0.00 | 0% | ok | 1.0x (12%) |
| direct3d | supernodal_lu.solve | f64 | 20^3 grid, nnz 53600 | 10.560 ms | 0.32 | 1.28 | 2%\* | ok | 1.0x (12%) |
| krylov | cg+ic0 | f64 | 128^2 grid, n 16384 | 32.647 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | minres+diag | f64 | 128^2 grid, n 16384 | 46.226 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicg+ilu0 | f64 | 128^2 grid, n 16384 | 28.470 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab+ilu0 | f64 | 128^2 grid, n 16384 | 13.521 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | bicgstab_ell+ilu0 | f64 | 128^2 grid, n 16384 | 17.972 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | cgs+ilu0 | f64 | 128^2 grid, n 16384 | 15.737 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | gmres30+ilu0 | f64 | 128^2 grid, n 16384 | 34.393 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | qmr+ilu0 | f64 | 128^2 grid, n 16384 | 29.202 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | tfqmr+ilu0 | f64 | 128^2 grid, n 16384 | 17.306 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s1+ilu0 | f64 | 128^2 grid, n 16384 | 18.604 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| krylov | idr_s2+ilu0 | f64 | 128^2 grid, n 16384 | 17.518 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s4+ilu0 | f64 | 128^2 grid, n 16384 | 17.644 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| krylov | idr_s8+ilu0 | f64 | 128^2 grid, n 16384 | 21.570 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | diagonal | f64 | 128^2 grid, rows scaled | 0.045 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilu_0 | f64 | 128^2 grid | 2.447 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ilut | f64 | 128^2 grid | 22.472 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ic_0 | f64 | 128^2 grid | 1.790 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| pc | ildl | f64 | 128^2 grid | 1.739 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | ssor | f64 | 128^2 grid | 0.030 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| pc | block_diagonal4 | f64 | 128^2 grid, rows scaled | 0.669 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| smoother | jacobi | f64 | 512^2 grid, nnz 1308672 | 1.575 ms | 1.66 | 18.62 | 35% | ok | 0.9x (12%) |
| smoother | gauss_seidel | f64 | 512^2 grid, nnz 1308672 | 1.778 ms | 1.47 | 16.49 | 31% | ok | 1.0x (12%) |
| smoother | sor1.5 | f64 | 512^2 grid, nnz 1308672 | 1.768 ms | 1.48 | 16.59 | 31% | ok | 1.0x (13%) |
| mg | poisson2d_vcycle | f64 | 255^2 grid, 6 levels, 10 V(1,1) | 15.271 ms | 2.52 | 19.03 | 36% | ok | 1.1x (13%) |
| sparse_eig | lanczos | f64 | 64x65 Laplacian, k=4, m=80 | 32.490 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | arnoldi | f64 | 64x65 convection-diffusion, k=4, m=80 | 32.797 ms | 0.00 | 0.00 | n/a | ok | 1.0x (13%) |
| sparse_eig | shift_invert | f64 | 64x65 Laplacian, k=4 nearest 0, m=40 | 52.581 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| sparse_eig | power_iteration | f64 | 64x65 Laplacian, 300 steps | 6.471 ms | 0.00 | 0.00 | n/a | ok | 1.0x (12%) |
| mixed | dot | f32>f64 | n=8388608 | 1.784 ms | 9.40 | 37.61 | 71% | ok | 1.0x (12%) |
| mixed | gemm | f32>f64 | 512x512x512 | 2.626 ms | 102.24 | 1.60 | 16% | ok | 5.8x (73%) |
| mixed | block_scaled_dot | i8+f32 | n=8388608, block 32 | 0.489 ms | 34.28 | 36.42 | 68% | ok | 1.0x (12%) |
| integer | dot | i32 | n=8388608, \|v\| <= 15 | 1.241 ms | 13.52 | 54.08 | n/a | ok | 1.0x (12%) |
| integer | dot | i16>i32 | n=8388608, \|v\| <= 15 | 0.463 ms | 36.27 | 72.53 | n/a | ok | 1.0x (13%) |
| integer | dot | i8>i32 | n=8388608, \|v\| <= 15 | 0.349 ms | 48.11 | 48.11 | n/a | ok | 1.0x (12%) |
| integer | dot | u8i8>i32 | n=8388608, \|v\| <= 15 | 0.317 ms | 52.94 | 52.94 | n/a | ok | 1.0x (12%) |
| integer | gemm | i32 | 512^3, \|v\| <= 2047 | 2.408 ms | 111.47 | 1.31 | n/a | ok | 6.2x (78%) |
| integer | gemm | i16>i32 | 512^3, \|v\| <= 2047 | 2.530 ms | 106.08 | 0.83 | n/a | ok | 6.8x (86%) |
| integer | gemm | i8>i32 | 512^3, \|v\| <= 127 | 2.552 ms | 105.19 | 0.62 | n/a | ok | 6.9x (87%) |
| integer | gemm_quad | i8>i32 | 512^3, \|v\| <= 127 | 0.662 ms | 405.68 | 2.38 | n/a | ok | 5.6x (70%) |
| integer | gemm_quad | u8i8>i32 | 512^3, \|v\| <= 127 | 0.662 ms | 405.19 | 2.37 | n/a | ok | 5.6x (69%) |

\* working set fits in the last-level cache: a cache rate, shown as a % of the DRAM roofline.

! more than 150% of the DRAM roofline for a bandwidth-bound result: the entry's byte model is probably wrong.

