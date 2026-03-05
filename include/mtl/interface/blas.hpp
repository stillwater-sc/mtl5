#pragma once
// MTL5 — Optional BLAS interface for hardware-accelerated operations
// Port from MTL4: boost/numeric/mtl/interface/blas.hpp
// Guarded by MTL5_HAS_BLAS (set by CMake when MTL5_ENABLE_BLAS=ON)

#ifdef MTL5_HAS_BLAS

#include <cstddef>

// ── Fortran BLAS declarations ──────────────────────────────────────────
extern "C" {

// Level 1 — vector operations
float  sdot_(const int* n, const float* x, const int* incx,
             const float* y, const int* incy);
double ddot_(const int* n, const double* x, const int* incx,
             const double* y, const int* incy);

void saxpy_(const int* n, const float* alpha, const float* x, const int* incx,
            float* y, const int* incy);
void daxpy_(const int* n, const double* alpha, const double* x, const int* incx,
            double* y, const int* incy);

void scopy_(const int* n, const float* x, const int* incx,
            float* y, const int* incy);
void dcopy_(const int* n, const double* x, const int* incx,
            double* y, const int* incy);

void sscal_(const int* n, const float* alpha, float* x, const int* incx);
void dscal_(const int* n, const double* alpha, double* x, const int* incx);

float  snrm2_(const int* n, const float* x, const int* incx);
double dnrm2_(const int* n, const double* x, const int* incx);

float  sasum_(const int* n, const float* x, const int* incx);
double dasum_(const int* n, const double* x, const int* incx);

int isamax_(const int* n, const float* x, const int* incx);
int idamax_(const int* n, const double* x, const int* incx);

// Level 2 — matrix-vector operations
void sgemv_(const char* trans, const int* m, const int* n,
            const float* alpha, const float* A, const int* lda,
            const float* x, const int* incx,
            const float* beta, float* y, const int* incy);
void dgemv_(const char* trans, const int* m, const int* n,
            const double* alpha, const double* A, const int* lda,
            const double* x, const int* incx,
            const double* beta, double* y, const int* incy);

void strsv_(const char* uplo, const char* trans, const char* diag,
            const int* n, const float* A, const int* lda,
            float* x, const int* incx);
void dtrsv_(const char* uplo, const char* trans, const char* diag,
            const int* n, const double* A, const int* lda,
            double* x, const int* incx);

// Level 3 — matrix-matrix operations
void sgemm_(const char* transa, const char* transb,
            const int* m, const int* n, const int* k,
            const float* alpha, const float* A, const int* lda,
            const float* B, const int* ldb,
            const float* beta, float* C, const int* ldc);
void dgemm_(const char* transa, const char* transb,
            const int* m, const int* n, const int* k,
            const double* alpha, const double* A, const int* lda,
            const double* B, const int* ldb,
            const double* beta, double* C, const int* ldc);

} // extern "C"

// ── C++ wrapper functions ──────────────────────────────────────────────

namespace mtl::interface::blas {

// ── Level 1 ────────────────────────────────────────────────────────────

inline float dot(int n, const float* x, int incx, const float* y, int incy) {
    return sdot_(&n, x, &incx, y, &incy);
}
inline double dot(int n, const double* x, int incx, const double* y, int incy) {
    return ddot_(&n, x, &incx, y, &incy);
}

inline void axpy(int n, float alpha, const float* x, int incx, float* y, int incy) {
    saxpy_(&n, &alpha, x, &incx, y, &incy);
}
inline void axpy(int n, double alpha, const double* x, int incx, double* y, int incy) {
    daxpy_(&n, &alpha, x, &incx, y, &incy);
}

inline void copy(int n, const float* x, int incx, float* y, int incy) {
    scopy_(&n, x, &incx, y, &incy);
}
inline void copy(int n, const double* x, int incx, double* y, int incy) {
    dcopy_(&n, x, &incx, y, &incy);
}

inline void scal(int n, float alpha, float* x, int incx) {
    sscal_(&n, &alpha, x, &incx);
}
inline void scal(int n, double alpha, double* x, int incx) {
    dscal_(&n, &alpha, x, &incx);
}

inline float nrm2(int n, const float* x, int incx) {
    return snrm2_(&n, x, &incx);
}
inline double nrm2(int n, const double* x, int incx) {
    return dnrm2_(&n, x, &incx);
}

// ── Level 2 ────────────────────────────────────────────────────────────

inline void gemv(char trans, int m, int n, float alpha,
                 const float* A, int lda, const float* x, int incx,
                 float beta, float* y, int incy) {
    sgemv_(&trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}
inline void gemv(char trans, int m, int n, double alpha,
                 const double* A, int lda, const double* x, int incx,
                 double beta, double* y, int incy) {
    dgemv_(&trans, &m, &n, &alpha, A, &lda, x, &incx, &beta, y, &incy);
}

inline void trsv(char uplo, char trans, char diag, int n,
                 const float* A, int lda, float* x, int incx) {
    strsv_(&uplo, &trans, &diag, &n, A, &lda, x, &incx);
}
inline void trsv(char uplo, char trans, char diag, int n,
                 const double* A, int lda, double* x, int incx) {
    dtrsv_(&uplo, &trans, &diag, &n, A, &lda, x, &incx);
}

// ── Level 3 ────────────────────────────────────────────────────────────

inline void gemm(char transa, char transb, int m, int n, int k,
                 float alpha, const float* A, int lda,
                 const float* B, int ldb,
                 float beta, float* C, int ldc) {
    sgemm_(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}
inline void gemm(char transa, char transb, int m, int n, int k,
                 double alpha, const double* A, int lda,
                 const double* B, int ldb,
                 double beta, double* C, int ldc) {
    dgemm_(&transa, &transb, &m, &n, &k, &alpha, A, &lda, B, &ldb, &beta, C, &ldc);
}

} // namespace mtl::interface::blas

#endif // MTL5_HAS_BLAS
