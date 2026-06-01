// Tests for the register-blocked GEMM micro-kernel (#88).
// Integer-valued data => products and sums are exact in float/double regardless
// of FMA/accumulation order, so we compare bit-exactly to a triple-loop ref.
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_template_test_macros.hpp>

#include <mtl/detail/gemm_microkernel.hpp>
#include <mtl/simd/blocking.hpp>

#include <cstddef>
#include <vector>

namespace {
const std::size_t kKc[] = {1, 2, 4, 7, 16, 64, 200};
}

TEMPLATE_TEST_CASE("gemm_microkernel: C += A*B for a packed MRxNR tile", "[detail][gemm][simd]", float, double) {
    constexpr std::size_t MR = mtl::simd::default_blocking<TestType>.mr;
    constexpr std::size_t NR = mtl::simd::default_blocking<TestType>.nr;
    INFO("MR=" << MR << " NR=" << NR << " W=" << mtl::simd::width<TestType>);

    for (std::size_t kc : kKc) {
        // A is MR x kc (row-major), B is kc x NR (row-major == packed B panel).
        std::vector<TestType> A(MR * kc), Bm(kc * NR), Ap(MR * kc);
        for (std::size_t i = 0; i < MR; ++i)
            for (std::size_t p = 0; p < kc; ++p)
                A[i * kc + p] = TestType((i + 2 * p) % 7) - 3;
        for (std::size_t p = 0; p < kc; ++p)
            for (std::size_t j = 0; j < NR; ++j)
                Bm[p * NR + j] = TestType((3 * p + j) % 5) - 2;

        // Pack A column-major: Ap[p*MR + i] == A(i,p).
        for (std::size_t p = 0; p < kc; ++p)
            for (std::size_t i = 0; i < MR; ++i)
                Ap[p * MR + i] = A[i * kc + p];

        // Reference C = A * B.
        std::vector<TestType> ref(MR * NR, TestType(0));
        for (std::size_t i = 0; i < MR; ++i)
            for (std::size_t j = 0; j < NR; ++j) {
                TestType s = 0;
                for (std::size_t p = 0; p < kc; ++p) s += A[i * kc + p] * Bm[p * NR + j];
                ref[i * NR + j] = s;
            }

        SECTION("into zeroed C (ldc == NR)") {
            std::vector<TestType> C(MR * NR, TestType(0));
            mtl::detail::gemm_microkernel<TestType, MR, NR>(kc, Ap.data(), Bm.data(), C.data(), NR);
            for (std::size_t k = 0; k < MR * NR; ++k) CHECK(C[k] == ref[k]);
        }
        SECTION("accumulate into preset C") {
            std::vector<TestType> C(MR * NR), pre(MR * NR);
            for (std::size_t k = 0; k < MR * NR; ++k) { C[k] = TestType(k % 5) - 2; pre[k] = C[k]; }
            mtl::detail::gemm_microkernel<TestType, MR, NR>(kc, Ap.data(), Bm.data(), C.data(), NR);
            for (std::size_t k = 0; k < MR * NR; ++k) CHECK(C[k] == pre[k] + ref[k]);
        }
        SECTION("non-trivial leading dimension (tile embedded in a wider C)") {
            const std::size_t ldc = NR + 5;
            std::vector<TestType> C(MR * ldc, TestType(0));
            mtl::detail::gemm_microkernel<TestType, MR, NR>(kc, Ap.data(), Bm.data(), C.data(), ldc);
            for (std::size_t i = 0; i < MR; ++i) {
                for (std::size_t j = 0; j < NR; ++j) CHECK(C[i * ldc + j] == ref[i * NR + j]);
                for (std::size_t j = NR; j < ldc; ++j) CHECK(C[i * ldc + j] == TestType(0)); // padding untouched
            }
        }
    }
}
