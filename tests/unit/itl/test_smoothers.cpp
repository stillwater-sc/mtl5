#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <mtl/mat/dense2D.hpp>
#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>
#include <mtl/vec/dense_vector.hpp>
#include <mtl/operation/operators.hpp>
#include <mtl/operation/norms.hpp>
#include <mtl/itl/smoother/gauss_seidel.hpp>
#include <mtl/itl/smoother/jacobi.hpp>
#include <mtl/itl/smoother/sor.hpp>
#include <mtl/math/accumulator_traits.hpp>

using namespace mtl;

// Helper: build a 4x4 diagonally dominant SPD dense matrix
static mat::dense2D<double> make_dense_spd() {
    mat::dense2D<double> A(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            A(i, j) = 0.0;
    A(0,0) = 4; A(0,1) = 1;
    A(1,0) = 1; A(1,1) = 4; A(1,2) = 1;
    A(2,1) = 1; A(2,2) = 4; A(2,3) = 1;
    A(3,2) = 1; A(3,3) = 4;
    return A;
}

// Helper: build same matrix as sparse
static mat::compressed2D<double> make_sparse_spd() {
    mat::compressed2D<double> A(4, 4);
    {
        mat::inserter<mat::compressed2D<double>> ins(A);
        ins[0][0] << 4.0; ins[0][1] << 1.0;
        ins[1][0] << 1.0; ins[1][1] << 4.0; ins[1][2] << 1.0;
        ins[2][1] << 1.0; ins[2][2] << 4.0; ins[2][3] << 1.0;
        ins[3][2] << 1.0; ins[3][3] << 4.0;
    }
    return A;
}

// Same SPD matrix in float, for the accumulator-routing test (double
// accumulation over narrower float data -- no Universal needed).
static mat::dense2D<float> make_dense_spd_f() {
    mat::dense2D<float> A(4, 4);
    for (std::size_t i = 0; i < 4; ++i)
        for (std::size_t j = 0; j < 4; ++j)
            A(i, j) = 0.0f;
    A(0,0)=4; A(0,1)=1;
    A(1,0)=1; A(1,1)=4; A(1,2)=1;
    A(2,1)=1; A(2,2)=4; A(2,3)=1;
    A(3,2)=1; A(3,3)=4;
    return A;
}

static mat::compressed2D<float> make_sparse_spd_f() {
    mat::compressed2D<float> A(4, 4);
    {
        mat::inserter<mat::compressed2D<float>> ins(A);
        ins[0][0] << 4.0f; ins[0][1] << 1.0f;
        ins[1][0] << 1.0f; ins[1][1] << 4.0f; ins[1][2] << 1.0f;
        ins[2][1] << 1.0f; ins[2][2] << 4.0f; ins[2][3] << 1.0f;
        ins[3][2] << 1.0f; ins[3][3] << 4.0f;
    }
    return A;
}

namespace {
// A wider-precision accumulator over float data: the running sum is held in
// double and every add_product is counted, so the test can assert the jacobi
// smoother actually routes its off-diagonal row sum through accumulator_traits
// rather than the naive value_type path.
int g_jacobi_addproduct_calls = 0;
struct counting_wide_acc { double v = 0.0; };
} // namespace

namespace mtl::math {
template <>
struct accumulator_traits<counting_wide_acc, float> {
    static void clear(counting_wide_acc& a) { a.v = 0.0; }
    static void assign(counting_wide_acc& a, const float& x) { a.v = static_cast<double>(x); }
    template <typename Result = float>
    static Result value(const counting_wide_acc& a) { return static_cast<Result>(a.v); }
    static void add_product(counting_wide_acc& a, const float& m, const float& x) {
        ++g_jacobi_addproduct_calls;
        a.v += static_cast<double>(m) * static_cast<double>(x);
    }
};
} // namespace mtl::math

TEST_CASE("Gauss-Seidel reduces residual (dense)", "[itl][smoother][gauss_seidel]") {
    auto A = make_dense_spd();
    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0};
    vec::dense_vector<double> x(4, 0.0);

    itl::smoother::gauss_seidel<mat::dense2D<double>> gs(A);

    auto r0 = A * x;
    for (std::size_t i = 0; i < 4; ++i) r0(i) = b(i) - r0(i);
    double norm0 = two_norm(r0);

    // Apply several sweeps
    for (int sweep = 0; sweep < 20; ++sweep)
        gs(x, b);

    auto r1 = A * x;
    for (std::size_t i = 0; i < 4; ++i) r1(i) = b(i) - r1(i);
    double norm1 = two_norm(r1);

    REQUIRE(norm1 < norm0 * 1e-6);
}

TEST_CASE("Gauss-Seidel reduces residual (sparse)", "[itl][smoother][gauss_seidel][sparse]") {
    auto A = make_sparse_spd();
    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0};
    vec::dense_vector<double> x(4, 0.0);

    itl::smoother::gauss_seidel<mat::compressed2D<double>> gs(A);

    for (int sweep = 0; sweep < 20; ++sweep)
        gs(x, b);

    auto r = A * x;
    for (std::size_t i = 0; i < 4; ++i) r(i) = b(i) - r(i);
    REQUIRE(two_norm(r) < 1e-6);
}

TEST_CASE("Jacobi reduces residual", "[itl][smoother][jacobi]") {
    auto A = make_dense_spd();
    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0};
    vec::dense_vector<double> x(4, 0.0);

    itl::smoother::jacobi<mat::dense2D<double>> jac(A);

    for (int sweep = 0; sweep < 50; ++sweep)
        jac(x, b);

    auto r = A * x;
    for (std::size_t i = 0; i < 4; ++i) r(i) = b(i) - r(i);
    REQUIRE(two_norm(r) < 1e-6);
}

TEST_CASE("Jacobi routes the row sum through accumulator_traits (#262)",
          "[itl][smoother][jacobi][accumulator]") {
    vec::dense_vector<float> b = {1.0f, 2.0f, 3.0f, 4.0f};

    SECTION("dense: double-accumulate-over-float matches the void default") {
        auto A = make_dense_spd_f();
        vec::dense_vector<float> x_default(4, 0.0f);
        vec::dense_vector<float> x_wide(4, 0.0f);

        itl::smoother::jacobi<mat::dense2D<float>> jac_default(A);
        itl::smoother::jacobi<mat::dense2D<float>, counting_wide_acc> jac_wide(A);

        g_jacobi_addproduct_calls = 0;
        for (int sweep = 0; sweep < 50; ++sweep) {
            jac_default(x_default, b);
            jac_wide(x_wide, b);
        }

        REQUIRE(g_jacobi_addproduct_calls > 0);   // routing actually exercised
        // Well-conditioned: the wider accumulator converges to the same solution.
        for (std::size_t i = 0; i < 4; ++i)
            REQUIRE_THAT(x_wide(i), Catch::Matchers::WithinAbs(x_default(i), 1e-4));
        auto r = A * x_wide;
        for (std::size_t i = 0; i < 4; ++i) r(i) = b(i) - r(i);
        REQUIRE(two_norm(r) < 1e-4f);
    }

    SECTION("sparse specialization also routes through the accumulator") {
        auto A = make_sparse_spd_f();
        vec::dense_vector<float> x(4, 0.0f);

        itl::smoother::jacobi<mat::compressed2D<float>, counting_wide_acc> jac(A);

        g_jacobi_addproduct_calls = 0;
        for (int sweep = 0; sweep < 50; ++sweep)
            jac(x, b);

        REQUIRE(g_jacobi_addproduct_calls > 0);   // the specialization routes too
        auto r = A * x;
        for (std::size_t i = 0; i < 4; ++i) r(i) = b(i) - r(i);
        REQUIRE(two_norm(r) < 1e-4f);
    }
}

TEST_CASE("Gauss-Seidel routes the row sum through accumulator_traits (#263)",
          "[itl][smoother][gauss_seidel][accumulator]") {
    vec::dense_vector<float> b = {1.0f, 2.0f, 3.0f, 4.0f};

    SECTION("dense: double-accumulate-over-float matches the void default") {
        auto A = make_dense_spd_f();
        vec::dense_vector<float> x_default(4, 0.0f);
        vec::dense_vector<float> x_wide(4, 0.0f);

        itl::smoother::gauss_seidel<mat::dense2D<float>> gs_default(A);
        itl::smoother::gauss_seidel<mat::dense2D<float>, counting_wide_acc> gs_wide(A);

        g_jacobi_addproduct_calls = 0;
        for (int sweep = 0; sweep < 30; ++sweep) {
            gs_default(x_default, b);
            gs_wide(x_wide, b);
        }

        REQUIRE(g_jacobi_addproduct_calls > 0);   // routing actually exercised
        // Well-conditioned: the wider accumulator converges to the same solution.
        for (std::size_t i = 0; i < 4; ++i)
            REQUIRE_THAT(x_wide(i), Catch::Matchers::WithinAbs(x_default(i), 1e-4));
        auto r = A * x_wide;
        for (std::size_t i = 0; i < 4; ++i) r(i) = b(i) - r(i);
        REQUIRE(two_norm(r) < 1e-4f);
    }

    SECTION("sparse specialization also routes through the accumulator") {
        auto A = make_sparse_spd_f();
        vec::dense_vector<float> x(4, 0.0f);

        itl::smoother::gauss_seidel<mat::compressed2D<float>, counting_wide_acc> gs(A);

        g_jacobi_addproduct_calls = 0;
        for (int sweep = 0; sweep < 30; ++sweep)
            gs(x, b);

        REQUIRE(g_jacobi_addproduct_calls > 0);   // the specialization routes too
        auto r = A * x;
        for (std::size_t i = 0; i < 4; ++i) r(i) = b(i) - r(i);
        REQUIRE(two_norm(r) < 1e-4f);
    }
}

TEST_CASE("SOR with omega=1.0 matches Gauss-Seidel", "[itl][smoother][sor]") {
    auto A = make_dense_spd();
    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0};
    vec::dense_vector<double> x_gs(4, 0.0);
    vec::dense_vector<double> x_sor(4, 0.0);

    itl::smoother::gauss_seidel<mat::dense2D<double>> gs(A);
    itl::smoother::sor<mat::dense2D<double>> sor_1(A, 1.0);

    for (int sweep = 0; sweep < 10; ++sweep) {
        gs(x_gs, b);
        sor_1(x_sor, b);
    }

    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE_THAT(x_sor(i), Catch::Matchers::WithinAbs(x_gs(i), 1e-12));
    }
}

TEST_CASE("SOR routes the row sum through accumulator_traits (#264)",
          "[itl][smoother][sor][accumulator]") {
    vec::dense_vector<float> b = {1.0f, 2.0f, 3.0f, 4.0f};
    const float omega = 1.1f;   // over-relaxation: exercises the omega blend

    SECTION("dense: double-accumulate-over-float matches the void default") {
        auto A = make_dense_spd_f();
        vec::dense_vector<float> x_default(4, 0.0f);
        vec::dense_vector<float> x_wide(4, 0.0f);

        itl::smoother::sor<mat::dense2D<float>> sor_default(A, omega);
        itl::smoother::sor<mat::dense2D<float>, counting_wide_acc> sor_wide(A, omega);

        g_jacobi_addproduct_calls = 0;
        for (int sweep = 0; sweep < 40; ++sweep) {
            sor_default(x_default, b);
            sor_wide(x_wide, b);
        }

        REQUIRE(g_jacobi_addproduct_calls > 0);   // routing actually exercised
        // Well-conditioned: the wider accumulator converges to the same solution.
        for (std::size_t i = 0; i < 4; ++i)
            REQUIRE_THAT(x_wide(i), Catch::Matchers::WithinAbs(x_default(i), 1e-4));
        auto r = A * x_wide;
        for (std::size_t i = 0; i < 4; ++i) r(i) = b(i) - r(i);
        REQUIRE(two_norm(r) < 1e-4f);
    }

    SECTION("sparse specialization also routes through the accumulator") {
        auto A = make_sparse_spd_f();
        vec::dense_vector<float> x(4, 0.0f);

        itl::smoother::sor<mat::compressed2D<float>, counting_wide_acc> sor_s(A, omega);

        g_jacobi_addproduct_calls = 0;
        for (int sweep = 0; sweep < 40; ++sweep)
            sor_s(x, b);

        REQUIRE(g_jacobi_addproduct_calls > 0);   // the specialization routes too
        auto r = A * x;
        for (std::size_t i = 0; i < 4; ++i) r(i) = b(i) - r(i);
        REQUIRE(two_norm(r) < 1e-4f);
    }
}

TEST_CASE("Sparse specialization matches generic version", "[itl][smoother][sparse]") {
    auto Ad = make_dense_spd();
    auto As = make_sparse_spd();
    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0};
    vec::dense_vector<double> x_dense(4, 0.0);
    vec::dense_vector<double> x_sparse(4, 0.0);

    itl::smoother::gauss_seidel<mat::dense2D<double>> gs_d(Ad);
    itl::smoother::gauss_seidel<mat::compressed2D<double>> gs_s(As);

    for (int sweep = 0; sweep < 10; ++sweep) {
        gs_d(x_dense, b);
        gs_s(x_sparse, b);
    }

    for (std::size_t i = 0; i < 4; ++i) {
        REQUIRE_THAT(x_sparse(i), Catch::Matchers::WithinAbs(x_dense(i), 1e-12));
    }
}

// -- backward + symmetric Gauss-Seidel (#269) ----------------------------

TEST_CASE("Backward Gauss-Seidel reaches the same fixed point as forward (#269)",
          "[itl][smoother][gauss_seidel][backward]") {
    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0};

    SECTION("dense") {
        auto A = make_dense_spd();
        vec::dense_vector<double> xf(4, 0.0), xb(4, 0.0);
        itl::smoother::gauss_seidel<mat::dense2D<double>> fwd(A);
        itl::smoother::backward_gauss_seidel<mat::dense2D<double>> bwd(A);
        for (int s = 0; s < 100; ++s) { fwd(xf, b); bwd(xb, b); }
        // Both convergent sweep orders reach the same A^{-1} b on an SPD system.
        for (std::size_t i = 0; i < 4; ++i)
            REQUIRE_THAT(xb(i), Catch::Matchers::WithinAbs(xf(i), 1e-9));
        auto r = A * xb;
        for (std::size_t i = 0; i < 4; ++i) r(i) = b(i) - r(i);
        REQUIRE(two_norm(r) < 1e-9);
    }

    SECTION("sparse specialization") {
        auto A = make_sparse_spd();
        vec::dense_vector<double> xb(4, 0.0);
        itl::smoother::backward_gauss_seidel<mat::compressed2D<double>> bwd(A);
        for (int s = 0; s < 100; ++s) bwd(xb, b);
        auto r = A * xb;
        for (std::size_t i = 0; i < 4; ++i) r(i) = b(i) - r(i);
        REQUIRE(two_norm(r) < 1e-9);
    }
}

TEST_CASE("Symmetric Gauss-Seidel = forward then backward, order-symmetric (#269)",
          "[itl][smoother][gauss_seidel][symmetric]") {
    auto A = make_dense_spd();
    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0};

    SECTION("one SGS application equals a forward sweep then a backward sweep") {
        vec::dense_vector<double> x_sgs(4, 0.0), x_manual(4, 0.0);
        itl::smoother::symmetric_gauss_seidel<mat::dense2D<double>> sgs(A);
        itl::smoother::gauss_seidel<mat::dense2D<double>> gs(A);
        sgs(x_sgs, b);
        gs.forward(x_manual, b);
        gs.backward(x_manual, b);
        for (std::size_t i = 0; i < 4; ++i)
            REQUIRE_THAT(x_sgs(i), Catch::Matchers::WithinAbs(x_manual(i), 1e-15));
    }

    SECTION("forward+backward and backward+forward reach the same fixed point") {
        // "Independent of which end it starts": both symmetric orderings converge
        // to the same solution on an SPD system.
        vec::dense_vector<double> x_fb(4, 0.0), x_bf(4, 0.0);
        itl::smoother::symmetric_gauss_seidel<mat::dense2D<double>> sgs(A);   // F then B
        itl::smoother::gauss_seidel<mat::dense2D<double>> gs(A);
        for (int s = 0; s < 60; ++s) {
            sgs(x_fb, b);
            gs.backward(x_bf, b);                                             // B then F
            gs.forward(x_bf, b);
        }
        for (std::size_t i = 0; i < 4; ++i)
            REQUIRE_THAT(x_fb(i), Catch::Matchers::WithinAbs(x_bf(i), 1e-9));
        auto r = A * x_fb;
        for (std::size_t i = 0; i < 4; ++i) r(i) = b(i) - r(i);
        REQUIRE(two_norm(r) < 1e-9);
    }

    SECTION("sparse specialization converges") {
        auto As = make_sparse_spd();
        vec::dense_vector<double> x(4, 0.0);
        itl::smoother::symmetric_gauss_seidel<mat::compressed2D<double>> sgs(As);
        for (int s = 0; s < 60; ++s) sgs(x, b);
        auto r = As * x;
        for (std::size_t i = 0; i < 4; ++i) r(i) = b(i) - r(i);
        REQUIRE(two_norm(r) < 1e-9);
    }
}

TEST_CASE("backward / symmetric Gauss-Seidel route through accumulator_traits (#269)",
          "[itl][smoother][gauss_seidel][symmetric][accumulator]") {
    vec::dense_vector<float> b = {1.0f, 2.0f, 3.0f, 4.0f};

    SECTION("backward (dense) invokes the accumulator") {
        auto A = make_dense_spd_f();
        vec::dense_vector<float> x(4, 0.0f);
        itl::smoother::backward_gauss_seidel<mat::dense2D<float>, counting_wide_acc> bwd(A);
        g_jacobi_addproduct_calls = 0;
        for (int s = 0; s < 30; ++s) bwd(x, b);
        REQUIRE(g_jacobi_addproduct_calls > 0);
        auto r = A * x;
        for (std::size_t i = 0; i < 4; ++i) r(i) = b(i) - r(i);
        REQUIRE(two_norm(r) < 1e-4f);
    }

    SECTION("symmetric (sparse) invokes the accumulator") {
        auto A = make_sparse_spd_f();
        vec::dense_vector<float> x(4, 0.0f);
        itl::smoother::symmetric_gauss_seidel<mat::compressed2D<float>, counting_wide_acc> sgs(A);
        g_jacobi_addproduct_calls = 0;
        for (int s = 0; s < 30; ++s) sgs(x, b);
        REQUIRE(g_jacobi_addproduct_calls > 0);
        auto r = A * x;
        for (std::size_t i = 0; i < 4; ++i) r(i) = b(i) - r(i);
        REQUIRE(two_norm(r) < 1e-4f);
    }
}

TEST_CASE("backward / symmetric Gauss-Seidel edge cases: 1x1 and empty (#269)",
          "[itl][smoother][gauss_seidel][edge]") {
    // Smoothers operate on a square system (they invert the diagonal), so the
    // relevant edge cases are the minimum 1x1 system and the empty system;
    // rectangular input is not applicable.
    SECTION("1x1 dense: one sweep is exact") {
        mat::dense2D<double> A(1, 1); A(0, 0) = 4.0;
        vec::dense_vector<double> b(1, 2.0);              // x = b/A = 0.5
        vec::dense_vector<double> xb(1, 0.0), xs(1, 0.0);
        itl::smoother::backward_gauss_seidel<mat::dense2D<double>> bwd(A);
        itl::smoother::symmetric_gauss_seidel<mat::dense2D<double>> sgs(A);
        bwd(xb, b); REQUIRE_THAT(xb(0), Catch::Matchers::WithinAbs(0.5, 1e-15));
        sgs(xs, b); REQUIRE_THAT(xs(0), Catch::Matchers::WithinAbs(0.5, 1e-15));
    }

    SECTION("1x1 sparse: one sweep is exact") {
        mat::compressed2D<double> A(1, 1);
        { mat::inserter<mat::compressed2D<double>> ins(A); ins[0][0] << 4.0; }
        vec::dense_vector<double> b(1, 2.0);
        vec::dense_vector<double> xb(1, 0.0), xs(1, 0.0);
        itl::smoother::backward_gauss_seidel<mat::compressed2D<double>> bwd(A);
        itl::smoother::symmetric_gauss_seidel<mat::compressed2D<double>> sgs(A);
        bwd(xb, b); REQUIRE_THAT(xb(0), Catch::Matchers::WithinAbs(0.5, 1e-15));
        sgs(xs, b); REQUIRE_THAT(xs(0), Catch::Matchers::WithinAbs(0.5, 1e-15));
    }

    SECTION("empty system is a safe no-op (dense)") {
        mat::dense2D<double> A(0, 0);
        vec::dense_vector<double> b(0), xb(0), xs(0);
        itl::smoother::backward_gauss_seidel<mat::dense2D<double>> bwd(A);
        itl::smoother::symmetric_gauss_seidel<mat::dense2D<double>> sgs(A);
        bwd(xb, b); REQUIRE(xb.size() == 0);
        sgs(xs, b); REQUIRE(xs.size() == 0);
    }

    SECTION("empty system is a safe no-op (sparse)") {
        // Exercise the compressed2D specialization's zero-row / zero-nnz path.
        mat::compressed2D<double> A(0, 0);
        vec::dense_vector<double> b(0), xb(0), xs(0);
        itl::smoother::backward_gauss_seidel<mat::compressed2D<double>> bwd(A);
        itl::smoother::symmetric_gauss_seidel<mat::compressed2D<double>> sgs(A);
        bwd(xb, b); REQUIRE(xb.size() == 0);
        sgs(xs, b); REQUIRE(xs.size() == 0);
    }
}

// -- backward + symmetric SOR / SSOR (#291) ------------------------------

TEST_CASE("Backward SOR reaches the same fixed point as forward (#291)",
          "[itl][smoother][sor][backward]") {
    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0};
    const double omega = 1.1;   // over-relaxation

    SECTION("dense") {
        auto A = make_dense_spd();
        vec::dense_vector<double> xf(4, 0.0), xb(4, 0.0);
        itl::smoother::sor<mat::dense2D<double>> fwd(A, omega);
        itl::smoother::backward_sor<mat::dense2D<double>> bwd(A, omega);
        for (int s = 0; s < 100; ++s) { fwd(xf, b); bwd(xb, b); }
        for (std::size_t i = 0; i < 4; ++i)
            REQUIRE_THAT(xb(i), Catch::Matchers::WithinAbs(xf(i), 1e-9));
        auto r = A * xb;
        for (std::size_t i = 0; i < 4; ++i) r(i) = b(i) - r(i);
        REQUIRE(two_norm(r) < 1e-9);
    }

    SECTION("sparse specialization") {
        auto A = make_sparse_spd();
        vec::dense_vector<double> xb(4, 0.0);
        itl::smoother::backward_sor<mat::compressed2D<double>> bwd(A, omega);
        for (int s = 0; s < 100; ++s) bwd(xb, b);
        auto r = A * xb;
        for (std::size_t i = 0; i < 4; ++i) r(i) = b(i) - r(i);
        REQUIRE(two_norm(r) < 1e-9);
    }
}

TEST_CASE("SSOR = forward then backward relaxed sweep, order-symmetric (#291)",
          "[itl][smoother][sor][symmetric][ssor]") {
    auto A = make_dense_spd();
    vec::dense_vector<double> b = {1.0, 2.0, 3.0, 4.0};
    const double omega = 1.25;

    SECTION("one SSOR application equals a forward then a backward relaxed sweep") {
        vec::dense_vector<double> x_ssor(4, 0.0), x_manual(4, 0.0);
        itl::smoother::symmetric_sor<mat::dense2D<double>> ssor(A, omega);
        itl::smoother::sor<mat::dense2D<double>> s(A, omega);
        ssor(x_ssor, b);
        s.forward(x_manual, b);
        s.backward(x_manual, b);
        for (std::size_t i = 0; i < 4; ++i)
            REQUIRE_THAT(x_ssor(i), Catch::Matchers::WithinAbs(x_manual(i), 1e-15));
    }

    SECTION("forward+backward and backward+forward reach the same fixed point") {
        vec::dense_vector<double> x_fb(4, 0.0), x_bf(4, 0.0);
        itl::smoother::symmetric_sor<mat::dense2D<double>> ssor(A, omega);   // F then B
        itl::smoother::sor<mat::dense2D<double>> s(A, omega);
        for (int step = 0; step < 80; ++step) {
            ssor(x_fb, b);
            s.backward(x_bf, b);                                             // B then F
            s.forward(x_bf, b);
        }
        for (std::size_t i = 0; i < 4; ++i)
            REQUIRE_THAT(x_fb(i), Catch::Matchers::WithinAbs(x_bf(i), 1e-9));
        auto r = A * x_fb;
        for (std::size_t i = 0; i < 4; ++i) r(i) = b(i) - r(i);
        REQUIRE(two_norm(r) < 1e-9);
    }

    SECTION("sparse specialization converges") {
        auto As = make_sparse_spd();
        vec::dense_vector<double> x(4, 0.0);
        itl::smoother::symmetric_sor<mat::compressed2D<double>> ssor(As, omega);
        for (int s = 0; s < 80; ++s) ssor(x, b);
        auto r = As * x;
        for (std::size_t i = 0; i < 4; ++i) r(i) = b(i) - r(i);
        REQUIRE(two_norm(r) < 1e-9);
    }
}

TEST_CASE("backward / symmetric SOR route through accumulator_traits (#291)",
          "[itl][smoother][sor][symmetric][accumulator]") {
    vec::dense_vector<float> b = {1.0f, 2.0f, 3.0f, 4.0f};
    const float omega = 1.1f;

    SECTION("backward (dense) invokes the accumulator") {
        auto A = make_dense_spd_f();
        vec::dense_vector<float> x(4, 0.0f);
        itl::smoother::backward_sor<mat::dense2D<float>, counting_wide_acc> bwd(A, omega);
        g_jacobi_addproduct_calls = 0;
        for (int s = 0; s < 40; ++s) bwd(x, b);
        REQUIRE(g_jacobi_addproduct_calls > 0);
        auto r = A * x;
        for (std::size_t i = 0; i < 4; ++i) r(i) = b(i) - r(i);
        REQUIRE(two_norm(r) < 1e-4f);
    }

    SECTION("SSOR (sparse) invokes the accumulator") {
        auto A = make_sparse_spd_f();
        vec::dense_vector<float> x(4, 0.0f);
        itl::smoother::symmetric_sor<mat::compressed2D<float>, counting_wide_acc> ssor(A, omega);
        g_jacobi_addproduct_calls = 0;
        for (int s = 0; s < 40; ++s) ssor(x, b);
        REQUIRE(g_jacobi_addproduct_calls > 0);
        auto r = A * x;
        for (std::size_t i = 0; i < 4; ++i) r(i) = b(i) - r(i);
        REQUIRE(two_norm(r) < 1e-4f);
    }
}

TEST_CASE("backward / symmetric SOR edge cases: 1x1 and empty (#291)",
          "[itl][smoother][sor][edge]") {
    const double omega = 1.1;
    SECTION("1x1 dense: one relaxed sweep from zero is omega * b/a") {
        mat::dense2D<double> A(1, 1); A(0, 0) = 4.0;
        vec::dense_vector<double> b(1, 2.0);
        // x0=0: gs_update = b/a = 0.5; x = omega*0.5 + (1-omega)*0 = 0.55
        vec::dense_vector<double> xb(1, 0.0), xs(1, 0.0);
        itl::smoother::backward_sor<mat::dense2D<double>> bwd(A, omega);
        bwd(xb, b); REQUIRE_THAT(xb(0), Catch::Matchers::WithinAbs(0.55, 1e-15));
        // SSOR = forward then backward: after forward x=0.55, backward gs_update
        // is still b/a=0.5 (1x1 has no off-diagonal), x = omega*0.5 + (1-omega)*0.55
        itl::smoother::symmetric_sor<mat::dense2D<double>> ssor(A, omega);
        ssor(xs, b);
        const double after_fwd = omega * 0.5;                        // 0.55
        const double expected  = omega * 0.5 + (1.0 - omega) * after_fwd;
        REQUIRE_THAT(xs(0), Catch::Matchers::WithinAbs(expected, 1e-15));
    }

    SECTION("1x1 sparse converges to the exact solution") {
        mat::compressed2D<double> A(1, 1);
        { mat::inserter<mat::compressed2D<double>> ins(A); ins[0][0] << 4.0; }
        vec::dense_vector<double> b(1, 2.0), xb(1, 0.0), xs(1, 0.0);
        itl::smoother::backward_sor<mat::compressed2D<double>> bwd(A, omega);
        itl::smoother::symmetric_sor<mat::compressed2D<double>> ssor(A, omega);
        for (int s = 0; s < 50; ++s) { bwd(xb, b); ssor(xs, b); }
        REQUIRE_THAT(xb(0), Catch::Matchers::WithinAbs(0.5, 1e-12));   // 2/4
        REQUIRE_THAT(xs(0), Catch::Matchers::WithinAbs(0.5, 1e-12));
    }

    SECTION("empty system is a safe no-op (dense + sparse)") {
        vec::dense_vector<double> b(0), xb(0), xs(0);
        mat::dense2D<double> Ad(0, 0);
        itl::smoother::backward_sor<mat::dense2D<double>> bwd_d(Ad, omega);
        itl::smoother::symmetric_sor<mat::dense2D<double>> ssor_d(Ad, omega);
        bwd_d(xb, b); REQUIRE(xb.size() == 0);
        ssor_d(xs, b); REQUIRE(xs.size() == 0);

        mat::compressed2D<double> As(0, 0);
        itl::smoother::backward_sor<mat::compressed2D<double>> bwd_s(As, omega);
        itl::smoother::symmetric_sor<mat::compressed2D<double>> ssor_s(As, omega);
        bwd_s(xb, b); REQUIRE(xb.size() == 0);
        ssor_s(xs, b); REQUIRE(xs.size() == 0);
    }
}
