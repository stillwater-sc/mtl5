#include <catch2/catch_test_macros.hpp>

// Stub test — will be filled when CG solver is ported
#include <mtl/concepts/linear_operator.hpp>
#include <mtl/concepts/preconditioner.hpp>

TEST_CASE("CG placeholder - concepts compile", "[itl][cg]") {
    // The actual CG test will verify convergence on a small SPD system
    // For now, verify the concepts required by CG are well-formed
    REQUIRE(true);
}
