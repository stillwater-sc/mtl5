// compressed2D_inserter (#825): a second inserter amends the matrix, the
// overflow map, slot_size 0, a 0-row matrix, and the index checks, which run in
// every build.
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <stdexcept>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/mat/inserter.hpp>

using namespace mtl;
using crs = mat::compressed2D<double>;

namespace {

/// The CRS arrays are well formed: starts non-decreasing and ending at nnz,
/// and each row's columns strictly increasing and inside the matrix.
void require_well_formed(const crs& A) {
    const auto& starts = A.ref_major();
    const auto& cols   = A.ref_minor();
    REQUIRE(starts.size() == A.num_rows() + 1);
    REQUIRE(starts.front() == 0);
    REQUIRE(starts.back() == A.nnz());
    REQUIRE(cols.size() == A.nnz());
    for (std::size_t r = 0; r < A.num_rows(); ++r) {
        REQUIRE(starts[r] <= starts[r + 1]);
        for (std::size_t k = starts[r]; k < starts[r + 1]; ++k) {
            REQUIRE(cols[k] < A.num_cols());
            if (k > starts[r]) REQUIRE(cols[k - 1] < cols[k]);
        }
    }
}

} // namespace

TEST_CASE("a second inserter amends the matrix", "[mat][inserter][825]") {
    crs A(3, 3);
    {
        mat::inserter<crs> ins(A);
        ins[0][0] << 1.0;
        ins[1][1] << 2.0;
    }
    SECTION("update_store: new positions are added, an existing one is replaced") {
        {
            mat::inserter<crs> ins(A);
            ins[2][2] << 3.0;
            ins[1][1] << 5.0;
        }
        require_well_formed(A);
        REQUIRE(A.nnz() == 3);
        REQUIRE(A(0, 0) == 1.0);   // was discarded: the second inserter rebuilt from its own entries
        REQUIRE(A(1, 1) == 5.0);
        REQUIRE(A(2, 2) == 3.0);
    }
    SECTION("update_plus adds to the stored value") {
        {
            mat::inserter<crs, mat::update_plus<double>> ins(A);
            ins[1][1] << 5.0;
            ins[1][0] << 4.0;
        }
        require_well_formed(A);
        REQUIRE(A.nnz() == 3);
        REQUIRE(A(0, 0) == 1.0);
        REQUIRE(A(1, 0) == 4.0);
        REQUIRE(A(1, 1) == 7.0);
    }
    SECTION("an inserter that inserts nothing leaves the matrix as it was") {
        { mat::inserter<crs> ins(A); }
        require_well_formed(A);
        REQUIRE(A.nnz() == 2);
        REQUIRE(A(0, 0) == 1.0);
        REQUIRE(A(1, 1) == 2.0);
    }
    SECTION("make_empty() first refills from scratch") {
        A.make_empty();
        {
            mat::inserter<crs> ins(A);
            ins[2][0] << 9.0;
        }
        REQUIRE(A.nnz() == 1);
        REQUIRE(A(0, 0) == 0.0);
        REQUIRE(A(2, 0) == 9.0);
    }
}

TEST_CASE("inserter overflow past slot_size", "[mat][inserter][825]") {
    crs A(2, 8);
    {
        // Two slots per row; row 0 gets six entries, in descending column order.
        mat::inserter<crs, mat::update_plus<double>> ins(A, 2);
        for (std::size_t c = 6; c-- > 0;) ins[0][c] << double(c + 1);
        ins[0][1] << 10.0;   // in the overflow map: updated there
        ins[0][5] << 10.0;   // in a slot
        ins[1][7] << 1.0;
    }
    require_well_formed(A);
    REQUIRE(A.nnz() == 7);
    REQUIRE(A(0, 0) == 1.0);
    REQUIRE(A(0, 1) == 12.0);
    REQUIRE(A(0, 2) == 3.0);
    REQUIRE(A(0, 5) == 16.0);
    REQUIRE(A(1, 7) == 1.0);

    SECTION("an existing row longer than slot_size amends through the overflow map") {
        {
            mat::inserter<crs, mat::update_plus<double>> ins(A, 2);
            ins[0][4] << 100.0;   // existing, loaded into the map
            ins[0][7] << 8.0;     // new, row already full
        }
        require_well_formed(A);
        REQUIRE(A.nnz() == 8);
        REQUIRE(A(0, 4) == 105.0);
        REQUIRE(A(0, 7) == 8.0);
        REQUIRE(A(0, 1) == 12.0);
    }
}

TEST_CASE("inserter with slot_size 0 puts everything in the overflow map", "[mat][inserter][825]") {
    crs A(3, 3);
    {
        mat::inserter<crs> ins(A, 0);
        ins[2][1] << 1.0;
        ins[0][2] << 2.0;
        ins[0][0] << 3.0;
        ins[0][2] << 4.0;   // update_store
    }
    require_well_formed(A);
    REQUIRE(A.nnz() == 3);
    REQUIRE(A(0, 0) == 3.0);
    REQUIRE(A(0, 2) == 4.0);
    REQUIRE(A(2, 1) == 1.0);
    {
        mat::inserter<crs> ins(A, 0);   // and amending works from there too
        ins[1][1] << 5.0;
    }
    REQUIRE(A.nnz() == 4);
    REQUIRE(A(0, 2) == 4.0);
}

TEST_CASE("inserter on a matrix with no rows or no columns", "[mat][inserter][825]") {
    crs A(0, 4);
    { mat::inserter<crs> ins(A); }
    require_well_formed(A);
    REQUIRE(A.nnz() == 0);

    crs B(3, 0);
    {
        mat::inserter<crs> ins(B);
        REQUIRE_THROWS_AS(ins[0][0], std::out_of_range);
    }
    require_well_formed(B);
    REQUIRE(B.nnz() == 0);
}

// The checks are in every build, NDEBUG included, by the "check once" policy
// recorded for #825/#826: at insertion, since the row indexes the slot arrays
// at once and finalize() runs in a destructor.
TEST_CASE("inserter rejects out-of-range indices", "[mat][inserter][825]") {
    crs A(2, 3);
    {
        mat::inserter<crs> ins(A);
        ins[0][0] << 1.0;
        REQUIRE_THROWS_AS(ins[2], std::out_of_range);
        REQUIRE_THROWS_AS(ins[1][3], std::out_of_range);
        ins[1][2] << 2.0;   // still usable after a rejected index
    }
    require_well_formed(A);
    REQUIRE(A.nnz() == 2);
    REQUIRE(A(1, 2) == 2.0);
}
