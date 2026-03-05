#include <catch2/catch_test_macros.hpp>
#include <mtl/vec/sparse_vector.hpp>
#include <mtl/vec/inserter.hpp>
#include <mtl/mat/inserter.hpp>  // for update_plus
#include <mtl/concepts/vector.hpp>
#include <mtl/concepts/collection.hpp>
#include <cmath>
#include <vector>

using namespace mtl;

// ── Concept satisfaction ────────────────────────────────────────────────

TEST_CASE("sparse_vector satisfies Collection and Vector concepts",
          "[sparse_vector][concepts]") {
    STATIC_REQUIRE(Collection<vec::sparse_vector<double>>);
    STATIC_REQUIRE(Vector<vec::sparse_vector<double>>);
    STATIC_REQUIRE(Collection<vec::sparse_vector<float>>);
    STATIC_REQUIRE(Vector<vec::sparse_vector<int>>);
}

// ── Construction ────────────────────────────────────────────────────────

TEST_CASE("sparse_vector default and sized construction",
          "[sparse_vector][construction]") {
    vec::sparse_vector<double> v0;
    REQUIRE(v0.size() == 0);
    REQUIRE(v0.nnz() == 0);
    REQUIRE(v0.empty());

    vec::sparse_vector<double> v(10);
    REQUIRE(v.size() == 10);
    REQUIRE(v.nnz() == 0);
    REQUIRE_FALSE(v.empty());
}

// ── Insert and sorted order ─────────────────────────────────────────────

TEST_CASE("sparse_vector insert maintains sorted order",
          "[sparse_vector][insert]") {
    vec::sparse_vector<double> v(100);

    // Insert in reverse order
    v.insert(50, 5.0);
    v.insert(10, 1.0);
    v.insert(90, 9.0);
    v.insert(30, 3.0);

    REQUIRE(v.nnz() == 4);

    // Indices must be sorted
    const auto& idx = v.indices();
    REQUIRE(idx[0] == 10);
    REQUIRE(idx[1] == 30);
    REQUIRE(idx[2] == 50);
    REQUIRE(idx[3] == 90);

    // Values must match
    const auto& vals = v.values();
    REQUIRE(vals[0] == 1.0);
    REQUIRE(vals[1] == 3.0);
    REQUIRE(vals[2] == 5.0);
    REQUIRE(vals[3] == 9.0);
}

// ── operator() read access ──────────────────────────────────────────────

TEST_CASE("sparse_vector operator() read access", "[sparse_vector][access]") {
    vec::sparse_vector<double> v(10);
    v.insert(3, 3.14);
    v.insert(7, 2.72);

    // Present indices
    REQUIRE(v(3) == 3.14);
    REQUIRE(v(7) == 2.72);

    // Absent indices return zero
    REQUIRE(v(0) == 0.0);
    REQUIRE(v(5) == 0.0);
    REQUIRE(v(9) == 0.0);
}

// ── operator[] write access ─────────────────────────────────────────────

TEST_CASE("sparse_vector operator[] write access", "[sparse_vector][access]") {
    vec::sparse_vector<double> v(10);

    // Insert via operator[]
    v[3] = 3.14;
    REQUIRE(v.nnz() == 1);
    REQUIRE(v(3) == 3.14);

    // Overwrite existing
    v[3] = 6.28;
    REQUIRE(v.nnz() == 1);
    REQUIRE(v(3) == 6.28);

    // Insert new element
    v[7] = 2.72;
    REQUIRE(v.nnz() == 2);
    REQUIRE(v(7) == 2.72);
}

// ── exists() ────────────────────────────────────────────────────────────

TEST_CASE("sparse_vector exists()", "[sparse_vector][exists]") {
    vec::sparse_vector<double> v(10);
    v.insert(3, 1.0);
    v.insert(7, 2.0);

    REQUIRE(v.exists(3));
    REQUIRE(v.exists(7));
    REQUIRE_FALSE(v.exists(0));
    REQUIRE_FALSE(v.exists(5));
    REQUIRE_FALSE(v.exists(9));
}

// ── clear() ─────────────────────────────────────────────────────────────

TEST_CASE("sparse_vector clear()", "[sparse_vector][clear]") {
    vec::sparse_vector<double> v(10);
    v.insert(3, 1.0);
    v.insert(7, 2.0);
    REQUIRE(v.nnz() == 2);

    v.clear();
    REQUIRE(v.nnz() == 0);
    REQUIRE(v.size() == 10);  // logical size unchanged
    REQUIRE_FALSE(v.exists(3));
}

// ── crop() ──────────────────────────────────────────────────────────────

TEST_CASE("sparse_vector crop()", "[sparse_vector][crop]") {
    vec::sparse_vector<double> v(10);
    v.insert(0, 0.001);
    v.insert(1, 1.0);
    v.insert(2, -0.0001);
    v.insert(3, 5.0);
    v.insert(4, -3.0);
    REQUIRE(v.nnz() == 5);

    v.crop(0.01);
    REQUIRE(v.nnz() == 3);
    REQUIRE_FALSE(v.exists(0));
    REQUIRE(v.exists(1));
    REQUIRE_FALSE(v.exists(2));
    REQUIRE(v.exists(3));
    REQUIRE(v.exists(4));
}

// ── Iterator access ─────────────────────────────────────────────────────

TEST_CASE("sparse_vector iterator", "[sparse_vector][iterator]") {
    vec::sparse_vector<double> v(10);
    v.insert(2, 20.0);
    v.insert(5, 50.0);
    v.insert(8, 80.0);

    std::vector<std::pair<std::size_t, double>> entries;
    for (auto [idx, val] : v) {
        entries.emplace_back(idx, val);
    }

    REQUIRE(entries.size() == 3);
    REQUIRE(entries[0] == std::pair<std::size_t, double>{2, 20.0});
    REQUIRE(entries[1] == std::pair<std::size_t, double>{5, 50.0});
    REQUIRE(entries[2] == std::pair<std::size_t, double>{8, 80.0});
}

// ── Orientation-aware dimensions ────────────────────────────────────────

TEST_CASE("sparse_vector orientation-aware dimensions",
          "[sparse_vector][orientation]") {
    // Default: col_major → column vector
    vec::sparse_vector<double> cv(5);
    REQUIRE(cv.num_rows() == 5);
    REQUIRE(cv.num_cols() == 1);

    // Row vector
    using row_params = vec::parameters<tag::row_major>;
    vec::sparse_vector<double, row_params> rv(5);
    REQUIRE(rv.num_rows() == 1);
    REQUIRE(rv.num_cols() == 5);
}

// ── Traits ──────────────────────────────────────────────────────────────

TEST_CASE("sparse_vector traits", "[sparse_vector][traits]") {
    using sv = vec::sparse_vector<double>;
    STATIC_REQUIRE(std::is_same_v<traits::category<sv>::type, tag::sparse>);
}

// ── Insert overwrite ────────────────────────────────────────────────────

TEST_CASE("sparse_vector insert overwrites existing",
          "[sparse_vector][insert]") {
    vec::sparse_vector<double> v(10);
    v.insert(5, 1.0);
    REQUIRE(v(5) == 1.0);

    v.insert(5, 2.0);
    REQUIRE(v.nnz() == 1);  // no duplicate
    REQUIRE(v(5) == 2.0);
}

// ── Inserter pattern ────────────────────────────────────────────────────

TEST_CASE("sparse_vector inserter with update_store",
          "[sparse_vector][inserter]") {
    vec::sparse_vector<double> v(10);
    {
        vec::sparse_vector_inserter<double> ins(v);
        ins[3] << 3.0;
        ins[7] << 7.0;
        ins[1] << 1.0;
    }
    REQUIRE(v.nnz() == 3);
    REQUIRE(v(1) == 1.0);
    REQUIRE(v(3) == 3.0);
    REQUIRE(v(7) == 7.0);
}

TEST_CASE("sparse_vector inserter with update_plus",
          "[sparse_vector][inserter]") {
    vec::sparse_vector<double> v(10);
    v.insert(3, 1.0);
    {
        vec::vec_inserter<vec::sparse_vector<double>, mat::update_plus<double>> ins(v);
        ins[3] << 2.0;  // accumulate: 1.0 + 2.0 = 3.0
        ins[5] << 5.0;
    }
    REQUIRE(v(3) == 3.0);
    REQUIRE(v(5) == 5.0);
}
