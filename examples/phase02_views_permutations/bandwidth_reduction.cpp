// bandwidth_reduction.cpp -- Example 2B: Cuthill-McKee, renumbering a mesh
//
// A mesh generator numbers its nodes in whatever order it creates them. The
// matrix of the mesh then has its nonzeros scattered far from the diagonal,
// and a banded solver, whose cost grows with the square of the bandwidth,
// pays for every one of them. Elizabeth Cuthill and James McKee (1969) gave
// the fix that is still in use: renumber the nodes breadth-first, level by
// level from a corner, so that neighbours get nearby numbers. Alan George
// (1971) observed that reversing the order (Reverse Cuthill-McKee, RCM) also
// reduces the fill of a later factorization.
//
// Renumbering is a permutation. The example uses permutations three ways:
//   1. as a vector p with p[new] = old, its inverse and its composition
//   2. applied symmetrically, B = P A P^T, by sparse::util::symmetric_permute
//   3. as a view, mapped(A, p, p), which reads B out of A without copying
//
// and measures the bandwidth max |i - j| over the nonzeros before and after.
//
// Run with --trace <file> to write the orderings for the docs-site animation.

#include <mtl/mtl.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "../common/trace.hpp"

using namespace mtl;

namespace {

using sparse_matrix = mat::compressed2D<double>;
using perm_t = std::vector<std::size_t>;

// The 5-point Laplacian of an m x m grid, nodes numbered row by row.
sparse_matrix grid_laplacian(std::size_t m) {
    const std::size_t N = m * m;
    sparse_matrix A(N, N);
    {
        mat::inserter<sparse_matrix> ins(A, 5);
        for (std::size_t r = 0; r < m; ++r)
            for (std::size_t c = 0; c < m; ++c) {
                const std::size_t p = r * m + c;
                ins[p][p] << 4.0;
                if (c > 0)     ins[p][p - 1] << -1.0;
                if (c + 1 < m) ins[p][p + 1] << -1.0;
                if (r > 0)     ins[p][p - m] << -1.0;
                if (r + 1 < m) ins[p][p + m] << -1.0;
            }
    }   // the inserter finishes the matrix here, before it is returned
    return A;
}

std::size_t bandwidth(const sparse_matrix& A) {
    const auto& starts = A.ref_major();
    const auto& cols = A.ref_minor();
    std::size_t bw = 0;
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (auto k = starts[i]; k < starts[i + 1]; ++k)
            bw = std::max(bw, i > cols[k] ? i - cols[k] : cols[k] - i);
    return bw;
}

// The profile (envelope): for each row, the distance from its first nonzero
// to the diagonal. A skyline solver stores exactly this many entries.
std::size_t profile(const sparse_matrix& A) {
    const auto& starts = A.ref_major();
    const auto& cols = A.ref_minor();
    std::size_t total = 0;
    for (std::size_t i = 0; i < A.num_rows(); ++i) {
        std::size_t first = i;
        for (auto k = starts[i]; k < starts[i + 1]; ++k) first = std::min(first, cols[k]);
        total += i - first;
    }
    return total;
}

// A shuffle that gives the same order on every platform: std::shuffle and the
// standard distributions are implementation-defined, std::mt19937 is not.
perm_t scrambled(std::size_t n, std::uint32_t seed) {
    perm_t p(n);
    for (std::size_t i = 0; i < n; ++i) p[i] = i;
    std::mt19937 gen(seed);
    for (std::size_t i = n - 1; i > 0; --i) std::swap(p[i], p[gen() % (i + 1)]);
    return p;
}

std::vector<std::vector<std::size_t>> nonzeros(const sparse_matrix& A) {
    std::vector<std::vector<std::size_t>> out;
    const auto& starts = A.ref_major();
    const auto& cols = A.ref_minor();
    for (std::size_t i = 0; i < A.num_rows(); ++i)
        for (auto k = starts[i]; k < starts[i + 1]; ++k) out.push_back({i, cols[k]});
    return out;
}

} // namespace

int main(int argc, char** argv) {
    example_trace::trace trace(argc, argv, "2B Cuthill-McKee bandwidth reduction");

    std::cout << "=============================================================\n";
    std::cout << " Phase 2B: Cuthill-McKee, Renumbering a Mesh\n";
    std::cout << "=============================================================\n\n";

    const std::size_t m = 12, N = m * m;
    const sparse_matrix natural = grid_laplacian(m);
    std::cout << m << " x " << m << " grid, " << N << " unknowns, " << natural.nnz()
              << " nonzeros.\n\n";

    // -- Permutations as vectors ------------------------------------------
    // p[new] = old: new node i is old node p[i]. Its inverse answers the other
    // question, "where did old node j go?".
    const perm_t shuffle = scrambled(N, 1969);
    const perm_t where = sparse::util::invert_permutation(shuffle);
    const perm_t round_trip = sparse::util::compose_permutations(shuffle, where);
    bool identity = true;
    for (std::size_t i = 0; i < N; ++i) identity = identity && round_trip[i] == i;
    std::cout << "A mesh generator's numbering: a shuffle p of 0.." << N - 1
              << " (valid: " << std::boolalpha << sparse::util::is_valid_permutation(shuffle)
              << ").\n  p[0..5] = ";
    for (std::size_t i = 0; i < 6; ++i) std::cout << shuffle[i] << ' ';
    std::cout << "\n  p composed with its inverse is the identity: " << identity << "\n\n";

    // -- Applying one: B = P A P^T --------------------------------------------
    const sparse_matrix mesh = sparse::util::symmetric_permute(natural, shuffle);

    // The same matrix as a view: mapped(A, p, p)(i, j) = A(p[i], p[j]). It
    // copies nothing, so it suits a look; the explicit B suits a solver.
    auto view = mapped(natural, shuffle, shuffle);
    bool same = true;
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j) same = same && view(i, j) == mesh(i, j);
    std::cout << "symmetric_permute(A, p) and mapped(A, p, p) agree on all " << N * N
              << " entries: " << same << "\n\n";

    // -- Cuthill-McKee ------------------------------------------------------
    // RCM on the scrambled matrix: rcm_order[new] = (scrambled) old.
    const perm_t rcm_order = sparse::ordering::rcm{}(mesh);
    const sparse_matrix reordered = sparse::util::symmetric_permute(mesh, rcm_order);

    std::cout << std::left << std::setw(26) << "numbering" << std::right << std::setw(11)
              << "bandwidth" << std::setw(10) << "profile" << std::setw(16)
              << "band n(2b+1)" << '\n';
    struct row { const char* name; const sparse_matrix* A; };
    for (auto r : {row{"row by row (natural)", &natural}, row{"mesh generator (shuffled)", &mesh},
                   row{"reverse Cuthill-McKee", &reordered}}) {
        const std::size_t b = bandwidth(*r.A);
        std::cout << std::left << std::setw(26) << r.name << std::right << std::setw(11) << b
                  << std::setw(10) << profile(*r.A) << std::setw(16) << N * (2 * b + 1) << '\n';
    }
    std::cout << "\nThe nonzeros are the same " << reordered.nnz()
              << " numbers in every case; only their positions change.\n\n";

    std::cout << "=== Key Takeaways ===\n";
    std::cout << "1. A permutation is a vector p[new] = old; invert_permutation\n";
    std::cout << "   answers where each old index went.\n";
    std::cout << "2. symmetric_permute(A, p) builds P A P^T; mapped(A, p, p) shows\n";
    std::cout << "   the same matrix without building it.\n";
    std::cout << "3. Numbering decides bandwidth: the same mesh is banded or not.\n";
    std::cout << "4. RCM finds a numbering close to the natural one without knowing\n";
    std::cout << "   the grid: it only follows the matrix's graph.\n";

    if (trace) {
        trace["m"] = m;
        trace["n"] = N;
        trace["seed"] = 1969;
        // Positions are given for the natural matrix's nonzeros; each ordering
        // is the map from a natural index to its index in that ordering.
        auto nz = example_trace::value::array();
        for (const auto& e : nonzeros(natural)) nz.push_back(example_trace::value::numbers(e));
        trace["natural_nonzeros"] = nz;
        // Natural node k sits at index where[k] in the shuffled numbering, and
        // shuffled index s sits at rcm_inverse[s] in the RCM numbering.
        const perm_t rcm_inverse = sparse::util::invert_permutation(rcm_order);
        perm_t to_rcm(N);
        for (std::size_t k = 0; k < N; ++k) to_rcm[k] = rcm_inverse[where[k]];
        auto& ords = trace["orderings"];
        ords = example_trace::value::array();
        struct ord { const char* name; perm_t index_of; const sparse_matrix* A; };
        perm_t ident(N);
        for (std::size_t k = 0; k < N; ++k) ident[k] = k;
        for (const auto& o : {ord{"natural", ident, &natural}, ord{"shuffled", where, &mesh},
                              ord{"rcm", to_rcm, &reordered}}) {
            auto v = example_trace::value::object();
            v["name"] = o.name;
            v["index_of"] = example_trace::value::numbers(o.index_of);
            v["bandwidth"] = bandwidth(*o.A);
            v["profile"] = profile(*o.A);
            ords.push_back(v);
        }
        trace["views_agree"] = same;
    }
    return trace.close();   // EXIT_FAILURE if a requested trace could not be written
}
