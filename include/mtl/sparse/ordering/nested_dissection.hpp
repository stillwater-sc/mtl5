#pragma once
// MTL5 -- Nested dissection ordering (multilevel or level-structure separators)
//
// Nested dissection orders a graph by divide and conquer. It finds a small
// vertex separator S whose removal splits the graph into two parts A and B with
// no edges between them, numbers A and B first and S last, and recurses on A and
// B. Eliminating a vertex of A can only create fill among its neighbours, which
// lie in A or S, so no fill ever joins A to B: the factor's off-diagonal fill is
// confined to the separator blocks. On a k x k grid the separators have O(k)
// vertices, which gives O(n log n) fill and O(n^1.5) work for n = k^2 -- optimal
// for planar graphs. The separator tree also exposes independent subtrees, the
// structure a parallel factorization wants.
//
// This is George's automatic nested dissection (SPARSPAK GENND), with separators
// taken from rooted level structures:
//
//   1. On the pattern of A + A^T, split the current vertex set into connected
//      components and treat each separately.
//   2. Find a pseudo-peripheral vertex with the George-Liu iteration: a
//      breadth-first search from the minimum-degree vertex of the deepest level,
//      repeated while the eccentricity grows.
//   3. Take a level as the separator S: the narrowest level whose larger side
//      holds at most `balance` (0.7) of the remaining vertices. Levels before it
//      form A, levels after it form B, and every edge joins the same or adjacent
//      levels, so S separates A from B.
//   4. Thin S: a vertex of S with no neighbour in the next level touches only A
//      and S, so it moves to A without connecting A to B.
//   5. Number A, then B, then S. Recurse on A and B.
//   6. Below `cutoff` vertices, order the subgraph with AMD instead. That is the
//      hybrid production codes use (METIS recurses to about 200 vertices and then
//      applies minimum degree): minimum degree is better on small pieces, and
//      nested dissection wins on the large-scale structure.
//
// The recursion runs on an explicit work stack, so deep separator trees cannot
// overflow the call stack.
//
// That is nd_separator::level_structure. The default, nd_separator::multilevel,
// replaces step 3 with a multilevel separator (multilevel_separator.hpp):
// coarsening by heavy-edge matching, an initial separator on the coarsest
// graph, and vertex-separator Fiduccia-Mattheyses refinement on the way back,
// as in METIS. The level-structure separator, refined the same way, is a second
// candidate, and the smaller balanced separator wins. Level structures are
// cheap and asymptotically optimal on grids; on irregular graphs a level can be
// far wider than the best separator, and the multilevel one is much better
// (random geometric graphs: half the fill, a quarter of the flops).
//
// References: George, "Nested Dissection of a Regular Finite Element Mesh",
//             SIAM J. Numer. Anal. 10(2), 1973.
//             George & Liu, "An Implementation of a Pseudoperipheral Node
//             Finder", ACM TOMS 5(3), 1979.
//             George & Liu, "Computer Solution of Large Sparse Positive
//             Definite Systems", Prentice-Hall, 1981, Ch. 8.
//             Lipton, Rose & Tarjan, "Generalized Nested Dissection",
//             SIAM J. Numer. Anal. 16(2), 1979.

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <mtl/mat/compressed2D.hpp>
#include <mtl/sparse/ordering/minimum_degree.hpp>
#include <mtl/sparse/ordering/multilevel_separator.hpp>

namespace mtl::sparse::ordering {

/// How nested_dissection finds its separators.
///   level_structure: one level of a breadth-first search from a
///                    pseudo-peripheral vertex (George's automatic ND).
///   multilevel:      coarsening by heavy-edge matching, an initial separator
///                    on the coarsest graph, and vertex-separator
///                    Fiduccia-Mattheyses refinement while uncoarsening
///                    (METIS-style); see multilevel_separator.hpp.
enum class nd_separator { level_structure, multilevel };

namespace detail {

/// Nested dissection on a symmetric pattern without diagonal, given as integer
/// CSC arrays (the A + A^T pattern build_aplusat_pattern produces). Returns
/// perm with perm[new] = old.
inline std::vector<std::size_t> nested_dissection_core(
    std::size_t n, const std::vector<idx>& Cp, const std::vector<idx>& Ci,
    std::size_t cutoff, double balance,
    nd_separator method = nd_separator::level_structure)
{
    std::vector<std::size_t> perm(n);
    if (n == 0) return perm;
    if (cutoff < 1) cutoff = 1;

    // owner[v] is the id of the work item v currently belongs to; a search
    // restricted to an item visits only vertices with that owner. Ids are never
    // reused, so a stale id can never match.
    std::vector<idx> owner(n, 0);
    idx next_id = 1;

    // Scratch reused by every search: the level of each visited vertex, and a
    // stamp marking vertices visited by the current search.
    std::vector<idx> level(n, -1);
    std::vector<idx> seen(n, -1);
    idx search = 0;

    // Breadth-first search from `root` within item `id`. Fills `order` with the
    // vertices in visiting order, `level[v]` with their depth, and `starts` with
    // the offset of each level in `order`. Returns the number of levels.
    std::vector<idx> order, starts;
    auto bfs = [&](idx root, idx id) {
        ++search;
        order.clear();
        starts.clear();
        order.push_back(root);
        seen[root] = search;
        level[root] = 0;
        starts.push_back(0);
        std::size_t head = 0;
        idx depth = 0;
        while (head < order.size()) {
            idx v = order[head++];
            if (level[v] != depth) {                 // first vertex of a new level
                depth = level[v];
                starts.push_back(static_cast<idx>(head - 1));
            }
            for (idx p = Cp[v]; p < Cp[v + 1]; ++p) {
                idx w = Ci[p];
                if (owner[w] == id && seen[w] != search) {
                    seen[w] = search;
                    level[w] = depth + 1;
                    order.push_back(w);
                }
            }
        }
        starts.push_back(static_cast<idx>(order.size()));
        return static_cast<idx>(starts.size() - 1);
    };

    auto degree_in = [&](idx v, idx id) {
        idx d = 0;
        for (idx p = Cp[v]; p < Cp[v + 1]; ++p)
            if (owner[Ci[p]] == id) ++d;
        return d;
    };

    // Order an item's vertices with AMD on its induced subgraph and write them
    // to perm[lo ...].
    std::vector<idx> local(n, -1);
    auto order_small = [&](const std::vector<idx>& verts, std::size_t lo, idx id) {
        const std::size_t m = verts.size();
        if (m <= 2) {
            for (std::size_t i = 0; i < m; ++i) perm[lo + i] = static_cast<std::size_t>(verts[i]);
            return;
        }
        for (std::size_t i = 0; i < m; ++i) local[verts[i]] = static_cast<idx>(i);
        std::vector<idx> Sp(m + 1, 0), Si;
        for (std::size_t i = 0; i < m; ++i) {
            idx v = verts[i];
            for (idx p = Cp[v]; p < Cp[v + 1]; ++p)
                if (owner[Ci[p]] == id) Si.push_back(local[Ci[p]]);
            Sp[i + 1] = static_cast<idx>(Si.size());
        }
        auto sub = minimum_degree_core(m, Sp, Si, dense_threshold(m));
        for (std::size_t i = 0; i < m; ++i) perm[lo + i] = static_cast<std::size_t>(verts[sub[i]]);
    };

    struct item {
        std::vector<idx> verts;
        std::size_t lo;     // first output position of this item
        idx id;
    };
    std::vector<item> stack;
    {
        item all{std::vector<idx>(n), 0, next_id++};
        for (std::size_t v = 0; v < n; ++v) {
            all.verts[v] = static_cast<idx>(v);
            owner[v] = all.id;
        }
        stack.push_back(std::move(all));
    }

    // Make a child item from `verts`, owned by a fresh id.
    auto make_item = [&](std::vector<idx> verts, std::size_t lo) {
        item it{std::move(verts), lo, next_id++};
        for (idx v : it.verts) owner[v] = it.id;
        return it;
    };

    while (!stack.empty()) {
        item it = std::move(stack.back());
        stack.pop_back();
        const std::size_t m = it.verts.size();
        if (m == 0) continue;
        if (m <= cutoff) {
            order_small(it.verts, it.lo, it.id);
            continue;
        }

        // Connected components: split into separate items if there is more than one.
        bfs(it.verts.front(), it.id);
        if (order.size() < m) {
            // One search per component, each collecting its vertices.
            std::size_t lo = it.lo;
            std::vector<std::vector<idx>> comps;
            std::vector<char> done(m, 0);
            for (std::size_t i = 0; i < m; ++i) local[it.verts[i]] = static_cast<idx>(i);
            for (std::size_t i = 0; i < m; ++i) {
                if (done[i]) continue;
                bfs(it.verts[i], it.id);
                for (idx w : order) done[local[w]] = 1;
                comps.push_back(order);
            }
            for (auto& c : comps) {
                const std::size_t sz = c.size();
                stack.push_back(make_item(std::move(c), lo));
                lo += sz;
            }
            continue;
        }

        // Level-structure split into A, B and S. False when the item has
        // diameter < 2, so that no level separates anything.
        std::vector<idx> part_a, part_b, part_s;
        auto level_split = [&]() -> bool {
            // Pseudo-peripheral vertex (George-Liu): restart from the minimum-degree
            // vertex of the last level while the number of levels grows.
            idx root = it.verts.front();
            {
                idx best = it.verts.front();
                idx bestd = degree_in(best, it.id);
                for (idx v : it.verts) {
                    idx d = degree_in(v, it.id);
                    if (d < bestd) { best = v; bestd = d; }
                }
                root = best;
            }
            idx nlev = bfs(root, it.id);
            for (;;) {
                idx cand = order[starts[nlev - 1]];
                idx cd = degree_in(cand, it.id);
                for (idx q = starts[nlev - 1]; q < starts[nlev]; ++q) {
                    idx d = degree_in(order[q], it.id);
                    if (d < cd) { cand = order[q]; cd = d; }
                }
                idx nl = bfs(cand, it.id);
                if (nl > nlev) { root = cand; nlev = nl; continue; }
                bfs(root, it.id);   // restore the deepest structure found
                break;
            }

            // A graph of diameter < 2 has no level that separates anything.
            if (nlev < 3) return false;

            // Separator level j in [1, nlev-2]: the narrowest level whose larger
            // side holds at most `balance` of the non-separator vertices. Balance
            // alone picks wide levels (a grid's middle anti-diagonal); width alone
            // picks lopsided splits that barely shrink the problem. If no level
            // meets the bound, take the most balanced one.
            idx sep = 1;
            {
                std::size_t best_width = static_cast<std::size_t>(-1);
                std::size_t best_big = static_cast<std::size_t>(-1);
                idx fallback = 1;
                for (idx j = 1; j <= nlev - 2; ++j) {
                    const std::size_t before = static_cast<std::size_t>(starts[j]);
                    const std::size_t after  = m - static_cast<std::size_t>(starts[j + 1]);
                    const std::size_t width  = static_cast<std::size_t>(starts[j + 1] - starts[j]);
                    const std::size_t big = before > after ? before : after;
                    if (big < best_big) { best_big = big; fallback = j; }
                    if (static_cast<double>(big) <= balance * static_cast<double>(before + after)
                        && width < best_width) {
                        best_width = width;
                        sep = j;
                    }
                }
                if (best_width == static_cast<std::size_t>(-1)) sep = fallback;
            }

            part_a.reserve(static_cast<std::size_t>(starts[sep]));
            for (idx q = 0; q < starts[sep]; ++q) part_a.push_back(order[q]);
            for (idx q = starts[sep + 1]; q < starts[nlev]; ++q) part_b.push_back(order[q]);
            // Thin the separator: a vertex with no neighbour in level sep+1 touches
            // only A and S, so it can join A.
            for (idx q = starts[sep]; q < starts[sep + 1]; ++q) {
                idx v = order[q];
                bool touches_b = false;
                for (idx p = Cp[v]; p < Cp[v + 1] && !touches_b; ++p) {
                    idx w = Ci[p];
                    if (owner[w] == it.id && seen[w] == search && level[w] == sep + 1) touches_b = true;
                }
                (touches_b ? part_s : part_a).push_back(v);
            }
            return true;
        };
        bool split = level_split();

        if (method == nd_separator::multilevel) {
            // The item's induced subgraph, with unit weights.
            for (std::size_t i = 0; i < m; ++i) local[it.verts[i]] = static_cast<idx>(i);
            nd_graph g;
            g.xadj.assign(m + 1, 0);
            g.vwgt.assign(m, 1);
            for (std::size_t i = 0; i < m; ++i) {
                const idx v = it.verts[i];
                for (idx p = Cp[v]; p < Cp[v + 1]; ++p)
                    if (owner[Ci[p]] == it.id) {
                        g.adj.push_back(local[Ci[p]]);
                        g.ewgt.push_back(1);
                    }
                g.xadj[i + 1] = static_cast<idx>(g.adj.size());
            }
            // A fixed seed per item: the same input gives the same ordering.
            const std::uint64_t seed = 0x6E64ull ^ (static_cast<std::uint64_t>(m) << 32)
                                       ^ static_cast<std::uint64_t>(it.lo);
            std::vector<signed char> chosen = multilevel_separator(g, balance, seed);
            // The level-structure separator, polished by the same refinement,
            // is a second candidate: on grids and regular meshes its lopsided
            // corner cuts are shorter than anything the coarse graphs suggest.
            if (split) {
                std::vector<signed char> ls(m);
                for (idx v : part_a) ls[local[v]] = nd_part_a;
                for (idx v : part_b) ls[local[v]] = nd_part_b;
                for (idx v : part_s) ls[local[v]] = nd_part_s;
                nd_refine(g, ls, balance);
                nd_thin_separator(g, ls);
                if (nd_part_weights(g, ls).better_than(nd_part_weights(g, chosen), balance))
                    chosen = std::move(ls);
            }
            const nd_weights pw = nd_part_weights(g, chosen);
            // A degenerate split (one side empty) keeps the plain level structure.
            if (pw.w[nd_part_a] > 0 && pw.w[nd_part_b] > 0) {
                part_a.clear();
                part_b.clear();
                part_s.clear();
                for (std::size_t i = 0; i < m; ++i)
                    (chosen[i] == nd_part_a ? part_a : chosen[i] == nd_part_b ? part_b : part_s)
                        .push_back(it.verts[i]);
                split = true;
            }
        }
        if (!split) {
            order_small(it.verts, it.lo, it.id);
            continue;
        }

        // Separator last, in its level order; then A and B as children.
        const std::size_t na = part_a.size(), nb = part_b.size(), ns = part_s.size();
        for (std::size_t i = 0; i < ns; ++i)
            perm[it.lo + na + nb + i] = static_cast<std::size_t>(part_s[i]);
        for (idx v : part_s) owner[v] = next_id++;   // out of every later search
        stack.push_back(make_item(std::move(part_b), it.lo + na));
        stack.push_back(make_item(std::move(part_a), it.lo));
    }
    return perm;
}

} // namespace detail

/// Nested dissection ordering.
/// Callable object: given a square sparse matrix, returns a permutation p
/// (p[new] = old) that reduces fill in a Cholesky or LDL^T factorization, and in
/// LU when the pivots stay near the diagonal. It works on the pattern of
/// A + A^T, as amd does.
///
/// `cutoff`: subgraphs of at most this many vertices are ordered with AMD
/// instead of being dissected further. Measured on 2-D grids, 16 to 256 change
/// the fill by under 3%; 64 is the default.
///
/// `balance`: a separator is acceptable when the larger of the two parts holds
/// at most this fraction of their vertices; among acceptable separators the
/// smallest is taken. On 2-D grids 0.7 gave the least fill of 0.5-0.9 for level
/// structures, and of 0.65-0.8 for multilevel separators.
///
/// `method`: nd_separator::multilevel (the default) or
/// nd_separator::level_structure. Multilevel gives less fill on every graph
/// measured, and much less on irregular ones. It costs 10-40x AMD's ordering
/// time (0.45 s on a 256 x 256 grid), against 3-4x for level structures.
struct nested_dissection {
    std::size_t cutoff = 64;
    double balance = 0.7;
    nd_separator method = nd_separator::multilevel;

    nested_dissection() = default;
    explicit nested_dissection(std::size_t cutoff_, double balance_ = 0.7,
                               nd_separator method_ = nd_separator::multilevel)
        : cutoff(cutoff_), balance(balance_), method(method_) {}
    explicit nested_dissection(nd_separator method_) : method(method_) {}

    template <typename Value, typename Parameters>
    std::vector<std::size_t> operator()(
        const mat::compressed2D<Value, Parameters>& A) const
    {
        if (A.num_rows() != A.num_cols())
            throw std::invalid_argument("nested_dissection: matrix must be square (got "
                + std::to_string(A.num_rows()) + "x" + std::to_string(A.num_cols()) + ")");
        const std::size_t n = A.num_rows();
        if (n == 0) return {};
        std::vector<detail::idx> Cp, Ci;
        detail::build_aplusat_pattern(A, Cp, Ci);
        return detail::nested_dissection_core(n, Cp, Ci, cutoff, balance, method);
    }
};

} // namespace mtl::sparse::ordering
