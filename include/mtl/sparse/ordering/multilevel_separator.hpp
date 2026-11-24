#pragma once
// MTL5 -- Multilevel vertex separators for nested dissection
//
// Finds a small vertex separator S of a connected, weighted graph, splitting the
// rest into parts A and B with no edge between them, in the style of METIS's
// nested dissection (Karypis & Kumar 1998):
//
//   1. Coarsen. Heavy-edge matching in a random vertex order collapses matched
//      pairs into one vertex: vertex weights count the original vertices, and
//      edge weights sum the edges they replace. Repeat until the graph has about
//      100 vertices or a level shrinks it by less than 15%. A cap on the vertex
//      weight keeps the coarse vertices of comparable size.
//   2. Initial separator on the coarsest graph. Several trials, each growing a
//      region breadth-first from a seed (the first is pseudo-peripheral, the rest
//      random) until it holds half the weight, taking the region's boundary as
//      the separator and refining it as in step 3. The best balanced one wins.
//   3. Uncoarsen. Project the labels to the next finer graph, which keeps the
//      separator valid: an edge between a fine A vertex and a fine B vertex would
//      be an edge between their coarse vertices. Then refine with vertex-separator
//      Fiduccia-Mattheyses: move a separator vertex v into A or B, which pulls
//      v's neighbours on the other side into S, so the separator shrinks by
//      w(v) - w(those neighbours). Each pass moves a vertex at most once, accepts
//      a bounded run of non-improving moves to climb out of local minima, and
//      rolls back to the best state seen. Moves keep the larger part at most
//      `balance` of the weight of A and B.
//   4. Thin. A separator vertex with no neighbour in B moves to A, and one with
//      no neighbour in A moves to B. Afterwards every vertex of S has a
//      neighbour in A and a neighbour in B: no single vertex can leave S.
//
// The random choices come from splitmix64 with a fixed seed, and every priority
// has a total order (gain, then vertex), so the result is the same with every
// compiler and standard library.
//
// References: Karypis & Kumar, "A Fast and High Quality Multilevel Scheme for
//             Partitioning Irregular Graphs", SIAM J. Sci. Comput. 20(1), 1998.
//             Hendrickson & Leland, "A Multilevel Algorithm for Partitioning
//             Graphs", Supercomputing '95.
//             Ashcraft & Liu, "Robust Ordering of Sparse Matrices using
//             Multisection", SIAM J. Matrix Anal. Appl. 19(3), 1998.
//             Fiduccia & Mattheyses, "A Linear-Time Heuristic for Improving
//             Network Partitions", 19th Design Automation Conference, 1982.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <mtl/sparse/ordering/minimum_degree.hpp>

namespace mtl::sparse::ordering::detail {

/// splitmix64 (Steele, Lea & Flood 2014): fixed output for a fixed seed on every
/// platform, unlike the std:: distributions.
struct nd_rng {
    std::uint64_t state;
    explicit nd_rng(std::uint64_t seed) : state(seed) {}
    std::uint64_t next() {
        std::uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    idx below(idx m) { return static_cast<idx>(next() % static_cast<std::uint64_t>(m)); }
};

/// A weighted undirected graph in CSR form, without self-loops; each edge is
/// stored in both directions.
struct nd_graph {
    std::vector<idx> xadj{0}, adj, ewgt, vwgt;
    idx size() const { return static_cast<idx>(vwgt.size()); }
};

/// Part labels.
inline constexpr signed char nd_part_a = 0;
inline constexpr signed char nd_part_b = 1;
inline constexpr signed char nd_part_s = 2;

/// One level of heavy-edge matching. Returns the coarse graph and fills cmap
/// (fine vertex -> coarse vertex).
inline nd_graph nd_coarsen(const nd_graph& g, std::vector<idx>& cmap, nd_rng& rng, idx maxvwgt) {
    const idx n = g.size();
    std::vector<idx> visit(static_cast<std::size_t>(n));
    for (idx i = 0; i < n; ++i) visit[i] = i;
    for (idx i = n - 1; i > 0; --i) std::swap(visit[i], visit[rng.below(i + 1)]);

    std::vector<idx> match(static_cast<std::size_t>(n), -1);
    cmap.assign(static_cast<std::size_t>(n), -1);
    std::vector<idx> rep;                     // rep[c] = first fine vertex of coarse c
    idx nc = 0;
    for (idx u : visit) {
        if (match[u] >= 0) continue;
        idx best = -1, bw = -1;
        for (idx p = g.xadj[u]; p < g.xadj[u + 1]; ++p) {
            const idx v = g.adj[p];
            if (match[v] < 0 && g.vwgt[u] + g.vwgt[v] <= maxvwgt && g.ewgt[p] > bw) {
                best = v;
                bw = g.ewgt[p];
            }
        }
        if (best < 0) {
            match[u] = u;
        } else {
            match[u] = best;
            match[best] = u;
            cmap[best] = nc;
        }
        cmap[u] = nc++;
        rep.push_back(u);
    }

    nd_graph c;
    c.vwgt.assign(static_cast<std::size_t>(nc), 0);
    c.xadj.assign(static_cast<std::size_t>(nc) + 1, 0);
    std::vector<idx> slot(static_cast<std::size_t>(nc), -1);   // position of an edge to c in c.adj
    for (idx cv = 0; cv < nc; ++cv) {
        const idx u = rep[cv], v = match[u];
        const idx start = static_cast<idx>(c.adj.size());
        const idx members[2] = {u, v};
        for (int k = 0; k < (v == u ? 1 : 2); ++k) {
            const idx w = members[k];
            c.vwgt[cv] += g.vwgt[w];
            for (idx p = g.xadj[w]; p < g.xadj[w + 1]; ++p) {
                const idx cw = cmap[g.adj[p]];
                if (cw == cv) continue;
                if (slot[cw] >= start) {
                    c.ewgt[slot[cw]] += g.ewgt[p];
                } else {
                    slot[cw] = static_cast<idx>(c.adj.size());
                    c.adj.push_back(cw);
                    c.ewgt.push_back(g.ewgt[p]);
                }
            }
        }
        c.xadj[cv + 1] = static_cast<idx>(c.adj.size());
    }
    return c;
}

/// Part weights of a labelling.
struct nd_weights {
    idx w[3] = {0, 0, 0};
    bool balanced(double balance) const {
        const idx big = w[0] > w[1] ? w[0] : w[1];
        return static_cast<double>(big) <= balance * static_cast<double>(w[0] + w[1]);
    }
    idx imbalance() const { return w[0] > w[1] ? w[0] - w[1] : w[1] - w[0]; }
    /// Better than `o`: balanced beats unbalanced; between balanced states the
    /// smaller separator, then the better balance; between unbalanced states
    /// the better balance.
    bool better_than(const nd_weights& o, double balance) const {
        const bool b = balanced(balance), ob = o.balanced(balance);
        if (b != ob) return b;
        if (!b) return imbalance() < o.imbalance();
        if (w[2] != o.w[2]) return w[2] < o.w[2];
        return imbalance() < o.imbalance();
    }
};

inline nd_weights nd_part_weights(const nd_graph& g, const std::vector<signed char>& where) {
    nd_weights pw;
    for (idx v = 0; v < g.size(); ++v) pw.w[where[v]] += g.vwgt[v];
    return pw;
}

/// Vertex-separator Fiduccia-Mattheyses refinement of `where`, in place.
inline void nd_refine(const nd_graph& g, std::vector<signed char>& where, double balance,
                      int max_passes = 10) {
    const idx n = g.size();
    nd_weights pw = nd_part_weights(g, where);
    const idx limit = std::clamp<idx>(n / 100, 15, 100);

    // side_w[2v + k]: the weight of v's neighbours in part k (A or B), kept
    // current through every relabelling so that a gain costs O(1).
    std::vector<idx> side_w(2 * static_cast<std::size_t>(n), 0);
    for (idx v = 0; v < n; ++v)
        for (idx p = g.xadj[v]; p < g.xadj[v + 1]; ++p) {
            const signed char lw = where[g.adj[p]];
            if (lw != nd_part_s) side_w[2 * v + lw] += g.vwgt[g.adj[p]];
        }
    auto relabel = [&](idx v, signed char to) {
        const signed char from = where[v];
        pw.w[from] -= g.vwgt[v];
        pw.w[to] += g.vwgt[v];
        where[v] = to;
        for (idx p = g.xadj[v]; p < g.xadj[v + 1]; ++p) {
            const idx u = g.adj[p];
            if (from != nd_part_s) side_w[2 * u + from] -= g.vwgt[v];
            if (to != nd_part_s) side_w[2 * u + to] += g.vwgt[v];
        }
    };
    // gain(v, to): how much the separator shrinks when v moves to side `to`,
    // pulling its neighbours on the other side into the separator.
    auto gain = [&](idx v, int to) { return g.vwgt[v] - side_w[2 * v + (1 - to)]; };

    std::vector<char> locked(static_cast<std::size_t>(n), 0);
    std::vector<std::pair<idx, signed char>> log;   // (vertex, previous label)
    using entry = std::pair<idx, idx>;              // (gain, vertex): a total order
    std::vector<entry> heap[2];
    auto push = [&](int to, idx v) {
        heap[to].push_back({gain(v, to), v});
        std::push_heap(heap[to].begin(), heap[to].end());
    };
    // Discard stale entries until the top is current; false if none is left.
    auto top = [&](int to, idx& v, idx& gn) {
        auto& h = heap[to];
        while (!h.empty()) {
            const entry e = h.front();
            std::pop_heap(h.begin(), h.end());
            h.pop_back();
            if (where[e.second] != nd_part_s || locked[e.second]) continue;
            if (gain(e.second, to) != e.first) { push(to, e.second); continue; }
            h.push_back(e);                              // still current: put it back
            std::push_heap(h.begin(), h.end());
            v = e.second;
            gn = e.first;
            return true;
        }
        return false;
    };

    for (int pass = 0; pass < max_passes; ++pass) {
        heap[0].clear();
        heap[1].clear();
        std::fill(locked.begin(), locked.end(), 0);
        log.clear();
        for (idx v = 0; v < n; ++v)
            if (where[v] == nd_part_s) { push(0, v); push(1, v); }

        nd_weights best = pw;
        std::size_t best_len = 0;
        idx nbad = 0;
        for (;;) {
            const bool unbalanced = !pw.balanced(balance);
            const int lighter = pw.w[0] <= pw.w[1] ? 0 : 1;
            int to = -1;
            idx v = -1, gn = 0;
            for (int side = 0; side < 2; ++side) {
                if (unbalanced && side != lighter) continue;
                idx sv, sg;
                if (!top(side, sv, sg)) continue;
                if (!unbalanced) {
                    nd_weights after = pw;
                    after.w[side] += g.vwgt[sv];
                    after.w[1 - side] -= g.vwgt[sv] - sg;
                    if (!after.balanced(balance)) continue;
                }
                if (to < 0 || sg > gn || (sg == gn && side == lighter)) {
                    to = side;
                    v = sv;
                    gn = sg;
                }
            }
            if (to < 0) break;

            const signed char other = static_cast<signed char>(1 - to);
            locked[v] = 1;
            log.push_back({v, nd_part_s});
            relabel(v, static_cast<signed char>(to));
            // v's separator neighbours now lose gain toward `other`; the stale
            // entries are corrected when they surface.
            for (idx p = g.xadj[v]; p < g.xadj[v + 1]; ++p) {
                const idx u = g.adj[p];
                if (where[u] != other) continue;
                log.push_back({u, other});
                relabel(u, nd_part_s);
                if (!locked[u]) { push(0, u); push(1, u); }
                // u left `other`, so its separator neighbours gain toward `to`.
                for (idx q = g.xadj[u]; q < g.xadj[u + 1]; ++q) {
                    const idx w = g.adj[q];
                    if (where[w] == nd_part_s && !locked[w]) push(to, w);
                }
            }

            if (pw.better_than(best, balance)) {
                best = pw;
                best_len = log.size();
                nbad = 0;
            } else if (++nbad > limit) {
                break;
            }
        }

        // Roll back to the best state.
        for (std::size_t i = log.size(); i > best_len; --i)
            relabel(log[i - 1].first, log[i - 1].second);
        if (best_len == 0) break;   // the pass improved nothing
    }
}

/// Initial separator on a (small) graph: grow a region breadth-first from a
/// seed to half the weight, take its boundary, refine; keep the best of
/// `trials`.
inline std::vector<signed char> nd_initial_separator(const nd_graph& g, double balance,
                                                     nd_rng& rng, int trials = 4) {
    const idx n = g.size();
    idx total = 0;
    for (idx w : g.vwgt) total += w;

    std::vector<idx> queue;
    std::vector<char> seen(static_cast<std::size_t>(n));
    auto bfs_last = [&](idx root) {
        std::fill(seen.begin(), seen.end(), 0);
        queue.assign(1, root);
        seen[root] = 1;
        for (std::size_t h = 0; h < queue.size(); ++h) {
            const idx v = queue[h];
            for (idx p = g.xadj[v]; p < g.xadj[v + 1]; ++p)
                if (!seen[g.adj[p]]) { seen[g.adj[p]] = 1; queue.push_back(g.adj[p]); }
        }
        return queue.back();
    };

    // Region sizes tried for each seed: from the smallest the balance bound
    // allows up to half the weight. A smaller part with a shorter boundary (a
    // corner cut of a grid) can beat a balanced one.
    const double lo_frac = std::max(0.0, 1.0 - balance);
    constexpr int fractions = 3;

    std::vector<signed char> best_where, where;
    nd_weights best;
    for (int t = 0; t < trials; ++t) {
        const idx seed = t == 0 ? bfs_last(bfs_last(0)) : rng.below(n);
        // Breadth-first order from the seed, restarting from unvisited vertices
        // if the search runs out (a disconnected graph).
        std::fill(seen.begin(), seen.end(), 0);
        queue.assign(1, seed);
        seen[seed] = 1;
        idx next_root = 0;
        for (std::size_t h = 0; h < static_cast<std::size_t>(n); ++h) {
            if (h == queue.size()) {
                while (seen[next_root]) ++next_root;
                seen[next_root] = 1;
                queue.push_back(next_root);
            }
            const idx v = queue[h];
            for (idx p = g.xadj[v]; p < g.xadj[v + 1]; ++p)
                if (!seen[g.adj[p]]) { seen[g.adj[p]] = 1; queue.push_back(g.adj[p]); }
        }
        for (int f = 0; f < fractions; ++f) {
            const double frac = lo_frac + (0.5 - lo_frac) * f / (fractions - 1);
            const double target = frac * static_cast<double>(total);
            // A is the shortest prefix of the order holding the target weight.
            where.assign(static_cast<std::size_t>(n), nd_part_b);
            idx wa = 0;
            for (std::size_t h = 0; h < queue.size() && static_cast<double>(wa) < target; ++h) {
                where[queue[h]] = nd_part_a;
                wa += g.vwgt[queue[h]];
            }
            // The boundary of A becomes the separator.
            for (idx v = 0; v < n; ++v) {
                if (where[v] != nd_part_a) continue;
                for (idx p = g.xadj[v]; p < g.xadj[v + 1]; ++p)
                    if (where[g.adj[p]] == nd_part_b) { where[v] = nd_part_s; break; }
            }
            nd_refine(g, where, balance);
            const nd_weights pw = nd_part_weights(g, where);
            if (best_where.empty() || pw.better_than(best, balance)) {
                best = pw;
                best_where = where;
            }
        }
    }
    return best_where;
}

/// Move separator vertices that touch only one side to that side. Afterwards
/// every separator vertex has a neighbour in A and a neighbour in B.
inline void nd_thin_separator(const nd_graph& g, std::vector<signed char>& where) {
    for (idx v = 0; v < g.size(); ++v) {
        if (where[v] != nd_part_s) continue;
        bool touches[2] = {false, false};
        for (idx p = g.xadj[v]; p < g.xadj[v + 1]; ++p) {
            const signed char lw = where[g.adj[p]];
            if (lw != nd_part_s) touches[lw] = true;
        }
        if (!touches[nd_part_b]) where[v] = nd_part_a;
        else if (!touches[nd_part_a]) where[v] = nd_part_b;
    }
}

/// Multilevel vertex separator of a connected graph. Returns where[v] in
/// {nd_part_a, nd_part_b, nd_part_s}, with no edge between A and B.
inline std::vector<signed char> multilevel_separator(const nd_graph& g, double balance,
                                                     std::uint64_t seed) {
    const idx n = g.size();
    if (n == 0) return {};
    constexpr idx coarsen_to = 100;
    nd_rng rng(seed);
    idx total = 0;
    for (idx w : g.vwgt) total += w;
    const idx maxvwgt = std::max<idx>(1, (3 * total) / (2 * coarsen_to));

    std::vector<nd_graph> coarse;               // coarse[i] is level i + 1
    std::vector<std::vector<idx>> cmaps;        // cmaps[i]: level i -> level i + 1
    auto level = [&](std::size_t i) -> const nd_graph& { return i == 0 ? g : coarse[i - 1]; };
    while (level(coarse.size()).size() > coarsen_to) {
        const nd_graph& fine = level(coarse.size());
        std::vector<idx> cmap;
        nd_graph c = nd_coarsen(fine, cmap, rng, maxvwgt);
        if (static_cast<double>(c.size()) > 0.85 * static_cast<double>(fine.size())) break;
        cmaps.push_back(std::move(cmap));
        coarse.push_back(std::move(c));
    }

    std::vector<signed char> where = nd_initial_separator(level(coarse.size()), balance, rng);
    for (std::size_t i = coarse.size(); i > 0; --i) {
        const nd_graph& fine = level(i - 1);
        const auto& cmap = cmaps[i - 1];
        std::vector<signed char> fw(static_cast<std::size_t>(fine.size()));
        for (idx v = 0; v < fine.size(); ++v) fw[v] = where[cmap[v]];
        where = std::move(fw);
        nd_refine(fine, where, balance);
    }
    nd_thin_separator(g, where);
    return where;
}

} // namespace mtl::sparse::ordering::detail
