#pragma once
// MTL5 -- Block Triangular Form (BTF) via Dulmage-Mendelsohn decomposition
//
// BTF permutes a square sparse matrix into block upper triangular form so that
// only the diagonal blocks need factorization -- the structure circuit-
// simulation (KLU) solvers exploit. It is computed in two stages:
//
//   1. Maximum matching (rows <-> columns) via augmenting paths. For a
//      structurally nonsingular matrix this gives a perfect matching, hence a
//      zero-free diagonal after permuting columns to their matched rows.
//   2. Strongly connected components of the directed graph induced by the
//      matched matrix (Tarjan's algorithm, iterative). The SCCs, ordered
//      topologically, are the diagonal blocks; the condensation is a DAG, so
//      the permuted matrix is block upper triangular.
//
// The result is a row permutation p and a column permutation q (both p[new] =
// old, matching the convention used elsewhere in mtl::sparse) plus block
// boundaries, such that A(p, q) is block upper triangular with a zero-free
// diagonal. This is the cs_dmperm interface from Davis; a single symmetric
// permutation does not suffice in general because the matching folds into the
// columns.
//
// Both the matching DFS and the SCC search are iterative (explicit stacks) to
// avoid stack overflow on large circuit graphs.
//
// References:
//   - Davis, "Direct Methods for Sparse Linear Systems", SIAM 2006, Ch. 7.
//   - Duff & Reid, MC21 (matching) and MC13 (block triangularization).

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <vector>

#include <mtl/mat/compressed2D.hpp>

namespace mtl::sparse::ordering {

/// Result of a Block Triangular Form decomposition.
/// A(row_perm, col_perm) is block upper triangular; block b spans new indices
/// [blocks[b], blocks[b+1]).
struct btf_result {
    std::vector<std::size_t> row_perm;   ///< p[new] = old row
    std::vector<std::size_t> col_perm;   ///< q[new] = old column
    std::vector<std::size_t> blocks;     ///< size nblocks()+1; block boundaries
    bool structurally_singular = false;  ///< true if no perfect matching exists

    std::size_t nblocks() const {
        return blocks.empty() ? 0 : blocks.size() - 1;
    }
};

/// Maximum-cardinality bipartite matching between rows and columns of A.
/// Returns match_row where match_row[i] is the column matched to row i, or -1
/// if row i is unmatched. Uses Kuhn's augmenting-path algorithm with an
/// explicit DFS stack (no recursion).
template <typename Value, typename Parameters>
std::vector<std::ptrdiff_t> maximum_matching(
    const mat::compressed2D<Value, Parameters>& A)
{
    const std::size_t nrows = A.num_rows();
    const std::size_t ncols = A.num_cols();
    const auto& rp = A.ref_major();   // row pointers (CSR)
    const auto& ci = A.ref_minor();   // column indices

    std::vector<std::ptrdiff_t> match_col(ncols, -1);  // column -> matched row
    std::vector<std::ptrdiff_t> match_row(nrows, -1);  // row -> matched column
    std::vector<char> seen(ncols, 0);

    // One DFS frame per row on the current alternating path.
    struct frame { std::size_t u; std::size_t k; std::ptrdiff_t c; };
    std::vector<frame> stack;
    stack.reserve(nrows);

    for (std::size_t s = 0; s < nrows; ++s) {
        std::fill(seen.begin(), seen.end(), char(0));
        stack.clear();
        stack.push_back({s, static_cast<std::size_t>(rp[s]), -1});
        bool child_succeeded = false;

        while (!stack.empty()) {
            // A child augmenting call returned true: commit this frame's edge.
            if (child_succeeded) {
                frame& f = stack.back();
                match_col[f.c] = static_cast<std::ptrdiff_t>(f.u);
                match_row[f.u] = f.c;
                stack.pop_back();
                continue;  // propagate success to the parent frame
            }

            frame& f = stack.back();
            bool advanced = false;
            const std::size_t end = static_cast<std::size_t>(rp[f.u + 1]);
            while (f.k < end) {
                std::size_t c = static_cast<std::size_t>(ci[f.k]);
                if (seen[c]) { ++f.k; continue; }
                seen[c] = 1;
                if (match_col[c] == -1) {
                    // Free column: match it here and unwind the path.
                    match_col[c] = static_cast<std::ptrdiff_t>(f.u);
                    match_row[f.u] = static_cast<std::ptrdiff_t>(c);
                    stack.pop_back();
                    child_succeeded = true;
                    advanced = true;
                    break;
                }
                // Column taken: try to re-match its current owner.
                f.c = static_cast<std::ptrdiff_t>(c);
                std::size_t w = static_cast<std::size_t>(match_col[c]);
                ++f.k;  // resume past c when this frame is revisited
                stack.push_back({w, static_cast<std::size_t>(rp[w]), -1});
                advanced = true;
                break;
            }
            if (advanced) continue;

            // Adjacency exhausted with no augmenting path from this frame.
            stack.pop_back();
            child_succeeded = false;
        }
    }

    return match_row;
}

namespace detail {

/// Iterative Tarjan SCC. adj is the adjacency list of an n-node digraph.
/// Fills comp[v] with the component id of v and returns the number of
/// components. Component ids are assigned in reverse topological order: if
/// there is an edge u -> v then comp[v] <= comp[u].
inline std::size_t tarjan_scc(
    const std::vector<std::vector<std::size_t>>& adj,
    std::vector<std::size_t>& comp)
{
    const std::size_t n = adj.size();
    constexpr std::ptrdiff_t UNVISITED = -1;

    std::vector<std::ptrdiff_t> index(n, UNVISITED);
    std::vector<std::ptrdiff_t> low(n, 0);
    std::vector<char> on_stack(n, 0);
    std::vector<std::size_t> scc_stack;
    comp.assign(n, 0);

    struct work { std::size_t v; std::size_t ei; };
    std::vector<work> call;
    std::ptrdiff_t counter = 0;
    std::size_t ncomp = 0;

    for (std::size_t v0 = 0; v0 < n; ++v0) {
        if (index[v0] != UNVISITED) continue;
        call.push_back({v0, 0});

        while (!call.empty()) {
            work& w = call.back();
            std::size_t v = w.v;

            if (w.ei == 0) {  // first time we enter v
                index[v] = low[v] = counter++;
                scc_stack.push_back(v);
                on_stack[v] = 1;
            }

            if (w.ei < adj[v].size()) {
                std::size_t to = adj[v][w.ei++];
                if (index[to] == UNVISITED) {
                    call.push_back({to, 0});
                } else if (on_stack[to]) {
                    low[v] = std::min(low[v], index[to]);
                }
                continue;
            }

            // All edges of v processed: maybe v is an SCC root.
            if (low[v] == index[v]) {
                while (true) {
                    std::size_t u = scc_stack.back();
                    scc_stack.pop_back();
                    on_stack[u] = 0;
                    comp[u] = ncomp;
                    if (u == v) break;
                }
                ++ncomp;
            }
            call.pop_back();
            if (!call.empty()) {
                std::size_t parent = call.back().v;
                low[parent] = std::min(low[parent], low[v]);
            }
        }
    }
    return ncomp;
}

} // namespace detail

/// Compute the Block Triangular Form of a square sparse matrix.
/// Returns row/column permutations and block boundaries such that
/// A(row_perm, col_perm) is block upper triangular with a zero-free diagonal.
/// If A is structurally singular, the result is flagged and the permutations
/// are still valid (matched columns first, leftover columns filling the rest),
/// but the block-triangular guarantee does not hold.
template <typename Value, typename Parameters>
btf_result block_triangular_form(
    const mat::compressed2D<Value, Parameters>& A)
{
    const std::size_t n = A.num_rows();
    if (A.num_cols() != n) {
        throw std::invalid_argument(
            "block_triangular_form: matrix must be square");
    }
    const auto& rp = A.ref_major();
    const auto& ci = A.ref_minor();

    btf_result result;
    if (n == 0) return result;

    auto match_row = maximum_matching(A);

    // Detect structural singularity (no perfect matching).
    for (std::size_t i = 0; i < n; ++i) {
        if (match_row[i] < 0) { result.structurally_singular = true; break; }
    }

    // match_col[c] = row matched to column c (for building the digraph).
    std::vector<std::ptrdiff_t> match_col(A.num_cols(), -1);
    for (std::size_t i = 0; i < n; ++i)
        if (match_row[i] >= 0)
            match_col[static_cast<std::size_t>(match_row[i])] =
                static_cast<std::ptrdiff_t>(i);

    if (result.structurally_singular) {
        // Best-effort valid permutations; no block structure guarantee.
        result.row_perm.resize(n);
        result.col_perm.resize(n);
        std::vector<char> col_used(A.num_cols(), 0);
        std::size_t pos = 0;
        for (std::size_t i = 0; i < n; ++i) {
            result.row_perm[pos] = i;
            if (match_row[i] >= 0) {
                std::size_t c = static_cast<std::size_t>(match_row[i]);
                result.col_perm[pos] = c;
                col_used[c] = 1;
            }
            ++pos;
        }
        // Fill columns of unmatched rows with unused column indices in order.
        std::size_t next_col = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (match_row[i] < 0) {
                while (next_col < A.num_cols() && col_used[next_col]) ++next_col;
                result.col_perm[i] = next_col;
                if (next_col < A.num_cols()) col_used[next_col] = 1;
            }
        }
        result.blocks = {0, n};  // single (uninformative) block
        return result;
    }

    // Build the digraph on the matched index space: edge i -> match_col[c]
    // for every nonzero A(i, c). Self-loops (the diagonal) are dropped.
    std::vector<std::vector<std::size_t>> adj(n);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t k = static_cast<std::size_t>(rp[i]);
             k < static_cast<std::size_t>(rp[i + 1]); ++k) {
            std::size_t c = static_cast<std::size_t>(ci[k]);
            std::ptrdiff_t to = match_col[c];
            if (to >= 0 && static_cast<std::size_t>(to) != i)
                adj[i].push_back(static_cast<std::size_t>(to));
        }
    }

    std::vector<std::size_t> comp;
    std::size_t ncomp = detail::tarjan_scc(adj, comp);

    // Tarjan assigns comp ids in reverse topological order (edge u->v gives
    // comp[v] <= comp[u]). Reverse to topological order so the permuted matrix
    // is block UPPER triangular: new_block = ncomp-1 - comp.
    std::vector<std::size_t> new_block(n);
    for (std::size_t i = 0; i < n; ++i)
        new_block[i] = ncomp - 1 - comp[i];

    // Stable bucket sort of nodes by block (ascending), node index as tiebreak.
    std::vector<std::size_t> block_size(ncomp, 0);
    for (std::size_t i = 0; i < n; ++i) ++block_size[new_block[i]];

    result.blocks.assign(ncomp + 1, 0);
    for (std::size_t b = 0; b < ncomp; ++b)
        result.blocks[b + 1] = result.blocks[b] + block_size[b];

    result.row_perm.resize(n);
    result.col_perm.resize(n);
    std::vector<std::size_t> cursor(result.blocks.begin(), result.blocks.end() - 1);
    for (std::size_t i = 0; i < n; ++i) {       // i ascending -> stable within block
        std::size_t b = new_block[i];
        std::size_t pos = cursor[b]++;
        result.row_perm[pos] = i;
        result.col_perm[pos] = static_cast<std::size_t>(match_row[i]);
    }

    return result;
}

} // namespace mtl::sparse::ordering
