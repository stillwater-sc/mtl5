#pragma once
// MTL5 -- Approximate Minimum Degree core (near-linear), shared by AMD and COLAMD
//
// Faithful port of Timothy Davis's CSparse `cs_amd` (Davis, "Direct Methods for
// Sparse Linear Systems", SIAM, 2006). The previous MTL5 amd/colamd were O(n^2)
// (linear min-degree scan + dense fill insertion); this is the production
// quotient-graph algorithm that runs in near-linear time:
//
//   - degree-bucket lists for O(1) amortized minimum-degree selection
//   - quotient graph with in-place element absorption + garbage collection
//   - approximate (external) degree updates
//   - supervariable (indistinguishable node) detection via hashing
//   - mass elimination and aggressive element absorption
//
// The single core drives both orderings via an `order` parameter:
//   order == 1 : AMD     -- minimum degree on the pattern of A + A^T
//   order == 2 : COLAMD  -- minimum degree on A^T*A with dense rows dropped
//
// Reference: Amestoy, Davis, Duff, "An Approximate Minimum Degree Ordering
//            Algorithm", SIAM J. Matrix Anal. Appl., 17(4), 1996; CSparse cs_amd.

#include <cmath>
#include <cstddef>
#include <type_traits>
#include <vector>

#include <mtl/mat/compressed2D.hpp>

namespace mtl::sparse::ordering::detail {

using idx = std::ptrdiff_t;

// CSparse marking macros: a dead/absorbed object j is stored as CS_FLIP(j).
constexpr idx cs_flip(idx i)   { return -(i) - 2; }
constexpr idx cs_unflip(idx i) { return (i < 0) ? cs_flip(i) : i; }

// a + b in the unsigned domain (well-defined modular wraparound), then back to
// signed (C++20: unsigned->signed conversion is modular two's complement). This
// reproduces CSparse's deliberate wraparound-detection without signed-overflow
// UB. With 64-bit idx the marks never actually overflow for any in-memory
// matrix; this just makes the arithmetic well-defined.
inline idx mark_add(idx a, idx b) {
    using u = std::make_unsigned_t<idx>;
    return static_cast<idx>(static_cast<u>(a) + static_cast<u>(b));
}

// Reset the working-mark array w[] when the running `mark` would overflow.
inline idx cs_wclear(idx mark, idx lemax, std::vector<idx>& w, std::size_t n) {
    if (mark < 2 || mark_add(mark, lemax) < 0) {
        for (std::size_t k = 0; k < n; ++k)
            if (w[k] != 0) w[k] = 1;
        mark = 2;
    }
    return mark;
}

// Iterative depth-first postorder of the assembly tree (CSparse cs_tdfs).
inline idx tdfs(idx j, idx k,
                std::vector<idx>& head, std::vector<idx>& next,
                std::vector<idx>& post, std::vector<idx>& stack) {
    idx top = 0;
    stack[0] = j;
    while (top >= 0) {
        idx p = stack[top];
        idx i = head[p];
        if (i == -1) {
            --top;                 // p has no unordered children left
            post[k++] = p;         // node p is the k-th node in postorder
        } else {
            head[p] = next[i];     // remove i from children of p
            stack[++top] = i;      // start dfs on child i
        }
    }
    return k;
}

/// Minimum-degree ordering on a prebuilt symmetric pattern C (no diagonal),
/// given as integer CSC arrays: Cp_in (size n+1) and Ci_in (size Cp_in[n]).
/// `dense` is the dense-node threshold. Returns permutation p with p[new]=old.
inline std::vector<std::size_t> minimum_degree_core(
    std::size_t n,
    const std::vector<idx>& Cp_in,
    const std::vector<idx>& Ci_in,
    idx dense)
{
    if (n == 0) return {};

    idx cnz = Cp_in[n];
    idx nn = static_cast<idx>(n);
    // Elbow room for fill during quotient-graph updates (CSparse: t). Computed
    // in the unsigned domain (well-defined modular arithmetic); the t < 1 clamp
    // then catches any wrap. For any in-memory matrix t stays far below idx max.
    idx t = mark_add(mark_add(cnz, cnz / 5), 2 * nn);
    if (t < 1) t = 1;

    std::vector<idx> Cp = Cp_in;                 // size n+1 (mutated)
    std::size_t ci_cap = static_cast<std::size_t>(t);   // t >= 1, guarded above
    std::vector<idx> Ci(ci_cap);
    for (idx p = 0; p < cnz; ++p) Ci[p] = Ci_in[p];
    idx nzmax = t;

    // Workspace, each length n+1 (index n is the dense-absorption element).
    std::vector<idx> len(n + 1), nv(n + 1), next(n + 1), head(n + 1),
                     elen(n + 1), degree(n + 1), w(n + 1), hhead(n + 1),
                     last(n + 1);
    std::vector<idx> P(n + 1);                    // result / postorder

    for (idx k = 0; k < nn; ++k) len[k] = Cp[k + 1] - Cp[k];
    len[nn] = 0;

    for (idx i = 0; i <= nn; ++i) {
        head[i] = -1; last[i] = -1; next[i] = -1; hhead[i] = -1;
        nv[i] = 1; w[i] = 1; elen[i] = 0; degree[i] = len[i];
    }
    idx mark = cs_wclear(0, 0, w, n);
    elen[nn] = -2;                                // n is a dead element
    Cp[nn] = -1;                                  // n is a root of assembly tree
    w[nn] = 0;

    idx lemax = 0, mindeg = 0, nel = 0;

    // --- initialize degree lists ---
    for (idx i = 0; i < nn; ++i) {
        idx d = degree[i];
        if (d == 0) {                             // empty node -> dead element
            elen[i] = -2; ++nel; Cp[i] = -1; w[i] = 0;
        } else if (d > dense) {                   // dense node -> absorb into n
            nv[i] = 0; elen[i] = -1; ++nel;
            Cp[i] = cs_flip(nn); ++nv[nn];
        } else {
            if (head[d] != -1) last[head[d]] = i;
            next[i] = head[d]; head[d] = i;
        }
    }

    while (nel < nn) {                            // while selecting pivots
        // --- select node of minimum approximate degree ---
        idx k = -1;
        for (; mindeg < nn && (k = head[mindeg]) == -1; ++mindeg) ;
        if (next[k] != -1) last[next[k]] = -1;
        head[mindeg] = next[k];                   // remove k from degree list
        idx elenk = elen[k];
        idx nvk = nv[k];
        nel += nvk;

        // --- garbage collection ---
        if (elenk > 0 && cnz + mindeg >= nzmax) {
            for (idx j = 0; j < nn; ++j) {
                idx p;
                if ((p = Cp[j]) >= 0) {           // live node/element
                    Cp[j] = Ci[p];
                    Ci[p] = cs_flip(j);
                }
            }
            idx q = 0, p = 0;
            while (p < cnz) {
                idx j = cs_flip(Ci[p++]);
                if (j >= 0) {                      // found object j
                    Ci[q] = Cp[j];                 // restore first entry of object
                    Cp[j] = q++;                   // new pointer to object j
                    for (idx k3 = 0; k3 < len[j] - 1; ++k3) Ci[q++] = Ci[p++];
                }
            }
            cnz = q;
        }

        // --- construct new element ---
        idx dk = 0;
        nv[k] = -nvk;                             // flag k as in Lk
        idx p = Cp[k];
        idx pk1 = (elenk == 0) ? p : cnz;
        idx pk2 = pk1;
        for (idx k1 = 1; k1 <= elenk + 1; ++k1) {
            idx e, pj, ln;
            if (k1 > elenk) { e = k; pj = p; ln = len[k] - elenk; }
            else            { e = Ci[p++]; pj = Cp[e]; ln = len[e]; }
            for (idx k2 = 1; k2 <= ln; ++k2) {
                idx i = Ci[pj++];
                idx nvi;
                if ((nvi = nv[i]) <= 0) continue; // dead or already in Lk
                dk += nvi;
                nv[i] = -nvi;                     // negate: i is in Lk
                Ci[pk2++] = i;
                if (next[i] != -1) last[next[i]] = last[i];
                if (last[i] != -1) next[last[i]] = next[i];
                else head[degree[i]] = next[i];
            }
            if (e != k) {
                Cp[e] = cs_flip(k);               // absorb e into k
                w[e] = 0;
            }
        }
        if (elenk != 0) cnz = pk2;                // Ck in Ci[cnz..nzmax-1]
        degree[k] = dk;
        Cp[k] = pk1;
        len[k] = pk2 - pk1;
        elen[k] = -2;                             // k is now an element

        // --- find set differences (scan 1) ---
        mark = cs_wclear(mark, lemax, w, n);
        for (idx pk = pk1; pk < pk2; ++pk) {
            idx i = Ci[pk];
            idx eln;
            if ((eln = elen[i]) <= 0) continue;
            idx nvi = -nv[i];
            idx wnvi = mark - nvi;
            for (idx pp = Cp[i]; pp <= Cp[i] + eln - 1; ++pp) {
                idx e = Ci[pp];
                if (w[e] >= mark)      w[e] -= nvi;
                else if (w[e] != 0)    w[e] = degree[e] + wnvi;
            }
        }

        // --- degree update (scan 2) ---
        for (idx pk = pk1; pk < pk2; ++pk) {
            idx i = Ci[pk];
            idx p1 = Cp[i];
            idx p2 = p1 + elen[i] - 1;
            idx pn = p1;
            idx h = 0, d = 0;
            for (idx pp = p1; pp <= p2; ++pp) {
                idx e = Ci[pp];
                if (w[e] != 0) {
                    idx dext = w[e] - mark;
                    if (dext > 0) {
                        d += dext; Ci[pn++] = e; h += e;
                    } else {
                        Cp[e] = cs_flip(k); w[e] = 0;  // aggressive absorption
                    }
                }
            }
            elen[i] = pn - p1 + 1;
            idx p3 = pn;
            idx p4 = p1 + len[i];
            for (idx pp = p2 + 1; pp < p4; ++pp) {
                idx j = Ci[pp];
                idx nvj;
                if ((nvj = nv[j]) <= 0) continue;
                d += nvj; Ci[pn++] = j; h += j;
            }
            if (d == 0) {                          // mass elimination
                Cp[i] = cs_flip(k);
                idx nvi = -nv[i];
                dk -= nvi; nvk += nvi; nel += nvi; nv[i] = 0; elen[i] = -1;
            } else {
                degree[i] = (degree[i] < d) ? degree[i] : d;
                Ci[pn] = Ci[p3];
                Ci[p3] = Ci[p1];
                Ci[p1] = k;
                len[i] = pn - p1 + 1;
                h %= nn; h = (h < 0) ? h + nn : h;
                next[i] = hhead[h]; hhead[h] = i;
                last[i] = h;
            }
        }
        degree[k] = dk;
        lemax = (lemax > dk) ? lemax : dk;
        mark = cs_wclear(mark_add(mark, lemax), lemax, w, n);

        // --- supervariable detection ---
        for (idx pk = pk1; pk < pk2; ++pk) {
            idx i = Ci[pk];
            if (nv[i] >= 0) continue;              // skip if not in Lk
            idx h = last[i];
            i = hhead[h];
            hhead[h] = -1;
            for (; i != -1 && next[i] != -1; i = next[i], ++mark) {
                idx ln = len[i];
                idx eln = elen[i];
                for (idx pp = Cp[i] + 1; pp <= Cp[i] + ln - 1; ++pp) w[Ci[pp]] = mark;
                idx jlast = i;
                for (idx j = next[i]; j != -1; ) {
                    bool ok = (len[j] == ln) && (elen[j] == eln);
                    for (idx pp = Cp[j] + 1; ok && pp <= Cp[j] + ln - 1; ++pp)
                        if (w[Ci[pp]] != mark) ok = false;
                    if (ok) {                      // i and j identical -> absorb j
                        Cp[j] = cs_flip(i);
                        nv[i] += nv[j];
                        nv[j] = 0; elen[j] = -1;
                        j = next[j];
                        next[jlast] = j;
                    } else {
                        jlast = j; j = next[j];
                    }
                }
            }
        }

        // --- finalize new element ---
        idx pf = pk1;
        for (idx pk = pk1; pk < pk2; ++pk) {
            idx i = Ci[pk];
            idx nvi;
            if ((nvi = -nv[i]) <= 0) continue;
            nv[i] = nvi;                           // restore nv[i]
            idx d = degree[i] + dk - nvi;
            idx cap = nn - nel - nvi;
            if (d > cap) d = cap;
            if (head[d] != -1) last[head[d]] = i;
            next[i] = head[d]; last[i] = -1; head[d] = i;
            mindeg = (mindeg < d) ? mindeg : d;
            degree[i] = d;
            Ci[pf++] = i;
        }
        nv[k] = nvk;
        if ((len[k] = pf - pk1) == 0) { Cp[k] = -1; w[k] = 0; }
        if (elenk != 0) cnz = pf;
    }

    // --- postordering ---
    for (idx i = 0; i < nn; ++i) Cp[i] = cs_flip(Cp[i]);
    for (idx j = 0; j <= nn; ++j) head[j] = -1;
    for (idx j = nn; j >= 0; --j) {               // unordered nodes
        if (nv[j] > 0) continue;
        next[j] = head[Cp[j]]; head[Cp[j]] = j;
    }
    for (idx e = nn; e >= 0; --e) {               // elements
        if (nv[e] <= 0) continue;
        if (Cp[e] != -1) { next[e] = head[Cp[e]]; head[Cp[e]] = e; }
    }
    idx kk = 0;
    for (idx i = 0; i <= nn; ++i)
        if (Cp[i] == -1) kk = tdfs(i, kk, head, next, P, w);

    std::vector<std::size_t> perm(n);
    for (std::size_t i = 0; i < n; ++i) perm[i] = static_cast<std::size_t>(P[i]);
    return perm;
}

// Dense-node threshold matching CSparse: max(16, 10*sqrt(n)), clamped to n-2.
inline idx dense_threshold(std::size_t n) {
    idx d = static_cast<idx>(10.0 * std::sqrt(static_cast<double>(n)));
    if (d < 16) d = 16;
    idx cap = static_cast<idx>(n) - 2;
    if (d > cap) d = cap;
    return d;
}

/// Build the symmetric pattern of A + A^T (no diagonal) as CSC integer arrays.
/// For the symmetric case (CSR == CSC), this is the AMD input.
template <typename Value, typename Parameters>
void build_aplusat_pattern(const mat::compressed2D<Value, Parameters>& A,
                           std::vector<idx>& Cp, std::vector<idx>& Ci) {
    std::size_t n = A.num_rows();
    const auto& starts  = A.ref_major();
    const auto& indices = A.ref_minor();

    // Distinct neighbor accumulation per node, deduped with a stamp array.
    std::vector<std::vector<idx>> rows_t(n);      // transpose adjacency (for A^T)
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t k = starts[i]; k < starts[i + 1]; ++k) {
            std::size_t j = indices[k];
            if (j != i) rows_t[j].push_back(static_cast<idx>(i));
        }

    std::vector<idx> stamp(n, -1);
    Cp.assign(n + 1, 0);
    Ci.clear();
    for (std::size_t i = 0; i < n; ++i) {
        // neighbors from row i of A
        for (std::size_t k = starts[i]; k < starts[i + 1]; ++k) {
            std::size_t j = indices[k];
            if (j != i && stamp[j] != static_cast<idx>(i)) {
                stamp[j] = static_cast<idx>(i);
                Ci.push_back(static_cast<idx>(j));
            }
        }
        // neighbors from column i of A (i.e. row i of A^T)
        for (idx j : rows_t[i]) {
            if (stamp[j] != static_cast<idx>(i)) {
                stamp[j] = static_cast<idx>(i);
                Ci.push_back(j);
            }
        }
        Cp[i + 1] = static_cast<idx>(Ci.size());
    }
}

/// Build the column-intersection pattern (A^T*A, no diagonal) over the columns
/// of A, dropping dense rows of A. This is the COLAMD input.
template <typename Value, typename Parameters>
void build_ata_pattern(const mat::compressed2D<Value, Parameters>& A,
                       std::vector<idx>& Cp, std::vector<idx>& Ci) {
    std::size_t m = A.num_rows();
    std::size_t n = A.num_cols();
    const auto& starts  = A.ref_major();
    const auto& indices = A.ref_minor();

    idx dense = dense_threshold(m == 0 ? n : m);

    // col -> list of (kept) rows containing that column.
    std::vector<std::vector<idx>> col_rows(n);
    for (std::size_t r = 0; r < m; ++r) {
        std::size_t rlen = starts[r + 1] - starts[r];
        if (static_cast<idx>(rlen) > dense) continue;   // drop dense row
        for (std::size_t k = starts[r]; k < starts[r + 1]; ++k)
            col_rows[indices[k]].push_back(static_cast<idx>(r));
    }

    std::vector<idx> stamp(n, -1);
    Cp.assign(n + 1, 0);
    Ci.clear();
    for (std::size_t c = 0; c < n; ++c) {
        for (idx r : col_rows[c]) {
            for (std::size_t k = starts[r]; k < starts[r + 1]; ++k) {
                std::size_t cc = indices[k];
                if (cc != c && stamp[cc] != static_cast<idx>(c)) {
                    stamp[cc] = static_cast<idx>(c);
                    Ci.push_back(static_cast<idx>(cc));
                }
            }
        }
        Cp[c + 1] = static_cast<idx>(Ci.size());
    }
}

/// order == 1 : AMD (A + A^T).   order == 2 : COLAMD (A^T A, drop dense rows).
template <typename Value, typename Parameters>
std::vector<std::size_t> minimum_degree(
    int order, const mat::compressed2D<Value, Parameters>& A)
{
    std::size_t n = (order == 1) ? A.num_rows() : A.num_cols();
    if (n == 0) return {};

    std::vector<idx> Cp, Ci;
    if (order == 1) build_aplusat_pattern(A, Cp, Ci);
    else            build_ata_pattern(A, Cp, Ci);

    return minimum_degree_core(n, Cp, Ci, dense_threshold(n));
}

} // namespace mtl::sparse::ordering::detail
