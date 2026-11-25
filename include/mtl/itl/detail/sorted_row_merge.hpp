#pragma once
// MTL5 -- the common columns of two sorted sparse rows, for the incomplete
// Cholesky factorizations (issue 925).
#include <cstddef>

namespace mtl::itl::detail {

/// Calls f(ri_k, rj_k, k) for every column k < limit stored in both rows, in
/// ascending k. Each row is a sequence of (column, value) pairs sorted by
/// column.
///
/// IC(0) and ILDL form L(i,j) from the sum over k < j of L(i,k) L(j,k). They
/// scanned row j from its start for every k of row i, which costs
/// O(len_i * len_j) per entry and is cubic in the row length: a banded SPD
/// matrix with n = 2000 took 0.056, 0.41 and 2.71 s at bandwidths 50, 100 and
/// 200 (issue 925). One merge of the two sorted rows costs
/// O(len_i + len_j), and it visits the same pairs in the same order, so the
/// sums are bit-identical.
template <typename Row, typename F>
void for_common_columns(const Row& ri, const Row& rj, std::size_t limit, F&& f) {
    auto a = ri.begin();
    auto c = rj.begin();
    const auto ae = ri.end();
    const auto ce = rj.end();
    while (a != ae && c != ce) {
        const auto ka = static_cast<std::size_t>(a->first);
        const auto kc = static_cast<std::size_t>(c->first);
        if (ka >= limit || kc >= limit) break;
        if (ka < kc) {
            ++a;
        } else if (kc < ka) {
            ++c;
        } else {
            f(a->second, c->second, ka);
            ++a;
            ++c;
        }
    }
}

} // namespace mtl::itl::detail
