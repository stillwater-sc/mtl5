#pragma once
// MTL5 -- run an accumulator_traits reduction, split across the thread pool
// when the accumulator can merge its partials (#582).
//
// The mixed-precision reductions (dot, dot_real, two_norm, frobenius_norm with
// an explicit Accumulator) sum their terms into one accumulator. Before #582
// that loop was serial: thread_pool::parallel_reduce combines partials with
// `+` on the result type, which a super-accumulator such as a quire does not
// offer. accumulator_traits::merge is that combine, so a reduction whose
// accumulator satisfies mergeable_accumulator now gives each chunk its own
// partial and merges them in thread order.
//
// What that means for the result:
//   * one thread (the default, MTL5_NUM_THREADS unset) -- the same serial loop
//     as before, bit for bit;
//   * an exact accumulator -- the same bits for any thread count, because the
//     partial sums and their merge are all exact;
//   * a rounding accumulator (a plain wider type, fma_accumulator) -- the terms
//     are regrouped by chunk, so the result is deterministic for a given thread
//     count but not bit-identical to the serial loop: the same contract as the
//     threaded float/double reductions that use parallel_reduce.
#include <algorithm>
#include <cstddef>

#include <mtl/detail/thread_pool.hpp>
#include <mtl/interface/dispatch_traits.hpp>
#include <mtl/math/accumulator_traits.hpp>

namespace mtl::detail {

/// Iteration units per chunk for a reduction of `Value` terms into `Acc`: the
/// smaller of the two types' budgets, since either a software-emulated element
/// or a software accumulator makes each term cost a call rather than a flop.
template <typename Acc, typename Value>
inline constexpr std::size_t accumulate_chunk_units =
    std::min(interface::parallel_chunk_units<Acc>, interface::parallel_chunk_units<Value>);

/// Accumulate `map(acc, begin, end)` over [0, n) into a cleared `Acc`.
///
/// Threaded across the pool, `grain` units per chunk at least, when `Threadable`
/// (the operands are plain dense storage, safe to read concurrently) and the
/// accumulator can merge. Otherwise one serial pass, as before #582.
template <typename Acc, typename Value, bool Threadable, typename MapFn>
Acc accumulate(std::size_t n, std::size_t grain, MapFn&& map) {
    using AT = math::accumulator_traits<Acc, Value>;
    auto init = [] {
        Acc a{};
        AT::clear(a);
        return a;
    };
    if constexpr (Threadable && math::mergeable_accumulator<Acc, Value>) {
        return thread_pool::instance().parallel_accumulate<Acc>(
            n, grain, init, map, [](Acc& into, const Acc& from) { AT::merge(into, from); });
    } else {
        Acc acc = init();
        map(acc, std::size_t{0}, n);
        return acc;
    }
}

} // namespace mtl::detail
