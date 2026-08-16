# Parallelizing algorithms in MTL5: patterns and pitfalls

A field guide distilled from the on-node threading rollout ([#297](https://github.com/stillwater-sc/mtl5/issues/297), building on the thread-pool epic [#221](https://github.com/stillwater-sc/mtl5/issues/221)). It records the parallelization *patterns* that worked, and — more valuably — the *correctness, caching, encapsulation, and exception-safety pitfalls* that code review surfaced, so the next algorithm we parallelize can reuse the insight instead of rediscovering it.

The concrete case studies are:

- **Batch 1 ([#298](https://github.com/stillwater-sc/mtl5/pull/298))** — dense LU trailing-submatrix update, Cholesky column update.
- **Batch 2 ([#300](https://github.com/stillwater-sc/mtl5/pull/300))** — Householder reflector application (`apply_householder_left`/`_right`), which powers QR / LQ / Hessenberg / the symmetric eigensolver.
- **Batch 3 pilot ([#301](https://github.com/stillwater-sc/mtl5/pull/301))** — sparse triangular solve via level scheduling.

---

## 1. The non-negotiable properties

Every parallel kernel in MTL5 must hold these, in priority order:

1. **Serial path unchanged.** At `MTL5_NUM_THREADS=1` (the default) the code runs the original serial path, byte-identical in behavior and codegen. The persistent `detail::thread_pool` makes this automatic: `parallel_for` runs a single `body(0, n)` when the team is size 1 or the work is below the grain, so *always calling `parallel_for`* is a zero-overhead no-op serially.
2. **Bit-identical across thread counts** wherever the serial code is deterministic. Not just "close" — exactly `==`. This is what lets us assert equality in tests and makes threaded results reproducible.
3. **Race-free**, proven under ThreadSanitizer.
4. **Deterministic schedule.** The partition of work is a pure function of the input, so results never depend on scheduling timing.

Determinism (2) is a *choice we design for*, not a given. See §3.

---

## 2. The two parallelism patterns

### 2a. Embarrassingly parallel over independent outputs

Most dense kernels reduce to: *"compute a set of outputs, each written once, each reading only shared read-only data or earlier-finalized outputs."* Partition the outputs into contiguous chunks; each chunk owns a disjoint output range.

| Kernel | Parallelized over | Each unit writes | reads (shared, read-only) |
|---|---|---|---|
| LU trailing update | rows `i > k` | row `i` | pivot row `k`, `A(k,k)` |
| Cholesky column update | rows `i > j` | `A(i,j)` | columns `< j`, column `j` |
| `apply_householder_left` | columns `j ≥ col` | column `j` | reflector `v` |
| `apply_householder_right` | rows `i ≥ row` | its `vlen` entries of row `i` | reflector `v` |

Because each output is produced by exactly one chunk with the same arithmetic in the same order as serial, these are **bit-identical and race-free "for free."** This is the pattern to reach for first — if you can phrase the kernel this way, the parallelization is almost mechanical.

### 2b. Dependency-structured parallelism → level scheduling

When outputs depend on each other (triangular solve, factorization panels), you can't partition arbitrarily. Group work into **levels**: `level[i] = 1 + max(level of i's dependencies)`. All items in a level are mutually independent, so a level is an embarrassingly-parallel step (pattern 2a); levels run in sequence with an implicit barrier between them. Parallelism is bounded by the width of each level, not the total work — a chain (bandwidth-1 matrix) has one item per level and no parallelism; a block-diagonal or arrow structure has wide levels.

Level scheduling needs a **preprocessing pass** (build the levels, O(nnz)) whose cost is comparable to one solve, so it only pays off when the schedule is **built once and reused** across many solves. That reuse requirement is the source of most of the pitfalls in §5.

---

## 3. Bit-identity is an ordering property — preserve the accumulation order

The subtle lesson from the sparse batch: *the same math in a different order is not bit-identical in floating point.* Two mathematically-equal formulations can give different last-bit results.

The sparse lower solve was a **CSC column-scatter**: `for each column j: x[j] /= L(j,j); for i>j: x[i] -= L(i,j)·x[j]`. Column `j` writes into many `x[i]`. Parallelizing columns is a double failure: (a) two columns write the same `x[i]` → **race**, and (b) their subtraction order is not fixed → **non-deterministic** and *not* bit-identical to serial.

The fix was to switch to a **row-oriented gather**: `x[i] = (b[i] − Σ_{k<i} L(i,k)·x[k]) / L(i,i)`. Now each `x[i]` is written once (race-free), *and* — the key — if the row's entries are iterated in **increasing column order**, the accumulation is the exact same sequence of `b[i]` minus successive products that the CSC scatter performed. So the gather is **bit-identical to the scatter**, and level scheduling over it is bit-identical across thread counts.

> **Rule:** to parallelize a reduction/accumulation deterministically, transform it into a form where each output is computed exactly once, **in the same operation order as the serial code**. Then contiguous chunking preserves that order per output.

`thread_pool::parallel_for` uses contiguous, deterministic chunking precisely so element-wise callers stay bit-identical across thread counts. (`parallel_reduce` does *not* — chunked summation changes associativity — so it is deterministic per thread count but not serial-exact; don't use it where you need `==` to serial.)

---

## 4. Grain: keep small work serial

Each kernel picks a grain so `parallel_for` stays serial until the work amortizes the hand-off, targeting ~64K flops per chunk: `grain = max(1, 65536 / work_per_unit)`, where `work_per_unit` is the inner-loop length (dot length for GEMV, `n-k` for the LU trailing row, average row nnz for the sparse solve). Consequence for tests: a matrix has to be genuinely large (and, for level scheduling, have wide levels) before the parallel path *splits* — see §6.

---

## 5. The pitfalls (mostly from reusable cached state)

The dense batches (2a) were nearly pitfall-free. The sparse pilot — which caches a reusable schedule inside a factorization object — drew a chain of review findings that generalize to **any algorithm that caches derived state**:

### 5a. Cache structure, not values (value-agnostic derived state)
The first schedule copied `L`'s numeric values. But a same-pattern re-factorization (the transient / mp-spice path: same sparsity, new numbers, in place) then left the cache holding stale values. **Fix:** the schedule stores only *positions* into `L`'s arrays and reads the numbers from `L` at solve time. Derived state that caches *structure* survives a value refresh; state that caches *values* does not.

### 5b. Bind cached state to its source; don't identify it with a weak key
The next attempt guarded the cache with `L.nnz()` as a staleness key. **`nnz` is not a pattern identity** — two matrices can share dimension and nonzero count but differ structurally, so the cached positions could silently address the wrong matrix. Chasing a run-time "has the source changed?" key is a losing game (a full pattern compare is as expensive as rebuilding). **Fix:** build the derived state *at the authoritative moment* (factorization time, when the factor is first produced) and **bind it to that object** so it can never refer to a different source.

### 5c. Enforce invariants; don't just document them
Building at factor time still left `L` and the schedule as public, independently-mutable members — a caller could replace one without the other. Documentation ("treat as immutable") is not enforcement. **Fix:** make the coupled state **private and read-only**, installed only *together* through a single setter (`set_factor`) that rebuilds the schedule, with a read-only accessor (`factor()`) for reads. The two can no longer drift out of sync, so no run-time key is needed. (Blast radius for privatizing a long-public member was one line — check, but usually small.)

### 5d. Validate before you mutate; be strongly exception-safe
The setter accepted a factor whose dimensions didn't match the analysis, and it replaced the factor *before* building the schedule. **Two bugs:** (i) a dimension mismatch would, in release builds, index a smaller RHS with the larger factor's rows → out of bounds; (ii) if schedule construction threw, the object was left with a **new factor and a stale schedule**. **Fix:** validate dimensions first (throw), then **build into a local and commit both members with `noexcept` moves** — so a throw leaves the object unchanged (strong exception guarantee).

### 5e. Asserts vanish in release — validate at the public boundary
`assert` is right for an internal primitive whose caller already guarantees the contract (e.g. the low-level solve, whose only entry point validates sizes with a `throw`). It is *not* a substitute for validating **untrusted input at a public API boundary** — those checks must be `throw`s that survive `NDEBUG`.

### 5f. Unsigned-index arithmetic underflows (dense batch)
Deriving the work range as `n - offset` on unsigned `size_type` wraps to a huge count when `offset > n` — where the original serial loop simply did nothing. `parallel_for` then iterates out of bounds. **Fix:** guard `if (offset >= n) return;` *before* computing the range. Current callers may never pass an out-of-range offset, but a parallelized primitive is often more public than the loop it replaced.

---

## 6. Testing recipe

The library kernels use the env-sized singleton pool, and **no CI lane currently sets `MTL5_NUM_THREADS`** (tracked in [#299](https://github.com/stillwater-sc/mtl5/issues/299)), so a threaded test must arrange its own threading and is the only thing exercising these paths today.

- **Force a multithreaded pool in-process.** Set `MTL5_NUM_THREADS` in a namespace-scope initializer *before the pool's first (lazy) use*; the whole test binary then runs threaded. (Each `test_*.cpp` is its own executable, so this is local.)
- **Assert bit-identity (`==`) against a serial reference.** Either a faithful in-test serial reimplementation (dense factorizations) or the existing serial routine the parallel one must match (`dense_lower_solve`). Exact equality catches races a tolerance check would hide.
- **Include a case that actually *splits*.** Because of the grain (§4), pick sizes large enough that `parallel_for` partitions — for level scheduling, a structure with a **wide level** (e.g. an arrow matrix: one level of `n−1` independent rows). Guard with `if (pool.size() < 2) WARN(...)` so single-core runners don't silently pass a vacuous test.
- **Run it under ThreadSanitizer** (`-DMTL5_SANITIZE=thread`). This is the actual race proof; the splitting case is what gives TSan something to find.
- **Boundary cases:** `1×1`, empty (`0×0` / `n=0`), the fully-sequential structure (dense triangular = one item per level), the maximally-parallel structure (diagonal = all level 0), and out-of-range offsets (§5f).

---

## 7. Checklist for the next algorithm

1. Can it be phrased as pattern 2a (independent outputs, each written once)? If so, partition the outputs, done — bit-identical and race-free by construction.
2. If outputs depend on each other, is there exploitable level/wavefront parallelism (2b)? Build a deterministic schedule; parallelize within a level.
3. Does parallelizing change the accumulation order? If yes, transform to a one-write-per-output form that **preserves the serial order** (§3), or accept non-bit-identical and document it.
4. Pick a grain so small inputs stay serial (§4).
5. If you cache reusable derived state: cache **structure not values** (5a); **bind it to its source** at the authoritative moment, no weak keys (5b); make the coupled state **private, set atomically** through a validating, **exception-safe** setter (5c–5d).
6. Validate untrusted input at public boundaries with `throw`, not `assert` (5e); guard unsigned index arithmetic (5f).
7. Test: env-thread the pool, `==` vs serial, a splitting case, TSan, boundaries (§6).

---

## References

- Thread pool + kernel rollout: [#221](https://github.com/stillwater-sc/mtl5/issues/221) (batches [#239–#243]), extended in [#297](https://github.com/stillwater-sc/mtl5/issues/297) (batches [#298], [#300], [#301]).
- Threaded CI coverage gap: [#299](https://github.com/stillwater-sc/mtl5/issues/299).
- Design context: [`on-node-threading.md`](../algorithms/on-node-threading.md), [`multicore-scaling-investigation.md`](../performance/multicore-scaling-investigation.md).
