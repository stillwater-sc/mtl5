# Session: a word that could repeat

**Date**: 2026-10-24 to 2026-10-25
**Duration**: A two-day working session
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Resolve #636: a thread pool for callers in a real-time loop, with mutex-free
dispatch and join, a spin-then-sleep wait, and a worst case that is written
down rather than implied.

**Merged:** mtl5 #647.

**Closed:** #636.

## The through-line

A handoff without a mutex can't block on a preempted lock holder, though
`run()` still sleeps once its spin runs out and waits for the slowest worker.
It can also simply be wrong. The first design
packed a 16-bit generation and a 16-bit team size into one 32-bit word. A
comment argued that a wrapped generation was harmless: a word equal to the one a
worker last saw also has the same team, so a worker that takes it for "no
change" was not part of that region either. That holds for a worker outside the
team. It does not hold for a team member preempted between its countdown and
its next load of the word, while the caller runs 65535 smaller regions. That
worker skips its tid, and the join waits for it forever. CodeRabbit traced the
sequence, and the fix is a 64-bit word with a 48-bit generation.

The cost question was whether an 8-byte `std::atomic::wait` is slower than a
4-byte one. libstdc++ waits on a 4-byte atomic with a futex directly, but on an
8-byte one through a shared table of proxies. The answer was measured, not
argued: with the workers asleep at 1 kHz, the dispatch median was 10.1–15.9 µs
before and 9.6–15.6 µs after.

## Work completed

### `spin_pool` (#647)

- **`include/mtl/parallel/spin_pool.hpp`** has `thread_pool`'s interface and
  contract. Dispatch is one store, a waiting worker spins then sleeps on the
  word, the join is an atomic countdown, and the busy flag and first-exception
  slot are atomics. A thread notifies only when another is asleep.
- **Selection.** `-DMTL5_POOL_SPIN` makes `parallel::pool()` return the process
  spin pool. The 17 kernel headers now call `pool()` instead of
  `thread_pool::instance()`.
- **No curiously recurring template pattern.** The plan first proposed a CRTP
  base for the shared loops. The maintainer asked why it was needed, and it
  wasn't: `parallel_for`, the reductions and the team rule became `detail::`
  function templates that take the pool, and both pool types forward to them.
- **Tests:** one TEMPLATE_TEST_CASE runs the contract, nesting, concurrency,
  exceptions, partitions, wake-after-gap, a slow worker, destruction, a stress
  test and an allocation check against three spin settings. GEMM and the
  factorizations also run rebuilt under `MTL5_POOL_SPIN`, and the shared-pool
  test checks the spin pool is shared across modules.
- **Measured** on the i7-12700K, P-cores pinned, T = 2–8, 4 rotated rounds:

  | Pool | Back to back, p50 / p99 | 1 kHz loop, p50 / p99 |
  |---|---|---|
  | `thread_pool` | 3.0–6.1 / 3.6–8.0 µs | 20–41 / 58–118 µs |
  | `spin_pool`, default spin | 0.23–0.48 / 0.25–0.57 µs | 11–16 / 61–110 µs |
  | `spin_pool`, spin outlasting the gap | 0.24–0.51 / 0.25–0.60 µs | 0.7–1.1 / 1.0–1.4 µs |

### Review fixes

CodeRabbit raised six findings, all valid, all fixed in one commit:
- the 16-bit generation wrap, above;
- a constructor that terminated, through a joinable `std::thread`, when starting
  a later worker threw; it now stops and joins the started workers and
  rethrows, through the same `shutdown()` as the destructor;
- the benchmark accepted `--regions 0` and a CSV it could not open;
- the sidecar recorded one region count for a run that uses two;
- the runner ran the arms in a fixed order, so each arm always sat in the same
  position; it now rotates the order and takes rounds in multiples of four.

## The parts that went wrong

- **The spin loop had no yield.** In the first run, the spinning pool at T = 8
  with a 1 ms gap had a dispatch median of 155 µs in every round, while the
  never-spinning variant held 20 µs. The scheduler had placed two of the pool's
  threads on one core, and the spinner held it. A yield every 64th iteration
  brought it to 16 µs, and the data was rerun from a clean commit.
- **An explanation went into the docs before it was checked.** Why the
  never-spinning variant was still fast back to back was first explained by
  guesswork. Reading libstdc++ showed that `atomic::wait` itself spins 16 times
  (12 pauses, then 4 yields) before the futex, and the docs now say that.
- **The wrap argument in the header's comment was wrong**, as described above.
  It was written as a justification and read as one, and nobody had walked a
  team member through it.

## Tooling notes

- **CodeRabbit's hourly allowance** covered the first review only; the two
  review-fix commits merged without an incremental review.
- **An 8-byte atomic wait is not a direct futex** in libstdc++ or libc++, which
  wait on a futex only for a 4-byte type (`__platform_wait_uses_type`). It works,
  and here it measured the same, but check it on any platform the latency
  matters on.

## Lessons

- **Test a wraparound argument against every party to the protocol**, not only
  the one the argument was written for. "Same word, so same team" was true and
  beside the point.
- **Rotate the arm order between rounds** in any benchmark whose arms run in
  sequence, in a multiple of the arm count, so position effects average out.
- **A spin wait needs a yield** whenever the pool can fill its cores. The
  scheduler does not promise one thread per core.

## Follow-ups

- Measure `spin_pool` on aarch64, where a pause is much shorter and the default
  4096 iterations spin for less time.
- mtl5-python and synapse adopt `mtl::parallel::pool()` and `MTL5::shared_pool`
  after the next MTL5 release.
- Earlier follow-ups stand: the Jetson and non-A78 tile measurements, bare-metal
  Zen 4, a `performance`-governor rerun of the #640 benchmark, and the
  maintainer's review of the relaxed `[586:eigen]` bound.
