# Session: every diagnosis was wrong, and the measurement said so

**Date**: 2026-08-05
**Duration**: Full day session
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Clear the remaining preconditioner and sparse backlog, then refresh the
benchmark data that #382 had made stale.

The refresh was meant to be housekeeping. It found a regression that had merged
the day before, and the rest of the day followed from that.

## The through-line

Every up-front diagnosis offered today was wrong. Not vague, not incomplete —
**wrong**, and in each case a measurement that took minutes said so:

| Claimed | Actually |
|---|---|
| `pc::ssor` needs a reimplementation with wide blast radius (#398, as filed) | One line. And it *improves* convergence |
| #382 regressed scaling via cache pressure from the larger tile (#408, as filed) | `mc` was coupled to `mr`; nothing to do with pressure |
| MKL's small-N anomaly is a cold P-core at 800 MHz | Refuted: 60.8 GFLOP/s cold vs 61.0 warm |
| "164/164 passing" after a change | The build had **failed**; ctest passed a stale binary |
| A benchmark comparison showing +5% at T=1 | Cross-session drift. That path *cannot* change |

The habit that saved each one was the same: **run the control**. Revert only the
header and see the test fail. Force the variable and see the number move. Build
both binaries and interleave them. None of it is sophisticated; it is just not
skipping the step.

## Work Completed

### The `pc::ssor` arc — a mis-filed issue, then the duplication behind it

The issue was filed claiming `ssor` ran "two SOR sweeps, not the classical
factored operator", needing a reimplementation. Wrong on the second half. The
sweeps **are** the factored operator — provided they start from `x = 0`:

```text
forward from 0:  F x1 = b                     F = D/w + L,  B = D/w + U
backward:        B x2 = b + (B - A) x1
so               x2 = B^-1 ((B - A) + F) F^-1 b,   (B - A) + F = D (2-w)/w
giving           x2 = [ w/(2-w) F D^-1 B ]^-1 b     <- Sheldon (1955), exactly
```

MTL5 initialised to `x = b`, adding a `G_B·G_F` term that is not symmetric even
when `A` and `M` both are. One line. The adjoint identity violation went
**1.1e-01 → 3.4e-16**, and every solver on symmetric `A` *improved* (cg 16→13,
gmres 13→12). The issue was rewritten before implementing (#398, #404).

Then the follow-up: `pc::ssor` was documented as a thin adapter over
`smoother::symmetric_sor` and was in fact a 120-line reimplementation of the same
sweeps. Making it the adapter it claimed to be deleted 129 lines, inherited the
`Accumulator` parameter for free — mixed-precision *preconditioning* had been
impossible while mixed-precision *smoothing* worked, with nothing saying so — and
made the #398 bug unrepresentable, since zeroing `x` became the only line
separating preconditioner from smoother (#405, #406).

### SpGEMM, and a correctness property no natural test would catch

`compressed2D * compressed2D` fell through to the dense `operator*`, materialising
an `n × n` intermediate *at the fine-grid size* — so the Galerkin triple product
`R·A·P`, the one product multigrid most needs sparse, was the one that was not.
Gustavson row-by-row over the existing sparse accumulator (#402, #407).

The part worth remembering is the sorting. `compressed2D::operator()` uses
`std::lower_bound`, so ascending column indices are a **correctness invariant**,
and the accumulator returns them in scatter order. Measured with the sort removed:

| checked how | result |
|---|---|
| `nnz` | correct |
| row sums from raw CSR | correct to 4.4e-16 |
| sparse matvec `A*x` | correct **to rounding** |
| element access `C(i,j)` | **wrong by 3.2** |

The matvec sums the same set of products whatever order the row is in, so it
still computes the right sum — correct *to rounding*, not bit-identical, since
floating-point addition is not associative. Either way it cannot detect the
corruption, which is the point: every natural test for a matrix product passes
on a corrupt matrix. So the tests assert index structure directly rather than
trusting a residual.

### The benchmark refresh that found a regression (#409)

Re-ran all five backends in one session — the headline is a *ratio* to OpenBLAS,
and a ratio is only defensible when both sides come from the same session. That
also supplied the attribution: the three untouched backends moved within noise
while `native-fast` moved +17.4%.

GEMM at N=1025 went 59.44 → 69.81 GFLOP/s and the #82 gate 82.3% → **97.4%** —
a change in standing, not a number: the single-core kernel is at parity with
OpenBLAS, so the gap #82 exists to close has essentially closed for this
operation on this machine.

Three findings were **retracted in place** rather than silently overwritten,
including one whose replacement explanation was tested and refuted. The MKL
small-N figure (17.5 GFLOP/s) did not reproduce — four independent measurements
gave 60.8–71.4, version unchanged since before both dates, large-N identical.
The obvious cause was a cold P-core; direct measurement killed it. Recorded as
**undetermined** rather than given a plausible-sounding story.

### The regression, root-caused (#408 → #410)

`#382`'s +22% single-thread win decayed with thread count and went **negative** at
N=1024/T=8. The filed hypothesis was cache pressure. The actual cause:

`mc = round_down((L2/2)/(kc·8), mr)` — the L2 budget gives exactly **64**, which
survives `mr=4` but drops to **60** under `mr=6`. Since ic-blocks are handed out
round-robin, that decides the critical path:

| N | mc=64 | mc=60 |
|---|---|---|
| 1024 | nib=16 → 128 rows, **ideal** | nib=18 → 180 rows, **1.41× ideal** |
| 2048 | nib=32 → 256 rows, **ideal** | nib=35 → 300 rows, 1.17× ideal |

A **cache** quantity had been coupled to the **register tile**. `mc` now derives
from the cache alone, and `detail::balanced_mc` chooses the block size at runtime
against `m` and the thread count. Measured interleaved, three reps, non-overlapping
ranges: **+27.3%** at N=1024 and **+10.2%** at N=2048 on 8 cores, T=1 unchanged.

Re-measured and corrected the page afterward (#411), including what the fix did
**not** achieve: efficiency is 85.5%, up from 79.2% but short of the pre-#382
90.5%, and `native-fast` is still last of four scalers. Softened, not withdrawn.

## What went wrong

**I introduced a silent-wrongness bug of exactly the kind this session kept
finding.** A draft of the #410 fix had `serial_nest` stepping `ic += MC` while
blocks were sized `min(MC_eff, m - ic)` — dropping `MC - MC_eff` rows of C per
block. **The entire existing GEMM suite passed it**, because every case used an
`m` fitting inside a single ic-block, where the step never matters. Caught only
by asking why a T=1 number moved when it provably could not.

**I reported a false green.** After that same patch I said "164/164 passing". The
build had failed on a `static_assert`; ctest ran the stale binary. The signal was
a `grep -c` returning 1, which I under-weighted. Now checking the build exit code,
not grep output.

**Three watchers looked armed and could not fire.** Two CI monitors used `jq`,
which is not installed here. Two process waiters used `pgrep -f <pattern>` where
the pattern appeared in the waiter's own command line — the self-match trap,
already in my notes from a previous session, hit twice more.

## Merged

`#397` `#399` `#400` `#403` `#404` `#406` `#407` `#409` `#410` `#411`

Closed: `#386` `#392` `#393` `#394` `#398` `#401` `#402` `#405` `#408`

## Open

- **#412** — `native-fast` loses ~4% of parallel speedup at **two** threads,
  growing to 7–12% at eight, with the ic partition now *provably* even
  (ratio 1.000 at every shape). The T=2 result rules out contention stories
  before anyone spends time on one; 1.92× at T=2 implies a 4.2% serial fraction
  that would predict 6.19× at T=8, but the measured 6.84× is better — so it is
  not a fixed serial region. Phase-level timing inside the parallel region is
  the measurement that would settle it, and does not exist yet.
- Two unattributed benchmark movements, flagged rather than explained: MKL's
  small-N GEMM and `native-fast` `axpy` (+37%, no commit in the window touches
  it).

## Carried forward

**A test that appears to cover something can be structurally incapable of it.**
Four instances today, one of them mine:

- `pc::identity` for `bicg`/`qmr` — the single preconditioner for which a doubled
  or wrongly-adjointed `M⁻¹` is indistinguishable from a correct one
- `test_multigrid.cpp` supplying `multigrid.hpp`'s own missing include at line 7
- Every GEMM test using an `m` that fits one ic-block, hiding my step bug
- Every natural SpGEMM check — nnz, row sums, `A*x` — passing on unsorted output

The defence that worked each time was the negative control: revert *only* the
fix, keep the tests, and watch them fail. If they do not, the test was never
testing what its name says.
