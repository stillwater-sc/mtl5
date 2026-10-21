# Session: the panel the partition padded

**Date**: 2026-10-20 to 2026-10-21
**Duration**: One long working session
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Three things, in order:
- finish the 5.19.0 release on the Python side;
- open the follow-up issue the release exposed there;
- resolve #458: the shipped `mc` sits in a trough of the multi-threaded response curve.

**Merged:** mtl5 #626; mtl5-python #106.

**Released:** MTL5 v5.19.0; mtl5-python v5.19.0, on PyPI.

**Closed:** #458.

**Opened:** mtl5-python #107.

## The through-line

#458 had a well-measured symptom and a suggested fix. On the Xeon the 2048³ T=6
`mc` curve was W-shaped, with a 13% spread at T=6 against 3.6% at T=1. The issue
thread proposed making the ic block count a multiple of the thread count. The
cause was neither cache residency nor block-count balance. It was the register
tile.

The serial path rounds `mc` down to whole `mr`-row micro-panels. The threaded path
didn't: it fixed the block count and kept whatever `mc` fell out. The micro-kernel
computes whole tiles, so a 32-row block at `mr = 6` costs 6 panels, not 5.33. Every
threaded block carried a padded panel. T=1 was flat because T=1 always ran whole
panels. The fix in the thread would have optimised a quantity that, fitted against
the measured curve, correlates with throughput at 0.00 to +0.23. The busiest
thread's panel count correlates at +0.74 to +0.83.

## Work completed

### The 5.19.0 release, finished

- MTL5 v5.19.0 tagged on the release merge; the release workflow passed.
- mtl5-python #106 bumped to MTL5 v5.19.0 and Universal v5.10.0.
- **Merging the bump released nothing.** mtl5-python's semantic-release only bumps the
  patch (`minor_tags` is empty) and a `chore(deps)` commit warrants no release at
  all. Minor releases there are cut by hand: tag, then `gh release create`, whose
  `release: published` event runs `wheels.yml`. A simulation showed the risk of
  leaving it: with no v5.19.0 tag, the next `fix:` would have released **5.18.2** and
  rewritten `pyproject.toml`'s 5.19.0. It was cut: 12 wheels and the sdist on PyPI.
- **mtl5-python #107**: `bicg` and `qmr` still refuse non-symmetric `ilu0`, `ilut` and
  `block_diagonal`, all of which now have real adjoints (`ilut` since 5.19.0).

### #626: choosing the threaded ic step by micro-panels

**Diagnosis.** A small planner probe printed each plan's `mc_used`, block count and
critical-path panels. These were fitted to the committed Xeon sweep
(`benchmarks/data/xeon-e5-2420-sweep`):

| cost model | kc = 512 | kc = 256 |
|---|---|---|
| block-count balance | +0.23 | 0.00 |
| critical rows | +0.21 | +0.02 |
| critical micro-panels | +0.83 | +0.74 |

**Rule.** The block-count result stays as the incumbent. A block size in
`[ceil(mc_max/2), mc_max]` replaces it only on strictly fewer critical panels:
- ties keep the old plan, so #408's tuned plans don't move;
- the window exists because a panel count alone always prefers one-panel blocks,
  and each block is another pass over packed B. `mc = 6` measured ×0.61–0.87 on the
  i7.

`detail::critical_panels` is a closed form, checked exhaustively against a
simulation in the tests.

**Measurement.** A new `legacymc` arm built the old rule from the same tree, so each
machine ran an interleaved A/B in one session:

| machine | changed plans | faster | range | slower |
|---|---|---|---|---|
| Xeon E5-2420 v2 | 16 | 16 | 7.5–16.2% | 0 |
| i7-12700K | 13 | 12 (13th within noise) | 3.1–9.2% | 0 |
| Ryzen 9 8945HS | 5 | 5 | 3.4–6.4% | 0 |

On the Xeon the panel model predicted each gain at r = +0.91, with measured gains
×1.010 of predicted. The issue's own point went 45.3 → 52.6 GF/s.

**Three machines, three ways of reaching them.** The i7 is this machine. The Xeon ran
over SSH, in `tmux`, after a key copy; its copy of the results was checked
byte-identical before deletion and its clone returned to `main`. The Zen 4 is a
Windows laptop whose WSL can reach here but not the other way. A reverse tunnel was
offered and judged too involved. A one-command profile was pushed to the branch
instead, the user ran it, and the results came back by `scp`. The sidecar carried
enough provenance to accept them without being there: commit, clean tree, `build_isa`,
pinning and load.

**Review.** CodeRabbit raised three points, all in the A/B runner's guards and the doc's
example command. Two of them were the same finding: the first fix for "an empty CSV
compares equal to another empty CSV" counted lines, and the second review pointed
out that a blank line is a line. The final check counts complete records.

## The parts that went wrong

- **The first planner probe was built without Highway.** It derived `nr = 2`,
  `kc = 1024`, `mc = 16`, which are not the shipped plans, and the first analysis ran on
  them. The first i7 sweep also forced `kc = 1024`. Both were discarded once
  `mc_used` was printed next to the configured values and didn't match the
  committed sweep.
- **The Zen 4 prediction assumed the i7's blocking.** The doc said that if Zen 4
  derived `mc_max = 64`, the same plans would change. It derived 128 (`kc = 128`), and
  only five plans changed. The hedge was in the text, but the number was a guess
  stated next to measured ones.
- **A fix shaped to the comment, not the failure.** The first runner guard answered
  "header-only CSV" literally. The underlying failure was "no complete record", and
  answering that the first time would have saved a review round.

## Tooling notes

- **WSL2 benchmark noise is thread-count dependent.** Up to T=4 the Zen 4 controls
  agreed within 1.7%. At T=6 and 8, two arms running the identical plan differed
  by −9% to +12%, with round spread up to 23%. `taskset` pins virtual CPUs and
  Windows places them. A/B interleaving keeps the comparison fair, but at high T on
  WSL only large effects resolve.
- **A profile is the cheapest remote protocol.** Pinning, threads, shapes, arms and
  output directory are fixed in a script on the branch, and the sidecar records the
  rest. Someone else can run it and nothing is lost in transcription.
- **`gh pr checks` has no `--json` in this `gh`;** `awk -F'\t' '{print $2}'` over its
  output gives the state column.
- **mtl5-python minor releases are human-cut** (now in memory): merge the bump, then
  tag and `gh release create` before anything else merges.

## Lessons

- **Fit the candidate cost models before choosing a fix.** The issue's proposal was
  plausible and wrong, and a ten-line correlation against data already in the repo
  separated it from the real cause in minutes.
- **Print what the code did, not what it was configured with.** `mc_used` beside `mc`
  exposed both the wrong-build probe and the runner calling a real A/B a consistency
  pair.
- **Keep the old rule buildable.** A compile-time switch for the legacy behaviour,
  plus an arm that sets it, made every measurement a same-tree, same-session A/B.
  That is what let a WSL laptop with VM noise still give a clean answer.

## Follow-ups

- mtl5-python #107: gate `bicg`/`qmr` on `has_exact_adjoint` and decide the MTL5
  floor.
- The Xeon's `mc = 171` point measures about 5% below the panel model. Its A block
  is far larger than L2, so the panel count probably needs a cache term for large
  `mc`.
- The `ondemand` governor on the Xeon and the WSL placement on the Zen 4 are both
  recorded but not controlled. A fixed-clock rerun would tighten the small-gain
  points, though none of the conclusions depend on them.
