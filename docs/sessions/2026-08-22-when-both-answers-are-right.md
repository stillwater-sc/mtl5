# Session: when both answers are right, no test can choose

**Date**: 2026-08-22
**Duration**: Full day session
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

Build the VNNI micro-kernel #468 had described and deliberately not written:
`vpdpbusd` consumes four k-values per instruction, so it needs a
quad-interleaved pack layout and a micro-kernel whose left operand is a
broadcast quad. Two PRs merged: #469 (the kernel) and #470 (the four-machine
correction to what #469 concluded).

## The through-line

The quad kernel and the widen-on-load kernel compute **bit-identical results** —
integer addition is associative, so a different summation grouping is
unobservable in the answer. That property was the point of the design and it is
also what made a real defect invisible: with the kernel selected by element type
rather than by an explicit argument, `mult`'s widening path was silently rerouted
through the quad kernel, and 179 passing tests said nothing at all. What said
something was the benchmark, where the two int8 arms began reporting the same
number to three digits.

The same shape recurred all day, in the numbers rather than the code:

| What looked settled | What checking it showed |
|---|---|
| Green suite means the dispatch is right | Both kernels agree exactly, so no *result* can distinguish them. Only a structural test can, and only the benchmark noticed |
| The quad kernel is worth ~1.25× | True of the Xeon and nothing else. Four machines: **1.27×–3.64×**, and the machine it was generalised from is the low outlier by ~3× |
| `native ÷ emulated` = the instruction's contribution | A **raw pairing ratio** — the pairings differ in signedness and decomposition path too. Netted: Zen 4 1.74–2.02×, not 2.23× |
| A VNNI purchase buys "something we already have" | Backwards. Only the machines *with* the instruction beat fp32 by a margin worth having: 1.01×/1.09× decomposed — beside it, not below — against 2.20×/3.29× native |
| An honest instruction measurement needs AVX10.2 hardware | **Partial support is the control.** Native and emulated arms already run in one binary on one machine |

The habit the previous session named — *the issue's remedy is a hypothesis* —
extended here to **a number from one machine**. Twice, and both times the
correction came from outside: once from three more machines, once from review.

## Work Completed

### #469 — the kernel, and the bug the tests could not see

`batch::quad_dot_broadcast_accumulate`, `gemm_quad_pack.hpp`,
`gemm_quad_microkernel.hpp`, `gemm_blocked<TC, TA, TB, Kern>`, `mult_quad`, two
benchmark arms. `TA`/`TB` split because `(u8,i8)` is VNNI's native shape and the
operands must differ; the five loops, cache blocks, thread grid, cooperative
B-pack and exception handling are shared, since none depend on which
micro-kernel runs.

The broadcast's lane assignment is checked **exactly, per lane**. The dot form's
contract is only "the total is right", which is enough for a reduction; for a
GEMM a lane **is** a column of C, so a permutation there is a wrong answer rather
than a reordering.

Kernel selection was first inferred from the element types. `(i8,i8)` is valid
input to both kernels, so that rerouted `mult`'s widening path — invisibly. It is
now an explicit `gemm_kernel` argument defaulting to `widening`, pinned by a
structural test comparing function addresses, since a result-based test provably
cannot tell the two apart.

That defect also motivated `verify_gemm_int`: `measure` only timed the lambda, so
no GEMM arm had ever *read* its output. Every integer arm now samples C against a
64-bit reference outside the timed region and fails the run. Verified by
tampering with the reference rather than assuming it fires.

### #469 — support is per pairing, and the ISAs disagree about which

x86 implements `u8×i8` first (`vpdpbusd`) and the symmetric pairings only at
AVX10.2; ARM implements `i8×i8`/`u8×u8` first (`SDOT`/`UDOT`) and `u8×i8` only
with I8MM. Read from Highway's own sources, then confirmed on hardware by the
two sidecars:

```text
Ryzen  AVX3_DL   u8*i8 native     i8*i8 emulated   u8*u8 emulated
Jetson NEON      u8*i8 emulated   i8*i8 native     u8*u8 native
```

`has_native_quad_dot_v<NA, NB>` replaces the single flag; `has_native_quad_dot`
keeps its original *any-pairing* meaning so no committed sidecar changes meaning.
`PARTIAL` joins the guard's states and is what every machine measured so far
reports.

### #470 — four machines, and what one could not say

GEMM at n=1024, best-of-iteration, single-threaded:

| machine | native | kernel | raw nat/emul | best int8 ÷ fp32 |
|---|---|---|---|---|
| Xeon E5-2420 v2 (SSE4) | none | 1.27× | — | 1.09× |
| i7-12700K (AVX2) | none | **2.70×** | — | 1.01× |
| Ryzen 9 8945HS (AVX3_DL) | `u8×i8` | 2.19× | 2.23× | **3.29×** |
| Jetson Orin Nano (NEON) | `i8×i8` | **3.64×** | 2.40× | **2.20×** |

The kernel effect rises with n on every machine — that part generalised — but its
magnitude is a property of the machine. The instruction, netted against the shape
control measured where both pairings are emulated, is **1.74–2.02×** on Zen 4
against ~1.2× for the same instruction on a dot: a dot is bandwidth-bound, a GEMM
is not.

## The parts that went wrong

- **I generalised from one machine, in a document about not doing that.** "~1.25×
  is the kernel result" was written from the only machine I could run, in the
  same file that records §6's 1.42×→1.17× correction for exactly that error. The
  Xeon turned out to be the low outlier by a factor of three.
- **I committed a two-variable ratio while writing the warning against it.** The
  column labelled `instruction` compared native `u8×i8` against emulated
  `i8×i8` — moving signedness and decomposition path alongside nativeness — two
  sections below a passage explaining why that is invalid. It was wrong in *both*
  directions at once: too high for Zen 4, too low for the Jetson. CodeRabbit
  caught it, not me. **Both slips were numbers that confirmed what I expected**;
  the scepticism went to the numbers I doubted.
- **I told the reader not to buy the hardware.** The plan said a VNNI purchase
  would be "buying something we already have". The data says only the machines
  with the instruction beat fp32 by a margin worth having — the decomposed parts
  sit beside it at 1.01× and 1.09×, the native ones at 2.20× and 3.29×. Wrong in
  the direction that costs money by *not* spending it.
- **I claimed a control did not exist that already did.** Partial support means
  native and emulated arms run in the same binary on the same machine. I argued
  at length that this needed AVX10.2 hardware.
- **Two monitors reported nothing and I nearly read the silence as success.**
  One used a `gh` flag this version does not support and failed on every
  iteration; one simply ran out of loop budget. Both exited 0 with no output. The
  status came from querying directly.
- **I broke CI with a section sign.** `§` in two comments of a new test file;
  sources here are ASCII-only and the gate is a hard failure.

## Tooling notes

- `gh pr edit --body-file` still fails against this repo with the projects-classic
  GraphQL deprecation error, and **it aborts the edit** — the body was unchanged
  and would have been reported as updated. `gh api repos/{o}/{r}/pulls/{n}
  --method PATCH -F body=@file` works.
- `gh pr checks --json` is unsupported in the installed `gh`; `gh pr view --json
  statusCheckRollup` is the form that works. A monitor built on the former fails
  silently, which is indistinguishable from "still running".
- Highway's `HWY_NATIVE_*_SUMOFMULQUADACCUMULATE` toggles are **not** usable as
  an external source of truth: on an SSE4 build, where all three pairings are
  emulated, both macros are still *defined*, because `generic_ops-inl.h` defines
  them when it supplies the fallback. Checked rather than assumed.
- `static_assert` inside a discarded `if constexpr` branch still fires outside a
  template. Target-conditional assertions need `#if`.
- Cross-target gate verification without the hardware: compile assertions against
  expectations read from the library's source, one arm per `-march`, **with a
  negative control per arm** to prove the harness rejects wrong expectations.
  Six x86 targets checked this way; ARM could not be, for want of an aarch64
  sysroot.

## The CHANGELOG gap this session closed

Nothing between **#457 and #468** had a CHANGELOG entry, and the run 2026-08-17
to 2026-08-21 has no session log at all. The last log is 2026-08-13; 5.10.0
shipped on 08-17, and twelve PRs landed after it without either.

The shape of the gap says what caused it. Those five days are one continuous
push — the `#451` integer-lane epic, phases 0 through 4, plus the `mc`
measurement thread (#453 falsified, #457/#458/#459) — and an epic has no natural
stopping point at which a wrapup feels due. The two prior gaps in this directory
close the same way: a session log appears when a *thread* ends, not when a day
does, and the previous ones were written as "fill the CHANGELOG gaps" commits
after the fact. The release on 08-17 also plausibly absorbed the attention that
would otherwise have gone to the changelog, since it is the one moment the file
is definitely being read.

Entries for all nine PRs are written here from their commit messages, which in
this repository carry the reasoning and the measurements — that is why the gap
was recoverable at all. **What is not recoverable is the session log**: the
commit messages record what each PR concluded, not what was tried and abandoned
between them, and no session log for 08-17..08-21 is reconstructed here for that
reason. Writing one from the merge history would be fabricating the interesting
half.

## Issues and PRs

- **Merged**: #469 (`16ebca8`) — the quad micro-kernel, 23 files, +2677/−103;
  #470 (`fd80d7d`) — the four-machine correction, docs only
- **Review**: 13 CodeRabbit findings across the two PRs, all verified against the
  code and fixed. Three were substantive: the pack UB, the unreachable
  `mult_quad` fallback, and the two-variable ratio
- **Validation**: 180/180 (was 176) on gcc+Highway, clang+Highway and the scalar
  fallback, plus UBSan over the integer kernels on both backends. Benchmarks on
  four machines, every arm passing `verify_gemm_int`
- **Backfilled**: CHANGELOG entries for #457, #459, #460, #463–#468, written from
  their commit messages

## Lessons

- **When two implementations agree bit-for-bit, correctness tests are blind to
  which one ran.** That is a property to design *for* — it is what makes the
  integer nest's reordering safe — and it removes an entire class of test. What
  remains is a structural assertion and a benchmark control, and the control only
  works if both arms are compiled into one binary.
- **A number from one machine is a hypothesis.** The kernel ratio, the
  instruction's share, and the hardware recommendation were each wrong from a
  single machine, and each in a different direction. The programme has now made
  this error three times: the dot's 1.42×, the Xeon's 1.25×, and the raw pairing
  ratio.
- **Scepticism is easiest to skip on the numbers you like.** Both ratios I got
  wrong were ones that flattered the work. The discipline is cheap to apply and I
  applied it selectively.
- **A capability exposed in pieces should not be modelled as a bit.** The single
  `has_native_quad_dot` mislabelled at least one arm on every machine measured,
  and by name-matching would have reported a 36× "machine gap" between Zen 4 and
  the Jetson that is almost entirely an artifact of which pairing each implements.
  This is §7's lesson one level down.
- **Silence from a watcher is not a result.** Two monitors produced no output for
  two different reasons, and neither meant "nothing to report".
- **Sanitizers do not catch a pointer that is merely formed.** ASan sees no bad
  load and UBSan checks wraparound, not escape from the object. The pack UB was
  found by reading, with clean sanitizer runs on both sides of the fix.
- **A changelog gap is recoverable; a session-log gap is not.** Nine PRs went
  five days without either, and the entries could be written afterwards *only*
  because this repository's commit messages carry the reasoning and the numbers.
  What no commit records is the path not taken — the measurement that came out
  flat, the design abandoned at lunchtime — which is most of what a session log
  is for. The wrapup wants a trigger that fires on a **calendar day**, not on the
  end of a thread, because an epic supplies no natural end.
