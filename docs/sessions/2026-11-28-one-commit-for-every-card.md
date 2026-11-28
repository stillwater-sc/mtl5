# Session: one commit for every card

**Date**: 2026-11-27 to 2026-11-28 (overnight; the previous log is `2026-11-27-the-unnormalized-vector.md`)
**Duration**: Three PRs, a benchmark campaign on six machines, and the 5.26.0 release
**Participants**: Theodore Omtzigt (Ravenwater), Claude Code

## Objective

- **#964:** the native SOR sweep on AArch64 ran at one of two speeds from commit to commit. Paid CodeRabbit reviews were allowed for this one.
- **#885:** the measurements left over from #737 and #778, now that every lab machine was on.
- **The 5.26.0 release,** so that mtl5-python gets `randsym_indefinite` (#970).

| PR | Issue | What |
|---|---|---|
| #979 | #964 | The CSR SOR and Gauss-Seidel sweeps sum around the diagonal, not past a test |
| #980 | #885 | A card page per class; medium is the standard card |
| #981 | #885 | The campaign at `b2d022726b3d`: ten cards and the #778 check |

## #964: an alignment, not a code change

- **The same instructions in both builds.** The slow and fast `mtl5_bench` binaries had the same SOR sweep, at different addresses. The inner row loop is 40 bytes, with a branch on every column index. It was slow when it straddled a 64-byte boundary. `-falign-functions=64` alone put `main` in the slow mode.
- **The fix is the loop's shape,** not an alignment directive. The sweep sums the off-diagonal terms in two loops around the diagonal's position, which is found at construction. Under all five alignment flags it runs at 2.96-2.98 ms.
- **Bit for bit, until FMA.** The first test checked the old and new forms for bit equality, and failed on CI's Ubuntu 26.04 GCC job. GCC's default GNU dialect contracts `sigma += a * b` to an FMA in the new loops, but not in the old branchy one. The failure reproduced locally with `-march=x86-64-v3`.
  - **The claim was corrected:** the results are bit for bit the same without contraction, and differ by rounding with it.
  - **The test checks bit equality where `__FP_FAST_FMA` is not defined,** and is within 64 ulps where it is.
  - CodeRabbit pointed out that the first tolerance, 64·ε·|b|, is not 64 ulps. It is a ulp distance now, with its own test.
- **The gate:** `smoother/gauss_seidel/f64` and `smoother/sor1.5/f64` joined the instruction-count kernels, which now number 17.

## #885: the campaign

**Decisions:**
- Re-run all five #737 machines at one commit.
- `medium` is the standard card.
- Try `--class large` on the x86 machines, and report any failure. None failed: the Xeon's 15 GB was enough.

**Per-class pages first (#980).** The publish script kept one page per machine, showing the latest session, so a large session would have replaced the medium card. It now writes `cards/<machine>-<class>.md` for any class but medium, and links that page after the card in the Systems table. CI's publish checks publish a `small` session, so they moved to that page.
- **One CI failure:** the macOS smoke job. A new helper used `sed '/…/{s/…/p}'`, which GNU sed accepts and BSD sed rejects. It is awk now.
- **One CodeRabbit finding:** the README's `git add` example staged only the medium page.

**The campaign** ran from the i7-12700K:
- **Five remote hosts in parallel,** each running its sessions in sequence.
- **The i7's own sessions locally,** without a reboot, because other sessions were running on it.
- **The Ryzen 7 8845HS** came along for #778.

What it found:
- **Windows Update ran during three Windows sessions.** The sidecars' `background_activity` showed it, and the Ryzen 9 pair's repeatability showed it too: 224 of 816 entries within ±5%, with a median ratio of 1.113.
  - **The affected pairs were re-run,** five sessions, each after a cold boot.
  - **The Ryzen 9's three reruns came out clean.** On the Ryzen 7 the update ran again in the first rerun.
  - **The Ryzen 7's pair** is two clean sessions, from two different runs.
  - **The rule:** check `background_activity` before publishing a Windows session.
- **#778 holds on the Xeon (1.9%) and the Ryzen 7 (1.000), not on the Ryzen 9 for f32.** In some runs the Ryzen 9's f32 kernels at W = 8 slow by 12-16%, while W = 6 stays put. Other runs of the same binary in the same session are at 1.000. The Ryzen 7 shares the Zen 4 entry and was at 1.000 in every run.
  - **Why no table entry fixes it:** W = 6 would cost the Ryzen 7 about 10%, and CPUID can't tell the parts apart.
  - **The maintainer's decision:** keep the entry, document the finding in `interleave-width.md`, and close #885.
- **Repeatability between each published pair** was 733-843 of 816-858 entries within ±5%. Each card's note gives its own figure.

## Process notes

- **CodeRabbit:**
  - **#979:** a paid review, $0.25.
  - **#980:** two paid reviews, $1.00 and $0.25.
  - **#981:** offered at $78.50 because of its 27 session directories, so it waited for the included review. The maintainer merged it.
- **A stale watcher event.** After a push, `pr-watch.sh` reported a CI failure for the previous head. Check the head sha in an event before acting on it.
- **The measured commit is not recorded in the #885 issue** (issues aren't commented on automatically). It is recorded in #981 and on the systems page.
