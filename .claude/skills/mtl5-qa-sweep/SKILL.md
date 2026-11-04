---
name: mtl5-qa-sweep
description: Daily QA sweep of MTL5. Builds and runs the unit tests, reviews the commits since the last sweep, takes a deep pass over one subsystem in rotation (with ASan/UBSan), and files the findings as one GitHub issue labelled qa-sweep. Meant for a scheduled run; read-only on the code.
disable-model-invocation: true
allowed-tools: Read Grep Glob Agent Bash(git *) Bash(cmake *) Bash(ctest *) Bash(gh issue *) Bash(gh label *) Bash(date *) Bash(nproc) Bash(mkdir *)
---

# MTL5 daily QA sweep

You run unattended, usually from a scheduled task on a fresh clone of `main`.
Nobody is watching, so don't stop to ask: when a step fails, record the
failure as a finding and go on to the next step.

## Ground rules

- **Read-only on the repository.** Don't edit tracked files, commit, push,
  open pull requests, or comment on any existing issue or PR. Your only write
  to GitHub is the one sweep issue in step 6, and creating the `qa-sweep`
  label if it is missing.
- **Builds:** at most `-j4`, in `build-ci/` and `build-asan/` only. Scratch
  programs go under `build-qa-scratch/`. Run long builds in the background and
  wait for them; a foreground command can time out.
- Use the `mtl5-qa-reviewer` agent for the reviews in steps 3 and 4. It knows
  what counts as a finding and what evidence to give.

## 1. Where to start

```bash
date -u +%F                         # the sweep's date
git rev-parse HEAD                  # the commit being swept
gh issue list --repo stillwater-sc/mtl5 --label qa-sweep --state all \
  --limit 1 --json number,body
```

The last sweep's issue body has a line `Swept commit: <sha>`. The range for
step 3 is `<sha>..HEAD`. If there is no earlier sweep, or the sha is not in
the history, use the commits of the last 7 days (`git log --since=7.days`).

## 2. Build and run the unit tests

```bash
cmake --preset ci -DMTL5_BUILD_EXAMPLES=OFF
cmake --build build-ci -j4
ctest --test-dir build-ci -j4 --output-on-failure
```

A compile error, warning-as-error or failing test on `main` is the sweep's
first finding. Give the target, the error or the failing assertion, and the
last commit that touched the code involved.

## 3. Review the commits since the last sweep

List the range with `git log --stat <range>`. Skip commits that touch only
`docs/`, `docs-site/`, Markdown or images. Give the rest to
`mtl5-qa-reviewer`, a few related commits at a time. For each, ask it to:

- check the change for functional bugs and performance problems;
- check whether a test reaches each new or changed path, and whether a test
  added with a fix would have failed before it.

If nothing but documentation changed, say so in the issue and go on.

## 4. Deep pass over one area

Pick today's area by the day of the year:

```bash
echo $(( 10#$(date -u +%j) % 11 ))
```

| # | Area | Headers | Test selection |
|---|---|---|---|
| 0 | Factorizations and solvers | `include/mtl/operation/`: `lu*`, `qr`, `lq`, `cholesky`, `ldlt*`, `svd`, `schur`, `hessenberg`, `eigenvalue*`, `householder`, `givens`, `sylvester`, `expm`, `inv`, `*trisolve`, `trsv`, `trsm` | `-R "$FACT"` |
| 1 | Kernels and element-wise operations | the rest of `include/mtl/operation/` | `-R '^op_test_' -E "$FACT"` |
| 2 | Sparse direct solvers | `include/mtl/sparse/` | `^sparse_` |
| 3 | Iterative solvers | `include/mtl/itl/` | `^itl_` |
| 4 | Matrix types | `include/mtl/mat/` | `^mat_` |
| 5 | Vector types | `include/mtl/vec/` | `^vec_` |
| 6 | External library bindings | `include/mtl/interface/` | `^iface_` |
| 7 | Recursion, threading, SIMD | `include/mtl/recursion/`, `parallel/`, `simd/` | `^(recursion\|parallel\|simd)_` |
| 8 | I/O and generators | `include/mtl/io/`, `generators/` | `^(io\|generators)_` |
| 9 | Foundations | `include/mtl/concepts/`, `traits/`, `tag/`, `math/`, `functor/`, `detail/`, `util/` | `^(concept\|math\|functor\|detail\|util)_` |
| 10 | Arrays and tensors | `include/mtl/array/`, `tensor/` | `^(array\|tensor)_` |

The `\|` in the patterns is Markdown escaping; the regex alternation is a
plain `|`, as in `^(io|generators)_`. Areas 0 and 1 split the operation tests
with one pattern, so the two sets are disjoint:

```bash
FACT='^op_test_(lu|qr|lq|cholesky|ldlt|svd|schur|hessenberg|eigen|householder|givens|sylvester|expm|inv|trisolve|trsv|trsm|factorization)'
```

Shell variables don't carry over between commands, so set `FACT` in the same
command as the `ctest` that uses it. For the other areas, the selection is
`-R '<pattern>'`.

Then:

1. **Review.** Give `mtl5-qa-reviewer` the area's headers and their tests
   under `tests/unit/` and `tests/regression/`. Ask for all three kinds of
   finding. For a large area, split the headers across two or three reviewer
   runs. The regression tests are for reading only: they are off by default
   (`MTL5_BUILD_REGRESSION_TESTS`) and too slow to run under the sanitizers,
   so step 2 doesn't build them.
2. **Sanitizers.** Build the area's unit tests with AddressSanitizer and
   UndefinedBehaviorSanitizer and run them:

   ```bash
   cmake -S . -B build-asan -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_STANDARD=20 \
     -DMTL5_BUILD_TESTS=ON -DMTL5_BUILD_EXAMPLES=OFF \
     -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined"
   ctest --test-dir build-asan -N <selection>   # the area's tests; a test's name is its target's
   cmake --build build-asan -j4 --target <the listed names>
   ctest --test-dir build-asan <selection> --output-on-failure
   ```

   Every sanitizer report is a functional bug. Give the test, the report's
   first frame inside `include/mtl/`, and the error kind.

## 5. Drop what is already reported

```bash
gh issue list --repo stillwater-sc/mtl5 --label qa-sweep --state open \
  --limit 50 --json number,title,body
gh issue list --repo stillwater-sc/mtl5 --state open --limit 100 --json number,title,body
```

Leave out a finding that an open issue already reports. Mention it under
"Still open" with the issue number, and don't comment on that issue.

## 6. File the sweep issue

Create the label if it doesn't exist:

```bash
gh label list --repo stillwater-sc/mtl5 --search qa-sweep --json name
gh label create qa-sweep --repo stillwater-sc/mtl5 --color 5319e7 \
  --description "Findings of the daily QA sweep"
```

File one issue, titled `QA sweep YYYY-MM-DD: <area>`, labelled `qa-sweep`,
with this body:

```markdown
Swept commit: <full sha>
Range reviewed: <first sha>..<last sha> (<n> commits, <m> after skipping docs-only)
Deep pass: <area> (<k> headers)
Build and tests: <passed / which failed>
Sanitizers: <passed / which reported>

## Functional bugs
- **<title>**: `<file:line>`. <triggering input and effect>. <confirmed with a test program / unconfirmed>.

## Regression gaps
- ...

## Performance
- ...

## Still open
- #<n>: <title>

🤖 Generated with [Claude Code](https://claude.com/claude-code) by the daily QA sweep
```

Leave out a section with no entries. If the sweep found nothing, still file
the issue, saying so, and close it right away:
`gh issue close <n> --reason completed`. The closed issue keeps the
`Swept commit` line for the next sweep, without adding an open issue.

End with the issue's URL and a one-line count: bugs, gaps, performance.
