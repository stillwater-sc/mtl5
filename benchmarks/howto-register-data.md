# Running mtl5_bench

## Run the benchmark

```bash

(p311) stillwater@sw-21:~/dev/stillwater/clones/mtl5$ ./benchmarks/platform.sh
mtl5 platform benchmark: class=medium variants=native,native-fast,blas threads=1,20 jobs=10
results: /home/stillwater/dev/stillwater/clones/mtl5/build-platform/results/sw-21-20261109T025123Z
preflight: warn: governor is 'powersave', not 'performance' -- clocks are not pinned
== build native
== build native-fast
== build blas
preflight: warn: governor is 'powersave', not 'performance' -- clocks are not pinned
== run native, 1 thread(s)
   107 of 107 checks passed
== run native, 20 thread(s)
   107 of 107 checks passed
== run native-fast, 1 thread(s)
   107 of 107 checks passed
== run native-fast, 20 thread(s)
   107 of 107 checks passed
== run blas, 1 thread(s)
   107 of 107 checks passed
== run blas, 20 thread(s)
   107 of 107 checks passed

summary: /home/stillwater/dev/stillwater/clones/mtl5/build-platform/results/sw-21-20261109T025123Z/summary.md

```

## Seeding the mtl5_bench database

On the Ryzen box:

```bash
  git checkout main
  git pull
  git checkout -b benchmark/ryzen-9-8945hs-data  # create the data branch
  ls build-platform/results/                     # find the session folder, e.g. <host>-20261108T231900Z
  cmake -DRESULTS=build-platform/results/<host>-20261108T231900Z \
        -DMACHINE=ryzen-9-8945hs -DPRELIMINARY=ON \
        -P benchmarks/publish_platform.cmake
```

PowerShell, if it's the Windows box:

```psl
  git fetch origin
  git switch -c benchmark/ryzen-9-8945hs-data origin/main
  dir build-platform\results
  cmake "-DRESULTS=build-platform\results\<host>-20261108T231900Z" `
        -DMACHINE=ryzen-9-8945hs -DPRELIMINARY=ON `
        -P benchmarks\publish_platform.cmake
```

### What it does:
- Copies the session into benchmarks/data/ryzen-9-8945hs/mtl5_bench/<session>/.
- Rewrites the card page docs/benchmarks/cards/ryzen-9-8945hs.md. Its header will then point at the committed data instead of saying "raw data not yet committed".

Then commit those files (git status shows them), push the branch, and open a PR.

If the script refuses:
- "built from a dirty tree": the session was built from a tree with uncommitted changes. Add -DALLOW_DIRTY=ON if you're happy to publish it anyway.
- Session name differs from 20261108T231900Z: the current page names that session, so if the folder you publish has a different time stamp, the page will switch to the new one. That's fine if it's the run you want.

Once the new PR merges, you can delete benchmark/ryzen-9 with git push origin --delete benchmark/ryzen-9.
