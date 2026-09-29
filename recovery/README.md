# Lumen MXFP4 AITER recovery snapshot

Updated: 2026-09-29

This is a non-submission recovery snapshot for the Lumen Qwen3-8B MXFP4
optimization work. Do not merge this branch wholesale into an AITER product
branch.

## What is preserved

- The 21 AITER code, test, benchmark, helper, and gfx950 config files from the
  dirty `/home/xdai/aiter` worktree are captured at their 2026-09-29 state.
  Seven files changed after the earlier `35e796da`/2026-09-28 snapshot.
- Four additional files preserve the completed backward-fusion correctness
  work and its exact-shape benchmark: two gfx950 configs, one test, and one
  benchmark.
- `recovery/dirty-code-paths.txt` lists all 21 files.
- `recovery/dirty-code-files.sha256` records their SHA256 checksums.
- `HANDOFF_AITER_LUMEN_2026-09-29.md` records the two operator branches,
  validation evidence, remaining work, Lumen state, and new-machine recovery.
- `recovery/mxfp4-pending-operators.bundle` preserves the exact clean
  dual-layout ref and stacked dequant-H16 WIP ref. Direct branch creation was
  blocked by an OAuth credential without GitHub's `workflow` scope.
- `recovery/bench-ecfff3f-lumen-base.bundle` preserves the exact runtime base
  commit `e35bb17f4f815903bf73598facedbb321e15af28`; it requires parent
  `ecfff3fa80f906c5c421a35a7f5e52842f000559`.
- `recovery/bench-ecfff3f-lumen-base.patch` is a human-readable fallback for
  reconstructing that one base commit.
- `.agents/skills/aiter-merge-preflight/` is agent-only support material and is
  deliberately separate from product code.

Artifact checksums:

```text
8db4c9870a7aa70b714a242ce225731ddba286553b59e6e1fc18857365f99067  recovery/bench-ecfff3f-lumen-base.bundle
1c1915165fe0576a78d1a0d68b7fb29fc7db0c1a4e777d8baaf56b93a77b30c7  recovery/bench-ecfff3f-lumen-base.patch
205f595307e231c5a05690666596c3dadab31c029a3a1f53f81767672cfed254  recovery/mxfp4-pending-operators.bundle
```

The source worktree fingerprint for its tracked diff is:

```text
fea7601c09d062b3bbb7fa8dcfc0888b87876dcfd133d10f3d1147d78409477a
```

## Relationship to Lumen

The committed AITER base `e35bb17f4` was built specifically for Lumen. It is
one commit above Lumen's `third_party/aiter` pin `ecfff3fa8` and adds the
SwiGLU forward/backward API and `hipb_mm_mixed` support used by Lumen paths.

The dirty snapshot adds three groups:

1. Lumen-integrated dependencies: split/eager-compatible SwiGLU and the
   two-source MXFP4 quantizer used by the packed gate/up weight cache, together
   with their shared helpers, tests, and benchmarks.
2. Positive micro-only WIP: fused SwiGLU forward plus dual-layout MXFP4 output.
   It is bitwise equal to the matching two-stage path and measured
   `1.0369x--1.0620x` faster in three fresh processes at `(16384, 12288)`, but
   has no Lumen E2E or NLL result.
3. Rejected full-backward fusion WIP: correctness reached `23 passed, 1 skipped`,
   but the production-parity BM256 implementation measured `0.9746x` versus the
   unfused chain and BM32 measured `0.7398x`. Keep it for redesign/reference;
   do not integrate it into Lumen as a speed optimization.

## Exact recovery procedure

Recover the two pending operator refs first:

```bash
git clone https://github.com/DaiXindi-AMD/aiter.git
cd aiter
git remote add upstream https://github.com/ROCm/aiter.git
git fetch upstream main
git fetch origin backup/2026-09-29/mxfp4-current-handoff

git show \
  origin/backup/2026-09-29/mxfp4-current-handoff:recovery/mxfp4-pending-operators.bundle \
  > /tmp/mxfp4-pending-operators.bundle
test "$(sha256sum /tmp/mxfp4-pending-operators.bundle | cut -d' ' -f1)" = \
  205f595307e231c5a05690666596c3dadab31c029a3a1f53f81767672cfed254

git bundle verify /tmp/mxfp4-pending-operators.bundle
git fetch /tmp/mxfp4-pending-operators.bundle \
  refs/heads/dai/mxfp4-dual-layout-upstream:refs/heads/restore/mxfp4-dual-layout \
  refs/heads/dai/mxfp4-dequant-h16-requant-upstream:refs/heads/restore/mxfp4-dequant-h16
```

To recover the research WIP in a separate checkout, extract the older runtime
base bundle and restore the 21 preserved paths:

```bash
git clone https://github.com/ROCm/aiter.git
cd aiter

git fetch https://github.com/DaiXindi-AMD/aiter.git \
  refs/heads/backup/2026-09-29/mxfp4-current-handoff:refs/remotes/recovery/mxfp4-current-handoff

git show refs/remotes/recovery/mxfp4-current-handoff:recovery/bench-ecfff3f-lumen-base.bundle \
  > /tmp/bench-ecfff3f-lumen-base.bundle
test "$(sha256sum /tmp/bench-ecfff3f-lumen-base.bundle | cut -d' ' -f1)" = \
  8db4c9870a7aa70b714a242ce225731ddba286553b59e6e1fc18857365f99067

git fetch /tmp/bench-ecfff3f-lumen-base.bundle \
  refs/heads/bench/ecfff3f-lumen:refs/heads/recovery/ecfff3f-lumen-base
git switch recovery/ecfff3f-lumen-base
test "$(git rev-parse HEAD)" = e35bb17f4f815903bf73598facedbb321e15af28

git show refs/remotes/recovery/mxfp4-current-handoff:recovery/dirty-code-paths.txt \
  > /tmp/aiter-lumen-mxfp4-dirty-code-paths.txt
xargs -a /tmp/aiter-lumen-mxfp4-dirty-code-paths.txt \
  git restore --source=refs/remotes/recovery/mxfp4-current-handoff --worktree --

git status --short
git diff --binary HEAD | sha256sum
```

The final command must print the tracked-diff fingerprint above. Verify the
untracked source files against `recovery/dirty-code-files.sha256` from the
recovery ref. Restore `.agents/skills/aiter-merge-preflight/` separately only
if the new Agent needs that local helper.
