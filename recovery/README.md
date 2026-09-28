# Lumen MXFP4 AITER recovery snapshot

This is a non-submission recovery snapshot for the Lumen Qwen3-8B MXFP4
optimization work. Do not merge this branch wholesale into an AITER product
branch.

## What is preserved

- The 17 AITER code, test, benchmark, helper, and gfx950 config files from the
  dirty `/home/xdai/aiter` worktree are byte-identical to commit
  `35e796da188e2131d004e5b17391b7b8e836d852` and this branch.
- `recovery/dirty-code-paths.txt` lists those 17 files.
- `recovery/dirty-code-files.sha256` records their SHA256 checksums.
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
```

The source worktree fingerprint for its tracked diff is:

```text
a5476078484e426e951c2a5e62233cb20f26c103f93656e02a690e294e9f9aa0
```

## Relationship to Lumen

The committed AITER base `e35bb17f4` was built specifically for Lumen. It is
one commit above Lumen's `third_party/aiter` pin `ecfff3fa8` and adds the
SwiGLU forward/backward API and `hipb_mm_mixed` support used by Lumen paths.

The dirty snapshot adds two groups:

1. Lumen-integrated dependencies: split/eager-compatible SwiGLU and the
   two-source MXFP4 quantizer used by the packed gate/up weight cache, together
   with their shared helpers, tests, and benchmarks.
2. Future WIP: fused SwiGLU plus dual-layout MXFP4 output, with a public wrapper,
   Triton kernel, gfx950 config, test, and benchmark. Current Lumen does not call
   this fused public API, so it must be evaluated and integrated separately.

## Exact recovery procedure

Fetch this recovery branch, extract the small bundle, recreate the exact AITER
base, and then restore only the 17 dirty paths into the worktree:

```bash
git clone https://github.com/ROCm/aiter.git
cd aiter

git fetch https://github.com/DaiXindi-AMD/aiter.git \
  refs/heads/backup/2026-09-28/ecfff3f-lumen-portable-wip:refs/remotes/recovery/ecfff3f-lumen-portable-wip

git show refs/remotes/recovery/ecfff3f-lumen-portable-wip:recovery/bench-ecfff3f-lumen-base.bundle \
  > /tmp/bench-ecfff3f-lumen-base.bundle
test "$(sha256sum /tmp/bench-ecfff3f-lumen-base.bundle | cut -d' ' -f1)" = \
  8db4c9870a7aa70b714a242ce225731ddba286553b59e6e1fc18857365f99067

git fetch /tmp/bench-ecfff3f-lumen-base.bundle \
  refs/heads/bench/ecfff3f-lumen:refs/heads/recovery/ecfff3f-lumen-base
git switch recovery/ecfff3f-lumen-base
test "$(git rev-parse HEAD)" = e35bb17f4f815903bf73598facedbb321e15af28

git show refs/remotes/recovery/ecfff3f-lumen-portable-wip:recovery/dirty-code-paths.txt \
  > /tmp/aiter-lumen-mxfp4-dirty-code-paths.txt
xargs -a /tmp/aiter-lumen-mxfp4-dirty-code-paths.txt \
  git restore --source=refs/remotes/recovery/ecfff3f-lumen-portable-wip --worktree --

git status --short
git diff --binary HEAD | sha256sum
```

The final command must print the tracked-diff fingerprint above. Verify the
untracked source files against `recovery/dirty-code-files.sha256` from the
recovery ref. Restore `.agents/skills/aiter-merge-preflight/` separately only
if the new Agent needs that local helper.

