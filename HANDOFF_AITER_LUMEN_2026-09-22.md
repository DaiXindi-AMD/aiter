# AITER–Lumen kernel migration handoff

Date: 2026-09-22

## Remote recovery refs

These are backup refs, not PR branches. Do not open PRs directly from them.

| Repository | Remote branch | Snapshot commit | Purpose |
| --- | --- | --- | --- |
| `DaiXindi-AMD/aiter` | `backup/2026-09-22/pending-aiter-kernels` | updated by the handoff-doc commit | Combined source for scale shuffle, RMSNorm, FP8 fix, and original MXFP4 migration work |
| `DaiXindi-AMD/aiter` | `backup/2026-09-22/fused-swiglu-dual-layout-wip` | `35e796da188e2131d004e5b17391b7b8e836d852` | Newer uncommitted fused SwiGLU/dual-layout and two-input quant prototype |
| `DaiXindi-AMD/Lumen` | `backup/2026-09-22/aiter-kernel-integration` | `dae77b2404df505fa00fb377330681ec4b50d054` | Lumen integration prototype and #5542 integration plan |

The clean operator refs are also present on `DaiXindi-AMD/aiter`:

- `dai/mxfp4-dual-layout` at `ddd226fbacacd2117c59debe72d68cffba85d9da`.
- `dai/mxfp4-dequant-h16-requant-v2` at `de0b691ca79533b4ae7118e6a253ccd96268dbb0`.

Example recovery on a new machine:

```bash
git clone https://github.com/DaiXindi-AMD/aiter.git
cd aiter
git fetch origin backup/2026-09-22/pending-aiter-kernels
git fetch origin backup/2026-09-22/fused-swiglu-dual-layout-wip
git fetch origin dai/mxfp4-dual-layout
git fetch origin dai/mxfp4-dequant-h16-requant-v2
```

For Lumen:

```bash
git clone https://github.com/DaiXindi-AMD/Lumen.git
cd Lumen
git fetch origin backup/2026-09-22/aiter-kernel-integration
```

## Current upstream status

- ROCm/aiter #5531 — merged: gfx950 stochastic MXFP4 quantization.
- ROCm/aiter #5538 — open, waiting to merge: packed FP4 logical transpose.
- ROCm/aiter #5542 — open, waiting to merge: fused SiLU-and-multiply backward.
- ROCm/aiter #5548 — open, waiting to merge: 32x32 block-scaled MXFP4 quantization.

Do not update the remaining branches merely to make CI rerun. Rebase each dependent
branch only when its prerequisite has landed in `ROCm/aiter:main`.

## Clean pending AITER operator branches

### Dual-layout H16 MXFP4

- Fork branch: `DaiXindi-AMD/aiter:dai/mxfp4-dual-layout`
- Commit: `ddd226fbacacd2117c59debe72d68cffba85d9da`
- Local worktree: `/home/xdai/aiter-pr-mxfp4-dual-layout`
- Tracking issue: ROCm/aiter #5552
- Contents: public wrapper, launchable kernel with repr, unit tests, and benchmark.
- State: clean and pushed; no ROCm/aiter PR has been opened.
- Next action: rebuild on current upstream after #5531. Remove the branch-local
  copies of MXFP4 scale, E8M0 decode, Philox, and SR packing helpers and reuse the
  canonical implementation introduced by #5531. Search again for overlap with the
  final #5548 implementation before opening the PR.
- It does not need #5538 or #5542: transpose and H16 are fused inside the operator.

### Dequant -> transpose -> H16 -> requant

- Fork branch: `DaiXindi-AMD/aiter:dai/mxfp4-dequant-h16-requant-v2`
- Commit: `de0b691ca79533b4ae7118e6a253ccd96268dbb0`
- Local worktree: `/home/xdai/aiter-pr-mxfp4-dequant-h16-requant-stack5548`
- Tracking issue: ROCm/aiter #5553
- Contents: public wrapper, launchable kernel with repr, unit tests, and benchmark.
- State: clean and pushed; no ROCm/aiter PR has been opened.
- Hard dependency: its direct parent is `afcc00cb3`, the local version of #5548,
  and its kernel imports `_mxfp4_pack_op` and `_mxfp4_scale_from_amax` from that
  implementation.
- Next action: after #5548 merges, create a fresh branch from upstream `main` and
  replay only `de0b691ca`; do not retain `afcc00cb3` in the submitted history.
- It does not need #5538 or #5542 because transpose and H16 are fused.

## Additional AITER work not yet isolated into submission branches

The combined source workspace is `/home/xdai/aiter-kernel-migration`. Its old
`codex/lumen-kernel-migration` history is archival and must not be submitted as-is.
The changes must be rebuilt as focused, signed commits on current upstream.

- gfx950 `shuffle_scale_gemm` Triton fast path:
  `_swizzle_mxfp4_scale_gfx950_kernel` in
  `aiter/ops/triton/_triton_kernels/quant/mxfp4_layout.py`.
  This is an optimization of an existing operator, not a required new API. Submit
  only with benchmark evidence that justifies replacing the existing path.
- Expanded 2-D scale shuffle:
  `_swizzle_expanded_2d_scale_kernel` plus `shuffle_scale_gemm_expanded` in the
  MXFP4 layout/shuffle files. If still required after direct-swizzled stores are
  finalized, it needs its own focused kernel PR.
- RMSNorm narrow persistent schedule: source commit `c284120c90519976e92080b10bd2c727122e3747`.
  It is independent of MXFP4 and should be a separate performance PR.
- FP8 lazy-dtype fix: source commit `2cd245c7b51d3a4fadd5bc59830e9ee49c4b1974`.
  It is independent of MXFP4 and should be a separate bugfix PR after completing
  Triton-only/AOT validation.

The older combined commits were authored with a development-agent identity and do
not have the required contributor DCO metadata. Do not push them as submission
history. New submission commits must be authored and signed off as:

```text
DaiXindi-AMD <xdai@amd.com>
```

## Newer uncommitted integration prototype

The dirty workspace `/home/xdai/aiter` on `bench/ecfff3f-lumen` contains a newer
prototype of:

- fused eager-rounded SwiGLU plus dual-layout MXFP4;
- two-input 32x32 MXFP4 quantization;
- shared activation and MXFP4 shuffle helpers;
- associated configs, tests, and benchmarks.

This is research/integration work, not a PR-ready branch. It overlaps #5542 and
#5548 and must be rebuilt after their final APIs land. Do not combine it with the
two simpler pending operator PRs without a fresh duplicate-kernel review.

## Superseded local drafts

These worktrees contain older or incomplete forms and should not be submitted:

- `/home/xdai/aiter-pr-mxfp4-sr` — superseded by merged #5531.
- `/home/xdai/aiter-pr-mxfp4-dequant-transpose` — standalone path folded into the
  production dequant-H16 operator.
- `/home/xdai/aiter-pr-mxfp4-dequant-h16-requant` — earlier dirty draft,
  superseded by `de0b691ca`.
- `/home/xdai/aiter-pr-mxfp4-dequant-h16-requant-5548` — intermediate stack,
  superseded by the clean `-stack5548` worktree.

Keep their backup refs only for recovery and comparison.

## Lumen integration

- Local worktree: `/home/xdai/Lumen-kernel-migration`
- Branch: `codex/aiter-kernel-migration`
- Prototype commit: `0f2cc90398ad05379d4eff125cfd365aaa71a7ce`

This prototype removes Lumen-owned copies and calls AITER APIs. It depends on all
four upstream PRs (#5531, #5538, #5542, #5548) and on the two pending production
operators above. It is not the final Lumen commit and its existing author metadata
must not be used for submission.

After every required AITER PR merges:

1. Start from current `ZhangDanyang-AMD/Lumen:main` or the requested target branch.
2. Pin `third_party/aiter` to one upstream ROCm/aiter commit containing all APIs.
3. Adapt Lumen imports to the final public AITER names; never import private kernel bodies.
4. Preserve the existing RHT locations and semantics.
5. Run AITER operator tests first, then Lumen unit tests, then Qwen3 MXFP4 Megatron
   and FSDP/FSDP2 smoke or short training runs.
6. Submit the Lumen migration as one focused Lumen commit.

## AITER acceptance checklist

Before opening each remaining AITER PR, verify:

1. Every launchable Triton kernel has `make_kernel_repr`/`repr=`.
2. The public wrapper has numerical unit tests.
3. A runnable benchmark and measured gfx950 evidence are present.
4. Files and tuning configs are in the correct operator/config directories.
5. No duplicate or near-duplicate helper/kernel remains after rebasing.
6. Shared activation, RHT, scale, packing, and shuffle helpers live in utils or the
   established shared kernel module rather than local copies.
7. Comments are concise.
8. Commits use `DaiXindi-AMD <xdai@amd.com>` and include the DCO sign-off, with no
   AI/Codex co-author trailer.

## Recommended continuation order

1. Fetch the latest `ROCm/aiter:main` containing merged #5531.
2. Rebuild and review the dual-layout branch, then open its PR.
3. Wait for #5548, then rebuild and review the dequant-H16 branch and open its PR.
4. Split and review the optional scale-shuffle optimization(s).
5. Split RMSNorm and FP8 fixes into independent PRs.
6. After #5538, #5542, #5548 and the two production MXFP4 operators merge, rebuild
   the single Lumen integration commit and run Megatron plus FSDP/FSDP2 validation.
