# AITER–Lumen MXFP4 kernel handoff and recovery guide

Updated: 2026-10-07 (US/Central)

The filename is retained because it is the original handoff entry point from
2026-09-22. This revision supersedes the old status in that document.

## Purpose

This guide is the authoritative handoff for continuing two pending AITER
operators and the later Lumen integration on another machine:

1. `dual_layout_quant_mxfp4`
2. `dequant_hadamard_quant_mxfp4`

The required product behavior is unchanged:

- keep every existing RHT/H16 point in the Lumen training chain;
- put new GPU kernels in AITER and call only AITER public wrappers from Lumen;
- submit one AITER PR per operator;
- submit the final Lumen migration as one commit after its AITER dependencies
  merge;
- use `DaiXindi-AMD <xdai@amd.com>` with DCO sign-off and no AI/Codex
  co-author trailer in submission commits.

## Read this first

- The dual-layout operator is the only one of the two that is currently
  reviewed, cleaned up, GPU-tested, and benchmarked on the new upstream base.
- The dequant-H16 operator is preserved as a clean stacked WIP commit. It is
  not PR-ready and must not be presented as tested after the upstream rebuild.
- AITER #5542 merged as `48b13652fd05ed6e65d82204d3e040373ff74708`.
  Its public backward API is `aiter.ops.triton.activation.silu_and_mul_backward`
  and its tuned support is gfx950 only.
- AITER #6124 is the still-open architecture-guard follow-up. Until it merges,
  Lumen must check the device architecture and fall back on non-gfx950 devices;
  importing the public symbol alone is not a sufficient capability check.
- The large fused SwiGLU/two-input-quant workspace is research WIP. It is
  backed up only so no source is lost and must not be submitted wholesale.
- The Lumen commit `0f2cc903` is an integration prototype, not final submission
  history. Its author metadata and old AITER imports must be replaced when the
  final Lumen commit is rebuilt.

## Upstream state at handoff

The latest fetched `ROCm/aiter:main` at this update is:

```text
b1cdc19ebf3a52b101221fee5b24a3ef7753c244
```

It was fetched on 2026-10-07. The relevant upstream PR state was checked from
the GitHub API:

| PR | State | Merge/head commit | Relevance |
| --- | --- | --- | --- |
| ROCm/aiter #5531 | merged | `2887d489943fbe9a8ed5bbe0db42bc2fad4bbc1b` | canonical stochastic MXFP4 conversion/helpers |
| ROCm/aiter #5538 | merged | `b220dbaf977d706e59e18db4eb920b7a9672d6b9` | packed FP4 logical transpose |
| ROCm/aiter #5542 | merged | `48b13652fd05ed6e65d82204d3e040373ff74708` | fused SiLU-and-multiply backward; gfx950 only |
| ROCm/aiter #5548 | merged | `6bb62a0978659a47176868c9cb5c0f05679ee383` | 32x32 block-scaled MXFP4 quantization |
| ROCm/aiter #6124 | **open** | head `b207635449d945d9e445df3f0d55717c7dab3d2d` | fail-closed architecture guard for #5542 on unsupported GPUs |

All four merged operator commits above are ancestors of `b1cdc19ebf`. #5542
is no longer a landing blocker. The complete Lumen migration still waits for
the two production MXFP4 operators in this guide. If #6124 has not merged when
Lumen integration starts, keep the equivalent gfx950-only guard in Lumen.

## Remote recovery refs

The recovery snapshots live in the user's forks. The `backup/...` refs must
not be used directly as PR branches.

At handoff, direct HTTPS creation of the two operator refs was rejected because
the available OAuth credential lacks GitHub's `workflow` scope and the newer
upstream history contains workflow-file changes. SSH was not configured. The
exact two refs and commit objects are therefore stored in
`recovery/mxfp4-pending-operators.bundle` on the AITER backup branch. This is
the authoritative portable copy until the refs are pushed with suitable
credentials.

### DaiXindi-AMD/aiter

| Remote branch | Expected source commit | Use |
| --- | --- | --- |
| `dai/mxfp4-dual-layout-upstream` | `deaf07a98bfa3c09f24f936e2531f7a0c85ee9ec` | intended PR ref; not published at handoff, recover from bundle |
| `dai/mxfp4-dequant-h16-requant-upstream-wip` | `46f57675342fde4622d17c446aef4d4e823e5ef5` | intended WIP ref; not published at handoff, recover from bundle |
| `backup/2026-09-29/mxfp4-current-handoff` | resolve after fetch | handoff, exact operator bundle, and current research-WIP snapshot |
| `backup/2026-09-28/ecfff3f-lumen-portable-wip` | `5d7178517e5f3947b7496fa5994696aa2fb9c50d` | previous portable snapshot retained for comparison |

### DaiXindi-AMD/Lumen

| Remote branch | Expected source commit | Use |
| --- | --- | --- |
| `backup/2026-09-29/aiter-kernel-migration-handoff` | contains `0f2cc90398ad05379d4eff125cfd365aaa71a7ce` | Lumen migration prototype and integration documents |

The operator bundle SHA256 is:

```text
205f595307e231c5a05690666596c3dadab31c029a3a1f53f81767672cfed254
```

The handoff procedure was not intended to open PRs automatically.

## Operator 1: dual-layout H16 MXFP4 quantization

### Exact state

```text
Local branch: dai/mxfp4-dual-layout-upstream
Base:         475cf0f607d71010a02288094f20e97254d7d2d2
Commit:       deaf07a98bfa3c09f24f936e2531f7a0c85ee9ec
Subject:      feat(triton): add dual-layout H16 MXFP4 quantization
State:        clean, one commit, DCO signed
```

Author and committer are both:

```text
DaiXindi-AMD <xdai@amd.com>
```

The commit contains no Codex co-author trailer.

### Function and semantics

The public `dual_layout_quant_mxfp4` API quantizes one input and produces both
layouts needed by training:

- the ordinary row-wise MXFP4 layout;
- the transposed, normalized H16-RHT layout used by the opposite GEMM side.

The production wrapper supports `block_size`, H16 sign `g`, scale swizzle, and
packed-column shuffle controls. It preserves the existing Lumen RHT behavior
instead of replacing it with an ordinary MXFP4 conversion.

### AITER reuse and layout work

The implementation reuses upstream AITER helpers rather than duplicating
MXFP4 conversion logic:

- `_mxfp4_scale_from_amax`
- `_mxfp4_sr_random_words`
- `_mxfp4_sr_pack`
- `_mxfp4_pack_op`

Shared H16, validation, endpoint-safe scale, scale-swizzle, and packed-layout
offset logic is in:

```text
aiter/ops/triton/utils/_triton/mxfp4.py
aiter/ops/triton/utils/mxfp4.py
```

The unnecessary `x must be a CUDA tensor` wrapper check was removed. The
compiler/runtime remains responsible for rejecting an invalid backend tensor.

### Files in the commit

```text
aiter/ops/triton/_triton_kernels/quant/dual_layout_mxfp4.py
aiter/ops/triton/configs/gfx950/triton/fusions/dual_layout_mxfp4/DEFAULT.json
aiter/ops/triton/quant/__init__.py
aiter/ops/triton/quant/dual_layout_mxfp4.py
aiter/ops/triton/utils/_triton/mxfp4.py
aiter/ops/triton/utils/mxfp4.py
op_tests/op_benchmarks/triton/bench_dual_layout_mxfp4.py
op_tests/triton_tests/quant/test_dual_layout_mxfp4.py
```

The gfx950 default config is:

```json
{
  "_dual_layout_quant_mxfp4_kernel": {
    "num_warps": 4,
    "num_stages": 1
  }
}
```

### Validation already completed

Hardware:

```text
AMD Instinct MI350X VF
gfx950
8 visible GPUs
```

GPU test command:

```bash
env PYTHONPATH=/path/to/aiter-dual-layout \
  python -m pytest \
  op_tests/triton_tests/quant/test_dual_layout_mxfp4.py -q
```

Result:

```text
27 passed in 9.38s
```

Static validation completed with Ruff and Black. On this host Black must use
one worker:

```bash
black --check --workers 1 <changed-python-files>
ruff check <changed-python-files>
git diff --check HEAD^
```

Representative benchmark at `4096x4096`:

```text
fused_rtn_layout:       0.045406 ms
decomposed_rtn_layout:  0.507832 ms
```

Measured fused/decomposed effective bandwidth:

| Shape | Fused | Decomposed |
| --- | ---: | ---: |
| `512x4096` | 469.69 GB/s | 32.91 GB/s |
| `2048x4096` | 991.66 GB/s | 129.76 GB/s |
| `4096x4096` | 1148.05 GB/s | 231.23 GB/s |
| `6144x4096` | 1216.60 GB/s | 239.12 GB/s |
| `12288x4096` | 1285.08 GB/s | 258.69 GB/s |
| `24576x4096` | 1244.53 GB/s | 252.50 GB/s |
| `4096x12288` | 1300.94 GB/s | 252.34 GB/s |

### Remaining action

The code is ready for a final remote diff/rebase check and PR creation. If
upstream main advances, rebase this single commit, rerun the targeted test and
benchmark, and confirm that no new equivalent kernel/helper has appeared.

## Operator 2: dequant -> transpose -> H16 -> requant

### Exact state

```text
Local branch: dai/mxfp4-dequant-h16-requant-upstream
Bundle ref:   refs/heads/dai/mxfp4-dequant-h16-requant-upstream
Base stack:   475cf0f607 -> deaf07a98 -> 46f576753
WIP commit:   46f57675342fde4622d17c446aef4d4e823e5ef5
Subject:      feat(triton): add MXFP4 dequant H16 requantization
State:        clean stacked source, DCO signed, not PR-ready
```

Only the old dequant-H16 commit was replayed on top of the cleaned dual-layout
commit. The old local #5548 implementation commit `afcc00cb3` was deliberately
not retained because upstream #5548 is merged.

### Current WIP files

```text
aiter/ops/triton/_triton_kernels/quant/mxfp4.py
aiter/ops/triton/quant/__init__.py
aiter/ops/triton/quant/mxfp4.py
op_tests/op_benchmarks/triton/bench_dequant_hadamard_quant_mxfp4.py
op_tests/triton_tests/quant/test_dequant_hadamard_quant_mxfp4.py
```

The generic `mxfp4.py` filenames are part of the unfinished state and must be
renamed before submission.

### Required continuation work

1. Rename both generic modules:

   ```text
   aiter/ops/triton/_triton_kernels/quant/mxfp4.py
   -> aiter/ops/triton/_triton_kernels/quant/dequant_hadamard_quant_mxfp4.py

   aiter/ops/triton/quant/mxfp4.py
   -> aiter/ops/triton/quant/dequant_hadamard_quant_mxfp4.py
   ```

2. Remove duplicate helpers and reuse the dual/upstream implementation:

   - `_decode_e8m0` -> `_mxfp4_e8m0_to_fp32`
   - `_normalized_hadamard16` -> `normalized_hadamard16`
   - `_prepare_h16_sign` -> `prepare_h16_sign`
   - local endpoint-safe scaling -> `mxfp4_apply_quant_scale`

   Keep `_decode_e2m1` locally only if a fresh upstream search still finds no
   equivalent general helper.

3. Add the production wrapper controls:

   ```python
   use_sr=False
   philox_seed=None
   philox_offset=0
   swizzle_scale=False
   shuffle_data=False
   in_scale_swizzled=False
   ```

4. Extend the kernel with upstream SR helpers, swizzled input-scale reads,
   swizzled output-scale stores, and shuffled packed-FP4 output stores.

5. Include these compile/tuning keys in the launchable kernel repr:

   ```text
   USE_SR
   SWIZZLE_SCALE
   SHUFFLE_DATA
   IN_SCALE_SWIZZLED
   num_warps
   ```

6. Add the required gfx950 tuning file:

   ```text
   aiter/ops/triton/configs/gfx950/triton/fusions/
     dequant_hadamard_quant_mxfp4/DEFAULT.json
   ```

   Initial measured candidate to validate, not blindly assume:

   ```json
   {
     "_dequant_hadamard_quant_mxfp4_kernel": {
       "num_warps": 4,
       "num_stages": 1
     }
   }
   ```

7. Remove the explicit `data_fp4 must be a CUDA tensor` check and any matching
   unit test that expects that artificial `ValueError`.

8. Preserve all existing numerical semantics:

   - input raw E8M0 value `255` poisons the complete corresponding output
     scale block;
   - poisoned output payload is zero and output scale is `255`;
   - decoded values are converted to BF16 before transpose/H16;
   - signed zero is canonicalized;
   - raw scale endpoints `0`, `254`, and `255` remain covered;
   - canonical FP4/E8M0 dtypes and strided inputs remain covered;
   - graph-capture-compatible sign validation remains covered.

9. Add production-layout tests:

   - fused scale swizzle equals `shuffle_scale_gemm(..., arch="gfx950",
     preshuffle_factor=32, scale_kwidth=8)`;
   - fused payload shuffle equals `shuffle_weight(..., arch="gfx950",
     layout=(16, 16))`;
   - `in_scale_swizzled=True` equals the row-major input path;
   - stochastic rounding is reproducible for a fixed seed/offset;
   - RTN and SR share the intended scale calculation;
   - invalid production metadata is rejected by the public wrapper.

10. Extend the benchmark with `fused_layout` and `decomposed_layout`. The
    decomposed reference must include dequant, transpose, H16, requant, scale
    shuffle, and payload shuffle so the comparison represents the actual
    training layout.

11. Run Black, Ruff, `py_compile`, the targeted GPU pytest, and the benchmark.
    Amend only the second commit. Keep the stack as:

    ```text
    475cf0f607  upstream base
    deaf07a98   dual-layout operator
    <new SHA>   completed dequant-H16 operator
    ```

After the dual-layout PR merges, rebase the completed second commit onto the
new upstream main and open the dequant-H16 PR without carrying the dual commit
in its submitted diff.

## Current research-WIP snapshot

The source workspace `/home/xdai/aiter` on `bench/ecfff3f-lumen` contains a
larger prototype covering fused SwiGLU/dual-layout quantization, two-input
MXFP4 quantization, shared activation/shuffle helpers, tests, benchmarks, and
configs. It is not one of the two submission branches.

The 2026-09-29 recovery branch updates seven files that changed after the
2026-09-28 portable snapshot and preserves 21 code paths in total. The four
late additions are two gfx950 configs, a backward-fusion test, and its
exact-shape benchmark.

The forward fused SwiGLU plus dual-layout prototype was bitwise equal to its
two-stage comparison and measured `1.0369x--1.0620x` faster at
`(16384, 12288)` in three fresh processes, but it has no Lumen end-to-end or
NLL result. The full-backward fusion reached `23 passed, 1 skipped`, but its
production-parity BM256 path measured `0.9746x` and BM32 measured `0.7398x`
versus the unfused chain. Keep the backward fusion for redesign/reference; do
not integrate it as a speed optimization in its current form.

The snapshot is research/integration source only:

- do not submit the backup branch as a PR;
- do not mix it into either production operator without a new duplicate search;
- re-evaluate it after the two production operators settle;
- use it to recover ideas or measured experiments, not as approved product code.

## Lumen integration state

The archival Lumen prototype is:

```text
Repository: /home/xdai/Lumen-kernel-migration
Branch:     codex/aiter-kernel-migration
Commit:     0f2cc90398ad05379d4eff125cfd365aaa71a7ce
```

It removes Lumen-owned kernels and calls the earlier AITER APIs. It is not the
final Lumen commit. In particular:

- its commit author is a development-agent identity;
- its AITER imports predate the final categorized public APIs;
- it must not be pushed as the final submission commit;
- its #5542 integration plan is preserved and updated in the Lumen backup ref.

The relevant production call sites in the prototype are in
`lumen/ops/quantize/linear.py`:

- dual-layout forward near line 1302;
- dual-layout backward near line 2631;
- dequant-H16 activation/WGrad preparation near line 2677.

Line numbers may move; search the public function names after recovery.

### Final Lumen change after AITER merges

1. Start from the requested current `ZhangDanyang-AMD/Lumen:main`.
2. Pin `third_party/aiter` to one reproducible ROCm/AITER revision containing
   #5531, #5538, #5542, #5548, dual-layout, and dequant-H16. Prefer a revision
   containing #6124 after that follow-up merges.
3. Import the final public categorized APIs, for example from
   `aiter.ops.triton.quant`; never import `_triton_kernels`.
4. Add guarded `_probe_aiter_*()` checks, an explicit gfx950 capability check,
   and a logged fallback. During testing, prove the AITER path actually
   executed so fallback cannot hide a failure.
5. Preserve the original RHT/H16 locations and stochastic-rounding choices.
6. Build the complete Lumen migration as one contributor-authored commit.
7. Run AITER operator tests first, then Lumen tests, then Qwen3 MXFP4 Megatron
   and FSDP/FSDP2 smoke or short training runs.

The #5542 integration plan explains why Megatron directly consumes the fused
backward while the current Hugging Face/FSDP MLP does not. It has been updated
for the merged API, gfx950-only support, and open follow-up #6124.

## GPU execution notes

Inside the managed Agent sandbox on the old machine:

```text
torch.cuda.device_count() == 8
torch.cuda.is_available() == False
/dev/kfd is absent
/dev/dri is absent
```

This is sandbox device isolation, not an Agent holding a GPU lock. With an
approved non-sandbox command, the same environment reports gfx950 MI350X and
can run the tests.

Use the complete AITER environment and point `PYTHONPATH` at the target
worktree. Do not set `AITER_TRITON_ONLY=1` in this environment; it caused an
import failure for `aiter.dtypes`.

## Recovery on the new machine

### AITER

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

git worktree add ../aiter-dual-layout \
  restore/mxfp4-dual-layout

git worktree add ../aiter-dequant-h16 \
  restore/mxfp4-dequant-h16

git worktree add ../aiter-recovery-snapshot \
  -b restore/mxfp4-current-handoff \
  origin/backup/2026-09-29/mxfp4-current-handoff
```

Verify before editing:

```bash
test "$(git -C ../aiter-dual-layout rev-parse HEAD)" = \
  deaf07a98bfa3c09f24f936e2531f7a0c85ee9ec

test "$(git -C ../aiter-dequant-h16 rev-parse HEAD)" = \
  46f57675342fde4622d17c446aef4d4e823e5ef5

git -C ../aiter-dual-layout status --short
git -C ../aiter-dequant-h16 status --short
```

The two status commands must be empty before continuation.

### Lumen

```bash
git clone https://github.com/DaiXindi-AMD/Lumen.git
cd Lumen
git fetch origin backup/2026-09-29/aiter-kernel-migration-handoff
git worktree add ../Lumen-aiter-handoff \
  -b restore/aiter-kernel-migration \
  origin/backup/2026-09-29/aiter-kernel-migration-handoff
```

Read these two documents first:

```text
HANDOFF_AITER_LUMEN_2026-09-29.md
outputs/aiter_5542_lumen_integration/AITER_5542_LUMEN_INTEGRATION_PLAN.md
```

## Required AITER acceptance checklist

Before opening either operator PR, confirm:

1. Every launchable kernel has `make_kernel_repr` metadata.
2. The public wrapper has numerical unit tests.
3. A runnable benchmark and current gfx950 measurements are included.
4. Wrapper, kernel, tests, benchmark, and nested tuning JSON are in the proper
   categorized folders.
5. A fresh search finds no duplicate or nearly duplicate AITER kernel/helper.
6. Reusable H16, activation, scale, packing, and shuffle functions are in
   shared utils or canonical upstream modules.
7. Source comments are concise; extended rationale belongs in the PR report.
8. No artificial `is_cuda`/"CUDA tensor" validation duplicates compiler
   behavior.
9. Commit author/committer and `Signed-off-by` are
   `DaiXindi-AMD <xdai@amd.com>`.
10. No AI/Codex co-author trailer is present.

## Recommended continuation order

1. Recover all refs and verify the exact SHAs above.
2. Check whether upstream main advanced beyond `b1cdc19ebf` and whether #6124
   merged.
3. Rebase and rerun the dual-layout targeted test/benchmark if required.
4. Open the dual-layout operator PR.
5. Complete the dequant-H16 checklist while keeping it stacked locally.
6. After dual-layout merges, rebase the second operator alone and open its PR.
7. Integrate merged #5542 through the public API on gfx950; keep a logged
   non-gfx950 fallback until the #6124 guard is present in the pinned AITER.
8. Rebuild one final Lumen commit and run Megatron plus FSDP/FSDP2 validation.

## Old-machine source paths

These paths are recorded only for forensic comparison before the old machine
is released:

```text
/home/xdai/aiter
/home/xdai/aiter-pr-mxfp4-dual-layout-upstream
/home/xdai/aiter-pr-mxfp4-dequant-h16-requant-upstream
/home/xdai/aiter-handoff-20260929
/home/xdai/aiter-backup-20260928
/home/xdai/aiter-kernel-migration
/home/xdai/Lumen-kernel-migration
```
