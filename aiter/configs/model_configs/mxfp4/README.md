# MXFP4 training case profiles

These tables were moved unchanged from
[Lumen 7b5931d21a56869f3e13b55992d7a4e1f8484267](https://github.com/ZhangDanyang-AMD/Lumen/tree/7b5931d21a56869f3e13b55992d7a4e1f8484267).
They contain historical gfx950 / 256-CU measurements, not measurements made during
this configuration refactor. The source project's MIT copyright and permission
notice is preserved in `SOURCE_LICENSE`.

| Profile | Original Lumen location | Case rows |
|---|---|---:|
| llama2_7b | examples/llama2/configs/llama2_7b_a4w4_blockscale_tuned_gemm.csv | 9 |
| llama31_8b | examples/llama31/configs/llama31_8b_a4w4_blockscale_tuned_gemm.csv | 7 |
| qwen3_8b | examples/qwen3/configs/qwen3_8b_a4w4_blockscale_tuned_gemm.csv | 8 |
| qwen3_small | examples/qwen3/configs/a4w4_blockscale_tuned_gemm.csv | 13 |
| qwen3_fsdp | qwen3_8b + qwen3_small, in that order | 21 |

The small-shape table was renamed for an explicit case identity; its bytes were
not changed. It includes one CK row, which is intentionally retained.

## API and data contract

Use `aiter.utility.mxfp4_tuning.configure_mxfp4_tuning(profile=...)` before the
first A4W4 lookup, or use its `--profile` CLI to produce a resolved CSV path.
`profiles.json` is the resource manifest. The parent-directory automatic scan
must not recursively merge every case here: different cases intentionally retain
different kernel choices for shared shapes.

- Keys are `(gfx, cu_num, M, N, K)`, with logical element K, not packed K/2.
- Values retain `kernelName`, `kernelId`, `splitK` and historical measurement
  columns. Do not replace kernel choices by comparing timings from different runs.
- Operands follow the existing gfx950 A4W4 API: packed E2M1 and E8M0 block-32
  scales, B preshuffle and the corresponding padded/swizzled scale layouts.
- A requested case precedes the original default table and top-level model tables.
  Explicit `AITER_CONFIG_GEMM_A4W4` replaces this selection.
- Ordered sources use first-wins resolution. Conflicting rows within one source
  are errors. Loading never edits the source tables.
- First lookup freezes one byte snapshot and writes only a content-addressed
  derived CSV. Its effective fingerprint is stable even if an external tuner
  later edits a source file. Configure/resolve publication is synchronized.

The public GEMM ABI and its ASM/CK selection are unchanged. Runtime Lumen backend
choice caches and shape logs are not kernel tuning tables and are not stored here.

CPU validation covers resource selection, conflicts, source immutability,
concurrent initialization and fingerprinting. No GPU correctness or performance
claim is added by this migration.
