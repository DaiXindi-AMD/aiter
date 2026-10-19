# MXFP4 MoE code preview

This branch is a source-reading snapshot, not a release or a PR-ready claim.
It contains the reviewed N0/A0p implementation and the immutable prepared config view over AITER base `3a80ec0cb84382f0757543fa48a1bde79347db3b`.

## Read the code

- [Public dense facade](../aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py): prepare, bind, late attach, F/D/W and packed replay.
- [Public contract and config metadata](../aiter/ops/triton/utils/mxfp4_contract.py).
- [Dense training tests](../op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4_training_contract.py).
- Lumen consumer snapshot: https://github.com/ZhangDanyang-AMD/Lumen/tree/mxfp4-moe-code-preview . Its AITER gitlink identifies the exact dependency commit.

Finite source review covers this preview and its publication-only fixes. All 17 changed Python files passed the actual repository Ruff 0.16.0 and Black 26.10.1 checks, and staged whitespace passed. The None-return spelling, short-circuit validation and export ordering follow-ups preserve the API/computational structure; this is not a GPU validation result.

These checks do not establish actual AITER/Lumen/Core import, preparation, collection, numerical, GPU/EP/DDP, runtime allocation, fault, training or performance correctness. Those checks remain unverified in this snapshot. Do not infer runtime success from test source or skipped cases.

This preview does not change the existing M1 source worktree or its dependency pins. No public PR, runtime deployment or resource action is part of this publication.
