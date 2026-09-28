---
name: aiter-merge-preflight
description: Preflight, audit, and repair ROCm/aiter pull requests before merge. Use for AITER PR readiness, required checks, CI failures, review-policy compliance, Triton kernel PRs, or deciding whether a red check is caused by the change. Do not use for repositories other than ROCm/aiter or its forks/worktrees.
---

# AITER merge preflight

Use this workflow for every ROCm/aiter PR before push, after push, and when a
check fails. Treat the live GitHub ruleset and the current repository files as
the sources of truth; the bundled reference is only a dated fallback.

## Establish scope

1. Confirm that a remote resolves to `ROCm/aiter` and identify the intended
   base branch, normally `main`.
2. Preserve unrelated user changes. Never clean, reset, or rewrite another
   worktree.
3. Read `AGENTS.md` files in scope, `CONTRIBUTE.md`, and every applicable
   `.github/instructions/*.instructions.md` file completely before editing.
4. Inspect `git status`, the merge base, commits, and `git diff
   <base>...HEAD`. A new launchable kernel must remain one concern per PR and
   carry its wrapper, unit test, and benchmark.

## Discover the current gates

When network access is available, query the effective rules on every audit:

```bash
gh api repos/ROCm/aiter/rules/branches/main
```

Extract the required status contexts and pull-request rules instead of relying
on old screenshots or memorized names. Then inspect the current workflow files
that produce those contexts. Read [references/aiter-ci.md](references/aiter-ci.md)
when a detailed workflow or label map is needed.

Do not equate an aggregate workflow name with a required context. Do not treat
`skipped`, `action_required`, or a check from an older head SHA as a passing
test on the current revision.

## Run local preflight

Run the deterministic checker from the target worktree:

```bash
python /path/to/aiter-merge-preflight/scripts/preflight.py \
  --repo "$PWD" --base upstream/main
```

Use `--online --pr <number>` after the PR exists to compare the local HEAD with
the remote PR and enforce the live rules/checks. The script is read-only. Its
local mode must pass before push; its online mode must pass before calling a
PR merge-ready. It is the deterministic first layer, not a substitute for the
targeted GPU tests and review described below.

The checker validates:

- repository identity, base availability, dirty state, and behind count;
- whitespace and commit trailers, including `Signed-off-by` and the absence
  of Codex co-author trailers;
- Black and Ruff versions derived from the current AITER workflow/docs;
- Black and Ruff on changed Python files;
- AITER Triton review invariants that are safe to detect mechanically;
- presence of wrapper, test, and benchmark when a launchable kernel is added;
- live base/head identity, required checks, applicable workflow conclusions,
  review decision, unresolved threads, and GitHub merge state when `--online`
  is requested.

If required tool versions are missing, create an isolated temporary virtual
environment. Do not change the user's global Python packages.

## Select tests from the diff

Always run the narrowest correctness test that exercises every changed public
API, plus nearby regression tests for code that was modified or reused.

For Triton changes:

1. Run the matching `op_tests/triton_tests/<category>/test_<op>.py` file on an
   appropriate ROCm GPU.
2. Run tests for an existing wrapper/helper modified by the PR.
3. If validation tests create CPU tensors, rerun the new test with
   `torch.set_default_device("cuda")` to catch leaked-default-device bugs.
4. Smoke-test the benchmark with one representative shape when practical.
5. Run the full relevant shard only when its cost and hardware requirements
   are reasonable locally; otherwise rely on the remote matrix and state the
   limitation explicitly.

Never test a private `_triton_kernels` entry point directly. Tests and
benchmarks must exercise the public wrapper and must not hardcode launch
configuration.

## Review before push

Check the complete diff against the current instructions, not only the last
edit. In particular verify:

- one launchable kernel/concern per PR;
- reuse of existing helpers and shuffle utilities;
- public wrapper and internal-kernel folder separation;
- absolute imports and torch-free internal kernel modules;
- config-aware `repr` for launchable kernels;
- lazy logger formatting;
- explicit device handling and output allocation on the input device;
- correctness assertions, boundary cases, non-contiguous/stride behavior,
  dtype/device errors, and benchmark coverage;
- author/committer identity and DCO trailer;
- no `Co-authored-by: Codex` or similar attribution.

After a rebase, use `git range-diff` against the old remote revision to prove
which patch changes were intentional. Push rewritten history only with
`--force-with-lease` and only when the user's request authorizes updating the
PR.

## Audit remote CI

Run the online preflight first. Use `$gh-fix-ci` for actual failing GitHub
Actions checks and their logs; it does not replace detection of missing checks,
`action_required`, or review blockers. If it is unavailable, inspect with
`gh pr checks`, the check-runs API, workflow-run API, and failed job logs.

For every failure classify it as one of:

1. **Change-related:** reproduce or reason from the changed path, fix it, rerun
   targeted tests, and update the same PR.
2. **Fixed upstream/stale base:** rebase to current `upstream/main`, preserve
   the PR patch, rerun local checks, and update with `--force-with-lease`.
3. **Infrastructure or unrelated baseline:** record the exact failing file and
   error, demonstrate that the PR does not touch that subsystem, and request a
   rerun or maintainer action. Do not add unrelated fixes to the kernel PR.

Fork workflows commonly end in `action_required`. That is neither success nor
failure; report that a ROCm/aiter maintainer must approve the run. Do not claim
the PR is fully green until every currently required context succeeds on the
exact remote head SHA and the review requirements are satisfied.

## Final report

Report:

- PR URL, base, exact head SHA, and mergeability;
- required checks and their current conclusions;
- local commands and pass/skip counts;
- any test not run and the concrete reason;
- review/thread/approval blockers;
- whether the branch was pushed and whether `--force-with-lease` was used.

Never promise that external GPU runners, Docker registries, networks, secrets,
or maintainer approvals will succeed. Guarantee only that code-side preflight
has passed and continue monitoring or repairing remote failures when the user
has asked for that work.
