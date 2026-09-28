# ROCm/aiter merge and CI reference

This is a fallback snapshot from 2026-09-16. Query the live effective rules
and read current workflows before relying on it.

## Effective `main` rules

Query:

```bash
gh api repos/ROCm/aiter/rules/branches/main
```

At the snapshot date, the required status contexts are:

- `Check Code Style with Black`
- `Check Code Style with Ruff`
- `Check Repository Dependency`
- `Standard Test Results`

The pull-request rule requires at least one approval, a configured team
approval for `*`, code-owner review, approval after the last push, resolution
of all review threads, and an extra approval for unattributed changes. New
pushes dismiss stale approvals. Only squash merge is allowed. Required checks
are not strict about being based on the newest `main`.

## Required-check producers

### Checks

`.github/workflows/pre-checks.yaml` produces:

- Black through `psf/black@stable`;
- Ruff, currently pinned in the workflow;
- repository dependency checkout. The current PR branch path only prints a
  temporary skip after recursive submodule checkout, so do not claim that it
  validates the CK pin unless the workflow changes.

Downstream GPU workflows call `.github/scripts/check_signal.sh` and wait for
the entire Checks workflow to succeed.

### Standard Test Results

`.github/workflows/aiter-test.yaml` builds the Python 3.12 AITER wheel and the
Triton wheel, splits `op_tests` into eight shards, and runs all shards on both
MI35X/gfx950 and MI300X/gfx942. `Standard Test Results` verifies that all 16
test logs exist.

The workflow also has `Aiter Test Gate`, but that context was not required by
the ruleset at the snapshot date. Inspect both the required context and the
underlying matrix jobs; an aggregate skipped because a dependency failed is
not evidence that tests passed.

## Other PR workflows

Treat these as quality gates when their path or label conditions apply, even
if they are not listed as hard required contexts:

- Triton Test: eight MI35X shards for changes under `aiter/ops/triton`, Triton
  tests, or Triton benchmarks. Label `ci:triton-300x` adds eight MI300X shards.
- OPUS Test: MI35X and MI300X for non-documentation PRs.
- `multigpu`: enables the AITER 8-GPU path.
- `ci:atom`, `ci:atom_full`, `ci:sglang`, `ci:kimi`, `ci:vllm`,
  `ci:performance`, `ci:vllm-di`, and `ci:all`: enable downstream suites.
- `ci:extended-test`: only dispatches from a same-repository branch, not a
  fork PR.
- Actionlint, documentation, Flash Attention, and FFM workflows are
  path/label specific.
- PR Title Tags & Labels derives component tags from changed paths.

## Local commands

Derive exact versions from the live workflow and contribution guide. At the
snapshot date CI uses Ruff 0.16.0 and the contribution guide pins Black
26.3.0.

```bash
black --check --diff <changed-python-files>
ruff check <changed-python-files>
git diff --check upstream/main...HEAD
```

For a Triton operator, also run its matching unit test and affected regression
tests. The full CI entry points are:

```bash
bash .github/scripts/split_tests.sh --shards 8 --test-type triton
bash .github/scripts/split_tests.sh --shards 8 --test-type aiter
bash .github/scripts/aiter_test.sh
bash op_tests/opus/run_tests.sh
```

These full paths require the correct ROCm images, GPU architectures, wheels,
and sometimes multiple GPUs; a normal local shell is not equivalent to the
self-hosted CI environment.

## CI interpretation

- Match checks to the exact PR head SHA.
- `action_required` means a maintainer has not approved fork workflows.
- `skipped` is not proof that the test body ran.
- A failure in a file untouched by the PR is not automatically unrelated:
  inspect imports, global state, device state, and shared helpers first.
- If the same unrelated failure appears across independent PRs on the same
  base, look for an upstream fix before adding changes to the operator PR.
- Do not mix infrastructure work or a different operator's fix into a
  one-kernel PR.
