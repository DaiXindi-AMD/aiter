#!/usr/bin/env python3
"""Read-only local and optional GitHub preflight for ROCm/aiter PRs."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote


@dataclass
class Finding:
    level: str
    message: str


@dataclass
class Commit:
    sha: str
    author_name: str
    author_email: str
    committer_name: str
    committer_email: str
    body: str


def run(
    args: list[str], cwd: Path, *, check: bool = False
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, cwd=cwd, text=True, capture_output=True, check=False)
    if check and result.returncode:
        detail = (result.stdout + result.stderr).strip()
        raise RuntimeError(f"{' '.join(args)} failed: {detail}")
    return result


def git(repo: Path, *args: str, check: bool = True) -> str:
    result = run(["git", *args], repo, check=check)
    return result.stdout.strip()


def version_from(pattern: str, path: Path) -> str | None:
    if not path.exists():
        return None
    match = re.search(pattern, path.read_text(encoding="utf-8"))
    return match.group(1) if match else None


def installed_version(tool: str, repo: Path) -> tuple[str | None, str | None]:
    executable = shutil.which(tool)
    if executable is None:
        return None, None
    result = run([executable, "--version"], repo)
    text = (result.stdout + result.stderr).strip()
    match = re.search(r"\b(\d+\.\d+\.\d+)\b", text)
    return executable, match.group(1) if match else None


def added_lines(repo: Path, base: str, path: str) -> list[str]:
    result = run(["git", "diff", "--unified=0", f"{base}...HEAD", "--", path], repo)
    return [
        line[1:]
        for line in result.stdout.splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]


def parse_commits(repo: Path, base: str) -> list[Commit]:
    raw = git(
        repo,
        "log",
        "--format=%H%x1f%an%x1f%ae%x1f%cn%x1f%ce%x1f%B%x1e",
        f"{base}..HEAD",
    )
    commits: list[Commit] = []
    for record in raw.split("\x1e"):
        record = record.strip()
        if not record:
            continue
        sha, author_name, author_email, committer_name, committer_email, body = (
            record.split("\x1f", 5)
        )
        commits.append(
            Commit(
                sha=sha.strip(),
                author_name=author_name.strip(),
                author_email=author_email.strip(),
                committer_name=committer_name.strip(),
                committer_email=committer_email.strip(),
                body=body.strip(),
            )
        )
    return commits


def local_preflight(repo: Path, base: str) -> tuple[list[Finding], list[str]]:
    findings: list[Finding] = []

    top = Path(git(repo, "rev-parse", "--show-toplevel")).resolve()
    repo = repo.resolve()
    if top != repo:
        findings.append(Finding("FAIL", f"--repo must be the worktree root: {top}"))

    remotes = git(repo, "remote", "-v")
    if "ROCm/aiter" not in remotes:
        findings.append(
            Finding("FAIL", "no remote URL identifies this worktree as ROCm/aiter")
        )

    if run(["git", "rev-parse", "--verify", base], repo).returncode:
        findings.append(Finding("FAIL", f"base ref does not exist: {base}"))
        return findings, []

    status = git(repo, "status", "--porcelain")
    if status:
        findings.append(Finding("FAIL", "worktree is dirty before preflight"))
    else:
        findings.append(Finding("PASS", "worktree is clean"))

    behind = int(git(repo, "rev-list", "--count", f"HEAD..{base}") or "0")
    if behind:
        findings.append(Finding("WARN", f"branch is {behind} commit(s) behind {base}"))
    else:
        findings.append(Finding("PASS", f"branch contains current local {base}"))

    changed = [
        line
        for line in git(
            repo,
            "diff",
            "--name-only",
            "--diff-filter=ACMRD",
            f"{base}...HEAD",
        ).splitlines()
        if line
    ]
    findings.append(Finding("INFO", f"changed files: {len(changed)}"))

    whitespace = run(["git", "diff", "--check", f"{base}...HEAD"], repo)
    if whitespace.returncode:
        findings.append(
            Finding("FAIL", whitespace.stdout.strip() or "diff check failed")
        )
    else:
        findings.append(Finding("PASS", "git diff --check"))

    commits = parse_commits(repo, base)
    if not commits:
        findings.append(Finding("FAIL", f"no commits found above {base}"))
    trailer_failure = False
    for commit in commits:
        short = commit.sha[:10]
        signoffs = re.findall(
            r"(?mi)^Signed-off-by:\s*(.*?)\s*<([^>]+)>\s*$", commit.body
        )
        if not signoffs:
            trailer_failure = True
            findings.append(Finding("FAIL", f"{short} has no Signed-off-by trailer"))
        elif not any(
            email.casefold() == commit.author_email.casefold()
            for _name, email in signoffs
        ):
            trailer_failure = True
            findings.append(
                Finding(
                    "FAIL",
                    f"{short} sign-off email does not match author "
                    f"{commit.author_email}",
                )
            )
        identity = (
            f"{commit.author_name} {commit.author_email} "
            f"{commit.committer_name} {commit.committer_email}"
        )
        if re.search(r"(?i)\b(?:codex|chatgpt|openai)\b", identity) or re.search(
            r"(?mi)^Co-authored-by:.*\b(?:Codex|ChatGPT|OpenAI)\b", commit.body
        ):
            trailer_failure = True
            findings.append(
                Finding("FAIL", f"{short} contains AI author/co-author attribution")
            )
    if commits and not trailer_failure:
        findings.append(Finding("PASS", f"commit trailers checked: {len(commits)}"))

    workflow = repo / ".github/workflows/pre-checks.yaml"
    contribute = repo / "CONTRIBUTE.md"
    expected_ruff = version_from(r"pip3 install ruff==(\d+\.\d+\.\d+)", workflow)
    expected_black = version_from(r"black==(\d+\.\d+\.\d+)", contribute)
    python_files = [
        path for path in changed if path.endswith(".py") and (repo / path).is_file()
    ]

    if python_files:
        black_bin, black_version = installed_version("black", repo)
        if black_bin is None:
            findings.append(Finding("FAIL", "Black is not installed"))
        elif expected_black is None:
            findings.append(
                Finding("FAIL", "cannot derive the repository Black version pin")
            )
        elif expected_black and black_version != expected_black:
            findings.append(
                Finding(
                    "FAIL",
                    f"Black version {black_version} != repository pin {expected_black}",
                )
            )
        else:
            black_failed = False
            for path in python_files:
                result = run([black_bin, "--check", "--workers", "1", path], repo)
                if result.returncode:
                    black_failed = True
                    detail = (result.stdout + result.stderr).strip()
                    findings.append(
                        Finding("FAIL", f"Black failed for {path}: {detail}")
                    )
            if not black_failed:
                findings.append(
                    Finding(
                        "PASS",
                        f"Black checked {len(python_files)} changed Python files",
                    )
                )

        ruff_bin, ruff_version = installed_version("ruff", repo)
        if ruff_bin is None:
            findings.append(Finding("FAIL", "Ruff is not installed"))
        elif expected_ruff is None:
            findings.append(Finding("FAIL", "cannot derive the CI Ruff version pin"))
        elif expected_ruff and ruff_version != expected_ruff:
            findings.append(
                Finding(
                    "FAIL", f"Ruff version {ruff_version} != CI pin {expected_ruff}"
                )
            )
        else:
            result = run([ruff_bin, "check", "--no-cache", *python_files], repo)
            if result.returncode:
                detail = (result.stdout + result.stderr).strip()
                findings.append(Finding("FAIL", f"Ruff failed: {detail}"))
            else:
                findings.append(
                    Finding(
                        "PASS",
                        f"Ruff checked {len(python_files)} changed Python files",
                    )
                )
    else:
        findings.append(Finding("INFO", "no changed Python files; Black/Ruff skipped"))

    new_kernel_paths: list[str] = []
    new_jit_decorators = 0
    for path in changed:
        lines = added_lines(repo, base, path)
        joined = "\n".join(lines)
        if path.startswith("op_tests/"):
            if re.search(
                r"from\s+aiter\.ops\.triton\._(?:triton|gluon)_kernels",
                joined,
            ):
                findings.append(
                    Finding("FAIL", f"{path} imports a private Triton/Gluon kernel")
                )
            if re.search(r"\b_\w*kernel\s*\[", joined):
                findings.append(
                    Finding("FAIL", f"{path} directly launches a private kernel")
                )
            if path.startswith("op_tests/triton_tests/") and re.search(
                r"\bprint\s*\(", joined
            ):
                findings.append(Finding("FAIL", f"{path} adds print() to a unit test"))
            if path.startswith("op_tests/triton_tests/") and "__main__" in joined:
                findings.append(
                    Finding("FAIL", f"{path} adds an ad-hoc __main__ test block")
                )
        if path.startswith("aiter/ops/triton/") and re.search(
            r"\b(?:_LOGGER|logger)\.(?:debug|info|warning|error|exception)\(f[\"']",
            joined,
        ):
            findings.append(Finding("FAIL", f"{path} adds eager f-string logging"))
        if path.startswith("aiter/ops/triton/") and re.search(
            r"^\s*from\s+\.", joined, re.MULTILINE
        ):
            findings.append(Finding("FAIL", f"{path} adds a relative import"))
        is_internal_kernel = path.startswith(
            (
                "aiter/ops/triton/_triton_kernels/",
                "aiter/ops/triton/_gluon_kernels/",
            )
        )
        is_torch_free_module = (
            is_internal_kernel
            or path.startswith("aiter/ops/triton/utils/_triton/")
            or bool(
                re.fullmatch(
                    r"aiter/ops/triton/utils/(?:config_utils|\w+_config_utils)\.py",
                    path,
                )
            )
        )
        if is_torch_free_module and re.search(
            r"^\s*(?:import\s+torch|from\s+torch\s+import|.*\btorch\.)",
            joined,
            re.MULTILINE,
        ):
            findings.append(
                Finding("FAIL", f"{path} adds torch to a torch-free module")
            )
        decorators = re.findall(r"^\s*@(?:triton|gluon)\.jit", joined, re.MULTILINE)
        if is_internal_kernel and decorators:
            new_kernel_paths.append(path)
            new_jit_decorators += len(decorators)
            if "make_kernel_repr" not in (repo / path).read_text(encoding="utf-8"):
                findings.append(
                    Finding("FAIL", f"{path} adds a kernel without make_kernel_repr")
                )
        elif (
            path.startswith("aiter/ops/triton/")
            and not is_internal_kernel
            and decorators
        ):
            findings.append(
                Finding("FAIL", f"{path} defines a JIT kernel in a public wrapper")
            )
        if (
            path.startswith("aiter/ops/triton/")
            and not path.startswith("aiter/ops/triton/_")
            and re.search(
                r"(?:device\s*=\s*[\"']cuda[\"']|torch\.device\([\"']cuda[\"']\)|\.cuda\s*\()",
                joined,
            )
        ):
            findings.append(
                Finding("FAIL", f"{path} adds hard-coded CUDA device selection")
            )

    if new_kernel_paths:
        for kernel_path in new_kernel_paths:
            if "/_gluon_kernels/" in kernel_path:
                relative = kernel_path.split("/_gluon_kernels/", 1)[1]
                parts = relative.split("/")
                category = parts[1] if len(parts) > 2 else ""
            else:
                relative = kernel_path.split("/_triton_kernels/", 1)[1]
                category = relative.split("/", 1)[0]
            has_wrapper = any(
                path.startswith(f"aiter/ops/triton/{category}/")
                and path.endswith(".py")
                for path in changed
            )
            has_test = any(
                path.startswith(f"op_tests/triton_tests/{category}/")
                and path.endswith(".py")
                for path in changed
            )
            has_bench = any(
                path.startswith("op_tests/op_benchmarks/triton/")
                and path.endswith(".py")
                for path in changed
            )
            for present, label in (
                (has_wrapper, "matching public wrapper"),
                (has_test, "matching Triton unit test"),
                (has_bench, "Triton benchmark"),
            ):
                if not present:
                    findings.append(
                        Finding("FAIL", f"new kernel {relative} has no changed {label}")
                    )
        if new_jit_decorators > 1:
            findings.append(
                Finding(
                    "WARN",
                    f"{new_jit_decorators} new JIT decorators found; manually verify "
                    "that only one is launchable and the rest are device helpers",
                )
            )

    if "3rdparty/composable_kernel" in changed:
        findings.append(
            Finding("WARN", "CK submodule changed; run .github/scripts/check_deps.sh")
        )

    if not any(f.level == "FAIL" for f in findings):
        findings.append(Finding("PASS", "local preflight completed without failures"))
    return findings, changed


def online_audit(
    repo: Path, base: str, changed: list[str], pr_number: int | None
) -> list[Finding]:
    findings: list[Finding] = []
    gh_bin = shutil.which("gh")
    if gh_bin is None:
        return [Finding("FAIL", "gh is not installed")]

    base_branch = base.partition("/")[2] or base
    encoded_base = quote(base_branch, safe="")

    auth = run([gh_bin, "auth", "status", "--hostname", "github.com"], repo)
    if auth.returncode:
        detail = (auth.stdout + auth.stderr).strip()
        return [Finding("FAIL", f"GitHub CLI authentication failed: {detail}")]

    rules = run(
        [gh_bin, "api", f"repos/ROCm/aiter/rules/branches/{encoded_base}"], repo
    )
    if rules.returncode:
        detail = (rules.stdout + rules.stderr).strip()
        return [Finding("FAIL", f"cannot read effective rules: {detail}")]
    required_contexts: list[str] = []
    pull_request_rule: dict[str, object] = {}
    try:
        payload = json.loads(rules.stdout)
        for item in payload:
            if item.get("type") == "required_status_checks":
                required_contexts.extend(
                    check["context"]
                    for check in item["parameters"]["required_status_checks"]
                )
            elif item.get("type") == "pull_request":
                pull_request_rule = item.get("parameters", {})
        if not required_contexts:
            raise KeyError("no required_status_checks rule")
        findings.append(
            Finding("INFO", "required checks: " + ", ".join(required_contexts))
        )
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        findings.append(Finding("FAIL", f"cannot parse effective rules: {exc}"))

    if pr_number is None:
        detected = run(
            [gh_bin, "pr", "view", "--json", "number", "--jq", ".number"], repo
        )
        if detected.returncode == 0 and detected.stdout.strip().isdigit():
            pr_number = int(detected.stdout.strip())
    if pr_number is None:
        findings.append(Finding("FAIL", "no PR number supplied or detected"))
        return findings

    view = run(
        [
            gh_bin,
            "pr",
            "view",
            str(pr_number),
            "--repo",
            "ROCm/aiter",
            "--json",
            "number,state,isDraft,baseRefName,headRefName,headRepository,mergeable,mergeStateStatus,reviewDecision,commits,labels,url",
        ],
        repo,
    )
    if view.returncode:
        detail = (view.stdout + view.stderr).strip()
        findings.append(Finding("FAIL", f"cannot read PR #{pr_number}: {detail}"))
        return findings
    try:
        data = json.loads(view.stdout)
    except json.JSONDecodeError as exc:
        findings.append(Finding("FAIL", f"cannot parse PR #{pr_number}: {exc}"))
        return findings
    commit_nodes = data.get("commits") or []
    remote_head = commit_nodes[-1].get("oid") if commit_nodes else None
    local_head = git(repo, "rev-parse", "HEAD")
    if remote_head != local_head:
        findings.append(
            Finding(
                "FAIL",
                f"PR head {remote_head or 'unknown'} != local HEAD {local_head}",
            )
        )
    else:
        findings.append(Finding("PASS", f"PR #{pr_number} matches local HEAD"))

    if data.get("baseRefName") != base_branch:
        findings.append(
            Finding(
                "FAIL",
                f"PR base {data.get('baseRefName')} != requested base {base_branch}",
            )
        )
    else:
        findings.append(Finding("PASS", f"PR targets {base_branch}"))

    live_base = run(
        [gh_bin, "api", f"repos/ROCm/aiter/git/ref/heads/{encoded_base}"], repo
    )
    if live_base.returncode:
        detail = (live_base.stdout + live_base.stderr).strip()
        findings.append(Finding("FAIL", f"cannot read live {base_branch}: {detail}"))
    else:
        try:
            live_base_sha = json.loads(live_base.stdout)["object"]["sha"]
            local_base_sha = git(repo, "rev-parse", base)
            if live_base_sha != local_base_sha:
                findings.append(
                    Finding(
                        "FAIL",
                        f"local {base} {local_base_sha} != live "
                        f"ROCm/aiter/{base_branch} {live_base_sha}; fetch first",
                    )
                )
            else:
                findings.append(
                    Finding("PASS", f"local {base} matches live {base_branch}")
                )
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            findings.append(Finding("FAIL", f"cannot parse live base ref: {exc}"))

    if data.get("state") != "OPEN":
        findings.append(
            Finding("FAIL", f"PR state is {data.get('state')} instead of OPEN")
        )
    if data.get("isDraft"):
        findings.append(Finding("FAIL", "PR is still a draft"))
    if data.get("mergeable") == "CONFLICTING":
        findings.append(Finding("FAIL", "PR has merge conflicts"))
    elif data.get("mergeable") == "UNKNOWN":
        findings.append(Finding("WARN", "GitHub has not determined mergeability yet"))

    merge_state = data.get("mergeStateStatus") or "UNKNOWN"
    if merge_state != "CLEAN":
        findings.append(
            Finding(
                "FAIL",
                f"GitHub merge state is {merge_state}; branch rules are not satisfied",
            )
        )

    required_approvals = int(
        pull_request_rule.get("required_approving_review_count", 0) or 0
    )
    review_decision = data.get("reviewDecision") or "NONE"
    if required_approvals and review_decision != "APPROVED":
        findings.append(
            Finding(
                "FAIL",
                f"review decision is {review_decision}; rules require approval",
            )
        )
    elif required_approvals:
        findings.append(Finding("PASS", "required review decision is APPROVED"))

    findings.append(
        Finding(
            "INFO",
            "PR state={state} mergeable={mergeable} mergeState={merge_state} "
            "review={review}".format(
                state=data.get("state"),
                mergeable=data.get("mergeable"),
                merge_state=data.get("mergeStateStatus"),
                review=data.get("reviewDecision") or "NONE",
            ),
        )
    )

    if remote_head:
        observed: dict[str, tuple[str, str]] = {}
        check_runs = run(
            [
                gh_bin,
                "api",
                f"repos/ROCm/aiter/commits/{remote_head}/check-runs?filter=latest&per_page=100",
            ],
            repo,
        )
        commit_status = run(
            [gh_bin, "api", f"repos/ROCm/aiter/commits/{remote_head}/status"],
            repo,
        )
        if check_runs.returncode or commit_status.returncode:
            detail = (
                check_runs.stdout
                + check_runs.stderr
                + commit_status.stdout
                + commit_status.stderr
            ).strip()
            findings.append(
                Finding("FAIL", f"cannot read checks for PR head: {detail}")
            )
        else:
            try:
                check_payload = json.loads(check_runs.stdout)
                status_payload = json.loads(commit_status.stdout)
                if check_payload.get("total_count", 0) > len(
                    check_payload.get("check_runs", [])
                ):
                    findings.append(
                        Finding(
                            "FAIL",
                            "more than 100 check runs exist; check audit is incomplete",
                        )
                    )
                for item in check_payload.get("check_runs", []):
                    observed[item["name"]] = (
                        item.get("status") or "unknown",
                        item.get("conclusion") or "pending",
                    )
                for item in status_payload.get("statuses", []):
                    observed.setdefault(
                        item["context"],
                        ("completed", item.get("state") or "unknown"),
                    )

                required_failures = 0
                for context in required_contexts:
                    state = observed.get(context)
                    if state is None:
                        required_failures += 1
                        findings.append(
                            Finding("FAIL", f"required check is missing: {context}")
                        )
                        continue
                    status, conclusion = state
                    if status != "completed" or conclusion != "success":
                        required_failures += 1
                        findings.append(
                            Finding(
                                "FAIL",
                                f"required check {context}: "
                                f"status={status} conclusion={conclusion}",
                            )
                        )
                    else:
                        findings.append(Finding("PASS", f"required check: {context}"))
                if required_contexts and not required_failures:
                    findings.append(Finding("PASS", "all live required checks passed"))
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                findings.append(Finding("FAIL", f"cannot parse PR checks: {exc}"))

        workflow_runs = run(
            [
                gh_bin,
                "api",
                f"repos/ROCm/aiter/actions/runs?head_sha={remote_head}&per_page=100",
            ],
            repo,
        )
        if workflow_runs.returncode:
            detail = (workflow_runs.stdout + workflow_runs.stderr).strip()
            findings.append(Finding("FAIL", f"cannot read workflow runs: {detail}"))
        else:
            try:
                run_payload = json.loads(workflow_runs.stdout)
                if run_payload.get("total_count", 0) > len(
                    run_payload.get("workflow_runs", [])
                ):
                    findings.append(
                        Finding(
                            "FAIL",
                            "more than 100 workflow runs exist; workflow audit is incomplete",
                        )
                    )
                candidates: dict[str, list[dict[str, object]]] = {}
                expected_head_ref = data.get("headRefName")
                expected_head_repo = (data.get("headRepository") or {}).get(
                    "nameWithOwner"
                )
                for item in run_payload.get("workflow_runs", []):
                    if item.get("event") not in {"pull_request", "pull_request_target"}:
                        continue
                    run_head_repo = (item.get("head_repository") or {}).get("full_name")
                    if (
                        item.get("head_branch") != expected_head_ref
                        or run_head_repo != expected_head_repo
                    ):
                        continue
                    associated_prs = item.get("pull_requests") or []
                    if associated_prs and not any(
                        pr.get("number") == pr_number for pr in associated_prs
                    ):
                        continue
                    name = item.get("name") or f"run {item.get('id')}"
                    candidates.setdefault(str(name), []).append(item)
                runs_by_name = {
                    name: max(
                        items,
                        key=lambda item: (
                            str(item.get("created_at") or ""),
                            int(item.get("run_attempt") or 0),
                            int(item.get("id") or 0),
                        ),
                    )
                    for name, items in candidates.items()
                }
                labels = {
                    item.get("name")
                    for item in data.get("labels", [])
                    if item.get("name")
                }
                documentation_only = bool(changed) and all(
                    path.endswith(".md")
                    or path.startswith("docs/")
                    or path in {"LICENSE", ".gitignore"}
                    for path in changed
                )
                expected_workflows = {
                    "Aiter Test",
                    "Checks",
                    "PR Title Tags & Labels",
                }
                if not documentation_only:
                    expected_workflows.add("OPUS Test")
                triton_paths = {
                    ".github/requirements/triton-test.txt",
                    ".github/scripts/build_aiter_triton.sh",
                    ".github/scripts/download_triton_wheel.sh",
                    ".github/scripts/install_triton.sh",
                    ".github/scripts/select_triton_tests.py",
                    ".github/scripts/split_tests.sh",
                    ".github/scripts/verify_triton_pin.py",
                    ".github/workflows/ci-config.yaml",
                    ".github/workflows/prepare-triton-wheel.yaml",
                    ".github/workflows/triton-test.yaml",
                }
                if any(
                    (
                        path.startswith(
                            (
                                "aiter/ops/triton/",
                                "op_tests/triton_tests/",
                                "op_tests/op_benchmarks/triton/",
                            )
                        )
                        and not path.endswith(".md")
                    )
                    or path in triton_paths
                    for path in changed
                ):
                    expected_workflows.add("Triton Test")
                if any(
                    path.startswith(".github/workflows/")
                    or path == ".github/actionlint.yaml"
                    for path in changed
                ):
                    expected_workflows.add("Actionlint")
                if any(
                    path.startswith("docs/") or path == ".github/workflows/docs.yml"
                    for path in changed
                ):
                    expected_workflows.add("Build and Deploy Documentation")
                flash_paths = (
                    "aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/",
                    "aiter/ops/triton/attention/mha_v3.py",
                    "aiter/ops/triton/__init__.py",
                    "aiter/ops/mha.py",
                    "csrc/py_itfs_ck/mha_",
                    "csrc/py_itfs_ck/attention_kernels.cu",
                    ".github/workflows/flash_attention_integration.yaml",
                    ".github/workflows/ci-config.yaml",
                )
                if any(
                    path.startswith(flash_paths) or path == "setup.py"
                    for path in changed
                ):
                    expected_workflows.add("Flash Attention Integration")
                monitor_paths = {
                    ".github/workflows/amd-ci-job-monitor.yml",
                    ".github/scripts/list_jobs.py",
                    ".github/scripts/query_job_status.py",
                    ".github/runner-config.yml",
                }
                if monitor_paths.intersection(changed):
                    expected_workflows.add("AMD CI Job Monitor")

                label_workflows = {
                    "ci:atom": {"Atom Test"},
                    "ci:atom_full": {"Atom Test"},
                    "ci:sglang": {"Sglang Downstream Test", "Kimi Downstream Test"},
                    "ci:kimi": {"Kimi Downstream Test"},
                    "ci:performance": {"Kimi Perf Downstream"},
                    "ci:vllm": {"vLLM Benchmark", "Kimi Downstream Test"},
                    "ci:vllm-di": {"vLLM disagg CI smoke workflow"},
                    "ci:extended-test": {"Extended test for PR"},
                    "ci:gfx1250-ffm-triton": {"FFM Triton Tests"},
                    "ci:all": {
                        "Atom Test",
                        "Kimi Downstream Test",
                        "Kimi Perf Downstream",
                        "Sglang Downstream Test",
                        "vLLM Benchmark",
                    },
                }
                for label in labels:
                    expected_workflows.update(label_workflows.get(str(label), set()))

                for name in sorted(expected_workflows):
                    item = runs_by_name.get(name)
                    if item is None:
                        findings.append(
                            Finding("FAIL", f"expected workflow is missing: {name}")
                        )
                        continue
                    status = item.get("status")
                    conclusion = item.get("conclusion")
                    if status != "completed" or conclusion != "success":
                        findings.append(
                            Finding(
                                "FAIL",
                                f"expected workflow {name}: status={status} "
                                f"conclusion={conclusion or 'pending'}",
                            )
                        )
                    else:
                        findings.append(Finding("PASS", f"workflow: {name}"))

                required_job_contexts = {
                    "OPUS Test": {"OPUS Tests (MI35X)", "OPUS Tests (MI300X)"},
                    "PR Title Tags & Labels": {"tag-title"},
                    "Triton Test": {"Triton Test Results"},
                }
                if "ci:triton-300x" in labels:
                    required_job_contexts["Triton Test"].add(
                        "Triton MI300X Test Results"
                    )
                for workflow_name in sorted(expected_workflows):
                    workflow_run = runs_by_name.get(workflow_name)
                    if (
                        workflow_run is None
                        or workflow_run.get("status") != "completed"
                        or workflow_run.get("conclusion") != "success"
                    ):
                        continue
                    for context in sorted(
                        required_job_contexts.get(workflow_name, set())
                    ):
                        state = observed.get(context)
                        if state is None:
                            findings.append(
                                Finding(
                                    "FAIL",
                                    f"{workflow_name} did not produce expected job: {context}",
                                )
                            )
                            continue
                        status, conclusion = state
                        if status != "completed" or conclusion != "success":
                            findings.append(
                                Finding(
                                    "FAIL",
                                    f"{workflow_name} job {context}: "
                                    f"status={status} conclusion={conclusion}",
                                )
                            )

                job_audit_workflows = expected_workflows - {
                    "Aiter Test",
                    "Checks",
                    *required_job_contexts,
                }
                for workflow_name in sorted(job_audit_workflows):
                    item = runs_by_name.get(workflow_name)
                    if (
                        item is None
                        or item.get("status") != "completed"
                        or item.get("conclusion") != "success"
                    ):
                        continue
                    run_id = item.get("id")
                    jobs = run(
                        [
                            gh_bin,
                            "api",
                            f"repos/ROCm/aiter/actions/runs/{run_id}/jobs?filter=latest&per_page=100",
                        ],
                        repo,
                    )
                    if jobs.returncode:
                        detail = (jobs.stdout + jobs.stderr).strip()
                        findings.append(
                            Finding(
                                "FAIL",
                                f"cannot inspect jobs for {workflow_name}: {detail}",
                            )
                        )
                        continue
                    try:
                        job_nodes = json.loads(jobs.stdout).get("jobs", [])
                        if not any(
                            job.get("conclusion") == "success" for job in job_nodes
                        ):
                            findings.append(
                                Finding(
                                    "FAIL",
                                    f"workflow {workflow_name} succeeded without "
                                    "an executed successful job",
                                )
                            )
                    except (TypeError, json.JSONDecodeError) as exc:
                        findings.append(
                            Finding(
                                "FAIL",
                                f"cannot parse jobs for {workflow_name}: {exc}",
                            )
                        )

                blocked_runs = []
                pending_runs = []
                failed_runs = []
                skipped_runs = []
                for name, item in runs_by_name.items():
                    status = item.get("status")
                    conclusion = item.get("conclusion")
                    if name in expected_workflows:
                        continue
                    if conclusion == "action_required":
                        blocked_runs.append(name)
                    elif status != "completed":
                        pending_runs.append(name)
                    elif conclusion in {
                        "failure",
                        "cancelled",
                        "timed_out",
                        "startup_failure",
                        "stale",
                    }:
                        failed_runs.append(f"{name}={conclusion}")
                    elif conclusion == "skipped":
                        skipped_runs.append(name)
                if blocked_runs:
                    findings.append(
                        Finding(
                            "FAIL",
                            "workflow approval required: "
                            + ", ".join(sorted(set(blocked_runs))),
                        )
                    )
                if pending_runs:
                    findings.append(
                        Finding(
                            "FAIL",
                            "workflow runs still pending: "
                            + ", ".join(sorted(set(pending_runs))),
                        )
                    )
                if failed_runs:
                    findings.append(
                        Finding(
                            "FAIL",
                            "workflow runs failed: "
                            + ", ".join(sorted(set(failed_runs))),
                        )
                    )
                if skipped_runs:
                    findings.append(
                        Finding(
                            "INFO",
                            "skipped workflows (not counted as passing): "
                            + ", ".join(sorted(set(skipped_runs))),
                        )
                    )
            except (TypeError, json.JSONDecodeError) as exc:
                findings.append(Finding("FAIL", f"cannot parse workflow runs: {exc}"))

    if pull_request_rule.get("required_review_thread_resolution"):
        query = """
query($owner: String!, $name: String!, $number: Int!) {
  repository(owner: $owner, name: $name) {
    pullRequest(number: $number) {
      reviewThreads(first: 100) {
        nodes { isResolved comments(first: 1) { nodes { url } } }
        pageInfo { hasNextPage }
      }
    }
  }
}
""".strip()
        threads = run(
            [
                gh_bin,
                "api",
                "graphql",
                "-F",
                "owner=ROCm",
                "-F",
                "name=aiter",
                "-F",
                f"number={pr_number}",
                "-f",
                f"query={query}",
            ],
            repo,
        )
        if threads.returncode:
            detail = (threads.stdout + threads.stderr).strip()
            findings.append(Finding("FAIL", f"cannot read review threads: {detail}"))
        else:
            try:
                thread_data = json.loads(threads.stdout)["data"]["repository"][
                    "pullRequest"
                ]["reviewThreads"]
                unresolved = [
                    node for node in thread_data["nodes"] if not node["isResolved"]
                ]
                if unresolved:
                    urls = [
                        node["comments"]["nodes"][0]["url"]
                        for node in unresolved
                        if node["comments"]["nodes"]
                    ]
                    findings.append(
                        Finding(
                            "FAIL",
                            f"unresolved review threads: {len(unresolved)}"
                            + (" (" + ", ".join(urls) + ")" if urls else ""),
                        )
                    )
                else:
                    findings.append(Finding("PASS", "all review threads are resolved"))
                if thread_data["pageInfo"]["hasNextPage"]:
                    findings.append(
                        Finding(
                            "FAIL",
                            "more than 100 review threads; audit pagination is incomplete",
                        )
                    )
            except (KeyError, TypeError, json.JSONDecodeError) as exc:
                findings.append(Finding("FAIL", f"cannot parse review threads: {exc}"))
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base", default="upstream/main")
    parser.add_argument("--online", action="store_true")
    parser.add_argument("--pr", type=int)
    args = parser.parse_args()

    if args.pr is not None and not args.online:
        parser.error("--pr requires --online")

    repo = args.repo.expanduser().resolve()
    findings, changed = local_preflight(repo, args.base)
    if args.online:
        findings.extend(online_audit(repo, args.base, changed, args.pr))

    for finding in findings:
        print(f"[{finding.level}] {finding.message}")
    if changed:
        print("[INFO] changed paths:")
        for path in changed:
            print(f"  - {path}")

    return 1 if any(f.level == "FAIL" for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
