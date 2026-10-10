---
name: babysit-ci
description: Use when asked to watch or fix CI on a PR or branch. Fixes mechanical failures, retriggers infrastructure flakes, and stops to report anything that needs judgment.
---

# Babysit CI

Monitor GitHub Actions CI for the current branch/PR. Auto-fix small issues (whitespace, missing
imports, formatting). Pause and describe anything that requires judgment. Retrigger flaky jobs.

## Step 1: Find the CI Run

```sh
# Get the current branch
git branch --show-current

# Find the latest CI run for this branch
gh run list --branch $(git branch --show-current) --limit 5

# Or for a specific PR
gh pr checks <PR_NUMBER>
```

## Step 2: Monitor Loop

Check CI status. For each failed job:

```sh
gh run view <RUN_ID>
gh run view <RUN_ID> --log-failed
```

## Step 3: Triage Each Failure

### Auto-fix (commit and push without asking)

| Failure | How to fix |
|---------|-----------|
| **Whitespace check** | Remove trailing whitespace, ensure final newline, no trailing blank lines |
| **Missing explicit import** | Add the missing import to the appropriate file |
| **Doctest output mismatch** | Update the expected output only if the PR intentionally changed what is printed; otherwise pause |
| **Typo in error message or docstring** | Fix the typo |
| **Unused import warning** | Remove the unused import |

After fixing, commit with a descriptive message and push. Continue monitoring.

### Retrigger (likely flaky)

"Unrelated to the changed files" is not enough on its own: show the log line that points to
infrastructure, or the same failure on `main`.

| Signal | Action |
|--------|--------|
| Test passed locally but fails in CI | Retrigger |
| Timeout with no test failure | Retrigger |
| Network/download error | Retrigger |
| `Pkg.instantiate` failure | Retrigger |
| CI infrastructure error | Retrigger |
| The same job fails the same way on `main` | Retrigger, and say it also fails on `main` |

```sh
gh run rerun <RUN_ID> --failed
# Or push an empty commit to retrigger all jobs
git commit --allow-empty -m "Retrigger CI"
git push
```

### Pause and describe (needs judgment)

Stop and explain the problem to the user for:

- **Test logic failure**: assertion fails that isn't a doctest output update
- **Type instability or GPU error**: "dynamic invocation" or similar
- **Regression test failure**: numerical results don't match
- **Build failure**: package won't precompile or load
- **Multiple interrelated failures**: several jobs with a common root cause

Report: which job(s) failed, relevant error output, likely cause, suggested fix options.

## Step 4: Confirm All Green

```sh
gh pr checks <PR_NUMBER>
```

Report the final status to the user.

## Notes

- Always check `gh run view <RUN_ID> --log-failed` before acting — don't guess from job names
- After pushing a fix, wait for CI to pick it up before checking again
- Never force-push or rewrite history to fix CI — always add new commits
- GPU jobs run on self-hosted `aws-linux-nvidia-gpu-*` runners (tests in `CI.yml`, docs in
  `Documentation.yml`); `Validation.yml` runs the BOMEX example
