# Git Workflow

This repository uses Git to preserve reviewable engineering decisions, not as a
runtime variant database or a long-term workspace. The goal is one integration
line, short-lived isolation where useful, and durable experiment provenance
without retaining rejected source in Production.

This file supplements `../rules/github.md`. The parent rule owns commit-message
syntax; this file owns repository branch, integration, and archive policy.

## Repository Shape

- `main` is the only permanent integration branch. A commit on `main` must meet
  the applicable validation level in `docs/change_policy.md`.
- Use a branch only when review, concurrent work, remote execution, or an
  isolated experiment makes it useful. Small sequential changes may be committed
  directly to `main` when the user requests commits.
- Short-lived branch names use `fix/<area>/<topic>`, `feat/<area>/<topic>`,
  `opt/<operator>/<feature>`, or `exp/<operator>/<hypothesis>`.
- Do not create permanent WIP, machine, agent, parameter, `v1/v2/v3`, or
  architecture-integration branches. Architecture is a source ownership
  boundary, not a reason for a second integration line.
- A branch name or branch head is never the feature/variant registry. Operator
  manifests record active and retired optimization facts.

Existing long-lived WIP, backup, and generated worktree branches are migration
backlog. Do not add unrelated work to them. Review their unique commits once,
integrate the decisions that still belong on `main`, preserve any branch-only
rejected experiment as described below, then remove the branch when explicitly
authorized. This policy does not itself delete or rewrite any existing ref.

## Commit Boundaries

Each commit represents one coherent decision and is independently reviewable:

- Keep implementation, focused tests, contract/schema updates, manifest state,
  and concise evidence for the same decision together when separating them would
  leave an invalid intermediate revision.
- Separate mechanical refactors from behavior changes; separate unrelated
  operators and independent optimization hypotheses.
- A generated calibration update may be its own commit when its schema and
  consumer are already present. Record the source commit and measurement
  environment in the calibration or result document.
- Do not split commits by file type merely to make them small, and do not combine
  several decisions merely because they were tested in one benchmark run.
- Keep raw traces, build products, caches, large profiler output, and temporary
  remote-sync files out of commits.

Before every commit:

1. inspect `git status --short` and preserve unrelated work;
2. stage explicit paths, never the entire dirty tree by habit;
3. run the required validation from `docs/change_policy.md`;
4. inspect `git diff --cached --check`, `git diff --cached --stat`, and the exact
   staged diff;
5. use the `<type>: <subject>` format from `../rules/github.md`.

Do not use `git stash` as durable experiment storage. Commit owned work on a
short-lived branch or leave clearly identified user changes untouched.

## Integration

- Integrate a short-lived branch once its decision is complete; do not repeatedly
  merge `main` and the work branch in both directions.
- Before integrating divergent work, inspect unique commits with a left/right
  or cherry-equivalence log and review the patch for each decision. Apply only
  coherent commits; do not merge a historical WIP branch wholesale merely to
  make branch tips equal.
- Prefer a fast-forward or a clean application of reviewed commits for local,
  unshared work. Preserve an existing reviewed merge when its topology carries
  useful collaboration context.
- Never rewrite shared history or force-push. Correct an integrated regression
  with a focused `revert` or follow-up fix that names the failed assumption.
- Committing, pushing, rebasing, branch deletion, tagging, or other repository
  mutation still requires the authorization stated in `AGENTS.md`.

## Experiments And Archives

Accepted experiments become ordinary reviewed commits on `main`; update their
manifest status and default/fallback evidence in the same decision.

For a rejected, superseded, or neutral experiment:

1. record correctness, benchmark method, result, and retirement reason in the
   operator manifest or a concise result document;
2. remove its source and default-build entrypoint from the active tree;
3. point the tombstone at a commit already reachable from `main` when possible;
4. if the runnable implementation exists only on a disposable branch, create an
   annotated `archive/<operator>/<feature>/<YYYYMMDD>` tag at its final validated
   commit before deleting that branch;
5. record that commit or tag in the tombstone and keep the reproduction command.

Do not create an archive tag for every benchmark or parameter sweep. A plain
commit id is sufficient when it remains reachable from `main`; a tag exists only
to keep otherwise unreachable branch-only source durable. Do not copy retired
source into an archive directory.

## Worktrees And Agents

- Concurrent workers use separate worktrees and separate short-lived branches.
  They must not edit or commit through the same branch simultaneously.
- Generated `worktree-*` branches are local implementation details. Do not push
  them or treat them as durable ownership boundaries.
- Sync source to remote benchmark machines from a known commit plus explicitly
  reported local changes. Remote copies are execution workspaces, not Git
  authorities.
- After integration, remove temporary worktrees and branches only when their
  unique commits and untracked files have been audited and deletion is
  explicitly authorized.

## Legacy Branch Convergence

To retire an existing long-lived branch without losing decisions:

1. compare it to `main` using commit equivalence, not only hashes;
2. classify each unique commit as integrate, already represented, rejected, or
   obsolete;
3. apply and validate the integrate set as focused commits on `main`;
4. add manifest/result provenance for rejected branch-only experiments and tag
   only those tips that would otherwise become unreachable;
5. verify no required commit or untracked artifact remains;
6. delete local and remote legacy refs only after explicit authorization.

Do not keep merging the converged branch back into `main`. Once audited, `main`
is the continuing integration line.
