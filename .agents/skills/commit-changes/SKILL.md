---
name: commit-changes
description: Stage and commit repository changes in focused groups using this repo's existing commit style. Use when the user asks Codex to commit, create a commit, split changes into commits, or prepare staged changes while avoiding unrelated files.
---

# Commit Changes

Use this skill from the repository root when committing changes for this repo.

## Inspect First

- Run `git status --short` before staging.
- Review recent style with `git log --oneline -20` when the message style is not fresh.
- Inspect unstaged and staged changes with `git diff` and `git diff --cached`.
- Treat unrelated untracked files as user-owned unless the user explicitly asks to include them.

## Stage Carefully

- Do not use `git add .`.
- Stage only files that belong to the requested change.
- Use explicit pathspecs, for example:

```bash
git add -- AGENTS.md .agents/skills/commit-changes/SKILL.md
```

- Use `git add -p` only when a file contains mixed unrelated edits and the split is needed.
- Re-run `git status --short` after staging and confirm unrelated files remain unstaged.

## Commit Message Style

- Follow the existing history: short English imperative subject, capitalized verb, no trailing period.
- Prefer messages like:
  - `Implement gfx1200 texture sampling instructions`
  - `Split rdna_translator into modules`
  - `Move Codex skill to agents directory`
  - `Add development environment setup`
- Use a one-line subject unless the change genuinely needs explanatory body text.
- Keep the message specific to the staged change; avoid generic subjects like `Update files`.

## Commit

- Before committing, confirm `git diff --cached --stat` matches the intended scope.
- Commit with:

```bash
git commit -m "<Subject>"
```

- After committing, report the new commit hash and subject.
- Mention any remaining untracked or modified files that were intentionally left out.
