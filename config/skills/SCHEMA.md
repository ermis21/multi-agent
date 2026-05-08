# Phoebe SKILL.md schema

Phoebe's skill discovery (`app/prompt_generator.py:_discover_skills`) parses
YAML frontmatter from each `config/skills/<name>/SKILL.md`. It complies with
the Anthropic Agent Skills spec for the canonical fields and extends it with
two optional Phoebe-specific fields used by the progressive-disclosure
render in the system prompt.

## Frontmatter fields

| Field | Required | Spec status | Notes |
|---|---|---|---|
| `name` | yes | canonical | kebab-case (e.g. `log-triage`); used as the skill identifier and as the folder name fallback |
| `description` | yes | canonical | one-paragraph summary, ≤ 1024 chars |
| `when-to-trigger` / `when_to_trigger` | no | **Phoebe extension** | trigger conditions; multi-line allowed |
| `when-not-to-trigger` / `when_not_to_trigger` | no | **Phoebe extension** | anti-trigger conditions; multi-line allowed |
| `license` | no | optional metadata | informational only; not parsed by Phoebe |
| `allowed-tools` | no | optional metadata | informational only; Phoebe enforces `allowed_tools` via `config/agents.yaml` per role |

Hyphenated keys are accepted alongside their underscore aliases for
compatibility with skills authored against the canonical Anthropic spec.

## Minimal example

```markdown
---
name: log-triage
description: Triage application logs to surface user-facing errors and the most recent stack traces.
when-to-trigger: User asks about errors, failures, traces, or 'why X is broken'.
when-not-to-trigger: User asks for routine status or non-error log volume metrics.
---

## When to use this skill

When a user reports a failing run...
```

## Discovery roots

Skills are scanned from two directories (Phoebe-internal first; project
skills win on name collision):

1. `config/skills/<name>/SKILL.md` — primary, edited via `skill_builder` agent or `skill_install` MCP tool
2. `/config/host_skills/<name>/SKILL.md` — optional read-only mount from `~/.agents/skills` on the host

## Bundled scripts

A SKILL.md can ship with sibling files (Python scripts, templates, supporting
markdown) in the same directory. The `skill_install` MCP tool fetches every
sibling with extension in `{.md, .py, .sh, .txt, .yaml, .yml, .json}`. Other
file types are skipped to keep the install scope tight.

## Cache invalidation

Skill discovery is cached on `config/skills/` directory mtime. When you write
or replace a SKILL.md (via `file_write` or `skill_install`), touch the
parent directory or the cache will not refresh until the next process
restart. `skill_install` does this automatically; manual edits via
`file_write` already invalidate because writing to a child changes the parent
mtime on most filesystems.
