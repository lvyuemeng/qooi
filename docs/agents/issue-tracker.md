# Issue tracker — local markdown

Issues are stored as markdown files under `.scratch/<feature-name>/` in this repo, one directory per feature area.

## Convention

```
.scratch/<feature-name>/
  ├── 001-short-title.md
  ├── 002-another-issue.md
  └── ...
```

Each issue file is a flat markdown document with YAML front matter:

```markdown
---
title: Short issue title
status: needs-triage
created: 2026-05-03
---

Description of the issue.
```

## Status labels

The `status` field in front matter maps to one of the five triage roles. See `docs/agents/triage-labels.md`.

## Tooling

- Create: write a new `.md` file in the appropriate `.scratch/<feature>/` directory
- List: `ls .scratch/**/*.md`
- Read: `cat` or open the file
- Update: edit the file directly

No `gh` CLI needed — this is purely file-based.
