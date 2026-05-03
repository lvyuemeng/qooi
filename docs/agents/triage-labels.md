# Triage labels

Five canonical roles — all use the default string as both label name and `status` front-matter value.

| Role | Label string | Meaning |
|------|-------------|---------|
| needs-triage | `needs-triage` | Maintainer needs to evaluate |
| needs-info | `needs-info` | Waiting on reporter for more details |
| ready-for-agent | `ready-for-agent` | Fully specified, AFK-ready (agent can pick up with no human context) |
| ready-for-human | `ready-for-human` | Needs human implementation |
| wontfix | `wontfix` | Will not be actioned |

## Usage in local-markdown

In `.scratch/<feature>/NNN-title.md`, the `status` field of the YAML front matter holds one of these strings:

```yaml
---
title: Something is broken
status: needs-triage
---
```
