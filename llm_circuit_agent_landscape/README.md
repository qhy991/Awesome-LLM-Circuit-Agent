# LLM Circuit Agent Landscape Maintainer

Data-driven toolkit for maintaining the **Awesome-LLM-Circuit-Agent** field map as
**Mermaid diagrams** synced into the root `README.md`.

```text
data/entries.yaml       ─┐
data/timeline.yaml       ├── validate → render_mermaid → sync_readme → README.md
data/timeline_dates.yaml ─┘
```

YAML is the source of truth; Mermaid blocks in the README are generated — do not
edit them by hand.

## What is included

```text
.
├── data/
│   ├── entries.yaml          # canonical papers / systems (imported from README)
│   ├── timeline.yaml         # curated report timeline buckets
│   ├── timeline_dates.yaml   # full-timeline date defaults + overrides
│   ├── mermaid_config.yaml   # README markers & diagram titles
│   ├── categories.yaml       # taxonomy (also used by category flowchart)
│   ├── sources.yaml
│   └── changelog.yaml
├── scripts/
│   ├── validate_entries.py
│   ├── render_mermaid.py     # YAML → .mmd
│   ├── sync_readme.py        # .mmd → README markers
│   ├── mermaid_lib.py
│   └── import_from_readme.py # README tables → entries.yaml
├── figures/                  # generated .mmd
└── docs/
```

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make all
```

CI runs at repo root: `.github/workflows/render-landscape.yml` (validates + syncs
README on push). It also fails if the README is out of sync.

## Common commands

| Command | Action |
|:--------|:-------|
| `make import` | Re-parse `../README.md` tables into `data/entries.yaml` |
| `make validate` | Check entries + timeline references |
| `make mermaid` | Regenerate `.mmd` only |
| `make sync-readme` | Mermaid + patch the root `README.md` |
| `make all` | validate + mermaid + sync-readme |

## Three views

| View | Source | README location |
|:-----|:-------|:----------------|
| **Report timeline** | `timeline.yaml` | Main landscape section |
| **Full timeline** | `entries.yaml` + `timeline_dates.yaml` | `<details>` collapsible |
| **Category map** | `entries.yaml` × categories | `<details>` collapsible |

See [`docs/contribution_guide.md`](docs/contribution_guide.md) for how to add a paper.
