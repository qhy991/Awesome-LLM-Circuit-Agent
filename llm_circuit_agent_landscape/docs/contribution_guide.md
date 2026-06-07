# Contributing to the Landscape

The landscape is **data-driven**: YAML in `data/` is the source of truth, and the
Mermaid blocks in the root `README.md` are generated. Never edit the Mermaid
blocks by hand — they are overwritten by `make sync-readme`.

## Add a paper

1. Add a row to the relevant table in the root `../README.md` (keep newest first).
2. Refresh the canonical data:

   ```bash
   make import      # parse README tables -> data/entries.yaml
   ```

   Or add the entry manually to `data/entries.yaml`:

   ```yaml
   - id: my_paper
     name: MyPaper
     year: 2026
     period: "2026+"
     category: Digital-CodeGen      # see data/categories.yaml
     subcategory: "RTL Generation, RAG"
     method_tags: [rtl, rag]
     url: https://arxiv.org/abs/XXXX.XXXXX
     source: awesome_llm_circuit_agent
     show_in_report: false
     show_in_full: true
     timeline_date: "2026-03"        # YYYY-MM
   ```

3. (Optional) Feature it in the compact view — `data/timeline.yaml`:

   ```yaml
   - date: "2026-03"
     items:
       - label: "MyPaper"
   ```

4. Regenerate and verify:

   ```bash
   make all
   ```

## Categories

Defined in `data/categories.yaml`. The category map and the SVG layout both read
from it. Add a new category there before referencing it in `entries.yaml`.

## Three views

| View | Source | README location |
|:-----|:-------|:----------------|
| Report timeline | `timeline.yaml` | Main landscape section |
| Full timeline | `entries.yaml` + `timeline_dates.yaml` | `<details>` collapsible |
| Category map | `entries.yaml` × categories | `<details>` collapsible |
