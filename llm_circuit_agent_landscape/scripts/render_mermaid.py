#!/usr/bin/env python3
"""Generate Mermaid diagrams from landscape YAML data (EN + ZH).

Usage:
    python scripts/render_mermaid.py
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
sys.path.insert(0, str(ROOT / "scripts"))

from mermaid_lib import (  # noqa: E402
    build_entries_by_id,
    entry_label,
    group_labels_by_date,
    load_landscape_data,
    render_timeline,
    resolve_entry_date,
    resolve_item_label,
    sanitize_mermaid_text,
)

LOCALES = ("en", "zh")


def timeline_title(cfg: dict[str, Any], key: str, locale: str) -> str:
    block = cfg["timelines"][key]
    return block[f"title_{locale}"]


def build_report_timeline(data: dict[str, Any], locale: str) -> str:
    entries_by_id = build_entries_by_id(data["entries"])
    buckets: list[tuple[str, list[str]]] = []

    for bucket in data["timeline"].get("buckets", []):
        date = bucket["date"]
        labels = [
            resolve_item_label(item, entries_by_id)
            for item in bucket.get("items", [])
        ]
        buckets.append((date, labels))

    title = timeline_title(data["config"], "report", locale)
    return render_timeline(title, buckets)


def build_full_timeline(data: dict[str, Any], locale: str) -> str:
    entries_by_id = build_entries_by_id(data["entries"])
    dates_cfg = data["timeline_dates"]
    period_defaults = dates_cfg.get("period_defaults", {})
    overrides = dates_cfg.get("overrides", {})

    items: list[tuple[str, str]] = []
    for entry in data["entries"]:
        if not entry.get("show_in_full", True):
            continue
        date = resolve_entry_date(entry, period_defaults, overrides)
        items.append((date, entry_label(entry)))

    if data["config"]["timelines"]["full"].get("include_external", True):
        for ext in dates_cfg.get("external", []):
            items.append((ext["date"], sanitize_mermaid_text(ext["label"])))

    buckets = group_labels_by_date(items)
    title = timeline_title(data["config"], "full", locale)
    return render_timeline(title, buckets)


def build_category_map(data: dict[str, Any], locale: str) -> str:
    cfg = data["config"]["category_map"]
    categories = data["categories"]["categories"]
    include = cfg.get("include_categories") or list(categories.keys())
    title_key = f"title_{locale}"

    lines = ["flowchart TB"]

    for category in include:
        cat_cfg = categories[category]
        cat_title = sanitize_mermaid_text(
            cat_cfg.get(title_key, cat_cfg.get("title_en", category))
        )
        node_id = category.replace(" ", "_").replace("/", "_").replace("-", "_").lower()
        lines.append(f'    subgraph {node_id}["{cat_title}"]')
        lines.append("        direction TB")

        cat_entries = [
            entry for entry in data["entries"]
            if entry["category"] == category and entry.get("show_in_full", True)
        ]
        cat_entries.sort(key=lambda e: (e.get("year", 0), e.get("name", "")))

        for entry in cat_entries:
            safe_id = re.sub(r"[^a-zA-Z0-9_]", "_", entry["id"])
            label = entry_label(entry)
            sub = sanitize_mermaid_text(entry.get("subcategory", ""))
            if sub:
                lines.append(f'        {safe_id}["{label}<br/><i>{sub}</i>"]')
            else:
                lines.append(f'        {safe_id}["{label}"]')

        lines.append("    end")

    return "\n".join(lines) + "\n"


def main() -> int:
    data = load_landscape_data()
    OUT.mkdir(parents=True, exist_ok=True)

    for locale in LOCALES:
        report = build_report_timeline(data, locale)
        full = build_full_timeline(data, locale)
        category = build_category_map(data, locale)

        (OUT / f"landscape_timeline_report.{locale}.mmd").write_text(report, encoding="utf-8")
        (OUT / f"landscape_timeline_full.{locale}.mmd").write_text(full, encoding="utf-8")
        (OUT / f"landscape_category_map.{locale}.mmd").write_text(category, encoding="utf-8")

        print(f"Wrote landscape_*.{locale}.mmd")

    report_count = sum(len(b.get("items", [])) for b in data["timeline"].get("buckets", []))
    full_count = len([e for e in data["entries"] if e.get("show_in_full", True)])
    full_count += len(data["timeline_dates"].get("external", []))
    print(f"Timeline items: report={report_count}, full={full_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
