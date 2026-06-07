#!/usr/bin/env python3
"""Validate landscape data files.

Usage:
    python scripts/validate_entries.py
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any
import re
import sys
import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# Circuit landscape keeps a leaner required set than the kernel variant because
# affiliation / backend metadata is not consistently available upstream.
REQUIRED_FIELDS = {
    "id", "name", "year", "period", "category",
    "source", "show_in_report", "show_in_full",
}
OPTIONAL_FIELDS = {
    "subcategory", "backends", "method_tags", "affiliation_short",
    "display_priority", "affiliation", "url", "venue", "notes", "highlight",
    "timeline_date", "timeline_label",
}
DATE_RE = re.compile(r"^\d{4}-\d{2}$")


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    cfg = load_yaml(DATA / "categories.yaml")
    entries = load_yaml(DATA / "entries.yaml")
    sources_cfg = load_yaml(DATA / "sources.yaml")

    valid_categories = set(cfg["categories"].keys())
    valid_periods = set(cfg["periods"].keys())
    valid_sources = {s["id"] for s in sources_cfg.get("sources", [])} | {"self"}

    if not isinstance(entries, list):
        raise ValueError("data/entries.yaml must contain a list of entries")

    ids = [entry.get("id") for entry in entries]
    duplicates = sorted(k for k, v in Counter(ids).items() if v > 1)
    if duplicates:
        raise ValueError(f"Duplicate ids: {duplicates}")

    errors: list[str] = []
    for index, entry in enumerate(entries):
        label = entry.get("id", f"entry-{index}")
        missing = sorted(REQUIRED_FIELDS - set(entry.keys()))
        if missing:
            errors.append(f"{label}: missing fields {missing}")
            continue

        if entry["category"] not in valid_categories:
            errors.append(f"{label}: invalid category {entry['category']!r}")
        if entry["period"] not in valid_periods:
            errors.append(f"{label}: invalid period {entry['period']!r}")
        if entry["source"] not in valid_sources:
            errors.append(f"{label}: invalid source {entry['source']!r}")
        if "backends" in entry and not isinstance(entry["backends"], list):
            errors.append(f"{label}: backends must be a list")
        if "method_tags" in entry and not isinstance(entry["method_tags"], list):
            errors.append(f"{label}: method_tags must be a list")
        if "display_priority" in entry and not isinstance(entry["display_priority"], int):
            errors.append(f"{label}: display_priority must be int")

        timeline_date = entry.get("timeline_date")
        if timeline_date is not None and not DATE_RE.match(str(timeline_date)):
            errors.append(f"{label}: timeline_date must be YYYY-MM, got {timeline_date!r}")

    timeline_path = DATA / "timeline.yaml"
    if timeline_path.exists():
        timeline_cfg = load_yaml(timeline_path)
        entry_ids = set(ids)
        for bucket in timeline_cfg.get("buckets", []):
            bucket_date = bucket.get("date", "?")
            for item in bucket.get("items", []):
                item_id = item.get("id")
                if item_id and item_id not in entry_ids:
                    errors.append(
                        f"timeline.yaml {bucket_date}: unknown entry id {item_id!r}"
                    )
                if not item.get("id") and not item.get("label"):
                    errors.append(
                        f"timeline.yaml {bucket_date}: item needs id or label"
                    )

    dates_path = DATA / "timeline_dates.yaml"
    if dates_path.exists():
        dates_cfg = load_yaml(dates_path)
        entry_ids = set(ids)
        for entry_id in dates_cfg.get("overrides", {}):
            if entry_id not in entry_ids:
                errors.append(f"timeline_dates.yaml: unknown override id {entry_id!r}")
        for ext in dates_cfg.get("external", []):
            if not DATE_RE.match(str(ext.get("date", ""))):
                errors.append(f"timeline_dates.yaml external: invalid date {ext!r}")
            if not ext.get("label"):
                errors.append(f"timeline_dates.yaml external: missing label {ext!r}")

    if errors:
        print("Validation failed:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1

    print(f"Validated {len(entries)} entries across {len(valid_categories)} categories.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
