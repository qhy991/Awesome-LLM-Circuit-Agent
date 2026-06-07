"""Shared helpers for Mermaid landscape rendering."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import re

import yaml

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def load_yaml(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def sanitize_mermaid_text(text: str) -> str:
    """Make text safe for Mermaid timeline / flowchart labels."""
    text = text.replace("\n", " ").strip()
    text = text.replace(":", " ·")
    text = text.replace('"', "'")
    text = text.replace("#", "")
    text = re.sub(r"\s+", " ", text)
    return text


def entry_label(entry: dict[str, Any]) -> str:
    if entry.get("timeline_label"):
        return sanitize_mermaid_text(str(entry["timeline_label"]))
    return sanitize_mermaid_text(str(entry["name"]))


def resolve_item_label(item: dict[str, Any], entries_by_id: dict[str, dict[str, Any]]) -> str:
    if "label" in item:
        return sanitize_mermaid_text(str(item["label"]))
    entry_id = item.get("id")
    if not entry_id:
        raise ValueError(f"Timeline item must have 'id' or 'label': {item!r}")
    entry = entries_by_id.get(entry_id)
    if entry is None:
        raise ValueError(f"Unknown entry id in timeline: {entry_id!r}")
    return entry_label(entry)


def resolve_entry_date(
    entry: dict[str, Any],
    period_defaults: dict[str, str],
    overrides: dict[str, str],
) -> str:
    if entry.get("timeline_date"):
        return str(entry["timeline_date"])
    entry_id = entry["id"]
    if entry_id in overrides:
        return overrides[entry_id]
    period = entry["period"]
    month = period_defaults.get(period, "06")
    return f"{entry['year']}-{month}"


def render_timeline(title: str, buckets: list[tuple[str, list[str]]]) -> str:
    lines = ["timeline", f"    title {sanitize_mermaid_text(title)}"]
    for date, labels in buckets:
        if not labels:
            continue
        lines.append(f"    {date} : {labels[0]}")
        for label in labels[1:]:
            lines.append(f"             : {label}")
    return "\n".join(lines) + "\n"


def group_labels_by_date(items: list[tuple[str, str]]) -> list[tuple[str, list[str]]]:
    grouped: dict[str, list[str]] = {}
    order: list[str] = []
    for date, label in items:
        if date not in grouped:
            grouped[date] = []
            order.append(date)
        grouped[date].append(label)
    return [(date, grouped[date]) for date in sorted(order)]


def build_entries_by_id(entries: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {entry["id"]: entry for entry in entries}


def load_landscape_data() -> dict[str, Any]:
    return {
        "entries": load_yaml(DATA / "entries.yaml"),
        "categories": load_yaml(DATA / "categories.yaml"),
        "timeline": load_yaml(DATA / "timeline.yaml"),
        "timeline_dates": load_yaml(DATA / "timeline_dates.yaml"),
        "config": load_yaml(DATA / "mermaid_config.yaml"),
    }
