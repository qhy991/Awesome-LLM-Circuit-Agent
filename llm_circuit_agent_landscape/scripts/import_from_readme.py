#!/usr/bin/env python3
"""Import landscape entries from the root README.md tables into entries.yaml.

This is a one-off / refreshable curation helper. It parses the markdown tables
under each section of ../README.md and emits data/entries.yaml. Dates are taken
from the README "Date" column, falling back to the arXiv id encoded in the URL.

Usage:
    python scripts/import_from_readme.py
"""
from __future__ import annotations

from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / ".." / "README.md").resolve()
OUT = ROOT / "data" / "entries.yaml"

SOURCE = "awesome_llm_circuit_agent"

# Ordered (heading-substring -> category). First match wins.
HEADING_MAP = [
    ("code generation & synthesis", "Digital-CodeGen"),
    ("verification & testing", "Digital-Verification"),
    ("optimization (ppa", "Digital-Optimization"),
    ("reinforcement learning approaches", "Digital-RL"),
    ("multi-agent systems & workflows", "Digital-MultiAgent"),
    ("reasoning & graph-based", "Digital-Reasoning"),
    ("topology & schematic generation", "Analog-Topology"),
    ("sizing & optimization", "Analog-Sizing"),
    ("workflows & multi-agent", "Analog-Workflow"),
    ("specialized applications", "Analog-Specialized"),
    ("datasets & benchmarks", "Datasets"),
    ("resources & learning", "Resources"),
    ("analog mind", None),          # skip — not LLM circuit-design papers
    ("contributing", None),
    ("citation", None),
    ("license", None),
]

TITLE_RE = re.compile(r"\[\*\*(.+?)\*\*\]\(([^)]+)\)")
DOT_DATE_RE = re.compile(r"(20\d{2})\.(\d{2})")
ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf|html)/(\d{2})(\d{2})\.\d{4,5}")
YEAR_ONLY_RE = re.compile(r"\b(20\d{2})\b")


def classify_heading(text: str):
    low = text.lower()
    for needle, category in HEADING_MAP:
        if needle in low:
            return category, True  # matched (category may be None = skip section)
    return None, False  # not a section heading we track


def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return text.strip("_")


def short_name(title: str) -> str:
    prefix = title.split(":", 1)[0].strip()
    if prefix and prefix != title and len(prefix) <= 48:
        return prefix
    words = title.split()
    if len(words) <= 8:
        return title
    return " ".join(words[:8])


def clean_subcategory(topic: str) -> str:
    tokens = [t.strip() for t in topic.split(",") if t.strip()]
    out: list[str] = []
    total = 0
    for token in tokens[:3]:
        if out and total + len(token) + 2 > 44:
            break
        out.append(token)
        total += len(token) + 2
    return ", ".join(out)


def derive_date(line: str, url: str):
    m = DOT_DATE_RE.search(line)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = ARXIV_RE.search(url)
    if m:
        return 2000 + int(m.group(1)), int(m.group(2))
    m = ARXIV_RE.search(line)
    if m:
        return 2000 + int(m.group(1)), int(m.group(2))
    m = YEAR_ONLY_RE.search(line)
    if m:
        return int(m.group(1)), None
    return None, None


def period_for(year: int, month) -> str:
    if year >= 2026:
        return "2026+"
    if year == 2025:
        if month and month >= 7:
            return "2025 H2"
        return "2025 H1"
    if year == 2024:
        return "2024"
    return "2023"


def split_cells(line: str) -> list[str]:
    parts = [c.strip() for c in line.strip().strip("|").split("|")]
    return parts


def method_tags_from_topic(topic: str) -> list[str]:
    tags = []
    for token in topic.split(","):
        slug = slugify(token)
        if slug:
            tags.append(slug)
    return tags[:4]


def yaml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def emit_entry(entry: dict) -> str:
    lines = [f"- id: {yaml_str(entry['id'])}"]
    lines.append(f"  name: {yaml_str(entry['name'])}")
    lines.append(f"  year: {entry['year']}")
    lines.append(f"  period: {yaml_str(entry['period'])}")
    lines.append(f"  category: {yaml_str(entry['category'])}")
    if entry.get("subcategory"):
        lines.append(f"  subcategory: {yaml_str(entry['subcategory'])}")
    if entry.get("venue"):
        lines.append(f"  venue: {yaml_str(entry['venue'])}")
    if entry.get("method_tags"):
        tags = ", ".join(yaml_str(t) for t in entry["method_tags"])
        lines.append(f"  method_tags: [{tags}]")
    if entry.get("url"):
        lines.append(f"  url: {yaml_str(entry['url'])}")
    lines.append(f"  source: {yaml_str(entry['source'])}")
    lines.append(f"  display_priority: {entry['display_priority']}")
    lines.append(f"  show_in_report: {str(entry['show_in_report']).lower()}")
    lines.append(f"  show_in_full: {str(entry['show_in_full']).lower()}")
    if entry.get("timeline_date"):
        lines.append(f"  timeline_date: {yaml_str(entry['timeline_date'])}")
    return "\n".join(lines)


def main() -> int:
    text = README.read_text(encoding="utf-8")
    current_category = None
    entries: list[dict] = []
    seen_ids: dict[str, int] = {}
    seen_urls: set[str] = set()

    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            heading = stripped.lstrip("#").strip()
            category, matched = classify_heading(heading)
            if matched:
                current_category = category
            continue

        if not stripped.startswith("|"):
            continue
        if current_category is None:
            continue

        m = TITLE_RE.search(line)
        if not m:
            continue  # header / separator / non-paper row
        title = re.sub(r"\s+", " ", m.group(1).strip())
        url = m.group(2).strip()
        if url in seen_urls:
            continue
        seen_urls.add(url)

        cells = split_cells(line)
        topic = cells[-1] if cells else ""
        venue = cells[1] if len(cells) >= 4 else ""

        year, month = derive_date(line, url)
        if year is None:
            year, month = 2026, None  # safe default for undated resources

        period = period_for(year, month)
        timeline_date = f"{year}-{month:02d}" if month else None

        base_id = slugify(short_name(title)) or slugify(title)
        base_id = base_id[:40].strip("_") or "entry"
        entry_id = base_id
        if entry_id in seen_ids:
            seen_ids[base_id] += 1
            entry_id = f"{base_id}_{seen_ids[base_id]}"
        else:
            seen_ids[base_id] = 1

        entries.append({
            "id": entry_id,
            "name": short_name(title),
            "year": year,
            "period": period,
            "category": current_category,
            "subcategory": clean_subcategory(re.sub(r"\s+", " ", topic)),
            "venue": venue,
            "method_tags": method_tags_from_topic(topic),
            "url": url,
            "source": SOURCE,
            "display_priority": 2,
            "show_in_report": False,
            "show_in_full": True,
            "timeline_date": timeline_date,
        })

    header = (
        "# Canonical data file for the LLM Circuit Agent Landscape.\n"
        "# Generated by scripts/import_from_readme.py from ../README.md, then\n"
        "# hand-curated. Run: make all\n"
        "#\n"
        "# Fields: id, name, year, period, category are required. Optional:\n"
        "#   subcategory, venue, method_tags, url, display_priority,\n"
        "#   timeline_date, timeline_label, highlight.\n"
        "# Curated report timeline: data/timeline.yaml\n\n"
    )
    body = "\n".join(emit_entry(e) for e in entries) + "\n"
    OUT.write_text(header + body, encoding="utf-8")

    by_cat: dict[str, int] = {}
    for e in entries:
        by_cat[e["category"]] = by_cat.get(e["category"], 0) + 1
    print(f"Wrote {len(entries)} entries to {OUT}")
    for cat, n in sorted(by_cat.items()):
        print(f"  {cat}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
