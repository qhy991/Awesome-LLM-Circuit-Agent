#!/usr/bin/env python3
"""Inject generated Mermaid blocks into the README files.

Usage:
    python scripts/sync_readme.py
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "scripts"))

from mermaid_lib import load_landscape_data  # noqa: E402


def wrap_mermaid(content: str) -> str:
    body = content.rstrip("\n")
    return f"```mermaid\n{body}\n```"


def replace_between(text: str, begin: str, end: str, replacement: str) -> str:
    start = text.find(begin)
    stop = text.find(end)
    if start == -1 or stop == -1 or stop < start:
        raise ValueError(f"Markers not found or out of order: {begin!r} ... {end!r}")
    stop += len(end)
    return text[:start] + begin + "\n\n" + replacement + "\n\n" + end + text[stop:]


def sync_mermaid_blocks(text: str, locale: str, markers: dict[str, dict[str, str]]) -> str:
    figures = ROOT / "figures"
    report = (figures / f"landscape_timeline_report.{locale}.mmd").read_text(encoding="utf-8")
    full = (figures / f"landscape_timeline_full.{locale}.mmd").read_text(encoding="utf-8")
    category = (figures / f"landscape_category_map.{locale}.mmd").read_text(encoding="utf-8")

    text = replace_between(
        text,
        markers["report_timeline"]["begin"],
        markers["report_timeline"]["end"],
        wrap_mermaid(report),
    )
    text = replace_between(
        text,
        markers["full_timeline"]["begin"],
        markers["full_timeline"]["end"],
        wrap_mermaid(full),
    )
    text = replace_between(
        text,
        markers["category_map"]["begin"],
        markers["category_map"]["end"],
        wrap_mermaid(category),
    )
    return text


def main() -> int:
    data = load_landscape_data()
    cfg = data["config"]
    markers = cfg["markers"]

    for key, file_cfg in cfg["readme_files"].items():
        readme_path = (ROOT / file_cfg["path"]).resolve()
        locale = file_cfg["locale"]
        if not readme_path.exists():
            print(f"Skip missing {readme_path}")
            continue
        text = readme_path.read_text(encoding="utf-8")
        text = sync_mermaid_blocks(text, locale, markers)
        readme_path.write_text(text, encoding="utf-8")
        print(f"Updated {readme_path} ({key})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
