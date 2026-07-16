#!/usr/bin/env python3
"""Reject broken, stale, or machine-specific committed notebook exports."""

from pathlib import Path
import hashlib
import json
import re
import sys
from functools import lru_cache
from typing import List, Optional, Set

DOCS = Path(__file__).resolve().parent
ROOT = DOCS / "notebooks-static"
SOURCES = DOCS / "notebooks"
DATA_PROVENANCE = SOURCES / "DATA_PROVENANCE.sha256"
HASH_META = "basissim-export-fingerprint"
EXPORT_CONTRACT = (
    "basissim-lean-v4|therapy=true|fragment=true|islands=true|verify=true|"
    "optimize=size|forced-fallback=01:z_slice,05:z_helical,11:z_idx"
)
BAD_HTML = {
    "rendered Pluto error": "<jlerror",
    "serialized Pluto exception": "plain_error",
}
BAD_TEXT = {
    "developer-machine path": "/Users/daleblack/",
    "CI workspace path": "/home/runner/work/",
}
BAD_REPORT_REASONS = ("failed to parse", "package ", "not found in current path")
FORCED_FALLBACKS = {
    "01_five_struct_api": (
        "z_slice",
        {"12000001-0000-4000-8000-000000000004"},
    ),
    "05_xcat_grid_to_recon": (
        "z_helical",
        {
            "05000012-0000-4000-8000-000000000040",
            "05000016-0000-4000-8000-000000000020",
        },
    ),
    "11_helical_scanning": (
        "z_idx",
        {"11000007-0000-4000-8000-000000000003"},
    ),
}


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def simulator_input_hash() -> str:
    root = DOCS.parent
    files = [root / "Project.toml", DATA_PROVENANCE]
    for directory in (root / "src",):
        if directory.is_dir():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    entries = [
        f"{path.relative_to(root)}:{source_hash(path)}"
        for path in sorted(files)
    ]
    return hashlib.sha256("\0".join(entries).encode("utf-8")).hexdigest()


def snapshot_tree_hash() -> str:
    manifest = (DOCS / "build_env" / "Manifest.toml").read_text(encoding="utf-8")
    section = re.search(
        r"\[\[deps\.Snapshot\]\](.*?)(?=\n\[\[deps\.|\Z)", manifest, re.S
    )
    tree = (
        re.search(r'^git-tree-sha1\s*=\s*"([0-9a-f]{40})"', section.group(1), re.M)
        if section else None
    )
    if tree is None:
        raise ValueError("missing locked Snapshot tree")
    return tree.group(1)


def export_fingerprint(path: Path, snapshot_tree: str) -> str:
    build_lock = source_hash(DOCS / "build_env" / "Manifest.toml")
    docs_lock = source_hash(DOCS / "Manifest.toml")
    driver = source_hash(DOCS / "extract_all.jl")
    payload = "\0".join(
        (source_hash(path), simulator_input_hash(), snapshot_tree, build_lock,
         docs_lock, driver, EXPORT_CONTRACT)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def recorded_hash(path: Path, fragment: bool) -> Optional[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = (
        rf"<!--\s*{HASH_META}:\s*([0-9a-f]{{64}})\s*-->"
        if fragment
        else rf'<meta\s+name="{HASH_META}"\s+content="([0-9a-f]{{64}})"'
    )
    match = re.search(pattern, text)
    return match.group(1) if match else None


def main() -> int:
    sources = sorted(SOURCES.glob("*.jl"))
    if not sources:
        print(f"no notebook sources found in {SOURCES}", file=sys.stderr)
        return 1

    failures: List[str] = []
    expected_pages: Set[Path] = set()
    try:
        snapshot_tree = snapshot_tree_hash()
    except (OSError, ValueError):
        print("invalid Snapshot export lock in docs/build_env/Manifest.toml", file=sys.stderr)
        return 1
    for source in sources:
        slug = source.stem
        digest = export_fingerprint(source, snapshot_tree)
        for suffix, fragment in ((".html", False), (".fragment.html", True)):
            page = ROOT / f"{slug}{suffix}"
            expected_pages.add(page)
            if not page.is_file():
                failures.append(f"{page.name}: missing export")
                continue
            found = recorded_hash(page, fragment)
            if found != digest:
                failures.append(
                    f"{page.name}: stale export (source {digest[:12]}, export {str(found)[:12]})"
                )
            text = page.read_text(encoding="utf-8", errors="replace").lower()
            for label, needle in BAD_HTML.items():
                if needle.lower() in text:
                    failures.append(f"{page.name}: {label} ({needle!r})")
            if fragment and "localstorage.getitem('snap-theme')" in text:
                failures.append(f"{page.name}: embedded fragment overrides host theme")

        report = ROOT / f"{slug}.islands" / "report.json"
        assets = ROOT / f"{slug}.islands"
        expected_fallback = FORCED_FALLBACKS.get(slug)
        if expected_fallback is None:
            if assets.exists():
                failures.append(
                    f"{assets.relative_to(ROOT)}: unexpected island/fallback assets"
                )
        else:
            bond, expected_cells = expected_fallback
            required = {
                "report.json", "coverage.json", "islands.json", "shim.js"
            }
            missing = sorted(name for name in required if not (assets / name).is_file())
            for name in missing:
                failures.append(
                    f"{assets.relative_to(ROOT)}/{name}: missing configured-fallback asset"
                )
            if not missing:
                reason = (
                    f"configured fallback (matched @bind {bond}); "
                    "island inference intentionally skipped"
                )
                try:
                    groups = json.loads((assets / "report.json").read_text(encoding="utf-8"))
                    manifest = json.loads((assets / "islands.json").read_text(encoding="utf-8"))
                    coverage = json.loads((assets / "coverage.json").read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    failures.append(
                        f"{assets.relative_to(ROOT)}: invalid configured-fallback JSON"
                    )
                else:
                    if len(groups) != 1:
                        failures.append(
                            f"{assets.relative_to(ROOT)}/report.json: expected one group"
                        )
                    else:
                        group = groups[0]
                        cells = group.get("cells", [])
                        actual_cells = {str(cell.get("id")) for cell in cells}
                        if (
                            group.get("bonds") != [bond]
                            or group.get("judgement") != "fallback"
                            or group.get("fallback_kind") != "configured"
                            or group.get("reasons") != [reason]
                            or actual_cells != expected_cells
                            or any(cell.get("ok") is not False for cell in cells)
                            or any(cell.get("reasons") != [reason] for cell in cells)
                        ):
                            failures.append(
                                f"{assets.relative_to(ROOT)}/report.json: "
                                "configured fallback contract mismatch"
                            )
                    expected_runtime = [{
                        "bonds": [bond],
                        "judgement": "fallback",
                        "fallback_kind": "configured",
                    }]
                    if (
                        manifest.get("groups") != []
                        or manifest.get("fallback_groups") != expected_runtime
                    ):
                        failures.append(
                            f"{assets.relative_to(ROOT)}/islands.json: "
                            "runtime fallback index mismatch"
                        )
                    count = len(expected_cells)
                    expected_coverage = {
                        "groups": {
                            "island": 0, "partial": 0, "fallback": 1, "total": 1
                        },
                        "cells": {
                            "interactive": 0, "fallback": count, "total": count
                        },
                    }
                    if coverage != expected_coverage:
                        failures.append(
                            f"{assets.relative_to(ROOT)}/coverage.json: "
                            "configured fallback coverage mismatch"
                        )

        if report.is_file():
            try:
                groups = json.loads(report.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                failures.append(f"{report.relative_to(ROOT)}: invalid JSON")
            else:
                for group in groups:
                    reasons = list(group.get("reasons", []))
                    reasons.extend(
                        reason
                        for cell in group.get("cells", [])
                        for reason in cell.get("reasons", [])
                    )
                    for reason in reasons:
                        low = str(reason).lower()
                        if any(needle in low for needle in BAD_REPORT_REASONS):
                            failures.append(
                                f"{report.relative_to(ROOT)}: invalid island diagnostic"
                            )

    actual_pages = set(ROOT.glob("*.html"))
    for orphan in sorted(actual_pages - expected_pages):
        failures.append(f"{orphan.name}: orphan export without source notebook")

    # Every textual artifact under notebooks-static is committed and copied by
    # Therapy. Catch machine paths even outside HTML (reports, maps, shims).
    for artifact in sorted(ROOT.rglob("*")):
        if not artifact.is_file() or artifact.suffix.lower() not in {
            ".html", ".json", ".js", ".css", ".map", ".txt"
        }:
            continue
        text = artifact.read_text(encoding="utf-8", errors="replace").lower()
        for label, needle in BAD_TEXT.items():
            if needle.lower() in text:
                failures.append(
                    f"{artifact.relative_to(ROOT)}: {label} ({needle!r})"
                )

    if failures:
        print("broken committed notebook exports:", file=sys.stderr)
        print("\n".join(f"  - {failure}" for failure in failures), file=sys.stderr)
        return 1

    print(f"verified {len(sources)} notebook source/export pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
