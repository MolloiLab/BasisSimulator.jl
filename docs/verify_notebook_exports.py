#!/usr/bin/env python3
"""Reject broken, stale, or machine-specific committed notebook exports.

Covers the BasisSimulator notebooks (docs/notebooks/<slug>.jl, rendered by docs/extract_all.jl)
and the BasisMonteCarlo notebooks (lib/BasisMonteCarlo/docs/notebooks/<stem>.jl, rendered by
docs/extract_montecarlo.jl as `mc-<stem>`). `BASISSIM_SKIP_NOTEBOOKS` (comma-separated slugs; a
trailing `*` matches a prefix, so `mc-*` skips every BasisMonteCarlo notebook) leaves notebooks
out, as it leaves them out of the site."""

from pathlib import Path
import hashlib
import json
import re
import sys
import os
from functools import lru_cache
from typing import List, Optional, Set, Tuple

DOCS = Path(__file__).resolve().parent
ROOT = DOCS / "notebooks-static"
SOURCES = DOCS / "notebooks"
DATA_PROVENANCE = SOURCES / "DATA_PROVENANCE.sha256"
MC_ROOT = DOCS.parent / "lib" / "BasisMonteCarlo"
MC_DOCS = MC_ROOT / "docs"
MC_SOURCES = MC_DOCS / "notebooks"
MC_PREFIX = "mc-"
HASH_META = "basissim-export-fingerprint"
EXPORT_CONTRACT = (
    "basissim-lean-v4|therapy=true|fragment=true|islands=true|verify=true|"
    "optimize=size|forced-fallback=01:z_slice,05:z_helical,11:z_idx"
)
# MC_EXPORT_CONTRACT in docs/extract_montecarlo.jl
MC_EXPORT_CONTRACT = (
    "basismc-lean-v1|therapy=true|fragment=true|islands=true|verify=true|"
    "optimize=size|cpu|forced-fallback=04,05,06,07,08:kvp|card=first-figure-webp"
)
BAD_HTML = {
    "rendered Pluto error": "<jlerror",
    "serialized Pluto exception": "plain_error",
}
BAD_TEXT = {
    "developer-machine path": "/Users/daleblack/",
    "CI workspace path": "/home/runner/work/",
}
BAD_TEXT_PATTERNS = {
    "macOS user path": re.compile(r"/(?:private/)?Users/[^/\s<]+/", re.I),
    "Unix home path": re.compile(r"/home/[^/\s<]+/", re.I),
    "temporary build path": re.compile(r"/(?:private/)?tmp/[^\s<\"']+", re.I),
    "mounted volume path": re.compile(r"/Volumes/[^\s<\"']+", re.I),
}
BAD_REPORT_REASONS = ("failed to parse", "package ", "not found in current path")
# The notebooks whose volume slider is forced to a static fallback (FORCE_FALLBACK_BONDS in
# extract_all.jl): the bond, and the one fallback group its report must contain. The group's cells
# must be cells of the notebook (checked against the source), so editing a notebook never needs
# a cell id here.
FORCED_FALLBACKS = {
    "01_five_struct_api": "z_slice",
    "05_xcat_grid_to_recon": "z_helical",
    "11_helical_scanning": "z_idx",
    # MC_FORCE_FALLBACK_BONDS in docs/extract_montecarlo.jl
    "mc-04_siemens_force": "kvp",
    "mc-05_siemens_flash": "kvp",
    "mc-06_ge_revolution_apex": "kvp",
    "mc-07_philips_brilliance_ict": "kvp",
    "mc-08_canon_aquilion_one": "kvp",
}


def source_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_input_hash(path: Path) -> str:
    """Hash package inputs while ignoring the release-only version field."""
    content = path.read_text(encoding="utf-8")
    normalized = re.sub(
        r'^version\s*=\s*"[^"]*"\s*$',
        'version = "<release-version>"',
        content,
        flags=re.M,
    )
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def simulator_input_hash() -> str:
    root = DOCS.parent
    files = [DATA_PROVENANCE]
    for directory in (root / "src",):
        if directory.is_dir():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    entries = [
        f"{path.relative_to(root)}:{source_hash(path)}"
        for path in sorted(files)
    ]
    entries.append(f"Project.toml:{project_input_hash(root / 'Project.toml')}")
    entries.sort()
    return hashlib.sha256("\0".join(entries).encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def mc_input_hash() -> str:
    """mc_input_hash in docs/extract_montecarlo.jl."""
    root = DOCS.parent
    # every lockfile: one per Julia minor version (Manifest-v1.12.toml, Manifest-v1.13.toml)
    files = [MC_DOCS / "Project.toml"] + sorted(
        path for path in MC_DOCS.iterdir() if re.fullmatch(r"Manifest(-v[0-9.]+)?\.toml", path.name)
    )
    for directory in (MC_ROOT / "src", MC_DOCS / "reference_data"):
        if directory.is_dir():
            files.extend(path for path in directory.rglob("*") if path.is_file())
    entries = [f"{path.relative_to(root).as_posix()}:{source_hash(path)}" for path in files]
    project = MC_ROOT / "Project.toml"
    entries.append(f"{project.relative_to(root).as_posix()}:{project_input_hash(project)}")
    entries.sort()
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


def mc_export_fingerprint(path: Path, snapshot_tree: str) -> str:
    """mc_export_fingerprint in docs/extract_montecarlo.jl."""
    build_lock = source_hash(DOCS / "build_env" / "Manifest.toml")
    exporter = source_hash(DOCS / "extract_all.jl")
    driver = source_hash(DOCS / "extract_montecarlo.jl")
    payload = "\0".join(
        (source_hash(path), mc_input_hash(), snapshot_tree, build_lock, exporter,
         driver, MC_EXPORT_CONTRACT)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def skipped(slug: str) -> bool:
    patterns = [p.strip() for p in os.environ.get("BASISSIM_SKIP_NOTEBOOKS", "").split(",") if p.strip()]
    return any(slug.startswith(p[:-1]) if p.endswith("*") else slug == p for p in patterns)


def recorded_hash(path: Path, fragment: bool) -> Optional[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = (
        rf"<!--\s*{HASH_META}:\s*([0-9a-f]{{64}})\s*-->"
        if fragment
        else rf'<meta\s+name="{HASH_META}"\s+content="([0-9a-f]{{64}})"'
    )
    match = re.search(pattern, text)
    return match.group(1) if match else None


def check_islands(source: Path, assets: Path, forced: Optional[str]) -> List[str]:
    """A notebook's `.islands/` must hold only compiled islands (every cell verified) and, for a
    notebook in FORCED_FALLBACKS, exactly the one configured fallback group of that bond; the
    runtime manifest and the coverage summary must agree with the report."""
    where = assets.relative_to(ROOT)
    failures: List[str] = []
    required = {"report.json", "coverage.json", "islands.json", "shim.js"}
    missing = sorted(name for name in required if not (assets / name).is_file())
    if missing:
        return [f"{where}/{name}: missing island asset" for name in missing]
    try:
        groups = json.loads((assets / "report.json").read_text(encoding="utf-8"))
        manifest = json.loads((assets / "islands.json").read_text(encoding="utf-8"))
        coverage = json.loads((assets / "coverage.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [f"{where}: invalid island JSON"]
    notebook_cells = set(
        re.findall(r"^# ╔═╡ ([0-9a-f-]{36})\s*$", source.read_text(encoding="utf-8"), re.M)
    )
    islands = [g for g in groups if g.get("judgement") == "island"]
    fallbacks = [g for g in groups if g.get("judgement") == "fallback"]
    if len(islands) + len(fallbacks) != len(groups):
        failures.append(f"{where}/report.json: a group that is neither an island nor the configured fallback")
    for g in islands:
        cells = g.get("cells", [])
        ids = {str(c.get("id")) for c in cells}
        if not cells or not ids <= notebook_cells or any(c.get("ok") is not True for c in cells):
            failures.append(f"{where}/report.json: island {g.get('bonds')} has an unverified or unknown cell")
    fallback_cells = set()
    if forced is None:
        if fallbacks:
            failures.append(f"{where}/report.json: unexpected fallback {[g.get('bonds') for g in fallbacks]}")
        expected_runtime = []
    else:
        reason = f"configured fallback (matched @bind {forced}); island inference intentionally skipped"
        if len(fallbacks) != 1:
            failures.append(f"{where}/report.json: expected exactly the configured fallback of {forced}")
        else:
            g = fallbacks[0]
            cells = g.get("cells", [])
            fallback_cells = {str(c.get("id")) for c in cells}
            if (
                g.get("bonds") != [forced]
                or g.get("fallback_kind") != "configured"
                or g.get("reasons") != [reason]
                or not fallback_cells
                or not fallback_cells <= notebook_cells
                or any(c.get("ok") is not False for c in cells)
                or any(c.get("reasons") != [reason] for c in cells)
            ):
                failures.append(f"{where}/report.json: configured fallback contract mismatch")
        expected_runtime = [{"bonds": [forced], "judgement": "fallback", "fallback_kind": "configured"}]
    if len(manifest.get("groups", [])) != len(islands) or manifest.get("fallback_groups") != expected_runtime:
        failures.append(f"{where}/islands.json: runtime index does not match the report")
    n_island_cells = sum(len(g.get("cells", [])) for g in islands)
    expected_coverage = {
        "groups": {"island": len(islands), "partial": 0, "fallback": len(fallbacks), "total": len(groups)},
        "cells": {"interactive": n_island_cells, "fallback": len(fallback_cells),
                  "total": n_island_cells + len(fallback_cells)},
    }
    if coverage != expected_coverage:
        failures.append(f"{where}/coverage.json: coverage does not match the report")
    return failures


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
    notebooks: List[Tuple[str, Path, str]] = [
        (source.stem, source, export_fingerprint(source, snapshot_tree)) for source in sources
    ]
    for source in sorted(MC_SOURCES.glob("*.jl")):
        slug = MC_PREFIX + source.stem
        if skipped(slug):
            notebooks.append((slug, source, ""))
            continue
        try:
            digest = mc_export_fingerprint(source, snapshot_tree)
        except OSError as error:
            print(f"cannot fingerprint {slug}: {error}", file=sys.stderr)
            return 1
        notebooks.append((slug, source, digest))
    checked = 0
    for slug, source, digest in notebooks:
        if skipped(slug):
            # Left out of the site: neither checked nor an orphan.
            expected_pages.update(ROOT / f"{slug}{suffix}" for suffix in (".html", ".fragment.html"))
            continue
        checked += 1
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

        assets = ROOT / f"{slug}.islands"
        report = assets / "report.json"
        forced = FORCED_FALLBACKS.get(slug)
        if assets.exists() or forced is not None:
            failures.extend(check_islands(source, assets, forced))

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
        for label, pattern in BAD_TEXT_PATTERNS.items():
            match = pattern.search(text)
            if match:
                failures.append(
                    f"{artifact.relative_to(ROOT)}: {label} ({match.group(0)!r})"
                )

    if failures:
        print("broken committed notebook exports:", file=sys.stderr)
        print("\n".join(f"  - {failure}" for failure in failures), file=sys.stderr)
        return 1

    print(f"verified {checked} notebook source/export pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
