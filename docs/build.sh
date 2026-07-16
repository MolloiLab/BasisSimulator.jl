#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Blessed Snapshot build hook for BasisSimulator.jl.
#
# Snapshot runs THIS in your own GitHub Actions (driven by ../snapshot.toml:
# type="build", [app] dir="docs") and publishes whatever it writes to docs/dist.
# It deliberately OVERRIDES Snapshot's default behavior:
#
#   • Default would be the notebook→WASM COLLECTION pipeline. snapshot.toml's
#     type="build" overrides that → Snapshot runs this script instead.
#   • A default app build would run `app.jl build` and try to RE-RENDER the Pluto
#     notebooks. We export BASISSIM_SKIP_NB_EXPORT=1 so it does NOT — the notebooks
#     run Metal/CUDA kernels CI can't execute, so they're rendered locally and
#     committed under docs/notebooks-static/; this build just mounts that HTML.
#   • app.jl reads SNAPSHOT_BASE_PATH (which Snapshot exports = /app/<owner>/<repo>)
#     so every href + asset resolves under Snapshot's hosting path, not GH Pages'.
#
# Net result: the SAME site as the regular docs CI (.github/workflows/docs.yml),
# built fast, with the notebooks never recompiled on CI.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"   # docs/ — runnable standalone, and exactly where Snapshot invokes us

export BASISSIM_SKIP_NB_EXPORT=1   # ← the key override: reuse committed notebooks-static/, never re-render

echo "▶ verify committed notebook exports"
python3 verify_notebook_exports.py

echo "▶ instantiate docs env (no-op if Snapshot already warmed it)"
# The committed lock is part of every notebook export fingerprint. Never
# resolve it during deployment: build exactly what was verified, and fail if
# instantiation unexpectedly rewrites the lock.
manifest_before="$(shasum -a 256 Manifest.toml | awk '{print $1}')"
julia --project=. -e 'using Pkg; Pkg.Registry.add("General"); Pkg.instantiate()'
manifest_after="$(shasum -a 256 Manifest.toml | awk '{print $1}')"
test "$manifest_before" = "$manifest_after" || {
    echo "docs/Manifest.toml changed during instantiate; refusing unverified build" >&2
    exit 1
}
python3 verify_notebook_exports.py

echo "▶ install Tailwind + DaisyUI"
npm install --no-audit --no-fund

echo "▶ Therapy build — notebooks NOT re-rendered, Tailwind auto-compiled → dist/"
julia --project=. --optimize=3 app.jl build

# Native Therapy routes already contain the fragment markup in their rendered
# `/examples/.../index.html`. Keep both export forms committed as reusable Pluto
# caches, but do not ship either redundant root HTML copy in Snapshot's bundle.
# Island directories and all referenced assets remain untouched.
find dist/notebooks-static -maxdepth 1 -type f -name '*.html' -delete
# Compiler diagnostics are review artifacts, not runtime assets. The browser
# needs islands.json + shim.js, but report/coverage files can contain source
# details and should never be published.
find dist/notebooks-static -type f \( -name 'report.json' -o -name 'coverage.json' \) -delete

echo "✓ dist/ ready ($(find dist -type f | wc -l | tr -d ' ') files)"
