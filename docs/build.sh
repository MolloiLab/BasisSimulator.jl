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

echo "▶ instantiate docs env (no-op if Snapshot already warmed it)"
# Mirrors docs.yml exactly: ensure the General registry exists + resolve BEFORE
# instantiate, so a drifted committed Manifest (vs the path-sourced BasisSimulator
# at ..) fixes itself instead of failing the build.
julia --project=. -e 'using Pkg; Pkg.Registry.add("General"); Pkg.resolve(); Pkg.instantiate()'

echo "▶ Therapy build — notebooks NOT re-rendered, Tailwind auto-compiled → dist/"
julia --project=. --optimize=3 app.jl build

echo "✓ dist/ ready ($(find dist -type f | wc -l | tr -d ' ') files)"
