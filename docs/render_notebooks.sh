#!/usr/bin/env bash
# Re-render the stale notebook exports (docs/notebooks-static/), one exporter per GPU.
#
#   docs/render_notebooks.sh              # every stale notebook
#   docs/render_notebooks.sh 04_pcct_vmi  # just these slugs (stale or not)
#
# Every exporter takes notebooks from one queue, so the GPUs stay busy however uneven the
# notebooks are. On an NVIDIA machine each exporter sees one GPU (CUDA_VISIBLE_DEVICES); set
# BASISSIM_GPUS="0 1" to choose them. Without nvidia-smi (a Mac with Metal, or the CPU) one
# exporter renders everything. Logs land in docs/render-logs/ (git-ignored).
#
# Data: notebook 05 reads the XCAT phantom from BASISSIM_XCAT_DIR (AGENTS.md, "Docs").
set -euo pipefail
cd "$(dirname "$0")"

julia --project=build_env -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.instantiate()'

if [ "$#" -gt 0 ]; then
    slugs=("$@")
else
    mapfile -t slugs < <(julia --project=build_env extract_all.jl --list-stale)
fi
if [ "${#slugs[@]}" -eq 0 ]; then
    echo "every notebook export is current"
    python3 verify_notebook_exports.py
    exit 0
fi

if [ -n "${BASISSIM_GPUS:-}" ]; then
    read -r -a gpus <<< "$BASISSIM_GPUS"
elif command -v nvidia-smi > /dev/null 2>&1; then
    mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader)
else
    gpus=("")
fi

queue="$(mktemp -d)"
trap 'rm -rf "$queue"' EXIT
for slug in "${slugs[@]}"; do
    [ -f "notebooks/$slug.jl" ] || { echo "no notebook docs/notebooks/$slug.jl" >&2; exit 1; }
    touch "$queue/$slug.todo"
done
# a forced list of slugs renders even when current
[ "$#" -gt 0 ] && export BASISSIM_FORCE_NB_REBUILD=1

mkdir -p render-logs
echo "rendering ${#slugs[@]} notebook(s) on ${#gpus[@]} lane(s): ${slugs[*]}"
pids=()
for gpu in "${gpus[@]}"; do
    log="render-logs/gpu${gpu:-cpu}.log"
    CUDA_VISIBLE_DEVICES="$gpu" julia --project=build_env extract_all.jl --queue "$queue" > "$log" 2>&1 &
    pids+=("$!")
    echo "  lane ${gpu:-cpu} → docs/$log"
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
grep -h '▸\|✓\|⚠' render-logs/*.log || true
[ "$status" -eq 0 ] || { echo "an exporter failed; see docs/render-logs/" >&2; exit 1; }
leftover="$(ls "$queue")"
[ -z "$leftover" ] || { echo "not rendered: $leftover" >&2; exit 1; }
python3 verify_notebook_exports.py
