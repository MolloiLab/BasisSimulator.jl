#!/usr/bin/env bash
# Re-render the stale notebook exports (docs/notebooks-static/), one exporter per GPU.
#
#   docs/render_notebooks.sh              # every stale notebook
#   docs/render_notebooks.sh 04_pcct_vmi  # just these slugs (stale or not)
#
# Every exporter takes notebooks from one queue, so the GPUs stay busy however uneven the
# notebooks are. On an NVIDIA machine each exporter sees one GPU (CUDA_VISIBLE_DEVICES); set
# BASISSIM_GPUS="0 1" to choose them, and BASISSIM_LANES_PER_GPU=2 (or more) to run several
# exporters per GPU when its memory allows (the lab's 96 GB cards take 3). Without nvidia-smi (a
# Mac with Metal, or the CPU) one exporter renders everything. Logs land in docs/render-logs/ (git-ignored).
#
# Data: notebook 05 reads the XCAT phantom from BASISSIM_XCAT_DIR (AGENTS.md, "The documentation site").
set -euo pipefail
cd "$(dirname "$0")"

julia --project=build_env -e 'using Pkg; Pkg.instantiate()'
julia --project=. -e 'using Pkg; Pkg.instantiate()'

if [ "$#" -gt 0 ]; then
    slugs=("$@")                      # named notebooks render whether stale or not
else
    # every stale notebook (every notebook with BASISSIM_FORCE_NB_REBUILD=1); a failure here stops
    stale="$(julia --project=build_env extract_all.jl --list-stale)"
    mapfile -t slugs < <(printf '%s\n' "$stale" | sed '/^$/d')
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
lanes_per_gpu="${BASISSIM_LANES_PER_GPU:-1}"
lanes=()
for gpu in "${gpus[@]}"; do
    for _ in $(seq "$lanes_per_gpu"); do lanes+=("$gpu"); done
done
mkdir -p render-logs
echo "rendering ${#slugs[@]} notebook(s) on ${#lanes[@]} lane(s) (${lanes_per_gpu} per GPU): ${slugs[*]}"

# A lane that dies leaves its claimed notebook behind; a second pass re-queues and renders it.
run_pass() {
    local pass="$1" pids=() i=0
    for gpu in "${lanes[@]}"; do
        i=$((i + 1))
        local log="render-logs/pass${pass}-lane${i}-gpu${gpu:-cpu}.log"
        CUDA_VISIBLE_DEVICES="$gpu" julia --project=build_env extract_all.jl --queue "$queue" > "$log" 2>&1 &
        pids+=("$!")
        echo "  pass $pass, lane $i (GPU ${gpu:-cpu}) → docs/$log"
    done
    local status=0
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    return "$status"
}
for pass in 1 2; do
    run_pass "$pass" || echo "a lane exited with an error in pass $pass (see docs/render-logs/)" >&2
    for f in "$queue"/*.claimed.*; do
        [ -e "$f" ] || continue
        slug="$(basename "$f")"; slug="${slug%%.claimed.*}"
        echo "  $slug was not finished; re-queued" >&2
        mv "$f" "$queue/$slug.todo"
    done
    [ -n "$(ls "$queue")" ] || break
done
grep -h '▸\|✓\|⚠' render-logs/*.log || true
leftover="$(ls "$queue")"
[ -z "$leftover" ] || { echo "not rendered: $leftover" >&2; exit 1; }
python3 verify_notebook_exports.py
