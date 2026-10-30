#!/usr/bin/env bash
# Re-render the stale notebook exports (docs/notebooks-static/), one exporter per GPU.
#
#   docs/render_notebooks.sh                          # every stale notebook
#   docs/render_notebooks.sh 04_pcct_vmi              # just these slugs (stale or not)
#   docs/render_notebooks.sh mc-04_siemens_force      # a BasisMonteCarlo notebook (CPU)
#
# Every exporter takes notebooks from one queue, so the GPUs stay busy however uneven the
# notebooks are. On an NVIDIA machine each exporter sees one GPU (CUDA_VISIBLE_DEVICES); set
# BASISSIM_GPUS="0 1" to choose them, and BASISSIM_LANES_PER_GPU=2 (or more) to run several
# exporters per GPU when its memory allows (the lab's 96 GB cards take 3). Without nvidia-smi (a
# Mac with Metal, or the CPU) one exporter renders everything. Logs land in docs/render-logs/ (git-ignored).
#
# The BasisMonteCarlo.jl notebooks (lib/BasisMonteCarlo/docs/notebooks/<stem>.jl, slug mc-<stem>)
# render on the CPU, in their own lib/BasisMonteCarlo/docs environment, through
# docs/extract_montecarlo.jl: one lane of their own beside the GPU lanes (BASISSIM_MC_LANES=2 for
# more; each notebook starts 8 workers). They take 5–15 min each, mc-10_scatter_library about 3.6 h.
# BASISSIM_SKIP_NOTEBOOKS=mc-10_scatter_library leaves one out of the stale set (`mc-*`: all of them).
#
# Data: notebook 05 reads the XCAT phantom from BASISSIM_XCAT_DIR (AGENTS.md, "The documentation site").
set -euo pipefail
cd "$(dirname "$0")"

mc_docs="../lib/BasisMonteCarlo/docs"
julia --project=build_env -e 'using Pkg; Pkg.instantiate()'

if [ "$#" -gt 0 ]; then
    slugs=("$@")                      # named notebooks render whether stale or not
else
    # every stale notebook (every notebook with BASISSIM_FORCE_NB_REBUILD=1); a failure here stops
    stale="$(julia --project=build_env extract_all.jl --list-stale)"
    mc_stale="$(julia --project=build_env extract_montecarlo.jl --list-stale)"
    mapfile -t slugs < <(printf '%s\n%s\n' "$stale" "$mc_stale" | sed '/^$/d')
fi
if [ "${#slugs[@]}" -eq 0 ]; then
    echo "every notebook export is current"
    python3 verify_notebook_exports.py
    exit 0
fi

queue="$(mktemp -d)"
trap 'rm -rf "$queue"' EXIT
mkdir "$queue/gpu" "$queue/mc"
n_gpu=0
n_mc=0
for slug in "${slugs[@]}"; do
    if [[ "$slug" == mc-* ]]; then
        [ -f "$mc_docs/notebooks/${slug#mc-}.jl" ] || { echo "no notebook $mc_docs/notebooks/${slug#mc-}.jl" >&2; exit 1; }
        touch "$queue/mc/$slug.todo"
        n_mc=$((n_mc + 1))
    else
        [ -f "notebooks/$slug.jl" ] || { echo "no notebook docs/notebooks/$slug.jl" >&2; exit 1; }
        touch "$queue/gpu/$slug.todo"
        n_gpu=$((n_gpu + 1))
    fi
done
[ "$n_gpu" -eq 0 ] || julia --project=. -e 'using Pkg; Pkg.instantiate()'
[ "$n_mc" -eq 0 ] || julia --project="$mc_docs" -e 'using Pkg; Pkg.instantiate()'

lanes=()   # "gpu:<index>" (empty index: the CPU) for docs/notebooks, "mc" for BasisMonteCarlo
if [ "$n_gpu" -gt 0 ]; then
    if [ -n "${BASISSIM_GPUS:-}" ]; then
        read -r -a gpus <<< "$BASISSIM_GPUS"
    elif command -v nvidia-smi > /dev/null 2>&1; then
        mapfile -t gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader)
    else
        gpus=("")
    fi
    lanes_per_gpu="${BASISSIM_LANES_PER_GPU:-1}"
    for gpu in "${gpus[@]}"; do
        for _ in $(seq "$lanes_per_gpu"); do lanes+=("gpu:$gpu"); done
    done
fi
if [ "$n_mc" -gt 0 ]; then
    for _ in $(seq "${BASISSIM_MC_LANES:-1}"); do lanes+=("mc"); done
fi
mkdir -p render-logs
echo "rendering ${#slugs[@]} notebook(s) on ${#lanes[@]} lane(s): ${slugs[*]}"

# A lane that dies leaves its claimed notebook behind; a second pass re-queues and renders it.
run_pass() {
    local pass="$1" pids=() i=0
    for lane in "${lanes[@]}"; do
        i=$((i + 1))
        if [ "$lane" = mc ]; then
            local log="render-logs/pass${pass}-lane${i}-mc-cpu.log"
            CUDA_VISIBLE_DEVICES="" julia --project=build_env extract_montecarlo.jl --queue "$queue/mc" > "$log" 2>&1 &
            pids+=("$!")
            echo "  pass $pass, lane $i (BasisMonteCarlo, CPU) → docs/$log"
        else
            local gpu="${lane#gpu:}"
            local log="render-logs/pass${pass}-lane${i}-gpu${gpu:-cpu}.log"
            CUDA_VISIBLE_DEVICES="$gpu" julia --project=build_env extract_all.jl --queue "$queue/gpu" > "$log" 2>&1 &
            pids+=("$!")
            echo "  pass $pass, lane $i (GPU ${gpu:-cpu}) → docs/$log"
        fi
    done
    local status=0
    for pid in "${pids[@]}"; do wait "$pid" || status=1; done
    return "$status"
}
for pass in 1 2; do
    run_pass "$pass" || echo "a lane exited with an error in pass $pass (see docs/render-logs/)" >&2
    for f in "$queue"/*/*.claimed.*; do
        [ -e "$f" ] || continue
        slug="$(basename "$f")"; slug="${slug%%.claimed.*}"
        echo "  $slug was not finished; re-queued" >&2
        mv "$f" "$(dirname "$f")/$slug.todo"
    done
    [ -n "$(find "$queue" -type f)" ] || break
done
grep -h '▸\|✓\|⚠' render-logs/*.log || true
leftover=""
for f in "$queue"/*/*; do [ -e "$f" ] && leftover="$leftover $(basename "$f")"; done
[ -z "$leftover" ] || { echo "not rendered:$leftover" >&2; exit 1; }
python3 verify_notebook_exports.py
