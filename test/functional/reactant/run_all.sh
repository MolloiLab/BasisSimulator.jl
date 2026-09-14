#!/usr/bin/env bash
# Run every Reactant/Enzyme smoke of the functional core SERIALLY, one Julia
# process at a time, behind the system-wide lock (XLA compiles are memory-hungry;
# overlapping them can hard-reset a 16 GB Mac).
#
#   bash test/functional/reactant/run_all.sh            # all smokes, on the default XLA client (CUDA where present)
#   bash test/functional/reactant/run_all.sh dd fbp     # a subset
#   BACKEND=cpu bash test/functional/reactant/run_all.sh   # pin the CPU client (regression baseline on a GPU box)
# Every smoke is traced at full f32 precision (run_full_precision.jl) — on CUDA the default TF32 matmuls are a ~-31 HU bias.
#
# First run: julia --project=envs/reactant -e 'using Pkg; Pkg.instantiate()'
set -uo pipefail
cd "$(dirname "$0")/../../.."
LOCK=/tmp/bs_reactant.lock
until mkdir "$LOCK" 2>/dev/null; do echo "waiting for $LOCK …"; sleep 30; done
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
PRELOAD=""; [ "${BACKEND:-}" = "cpu" ] && PRELOAD="-L test/functional/reactant/cpu_backend.jl"
names=("$@")
if [ ${#names[@]} -eq 0 ]; then
    names=(dd fbp eict pcct hir denoise vmi nchannel pipeline windowed gpu_parity)
fi
status=0
for n in "${names[@]}"; do
    f="test/functional/reactant/smoke_$n.jl"
    [ -f "$f" ] || { echo "skip $n (no $f)"; continue; }
    echo "════════ smoke_$n.jl ════════"
    if julia --project=envs/reactant -t 2 --heap-size-hint=3G $PRELOAD test/functional/reactant/run_full_precision.jl "$f"; then
        echo "PASS smoke_$n"
    else
        echo "FAIL smoke_$n"; status=1
    fi
done
exit $status
