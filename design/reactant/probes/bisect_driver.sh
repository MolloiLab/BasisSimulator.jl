#!/bin/bash
# Runs the offset bisect across backends and sizes, SERIALLY (one Reactant process at a time).
cd "$(dirname "$0")/../../.."
P=design/reactant/probes; OUT=$P/bisect_results.txt; : > "$OUT"
[ -f envs/reactant/Manifest.toml ] || julia --project=envs/reactant -e 'using Pkg; Pkg.instantiate()' >> "$OUT" 2>&1
run() { local label=$1; shift
  echo "################ $label ################" >> "$OUT"
  env "$@" timeout 3600 julia --project=envs/reactant --heap-size-hint=24G $P/bisect_compiled_offset.jl 2>&1 \
    | grep -E '^\[(gpu|cpu)\]|ERROR|LoadError|RESOURCE_EXHAUSTED|OutOfMemory' >> "$OUT"
  echo "(julia exit ${PIPESTATUS[0]})" >> "$OUT"; }
run "A: 32^2 recon, 32^2x2 phantom, 40 views, GPU"  BACKEND=gpu NV=32  NS=2  R=32  NZ=2 VIEWS=40
run "B: same, CPU"                                    BACKEND=cpu NV=32  NS=2  R=32  NZ=2 VIEWS=40
run "C: 512^2x8 recon, 256^2x16 phantom, 48 views, GPU (offset was seen here)" BACKEND=gpu NV=256 NS=16 R=512 NZ=8 VIEWS=48 FULLPROG=0
echo "BISECT_DRIVER_DONE" >> "$OUT"
