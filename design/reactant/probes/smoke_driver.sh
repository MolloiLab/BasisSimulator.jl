#!/bin/bash
# Run the Reactant smokes SERIALLY for one backend, no shared lock, results to one file.
#   smoke_driver.sh <label> [extra julia args...]      (env CUDA_VISIBLE_DEVICES selects the card)
cd "$(dirname "$0")/../../.."; P=design/reactant/probes; label=$1; shift
OUT=$P/smoke_results_$label.txt; : > "$OUT"; echo "backend=$label CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES-<unset>}' start $(date +%H:%M:%S)" >> "$OUT"
for n in gpu_parity compiled windowed dd fbp eict pcct hir denoise vmi nchannel pipeline; do
  f=test/functional/reactant/smoke_$n.jl; [ -f "$f" ] || { echo "skip $n" >> "$OUT"; continue; }
  t0=$(date +%s)
  timeout 3600 julia --project=envs/reactant -t 4 --heap-size-hint=8G "$@" test/functional/reactant/run_full_precision.jl "$f" > $P/smoke_${n}_$label.log 2>&1; rc=$?
  last=$(grep -vE 'Source Location|hlo_memory|external/xla|absl::|^I0000|^W0000|ptxas|^\s*$' $P/smoke_${n}_$label.log | tail -1 | cut -c1-150)
  printf "%-10s %s  (%4ds)  %s\n" "$n" "$([ $rc -eq 0 ] && echo PASS || echo "FAIL rc=$rc")" $(( $(date +%s) - t0 )) "$last" >> "$OUT"
done
echo "SMOKES_DONE $(date +%H:%M:%S)" >> "$OUT"
