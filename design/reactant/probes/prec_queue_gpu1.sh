#!/bin/bash
cd "$(dirname "$0")/../../.."; P=design/reactant/probes
for cfg in "PREC=DEFAULT" "PREC=HIGHEST" "PREC=DEFAULT ALG=TF32_TF32_F32_X3"; do
  tag=$(echo "$cfg" | tr ' =' '__')
  env $cfg CUDA_VISIBLE_DEVICES=1 BACKEND=gpu NV=256 NS=16 R=512 NZ=8 VIEWS=48 BUDGET=16384 \
    timeout 3600 julia --project=envs/reactant --heap-size-hint=24G $P/bisect_precision.jl > $P/prec_${tag}_512_gpu1.log 2>&1
done
echo QUEUE_DONE > $P/prec_queue_gpu1.done
