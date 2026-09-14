#!/bin/bash
cd "$(dirname "$0")/../../.."; P=design/reactant/probes
while [ ! -f $P/prec_queue_gpu1.done ]; do sleep 20; done
CUDA_VISIBLE_DEVICES=1 PREC=DEFAULT BACKEND=gpu NV=256 NS=16 R=512 NZ=8 VIEWS=48 BUDGET=16384 timeout 3600 julia --project=envs/reactant --heap-size-hint=24G $P/bisect_precision.jl > $P/prec_PREC_DEFAULT_512_gpu1.log 2>&1
echo DONE > $P/prec_default_after.done
