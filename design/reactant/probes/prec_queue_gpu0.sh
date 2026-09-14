#!/bin/bash
cd "$(dirname "$0")/../../.."; P=design/reactant/probes
CUDA_VISIBLE_DEVICES=0 PREC=HIGHEST BACKEND=gpu NV=32 NS=2 R=32 NZ=2 VIEWS=40 timeout 1800 julia --project=envs/reactant --heap-size-hint=16G $P/bisect_precision.jl > $P/prec_HIGHEST_32_gpu0.log 2>&1
CUDA_VISIBLE_DEVICES=0 FULLPROG=0 BACKEND=gpu NV=32 NS=2 R=32 NZ=2 VIEWS=40 timeout 1800 julia --project=$HOME/.cache/bs_clean_env --heap-size-hint=16G $P/bisect_compiled_offset.jl > $P/bisect_cleanHEAD_driver_gpu0.log 2>&1
echo QUEUE_DONE > $P/prec_queue_gpu0.done
