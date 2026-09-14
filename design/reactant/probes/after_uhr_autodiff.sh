#!/bin/bash
cd "$(dirname "$0")/../../.."; P=design/reactant/probes
while kill -0 3284824 2>/dev/null; do sleep 20; done
CUDA_VISIBLE_DEVICES=0 VIEWS=24 BUDGET=32768 timeout 3600 julia --project=envs/reactant --heap-size-hint=30G $P/uhr_legacy_parity.jl > $P/uhr_legacy_parity_gpu0.log 2>&1
