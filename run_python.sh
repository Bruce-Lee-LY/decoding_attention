# Copyright 2023. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 20:42:28 on Sun, Feb 12, 2023
#
# Description: run python script

#!/bin/bash

set -euo pipefail

WORK_PATH=$(cd $(dirname $0) && pwd) && cd $WORK_PATH

export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1

rm -rf log && mkdir -p log

# test
python3 $WORK_PATH/tests/test_decoding_attn.py
# pytest $WORK_PATH/tests/test_decoding_attn.py

# FP16
# python3 $WORK_PATH/benchmarks/python/benchmark_decoding_attn.py --head_q 32 --head_k 32 --dim 128 --dim_v 128 --warmup_iterations 1 --profiling_iterations 10 > log/benchmark_decoding_attn.log 2>&1

# BF16
# python3 $WORK_PATH/benchmarks/python/benchmark_decoding_attn.py --head_q 32 --head_k 32 --dim 128 --dim_v 128 --is_bf16 --warmup_iterations 1 --profiling_iterations 10 > log/benchmark_decoding_attn.log 2>&1

# MLA
# python3 $WORK_PATH/benchmarks/python/benchmark_decoding_attn.py --head_q 128 --head_k 1 --dim 576 --dim_v 512 --warmup_iterations 1 --profiling_iterations 10 > log/benchmark_decoding_attn.log 2>&1
