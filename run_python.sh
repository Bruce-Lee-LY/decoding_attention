# Copyright 2023. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 20:42:28 on Sun, Feb 12, 2023
#
# Description: run python script

#!/bin/bash

set -euo pipefail

WORK_PATH=$(cd $(dirname $0) && pwd) && cd $WORK_PATH

export CUDA_VISIBLE_DEVICES=0

rm -rf log && mkdir -p log

# test
python3 $WORK_PATH/tests/test_decoding_attn.py
# pytest $WORK_PATH/tests/test_decoding_attn.py

# FP16
# python3 $WORK_PATH/benchmarks/python/benchmark_decoding_attn.py --head_q 32 --head_k 32 --dim 128 --profiling_iterations 10 > log/benchmark_decoding_attn.log 2>&1

# BF16
# python3 $WORK_PATH/benchmarks/python/benchmark_decoding_attn.py --head_q 32 --head_k 32 --dim 128 --is_bf16 --profiling_iterations 10 > log/benchmark_decoding_attn.log 2>&1
