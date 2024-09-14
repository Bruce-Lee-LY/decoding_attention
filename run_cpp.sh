# Copyright 2023. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 20:42:28 on Sun, Feb 12, 2023
#
# Description: run cpp script

#!/bin/bash

set -euo pipefail

WORK_PATH=$(cd $(dirname $0) && pwd) && cd $WORK_PATH

export CUDA_VISIBLE_DEVICES=0

rm -rf log ncu && mkdir -p log ncu

# $1: b, $2: sq, $3: sk, $4: hq, $5: hk, $6: d, $7: is_causal, $8: log_path
evaluate_da() {
    echo "Evaluating ${1} * ${2} * ${3} * ${4} * ${5} * ${6} * ${7} * ${8}"
    $WORK_PATH/output/bin/benchmark_decoding_attn -b=$1 -sq=$2 -sk=$3 -hq=$4 -hk=$5 -d=$6 -is_causal=$7 -is_alibi=false -is_bf16=false -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > log/${8}/da_${1}_${2}_${3}_${4}_${5}_${6}.log 2>&1
    sleep 3
}

# $1: b, $2: sq, $3: sk, $4: hq, $5: hk, $6: d, $7: is_causal, $8: log_path
ncu_da() {
    echo "NCU ${1} * ${2} * ${3} * ${4} * ${5} * ${6} * ${7} * ${8}"
    sudo ncu --set full --target-processes all --force-overwrite -o ncu/${8}/da_${1}_${2}_${3}_${4}_${5}_${6} $WORK_PATH/output/bin/benchmark_decoding_attn -b=$1 -sq=$2 -sk=$3 -hq=$4 -hk=$5 -d=$6 -is_causal=$7 -is_alibi=false -is_bf16=false -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/${8}/ncu_da_${1}_${2}_${3}_${4}_${5}_${6}.log 2>&1
    sleep 3
}

benchmark_da_decoding_seq() {
    echo "Evaluating Decoding Seq"
    b=1
    sq=1
    seq_k=(1 8 16 32 64 128 256 512 1024 2048 3072 4096 5120 6144 7168 8192)
    hq=32
    hk=32
    d=128
    ic=false
    lp=decoding_seq

    mkdir -p log/$lp ncu/$lp

    for sk in ${seq_k[@]};
    do
        evaluate_da $b $sq $sk $hq $hk $d $ic $lp
        # ncu_da $b $sq $sk $hq $hk $d $ic $lp
    done
}

benchmark_da_decoding_batch() {
    echo "Evaluating Decoding Batch"
    batch=(1 2 4 8 16 32 64 128 256 512 768 1024 1536 2048)
    sq=1
    sk=128
    hq=32
    hk=32
    d=128
    ic=false
    lp=decoding_batch

    mkdir -p log/$lp ncu/$lp

    for b in ${batch[@]};
    do
        evaluate_da $b $sq $sk $hq $hk $d $ic $lp
        # ncu_da $b $sq $sk $hq $hk $d $ic $lp
    done
}

benchmark_da() {
    benchmark_da_decoding_seq
    benchmark_da_decoding_batch
}

# FP16
nohup $WORK_PATH/output/bin/benchmark_decoding_attn -b=2 -sq=1 -sk=256 -hq=32 -hk=32 -d=128 -is_causal=false -is_alibi=false -is_bf16=false -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=true > log/da_2_1_256_32_32_128.log 2>&1 &
# sudo ncu --set full --target-processes all --force-overwrite -o ncu/da_2_1_256_32_32_128 $WORK_PATH/output/bin/benchmark_decoding_attn -b=2 -sq=1 -sk=256 -hq=32 -hk=32 -d=128 -is_causal=false -is_alibi=false -is_bf16=false -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_da_2_1_256_32_32_128.log 2>&1

# BF16
# nohup $WORK_PATH/output/bin/benchmark_decoding_attn -b=2 -sq=1 -sk=256 -hq=32 -hk=32 -d=128 -is_causal=false -is_alibi=false -is_bf16=true -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=true > log/da_2_1_256_32_32_128.log 2>&1 &
# sudo ncu --set full --target-processes all --force-overwrite -o ncu/da_2_1_256_32_32_128 $WORK_PATH/output/bin/benchmark_decoding_attn -b=2 -sq=1 -sk=256 -hq=32 -hk=32 -d=128 -is_causal=false -is_alibi=true -is_bf16=false -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_da_2_1_256_32_32_128.log 2>&1

# GQA/MQA
# nohup $WORK_PATH/output/bin/benchmark_decoding_attn -b=2 -sq=1 -sk=256 -hq=64 -hk=8 -d=128 -is_causal=false -is_alibi=false -is_bf16=false -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=true > log/da_2_1_256_64_8_128.log 2>&1 &
# sudo ncu --set full --target-processes all --force-overwrite -o ncu/da_2_1_256_64_8_128 $WORK_PATH/output/bin/benchmark_decoding_attn -b=2 -sq=256 -sk=256 -hq=64 -hk=8 -d=128 -is_causal=true -is_alibi=false -is_bf16=false -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_da_2_1_256_64_8_128.log 2>&1

# Alibi
# nohup $WORK_PATH/output/bin/benchmark_decoding_attn -b=2 -sq=1 -sk=256 -hq=32 -hk=32 -d=128 -is_causal=false -is_alibi=true -is_bf16=false -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=true > log/da_2_1_256_32_32_128.log 2>&1 &
# sudo ncu --set full --target-processes all --force-overwrite -o ncu/da_2_1_256_32_32_128 $WORK_PATH/output/bin/benchmark_decoding_attn -b=2 -sq=256 -sk=256 -hq=32 -hk=32 -d=128 -is_causal=true -is_alibi=true -is_bf16=false -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_da_2_1_256_32_32_128.log 2>&1

# benchmark_da
