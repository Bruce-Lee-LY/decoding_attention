# Copyright 2023. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 20:42:28 on Sun, Feb 12, 2023
#
# Description: run cpp script

#!/bin/bash

set -euo pipefail

WORK_PATH=$(cd $(dirname $0) && pwd) && cd $WORK_PATH

export CUDA_VISIBLE_DEVICES=0
# export CUDA_LAUNCH_BLOCKING=1

rm -rf log ncu && mkdir -p log ncu

# $1: b, $2: sq, $3: sk, $4: hq, $5: hk, $6: d, $7: dv, $8: is_causal, $9: log_path
evaluate_da() {
    echo "Evaluating ${1} * ${2} * ${3} * ${4} * ${5} * ${6} * ${7} * ${8} * ${9}"
    $WORK_PATH/output/bin/benchmark_decoding_attn -b=$1 -sq=$2 -sk=$3 -hq=$4 -hk=$5 -d=$6 -dv=$7 -is_causal=$8 -is_alibi=false -is_bf16=false -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=false > log/${9}/da_${1}_${2}_${3}_${4}_${5}_${6}_${7}.log 2>&1
    sleep 3
}

# $1: b, $2: sq, $3: sk, $4: hq, $5: hk, $6: d, $7: dv, $8: is_causal, $9: log_path
ncu_da() {
    echo "NCU ${1} * ${2} * ${3} * ${4} * ${5} * ${6} * ${7} * ${8} * ${9}"
    sudo ncu --set full --target-processes all --force-overwrite -o ncu/${8}/da_${1}_${2}_${3}_${4}_${5}_${6}_${7} $WORK_PATH/output/bin/benchmark_decoding_attn -b=$1 -sq=$2 -sk=$3 -hq=$4 -hk=$5 -d=$6 -dv=$7 -is_causal=$8 -is_alibi=false -is_bf16=false -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/${9}/ncu_da_${1}_${2}_${3}_${4}_${5}_${6}_${7}.log 2>&1
    sleep 3
}

# $1: hq, $2: hk, $3: d, $4: dv
benchmark_da_decoding_seq() {
    echo "Evaluating Decoding Seq"
    b=1
    sq=1
    seq_k=(1 8 16 32 64 128 256 512 1024 2048 3072 4096 5120 6144 7168 8192)
    hq=$1
    hk=$2
    d=$3
    dv=$4
    ic=false
    lp=decoding_seq

    mkdir -p log/$lp ncu/$lp

    for sk in ${seq_k[@]};
    do
        evaluate_da $b $sq $sk $hq $hk $d $dv $ic $lp
        # ncu_da $b $sq $sk $hq $hk $d $dv $ic $lp
    done
}

# $1: hq, $2: hk, $3: d, $4: dv
benchmark_da_decoding_batch() {
    echo "Evaluating Decoding Batch"
    batch=(1 2 4 8 16 32 64 128 256 512 768 1024 1536 2048)
    sq=1
    sk=128
    hq=$1
    hk=$2
    d=$3
    dv=$4
    ic=false
    lp=decoding_batch

    mkdir -p log/$lp ncu/$lp

    for b in ${batch[@]};
    do
        evaluate_da $b $sq $sk $hq $hk $d $dv $ic $lp
        # ncu_da $b $sq $sk $hq $hk $d $dv $ic $lp
    done
}

benchmark_da() {
    benchmark_da_decoding_seq 32 32 128 128
    benchmark_da_decoding_batch 32 32 128 128
}

benchmark_mla() {
    benchmark_da_decoding_seq 128 1 576 512
    benchmark_da_decoding_batch 128 1 576 512
}

# FP16
# nohup $WORK_PATH/output/bin/benchmark_decoding_attn -b=2 -sq=1 -sk=256 -hq=32 -hk=32 -d=128 -dv=128 -is_causal=false -is_alibi=false -is_bf16=false -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=true > log/da_2_1_256_32_32_128_128.log 2>&1 &
# sudo ncu --set full --target-processes all --force-overwrite -o ncu/da_2_1_256_32_32_128_128 $WORK_PATH/output/bin/benchmark_decoding_attn -b=2 -sq=1 -sk=256 -hq=32 -hk=32 -d=128 -dv=128 -is_causal=false -is_alibi=false -is_bf16=false -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_da_2_1_256_32_32_128_128.log 2>&1

# BF16
# nohup $WORK_PATH/output/bin/benchmark_decoding_attn -b=2 -sq=1 -sk=256 -hq=32 -hk=32 -d=128 -dv=128 -is_causal=false -is_alibi=false -is_bf16=true -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=true > log/da_2_1_256_32_32_128_128.log 2>&1 &
# sudo ncu --set full --target-processes all --force-overwrite -o ncu/da_2_1_256_32_32_128_128 $WORK_PATH/output/bin/benchmark_decoding_attn -b=2 -sq=1 -sk=256 -hq=32 -hk=32 -d=128 -dv=128 -is_causal=false -is_alibi=true -is_bf16=false -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_da_2_1_256_32_32_128_128.log 2>&1

# GQA/MQA
# nohup $WORK_PATH/output/bin/benchmark_decoding_attn -b=2 -sq=1 -sk=256 -hq=64 -hk=8 -d=128 -dv=128 -is_causal=false -is_alibi=false -is_bf16=false -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=true > log/da_2_1_256_64_8_128_128.log 2>&1 &
# sudo ncu --set full --target-processes all --force-overwrite -o ncu/da_2_1_256_64_8_128_128 $WORK_PATH/output/bin/benchmark_decoding_attn -b=2 -sq=256 -sk=256 -hq=64 -hk=8 -d=128 -dv=128 -is_causal=true -is_alibi=false -is_bf16=false -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_da_2_1_256_64_8_128_128.log 2>&1

# Alibi
# nohup $WORK_PATH/output/bin/benchmark_decoding_attn -b=2 -sq=1 -sk=256 -hq=32 -hk=32 -d=128 -dv=128 -is_causal=false -is_alibi=true -is_bf16=false -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=true > log/da_2_1_256_32_32_128_128.log 2>&1 &
# sudo ncu --set full --target-processes all --force-overwrite -o ncu/da_2_1_256_32_32_128_128 $WORK_PATH/output/bin/benchmark_decoding_attn -b=2 -sq=256 -sk=256 -hq=32 -hk=32 -d=128 -dv=128 -is_causal=true -is_alibi=true -is_bf16=false -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_da_2_1_256_32_32_128_128.log 2>&1

# MLA
nohup $WORK_PATH/output/bin/benchmark_decoding_attn -b=2 -sq=1 -sk=256 -hq=128 -hk=1 -d=576 -dv=512 -is_causal=false -is_alibi=false -is_bf16=false -warmup_iterations=1 -profiling_iterations=10 -sleep_duration=100 -enable_check=true > log/da_2_1_256_128_1_576_512.log 2>&1 &
# sudo ncu --set full --target-processes all --force-overwrite -o ncu/da_2_1_256_128_1_576_512 $WORK_PATH/output/bin/benchmark_decoding_attn -b=2 -sq=1 -sk=256 -hq=128 -hk=1 -d=576 -dv=512 -is_causal=false -is_alibi=false -is_bf16=false -warmup_iterations=1 -profiling_iterations=1 -sleep_duration=100 -enable_check=false > log/ncu_da_2_1_256_128_1_576_512.log 2>&1

# benchmark_da
# benchmark_mla
