# Copyright 2023. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 21:14:13 on Tue, Oct 31, 2023
#
# Description: benchmark decoding attn using python api

# !/usr/bin/python3
# coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement

import optparse
import time
import torch
from torch.nn import functional as F

from decoding_attn import decoding_attn_fwd

try:
    # flash decoding
    from flash_attn import flash_attn_with_kvcache
except ImportError:
    flash_attn_with_kvcache = None

try:
    from flashinfer import single_decode_with_kv_cache, BatchDecodeWithPagedKVCacheWrapper
except ImportError:
    single_decode_with_kv_cache = None
    BatchDecodeWithPagedKVCacheWrapper = None

try:
    from flash_mla import get_mla_metadata, flash_mla_with_kvcache
    import triton
except ImportError:
    get_mla_metadata = None
    flash_mla_with_kvcache = None


def get_cu_seq(seqs: torch.tensor) -> torch.tensor:
    """
    Arguments:
        seqs: [batch], dtype torch.int32, sequence length of each batch.
    Return:
        cu_seq: [batch + 1], dtype torch.int32. The cumulative sequence lengths of the sequences
            in the batch.
    """
    return F.pad(seqs.cumsum(dim=0, dtype=torch.int32), (1, 0))


def compute_flops_and_bandwidth(batch, seq_q, seq_k, head_q, head_k, dim, dim_v, time):
    throughput = (batch * seq_q * seq_k * head_q * (dim + dim_v)
                  * 2 * 10**(-12)) / (time * 10**(-3))
    bandwidth = ((batch * seq_q * head_q * dim + batch * seq_k * head_k *
                 dim + batch * seq_q * head_q * dim_v) * 2 * 10**(-9)) / (time * 10**(-3))
    return throughput, bandwidth


def benchmark_flash_attn(q, k, v, profiling_iterations=10):
    batch = q.shape[0]
    seq_q = q.shape[1]
    head_q = q.shape[2]
    dim = q.shape[3]
    seq_k = k.shape[1]
    head_k = k.shape[2]
    dim_v = v.shape[3]

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # warm up
    output = flash_attn_with_kvcache(q, k, v)
    # print(f"Flash-Decoding output: {output}")

    start.record()
    for _ in range(profiling_iterations):
        __ = flash_attn_with_kvcache(q, k, v)
    end.record()
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end) / profiling_iterations
    throughput, bandwidth = compute_flops_and_bandwidth(
        batch, seq_q, seq_k, head_q, head_k, dim, dim_v, elapsed_time)
    print("Flash-Decoding {}-{} profiling time: {:.4f} ms, throughput: {:.4f} TFLOPS, bandwidth: {:.3f} GB/s".format(
        batch, seq_k, elapsed_time, throughput, bandwidth))


def benchmark_flashinfer_single(q, k, v, profiling_iterations=10):
    batch = 1
    seq_q = 1
    head_q = q.shape[0]
    dim = q.shape[1]
    seq_k = k.shape[0] // batch
    head_k = k.shape[1]
    dim_v = v.shape[2]

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # warm up
    output = single_decode_with_kv_cache(q, k, v)
    # print(f"FlashInfer-Single output: {output}")

    start.record()
    for _ in range(profiling_iterations):
        __ = single_decode_with_kv_cache(q, k, v)
    end.record()
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end) / profiling_iterations
    throughput, bandwidth = compute_flops_and_bandwidth(
        batch, seq_q, seq_k, head_q, head_k, dim, dim_v, elapsed_time)
    print("FlashInfer {}-{} profiling time: {:.4f} ms, throughput: {:.4f} TFLOPS, bandwidth: {:.3f} GB/s".format(
        batch, seq_k, elapsed_time, throughput, bandwidth))


def benchmark_flashinfer_batch(q, k, v, profiling_iterations=10):
    batch = q.shape[0]
    seq_q = 1
    head_q = q.shape[1]
    dim = q.shape[2]
    seq_k = k.shape[0] // batch
    head_k = k.shape[1]
    dim_v = v.shape[2]

    # page_size: the page size of the paged kv cache
    page_size = 1
    num_pages_per_seq = (seq_k + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch
    # NHD: the last 3 dimensions are organized as [seq_k, head_k, dim]
    kv_layout = "NHD"
    k = k.unsqueeze(1).unsqueeze(2)
    v = v.unsqueeze(1).unsqueeze(2)
    # kv_data: [total_num_pages, 2, page_size, head_k, dim]
    kv_data = torch.cat((k, v), 1)
    kv_indptr = torch.arange(0, batch + 1, dtype=torch.int32,
                             device=torch.device('cuda')) * num_pages_per_seq
    kv_indices = torch.arange(
        0, total_num_pages, dtype=torch.int32, device=torch.device('cuda'))
    kv_last_page_len = torch.full(
        (batch,), (seq_k - 1) % page_size + 1, dtype=torch.int32, device=torch.device('cuda'))
    # the device of the workspace buffer should be the same as the device of the input tensors
    # in the split-k algorithm
    workspace_buffer = torch.empty(
        batch * seq_q * head_q * dim, dtype=torch.float32, device=torch.device('cuda'))
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout)
    wrapper.plan(kv_indptr, kv_indices, kv_last_page_len, head_q,
                 head_k, dim, page_size, data_type=k.dtype, q_data_type=q.dtype)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # warm up
    output = wrapper.run(q, kv_data)
    # print(f"FlashInfer-Batch output: {output}")

    start.record()
    for _ in range(profiling_iterations):
        __ = wrapper.run(q, kv_data)
    end.record()
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end) / profiling_iterations
    throughput, bandwidth = compute_flops_and_bandwidth(
        batch, seq_q, seq_k, head_q, head_k, dim, dim_v, elapsed_time)
    print("FlashInfer {}-{} profiling time: {:.4f} ms, throughput: {:.4f} TFLOPS, bandwidth: {:.3f} GB/s".format(
        batch, seq_k, elapsed_time, throughput, bandwidth))


def benchmark_flashmla(q, k, dim_v, profiling_iterations=10):
    batch = q.shape[0]
    seq_q = q.shape[1]
    head_q = q.shape[2]
    dim = q.shape[3]
    seq_k = k.shape[1]
    head_k = k.shape[2]

    cache_seqlens = torch.full(
        (batch,), seq_k, dtype=torch.int32, device=torch.device('cuda'))
    block_size = 64
    block_table = torch.arange(batch * seq_k // block_size, dtype=torch.int32,
                               device=torch.device('cuda')).view(batch, seq_k // block_size)
    blocked_k = k.reshape(block_table.numel(), block_size, head_k, dim)
    for i in range(batch):
        blocked_k.view(batch, seq_k, head_k, dim)[i, cache_seqlens[i].item():] = (
            float("nan")
        )

    tile_scheduler_metadata, num_splits = get_mla_metadata(
        cache_seqlens, seq_q * head_q // head_k, head_k
    )

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # warm up
    output, _ = flash_mla_with_kvcache(
        q, blocked_k, block_table, cache_seqlens, dim_v, tile_scheduler_metadata, num_splits, causal=True)
    # print(f"FlashMLA output: {output}")

    start.record()
    for _ in range(profiling_iterations):
        __, ___ = flash_mla_with_kvcache(
            q, blocked_k, block_table, cache_seqlens, dim_v, tile_scheduler_metadata, num_splits, causal=True)
    end.record()
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end) / profiling_iterations
    throughput, bandwidth = compute_flops_and_bandwidth(
        batch, seq_q, seq_k, head_q, head_k, dim, dim_v, elapsed_time)
    print("FlashMLA {}-{} profiling time: {:.4f} ms, throughput: {:.4f} TFLOPS, bandwidth: {:.3f} GB/s".format(
        batch, seq_k, elapsed_time, throughput, bandwidth))


def benchmark_decoding_attn(q, k, v, dim_v, is_alibi, profiling_iterations=10):
    batch = q.shape[0]
    seq_q = 1
    head_q = q.shape[1]
    dim = q.shape[2]
    seq_k = k.shape[0] // batch
    head_k = k.shape[1]

    cu_seq_k = get_cu_seq(torch.full(
        (batch,), seq_k, dtype=torch.int32, device=torch.device('cuda')))

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    # warm up
    output = decoding_attn_fwd(q, k, v, cu_seq_k, seq_k, dim_v, is_alibi)
    # print(f"Decoding-Attention output: {output}")

    start.record()
    for _ in range(profiling_iterations):
        __ = decoding_attn_fwd(q, k, v, cu_seq_k, seq_k, dim_v, is_alibi)
    end.record()
    torch.cuda.synchronize()
    elapsed_time = start.elapsed_time(end) / profiling_iterations
    throughput, bandwidth = compute_flops_and_bandwidth(
        batch, seq_q, seq_k, head_q, head_k, dim, dim_v, elapsed_time)
    print("Decoding-Attention {}-{} profiling time: {:.4f} ms, throughput: {:.4f} TFLOPS, bandwidth: {:.3f} GB/s".format(
        batch, seq_k, elapsed_time, throughput, bandwidth))


def benchmark_forward(batch, seq_q, seq_k, head_q, head_k, dim, dim_v, is_alibi, is_bf16, profiling_iterations=10):
    torch.cuda.empty_cache()

    dtype = torch.bfloat16 if is_bf16 else torch.float16
    total_q = batch * seq_q
    total_k = batch * seq_k

    q = torch.randn(total_q, head_q, dim,
                    device=torch.device('cuda'), dtype=dtype)
    k = torch.randn(total_k, head_k, dim,
                    device=torch.device('cuda'), dtype=dtype)
    if dim == dim_v:
        v = torch.randn(total_k, head_k, dim_v,
                        device=torch.device('cuda'), dtype=dtype)
    else:
        v = None

    if dim == dim_v and flash_attn_with_kvcache is not None:
        q4 = q.reshape(batch, seq_q, head_q, dim)
        k4 = k.reshape(batch, seq_k, head_k, dim)
        v4 = v.reshape(batch, seq_k, head_k, dim_v)

        time.sleep(0.1)
        benchmark_flash_attn(q4, k4, v4, profiling_iterations)

    if batch == 1 and dim == dim_v and single_decode_with_kv_cache is not None:
        q2 = q.reshape(head_q, dim)

        time.sleep(0.1)
        benchmark_flashinfer_single(q2, k, v, profiling_iterations)

    if batch > 1 and dim == dim_v and BatchDecodeWithPagedKVCacheWrapper is not None:
        time.sleep(0.1)
        benchmark_flashinfer_batch(q, k, v, profiling_iterations)

    if dim != dim_v and flash_mla_with_kvcache is not None:
        q4 = q.reshape(batch, seq_q, head_q, dim)
        k4 = k.reshape(batch, seq_k, head_k, dim)

        time.sleep(0.1)
        benchmark_flashmla(q4, k4, dim_v, profiling_iterations)

    time.sleep(0.1)
    benchmark_decoding_attn(q, k, v, dim_v, is_alibi, profiling_iterations)


def benchmark_seq(head_q=32, head_k=32, dim=128, dim_v=128, is_alibi=False, is_bf16=False, profiling_iterations=10):
    print("------------------------------- Benchmark Seq -------------------------------")
    batch = 1
    seq_q = 1
    if dim == dim_v:
        seq_ks = [1, 8, 16, 32, 64, 128, 256, 512, 1024,
                  2048, 3072, 4096, 5120, 6144, 7168, 8192]
    else:
        seq_ks = [256, 512, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]

    for seq_k in seq_ks:
        benchmark_forward(batch, seq_q, seq_k, head_q, head_k,
                          dim, dim_v, is_alibi, is_bf16, profiling_iterations)


def benchmark_batch(head_q=32, head_k=32, dim=128, dim_v=128, is_alibi=False, is_bf16=False, profiling_iterations=10):
    print("------------------------------- Benchmark Batch -------------------------------")
    batchs = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 768, 1024, 1536, 2048]
    seq_q = 1
    if dim == dim_v:
        seq_k = 128
    else:
        seq_k = 4096

    for batch in batchs:
        benchmark_forward(batch, seq_q, seq_k, head_q, head_k,
                          dim, dim_v, is_alibi, is_bf16, profiling_iterations)


def main():
    usage = "python3 benchmark_decoding_attn.py --head_q 32 --head_k 32 --dim 128 --dim_v 128 --profiling_iterations 10"
    parser = optparse.OptionParser(usage)
    parser.add_option("--head_q", dest="head_q", type="int", default="32")
    parser.add_option("--head_k", dest="head_k", type="int", default="32")
    parser.add_option("--dim", dest="dim", type="int", default="128")
    parser.add_option("--dim_v", dest="dim_v", type="int", default="128")
    parser.add_option("--is_alibi", action="store_true",
                      dest="is_alibi", default=False)
    parser.add_option("--is_bf16", action="store_true",
                      dest="is_bf16", default=False)
    parser.add_option("--profiling_iterations",
                      dest="profiling_iterations", type="int", default="10")

    options, args = parser.parse_args()
    head_q = options.head_q
    head_k = options.head_k
    dim = options.dim
    dim_v = options.dim_v
    is_alibi = options.is_alibi
    is_bf16 = options.is_bf16
    profiling_iterations = options.profiling_iterations

    print(
        f"Benchmark Decoding Attention: head q: {head_q}, head k: {head_k}, dim: {dim}, dim v: {dim_v}, is alibi: {is_alibi}, is bf16: {is_bf16}, profiling iterations: {profiling_iterations}")

    benchmark_seq(head_q, head_k, dim, dim_v, is_alibi,
                  is_bf16, profiling_iterations)
    benchmark_batch(head_q, head_k, dim, dim_v, is_alibi,
                    is_bf16, profiling_iterations)


if __name__ == "__main__":
    main()
