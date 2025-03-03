# Copyright 2023. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 21:14:13 on Tue, Oct 31, 2023
#
# Description: test decoding attn python api

# !/usr/bin/python3
# coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement

import math
import pytest
import torch
from torch.nn import functional as F

from decoding_attn import decoding_attn_fwd


def get_cu_seq(seqs: torch.tensor) -> torch.tensor:
    """
    Arguments:
        seqs: [batch], dtype torch.int32, sequence length of each batch.
    Return:
        cu_seq: [batch + 1], dtype torch.int32. The cumulative sequence lengths of the sequences 
            in the batch.
    """
    return F.pad(seqs.cumsum(dim=0, dtype=torch.int32), (1, 0))


# sopprt MHA/MQA/GQA/MLA
def attn_cpu(q: torch.tensor, k: torch.tensor, v_: torch.tensor, dim_v: int) -> torch.tensor:
    """
    Arguments:
        q: [batch, seq_q, head_q, dim]
        k: [batch, seq_k, head_k, dim], MLA: kv_c_and_k_pe_cache.
        v: [batch, seq_k, head_k, dim_v], MLA: None.
    Return:
        o: [batch, seq_q, head_q, dim_v]
    """
    v = v_ if v_ is not None else k[..., :dim_v]
    head_q = q.shape[2]
    dim = q.shape[3]
    head_k = k.shape[2]
    head_ratio = head_q // head_k
    qt = q.transpose(1, 2)
    kt = k.transpose(1, 2).repeat_interleave(head_ratio, dim=1)
    vt = v.transpose(1, 2).repeat_interleave(head_ratio, dim=1)
    s = torch.matmul(qt, kt.transpose(-2, -1)) / math.sqrt(dim)
    p = F.softmax(s, dim=-1)
    o = torch.matmul(p, vt)

    return o.transpose(1, 2)


# test MHA/GQA
@pytest.mark.parametrize("head_q", [32, 64])
@pytest.mark.parametrize("head_k", [8, 32])
@pytest.mark.parametrize("dim", [64, 96, 128, 256])
@pytest.mark.parametrize("batch", [1, 2, 16])
@pytest.mark.parametrize("seq_k", [1, 128, 512])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_decoding_attn_fwd(head_q, head_k, dim, batch, seq_k, dtype):
    seq_q = 1
    total_q = batch * seq_q
    total_k = batch * seq_k
    is_alibi = False

    q = torch.randn(total_q, head_q, dim,
                    device=torch.device('cuda'), dtype=dtype)
    k = torch.randn(total_k, head_k, dim,
                    device=torch.device('cuda'), dtype=dtype)
    v = torch.randn(total_k, head_k, dim,
                    device=torch.device('cuda'), dtype=dtype)

    cu_seq_k = get_cu_seq(torch.full(
        (batch,), seq_k, dtype=torch.int32, device=torch.device('cuda')))

    q4 = q.reshape(batch, seq_q, head_q, dim)
    k4 = k.reshape(batch, seq_k, head_k, dim)
    v4 = v.reshape(batch, seq_k, head_k, dim)

    attn = attn_cpu(q4, k4, v4, dim)
    output = attn.reshape(total_q, head_q, dim)
    print(f"Attn-CPU output: {output}")

    da_output = decoding_attn_fwd(q, k, v, cu_seq_k, seq_k, dim, is_alibi)
    print(f"Decoding-Attention output: {da_output}")

    assert (output - da_output).abs().mean().item() <= 5e-3


@pytest.mark.parametrize("head_q", [16, 32, 64, 128])
@pytest.mark.parametrize("head_k", [1])
@pytest.mark.parametrize("dim", [576])
@pytest.mark.parametrize("dim_v", [512])
@pytest.mark.parametrize("batch", [1, 2, 16])
@pytest.mark.parametrize("seq_k", [1, 128, 512])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_decoding_mla_fwd(head_q, head_k, dim, dim_v, batch, seq_k, dtype):
    seq_q = 1
    total_q = batch * seq_q
    total_k = batch * seq_k
    is_alibi = False

    q = torch.randn(total_q, head_q, dim,
                    device=torch.device('cuda'), dtype=dtype)
    k = torch.randn(total_k, head_k, dim,
                    device=torch.device('cuda'), dtype=dtype)

    cu_seq_k = get_cu_seq(torch.full(
        (batch,), seq_k, dtype=torch.int32, device=torch.device('cuda')))

    q4 = q.reshape(batch, seq_q, head_q, dim)
    k4 = k.reshape(batch, seq_k, head_k, dim)

    attn = attn_cpu(q4, k4, None, dim_v)
    output = attn.reshape(total_q, head_q, dim_v)
    print(f"MLA-CPU output: {output}")

    da_output = decoding_attn_fwd(q, k, None, cu_seq_k, seq_k, dim_v, is_alibi)
    print(f"Decoding-Attention output: {da_output}")

    assert (output - da_output).abs().mean().item() <= 5e-3


def main():
    test_decoding_attn_fwd(32, 32, 128, 2, 128, torch.float16)
    test_decoding_mla_fwd(128, 1, 576, 512, 2, 128, torch.float16)


if __name__ == "__main__":
    main()
