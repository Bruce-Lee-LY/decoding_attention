# Copyright 2023. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 21:14:13 on Tue, Oct 31, 2023
#
# Description: decoding attn interface

# !/usr/bin/python3
# coding=utf-8

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
from __future__ import with_statement

import torch

import decoding_attn_cuda


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def decoding_attn_fwd(
    q: torch.tensor,
    k: torch.tensor,
    v: torch.tensor,
    cu_seq_k: torch.tensor,
    max_seq_k: int,
    dim_v: int,
    is_alibi: bool
) -> torch.tensor:
    """
    Arguments:
        q: [total_q, head_q, dim], torch.float16 / torch.bfloat16, where total_q = total number of 
            query tokens in the batch.
        k: [total_k, head_k, dim], torch.float16 / torch.bfloat16, where total_k = total number of 
            key tokens in the batch, MLA: kv_c_and_k_pe_cache.
        v: [total_k, head_k, dim_v], torch.float16 / torch.bfloat16, where total_k = total number of 
            value tokens in the batch, MLA: None.
        cu_seq_k: [batch + 1], dtype torch.int32. The cumulative sequence lengths of the sequences 
            in the batch, used to index into k / v.
        max_seq_k: Maximum key sequence length in the batch.
        dim_v: Head dimension of v.
        is_alibi: Whether to apply alibi.
    Return:
        o: [total_q, head_q, dim_v], torch.float16 / torch.bfloat16, where total_q = total number of 
            output tokens in the batch.
    """
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    o = decoding_attn_cuda.fwd(
        q, k, v, None, cu_seq_k, max_seq_k, dim_v, is_alibi)
    return o
