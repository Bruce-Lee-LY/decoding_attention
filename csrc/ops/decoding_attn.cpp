// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:14:13 on Tue, Oct 31, 2023
//
// Description: decoding attn

#include "decoding_attn.h"

#include <cmath>

#include "decoding_attn/decoding.h"

void set_dmha_fwd_params(DecodingParams &params, void *q, void *k, void *v, void *o, int *cu_seq_k, size_t max_seq_k,
                         size_t batch, size_t head_q, size_t head_k, size_t dim, size_t dim_v, bool is_alibi,
                         bool is_bf16, cudaStream_t stream) {
    DA_CHECK(q);
    DA_CHECK(k);
    DA_CHECK(o);
    DA_CHECK(cu_seq_k);
    DA_CHECK_EQ(head_q % head_k, 0);
    DA_CHECK_LE(dim, 576);
    DA_CHECK_LE(dim_v, 512);

    // Reset the parameters
    memset(&params, 0, sizeof(params));

    // Set the pointers and strides.
    params.q_ptr = q;
    params.k_ptr = k;
    params.v_ptr = v ? v : k;

    params.q_row_stride = head_q * dim;
    params.k_row_stride = head_k * dim;
    params.v_row_stride = head_k * dim;
    params.q_head_stride = dim;
    params.k_head_stride = dim;
    params.v_head_stride = dim_v;

    params.h = head_q;
    params.h_k = head_k;
    params.h_h_k_ratio = params.h / params.h_k;

    params.o_ptr = o;

    params.o_row_stride = head_q * dim_v;
    params.o_head_stride = dim_v;

    // Set the dimensions.
    params.b = batch;
    params.max_seq_k = max_seq_k;
    params.d = dim;
    params.d_v = dim_v;

    params.scale_softmax = 1.0 / std::sqrt(dim);

    params.cu_seq_k = cu_seq_k;

    params.is_alibi = is_alibi;
    params.is_bf16 = is_bf16;

    params.stream = stream;
}

void run_dmha_fwd(const DecodingParams &params) {
    DA_FP16_SWITCH(!params.is_bf16, [&] {
        DA_HEADDIM_SWITCH(params.d, params.d_v, [&] { run_dmha_fwd_<elem_type, head_dim, head_dim_v>(params); });
    });
}

void decoding_attn(void *q, void *k, void *v, void *o, int *cu_seq_k, size_t max_seq_k, size_t batch, size_t head_q,
                   size_t head_k, size_t dim, size_t dim_v, bool is_alibi, bool is_bf16, cudaStream_t stream) {
    DecodingParams params;
    set_dmha_fwd_params(params, q, k, v, o, cu_seq_k, max_seq_k, batch, head_q, head_k, dim, dim_v, is_alibi, is_bf16,
                        stream);
    run_dmha_fwd(params);
}
