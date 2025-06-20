// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:14:13 on Tue, Oct 31, 2023
//
// Description: decoding attn

#include <cstddef>

#include "cuda_runtime_api.h"

/**
 * @brief decoding attn api: sopprt MHA/MQA/GQA/MLA
 *
 * @param q [total_q * head_q * dim]
 * @param k [total_k * head_k * dim], kv_c_and_k_pe_cache for MLA
 * @param v [total_k * head_k * dim_v], nullptr for MLA
 * @param o [total_q * head_q * dim_v]
 * @param cu_seq_k [batch + 1]
 * @param max_seq_k
 * @param batch
 * @param head_q
 * @param head_k
 * @param dim
 * @param dim_v
 * @param is_alibi
 * @param is_bf16
 * @param stream
 */
void decoding_attn(void *q, void *k, void *v, void *o, int *cu_seq_k, size_t max_seq_k, size_t batch, size_t head_q,
                   size_t head_k, size_t dim, size_t dim_v, bool is_alibi, bool is_bf16, cudaStream_t stream);
