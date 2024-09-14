// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:14:13 on Tue, Oct 31, 2023
//
// Description: decoding attn

#include <cstddef>

#include "cuda_runtime_api.h"

/**
 * @brief decoding attn api
 *
 * @param q [total_q * head_q * dim]
 * @param k [total_k * head_k * dim]
 * @param v [total_k * head_k * dim]
 * @param o [total_q * head_q * dim]
 * @param cu_seq_k [batch + 1]
 * @param max_seq_k
 * @param batch
 * @param head_q
 * @param head_k
 * @param dim
 * @param is_alibi
 * @param is_bf16
 * @param stream
 */
void decoding_attn(void *q, void *k, void *v, void *o, int *cu_seq_k, size_t max_seq_k, size_t batch, size_t head_q,
                   size_t head_k, size_t dim, bool is_alibi, bool is_bf16, cudaStream_t stream);
