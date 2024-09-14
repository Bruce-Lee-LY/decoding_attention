// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:14:13 on Tue, Oct 31, 2023
//
// Description: block info

#pragma once

#include "cuda_runtime_api.h"

struct DecodingBlockInfo {
    template <typename Params>
    __device__ DecodingBlockInfo(const Params &params, const int bidb, const int bidh)
        : b(bidb),
          h(bidh),
          h_k(h / params.h_h_k_ratio),
          sum_seq_q(b),
          sum_seq_k(params.cu_seq_k[b]),
          actual_seq_k(params.cu_seq_k[b + 1] - sum_seq_k),
          row_shift(actual_seq_k - actual_seq_q),
          h_slope(1.0 / exp2f(8.0 * (h + 1) / params.h)) {}

    inline __device__ size_t q_offset(const size_t row_stride, const size_t head_stride, const size_t dim_idx) const {
        return static_cast<size_t>(sum_seq_q) * row_stride + h * head_stride + dim_idx;
    }

    inline __device__ size_t k_offset(const size_t seq_k, const size_t row_stride, const size_t head_stride,
                                      const size_t dim_idx) const {
        return static_cast<size_t>(sum_seq_k + seq_k) * row_stride + h_k * head_stride + dim_idx;
    }

    const int b;
    const int h;
    const int h_k;
    const int sum_seq_q;
    const int sum_seq_k;
    const int actual_seq_k;
    const int row_shift;
    const float h_slope;

    const int actual_seq_q = 1;
};
