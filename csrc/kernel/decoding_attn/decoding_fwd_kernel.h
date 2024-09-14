// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:14:13 on Tue, Oct 31, 2023
//
// Description: decoding fwd kernel

#pragma once

#include "decoding_attn/block_info.h"
#include "decoding_attn/decoding.h"
#include "decoding_attn/kernel_traits.h"

template <typename T, typename KernelTraits, bool IsAlibi>
__global__ void dmha_fwd_kernel(const DecodingParams params) {
    const DecodingBlockInfo binfo(params, blockIdx.x, blockIdx.y);
    if (binfo.actual_seq_k <= 0) {
        return;
    }

    constexpr size_t head_dim = KernelTraits::head_dim;
    constexpr size_t threads_per_block = KernelTraits::threads_per_block;
    constexpr size_t threads_per_group = KernelTraits::threads_per_group;

    constexpr size_t warp_size = KernelTraits::warp_size;
    constexpr size_t warps_per_block = KernelTraits::warps_per_block;

    constexpr size_t groups_per_warp = KernelTraits::groups_per_warp;
    constexpr size_t groups_per_block = KernelTraits::groups_per_block;

    constexpr size_t thread_copy_elem_nums = KernelTraits::thread_copy_elem_nums;

    constexpr size_t thread_elem_nums = KernelTraits::thread_elem_nums;
    constexpr size_t thread_iters = KernelTraits::thread_iters;

    constexpr unsigned int shfl_mask = KernelTraits::shfl_mask;

    const size_t warp_id = threadIdx.x / warp_size;
    const size_t lane_id = threadIdx.x % warp_size;
    const size_t group_id = lane_id / threads_per_group;
    const size_t group_lane_id = lane_id % threads_per_group;

    T *q_ptr = reinterpret_cast<T *>(params.q_ptr);
    T *k_ptr = reinterpret_cast<T *>(params.k_ptr);
    T *v_ptr = reinterpret_cast<T *>(params.v_ptr);
    T *o_ptr = reinterpret_cast<T *>(params.o_ptr);

    // S = Q * K^T
    T RQ[thread_elem_nums];

#pragma unroll
    for (size_t i = 0; i < thread_iters; ++i) {
        *(int4 *)(&RQ[i * thread_copy_elem_nums]) =
            *(int4 *)(&q_ptr[binfo.q_offset(params.q_row_stride, params.q_head_stride,
                                            (i * threads_per_group + group_lane_id) * thread_copy_elem_nums)]);
    }

    extern __shared__ float S_smem[];
    float S_max = -std::numeric_limits<float>::max();

#pragma unroll
    for (size_t base_seq_k = warp_id * groups_per_warp; base_seq_k < binfo.actual_seq_k;
         base_seq_k += groups_per_block) {
        size_t seq_k = base_seq_k + group_id;
        T RK[thread_elem_nums];

        float acc = 0.0;
        if (seq_k < binfo.actual_seq_k) {
#pragma unroll
            for (size_t i = 0; i < thread_iters; ++i) {
                *(int4 *)(&RK[i * thread_copy_elem_nums]) =
                    *(int4 *)(&k_ptr[binfo.k_offset(seq_k, params.k_row_stride, params.k_head_stride,
                                                    (i * threads_per_group + group_lane_id) * thread_copy_elem_nums)]);
            }

#pragma unroll
            for (size_t i = 0; i < thread_elem_nums; ++i) {
                if constexpr (std::is_same_v<T, half>) {
                    acc += (__half2float(RQ[i]) * __half2float(RK[i]));
                } else {
                    acc += (__bfloat162float(RQ[i]) * __bfloat162float(RK[i]));
                }
            }
        }

#pragma unroll
        for (size_t i = threads_per_group / 2; i >= 1; i /= 2) {
            acc += __shfl_xor_sync(shfl_mask, acc, i);
        }

        if (group_lane_id == 0 && seq_k < binfo.actual_seq_k) {
            acc *= params.scale_softmax;

            if (IsAlibi) {
                acc += (binfo.h_slope * (static_cast<int>(seq_k) - binfo.actual_seq_q - binfo.row_shift));
            }

            S_smem[seq_k] = acc;
            S_max = fmaxf(acc, S_max);
        }
    }

    // P = Softmax(S)
    __shared__ float softmax_smem[warps_per_block];

#pragma unroll
    for (size_t i = warp_size / 2; i >= 1; i /= 2) {
        S_max = fmaxf(S_max, __shfl_xor_sync(shfl_mask, S_max, i));
    }

    if (lane_id == 0) {
        softmax_smem[warp_id] = S_max;
    }

    __syncthreads();

    if (lane_id < warps_per_block) {
        S_max = softmax_smem[lane_id];
    } else {
        S_max = -std::numeric_limits<float>::max();
    }

#pragma unroll
    for (size_t i = warps_per_block / 2; i >= 1; i /= 2) {
        S_max = fmaxf(S_max, __shfl_xor_sync(shfl_mask, S_max, i));
    }

    S_max = __shfl_sync(shfl_mask, S_max, 0);

    float exp_sum = 0.0;
#pragma unroll
    for (size_t seq_k = threadIdx.x; seq_k < binfo.actual_seq_k; seq_k += threads_per_block) {
        S_smem[seq_k] -= S_max;
        S_smem[seq_k] = exp(S_smem[seq_k]);
        exp_sum += S_smem[seq_k];
    }

#pragma unroll
    for (size_t i = warp_size / 2; i >= 1; i /= 2) {
        exp_sum += __shfl_xor_sync(shfl_mask, exp_sum, i);
    }

    if (lane_id == 0) {
        softmax_smem[warp_id] = exp_sum;
    }

    __syncthreads();

    if (lane_id < warps_per_block) {
        exp_sum = softmax_smem[lane_id];
    }

#pragma unroll
    for (size_t i = warps_per_block / 2; i >= 1; i /= 2) {
        exp_sum += __shfl_xor_sync(shfl_mask, exp_sum, i);
    }
    exp_sum = __shfl_sync(shfl_mask, exp_sum, 0);

#pragma unroll
    for (size_t seq_k = threadIdx.x; seq_k < binfo.actual_seq_k; seq_k += threads_per_block) {
        S_smem[seq_k] /= exp_sum;
    }

    __syncthreads();

    // O = P * V
    T RV[thread_elem_nums];
    float RO[thread_elem_nums];

    memset(RO, 0, sizeof(RO));

#pragma unroll
    for (size_t base_seq_k = warp_id * groups_per_warp; base_seq_k < binfo.actual_seq_k;
         base_seq_k += groups_per_block) {
        size_t seq_k = base_seq_k + group_id;

        if (seq_k < binfo.actual_seq_k) {
#pragma unroll
            for (size_t i = 0; i < thread_iters; ++i) {
                *(int4 *)(&RV[i * thread_copy_elem_nums]) =
                    *(int4 *)(&v_ptr[binfo.k_offset(seq_k, params.v_row_stride, params.v_head_stride,
                                                    (i * threads_per_group + group_lane_id) * thread_copy_elem_nums)]);
            }

#pragma unroll
            for (size_t i = 0; i < thread_elem_nums; ++i) {
                if constexpr (std::is_same_v<T, half>) {
                    RO[i] += (S_smem[seq_k] * __half2float(RV[i]));
                } else {
                    RO[i] += (S_smem[seq_k] * __bfloat162float(RV[i]));
                }
            }
        }
    }

#pragma unroll
    for (size_t i = 0; i < thread_elem_nums; ++i) {
#pragma unroll
        for (size_t j = threads_per_group; j <= warp_size / 2; j *= 2) {
            RO[i] += __shfl_xor_sync(shfl_mask, RO[i], j);
        }
    }

    __syncthreads();

#pragma unroll
    for (size_t i = threadIdx.x; i < head_dim; i += threads_per_block) {
        S_smem[i] = 0.0;
    }

    __syncthreads();

    if (lane_id < threads_per_group) {
#pragma unroll
        for (size_t i = 0; i < thread_iters; ++i) {
#pragma unroll
            for (size_t j = 0; j < thread_copy_elem_nums; ++j) {
                atomicAdd(S_smem + (i * threads_per_group + lane_id) * thread_copy_elem_nums + j,
                          RO[i * thread_copy_elem_nums + j]);
            }
        }
    }

    __syncthreads();

#pragma unroll
    for (size_t i = threadIdx.x; i < head_dim; i += threads_per_block) {
        if constexpr (std::is_same_v<T, half>) {
            o_ptr[binfo.q_offset(params.o_row_stride, params.o_head_stride, i)] = __float2half(S_smem[i]);
        } else {
            o_ptr[binfo.q_offset(params.o_row_stride, params.o_head_stride, i)] = __float2bfloat16(S_smem[i]);
        }
    }
}
