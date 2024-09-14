// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:14:13 on Tue, Oct 31, 2023
//
// Description: decoding fwd launch template

#pragma once

#include "decoding_attn/decoding_fwd_kernel.h"

template <typename T, size_t HeadDim, size_t ThreadsPerBlock, size_t ThreadsPerGroup>
void dmha_fwd(const DecodingParams &params) {
    constexpr size_t warp_size = 32;
    constexpr size_t static_smem_size = ThreadsPerBlock / warp_size * sizeof(float);
    const size_t dynamic_smem_size = std::max(params.seq_k * sizeof(float), params.d * sizeof(float));
    const size_t total_smem_size = static_smem_size + dynamic_smem_size;
    dim3 block(ThreadsPerBlock);
    dim3 grid(params.b, params.h);

    DA_BOOL_SWITCH(params.is_alibi, IsAlibi, [&] {
        auto kernel = &dmha_fwd_kernel<T, DecodingKernelTraits<T, HeadDim, ThreadsPerBlock, ThreadsPerGroup>, IsAlibi>;
        if (total_smem_size >= 48 * 1024) {
            DA_CHECK_CUDART_ERROR(
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, total_smem_size));
        }

        kernel<<<grid, block, total_smem_size, params.stream>>>(params);
        DA_CHECK_CUDART_ERROR(cudaPeekAtLastError());
    });
}
