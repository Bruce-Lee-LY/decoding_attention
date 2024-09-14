// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:14:13 on Tue, Oct 31, 2023
//
// Description: kernel traits

#pragma once

#include <cstddef>

template <typename T, size_t HeadDim, size_t ThreadsPerBlock, size_t ThreadsPerGroup>
struct DecodingKernelTraits {
    static constexpr size_t head_dim = HeadDim;
    static constexpr size_t threads_per_block = ThreadsPerBlock;
    static constexpr size_t threads_per_group = ThreadsPerGroup;

    static constexpr size_t warp_size = 32;
    static constexpr size_t warps_per_block = threads_per_block / warp_size;

    static constexpr size_t groups_per_warp = warp_size / threads_per_group;
    static constexpr size_t groups_per_block = groups_per_warp * warps_per_block;

    static constexpr size_t thread_copy_bytes = 16;
    static constexpr size_t thread_copy_elem_nums = thread_copy_bytes / sizeof(T);

    static constexpr size_t thread_elem_nums = head_dim / threads_per_group;
    static constexpr size_t thread_iters = thread_elem_nums / thread_copy_elem_nums;

    static constexpr unsigned int shfl_mask = 0xffffffff;
};
