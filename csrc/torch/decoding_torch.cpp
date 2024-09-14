// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:14:13 on Tue, Oct 31, 2023
//
// Description: decoding torch

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/nn/functional.h>
#include <torch/python.h>

#include "decoding_attn.h"

#define DA_CHECK_DEVICE(x) TORCH_CHECK(x.is_cuda(), #x " must be on CUDA")
#define DA_CHECK_SHAPE(x, ...) \
    TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define DA_CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

at::Tensor dmha_fwd(const at::Tensor &q, const at::Tensor &k, const at::Tensor &v, c10::optional<at::Tensor> &o_,
                    const at::Tensor &cu_seq_k, const int max_seq_k, const bool is_alibi) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    auto q_dtype = q.dtype();
    TORCH_CHECK(q_dtype == torch::kFloat16 || q_dtype == torch::kBFloat16,
                "Decoding-Attention only support FP16 and BF16 data type");
    bool is_bf16 = q_dtype == torch::kBFloat16;
    TORCH_CHECK(k.dtype() == q_dtype, "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q_dtype, "query and value must have the same dtype");
    TORCH_CHECK(cu_seq_k.dtype() == torch::kInt32, "cu_seq_k must have dtype int32");

    DA_CHECK_DEVICE(q);
    DA_CHECK_DEVICE(k);
    DA_CHECK_DEVICE(v);
    DA_CHECK_DEVICE(cu_seq_k);

    TORCH_CHECK(q.stride(-1) == 1, "query must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "key must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "value must have contiguous last dimension");
    DA_CHECK_CONTIGUOUS(cu_seq_k);

    const int batch = cu_seq_k.numel() - 1;
    const int total_q = q.size(0);
    const int head_q = q.size(1);
    const int dim = q.size(2);
    const int total_k = k.size(0);
    const int head_k = k.size(1);
    TORCH_CHECK(batch > 0, "batch size must be positive");
    TORCH_CHECK(dim <= 256, "dim should be less than 256");
    TORCH_CHECK(head_q % head_k == 0, "number of heads in key/value must divide number of heads in query");

    DA_CHECK_SHAPE(q, total_q, head_q, dim);
    DA_CHECK_SHAPE(k, total_k, head_k, dim);
    DA_CHECK_SHAPE(v, total_k, head_k, dim);
    DA_CHECK_SHAPE(cu_seq_k, batch + 1);

    at::Tensor o;
    if (o_.has_value()) {
        o = o_.value();
        TORCH_CHECK(o.dtype() == q_dtype, "Output must have the same dtype as inputs");
        DA_CHECK_DEVICE(o);
        TORCH_CHECK(o.stride(-1) == 1, "Output tensor must have contiguous last dimension");
        DA_CHECK_SHAPE(o, total_q, head_q, dim);
    } else {
        o = torch::empty_like(q);
    }

    decoding_attn(q.data_ptr(), k.data_ptr(), v.data_ptr(), o.data_ptr(), reinterpret_cast<int *>(cu_seq_k.data_ptr()),
                  max_seq_k, batch, head_q, head_k, dim, is_alibi, is_bf16, stream);

    return o;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Decoding-Attention";
    m.def("fwd", &dmha_fwd, "Forward pass");
}
