// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:14:13 on Tue, Oct 31, 2023
//
// Description: decoding fwd hdim64

#include "decoding_attn/decoding_fwd_launch_template.h"

template <>
void run_dmha_fwd_<half, 64>(const DecodingParams &params) {
    dmha_fwd<half, 64, 256, 4>(params);
}

template <>
void run_dmha_fwd_<__nv_bfloat16, 64>(const DecodingParams &params) {
    dmha_fwd<__nv_bfloat16, 64, 256, 4>(params);
}
