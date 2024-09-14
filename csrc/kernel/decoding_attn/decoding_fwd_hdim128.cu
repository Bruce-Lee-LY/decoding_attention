// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:14:13 on Tue, Oct 31, 2023
//
// Description: decoding fwd hdim128

#include "decoding_attn/decoding_fwd_launch_template.h"

template <>
void run_dmha_fwd_<half, 128>(const DecodingParams &params) {
    if (params.b <= 4) {
        dmha_fwd<half, 128, 256, 8>(params);
    } else {
        dmha_fwd<half, 128, 128, 16>(params);
    }
}

template <>
void run_dmha_fwd_<__nv_bfloat16, 128>(const DecodingParams &params) {
    if (params.b <= 4) {
        dmha_fwd<__nv_bfloat16, 128, 256, 8>(params);
    } else {
        dmha_fwd<__nv_bfloat16, 128, 128, 16>(params);
    }
}
