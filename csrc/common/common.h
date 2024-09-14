// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 20:42:28 on Sun, Feb 12, 2023
//
// Description: common macro

#pragma once

#include "cuda_bf16.h"
#include "cuda_fp16.h"
#include "logging.h"
#include "util.h"

#define DA_LIKELY(x) __builtin_expect(!!(x), 1)
#define DA_UNLIKELY(x) __builtin_expect(!!(x), 0)

#define DA_CHECK(x)                       \
    do {                                  \
        if (DA_UNLIKELY(!(x))) {          \
            DLOG("Check failed: %s", #x); \
            exit(EXIT_FAILURE);           \
        }                                 \
    } while (0)

#define DA_CHECK_EQ(x, y) DA_CHECK((x) == (y))
#define DA_CHECK_NE(x, y) DA_CHECK((x) != (y))
#define DA_CHECK_LE(x, y) DA_CHECK((x) <= (y))
#define DA_CHECK_LT(x, y) DA_CHECK((x) < (y))
#define DA_CHECK_GE(x, y) DA_CHECK((x) >= (y))
#define DA_CHECK_GT(x, y) DA_CHECK((x) > (y))

#define DA_DISALLOW_COPY_AND_ASSIGN(TypeName) \
    TypeName(const TypeName &) = delete;      \
    void operator=(const TypeName &) = delete

#define DA_CHECK_CUDART_ERROR(_expr_)                                                             \
    do {                                                                                          \
        cudaError_t _ret_ = _expr_;                                                               \
        if (DA_UNLIKELY(_ret_ != cudaSuccess)) {                                                  \
            const char *_err_str_ = cudaGetErrorName(_ret_);                                      \
            int _rt_version_ = 0;                                                                 \
            cudaRuntimeGetVersion(&_rt_version_);                                                 \
            int _driver_version_ = 0;                                                             \
            cudaDriverGetVersion(&_driver_version_);                                              \
            DLOG("CUDA Runtime API error = %04d \"%s\", runtime version: %d, driver version: %d", \
                 static_cast<int>(_ret_), _err_str_, _rt_version_, _driver_version_);             \
            exit(EXIT_FAILURE);                                                                   \
        }                                                                                         \
    } while (0)

#define DA_BOOL_SWITCH(COND, CONST_NAME, ...)         \
    [&] {                                             \
        if (COND) {                                   \
            constexpr static bool CONST_NAME = true;  \
            return __VA_ARGS__();                     \
        } else {                                      \
            constexpr static bool CONST_NAME = false; \
            return __VA_ARGS__();                     \
        }                                             \
    }()

#define DA_FP16_SWITCH(COND, ...)            \
    [&] {                                    \
        if (COND) {                          \
            using elem_type = half;          \
            return __VA_ARGS__();            \
        } else {                             \
            using elem_type = __nv_bfloat16; \
            return __VA_ARGS__();            \
        }                                    \
    }()

#define DA_HEADDIM_SWITCH(HEADDIM, ...)            \
    [&] {                                          \
        if (HEADDIM == 64) {                       \
            constexpr static size_t HeadDim = 64;  \
            return __VA_ARGS__();                  \
        } else if (HEADDIM == 96) {                \
            constexpr static size_t HeadDim = 96;  \
            return __VA_ARGS__();                  \
        } else if (HEADDIM == 128) {               \
            constexpr static size_t HeadDim = 128; \
            return __VA_ARGS__();                  \
        } else if (HEADDIM == 256) {               \
            constexpr static size_t HeadDim = 256; \
            return __VA_ARGS__();                  \
        }                                          \
    }()
