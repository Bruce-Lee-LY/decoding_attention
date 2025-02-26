# Copyright 2023. All Rights Reserved.
# Author: Bruce-Lee-LY
# Date: 21:14:13 on Tue, Oct 31, 2023
#
# Description: setup decoding attention

# !/usr/bin/python3
# coding=utf-8

import os
import sys
from pathlib import Path
from setuptools import setup, find_packages

import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


def append_nvcc_threads(nvcc_extra_args):
    nvcc_threads = os.getenv("NVCC_THREADS") or "4"
    return nvcc_extra_args + ["--threads", nvcc_threads]


class NinjaBuildExtension(BuildExtension):
    def __init__(self, *args, **kwargs) -> None:
        # do not override env MAX_JOBS if already exists
        if not os.environ.get("MAX_JOBS"):
            import psutil

            # calculate the maximum allowed NUM_JOBS based on cores
            max_num_jobs_cores = max(1, os.cpu_count() // 2)

            # calculate the maximum allowed NUM_JOBS based on free memory
            free_memory_gb = psutil.virtual_memory().available / \
                (1024 ** 3)  # free memory in GB
            # each JOB peak memory cost is ~8-9GB when threads = 4
            max_num_jobs_memory = int(free_memory_gb / 9)

            # pick lower value of jobs based on cores vs memory metric to minimize oom and swap usage during compilation
            max_jobs = max(1, min(max_num_jobs_cores, max_num_jobs_memory))
            os.environ["MAX_JOBS"] = str(max_jobs)

        super().__init__(*args, **kwargs)


print("python version: {}".format(sys.version))
print("torch version: {}".format(torch.__version__))

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))

# Check, if ATen/CUDAGeneratorImpl.h is found, otherwise use ATen/cuda/CUDAGeneratorImpl.h
# See https://github.com/pytorch/pytorch/pull/70650
generator_flag = []
torch_dir = torch.__path__[0]
if os.path.exists(os.path.join(torch_dir, "include", "ATen", "CUDAGeneratorImpl.h")):
    generator_flag = ["-DOLD_GENERATOR_PATH"]

cc_flag = []
cc_flag.append("-gencode")
cc_flag.append("arch=compute_80,code=sm_80")
cc_flag.append("-gencode")
cc_flag.append("arch=compute_86,code=sm_86")
cc_flag.append("-gencode")
cc_flag.append("arch=compute_89,code=sm_89")
cc_flag.append("-gencode")
cc_flag.append("arch=compute_90,code=sm_90")

# HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
# torch._C._GLIBCXX_USE_CXX11_ABI
# https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
torch._C._GLIBCXX_USE_CXX11_ABI = False

ext_modules = []
ext_modules.append(
    CUDAExtension(
        name="decoding_attn_cuda",
        sources=[
            "csrc/torch/decoding_torch.cpp",
            "csrc/ops/decoding_attn.cpp",
            "csrc/kernel/decoding_attn/decoding_fwd_hd64_hdv64.cu",
            "csrc/kernel/decoding_attn/decoding_fwd_hd96_hdv96.cu",
            "csrc/kernel/decoding_attn/decoding_fwd_hd128_hdv128.cu",
            "csrc/kernel/decoding_attn/decoding_fwd_hd256_hdv256.cu",
            "csrc/kernel/decoding_attn/decoding_fwd_hd576_hdv512.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"] + generator_flag,
            "nvcc": append_nvcc_threads(
                [
                    "-O3",
                    "-std=c++17",
                    "-U__CUDA_NO_HALF_OPERATORS__",
                    "-U__CUDA_NO_HALF_CONVERSIONS__",
                    "-U__CUDA_NO_HALF2_OPERATORS__",
                    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                    "--expt-relaxed-constexpr",
                    "--expt-extended-lambda",
                    "--use_fast_math",
                    # "--ptxas-options=-v",
                    # "--ptxas-options=-O2",
                    # "-lineinfo",
                ]
                + generator_flag
                + cc_flag
            ),
        },
        include_dirs=[
            Path(this_dir) / "csrc" / "common",
            Path(this_dir) / "csrc" / "kernel",
            Path(this_dir) / "csrc" / "ops",
        ],
    )
)


setup(
    name="decoding_attn",
    version="0.1.0",
    packages=find_packages(
        exclude=(
            "csrc",
            "decodng_attn",
        )
    ),
    author="Bruce-Lee-LY",
    description="Decoding Attention",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Bruce-Lee-LY/decoding_attention",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"build_ext": NinjaBuildExtension},
    python_requires=">=3.8",
    install_requires=[
        "torch",
    ],
    setup_requires=[
        "psutil",
        "ninja",
    ],
)
