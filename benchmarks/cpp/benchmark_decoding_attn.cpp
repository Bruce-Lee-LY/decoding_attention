// Copyright 2023. All Rights Reserved.
// Author: Bruce-Lee-LY
// Date: 21:08:30 on Sun, Aug 27, 2023
//
// Description: benchmark decoding attn using cpp api

#include "decoding_attn.h"
#include "gflags/gflags.h"
#include "omp.h"
#include "tester.h"

DEFINE_uint32(b, 2, "batch size");
DEFINE_uint32(sq, 1, "q seq len");
DEFINE_uint32(sk, 256, "kv seq len");
DEFINE_uint32(hq, 32, "q head num");
DEFINE_uint32(hk, 32, "kv head num");
DEFINE_uint32(d, 128, "head dim");
DEFINE_uint32(dv, 128, "head dim v");
DEFINE_bool(is_causal, false, "causal mask");
DEFINE_bool(is_alibi, false, "enable alibi");
DEFINE_bool(is_bf16, false, "data type of q, k, v and o");
DEFINE_uint32(warmup_iterations, 1, "warmup iteration numbers and average the result");
DEFINE_uint32(profiling_iterations, 10, "profiling iteration numbers and average the result");
DEFINE_uint32(sleep_duration, 100, "sleep_milliseconds between profiling");
DEFINE_bool(enable_check, false, "check the GPU result against the CPU result");
DEFINE_uint32(cpu_procs, omp_get_num_procs(), "processor num used of CPU");
DEFINE_uint32(gpu_rank, 0, "the used GPU rank");

template <typename T>
void test_decoding_attn(size_t batch = 2, size_t seq_q = 1, size_t seq_k = 256, size_t head_q = 32, size_t head_k = 32,
                        size_t dim = 128, size_t dim_v = 128, bool is_causal = false, bool is_alibi = false,
                        cudaStream_t stream = nullptr, size_t warmup_iterations = 1, size_t profiling_iterations = 10,
                        size_t sleep_duration = 100, bool enable_check = false) {
    Tester<T> tester(batch, seq_q, seq_k, head_q, head_k, dim, dim_v, is_causal, is_alibi, stream, warmup_iterations,
                     profiling_iterations, sleep_duration, enable_check);
    tester.evaluate(decoding_attn, "Decoding-Attention");
}

int main(int argc, char *argv[]) {
    GFLAGS_NAMESPACE::ParseCommandLineFlags(&argc, &argv, true);

    omp_set_num_threads(FLAGS_cpu_procs);
    DA_CHECK_CUDART_ERROR(cudaSetDevice(FLAGS_gpu_rank));

    cudaDeviceProp dev_prop;
    DA_CHECK_CUDART_ERROR(cudaGetDeviceProperties(&dev_prop, FLAGS_gpu_rank));
    DLOG("Decoding Attention start with %u CPU processes on the %u-th GPU: %s", FLAGS_cpu_procs, FLAGS_gpu_rank,
         dev_prop.name);

    int driver_version = 0;
    int runtime_version = 0;
    DA_CHECK_CUDART_ERROR(cudaDriverGetVersion(&driver_version));
    DA_CHECK_CUDART_ERROR(cudaRuntimeGetVersion(&runtime_version));
    DLOG("CUDA driver version / runtime version: %d.%d / %d.%d", driver_version / 1000, (driver_version % 100) / 10,
         runtime_version / 1000, (runtime_version % 100) / 10);

    DLOG("CUDA capability major/minor version number: %d.%d", dev_prop.major, dev_prop.minor);
    DLOG("%d multiprocessors, %d CUDA cores/MP: %d CUDA cores", dev_prop.multiProcessorCount,
         convert_SM_to_cores(dev_prop.major, dev_prop.minor),
         convert_SM_to_cores(dev_prop.major, dev_prop.minor) * dev_prop.multiProcessorCount);
    DLOG("GPU max clock rate: %.0f MHz (%0.2f GHz)", static_cast<double>(dev_prop.clockRate) * 1e-3,
         static_cast<double>(dev_prop.clockRate) * 1e-6);
    DLOG("Memory clock rate: %.0f MHz (%0.2f GHz)", static_cast<double>(dev_prop.memoryClockRate) * 1e-3,
         static_cast<double>(dev_prop.memoryClockRate) * 1e-6);
    DLOG("Memory bus width: %d-bit", dev_prop.memoryBusWidth);
    DLOG("Total amount of global memory: %.0f MBytes (%zu Bytes)",
         static_cast<double>(dev_prop.totalGlobalMem) / 1048576, dev_prop.totalGlobalMem);
    DLOG("Total amount of constant memory: %.0f KBytes (%zu Bytes)", static_cast<double>(dev_prop.totalConstMem) / 1024,
         dev_prop.totalConstMem);
    DLOG("Total amount of shared memory per block: %.0f KBytes (%zu Bytes)",
         static_cast<double>(dev_prop.sharedMemPerBlock) / 1024, dev_prop.sharedMemPerBlock);
    DLOG("Total shared memory per multiprocessor: %.0f KBytes (%zu Bytes)",
         static_cast<double>(dev_prop.sharedMemPerMultiprocessor) / 1024, dev_prop.sharedMemPerMultiprocessor);
    DLOG("L2 cache size: %.0f KBytes (%d Bytes)", static_cast<double>(dev_prop.l2CacheSize) / 1024,
         dev_prop.l2CacheSize);
    DLOG("Total number of registers available per block: %d", dev_prop.regsPerBlock);
    DLOG("Warp size: %d", dev_prop.warpSize);
    DLOG("Max number of threads per multiprocessor: %d", dev_prop.maxThreadsPerMultiProcessor);
    DLOG("Max number of threads per block: %d", dev_prop.maxThreadsPerBlock);
    DLOG("Max dimension size of a thread block (x,y,z): (%d, %d, %d)", dev_prop.maxThreadsDim[0],
         dev_prop.maxThreadsDim[1], dev_prop.maxThreadsDim[2]);
    DLOG("Max dimension size of a grid size (x,y,z): (%d, %d, %d)", dev_prop.maxGridSize[0], dev_prop.maxGridSize[1],
         dev_prop.maxGridSize[2]);

    cudaStream_t stream = nullptr;

    DLOG(
        "DMHA: Softmax (Q (%u x %u x %u x %u) * K^T (%u x %u x %u x %u)) * V (%u x %u x %u x %u) = O (%u x %u x %u x "
        "%u)",
        FLAGS_b, FLAGS_sq, FLAGS_hq, FLAGS_d, FLAGS_b, FLAGS_sk, FLAGS_hk, FLAGS_d, FLAGS_b, FLAGS_sk, FLAGS_hk,
        FLAGS_dv, FLAGS_b, FLAGS_sq, FLAGS_hq, FLAGS_dv);
    DLOG(
        "Profiling: is causal: %d, stream: %p, is alibi: %d, is bf16: %d, warmup iterations: %u, profiling iterations: "
        "%u, sleep duration: %u ms, enable check: %d",
        FLAGS_is_causal, stream, FLAGS_is_alibi, FLAGS_is_bf16, FLAGS_warmup_iterations, FLAGS_profiling_iterations,
        FLAGS_sleep_duration, FLAGS_enable_check);

    if (FLAGS_is_bf16) {
        test_decoding_attn<__nv_bfloat16>(FLAGS_b, FLAGS_sq, FLAGS_sk, FLAGS_hq, FLAGS_hk, FLAGS_d, FLAGS_dv,
                                          FLAGS_is_causal, FLAGS_is_alibi, stream, FLAGS_warmup_iterations,
                                          FLAGS_profiling_iterations, FLAGS_sleep_duration, FLAGS_enable_check);
    } else {
        test_decoding_attn<half>(FLAGS_b, FLAGS_sq, FLAGS_sk, FLAGS_hq, FLAGS_hk, FLAGS_d, FLAGS_dv, FLAGS_is_causal,
                                 FLAGS_is_alibi, stream, FLAGS_warmup_iterations, FLAGS_profiling_iterations,
                                 FLAGS_sleep_duration, FLAGS_enable_check);
    }

    GFLAGS_NAMESPACE::ShutDownCommandLineFlags();

    DLOG("Done");

    return 0;
}
