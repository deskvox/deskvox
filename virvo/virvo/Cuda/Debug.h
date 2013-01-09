// Debug.h


#pragma once


#include <cuda_runtime_api.h>

#include <stdio.h>


#ifndef VV_CUDA_DEBUGGING
#   ifndef NDEBUG
#       define VV_CUDA_DEBUGGING 1 // Enable in debug builds.
#   else
#       define VV_CUDA_DEBUGGING 0
#   endif
#endif

#if VV_CUDA_DEBUGGING
#   define VV_CUDA_CALL(X) virvo::cuda::debug_call(X, __FILE__, __LINE__)
#else
#   define VV_CUDA_CALL(X) X
#endif


namespace virvo
{
namespace cuda
{


    inline void debug_report_error(cudaError_t err, char const* file, int line)
    {
        // Just print a message and continue execution.
        fprintf(stderr, "%s(%d) : CUDA error: %s\n", file, line, cudaGetErrorString(err));
    }


    inline cudaError_t debug_call(cudaError_t err, char const* file, int line)
    {
        if (err != cudaSuccess)
            debug_report_error(err, file, line);

        return err;
    }


} // namespace cuda
} // namespace virvo
