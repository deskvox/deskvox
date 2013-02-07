// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

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
