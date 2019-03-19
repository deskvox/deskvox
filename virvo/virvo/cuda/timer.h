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

#ifndef VV_CUDA_TIMER_H
#define VV_CUDA_TIMER_H

#include <cuda_runtime_api.h>

namespace virvo
{

class CudaTimer
{
public:

    CudaTimer()
    {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);

        reset();
    }

    ~CudaTimer()
    {
        cudaEventDestroy(stop_);
        cudaEventDestroy(start_);
    }

    void reset()
    {
        cudaEventRecord(start_);
    }

    double elapsed() const
    {
        cudaEventRecord(stop_);
        cudaEventSynchronize(stop_);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_, stop_);
        return static_cast<double>(ms) / 1000.0;
    }

private:

    cudaEvent_t start_;
    cudaEvent_t stop_;

};

} // namespace virvo

#endif // VV_CUDA_TIMER_H
