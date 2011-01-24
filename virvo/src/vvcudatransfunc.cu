// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 2010 University of Cologne
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

#include <iostream>
using std::cerr;
using std::endl;

#include <cmath>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "vvdebugmsg.h"
#include "vvvecmath.h"
#include "vvstopwatch.h"
#include "vvcuda.h"
#include "vvcudautils.h"
#include "vvcudatransfunc.h"

__global__ void makePreintLUTCorrectKernel(int width, uchar4 *__restrict__ preIntTable,
        float thickness, float min, float max, const float4 *__restrict__ rgba)
{
  const int minLookupSteps = 2;
  const int addLookupSteps = 1;

  const int sb = blockIdx.x;
  const int sf = threadIdx.x;

  int n=minLookupSteps+addLookupSteps*abs(sb-sf);
  float stepWidth = 1.f/n;
  float r=0.f, g=0.f, b=0.f, tau=0.f;
  for (int i=0; i<n; ++i)
  {
      const float s = sf+(sb-sf)*i*stepWidth;
      const int is = (int)s;
      const float fract_s = s-floorf(s);
      const float tauc = thickness*stepWidth*(rgba[is].w*fract_s+rgba[is+1].w*(1.f-fract_s));
      const float e_tau = expf(-tau);
#ifdef STANDARD
      /* standard optical model: r,g,b densities are multiplied with opacity density */
      const float rc = e_tau*tauc*(rgba[is].x*fract_s+rgba[is+1].x*(1.f-fract_s));
      const float gc = e_tau*tauc*(rgba[is].y*fract_s+rgba[is+1].y*(1.f-fract_s));
      const float bc = e_tau*tauc*(rgba[is].z*fract_s+rgba[is+1].z*(1.f-fract_s));

#else
      /* Willhelms, Van Gelder optical model: r,g,b densities are not multiplied */
      const float rc = e_tau*stepWidth*(rgba[is].x*fract_s+rgba[is+1].x*(1.f-fract_s));
      const float gc = e_tau*stepWidth*(rgba[is].y*fract_s+rgba[is+1].y*(1.f-fract_s));
      const float bc = e_tau*stepWidth*(rgba[is].z*fract_s+rgba[is+1].z*(1.f-fract_s));
#endif

      r = r+rc;
      g = g+gc;
      b = b+bc;
      tau = tau + tauc;
  }

  clamp(r);
  clamp(g);
  clamp(b);

  preIntTable[sf*width+sb] = make_uchar4(r*255.99f, g*255.99f, b*255.99f, (1.f-expf(-tau))*255.99f);
}

bool makePreintLUTCorrectCuda(int width, uchar *preIntTable, float thickness, float min, float max, const float *rgba)
{
    float4 *d_rgba = NULL;
    uchar4 *d_preIntTable = NULL;

    bool ok = true;
    vvCuda::checkError(&ok, cudaMalloc(&d_rgba, sizeof(float4)*(width+1)), "cudaMalloc d_rgba");
    if(ok)
        vvCuda::checkError(&ok, cudaMalloc(&d_preIntTable, sizeof(uchar4)*width*width), "cudaMalloc d_preIntTable");
    if(ok)
        vvCuda::checkError(&ok, cudaMemcpy(d_rgba, rgba, sizeof(float4)*(width+1), cudaMemcpyHostToDevice), "cudaMemcpy rgba");

    if(ok)
        makePreintLUTCorrectKernel<<<width, width>>>(width, d_preIntTable, thickness, min, max, d_rgba);

    if(ok)
        vvCuda::checkError(&ok, cudaMemcpy(preIntTable, d_preIntTable, sizeof(uchar4)*width*width, cudaMemcpyDeviceToHost), "cudaMemcpy preIntTable");

    vvCuda::checkError(&ok, cudaFree(d_rgba), "cudaFree d_rgba");
    vvCuda::checkError(&ok, cudaFree(d_preIntTable), "cudaFree d_preIntTable");

    return ok;
}
