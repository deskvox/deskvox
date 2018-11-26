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

#ifndef __CUDACC_EXTENDED_LAMBDA__
#error "Compile w/ option --expt-extended-lambda"
#endif

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#undef MATH_NAMESPACE
#include <visionaray/math/detail/math.h> // div_up
#include <visionaray/math/aabb.h>
#include <visionaray/morton.h>
#undef MATH_NAMESPACE

#include "../cuda/timer.h"
#include "../vvvoldesc.h"
#include "lbvh.h"

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Compact brick data structure. max-corner is explicitly given by adding global brick-size
// to min-corner!
//

struct VSNRAY_ALIGN(16) Brick
{
  int min_corner[3];
  int is_empty = true;
};


//-------------------------------------------------------------------------------------------------
// Kernels
//

template <typename TransfuncTex>
__global__ void findNonEmptyBricks(const float* voxels, TransfuncTex transfunc, Brick* bricks)
{
  unsigned brick_index = blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x;
  unsigned brick_offset = brick_index * blockDim.x * blockDim.y * blockDim.z;

  unsigned index = brick_offset + threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;

  __shared__ int shared_empty;
  shared_empty = true;

  __syncthreads();

  bool empty = tex1D(transfunc, voxels[index]).w < 0.0001f;
  // All threads in block vote
  if (!empty)
    atomicExch(&shared_empty, false);

  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
  {
    bricks[brick_index].min_corner[0] = blockIdx.x;
    bricks[brick_index].min_corner[1] = blockIdx.y;
    bricks[brick_index].min_corner[2] = blockIdx.z;

    if (!shared_empty)
      atomicExch(&bricks[brick_index].is_empty, false);
  }
}


//-------------------------------------------------------------------------------------------------
// BVH private implementation
//

struct BVH::Impl
{
  vec3i vox;
  vec3 dist;
  float scale;
  // Brickwise (8x8x8) sorted on a z-order curve, "natural" layout inside!
  thrust::device_vector<float> voxels;
};


//-------------------------------------------------------------------------------------------------
// BVH
//

BVH::BVH()
  : impl_(new Impl)
{
}

BVH::~BVH()
{
}

void BVH::updateVolume(vvVolDesc const& vd, int channel)
{
  impl_->vox = vec3i(vd.vox.x, vd.vox.y, vd.vox.z);
  impl_->dist = vec3(vd.getDist().x, vd.getDist().y, vd.getDist().z);
  impl_->scale = vd._scale;

  vec3i brick_size(8,8,8);

  vec3i num_bricks(div_up(impl_->vox[0], brick_size.x),
      div_up(impl_->vox[1], brick_size.y),
      div_up(impl_->vox[2], brick_size.z));

  size_t num_voxels = num_bricks.x*brick_size.x * num_bricks.y*brick_size.y * num_bricks.z*brick_size.z;

  thrust::host_vector<float> host_voxels(num_voxels);

  for (int bz = 0; bz < num_bricks.z; ++bz)
  {
    for (int by = 0; by < num_bricks.y; ++by)
    {
      for (int bx = 0; bx < num_bricks.x; ++bx)
      {
        // Brick index
        int brick_index = bz * num_bricks.x * num_bricks.y + by * num_bricks.x + bx;
        // Brick offset in voxels array
        int brick_offset = brick_index * brick_size.x * brick_size.y * brick_size.z;

        for (int zz = 0; zz < brick_size.z; ++zz)
        {
          for (int yy = 0; yy < brick_size.y; ++yy)
          {
            for (int xx = 0; xx < brick_size.x; ++xx)
            {
              // Index into voxels array
              int index = brick_offset + zz * brick_size.x * brick_size.y + yy * brick_size.x + xx;

              // Indices into voldesc
              int x = bx * brick_size.x + xx;
              int y = by * brick_size.y + yy;
              int z = bz * brick_size.z + zz;

              if (x < impl_->vox[0] && y < impl_->vox[1] && z < impl_->vox[2])
              {
                host_voxels[index] = vd.getChannelValue(vd.getCurrentFrame(),
                    x,
                    y,
                    z,
                    channel);
              }
              else
                host_voxels[index] = 0.f;
            }
          }
        }
      }
    }
  }

  impl_->voxels.resize(host_voxels.size());
  thrust::copy(host_voxels.begin(), host_voxels.end(), impl_->voxels.begin());
}

void BVH::updateTransfunc(BVH::TransfuncTex transfunc)
{
  cuda_texture<visionaray::vec4, 1> cuda_transfunc(transfunc.data(),
      transfunc.width(),
      transfunc.get_address_mode(),
      transfunc.get_filter_mode());

  dim3 block_size(8, 8, 8);
  dim3 grid_size(div_up(impl_->vox[0], (int)block_size.x),
                 div_up(impl_->vox[1], (int)block_size.y),
                 div_up(impl_->vox[2], (int)block_size.z));

  // Identify non-empty bricks
  thrust::device_vector<Brick> bricks(grid_size.x * grid_size.y * grid_size.z);

  virvo::CudaTimer t;
  findNonEmptyBricks<<<grid_size, block_size>>>(
      thrust::raw_pointer_cast(impl_->voxels.data()),
      cuda_texture_ref<visionaray::vec4, 1>(cuda_transfunc),
      thrust::raw_pointer_cast(bricks.data()));
  std::cout << t.elapsed() << '\n';
  t.reset();

  // Compact non-empty bricks to the left of the list
  thrust::device_vector<Brick> compact_bricks(grid_size.x * grid_size.y * grid_size.z);

  auto last = thrust::copy_if(
      thrust::device,
      bricks.begin(),
      bricks.end(),
      compact_bricks.begin(),
      [] __device__ (Brick b) { return !b.is_empty; });
  std::cout << t.elapsed() << '\n';
  t.reset();

  thrust::stable_sort(
      thrust::device,
      compact_bricks.begin(),
      last,
      [] __device__ (Brick l, Brick r)
      {
        auto ml = morton_encode3D(l.min_corner[0], l.min_corner[1], l.min_corner[2]);
        auto mr = morton_encode3D(r.min_corner[0], r.min_corner[1], r.min_corner[2]);
        return ml < mr;
      });
  std::cout << t.elapsed() << '\n';
}

void BVH::renderGL(vvColor color) const
{
}

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
