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

#include <iostream>
#include <ostream>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#undef MATH_NAMESPACE

#include <visionaray/math/detail/math.h>
#include <visionaray/math/aabb.h>
#include <visionaray/math/axis.h>
#include <visionaray/math/forward.h>
#include <visionaray/math/io.h>
#include <visionaray/math/vector.h>

#undef MATH_NAMESPACE

#include <virvo/vvclock.h>
#include <virvo/vvvoldesc.h>

#include "cudakdtree.h"

using namespace visionaray;

#define BUILD_TIMING 1

//-------------------------------------------------------------------------------------------------
// CUDA Summed-volume table
//

template <typename Tex, typename T>
__global__ void svt_apply_transfunc(Tex transfunc,
      T* data,
      const float* voxels,
      int width,
      int height,
      int depth)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= width || y >= height || z >= depth)
    return;

  int index = z * width * height + y * width + x;
  if (tex1D(transfunc, voxels[index]).w < 0.0001)
    data[index] = T(0);
  else
    data[index] = T(1);
}

template <typename T>
__global__ void svt_build_x(T* data, int width, int height, int depth)
{
  int y = blockIdx.x * blockDim.x + threadIdx.x;
  int z = blockIdx.y * blockDim.y + threadIdx.y;

  if (y >= height || z >= depth)
    return;

  int off = 16;//32;

  for (int bx = 0; bx < width; bx += off)
  {
    for (int x = bx + 1; x < min(bx + off, width); ++x)
    {
      int i1 = z * width * height + y * width + x;
      int i2 = z * width * height + y * width + (x - 1);
      data[i1] += data[i2];
    }
  }
}

template <typename T>
__global__ void svt_build_y(T* data, int width, int height, int depth)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int z = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || z >= depth)
    return;

  int off = 8;//32;

  for (int by = 0; by < height; by += off)
  {
    for (int y = by + 1; y < min(by + off, height); ++y)
    {
      int i1 = z * width * height + y * width + x;
      int i2 = z * width * height + (y - 1) * width + x;
      data[i1] += data[i2];
    }
  }
}

template <typename T>
__global__ void svt_build_z(T* data, int width, int height, int depth)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  int off = 8;//24;

  for (int bz = 0; bz < depth; bz += off)
  {
    for (int z = bz + 1; z < min(bz + off, depth); ++z)
    {
      int i1 = z * width * height + y * width + x;
      int i2 = (z - 1) * width * height + y * width + x;
      data[i1] += data[i2];
    }
  }
}

template <typename T>
__global__ void svt_build(T* data, int width, int height, int depth)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x * 2;
  int y = blockIdx.y * blockDim.y + threadIdx.y * 2;
  int z = blockIdx.z * blockDim.z + threadIdx.z * 2;

  if (x >= width || y >= height || z >= depth)
    return;

#define BX 16
#define BY 8
#define BZ 8

  __shared__ T smem[BX*2][BY*2][BZ*2];

  // Copy 2x2x2 neighborhood to shared memory

  #pragma unroll
  for (int k = 0; k < 2; ++k)
  {
    for (int j = 0; j < 2; ++j)
    {
      for (int i = 0; i < 2; ++i)
      {
        int index = (z+k)* width * height + (y+j) * width + (x+i);
        smem[threadIdx.x + i][threadIdx.y + j][threadIdx.z + k] = data[index];
      }
    }
  }

  __syncthreads();

  // Reduction over x
  for (int l = 1; l < BX; l *= 2)
  {
    smem[threadIdx.x + 1][threadIdx.y][threadIdx.z] += smem[threadIdx.x][threadIdx.y][threadIdx.z];
  }

  __syncthreads();

  // Reduction over y
  for (int l = 1; l < BY; l *= 2)
  {
    smem[threadIdx.x][threadIdx.y + 1][threadIdx.z] += smem[threadIdx.x][threadIdx.y][threadIdx.z];
  }

  __syncthreads();

  // Reduction over z
  for (int l = 1; l < BZ; l *= 2)
  {
    smem[threadIdx.x][threadIdx.y][threadIdx.z + 1] += smem[threadIdx.x][threadIdx.y][threadIdx.z];
  }

  __syncthreads();

  // Copy back to global memory

  #pragma unroll
  for (int k = 0; k < 2; ++k)
  {
    for (int j = 0; j < 2; ++j)
    {
      for (int i = 0; i < 2; ++i)
      {
        int index = (z+k)* width * height + (y+j) * width + (x+i);
        data[index] = smem[threadIdx.x + i][threadIdx.y + j][threadIdx.z + k];
      }
    }
  }
}

template <typename T>
struct CudaSVT
{
  void reset(vvVolDesc const& vd, aabbi bbox, int channel = 0);

  template <typename Tex>
  void build(Tex transfunc);

  aabbi boundary(aabbi bbox) const;

  T& operator()(int x, int y, int z)
  {
    return data_[z * width * height + y * width + x];
  }

  T& at(int x, int y, int z)
  {
    return data_[z * width * height + y * width + x];
  }

  T const& at(int x, int y, int z) const
  {
    return data_[z * width * height + y * width + x];
  }

  T border_at(int x, int y, int z) const
  {
    if (x < 0 || y < 0 || z < 0)
      return 0;

    return data_[z * width * height + y * width + x];
  }

  T last() const
  {
    return data_.back();
  }

  T get_count(basic_aabb<int> bounds) const
  {
    bounds.min -= vec3i(1);
    bounds.max -= vec3i(1);

    return border_at(bounds.max.x, bounds.max.y, bounds.max.z)
         - border_at(bounds.max.x, bounds.max.y, bounds.min.z)
         - border_at(bounds.max.x, bounds.min.y, bounds.max.z)
         - border_at(bounds.min.x, bounds.max.y, bounds.max.z)
         + border_at(bounds.min.x, bounds.min.y, bounds.max.z)
         + border_at(bounds.min.x, bounds.max.y, bounds.min.z)
         + border_at(bounds.max.x, bounds.min.y, bounds.min.z)
         - border_at(bounds.min.x, bounds.min.y, bounds.min.z);
  }

  // Channel values from volume description
  thrust::device_vector<float> voxels_;
  // SVT array
  thrust::device_vector<T> data_;
  int width;
  int height;
  int depth;
};

template <typename T>
void CudaSVT<T>::reset(vvVolDesc const& vd, aabbi bbox, int channel)
{
  size_t size = bbox.size().x * bbox.size().y * bbox.size().z;
  data_.resize(size);
  width  = bbox.size().x;
  height = bbox.size().y;
  depth  = bbox.size().z;

  thrust::host_vector<float> host_voxels(size);

  for (int z = 0; z < depth; ++z)
  {
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        size_t index = z * width * height + y * width + x;
        host_voxels[index] = vd.getChannelValue(vd.getCurrentFrame(),
                bbox.min.x + x,
                bbox.min.y + y,
                bbox.min.z + z,
                channel);
      }
    }
  }

  voxels_ = host_voxels;
}

template <typename T>
template <typename Tex>
void CudaSVT<T>::build(Tex transfunc)
{
  // Apply transfer function
  {
    // Launch blocks of size 16x8x8 to e.g.
    // meet 1024 thread limit on Kepler
    dim3 block_size(16, 8, 8);
    dim3 grid_size(div_up(width,  (int)block_size.x),
                   div_up(height, (int)block_size.y),
                   div_up(depth,  (int)block_size.z));

    svt_apply_transfunc<<<grid_size, block_size>>>(
            transfunc,
            thrust::raw_pointer_cast(data_.data()),
            thrust::raw_pointer_cast(voxels_.data()),
            width,
            height,
            depth);
    cudaDeviceSynchronize();
  }

  // Build SVT
#if 0
  {
    dim3 block_size(8, 8);
    dim3 grid_size(div_up(width,  (int)block_size.x),
                   div_up(height, (int)block_size.y));

    svt_build_x<<<grid_size, block_size>>>(
            thrust::raw_pointer_cast(data_.data()),
            width,
            height,
            depth);
    cudaDeviceSynchronize();
  }

  {
    dim3 block_size(16, 8);
    dim3 grid_size(div_up(width,  (int)block_size.x),
                   div_up(height, (int)block_size.y));

    svt_build_y<<<grid_size, block_size>>>(
            thrust::raw_pointer_cast(data_.data()),
            width,
            height,
            depth);
    cudaDeviceSynchronize();
  }

  {
    dim3 block_size(16, 8);
    dim3 grid_size(div_up(width,  (int)block_size.x),
                   div_up(height, (int)block_size.y));

    svt_build_z<<<grid_size, block_size>>>(
            thrust::raw_pointer_cast(data_.data()),
            width,
            height,
            depth);
    cudaDeviceSynchronize();
  }
#else
  {
    dim3 block_size(16, 8, 8);
    dim3 grid_size(div_up(width/2,  (int)block_size.x),
                   div_up(height/2, (int)block_size.y),
                   div_up(depth/2,  (int)block_size.z));

    svt_build<<<grid_size, block_size>>>(
            thrust::raw_pointer_cast(data_.data()),
            width,
            height,
            depth);
    cudaDeviceSynchronize();
  }
#endif
}

namespace virvo
{

struct CudaKdTree::Impl
{
  CudaSVT<uint16_t> svt;

  visionaray::vec3i vox;
  visionaray::vec3 dist;
  float scale;
};

CudaKdTree::CudaKdTree()
  : impl_(new Impl)
{
}

CudaKdTree::~CudaKdTree()
{
}

void CudaKdTree::updateVolume(vvVolDesc const& vd, int channel)
{
  using visionaray::aabbi;
  using visionaray::vec3i;
  using visionaray::vec3;

  impl_->vox = vec3i(vd.vox.x, vd.vox.y, vd.vox.z);
  impl_->dist = vec3(vd.getDist().x, vd.getDist().y, vd.getDist().z);
  impl_->scale = vd._scale;

  impl_->svt.reset(vd, aabbi(vec3i(0), impl_->vox), channel);
}

void CudaKdTree::updateTransfunc(const visionaray::texture_ref<visionaray::vec4, 1>& transfunc)
{
  cuda_texture<visionaray::vec4, 1> cuda_transfunc(transfunc.data(),
      transfunc.width(),
      transfunc.get_address_mode(),
      transfunc.get_filter_mode());
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  for (int i=0; i<10; ++i) {
#ifdef BUILD_TIMING
  cudaEventRecord(start);
  //vvStopwatch sw; sw.start();
#endif
  impl_->svt.build(cuda_texture_ref<visionaray::vec4, 1>(cuda_transfunc));
#ifdef BUILD_TIMING
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  ms /= 1000.0f;
  std::cout << std::fixed << std::setprecision(3) << "svt update: " << ms << " sec.\n";
#endif
  }
  cudaEventDestroy(stop);
  cudaEventDestroy(start);
}

} // virvo

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
