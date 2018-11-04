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

#include <virvo/vvvoldesc.h>

#include "cudakdtree.h"

using namespace visionaray;

#define BUILD_TIMING 1

// Blocks of 16*8*8=1024 threads
#define BX 4
#define BY 8
#define BZ 8

//-------------------------------------------------------------------------------------------------
//
//

template <typename T>
__host__ __device__
void swap(T& a, T& b)
{
  T t(a);
  a = b;
  b = t;
}


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
#define NUM_BANKS 16
#define LOG_NUM_BANKS 4
#define CONFLICT_FREE_OFFSET(n) \
    ((n) >> NUM_BANKS + (n) >> (2 * LOG_NUM_BANKS))

  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (y >= height || z >= depth)
    return;

  __shared__ T smem[BX*2][BY][BZ];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int W = min(BX*2, width);

  if (threadIdx.x >= W/2)
    return;

  int base = z * width * height + y * width + blockIdx.x * BX * 2;
  int ai = tx;
  int bi = tx + W/2;
  smem[ai/* + CONFLICT_FREE_OFFSET(ai)*/][ty][tz] = data[base + ai];
  smem[bi/* + CONFLICT_FREE_OFFSET(bi)*/][ty][tz] = data[base + bi];

  #pragma unroll
  for (int i = 0; i < 3; ++i)
  {
    // Reduction
    int stride;
    for (stride = 1; stride <= BX; stride <<= 1) {
      int index = (tx + 1) * stride * 2 - 1;
      if (index < 2 * BX)
        smem[index][ty][tz] += smem[index - stride][ty][tz];
      __syncthreads();
    }

    // Post reduction
    for (stride = BX >> 1; stride; stride >>= 1) {
      int index = (tx + 1) * stride * 2 - 1;
      if (index + stride < 2 * BX)
        smem[index + stride][ty][tz] += smem[index][ty][tz];
      __syncthreads();
    }

    if (i == 0)
    {
      if (ai > ty)
        swap(smem[ai][ty][tz], smem[ty][ai][tz]);
      if (bi >= ty)
        swap(smem[bi][ty][tz], smem[ty][bi][tz]);
    }
    else if (i == 1)
    {
      if (tz > ai)
        swap(smem[tz][ty][ai], smem[ai][ty][tz]);
      if (tz >= bi)
        swap(smem[tz][ty][bi], smem[bi][ty][tz]);
    }
    else if (i == 2)
    {
      if (ty > ai)
        swap(smem[ty][ai][tz], smem[ai][ty][tz]);
      if (ty > bi)
        swap(smem[ty][bi][tz], smem[bi][ty][tz]);
    }
  }

  // Copy back to global memory

  data[base + ai] = smem[ai/* + CONFLICT_FREE_OFFSET(ai)*/][ty][tz];
  data[base + bi] = smem[bi/* + CONFLICT_FREE_OFFSET(bi)*/][ty][tz];
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
    dim3 block_size(BX, BY, BZ);
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
  // Test
#if 1
  {
    int W=8,H=8,D=8;
    std::vector<uint16_t> data(W*H*D);
    std::fill(data.begin(), data.end(), 0);
    int x=1,y=0,z=0;
    data[z*W*H + y*W + x] = 1;

    thrust::device_vector<uint16_t> d_data(data);

    dim3 block_size(BX, BY, BZ);
    dim3 grid_size(div_up(W/2, (int)block_size.x),
                   div_up(D, (int)block_size.y),
                   div_up(H, (int)block_size.z));
    svt_build<<<grid_size, block_size>>>(
            thrust::raw_pointer_cast(d_data.data()),
            W,
            D,
            H);

    thrust::host_vector<uint16_t> h_data(d_data);

    int idx = 0;
    for (int i = 0; i < D; ++i)
    {
      for (int j = 0; j < H; ++j)
      {
        for (int k = 0; k < W; ++k)
        {
          std::cout << h_data[idx++] << ' ';
        }
        std::cout << '\n';
      }
      std::cout << '\n';
      std::cout << '\n';
    }
    exit(0);
  }
#endif

  {
    dim3 block_size(BX, BY, BZ);
    dim3 grid_size(div_up(width/2,  (int)block_size.x),
                   div_up(height, (int)block_size.y),
                   div_up(depth,  (int)block_size.z));

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

//-------------------------------------------------------------------------------------------------
// Private implementation
//

struct CudaKdTree::Impl
{
  void node_splitting();

  CudaSVT<uint16_t> svt;

  visionaray::vec3i vox;
  visionaray::vec3 dist;
  float scale;

  // Array with one bbox per partial SVT
  thrust::device_vector<aabbi> bounds_;

};

void CudaKdTree::Impl::node_splitting()
{
  // We maintain one bbox per partial SVT in global
  // memory that is updated during splitting
  {
    dim3 block_size(BX, BY, BZ);
    dim3 grid_size(div_up(vox[0]/2, (int)block_size.x),
                   div_up(vox[1]/2, (int)block_size.y),
                   div_up(vox[2]/2, (int)block_size.z));

    bounds_.resize(grid_size.x * grid_size.y * grid_size.z);
  }
}


//-------------------------------------------------------------------------------------------------
// (Semi-)public interface
//

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
#ifdef BUILD_TIMING
  cudaEventRecord(start);
#endif
  impl_->svt.build(cuda_texture_ref<visionaray::vec4, 1>(cuda_transfunc));
#ifdef BUILD_TIMING
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  float sec = ms / 1000.0f;
  std::cout << std::fixed << std::setprecision(3) << "svt update: " << sec << " sec.\n";
  cudaEventRecord(start);
  cudaEventRecord(stop);
#endif
  impl_->node_splitting();
#ifdef BUILD_TIMING
  cudaEventSynchronize(stop);
  ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  sec = ms / 1000.0f;
  std::cout << "splitting: " << sec << " sec.\n";
  cudaEventDestroy(stop);
  cudaEventDestroy(start);
#endif
}

} // virvo

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
