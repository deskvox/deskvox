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

#include <thrust/system/cuda/vector.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/generate.h>
#include <thrust/pair.h>
#include <iostream>
#include <map>

#include <cassert>
#include <iostream>
#include <ostream>

#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/reduce.h>

#undef MATH_NAMESPACE

#include <visionaray/math/detail/math.h>
#include <visionaray/math/aabb.h>
#include <visionaray/math/axis.h>
#include <visionaray/math/forward.h>
#include <visionaray/math/io.h>
#include <visionaray/math/vector.h>
#include <visionaray/morton.h>

#undef MATH_NAMESPACE

#include <virvo/cuda/timer.h>
#include <virvo/vvopengl.h>
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

class cached_allocator
{
public:
    // just allocate bytes
    typedef char value_type;
 
    cached_allocator() = default;
 
    ~cached_allocator()
    {
        // free all allocations when cached_allocator goes out of scope
        free_all();
    }
 
    char* allocate(std::ptrdiff_t num_bytes)
    {
        char* result = 0;
 
        // search the cache for a free block
        free_blocks_type::iterator free_block = free_blocks.find(num_bytes);
 
        if (free_block != free_blocks.end())
        {
            //std::cout << "cached_allocator::allocator(): found a hit" << std::endl;
 
            // get the pointer
            result = free_block->second;
 
            // erase from the free_blocks map
            free_blocks.erase(free_block);
        }
        else
        {
            // no allocation of the right size exists
            // create a new one with cuda::malloc
            // throw if cuda::malloc can't satisfy the request
            try
            {
                std::cout << "cached_allocator::allocator(): no free block found; calling cuda::malloc" << std::endl;
 
                // allocate memory and convert cuda::pointer to raw pointer
                result = thrust::cuda::malloc<char>(num_bytes).get();
            }
            catch(std::runtime_error &e)
            {
                throw;
            }
        }
 
        // insert the allocated pointer into the allocated_blocks map
        allocated_blocks.insert(std::make_pair(result, num_bytes));
 
        return result;
    }
 
    void deallocate(char* ptr, size_t n)
    {
        // erase the allocated block from the allocated blocks map
        allocated_blocks_type::iterator iter = allocated_blocks.find(ptr);
        std::ptrdiff_t num_bytes = iter->second;
        allocated_blocks.erase(iter);
 
        // insert the block into the free blocks map
        free_blocks.insert(std::make_pair(num_bytes, ptr));
    }
 
private:
    typedef std::multimap<std::ptrdiff_t, char*> free_blocks_type;
    typedef std::map<char*, std::ptrdiff_t> allocated_blocks_type;
 
    free_blocks_type free_blocks;
    allocated_blocks_type allocated_blocks;
 
    void free_all()
    {
        std::cout << "cached_allocator::free_all(): cleaning up after ourselves..." << std::endl;
 
        // deallocate all outstanding blocks in both lists
        for (free_blocks_type::iterator i = free_blocks.begin();
                i != free_blocks.end(); i++)
        {
            // transform the pointer to cuda::pointer before calling cuda::free
            thrust::cuda::free(thrust::cuda::pointer<char>(i->second));
        }
 
        for (allocated_blocks_type::iterator i = allocated_blocks.begin();
                i != allocated_blocks.end(); i++)
        {
            // transform the pointer to cuda::pointer before calling cuda::free
            thrust::cuda::free(thrust::cuda::pointer<char>(i->first));
        }
    }
};

template <typename T>
__host__ __device__
void swap(T& a, T& b)
{
  T t(a);
  a = b;
  b = t;
}


//-------------------------------------------------------------------------------------------------
// CUDA kernels
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
__global__ void svt_build_boxes(T* data, aabbi* boxes, int width, int height, int depth)
{
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (y >= height || z >= depth)
    return;

  __shared__ T smem[BX*2][BY][BZ];

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  int W = min(BX*2, width);
  int H = BY;
  int D = BZ;

  if (threadIdx.x >= W/2)
    return;

  int base = z * width * height + y * width + blockIdx.x * BX * 2;
  int ai = tx;
  int bi = tx + W/2;
  smem[ai][ty][tz] = data[base + ai];
  smem[bi][ty][tz] = data[base + bi];

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
      if (tz > ai)
        swap(smem[tz][ty][ai], smem[ai][ty][tz]);
      if (tz >= bi)
        swap(smem[tz][ty][bi], smem[bi][ty][tz]);

      __syncthreads();

      if (ai > ty)
        swap(smem[ai][ty][tz], smem[ty][ai][tz]);
      if (bi >= ty)
        swap(smem[bi][ty][tz], smem[ty][bi][tz]);

    }

    __syncthreads();
  }

  // Calculate local bounding boxes

  auto border_at = [&](int x, int y, int z)
  {
    if (x < 0 || y < 0 || z < 0)
      return T(0);
    else
      return smem[x][y][z];
  };

  auto get_count = [&](aabbi bounds)
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
  };

  aabbi bounds(vec3i(0,0,0), vec3i(8,8,8));

  uint16_t voxels = get_count(bounds);

  int box_index = bz * gridDim.x * gridDim.y + by * gridDim.x + bx;

  if (voxels == 0)
  {
    // Make an invalid bounding box that we however can
    // identify as belonging to this brick!
    boxes[box_index].min = vec3i(4,4,4) + vec3i(bx * W, by * H, bz * D);
    boxes[box_index].max = vec3i(3,3,3) + vec3i(bx * W, by * H, bz * D);
  }
  else if (tx == 0 && ty == 0 && tz == 0)
  {
    // Search for the minimal volume bounding box
    // that contains #voxels contained in bbox!

    vec3i min_corner = bounds.min;
    vec3i max_corner = bounds.max;

    // X boundary
    int x = (bounds.max.x - bounds.min.x) / 2;

    while (x >= 1)
    {
      aabbi lbox(min_corner, bounds.max);
      lbox.min.x += x;

      if (get_count(lbox) == voxels)
      {
        min_corner = lbox.min;
      }

      aabbi rbox(bounds.min, max_corner);
      rbox.max.x -= x;

      if (get_count(rbox) == voxels)
      {
        max_corner = rbox.max;
      }

      x /= 2;
    }

    // Y boundary from left
    int y = (bounds.max.y - bounds.min.y) / 2;

    while (y >= 1)
    {
      aabbi lbox(min_corner, bounds.max);
      lbox.min.y += y;

      if (get_count(lbox) == voxels)
      {
        min_corner = lbox.min;
      }

      aabbi rbox(bounds.min, max_corner);
      rbox.max.y -= y;

      if (get_count(rbox) == voxels)
      {
        max_corner = rbox.max;
      }

      y /= 2;
    }

    // Z boundary from left
    int z = (bounds.max.z - bounds.min.z) / 2;

    while (z >= 1)
    {
      aabbi lbox(min_corner, bounds.max);
      lbox.min.z += z;

      if (get_count(lbox) == voxels)
      {
        min_corner = lbox.min;
      }

      aabbi rbox(bounds.min, max_corner);
      rbox.max.z -= z;

      if (get_count(rbox) == voxels)
      {
        max_corner = rbox.max;
      }

      z /= 2;
    }

    bounds = aabbi(min_corner, max_corner);

    bounds.min += vec3i(bx * W, by * H, bz * D);
    bounds.max += vec3i(bx * W, by * H, bz * D);
    boxes[box_index] = bounds;
  }
}


//-------------------------------------------------------------------------------------------------
// CUDA Summed-volume table
//

template <typename T>
struct CudaSVT
{
  void reset(vvVolDesc const& vd, aabbi bbox, int channel = 0);

  template <typename Tex>
  void build(Tex transfunc);

  aabbi boundary(aabbi bbox, const cudaStream_t& stream = 0) const;

  // Channel values from volume description
  thrust::device_vector<float> voxels_;
  // SVT array
  thrust::device_vector<T> data_;
  // Local bounding boxes
  thrust::device_vector<aabbi> boxes_;

  int width;
  int height;
  int depth;
};

template <typename T>
void CudaSVT<T>::reset(vvVolDesc const& vd, aabbi bbox, int channel)
{
  std::cout << cudaGetLastError() << '\n';
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

struct device_compare_morton
{
  __device__
  bool operator()(aabbi l, aabbi r)
  {
    auto cl = l.min / vec3i(8,8,8);
    auto cr = r.min / vec3i(8,8,8);

    auto ml = morton_encode3D(cl.x, cl.y, cl.z);
    auto mr = morton_encode3D(cr.x, cr.y, cr.z);

    return ml < mr;
  }
};

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
  }

  // Build 8x8x8 SVTs
  {
    dim3 block_size(BX, BY, BZ);
    dim3 grid_size(div_up(width/2, (int)block_size.x),
                   div_up(height,  (int)block_size.y),
                   div_up(depth,   (int)block_size.z));

    boxes_.resize(grid_size.x * grid_size.y * grid_size.z);

    svt_build_boxes<<<grid_size, block_size>>>(
            thrust::raw_pointer_cast(data_.data()),
            thrust::raw_pointer_cast(boxes_.data()),
            width,
            height,
            depth);
  }

  thrust::stable_sort(
        thrust::device,
        boxes_.begin(),
        boxes_.end(),
        device_compare_morton()
        );
//
//    thrust::host_vector<aabbi> h_boxes(boxes_);
//
//    for (auto b : h_boxes)
//        if (b.valid()) std::cout << b.min << b.max << std::endl;
}

struct device_combine
{
  __device__
  aabbi operator()(aabbi l, aabbi r)
  {
    l = intersect(l, check);
    r = intersect(r, check);

    if (l.empty())
      l.invalidate();

    if (r.empty())
      r.invalidate();

    return combine(l, r);
  }

  aabbi check;
};

template <typename T>
aabbi CudaSVT<T>::boundary(aabbi bbox, const cudaStream_t& stream) const
{
  bbox.min.x = std::max(0, round_down(bbox.min.x, 8));
  bbox.min.y = std::max(0, round_down(bbox.min.y, 8));
  bbox.min.z = std::max(0, round_down(bbox.min.z, 8));

  bbox.max.x = std::min(width,  round_up(bbox.max.x, 8));
  bbox.max.y = std::min(height, round_up(bbox.max.y, 8));
  bbox.max.z = std::min(depth,  round_up(bbox.max.z, 8));

  // Calculate minimum and maximum morton indices of box
  // So we can restrict the range of the ensuing reduction
  vec3i min(bbox.min.x / 8, bbox.min.y / 8, bbox.min.z / 8);
  vec3i max(bbox.max.x / 8, bbox.max.y / 8, bbox.max.z / 8);
//std::cout << min << max << '\n';
  auto min_index = morton_encode3D(min.x, min.y, min.z);
  auto max_index = std::min((uint64_t)morton_encode3D(max.x, max.y, max.z), (uint64_t)boxes_.size());
//std::cout << min_index << ' ' << max_index << '\n';

  thrust::host_vector<aabbi> check(1);
  thrust::copy(&bbox, &bbox + 1, check.begin());
  device_combine comb;
  comb.check = bbox;//thrust::raw_pointer_cast(check.data());

  aabbi init;
  init.invalidate();

  static cached_allocator alloc;
  return thrust::reduce(
      thrust::cuda::par(alloc),
      //thrust::device,
      boxes_.begin() + min_index,
      boxes_.begin() + max_index + 1,
      init,
      comb);
}

namespace virvo
{

//-------------------------------------------------------------------------------------------------
// Private implementation
//

struct CudaKdTree::Impl
{
  struct Node;
  typedef std::unique_ptr<Node> NodePtr;

  struct Node
  {
    visionaray::aabbi bbox;
    NodePtr left  = nullptr;
    NodePtr right = nullptr;
    int axis = -1;
    int splitpos = -1;
    int depth;
  };

  template <typename Func>
  void traverse(NodePtr const& n, visionaray::vec3 eye, Func f, bool frontToBack = true) const
  {
    if (n != nullptr)
    {
      f(n);

      if (n->axis >= 0)
      {
        int spi = n->splitpos;
        if (n->axis == 1 || n->axis == 2)
          spi = vox[n->axis] - spi - 1;
        float splitpos = (spi - vox[n->axis]/2.f) * dist[n->axis] * scale;

        // TODO: puh..
        if (n->axis == 0 && eye[n->axis] < splitpos || n->axis == 1 && eye[n->axis] >= splitpos || n->axis == 2 && eye[n->axis] >= splitpos)
        {
          traverse(frontToBack ? n->left : n->right, eye, f, frontToBack);
          traverse(frontToBack ? n->right : n->left, eye, f, frontToBack);
        }
        else
        {
          traverse(frontToBack ? n->right : n->left, eye, f, frontToBack);
          traverse(frontToBack ? n->left : n->right, eye, f, frontToBack);
        }
      }
    }
  }

  void node_splitting(NodePtr& n);
  void renderGL(NodePtr const& n, vvColor color) const;

  CudaSVT<uint16_t> svt;

  visionaray::vec3i vox;
  visionaray::vec3 dist;
  float scale;

  NodePtr root = nullptr;
};

void CudaKdTree::Impl::node_splitting(NodePtr& n)
{
  using visionaray::aabbi;
  using visionaray::vec3i;

  // Expand node's bounding box so it falls on multiples of eights
  n->bbox.min.x = max(0, round_down(n->bbox.min.x, 8));
  n->bbox.min.y = max(0, round_down(n->bbox.min.y, 8));
  n->bbox.min.z = max(0, round_down(n->bbox.min.z, 8));

  n->bbox.max.x = min(vox[0], round_up(n->bbox.max.x, 8));
  n->bbox.max.y = min(vox[1], round_up(n->bbox.max.y, 8));
  n->bbox.max.z = min(vox[2], round_up(n->bbox.max.z, 8));

  // Halting criterion 1.)
  if (volume(n->bbox) < volume(root->bbox) / 10)
    return;

  // Split along longest axis
  vec3i len = n->bbox.max - n->bbox.min;

  int axis = 0;
  if (len.y > len.x && len.y > len.z)
    axis = 1;
  else if (len.z > len.x && len.z > len.y)
    axis = 2;

  static const int NumBins = 16;

  int dl = len[axis] / NumBins;

  int num_planes = NumBins- 1;

  int min_cost = INT_MAX;
  int best_p = -1;

  aabbi lbox = n->bbox;
  aabbi rbox = n->bbox;

  int first = lbox.min[axis];

  int vol = volume(n->bbox);

  int off = n->bbox.min[axis];

  for (int p = 1; p <= num_planes; ++p)
  {
    aabbi ltmp = n->bbox;
    aabbi rtmp = n->bbox;

    ltmp.max[axis] = first + dl * p;
    rtmp.min[axis] = first + dl * p;

    ltmp = svt.boundary(ltmp);
    rtmp = svt.boundary(rtmp);

    int c = volume(ltmp) + volume(rtmp);

    // empty-space volume
    int ev = vol - c;

    // Halting criterion 2.)
    if (ev <= vol / 20)
      continue;

    if (c < min_cost)
    {
      min_cost = c;
      lbox = ltmp;
      rbox = rtmp;
      best_p = p;std::cout << best_p << '\n';
    }

    off += dl;
    if (dl >= n->bbox.max[axis])
      break;
  }

  // Halting criterion 2.)
  if (best_p < 0)
    return;

  // Store split plane for traversal
  n->axis = axis;
  n->splitpos = first + dl * best_p;

  n->left.reset(new Node);
  n->left->bbox = lbox;
  n->left->depth = n->depth + 1;
  node_splitting(n->left);

  n->right.reset(new Node);
  n->right->bbox = rbox;
  n->right->depth = n->depth + 1;
  node_splitting(n->right);
}

void CudaKdTree::Impl::renderGL(CudaKdTree::Impl::NodePtr const& n, vvColor color) const
{
  using visionaray::vec3;

  if (n != nullptr)
  {
    if (n->left == nullptr && n->right == nullptr)
    {
      auto bbox = n->bbox;
      bbox.min.y = vox[1] - n->bbox.max.y;
      bbox.max.y = vox[1] - n->bbox.min.y;
      bbox.min.z = vox[2] - n->bbox.max.z;
      bbox.max.z = vox[2] - n->bbox.min.z;
      vec3 bmin = (vec3(bbox.min) - vec3(vox)/2.f) * dist * scale;
      vec3 bmax = (vec3(bbox.max) - vec3(vox)/2.f) * dist * scale;

      glBegin(GL_LINES);
      glColor3f(color[0], color[1], color[2]);

      glVertex3f(bmin.x, bmin.y, bmin.z);
      glVertex3f(bmax.x, bmin.y, bmin.z);

      glVertex3f(bmax.x, bmin.y, bmin.z);
      glVertex3f(bmax.x, bmax.y, bmin.z);

      glVertex3f(bmax.x, bmax.y, bmin.z);
      glVertex3f(bmin.x, bmax.y, bmin.z);

      glVertex3f(bmin.x, bmax.y, bmin.z);
      glVertex3f(bmin.x, bmin.y, bmin.z);

      //
      glVertex3f(bmin.x, bmin.y, bmax.z);
      glVertex3f(bmax.x, bmin.y, bmax.z);

      glVertex3f(bmax.x, bmin.y, bmax.z);
      glVertex3f(bmax.x, bmax.y, bmax.z);

      glVertex3f(bmax.x, bmax.y, bmax.z);
      glVertex3f(bmin.x, bmax.y, bmax.z);

      glVertex3f(bmin.x, bmax.y, bmax.z);
      glVertex3f(bmin.x, bmin.y, bmax.z);

      //
      glVertex3f(bmin.x, bmin.y, bmin.z);
      glVertex3f(bmin.x, bmin.y, bmax.z);

      glVertex3f(bmax.x, bmin.y, bmin.z);
      glVertex3f(bmax.x, bmin.y, bmax.z);

      glVertex3f(bmax.x, bmax.y, bmin.z);
      glVertex3f(bmax.x, bmax.y, bmax.z);

      glVertex3f(bmin.x, bmax.y, bmin.z);
      glVertex3f(bmin.x, bmax.y, bmax.z);
      glEnd();
    }

    renderGL(n->left, color);
    renderGL(n->right, color);
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

#ifdef BUILD_TIMING
  CudaTimer timer;
#endif
  impl_->svt.build(cuda_texture_ref<visionaray::vec4, 1>(cuda_transfunc));
#ifdef BUILD_TIMING
  std::cout << std::fixed << std::setprecision(8) << "svt update: " << timer.elapsed() << " sec.\n";
#endif

#ifdef BUILD_TIMING
  timer.reset();
#endif
  impl_->root.reset(new Impl::Node);
  impl_->root->bbox = impl_->svt.boundary(visionaray::aabbi(
        visionaray::vec3i(0),
        visionaray::vec3i(impl_->vox[0], impl_->vox[1], impl_->vox[2])));
  impl_->root->depth = 0;
  impl_->node_splitting(impl_->root);
#ifdef BUILD_TIMING
  std::cout << "splitting: " << timer.elapsed() << " sec.\n";
#endif
}

std::vector<visionaray::aabb> CudaKdTree::get_leaf_nodes(visionaray::vec3 eye, bool frontToBack) const
{
  using visionaray::aabb;
  using visionaray::vec3;

  std::vector<aabb> result;
size_t vol = 0;
  impl_->traverse(impl_->root, eye, [&vol,&result,this](Impl::NodePtr const& n)
  {
    if (n->left == nullptr && n->right == nullptr)
    {
      auto bbox = n->bbox;vol += volume(bbox);
      bbox.min.y = impl_->vox[1] - n->bbox.max.y;
      bbox.max.y = impl_->vox[1] - n->bbox.min.y;
      bbox.min.z = impl_->vox[2] - n->bbox.max.z;
      bbox.max.z = impl_->vox[2] - n->bbox.min.z;
      vec3 bmin = (vec3(bbox.min) - vec3(impl_->vox)/2.f) * impl_->dist * impl_->scale;
      vec3 bmax = (vec3(bbox.max) - vec3(impl_->vox)/2.f) * impl_->dist * impl_->scale;

      result.push_back(aabb(bmin, bmax));
    }
  }, frontToBack);vol = (impl_->vox[0] * impl_->vox[1] * impl_->vox[2]) - vol;std::cout << vol << std::endl;

  return result;
}

void CudaKdTree::renderGL(vvColor color) const
{
  impl_->renderGL(impl_->root, color);
}

} // virvo

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
