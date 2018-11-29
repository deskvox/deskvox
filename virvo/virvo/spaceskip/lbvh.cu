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
#include <thrust/execution_policy.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#undef MATH_NAMESPACE
#include <visionaray/math/detail/math.h> // div_up
#include <visionaray/math/aabb.h>
#include <visionaray/morton.h>
#undef MATH_NAMESPACE

#include "../cuda/timer.h"
#include "../vvopengl.h"
#include "../vvspaceskip.h"
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
  // After compaction, all bricks are non-empty
  // anyway ==> reuse this field afterwards
  union
  {
    unsigned morton_code;
    int is_empty = true;
  };
};


//-------------------------------------------------------------------------------------------------
// Tree node that can be stored in device memory
//

struct Node
{
  __device__ Node()
    : bbox(vec3i(INT_MAX), vec3i(0))
  {
  }

  aabbi bbox;
  int left = -1;
  int right = -1;
  int parent = -1;
};


//-------------------------------------------------------------------------------------------------
// Helpers
//

template <typename T>
__device__
inline int signum(T a)
{
    return (T(0.0) < a) - (a < T(0.0));
}


//-------------------------------------------------------------------------------------------------
// Find node range that an inner node overlaps
//

__device__
vec2i determine_range(Brick* bricks, int num_bricks, int i)
{
  auto delta = [&](int i, int j)
  {
    // Karras' delta(i,j) function
    // Denotes the length of the longest common
    // prefix between keys k_i and k_j

    // Cf. Figure 4: "for simplicity, we define that
    // delta(i,j) = -1 when j not in [0,n-1]"
    if (j < 0 || j >= num_bricks)
      return -1;

    return __clz(bricks[i].morton_code ^ bricks[j].morton_code);
  };

  int num_inner = num_bricks - 1;

  if (i == 0)
    return { 0, num_inner };

  // Determine direction of the range (+1 or -1)
  int d = signum(delta(i, i + 1) - delta(i, i - 1));

  // Compute upper bound for the length of the range
  int delta_min = delta(i, i - d);
  int l_max = 2;
  while (delta(i, i + l_max * d) > delta_min)
  {
    l_max *= 2;
  }

  // Find the other end using binary search
  int l = 0;
  for (int t = l_max >> 1; t >= 1; t >>= 1)
  {
    if (delta(i, i + (l + t) * d) > delta_min)
      l += t;
  }

  if (d == 1)
    return vec2i(i, i + l * d);
  else
    return vec2i(i + l * d, i);
}


//-------------------------------------------------------------------------------------------------
// Find split positions based on Morton codes
//

__device__
int find_split(Brick* bricks, int first, int last)
{
  unsigned code_first = bricks[first].morton_code;
  unsigned code_last  = bricks[last].morton_code;

  if (code_first == code_last)
  {
    return (first + last) / 2;
  }

  unsigned common_prefix = __clz(code_first ^ code_last);

  int result = first;
  int step = last - first;

  do
  {
    step = (step + 1) / 2;
    int next = result + step;

    if (next < last)
    {
      unsigned code = bricks[next].morton_code;
      if (code_first == code || __clz(code_first ^ code) > common_prefix)
      {
        result = next;
      }
    }
  }
  while (step > 1);

  return result;
}


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
  if (shared_empty && !empty)
    atomicExch(&shared_empty, false);

  __syncthreads();

  if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
  {
    bricks[brick_index].min_corner[0] = blockIdx.x;
    bricks[brick_index].min_corner[1] = blockIdx.y;
    bricks[brick_index].min_corner[2] = blockIdx.z;

    if (!shared_empty)
      bricks[brick_index].is_empty = false;
  }
}

__global__ void assignMortonCodes(Brick* bricks, int num_bricks)
{
  unsigned index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < num_bricks)
  {
    Brick& b = bricks[index];
    b.morton_code = morton_encode3D(b.min_corner[0], b.min_corner[1], b.min_corner[2]);
  }
}

__global__ void nodeSplitting(Brick* bricks, int num_bricks, Node* leaves, Node* inner)
{
  int num_leaves = num_bricks;
  int num_inner = num_leaves - 1;

  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index < num_inner)
  {
    // NOTE: This is [first..last], not [first..last)!!
    vec2i range = determine_range(bricks, num_bricks, index);
    int first = range.x;
    int last = range.y;

    int split = find_split(bricks, first, last);
    //printf("%d: %d %d %d\n", index, first, split, last);

    int left = split;
    int right = split + 1;

    if (left == first)
    {
      // left child is leaf
      inner[index].left = ~left;
      leaves[left].parent = index;
    }
    else
    {
      // left child is inner
      inner[index].left = left;
      inner[left].parent = index;
    }

    if (right == last)
    {
      // right child is leaf
      inner[index].right = ~right;
      leaves[right].parent = index;
    }
    else
    {
      // right child is inner
      inner[index].right = right;
      inner[right].parent = index;
    }
  }
}

__global__ void buildHierarchy(Node* inner,
        int num_inner,
        Node* leaves,
        int num_leaves,
        Brick* bricks,
        virvo::SkipTreeNode* nodes,
        vec3i vox,
        vec3 dist,
        float scale)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;

  if (index >= num_leaves)
    return;

  // Leaf's bounding box
  aabbi bbox(vec3i(bricks[index].min_corner)*8, vec3i(bricks[index].min_corner)*8 + vec3i(8,8,8));
  leaves[index].bbox = bbox;

  // Atomically combine child bounding boxes and update parents
  int next = leaves[index].parent;

  while (next >= 0)
  {
    atomicMin(&inner[next].bbox.min.x, bbox.min.x);
    atomicMin(&inner[next].bbox.min.y, bbox.min.y);
    atomicMin(&inner[next].bbox.min.z, bbox.min.z);
    atomicMax(&inner[next].bbox.max.x, bbox.max.x);
    atomicMax(&inner[next].bbox.max.y, bbox.max.y);
    atomicMax(&inner[next].bbox.max.z, bbox.max.z);
    next = inner[next].parent;
  }

  __threadfence();

  // Convert aabbi to aabb. Each thread (but one) processes an inner node and a leaf
  // Also set indices while we're at it!

  if (index < num_inner)
  {
    auto bbox = inner[index].bbox;
    bbox.min.y = vox[1] - inner[index].bbox.max.y;
    bbox.max.y = vox[1] - inner[index].bbox.min.y;
    bbox.min.z = vox[2] - inner[index].bbox.max.z;
    bbox.max.z = vox[2] - inner[index].bbox.min.z;
    vec3 bmin = (vec3(bbox.min) - vec3(vox)/2.f) * dist * scale;
    vec3 bmax = (vec3(bbox.max) - vec3(vox)/2.f) * dist * scale;
    nodes[index].min_corner[0] = bmin.x;
    nodes[index].min_corner[1] = bmin.y;
    nodes[index].min_corner[2] = bmin.z;
    nodes[index].left = inner[index].left >= 0 ? inner[index].left : ~(num_inner + ~inner[index].left);
    nodes[index].max_corner[0] = bmax.x;
    nodes[index].max_corner[1] = bmax.y;
    nodes[index].max_corner[2] = bmax.z;
    nodes[index].right = inner[index].right >= 0 ? inner[index].right : ~(num_inner + ~inner[index].right);
  }

//if (index < num_leaves)
  {
    auto bbox = leaves[index].bbox;
    bbox.min.y = vox[1] - leaves[index].bbox.max.y;
    bbox.max.y = vox[1] - leaves[index].bbox.min.y;
    bbox.min.z = vox[2] - leaves[index].bbox.max.z;
    bbox.max.z = vox[2] - leaves[index].bbox.min.z;
    vec3 bmin = (vec3(bbox.min) - vec3(vox)/2.f) * dist * scale;
    vec3 bmax = (vec3(bbox.max) - vec3(vox)/2.f) * dist * scale;
    nodes[num_inner + index].min_corner[0] = bmin.x;
    nodes[num_inner + index].min_corner[1] = bmin.y;
    nodes[num_inner + index].min_corner[2] = bmin.z;
    nodes[num_inner + index].left = -1;
    nodes[num_inner + index].max_corner[0] = bmax.x;
    nodes[num_inner + index].max_corner[1] = bmax.y;
    nodes[num_inner + index].max_corner[2] = bmax.z;
    nodes[num_inner + index].right = -1;
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
  thrust::device_vector<virvo::SkipTreeNode> nodes;
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
  std::cout << "Find empty: " << t.elapsed() << '\n';
  t.reset();

  // Compact non-empty bricks to the left of the list
  thrust::device_vector<Brick> compact_bricks(grid_size.x * grid_size.y * grid_size.z);

  auto last = thrust::copy_if(
      thrust::device,
      bricks.begin(),
      bricks.end(),
      compact_bricks.begin(),
      [] __device__ (Brick b) { return !b.is_empty; });
  std::cout << "Compaction: " << t.elapsed() << '\n';
  t.reset();

  size_t numNonEmptyBricks = last - compact_bricks.begin();
  size_t numThreads = 1024;

  assignMortonCodes<<<div_up(numNonEmptyBricks, numThreads), numThreads>>>(
      thrust::raw_pointer_cast(compact_bricks.data()),
      numNonEmptyBricks);
  std::cout << "Assign Morton: " << t.elapsed() << '\n';
  t.reset();

  thrust::stable_sort(
      thrust::device,
      compact_bricks.begin(),
      last,
      [] __device__ (Brick l, Brick r)
      {
        return l.morton_code < r.morton_code;
      });
  std::cout << "Sorting: " << t.elapsed() << '\n';
  t.reset();

#if 0
  thrust::host_vector<Brick> h_karras_bricks(8);
  h_karras_bricks[0].morton_code = 1;
  h_karras_bricks[1].morton_code = 2;
  h_karras_bricks[2].morton_code = 4;
  h_karras_bricks[3].morton_code = 5;
  h_karras_bricks[4].morton_code = 19;
  h_karras_bricks[5].morton_code = 24;
  h_karras_bricks[6].morton_code = 25;
  h_karras_bricks[7].morton_code = 30;
  thrust::device_vector<Brick> karras_bricks(h_karras_bricks);
  thrust::device_vector<Node> leaves(8);
  thrust::device_vector<Node> inner(7);
  nodeSplitting<<<div_up(numNonEmptyBricks, numThreads), numThreads>>>(
        thrust::raw_pointer_cast(karras_bricks.data()),
        8,
        thrust::raw_pointer_cast(leaves.data()),
        thrust::raw_pointer_cast(inner.data()));
#else
  thrust::device_vector<Node> leaves(numNonEmptyBricks);
  thrust::device_vector<Node> inner(numNonEmptyBricks - 1);
  nodeSplitting<<<div_up(numNonEmptyBricks, numThreads), numThreads>>>(
      thrust::raw_pointer_cast(compact_bricks.data()),
      numNonEmptyBricks,
      thrust::raw_pointer_cast(leaves.data()),
      thrust::raw_pointer_cast(inner.data()));
  std::cout << "Splitting: " << t.elapsed() << '\n';
  t.reset();

#endif
#if 0
  thrust::host_vector<Node> h_inner(inner);
  int i = 0;
  for (auto n : h_inner)
  {
    auto l = n.left >= 0 ? n.left : ~n.left;
    auto r = n.right >= 0 ? n.right : ~n.right;
    auto strl = n.left >= 0 ? "INNER" : "LEAF";
    auto strr = n.right >= 0 ? "INNER" : "LEAF";
    std::cout << i++ << ": "
              << "Left: " << strl << '(' << l << "), "
              << "right: " << strr << '(' << r << "), "
              << "parent: " << n.parent << '\n';
  }
#endif

  virvo::SkipTreeNode init = { FLT_MAX, FLT_MAX, FLT_MAX, -1, -FLT_MAX, -FLT_MAX, -FLT_MAX, -1 };
  impl_->nodes.resize(inner.size() + leaves.size(), init);

  buildHierarchy<<<div_up(leaves.size(), numThreads), numThreads>>>(
      thrust::raw_pointer_cast(inner.data()),
      inner.size(),
      thrust::raw_pointer_cast(leaves.data()),
      leaves.size(),
      thrust::raw_pointer_cast(compact_bricks.data()),
      thrust::raw_pointer_cast(impl_->nodes.data()),
      impl_->vox,
      impl_->dist,
      impl_->scale);
  std::cout << "Build hierarchy: " << t.elapsed() << '\n';
}

virvo::SkipTreeNode* BVH::getNodes(int& numNodes)
{
  numNodes = impl_->nodes.size();
  return thrust::raw_pointer_cast(impl_->nodes.data());
}

std::vector<aabb> BVH::get_leaf_nodes(vec3 eye, bool frontToBack) const
{
  // TODO: it should also be possible to directly return
  // a device pointer to the leaf nodes

  // There are n-1 inner nodes followed by n leaves
  int num_inner = impl_->nodes.size() / 2;
  int num_leaves = impl_->nodes.size() - num_inner;

  std::vector<virvo::SkipTreeNode> leaves(num_leaves);
  thrust::copy(
      impl_->nodes.data() + num_inner,
      impl_->nodes.data() + num_inner + num_leaves,
      leaves.data());

  std::vector<aabb> result(num_leaves);

  for (size_t i = 0; i < leaves.size(); ++i)
  {
    result[i].min = vec3(leaves[i].min_corner);
    result[i].max = vec3(leaves[i].max_corner);
  }

  std::sort(
      result.begin(),
      result.end(),
      [eye, frontToBack](aabb const& l, aabb const& r)
      {
        auto distl = length(eye - l.center());
        auto distr = length(eye - r.center());

        if (frontToBack)
          return distl < distr;
        else
          return distr < distl;
      });

  return result;
}

void BVH::renderGL(vvColor color) const
{
  int numNodes = 0;
  auto nodes = const_cast<BVH*>(this)->getNodes(numNodes); // TODO..

  std::vector<virvo::SkipTreeNode> h_nodes(numNodes);
  cudaMemcpy(h_nodes.data(),
      nodes,
      numNodes * sizeof(virvo::SkipTreeNode),
      cudaMemcpyDeviceToHost);

  for (auto n : h_nodes)
  {
    vec3 bmin(n.min_corner);
    vec3 bmax(n.max_corner);

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
}

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
