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

#include <cassert>

#undef MATH_NAMESPACE

#include <visionaray/math/aabb.h>
#include <visionaray/math/forward.h>
#include <visionaray/math/matrix.h>
#include <visionaray/math/rectangle.h>
#include <visionaray/math/vector.h>
#include <visionaray/texture/texture.h>

#undef MATH_NAMESPACE

#include "vvclock.h"
#include "vvopengl.h"
#include "vvspaceskip.h"
#include "vvvoldesc.h"

using namespace visionaray;

#define FRAME_TIMING 0
#define BUILD_TIMING 1
#define KDTREE       1

//-------------------------------------------------------------------------------------------------
// Summed-volume table
//

template <typename T>
struct SVT
{
  void reset(aabbi bbox);
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

  T* data()
  {
    return data_.data();
  }

  T const* data() const
  {
    return data_.data();
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
  std::vector<float> voxels_;
  // SVT array
  std::vector<T> data_;
  int width;
  int height;
  int depth;
};

template <typename T>
void SVT<T>::reset(aabbi bbox)
{
  data_.resize(bbox.size().x * bbox.size().y * bbox.size().z);
  width  = bbox.size().x;
  height = bbox.size().y;
  depth  = bbox.size().z;
}

template <typename T>
void SVT<T>::reset(vvVolDesc const& vd, aabbi bbox, int channel)
{
  voxels_.resize(bbox.size().x * bbox.size().y * bbox.size().z);
  data_.resize(bbox.size().x * bbox.size().y * bbox.size().z);
  width  = bbox.size().x;
  height = bbox.size().y;
  depth  = bbox.size().z;


  for (int z = 0; z < depth; ++z)
  {
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        size_t index = z * width * height + y * width + x;
        voxels_[index] = vd.getChannelValue(0,
                bbox.min.x + x,
                bbox.min.y + y,
                bbox.min.z + z,
                channel);
      }
    }
  }
}

template <typename T>
template <typename Tex>
void SVT<T>::build(Tex transfunc)
{
  // Apply transfer function
  for (int z = 0; z < depth; ++z)
  {
    for (int y = 0; y < height; ++y)
    {
      for (int x = 0; x < width; ++x)
      {
        size_t index = z * width * height + y * width + x;
        if (tex1D(transfunc, voxels_[index]).w < 0.0001)
          at(x, y, z) = T(0);
        else
          at(x, y, z) = T(1);
      }
    }
  }


  // Build summed volume table

  // Init 0-border voxel
  //at(0, 0, 0) = at(0, 0, 0);

  // Init 0-border edges (prefix sum)
  for (int x=1; x<width; ++x)
  {
    at(x, 0, 0) = at(x, 0, 0) + at(x-1, 0, 0);
  }

  for (int y=1; y<height; ++y)
  {
    at(0, y, 0) = at(0, y, 0) + at(0, y-1, 0);
  }

  for (int z=1; z<depth; ++z)
  {
    at(0, 0, z) = at(0, 0, z) + at(0, 0, z-1);
  }


  // Init 0-border planes (summed-area tables)
  for (int y=1; y<height; ++y)
  {
    for (int x=1; x<width; ++x)
    {
      at(x, y, 0) = at(x, y, 0)
        + at(x-1, y, 0) + at(x, y-1, 0)
        - at(x-1, y-1, 0);
    }
  }

  for (int z=1; z<depth; ++z)
  {
    for (int y=1; y<height; ++y)
    {
      at(0, y, z) = at(0, y, z)
        + at(0, y-1, z) + at(0, y, z-1)
        - at(0, y-1, z-1);
    }
  }

  for (int x=1; x<width; ++x)
  {
    for (int z=1; z<depth; ++z)
    {
      at(x, 0, z) = at(x, 0, z)
        + at(x-1, 0, z) + at(x, 0, z-1)
        - at(x-1, 0, z-1);
    }
  }


  // Build up SVT
  for (int z=1; z<depth; ++z)
  {
    for (int y=1; y<height; ++y)
    {
      for (int x=1; x<width; ++x)
      {
        at(x, y, z) = at(x, y, z) + at(x-1, y-1, z-1)
          + at(x-1, y, z) - at(x, y-1, z-1)
          + at(x, y-1, z) - at(x-1, y, z-1)
          + at(x, y, z-1) - at(x-1, y-1, z);
      }
    }
  }
}

// produce a boundary around the *non-empty* voxels in bbox
template <typename T>
aabbi SVT<T>::boundary(aabbi bbox) const
{
  aabbi bounds = bbox;

  // Search for the minimal volume bounding box
  // that contains #voxels contained in bbox!
  uint16_t voxels = get_count(bounds);


  // X boundary
  int x = (bounds.min.x + bounds.max.x) / 2;

  while (x >= 1)
  {
    aabbi lbox = bounds;
    lbox.min.x += x;

    if (get_count(lbox) == voxels)
    {
      bounds = lbox;
    }

    aabbi rbox = bounds;
    rbox.max.x -= x;

    if (get_count(rbox) == voxels)
    {
      bounds = rbox;
    }

    x /= 2;
  }

  // Y boundary from left
  int y = (bounds.min.y + bounds.max.y) / 2;

  while (y >= 1)
  {
    aabbi lbox = bounds;
    lbox.min.y += y;

    if (get_count(lbox) == voxels)
    {
      bounds = lbox;
    }

    aabbi rbox = bounds;
    rbox.max.y -= y;

    if (get_count(rbox) == voxels)
    {
      bounds = rbox;
    }

    y /= 2;
  }

  // Z boundary from left
  int z = (bounds.min.z + bounds.max.z) / 2;

  while (z >= 1)
  {
    aabbi lbox = bounds;
    lbox.min.z += z;

    if (get_count(lbox) == voxels)
    {
      bounds = lbox;
    }

    aabbi rbox = bounds;
    rbox.max.z -= z;

    if (get_count(rbox) == voxels)
    {
      bounds = rbox;
    }

    z /= 2;
  }

  return bounds;
}

//-------------------------------------------------------------------------------------------------
// Partial SVT
//

struct PartialSVT
{
  typedef SVT<uint16_t> svt_t;

  void reset(vvVolDesc const& vd, aabbi bbox, int channel = 0);

  template <typename Tex>
    void build(Tex transfunc);

  aabbi boundary(aabbi bbox) const;

  uint64_t get_count(aabbi bounds) const;

  vec3i bricksize = vec3i(32, 32, 32);

  vec3i num_svts;
  std::vector<svt_t> svts;
};

void PartialSVT::reset(vvVolDesc const& vd, aabbi bbox, int channel)
{
  num_svts = vec3i(div_up(bbox.max.x, bricksize.x),
                   div_up(bbox.max.y, bricksize.y),
                   div_up(bbox.max.z, bricksize.z));

  svts.resize(num_svts.x * num_svts.y * num_svts.z);


  // Fill with volume channel values

  int bz = 0;
  for (int z = 0; z < num_svts.z; ++z)
  {
    int by = 0;
    for (int y = 0; y < num_svts.y; ++y)
    {
      int bx = 0;
      for (int x = 0; x < num_svts.x; ++x)
      {
        vec3i bmin(bx, by, bz);
        vec3i bmax(min(bbox.max.x, bx + bricksize.x),
                   min(bbox.max.y, by + bricksize.y),
                   min(bbox.max.z, bz + bricksize.z));
        svts[z * num_svts.x * num_svts.y + y * num_svts.x + x].reset(vd, aabbi(bmin, bmax), channel);

        bx += bricksize.x;
      }

      by += bricksize.y;
    }

    bz += bricksize.z;
  }
}

template <typename Tex>
void PartialSVT::build(Tex transfunc)
{
    #pragma omp parallel for
    for (size_t i = 0; i < svts.size(); ++i)
    {
        svts[i].build(transfunc);
    }
}

aabbi PartialSVT::boundary(aabbi bbox) const
{
  aabbi bounds = bbox;

  vec3i min_brick = bounds.min / bricksize;
  vec3i min_bpos = bounds.min - min_brick * bricksize;

  vec3i max_brick = bounds.max / bricksize;
  vec3i max_bpos = bounds.max - max_brick * bricksize;

  vec3i num_bricks = max_brick - min_brick + vec3i(1);
  std::vector<aabbi> brick_boundaries(num_bricks.x * num_bricks.y * num_bricks.z);

  #pragma omp parallel for collapse(3)
  for (int bz = min_brick.z; bz <= max_brick.z; ++bz)
  {
    for (int by = min_brick.y; by <= max_brick.y; ++by)
    {
      for (int bx = min_brick.x; bx <= max_brick.x; ++bx)
      {
        int minz = bz == min_brick.z ? min_bpos.z : 0;
        int maxz = bz == max_brick.z ? max_bpos.z : bricksize.z;

        int miny = by == min_brick.y ? min_bpos.y : 0;
        int maxy = by == max_brick.y ? max_bpos.y : bricksize.y;

        int minx = bx == min_brick.x ? min_bpos.x : 0;
        int maxx = bx == max_brick.x ? max_bpos.x : bricksize.x;

        int i = (bz - min_brick.z) * num_bricks.x * num_bricks.y + (by - min_brick.y) * num_bricks.x + (bx - min_brick.x);

        auto& svt = svts[bz * num_svts.x * num_svts.y + by * num_svts.x + bx];
        aabbi test(vec3i(minx, miny, minz), vec3i(maxx, maxy, maxz));

        if (svt.get_count(test) != 0)
          brick_boundaries[i] = svt.boundary(test);
        else
          brick_boundaries[i] = aabbi(vec3i(0), vec3i(0));

        brick_boundaries[i].min += vec3i(bx * bricksize.x, by * bricksize.y, bz * bricksize.z);
        brick_boundaries[i].max += vec3i(bx * bricksize.x, by * bricksize.y, bz * bricksize.z);
      }
    }
  }

  bounds.invalidate();

  for (size_t i = 0; i < brick_boundaries.size(); ++i)
  {
    aabbi bb = brick_boundaries[i];

    if (bb.invalid() || volume(bb) <= 0)
      continue;

    if (!bbox.contains(bb))
      continue;

    bounds = combine(bounds, bb);
  }

  if (bounds.invalid())
    return bbox;
  else
    return bounds;
}

uint64_t PartialSVT::get_count(aabbi bounds) const
{
  vec3i min_brick = bounds.min / bricksize;
  vec3i min_bpos = bounds.min - min_brick * bricksize;

  vec3i max_brick = bounds.max / bricksize;
  vec3i max_bpos = bounds.max - max_brick * bricksize;

  uint64_t count = 0;

  for (int bz = min_brick.z; bz <= max_brick.z; ++bz)
  {
    int minz = bz == min_brick.z ? min_bpos.z : 0;
    int maxz = bz == max_brick.z ? max_bpos.z : bricksize.z;

    for (int by = min_brick.y; by <= max_brick.y; ++by)
    {
      int miny = by == min_brick.y ? min_bpos.y : 0;
      int maxy = by == max_brick.y ? max_bpos.y : bricksize.y;

      for (int bx = min_brick.x; bx <= max_brick.x; ++bx)
      {
        int minx = bx == min_brick.x ? min_bpos.x : 0;
        int maxx = bx == max_brick.x ? max_bpos.x : bricksize.x;

        // 32**3 bricks can store 16 bit values, but the
        // overall count will generally not fit in 16 bits
        count += static_cast<uint64_t>(svts[bz * num_svts.x * num_svts.y + by * num_svts.x + bx].get_count(
              aabbi(vec3i(minx, miny, minz), vec3i(maxx, maxy, maxz))));
      }
    }
  }

  return count;
}


//-------------------------------------------------------------------------------------------------
// Kd-tree (Vidal et al. 2008)
//

struct KdTree
{
  struct Node;
  typedef std::unique_ptr<Node> NodePtr;

  struct Node
  {
    aabbi bbox;
    NodePtr left  = nullptr;
    NodePtr right = nullptr;
    int axis = -1;
    int splitpos = -1;
    int depth;
  };

  template <typename Func>
    void traverse(NodePtr const& n, vec3 eye, Func f) const
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
            traverse(n->left, eye, f);
            traverse(n->right, eye, f);
          }
          else
          {
            traverse(n->right, eye, f);
            traverse(n->left, eye, f);
          }
        }
      }
    }

  PartialSVT psvt;
  //SVT<uint64_t> psvt;

  NodePtr root = nullptr;

  vec3i vox;
  vec3 dist;
  float scale;

  void updateVolume(vvVolDesc const& vd, int channel = 0);

  template <typename Tex>
    void updateTransfunc(Tex transfunc);

  void node_splitting(NodePtr& n);

  std::vector<aabb> get_leaf_nodes(vec3 eye) const;

  // Need OpenGL context!
  void renderGL() const;
  // Need OpenGL context!
  void renderGL(NodePtr const& n) const;
};

void KdTree::updateVolume(vvVolDesc const& vd, int channel)
{
  vox = vec3i(vd.vox.x, vd.vox.y, vd.vox.z);
  dist = vec3(vd.getDist().x, vd.getDist().y, vd.getDist().z);
  scale = vd._scale;

  psvt.reset(vd, aabbi(vec3i(0), vox), channel);
}

template <typename Tex>
void KdTree::updateTransfunc(Tex transfunc)
{
#ifdef BUILD_TIMING
  vvStopwatch sw; sw.start();
#endif
  psvt.build(transfunc);
#ifdef BUILD_TIMING
  std::cout << std::fixed << std::setprecision(3) << "svt update: " << sw.getTime() << " sec.\n";
#endif

#ifdef BUILD_TIMING
  sw.start();
#endif
  root.reset(new Node);
  root->bbox = psvt.boundary(aabbi(vec3i(0), vec3i(vox[0], vox[1], vox[2])));
  root->depth = 0;
  node_splitting(root);
#ifdef BUILD_TIMING
  std::cout << "splitting: " << sw.getTime() << " sec.\n";
#endif
}

void KdTree::node_splitting(KdTree::NodePtr& n)
{
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

  int lmax = len[axis];

  static const int dl = 4; // ``we set dl to be 4 for 256^3 data sets..''
  //  static const int dl = 8; // ``.. and 8 for 512^3 data sets.''

  // Halting criterion 1.b) (should not really get here..)
  if (lmax < dl)
    return;

  int num_planes = lmax / dl;

  int min_cost = INT_MAX;
  int best_p = -1;

  aabbi lbox = n->bbox;
  aabbi rbox = n->bbox;

  int first = lbox.min[axis];

  int vol = volume(n->bbox);

  for (int p = 1; p < num_planes; ++p)
  {
    aabbi ltmp = n->bbox;
    aabbi rtmp = n->bbox;

    ltmp.max[axis] = first + dl * p;
    rtmp.min[axis] = first + dl * p;

    ltmp = psvt.boundary(ltmp);
    rtmp = psvt.boundary(rtmp);

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
      best_p = p;
    }
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

std::vector<aabb> KdTree::get_leaf_nodes(vec3 eye) const
{
  std::vector<aabb> result;

  traverse(root, eye, [&result,this,eye](NodePtr const& n)
  {
    if (n->left == nullptr && n->right == nullptr)
    {
      auto bbox = n->bbox;
      bbox.max.y = vox[1] - bbox.max.y - 1;
      bbox.min.y = vox[1] - bbox.min.y - 1;
      bbox.max.z = vox[2] - bbox.max.z - 1;
      bbox.min.z = vox[2] - bbox.min.z - 1;
      vec3 bmin = (vec3(bbox.min) - vec3(vox)/2.f) * dist * scale;
      vec3 bmax = (vec3(bbox.max) - vec3(vox)/2.f) * dist * scale;
//std::cout << length(aabb(bmin, bmax).center() - eye) << '\n';

      result.push_back(aabb(bmin, bmax));
    }
  });

  return result;
}

void KdTree::renderGL() const
{
  renderGL(root);
}

void KdTree::renderGL(KdTree::NodePtr const& n) const
{
  if (n != nullptr)
  {
    if (n->left == nullptr && n->right == nullptr)
    {
      auto bbox = n->bbox;
      bbox.max.y = vox[1] - bbox.max.y - 1;
      bbox.min.y = vox[1] - bbox.min.y - 1;
      bbox.max.z = vox[2] - bbox.max.z - 1;
      bbox.min.z = vox[2] - bbox.min.z - 1;
      vec3 bmin = (vec3(bbox.min) - vec3(vox)/2.f) * dist * scale;
      vec3 bmax = (vec3(bbox.max) - vec3(vox)/2.f) * dist * scale;

      glBegin(GL_LINES);
      glColor3f(0,0,0);

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

    renderGL(n->left);
    renderGL(n->right);
  }
}

namespace virvo
{

struct SkipTree::Impl
{
  SkipTree::Technique technique;

  KdTree kdtree;
};

SkipTree::SkipTree(SkipTree::Technique tech)
  : impl_(new Impl)
{
  impl_->technique = tech;
}

SkipTree::~SkipTree()
{
}

void SkipTree::updateVolume(const vvVolDesc& vd)
{
  if (impl_->technique == SVTKdTree)
    impl_->kdtree.updateVolume(vd);
}

void SkipTree::updateTransfunc(const uint8_t* data,
        int numEntriesX,
        int numEntriesY,
        int numEntriesZ,
        PixelFormat format)
{
  using namespace visionaray;

  assert(numEntriesX == 1 && numEntriesY == 1); // Currently only 1D TF support

  if (format == PF_RGBA32F)
  {
    texture_ref<visionaray::vec4, 1> transfunc(numEntriesX);
    transfunc.reset(reinterpret_cast<const visionaray::vec4*>(data));
    transfunc.set_address_mode(Clamp);
    transfunc.set_filter_mode(Nearest);

    impl_->kdtree.updateTransfunc(transfunc);
  }
}

std::vector<aabb> SkipTree::getSortedBricks(vec3 eye)
{
  std::vector<aabb> result;

  if (impl_->technique == SVTKdTree)
  {
    auto leaves = impl_->kdtree.get_leaf_nodes(visionaray::vec3(eye.x, eye.y, eye.z));

    result.resize(leaves.size());

    for (size_t i = 0; i < leaves.size(); ++i)
    {
      const auto& leaf = leaves[i];

      result[i].min = virvo::vec3(leaf.min.x, leaf.min.y, leaf.min.z);
      result[i].max = virvo::vec3(leaf.max.x, leaf.max.y, leaf.max.z);
    }
  }

  return result;
}

} // namespace virvo

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
