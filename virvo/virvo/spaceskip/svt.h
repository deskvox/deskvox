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

#ifndef VV_SPACESKIP_SVT_H
#define VV_SPACESKIP_SVT_H

#include <thread>
#include <vector>

#undef MATH_NAMESPACE

#include <visionaray/detail/parallel_for.h> // detail!
#include <visionaray/math/aabb.h>
#include <visionaray/math/forward.h>
#include <visionaray/math/matrix.h>
#include <visionaray/math/rectangle.h>
#include <visionaray/math/vector.h>
#include <visionaray/texture/texture.h>

#undef MATH_NAMESPACE

#include "vvvoldesc.h"

//-------------------------------------------------------------------------------------------------
// Summed-volume table
//

template <typename T>
struct SVT
{
  void reset(visionaray::aabbi bbox);
  void reset(vvVolDesc const& vd, visionaray::aabbi bbox, int channel = 0);

  template <typename Tex>
  void build(Tex transfunc);

  visionaray::aabbi boundary(visionaray::aabbi bbox) const;

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

  T get_count(visionaray::basic_aabb<int> bounds) const
  {
    bounds.min -= visionaray::vec3i(1);
    bounds.max -= visionaray::vec3i(1);

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
void SVT<T>::reset(visionaray::aabbi bbox)
{
  data_.resize(bbox.size().x * bbox.size().y * bbox.size().z);
  width  = bbox.size().x;
  height = bbox.size().y;
  depth  = bbox.size().z;
}

template <typename T>
void SVT<T>::reset(vvVolDesc const& vd, visionaray::aabbi bbox, int channel)
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
        voxels_[index] = vd.getChannelValue(vd.getCurrentFrame(),
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
  using namespace visionaray;

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
visionaray::aabbi SVT<T>::boundary(visionaray::aabbi bbox) const
{
  using namespace visionaray;

  aabbi bounds = bbox;

  // Search for the minimal volume bounding box
  // that contains #voxels contained in bbox!
  uint16_t voxels = get_count(bounds);


  // X boundary
  int x = (bounds.max.x - bounds.min.x) / 2;

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
  int y = (bounds.max.y - bounds.min.y) / 2;

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
  int z = (bounds.max.z - bounds.min.z) / 2;

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

  PartialSVT()
    : pool(std::thread::hardware_concurrency())
  {
  }

  void reset(vvVolDesc const& vd, visionaray::aabbi bbox, int channel = 0);

  template <typename Tex>
  void build(Tex transfunc);

  visionaray::aabbi boundary(visionaray::aabbi bbox);

  uint64_t get_count(visionaray::aabbi bounds) const;

  visionaray::vec3i bricksize = visionaray::vec3i(32, 32, 32);

  visionaray::vec3i num_svts;
  std::vector<svt_t> svts;

  visionaray::thread_pool pool;
};

inline void PartialSVT::reset(vvVolDesc const& vd, visionaray::aabbi bbox, int channel)
{
  using namespace visionaray;

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
inline void PartialSVT::build(Tex transfunc)
{
  using namespace visionaray;

  parallel_for(pool, range1d<size_t>(0, svts.size()), [this, transfunc](size_t i)
  {
    svts[i].build(transfunc);
  });
}

inline visionaray::aabbi PartialSVT::boundary(visionaray::aabbi bbox)
{
  using namespace visionaray;

  aabbi bounds = bbox;

  vec3i min_brick = bounds.min / bricksize;
  vec3i min_bpos = bounds.min - min_brick * bricksize;

  vec3i max_brick = bounds.max / bricksize;
  vec3i max_bpos = bounds.max - max_brick * bricksize;

  vec3i num_bricks = max_brick - min_brick + vec3i(1);
  int n = num_bricks.x * num_bricks.y * num_bricks.z;
  std::vector<aabbi> brick_boundaries(n);

  parallel_for(pool, range1d<int>(0, n), [&](int b)
  {
    int bz = min_brick.z + b / (num_bricks.x * num_bricks.y);
    int by = min_brick.y + (b / num_bricks.x) % num_bricks.y;
    int bx = min_brick.x + b % num_bricks.x;//std::cout << bx << ' ' << by << ' ' << bz << '\n';

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
  });

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

inline uint64_t PartialSVT::get_count(visionaray::aabbi bounds) const
{
  using namespace visionaray;

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

#endif // VV_SPACESKIP_SVT_H
