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
#include "spaceskip/kdtree.h"
#undef MATH_NAMESPACE

#include "vvspaceskip.h"
#include "vvvoldesc.h"

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

  (void)numEntriesY; (void)numEntriesZ;
  assert(numEntriesY == 1 && numEntriesZ == 1); // Currently only 1D TF support

  if (format == PF_RGBA32F)
  {
    texture_ref<visionaray::vec4, 1> transfunc(numEntriesX);
    transfunc.reset(reinterpret_cast<const visionaray::vec4*>(data));
    transfunc.set_address_mode(Clamp);
    transfunc.set_filter_mode(Nearest);

    impl_->kdtree.updateTransfunc(transfunc);
  }
}

std::vector<aabb> SkipTree::getSortedBricks(vec3 eye, bool frontToBack)
{
  std::vector<aabb> result;

  if (impl_->technique == SVTKdTree)
  {
    auto leaves = impl_->kdtree.get_leaf_nodes(
        visionaray::vec3(eye.x, eye.y, eye.z),
        frontToBack
        );

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

void SkipTree::renderGL(vvColor color)
{
  impl_->kdtree.renderGL(color);
}

} // namespace virvo

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
