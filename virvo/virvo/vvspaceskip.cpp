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

#include "spaceskip/kdtree.h"
#include "spaceskip/lbvh.h"

#include "spaceskip/grid.h"
#include "spaceskip/cudakdtree.h"

#include "vvclock.h"
#include "vvopengl.h"
#include "vvspaceskip.h"
#include "vvvoldesc.h"

namespace virvo
{

struct SkipTree::Impl
{
  SkipTree::Technique technique;

  KdTree kdtree;
  CudaKdTree cuda_kdtree;
  BVH bvh;
  GridAccelerator grid;
};

SkipTree::SkipTree(SkipTree::Technique tech)
  : impl_(new Impl)
{
  impl_->technique = tech;
}

SkipTree::~SkipTree()
{
}

SkipTree::Technique SkipTree::getTechnique() const
{
  return impl_->technique;
}

void SkipTree::updateVolume(const vvVolDesc& vd)
{
  if (impl_->technique == SVTKdTree)
    impl_->kdtree.updateVolume(vd);
  else if (impl_->technique == SVTKdTreeCU)
    impl_->cuda_kdtree.updateVolume(vd);
  else if (impl_->technique == LBVH)
    impl_->bvh.updateVolume(vd);
  else if (impl_->technique == Grid)
    impl_->grid.updateVolume(vd);
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

    if (impl_->technique == SVTKdTree)
      impl_->kdtree.updateTransfunc(transfunc);
    else if (impl_->technique == SVTKdTreeCU)
      impl_->cuda_kdtree.updateTransfunc(transfunc);
    else if (impl_->technique == LBVH)
      impl_->bvh.updateTransfunc(transfunc);
    else if (impl_->technique == Grid)
      impl_->grid.updateTransfunc(transfunc);
  }
}

SkipTreeNode* SkipTree::getNodesDevPtr(int& numNodes)
{
  if (impl_->technique == SVTKdTree)
    return impl_->kdtree.getNodesDevPtr(numNodes);
  else if (impl_->technique == SVTKdTreeCU)
    return impl_->cuda_kdtree.getNodesDevPtr(numNodes);
  else if (impl_->technique == LBVH)
    return impl_->bvh.getNodesDevPtr(numNodes);

  numNodes = 0;
  return nullptr;
}

std::vector<aabb> SkipTree::getSortedBricks(vec3 eye, bool frontToBack)
{
  std::vector<aabb> result;

  std::vector<visionaray::aabb> leaves;

  if (impl_->technique == SVTKdTree)
  {
    leaves = impl_->kdtree.get_leaf_nodes(
        visionaray::vec3(eye.x, eye.y, eye.z),
        frontToBack);
  }
  else if (impl_->technique == LBVH)
  {
    leaves = impl_->bvh.get_leaf_nodes(
        visionaray::vec3(eye.x, eye.y, eye.z),
        frontToBack);
  }
  else if (impl_->technique == SVTKdTreeCU)
  {
    leaves = impl_->cuda_kdtree.get_leaf_nodes(
        visionaray::vec3(eye.x, eye.y, eye.z),
        frontToBack
        );
  }

  result.resize(leaves.size());

  for (size_t i = 0; i < leaves.size(); ++i)
  {
    const auto& leaf = leaves[i];

    result[i].min = virvo::vec3(leaf.min.x, leaf.min.y, leaf.min.z);
    result[i].max = virvo::vec3(leaf.max.x, leaf.max.y, leaf.max.z);
  }

  return result;
}

SkipGrid SkipTree::getGrid() const
{
  assert(impl_->technique == Grid);

  return {
    reinterpret_cast<virvo::vec2 const*>(impl_->grid.cellRange),
    virvo::vec3i(impl_->grid.gridDimensions.data()),
    impl_->grid.maxOpacities
    };
}

void SkipTree::renderGL(vvColor color)
{
  if (impl_->technique == SVTKdTree)
    impl_->kdtree.renderGL(color);
  else if (impl_->technique == SVTKdTreeCU)
    impl_->cuda_kdtree.renderGL(color);
  else if (impl_->technique == LBVH)
    impl_->bvh.renderGL(color);
}

} // namespace virvo

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
