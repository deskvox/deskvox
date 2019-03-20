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

#ifndef VV_SPACESKIP_CUDA_KDTREE_H
#define VV_SPACESKIP_CUDA_KDTREE_H

#include <memory>

#undef MATH_NAMESPACE

#include <visionaray/math/aabb.h>
#include <visionaray/math/forward.h>

#include <visionaray/texture/texture.h>

#undef MATH_NAMESPACE

#include <virvo/vvcolor.h>

class vvVolDesc;

namespace virvo
{

  struct SkipTreeNode;

  class CudaKdTree
  {
  public:

    CudaKdTree();
   ~CudaKdTree();

    void updateVolume(vvVolDesc const& vd, int channel = 0);

    void updateTransfunc(const visionaray::texture_ref<visionaray::vec4, 1>& transfunc);

    SkipTreeNode* getNodesDevPtr(int& numNodes);

    std::vector<visionaray::aabb> get_leaf_nodes(visionaray::vec3 eye, bool frontToBack) const;

    void renderGL(vvColor color) const;

  private:

    struct Impl;
    std::unique_ptr<Impl> impl_;

  };

} // virvo

#endif // VV_SPACESKIP_CUDA_KDTREE_H

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
