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

#ifndef VV_SPACESKIP_H
#define VV_SPACESKIP_H

#include <cstring>
#include <memory>
#include <vector>

#include "math/aabb.h"
#include "math/vector.h"
#include "vvcolor.h"
#include "vvexport.h"
#include "vvmacros.h"
#include "vvpixelformat.h"

class vvVolDesc;

namespace virvo
{
  /** Compact node returned by \see SkipTree::getNodes()
   */
  struct VV_ALIGN(32) SkipTreeNode
  {
    float min_corner[3];
    int left;
    float max_corner[3];
    int right;
  };


  /** Space skipping tree wrapper, internally supports multiple
   *  construction techniques
   */
  class SkipTree
  {
  public:

    enum Technique
    {
      /** Space skipping k-d tree technique from
       * "Rapid k-d Tree Construction for Sparse Volume Data"
       */
      SVTKdTree,

      /** Rapid k-d Tree Construction, with CUDA
       */
      SVTKdTreeCU,

      /** Space skipping technique from
       * "A Linear Time BVH Construction Algorithm for Sparse Volumes"
       */
      LBVH,
    };

    VVAPI SkipTree(Technique tech);
    VVAPI ~SkipTree();

    VVAPI Technique getTechnique() const;

    VVAPI void updateVolume(const vvVolDesc& vd);
    VVAPI void updateTransfunc(const uint8_t* data,
        int numEntriesX,
        int numEntriesY = 1, // for 2D TF
        int numEntriesZ = 1, // for 3D TF
        PixelFormat format = PF_RGBA32F);

    /**
     * @brief Get pointer to tree nodes
     *    Not all techniques implement this function!
     */
    VVAPI SkipTreeNode* getNodes(int& numNodes);

    /**
     * @brief Produce a sorted list of bricks that contain non-empty voxels
     */
    VVAPI std::vector<aabb> getSortedBricks(vec3 eye, bool frontToBack = true);


    /**
     * @brief Render with OpenGL (need an OpenGL context)
     */
    VVAPI void renderGL(vvColor color = vvColor(1.0f, 1.0f, 1.0f));

  private:

    struct Impl;
    std::unique_ptr<Impl> impl_;

  };

} // virvo

#endif // VV_SPACESKIP_H

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
