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

#ifndef VV_SPACESKIP_KDTREE_H
#define VV_SPACESKIP_KDTREE_H

#include <memory>
#include <vector>

#include "svt.h"

#define FRAME_TIMING 0
#define BUILD_TIMING 1
#define KDTREE       1

//-------------------------------------------------------------------------------------------------
// Kd-tree (Vidal et al. 2008)
//

struct KdTree
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
  void traverse(NodePtr const& n, Func f, bool frontToBack = true)
  {
    if (n != nullptr)
    {
      f(n);

      if (frontToBack)
      {
        traverse(n->left, f, frontToBack);
        traverse(n->right, f, frontToBack);
      }
      else
      {
        traverse(n->right, f, frontToBack);
        traverse(n->left, f, frontToBack);
      }
    }
  }

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

  PartialSVT psvt;
  //SVT<uint64_t> psvt;

  NodePtr root = nullptr;

  visionaray::vec3i vox;
  visionaray::vec3 dist;
  float scale;

  void updateVolume(vvVolDesc const& vd, int channel = 0);

  template <typename Tex>
  void updateTransfunc(Tex transfunc);

  void node_splitting(NodePtr& n);

  std::vector<visionaray::aabb> get_leaf_nodes(visionaray::vec3 eye, bool frontToBack) const;

  // Need OpenGL context!
  void renderGL(vvColor color) const;
  // Need OpenGL context!
  void renderGL(NodePtr const& n, vvColor color) const;
};

#include "kdtree.inl"

#endif // VV_SPACESKIP_KDTREE_H
