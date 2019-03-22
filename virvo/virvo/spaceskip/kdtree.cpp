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

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#ifdef VV_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#include <virvo/vvopengl.h>

#include "kdtree.h"

#include <virvo/vvspaceskip.h>

KdTree::~KdTree()
{
  cudaFree(d_nodes);
}

void KdTree::updateVolume(vvVolDesc const& vd, int channel)
{
  using namespace visionaray;

  vox = vec3i(vd.vox.x, vd.vox.y, vd.vox.z);
  dist = vec3(vd.getDist().x, vd.getDist().y, vd.getDist().z);
  scale = vd._scale;

  psvt.reset(vd, aabbi(vec3i(0), vox), channel);
}

void KdTree::node_splitting(int index)
{
  using namespace visionaray;

  auto s = nodes[index].bbox.size();
  int64_t vol = static_cast<int64_t>(s.x) * s.y * s.z;

  auto rs = nodes[0].bbox.size();
  int64_t root_vol = static_cast<int64_t>(rs.x) * rs.y * rs.z;

  // Halting criterion 1.)
  //if (vol < root_vol / 10)
  if (vol <= 8*8*8)
    return;

  // Split along longest axis
  vec3i len = nodes[index].bbox.max - nodes[index].bbox.min;

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

  aabbi lbox = nodes[index].bbox;
  aabbi rbox = nodes[index].bbox;

  int first = lbox.min[axis];

  for (int p = 1; p < num_planes; ++p)
  {
    aabbi ltmp = nodes[index].bbox;
    aabbi rtmp = nodes[index].bbox;

    ltmp.max[axis] = first + dl * p;
    rtmp.min[axis] = first + dl * p;

    ltmp = psvt.boundary(ltmp);
    rtmp = psvt.boundary(rtmp);

    auto ls = ltmp.size();
    auto rs = rtmp.size();

    int64_t lvol = static_cast<int64_t>(ls.x) * ls.y * ls.z;
    int64_t rvol = static_cast<int64_t>(rs.x) * rs.y * rs.z;
    int64_t c = volume(ltmp) + volume(rtmp);

    // empty-space volume
    int64_t ev = vol - c;

    // Halting criterion 2.)
    //if (ev <= std::max(vol / 2000, (int64_t)13824))
    //if (ev <= 32*32*32)
    if (ev <= vol / 1000 || vol <= 16*16*16)
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
  nodes[index].axis = axis;
  nodes[index].splitpos = first + dl * best_p;

  nodes[index].left = static_cast<int>(nodes.size());
  nodes[index].right = static_cast<int>(nodes.size()) + 1;

  Node left;
  left.bbox = lbox;
  nodes.emplace_back(left);

  Node right;
  right.bbox = rbox;
  nodes.emplace_back(right);

  node_splitting(nodes[index].left);
  node_splitting(nodes[index].right);
}

virvo::SkipTreeNode* KdTree::getNodesDevPtr(int& numNodes)
{
#ifdef VV_HAVE_CUDA
  using visionaray::vec3;

  numNodes = static_cast<int>(nodes.size());

  std::vector<virvo::SkipTreeNode> tmp(numNodes);

  for (int i = 0; i < numNodes; ++i)
  {
    auto bbox = nodes[i].bbox;
    bbox.min.y = vox[1] - nodes[i].bbox.max.y;
    bbox.max.y = vox[1] - nodes[i].bbox.min.y;
    bbox.min.z = vox[2] - nodes[i].bbox.max.z;
    bbox.max.z = vox[2] - nodes[i].bbox.min.z;
    vec3 bmin = (vec3(bbox.min) - vec3(vox)/2.f) * dist * scale;
    vec3 bmax = (vec3(bbox.max) - vec3(vox)/2.f) * dist * scale;

    tmp[i].min_corner[0] = bmin.x;
    tmp[i].min_corner[1] = bmin.y;
    tmp[i].min_corner[2] = bmin.z;
    tmp[i].left          = nodes[i].left;

    tmp[i].max_corner[0] = bmax.x;
    tmp[i].max_corner[1] = bmax.y;
    tmp[i].max_corner[2] = bmax.z;
    tmp[i].right         = nodes[i].right;
  }

  cudaFree(d_nodes);
  cudaError_t err = cudaMalloc(reinterpret_cast<void**>(&d_nodes), sizeof(virvo::SkipTreeNode) * nodes.size());
  if (err != cudaSuccess)
  {
    numNodes = 0;
    return nullptr;
  }

  err = cudaMemcpy(d_nodes, tmp.data(), sizeof(virvo::SkipTreeNode) * nodes.size(), cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
    cudaFree(d_nodes);
    d_nodes = nullptr;
    numNodes = 0;
  }

  return d_nodes;
#else
  numNodes = 0;
  return nullptr;
#endif
}

std::vector<visionaray::aabb> KdTree::get_leaf_nodes(visionaray::vec3 eye, bool frontToBack) const
{
  using namespace visionaray;

  std::vector<aabb> result;

  traverse(0 /*root*/, eye, [&result,this](Node const& n)
  {
    if (n.left == -1 && n.right == -1)
    {
      auto bbox = n.bbox;
      bbox.min.y = vox[1] - n.bbox.max.y;
      bbox.max.y = vox[1] - n.bbox.min.y;
      bbox.min.z = vox[2] - n.bbox.max.z;
      bbox.max.z = vox[2] - n.bbox.min.z;
      vec3 bmin = (vec3(bbox.min) - vec3(vox)/2.f) * dist * scale;
      vec3 bmax = (vec3(bbox.max) - vec3(vox)/2.f) * dist * scale;

      result.push_back(aabb(bmin, bmax));
    }
  }, frontToBack);

  return result;
}

void KdTree::renderGL(vvColor color) const
{
  renderGL(0 /*root*/, color);
}

void KdTree::renderGL(int index, vvColor color) const
{
  using namespace visionaray;

  if (index >= 0 && index < nodes.size())
  {
    Node const& n = nodes[index];

    if (n.left == -1 && n.right == -1)
    {
      auto bbox = n.bbox;
      bbox.min.y = vox[1] - n.bbox.max.y;
      bbox.max.y = vox[1] - n.bbox.min.y;
      bbox.min.z = vox[2] - n.bbox.max.z;
      bbox.max.z = vox[2] - n.bbox.min.z;
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

    renderGL(n.left, color);
    renderGL(n.right, color);
  }
}
