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

#include <virvo/vvopengl.h>

#include "kdtree.h"

void KdTree::updateVolume(vvVolDesc const& vd, int channel)
{
  using namespace visionaray;

  vox = vec3i(vd.vox.x, vd.vox.y, vd.vox.z);
  dist = vec3(vd.getDist().x, vd.getDist().y, vd.getDist().z);
  scale = vd._scale;

  psvt.reset(vd, aabbi(vec3i(0), vox), channel);
}

void KdTree::node_splitting(KdTree::NodePtr& n)
{
  using namespace visionaray;

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

std::vector<visionaray::aabb> KdTree::get_leaf_nodes(visionaray::vec3 eye, bool frontToBack) const
{
  using namespace visionaray;

  std::vector<aabb> result;

  traverse(root, eye, [&result,this](NodePtr const& n)
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

      result.push_back(aabb(bmin, bmax));
    }
  }, frontToBack);

  return result;
}

void KdTree::renderGL(vvColor color) const
{
  renderGL(root, color);
}

void KdTree::renderGL(KdTree::NodePtr const& n, vvColor color) const
{
  using namespace visionaray;

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
