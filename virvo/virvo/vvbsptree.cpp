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

#include <cmath>
#include <set>

#include "vvbrick.h"
#include "vvbsptree.h"
#include "vvdebugmsg.h"
#include "vvgltools.h"
#include "vvopengl.h"
#include "vvtoolshed.h"
#include "vvvoldesc.h"

//============================================================================
// vvBspNode Method Definitions
//============================================================================

vvBspNode::vvBspNode(const vvAABBi& aabb)
  : _aabb(aabb)
{
  _childLeft = NULL;
  _childRight = NULL;
}

vvBspNode::~vvBspNode()
{
  delete _childLeft;
  delete _childRight;
}

void vvBspNode::accept(vvVisitor* visitor)
{
  visitor->visit(this);
}

void vvBspNode::addChild(vvBspNode* child)
{
  if (_childLeft == NULL)
  {
    _childLeft = child;
  }
  else
  {
    _childRight = child;
  }
}

bool vvBspNode::isLeaf() const
{
  return (_childLeft == NULL && _childRight == NULL);
}

void vvBspNode::setId(const int id)
{
  _id = id;
}

void vvBspNode::setAabb(const vvAABBi& aabb)
{
  _aabb = aabb;
}

int vvBspNode::getId() const
{
  return _id;
}

vvBspNode* vvBspNode::getChildLeft() const
{
  return _childLeft;
}

vvBspNode* vvBspNode::getChildRight() const
{
  return _childRight;
}

const vvAABBi& vvBspNode::getAabb() const
{
  return _aabb;
}

void vvBspNode::clipProbe(vvVector3& probeMin, vvVector3& probeMax,
                          vvVector3&, vvVector3&) const
{
  const vvVector3i probeMinI = vvVector3i(static_cast<int>(probeMin[0]),
                                          static_cast<int>(probeMin[1]),
                                          static_cast<int>(probeMin[2]));

  const vvVector3i probeMaxI = vvVector3i(static_cast<int>(probeMax[0]),
                                          static_cast<int>(probeMax[1]),
                                          static_cast<int>(probeMax[2]));

  vvAABBi probe(probeMinI, probeMaxI);
  probe.intersect(_aabb);

  for (int i = 0; i < 3; ++i)
  {
    probeMin[i] = static_cast<float>(probe.getMin()[i]);
    probeMax[i] = static_cast<float>(probe.getMax()[i]);
  }
}

//============================================================================
// vvBspTree Method Definitions
//============================================================================

vvBspTree::vvBspTree(vvVolDesc* vd, const vvBspData& data)
  : _vd(vd)
  , _root(NULL)
  , _visitor(NULL)
  , _data(data)
{
  // load balance vector is optional
  if (_data.loadBalance.size() < 1)
  {
    const float fraction = 1.0f / static_cast<float>(_data.numLeafs);

    for (int i = 0; i < _data.numLeafs; ++i)
    {
      _data.loadBalance.push_back(fraction);
    }
  }

  float totalLoad = 0.0f;
  for (std::vector<float>::const_iterator it = _data.loadBalance.begin();
       it != _data.loadBalance.end(); ++it)
  {
    totalLoad += *it;
  }

  // check if total load sums up to appr. 1
  if ((totalLoad < 1.0f - FLT_EPSILON) || (totalLoad > 1.0f + FLT_EPSILON))
  {
    _root = NULL;
    vvDebugMsg::msg(0, "vvBspTree::vvBspTree() - Error: load balance must sum up to 1: ", totalLoad);
    return;
  }

  vvVector3i voxMin = vvVector3i(0, 0, 0);
  vvVector3i voxMax = vd->vox;
  _leafs.resize(_data.loadBalance.size());

  if (_leafs.size() < 1)
  {
    vvDebugMsg::msg(0, "vvBspTree::vvBspTree() - Error: no leafs");
    return;
  }

  if (_leafs.size() > 1)
  {
    _root = new vvBspNode(vvAABBi(voxMin, voxMax));
    buildHierarchy(_root, 0);
  }
  else
  {
    _root = new vvBspNode(vvAABBi(voxMin, voxMax));
    _root->setId(0);
    _leafs[0] = _root;
  }
}

vvBspTree::~vvBspTree()
{
  // only delete _root and _leafs, the renderer is responsible
  // for deleting the single _visitor instance
  delete _root;
}

void vvBspTree::traverse(const vvVector3i& pos) const
{
  traverse(pos, _root);
}

const std::vector<vvBspNode*>& vvBspTree::getLeafs() const
{
  return _leafs;
}

void vvBspTree::setVisitor(vvVisitor* visitor)
{
  _visitor = visitor;
}

void vvBspTree::buildHierarchy(vvBspNode* node, const uint leafIdx)
{
  const float fraction = calcRelativeFraction(leafIdx);
  const vvAABBi aabb = node->getAabb();
  vvVecmath::AxisType axis;
  const int length = aabb.getLongestSide(axis);
  const float split = static_cast<float>(length) * fraction;
  std::pair<vvAABBi, vvAABBi> splitted = aabb.split(axis, static_cast<int>(split));

  if (leafIdx == _leafs.size() - 2)
  {
    vvBspNode* childLeft = new vvBspNode(splitted.first);
    vvBspNode* childRight = new vvBspNode(splitted.second);
    node->addChild(childLeft);
    node->addChild(childRight);

    // make left child a leaf
    _leafs.at(leafIdx) = childLeft;
    childLeft->setId(leafIdx);

    // make right child a leaf
    _leafs.at(leafIdx + 1) = childRight;
    childRight->setId(leafIdx + 1);
  }
  else if (leafIdx < _leafs.size() - 2)
  {
    vvBspNode* childLeft = new vvBspNode(splitted.first);
    vvBspNode* childRight = new vvBspNode(splitted.second);
    node->addChild(childLeft);
    node->addChild(childRight);

    // make left child a leaf
    _leafs.at(leafIdx) = childLeft;
    childLeft->setId(leafIdx);
    
    // recurse with right child
    buildHierarchy(childRight, leafIdx + 1);
  }
}

float vvBspTree::calcRelativeFraction(const int leafIdx)
{
  float total = 0.0f;
  for (int i = leafIdx; i < static_cast<int>(_leafs.size()); ++i)
  {
    total += _data.loadBalance[i];
  }
  return _data.loadBalance[leafIdx] / total;
}

void vvBspTree::traverse(const vvVector3i& pos, vvBspNode* node) const
{
  if (node->isLeaf())
  {
    // since this bsp tree implementation utilized the visitor
    // pattern, rendering is initiated using the visit() / accept()
    // mechanism of the half space node / its visitor
    node->accept(_visitor);
  }
  else
  {
    vvVector3i minval = node->getChildLeft()->getAabb().getMin();
    vvVector3i maxval = node->getChildLeft()->getAabb().getMax();

    for (int i = 0; i < 3; ++i)
    {
      if (minval[i] == node->getAabb().getMin()[i])
      {
        minval[i] = -std::numeric_limits<int>::max() + 1;
      }

      if (maxval[i] == node->getAabb().getMax()[i])
      {
        maxval[i] = std::numeric_limits<int>::max();
      }
    }
    const vvAABBi aabb = vvAABBi(minval, maxval);

    // back-to-front traversal
    if (aabb.contains(pos))
    {
      traverse(pos, node->getChildRight());
      traverse(pos, node->getChildLeft());
    }
    else
    {
      traverse(pos, node->getChildLeft());
      traverse(pos, node->getChildRight());
    }
  }
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
