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

#include <GL/glew.h>

#include "vvaabb.h"
#include "vvbsptree.h"
#include "vvbsptreevisitors.h"
#include "vvdebugmsg.h"
#include "vvgltools.h"
#include "vvimage.h"
#include "vvrenderer.h"
#include "vvvoldesc.h"

vvSortLastVisitor::vvSortLastVisitor()
  : vvVisitor()
{
  vvDebugMsg::msg(1, "vvSortLastVisitor::vvSortLastVisitor()");
}

vvSortLastVisitor::~vvSortLastVisitor()
{
  vvDebugMsg::msg(1, "vvSortLastVisitor::~vvSortLastVisitor()");
}

//----------------------------------------------------------------------------
/** Sort-last visitor visit method. Supplies logic to render results of worker
    thread.
  @param obj  node to render
*/
void vvSortLastVisitor::visit(vvVisitable* obj) const
{
  vvDebugMsg::msg(3, "vvSortLastVisitor::visit()");

  vvBspNode* node = dynamic_cast<vvBspNode*>(obj);

  if (node->isLeaf())
  {
    const Texture& tex = _textures.at(node->getId());
    glWindowPos2i(tex.rect->x, tex.rect->y);
    glDrawPixels(tex.rect->width, tex.rect->height, GL_RGBA, GL_FLOAT, &(*tex.pixels)[0]);
  }
}

void vvSortLastVisitor::setTextures(const std::vector<Texture>& textures)
{
  vvDebugMsg::msg(3, "vvSortLastVisitor::setTextures()");

  _textures = textures;
}

vvSimpleRenderVisitor::vvSimpleRenderVisitor(const std::vector<vvRenderer*>& renderers)
  : vvVisitor()
  , _renderers(renderers)
{
  vvDebugMsg::msg(1, "vvSimpleRenderVisitor::vvSimpleRenderVisitor()");
}

vvSimpleRenderVisitor::~vvSimpleRenderVisitor()
{
  vvDebugMsg::msg(1, "vvSimpleRenderVisitor::~vvSimpleRenderVisitor()");
}

void vvSimpleRenderVisitor::visit(vvVisitable* obj) const
{
  vvDebugMsg::msg(1, "vvSimpleRenderVisitor::visit()");
  vvBspNode* node = dynamic_cast<vvBspNode*>(obj);
  if (node->isLeaf())
  {
    _renderers[node->getId()]->renderVolumeGL();
  }
}

vvShowBricksVisitor::vvShowBricksVisitor(vvVolDesc* vd)
  : vvVisitor()
  , _vd(vd)
{
  vvDebugMsg::msg(1, "vvShowBricksVisitor::vvShowBricksVisitor()");
}

vvShowBricksVisitor::~vvShowBricksVisitor()
{
  vvDebugMsg::msg(1, "vvShowBricksVisitor::~vvShowBricksVisitor()");
}

void vvShowBricksVisitor::visit(vvVisitable* obj) const
{
  vvDebugMsg::msg(3, "vvShowBricksVisitor::visit()");

  vvBspNode* node = dynamic_cast<vvBspNode*>(obj);

  if (node->isLeaf())
  {
    // convert voxel to obj coordinates
    vvAABB objAabb = vvAABB(_vd->objectCoords(node->getAabb().getMin()),
                            _vd->objectCoords(node->getAabb().getMax()));
    vvGLTools::render(objAabb);
  }
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
