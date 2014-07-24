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

#include "vvibrserver.h"
#include "vvibr.h"
#include "vvdebugmsg.h"
#include "vvvoldesc.h"
#include "vvsocketio.h"

#include "private/vvgltools.h"
#include "private/vvibrimage.h"

using virvo::recti;
using virvo::vec2f;


vvIbrServer::vvIbrServer(vvSocket *socket)
: vvRemoteServer(socket)
{
  vvDebugMsg::msg(1, "vvIbrServer::vvIbrServer()");
}

vvIbrServer::~vvIbrServer()
{
  vvDebugMsg::msg(1, "vvIbrServer::~vvIbrServer()");
}

//----------------------------------------------------------------------------
/** Perform remote rendering, read back pixel data and send it over socket
    connections using a vvImage instance.
*/
void vvIbrServer::renderImage(const vvMatrix& pr, const vvMatrix& mv, vvRenderer* renderer)
{
  vvDebugMsg::msg(3, "vvIbrServer::renderImage()");

  // Update matrices
  vvGLTools::setProjectionMatrix(pr);
  vvGLTools::setModelviewMatrix(mv);

  // Render volume:
  renderer->beginFrame(virvo::CLEAR_COLOR | virvo::CLEAR_DEPTH);
  renderer->renderVolumeGL();
  renderer->endFrame();

  // Compute depth range
  vvAABB aabb = vvAABB(vvVector3(), vvVector3());

  renderer->getVolDesc()->getBoundingBox(aabb);

  float drMin = 0.0f;
  float drMax = 0.0f;

  virvo::ibr::calcDepthRange(pr, mv, aabb, drMin, drMax);

  renderer->setParameter(vvRenderer::VV_IBR_DEPTH_RANGE, vec2f(drMin, drMax));

  virvo::RenderTarget* rt = renderer->getRenderTarget();

  int w = rt->width();
  int h = rt->height();

  // Create a new IBR image
  virvo::IbrImage image(w, h, rt->colorFormat(), rt->depthFormat());

  image.setDepthMin(drMin);
  image.setDepthMax(drMax);
  image.setViewMatrix(mv);
  image.setProjMatrix(pr);
  image.setViewport(recti(0, 0, w, h));

  // Fetch rendered image
  if (!rt->downloadColorBuffer(image.colorBuffer().data().ptr(), image.colorBuffer().size()))
  {
    return;
  }
  if (!rt->downloadDepthBuffer(image.depthBuffer().data().ptr(), image.depthBuffer().size()))
  {
    return;
  }

  // Compress the image
  image.compress();

  // Send the image
  if (vvSocket::VV_OK != _socketio->putIbrImage(image))
  {
    return;
  }
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
