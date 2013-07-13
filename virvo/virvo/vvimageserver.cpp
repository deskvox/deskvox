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

#include "vvimageserver.h"
#include "vvdebugmsg.h"
#include "vvrenderer.h"
#include "vvsocketio.h"

#include "private/vvgltools.h"
#include "private/vvimage.h"

vvImageServer::vvImageServer(vvSocket *socket)
  : vvRemoteServer(socket)
{
  vvDebugMsg::msg(1, "vvImageServer::vvImageServer()");
}

vvImageServer::~vvImageServer()
{
  vvDebugMsg::msg(1, "vvImageServer::~vvImageServer()");
}

//----------------------------------------------------------------------------
/** Perform remote rendering, read back pixel data and send it over socket
    connections using a vvImage instance.
*/
void vvImageServer::renderImage(const vvMatrix& pr, const vvMatrix& mv, vvRenderer* renderer)
{
  vvDebugMsg::msg(3, "vvImageServer::renderImage()");

  // Update matrices
  vvGLTools::setProjectionMatrix(pr);
  vvGLTools::setModelviewMatrix(mv);

  renderer->beginFrame(virvo::CLEAR_COLOR | virvo::CLEAR_DEPTH);
  renderer->renderVolumeGL();
  renderer->endFrame();

  virvo::RenderTarget* rt = renderer->getRenderTarget();

  int w = rt->width();
  int h = rt->height();

  virvo::Image image(w, h, rt->colorFormat());

  // Fetch rendered image
  if (!rt->downloadColorBuffer(image.data(), image.size()))
  {
    return;
  }

  // Compress the image
  image.compress();

  // Send the image
  if (vvSocket::VV_OK != _socketio->putImage(image))
  {
    return;
  }
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
