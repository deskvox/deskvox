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

#include "vvgltools.h"
#include "vvibr.h"
#include "vvibrrenderer.h"
#include "vvibrserver.h"
#include "vvdebugmsg.h"
#include "vvvoldesc.h"
#include "vvibrimage.h"
#include "vvsocketio.h"
#include "vvtoolshed.h"

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

vvIbrServer::vvIbrServer(vvSocket *socket)
: vvRemoteServer(socket)
, _ibrMode(vvRenderer::VV_GRADIENT)
, _image(NULL)
{
  vvDebugMsg::msg(1, "vvIbrServer::vvIbrServer()");
}

vvIbrServer::~vvIbrServer()
{
  vvDebugMsg::msg(1, "vvIbrServer::~vvIbrServer()");

  delete _image;
}

//----------------------------------------------------------------------------
/** Perform remote rendering, read back pixel data and send it over socket
    connections using a vvImage instance.
*/
void vvIbrServer::renderImage(const vvMatrix& pr, const vvMatrix& mv, vvRenderer* renderer)
{
  vvDebugMsg::msg(3, "vvIbrServer::renderImage()");

  // Render volume:
  float matrixGL[16];

  glMatrixMode(GL_PROJECTION);
  pr.getGL(matrixGL);
  glLoadMatrixf(matrixGL);

  glMatrixMode(GL_MODELVIEW);
  mv.getGL(matrixGL);
  glLoadMatrixf(matrixGL);

  vvIbrRenderer* ibrRenderer = dynamic_cast<vvIbrRenderer*>(renderer);
  if (ibrRenderer == NULL)
  {
    vvDebugMsg::msg(0, "No IBR rendering supported. Aborting...");
    return;
  }

  float drMin = 0.0f;
  float drMax = 0.0f;
  vvAABB aabb = vvAABB(vvVector3(), vvVector3());
  renderer->getVolDesc()->getBoundingBox(aabb);
  vvIbr::calcDepthRange(pr, mv, aabb, drMin, drMax);

  renderer->setParameter(vvRenderer::VV_IBR_DEPTH_RANGE, vvVector2(drMin, drMax));

  int dp = renderer->getParameter(vvRenderer::VV_IBR_DEPTH_PREC);
  ibrRenderer->compositeVolume();

  // Fetch rendered image
  virvo::Viewport vp = vvGLTools::getViewport();
  const int w = vp[2];
  const int h = vp[3];
  if(!_image || _image->getWidth() != w || _image->getHeight() != h
      || _image->getDepthPrecision() != dp)
  {
    _pixels.resize(w*h*4);
    _depth.resize(w*h*(dp/8));
    if(_image)
    {
      _image->setDepthPrecision(dp);
      _image->setNewImage(h, w, &_pixels[0]);
    }
    else
    {
      _image = new vvIbrImage(h, w, &_pixels[0], dp);
    }
    _image->alloc_pd();
  }
  else
  {
    _image->setNewImagePtr(&_pixels[0]);
  }
  _image->setNewDepthPtr(&_depth[0]);

  uchar* p = &_pixels[0];
  ibrRenderer->getColorBuffer(&p);
  p = &_depth[0];
  ibrRenderer->getDepthBuffer(&p);

  _image->setModelViewMatrix(mv);
  _image->setProjectionMatrix(pr);
  _image->setViewport(vp);
  _image->setDepthRange(drMin, drMax);
  const int size = _image->encode(_codetype, 0, h-1, 0, w-1);
  if (size > 0)
  {
    if (_socketio->putIbrImage(_image) != vvSocket::VV_OK)
    {
      vvDebugMsg::msg(1, "Error sending image over socket...");
    }
  }
  else
  {
    vvDebugMsg::msg(1, "Error encoding image...");
  }
}

void vvIbrServer::resize(const int w, const int h)
{
  vvRemoteServer::resize(w, h);
  glViewport(0, 0, w, h);
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
