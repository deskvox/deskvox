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
#include "vvimageserver.h"
#include "vvrenderer.h"
#include "vvsocketio.h"
#include "vvopengl.h"
#include "vvdebugmsg.h"
#include "vvimage.h"

vvImageServer::vvImageServer()
  : vvRemoteServer()
  , _image(NULL)
  , _pixels(NULL)
{
}

vvImageServer::~vvImageServer()
{
  delete _image;
  delete[] _pixels;
}

//----------------------------------------------------------------------------
/** Perform remote rendering, read back pixel data and send it over socket
    connections using a vvImage instance.
*/
void vvImageServer::renderImage(vvMatrix& pr, vvMatrix& mv, vvRenderer* renderer)
{
  vvDebugMsg::msg(3, "vvIsaServer::renderImage()");

  // Render volume:
  float matrixGL[16];

  glMatrixMode(GL_PROJECTION);
  pr.makeGL(matrixGL);
  glLoadMatrixf(matrixGL);

  glMatrixMode(GL_MODELVIEW);
  mv.makeGL(matrixGL);
  glLoadMatrixf(matrixGL);

  glClearColor(0., 0., 0., 0.);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  renderer->renderVolumeGL();

  // Fetch rendered image
  vvGLTools::Viewport vp = vvGLTools::getViewport();
  const int w = vp[2];
  const int h = vp[3];
  if(!_image || _image->getWidth() != w || _image->getHeight() != h)
  {
    delete[] _pixels;
    _pixels = new uchar[w*h*4];
    if(_image)
      _image->setNewImage(h, w, _pixels);
    else
      _image = new vvImage(h, w, _pixels);
  }
  else
  {
    _image->setNewImagePtr(_pixels);
  }

  glReadPixels(0, 0, w, h, GL_RGBA, GL_BYTE, _pixels);

  _image->encode(_codetype, 0, h-1, 0, w-1);
  _socket->putImage(_image);
}

void vvImageServer::resize(const int w, const int h)
{
  glViewport(0, 0, w, h);
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
