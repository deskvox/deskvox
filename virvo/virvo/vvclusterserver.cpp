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

#include "vvgltools.h"
#include "vvclusterserver.h"
#include "vvtexrend.h"

#ifdef HAVE_BONJOUR
#include "vvbonjour/vvbonjourregistrar.h"
#endif

using std::cerr;
using std::endl;

vvClusterServer::vvClusterServer(const BufferPrecision compositingPrecision)
  : vvRemoteServer(),
    _offscreenBuffer(0), _compositingPrecision(compositingPrecision)
{
  vvDebugMsg::msg(1, "vvClusterServer::vvClusterServer()");
}

vvClusterServer::~vvClusterServer()
{
  vvDebugMsg::msg(1, "vvClusterServer::~vvClusterServer()");

  delete _offscreenBuffer;
}

void vvClusterServer::setCompositingPrecision(const BufferPrecision compositingPrecision)
{
  vvDebugMsg::msg(1, "vvClusterServer::setCompositingPrecision()");

  _compositingPrecision = compositingPrecision;
}

BufferPrecision vvClusterServer::getCompositingPrecision() const
{
  vvDebugMsg::msg(1, "vvClusterServer::getCompositingPrecision()");

  return _compositingPrecision;
}

vvRemoteServer::ErrorType vvClusterServer::initBricks(std::vector<vvBrick*>& bricks) const
{
  vvDebugMsg::msg(1, "vvClusterServer::initBricks()");

  const vvSocket::ErrorType err = _socket->getBricks(bricks);
  switch (err)
  {
  case vvSocket::VV_OK:
    cerr << "Brick outlines received" << endl;
    break;
  default:
    cerr << "Unable to retrieve brick outlines" << endl;
    return VV_SOCKET_ERROR;
  }
  return VV_OK;
}

void vvClusterServer::renderLoop(vvRenderer* renderer)
{
  vvDebugMsg::msg(1, "vvClusterServer::renderLoop()");

  vvGLTools::printGLError("enter vvClusterServer::renderLoop()");

  _offscreenBuffer = new vvOffscreenBuffer(1.0f, _compositingPrecision);
  _offscreenBuffer->initForRender();

  vvRemoteServer::renderLoop(renderer);

  vvGLTools::printGLError("leave vvClusterServer::renderLoop()");
}

//----------------------------------------------------------------------------
/** Perform remote rendering, read back pixel data and send it over socket
    connections using a vvImage instance.
*/
void vvClusterServer::renderImage(vvMatrix& pr, vvMatrix& mv, vvRenderer* renderer)
{
  vvDebugMsg::msg(3, "vvClusterServer::renderImage()");

  vvGLTools::printGLError("enter vvClusterServer::renderImage()");

  if (vvDebugMsg::getDebugLevel() > 0)
  {
    _offscreenBuffer->initForRender();
  }

  _offscreenBuffer->bindFramebuffer();
  _offscreenBuffer->clearBuffer();

  // Draw volume:
  float matrixGL[16];

  glMatrixMode(GL_PROJECTION);
  pr.get(matrixGL);
  glLoadMatrixf(matrixGL);

  glMatrixMode(GL_MODELVIEW);
  mv.get(matrixGL);
  glLoadMatrixf(matrixGL);


  vvTexRend* texRend = dynamic_cast<vvTexRend*>(renderer);
  assert(texRend != NULL);

  vvRect* screenRect = texRend->getProbedMask().getProjectedScreenRect();

  texRend->setIsSlave(true);
  texRend->renderVolumeGL();

  glFlush();

  uchar* pixels = new uchar[screenRect->width * screenRect->height * 4];
  glReadPixels(screenRect->x, screenRect->y, screenRect->width, screenRect->height, GL_RGBA, GL_UNSIGNED_BYTE, pixels);
  vvImage img(screenRect->height, screenRect->width, pixels);

  _socket->putImage(&img);
  delete[] pixels;
  _offscreenBuffer->unbindFramebuffer();

  if (vvDebugMsg::getDebugLevel() > 0)
  {
    _offscreenBuffer->writeBack();
  }

  vvGLTools::printGLError("leave vvClusterServer::renderImage()");
}

void vvClusterServer::resize(const int w, const int h)
{
  vvDebugMsg::msg(1, "vvClusterServer::resize()");

  _offscreenBuffer->resize(w, h);
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
