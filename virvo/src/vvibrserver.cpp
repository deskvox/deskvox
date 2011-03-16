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
#include "vvibrserver.h"
#include "vvrayrend.h"
#include "vvcudaimg.h"

#ifdef HAVE_BONJOUR
#include "vvbonjour/vvbonjourregistrar.h"
#endif

vvIbrServer::vvIbrServer(const vvImage2_5d::DepthPrecision dp)
  : vvRemoteServer(), _depthPrecision(dp)
{
}

vvIbrServer::~vvIbrServer()
{
  delete _socket;
}

void vvIbrServer::setDepthPrecision(const vvImage2_5d::DepthPrecision dp)
{
  _depthPrecision = dp;
}

//----------------------------------------------------------------------------
/** Perform remote rendering, read back pixel data and send it over socket
    connections using a vvImage instance.
*/
void vvIbrServer::renderImage(vvMatrix& pr, vvMatrix& mv, vvRenderer* renderer)
{
  vvDebugMsg::msg(3, "vvIsaServer::renderImage()");

  // Render volume:
  float matrixGL[16];

  glMatrixMode(GL_PROJECTION);
  pr.get(matrixGL);
  glLoadMatrixf(matrixGL);

  glMatrixMode(GL_MODELVIEW);
  mv.get(matrixGL);
  glLoadMatrixf(matrixGL);

  vvRect* screenRect = renderer->getVolDesc()->getBoundingBox().getProjectedScreenRect();

  vvRayRend* rayRend = dynamic_cast<vvRayRend*>(renderer);
  assert(rayRend != NULL);

  rayRend->setDepthPrecision(_depthPrecision);
  rayRend->compositeVolume(screenRect->width, screenRect->height);

  glFlush();

  // Fetch rendered image
  uchar* pixels = new uchar[screenRect->width*screenRect->height*4];
  cudaMemcpy(pixels, dynamic_cast<vvCudaImg*>(rayRend->intImg)->getDImg(), screenRect->width * screenRect->height*4, cudaMemcpyDeviceToHost);

  vvImage2_5d* im2;
  switch(_depthPrecision)
  {
  case vvImage2_5d::VV_UCHAR:
    {
      im2 = new vvImage2_5d(screenRect->height, screenRect->width, (unsigned char*)pixels, vvImage2_5d::VV_UCHAR);
      im2->setDepthPrecision(_depthPrecision);
      im2->alloc_pd();
      uchar* depthUchar = im2->getpixeldepthUchar();
      cudaMemcpy(depthUchar, rayRend->_depthUchar, screenRect->width * screenRect->height *sizeof(uchar), cudaMemcpyDeviceToHost);
      cudaFree(rayRend->_depthUchar);
    }
    break;
  case vvImage2_5d::VV_USHORT:
    {
      im2 = new vvImage2_5d(screenRect->height, screenRect->width, (unsigned char*)pixels, vvImage2_5d::VV_USHORT);
      im2->setDepthPrecision(_depthPrecision);
      im2->alloc_pd();
      ushort* depthUshort = im2->getpixeldepthUshort();
      cudaMemcpy(depthUshort, rayRend->_depthUshort, screenRect->width * screenRect->height *sizeof(ushort), cudaMemcpyDeviceToHost);
      cudaFree(rayRend->_depthUshort);
    }
    break;
  case vvImage2_5d::VV_UINT:
    {
      im2 = new vvImage2_5d(screenRect->height, screenRect->width, (unsigned char*)pixels, vvImage2_5d::VV_UINT);
      im2->alloc_pd();
      uint* depthUint = im2->getpixeldepthUint();
      cudaMemcpy(depthUint, rayRend->_depthUint, screenRect->width * screenRect->height *sizeof(uint), cudaMemcpyDeviceToHost);
      cudaFree(rayRend->_depthUint);
    }
    break;
  }
  _socket->putImage2_5d(im2);

  delete[] pixels;
  delete im2;
}

void vvIbrServer::resize(const int w, const int h)
{
  glViewport(0, 0, w, h);
}
