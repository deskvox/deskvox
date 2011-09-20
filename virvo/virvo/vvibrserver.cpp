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
#include "vvibrserver.h"
#include "vvrayrend.h"
#include "vvcudaimg.h"

#ifdef HAVE_BONJOUR
#include "vvbonjour/vvbonjourregistrar.h"
#endif

#ifdef HAVE_CUDA
vvIbrServer::vvIbrServer(const vvIbrImage::DepthPrecision dp,
                         const vvRayRend::IbrMode mode)
  : vvRemoteServer(), _depthPrecision(dp), _ibrMode(mode)
{
}

vvIbrServer::~vvIbrServer()
{

}

void vvIbrServer::setDepthPrecision(const vvIbrImage::DepthPrecision dp)
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
  pr.makeGL(matrixGL);
  glLoadMatrixf(matrixGL);

  glMatrixMode(GL_MODELVIEW);
  mv.makeGL(matrixGL);
  glLoadMatrixf(matrixGL);

  vvRect* screenRect = renderer->getVolDesc()->getBoundingBox().getProjectedScreenRect();

  vvRayRend* rayRend = dynamic_cast<vvRayRend*>(renderer);
  assert(rayRend != NULL);

  // calculate bounding sphere
  vvAABB bbox(renderer->getVolDesc()->getBoundingBox().min(), renderer->getVolDesc()->getBoundingBox().max());
  vvVector4 center4(bbox.getCenter()[0], bbox.getCenter()[1], bbox.getCenter()[2], 1.0f);
  vvVector4 min4(bbox.min()[0], bbox.min()[1], bbox.min()[2], 1.0f);
  vvVector4 max4(bbox.max()[0], bbox.max()[1], bbox.max()[2], 1.0f);

  center4.multiply(&mv);
  min4.multiply(&mv);
  max4.multiply(&mv);

  vvVector3 center(center4[0], center4[1], center4[2]);
  vvVector3 min(min4.e[0], min4.e[1], min4.e[2]);
  vvVector3 max(max4.e[0], max4.e[1], max4.e[2]);

  float radius = (max-min).length() * 0.5f;

  // Depth buffer of ibrPlanes
  vvVector3 scal(center);
  scal.normalize();
  scal.scale(radius);
  min = center - scal;
  max = center + scal;

  min4 = vvVector4(&min, 1.f);
  max4 = vvVector4(&max, 1.f);
  min4.multiply(&pr);
  max4.multiply(&pr);
  min4.perspectiveDivide();
  max4.perspectiveDivide();

  rayRend->_ibrPlanes[0] = (min4[2]+1.f)/2.f;
  rayRend->_ibrPlanes[1] = (max4[2]+1.f)/2.f;

  rayRend->setDepthPrecision(_depthPrecision);
  rayRend->compositeVolume();

  // Fetch rendered image
  vvGLTools::Viewport vp = vvGLTools::getViewport();
  uchar* pixels = new uchar[vp[2]*vp[3]*4];
  cudaMemcpy(pixels, dynamic_cast<vvCudaImg*>(rayRend->intImg)->getDeviceImg(), vp[2]*vp[3]*4, cudaMemcpyDeviceToHost);

  vvIbrImage* im2;
  switch(_depthPrecision)
  {
  case vvIbrImage::VV_UCHAR:
    {
      im2 = new vvIbrImage(vp[3], vp[2], (unsigned char*)pixels, vvIbrImage::VV_UCHAR);
      im2->setDepthPrecision(_depthPrecision);
      im2->alloc_pd();
      uchar* depthUchar = im2->getpixeldepthUchar();
      cudaMemcpy(depthUchar, rayRend->_depthUchar, vp[2]*vp[3]*sizeof(uchar), cudaMemcpyDeviceToHost);
      cudaFree(rayRend->_depthUchar);
    }
    break;
  case vvIbrImage::VV_USHORT:
    {
      im2 = new vvIbrImage(vp[3], vp[2], (unsigned char*)pixels, vvIbrImage::VV_USHORT);
      im2->setDepthPrecision(_depthPrecision);
      im2->alloc_pd();
      ushort* depthUshort = im2->getpixeldepthUshort();
      cudaMemcpy(depthUshort, rayRend->_depthUshort, vp[2]*vp[3]*sizeof(ushort), cudaMemcpyDeviceToHost);
      cudaFree(rayRend->_depthUshort);
    }
    break;
  case vvIbrImage::VV_UINT:
    {
      im2 = new vvIbrImage(vp[3], vp[2], (unsigned char*)pixels, vvIbrImage::VV_UINT);
      im2->alloc_pd();
      uint* depthUint = im2->getpixeldepthUint();
      cudaMemcpy(depthUint, rayRend->_depthUint, vp[2]*vp[3]*sizeof(uint), cudaMemcpyDeviceToHost);
      cudaFree(rayRend->_depthUint);
    }
    break;
  }
  vvMatrix vpMatrix;
  vpMatrix.identity();
  vpMatrix.scale(1.0f / (0.5f * vp[2]),
                 1.0f / (0.5f * vp[3]),
                 2.0f);
  vpMatrix.translate((vp[0] / (0.5f * vp[2])) - 1.0f,
                     (vp[1] / (0.5f * vp[3])) - 1.0f,
                     -1.0f);

  vvMatrix invModelviewProjection = mv * pr;
  invModelviewProjection.invert();

  vvMatrix depthScaleMatrix;
  depthScaleMatrix.identity();
  depthScaleMatrix.scale(1.0f, 1.0f, (rayRend->_ibrPlanes[1] - rayRend->_ibrPlanes[0]));
  depthScaleMatrix.translate(0.0f, 0.0f, rayRend->_ibrPlanes[0]);

  im2->setReprojectionMatrix(depthScaleMatrix * vpMatrix * invModelviewProjection);
  _socket->putIbrImage(im2);

  delete[] pixels;
  delete im2;
}

void vvIbrServer::resize(const int w, const int h)
{
  glViewport(0, 0, w, h);
}

#endif
