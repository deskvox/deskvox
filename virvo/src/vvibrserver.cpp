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

#include "vvfileio.h"
#include "vvgltools.h"
#include "vvibrserver.h"
#include "vvrayrend.h"
#include "vvcudaimg.h"

#ifdef HAVE_BONJOUR
#include "vvbonjour/vvbonjourregistrar.h"
#endif

vvIbrServer::vvIbrServer(const BufferPrecision compositingPrecision)
  : _socket(0)
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

vvIbrServer::ErrorType vvIbrServer::initSocket(const int port, const vvSocket::SocketType st)
{
  delete _socket;
  _socket = new vvSocketIO(port, st);
  _socket->set_debuglevel(vvDebugMsg::getDebugLevel());

#ifdef HAVE_BONJOUR
  // Register the bonjour service for the slave.
  vvBonjourRegistrar* registrar = new vvBonjourRegistrar();
  const vvBonjourEntry entry = vvBonjourEntry("Virvo render slave",
                                              "_distrendering._tcp", "");
  registrar->registerService(entry, port);
#endif

  cerr << "Listening on port " << port << endl;

  const vvSocket::ErrorType err = _socket->init();
  _socket->no_nagle();

  if (err != vvSocket::VV_OK)
  {
    return VV_SOCKET_ERROR;
  }
  else
  {
    return VV_OK;
  }
}

vvIbrServer::ErrorType vvIbrServer::initData(vvVolDesc*& vd)
{
  _socket->getBool(_loadVolumeFromFile);

  if (_loadVolumeFromFile)
  {
    char* fn = 0;
    _socket->getFileName(fn);
    cerr << "Load volume from file: " << fn << endl;
    vd = new vvVolDesc(fn);

    vvFileIO* fio = new vvFileIO();
    if (fio->loadVolumeData(vd) != vvFileIO::OK)
    {
      cerr << "Error loading volume file" << endl;
      delete vd;
      delete fio;
      return VV_FILEIO_ERROR;
    }
    else
    {
      vd->printInfoLine();
      delete fio;
    }
  }
  else
  {
    cerr << "Wait for volume data to be transferred..." << endl;
    vd = new vvVolDesc();

    // Get a volume
    switch (_socket->getVolume(vd))
    {
    case vvSocket::VV_OK:
      cerr << "Volume transferred successfully" << endl;
      break;
    case vvSocket::VV_ALLOC_ERROR:
      cerr << "Not enough memory" << endl;
      return VV_SOCKET_ERROR;
    default:
      cerr << "Cannot read volume from socket" << endl;
      return VV_SOCKET_ERROR;
    }
  }
  return VV_OK;
}

vvIbrServer::ErrorType vvIbrServer::initBricks(std::vector<vvBrick*>& bricks) const
{
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

void vvIbrServer::renderLoop(vvRayRend* renderer)
{
  vvSocketIO::CommReason commReason;
  vvMatrix pr;
  vvMatrix mv;
  float quality;
  int mipMode;
  vvVector3 position;
  vvVector3 viewDir;
  vvVector3 objDir;
  int w;
  int h;
  bool interpolation;
  bool roiEnabled;
  vvVector3 roiPos;
  vvVector3 roiSize;
  int currentFrame;
  vvTransFunc tf;

  while (1)
  {
    vvSocket::ErrorType err = _socket->getCommReason(commReason);

    if (err == vvSocket::VV_OK)
    {
      switch (commReason)
      {
      case vvSocketIO::VV_CURRENT_FRAME:
        if ((_socket->getInt32(currentFrame)) == vvSocket::VV_OK)
        {
          renderer->setCurrentFrame(currentFrame);
        }
        break;
      case vvSocketIO::VV_EXIT:
        return;
      case vvSocketIO::VV_MATRIX:
        if ((_socket->getMatrix(&pr) == vvSocket::VV_OK)
           && (_socket->getMatrix(&mv) == vvSocket::VV_OK))
        {
          renderImage(pr, mv, renderer);
        }
        break;
      case vvSocketIO::VV_MIPMODE:
        if ((_socket->getInt32(mipMode)) == vvSocket::VV_OK)
        {
          renderer->setParameter(vvRenderState::VV_MIP_MODE, mipMode);
        }
        break;
      case vvSocketIO::VV_OBJECT_DIRECTION:
        if ((_socket->getVector3(objDir)) == vvSocket::VV_OK)
        {
          renderer->setObjectDirection(&objDir);
        }
        break;
      case vvSocketIO::VV_QUALITY:
        if ((_socket->getFloat(quality)) == vvSocket::VV_OK)
        {
          renderer->setParameter(vvRenderState::VV_QUALITY, quality);
        }
        break;
      case vvSocketIO::VV_POSITION:
        if ((_socket->getVector3(position)) == vvSocket::VV_OK)
        {
          renderer->setPosition(&position);
        }
        break;
      case vvSocketIO::VV_RESIZE:
        if ((_socket->getWinDims(w, h)) == vvSocket::VV_OK)
        {
          glViewport(0, 0, w, h);
        }
        break;
      case vvSocketIO::VV_INTERPOLATION:
        if ((_socket->getBool(interpolation)) == vvSocket::VV_OK)
        {
          renderer->setParameter(vvRenderer::VV_SLICEINT, interpolation ? 1.0f : 0.0f);
        }
        break;
      case vvSocketIO::VV_TOGGLE_BOUNDINGBOX:
        renderer->setParameter(vvRenderState::VV_BOUNDARIES, !((bool)renderer->getParameter(vvRenderState::VV_BOUNDARIES)));
        break;
      case vvSocketIO::VV_TOGGLE_ROI:
        if ((_socket->getBool(roiEnabled)) == vvSocket::VV_OK)
        {
          renderer->setROIEnable(roiEnabled);
        }
        break;
      case vvSocketIO::VV_ROI_POSITION:
        if ((_socket->getVector3(roiPos)) == vvSocket::VV_OK)
        {
          renderer->setParameterV3(vvRenderState::VV_ROI_POS, roiPos);
        }
        break;
      case vvSocketIO::VV_ROI_SIZE:
        if ((_socket->getVector3(roiSize)) == vvSocket::VV_OK)
        {
          renderer->setParameterV3(vvRenderState::VV_ROI_SIZE, roiSize);
        }
        break;
      case vvSocketIO::VV_TRANSFER_FUNCTION:
        tf._widgets.removeAll();
        if ((_socket->getTransferFunction(tf)) == vvSocket::VV_OK)
        {
          renderer->getVolDesc()->tf = tf;
          renderer->updateTransferFunction();
        }
        break;
      case vvSocketIO::VV_VIEWING_DIRECTION:
        if ((_socket->getVector3(viewDir)) == vvSocket::VV_OK)
        {
          renderer->setViewingDirection(&viewDir);
        }
        break;
      default:
        break;
      }
    }
    else if (err == vvSocket::VV_PEER_SHUTDOWN)
    {
      delete _socket;
      _socket = NULL;
      return;
    }
  }
}

//----------------------------------------------------------------------------
/** Perform remote rendering, read back pixel data and send it over socket
    connections using a vvImage instance.
*/
void vvIbrServer::renderImage(vvMatrix& pr, vvMatrix& mv, vvRayRend* renderer)
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
  renderer->setDepthPrecision(_depthPrecision);
  renderer->compositeVolume(screenRect->width, screenRect->height);

  glFlush();

  // Fetch rendered image
  uchar* pixels = new uchar[screenRect->width*screenRect->height*4];
  cudaMemcpy(pixels, dynamic_cast<vvCudaImg*>(renderer->intImg)->getDImg(), screenRect->width * screenRect->height*4, cudaMemcpyDeviceToHost);

  vvImage2_5d* im2;
  switch(_depthPrecision)
  {
  case vvImage2_5d::VV_UCHAR:
    {
      im2 = new vvImage2_5d(screenRect->height, screenRect->width, (unsigned char*)pixels, vvImage2_5d::VV_UCHAR);
      im2->setDepthPrecision(_depthPrecision);
      im2->alloc_pd();
      uchar* depthUchar = im2->getpixeldepthUchar();
      cudaMemcpy(depthUchar, renderer->_depthUchar, screenRect->width * screenRect->height *sizeof(uchar), cudaMemcpyDeviceToHost);
      cudaFree(renderer->_depthUchar);
    }
    break;
  case vvImage2_5d::VV_USHORT:
    {
      im2 = new vvImage2_5d(screenRect->height, screenRect->width, (unsigned char*)pixels, vvImage2_5d::VV_USHORT);
      im2->setDepthPrecision(_depthPrecision);
      im2->alloc_pd();
      ushort* depthUshort = im2->getpixeldepthUshort();
      cudaMemcpy(depthUshort, renderer->_depthUshort, screenRect->width * screenRect->height *sizeof(ushort), cudaMemcpyDeviceToHost);
      cudaFree(renderer->_depthUshort);
    }
    break;
  case vvImage2_5d::VV_UINT:
    {
      im2 = new vvImage2_5d(screenRect->height, screenRect->width, (unsigned char*)pixels, vvImage2_5d::VV_UINT);
      im2->alloc_pd();
      uint* depthUint = im2->getpixeldepthUint();
      cudaMemcpy(depthUint, renderer->_depthUint, screenRect->width * screenRect->height *sizeof(uint), cudaMemcpyDeviceToHost);
      cudaFree(renderer->_depthUint);
    }
    break;
  }

  // Send image to client
  _socket->putImage2_5d(im2);

  delete[] pixels;
  delete im2;
}
