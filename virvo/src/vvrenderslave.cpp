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
#include "vvrenderslave.h"
#include "vvtexrend.h"

#ifdef HAVE_BONJOUR
#include "vvbonjour/vvbonjourregistrar.h"
#endif

vvRenderSlave::vvRenderSlave(const BufferPrecision compositingPrecision)
  : _offscreenBuffer(0), _socket(0), _compositingPrecision(compositingPrecision)
{

}

vvRenderSlave::~vvRenderSlave()
{
  delete _offscreenBuffer;
  delete _socket;
}

void vvRenderSlave::setCompositingPrecision(const BufferPrecision compositingPrecision)
{
  _compositingPrecision = compositingPrecision;
}

BufferPrecision vvRenderSlave::getCompositingPrecision() const
{
  return _compositingPrecision;
}

vvRenderSlave::ErrorType vvRenderSlave::initSocket(const int port, const vvSocket::SocketType st)
{
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

vvRenderSlave::ErrorType vvRenderSlave::initData(vvVolDesc*& vd) const
{
  bool loadVolumeFromFile;
  _socket->getBool(loadVolumeFromFile);

  if (loadVolumeFromFile)
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

vvRenderSlave::ErrorType vvRenderSlave::initBricks(std::vector<vvBrick*>& bricks) const
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

void vvRenderSlave::renderLoop(vvTexRend* renderer)
{
  renderer->setIsSlave(true);

  _offscreenBuffer = new vvOffscreenBuffer(1.0f, _compositingPrecision);
  _offscreenBuffer->initForRender();

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
  int currentFrame;

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
          renderer->_renderState._mipMode = mipMode;
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
          renderer->_renderState._quality = quality;
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
          _offscreenBuffer->resize(w, h);
        }
        break;
      case vvSocketIO::VV_INTERPOLATION:
        if ((_socket->getBool(interpolation)) == vvSocket::VV_OK)
        {
          renderer->setParameter(vvRenderer::VV_SLICEINT, interpolation ? 1.0f : 0.0f);
        }
        break;
      case vvSocketIO::VV_TOGGLE_BOUNDINGBOX:
        renderer->_renderState._boundaries = !renderer->_renderState._boundaries;
        break;
      case vvSocketIO::VV_TOGGLE_ROI:
        if ((_socket->getBool(roiEnabled)) == vvSocket::VV_OK)
        {
          renderer->setROIEnable(roiEnabled);
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
void vvRenderSlave::renderImage(vvMatrix& pr, vvMatrix& mv, vvTexRend* renderer)
{
  vvDebugMsg::msg(3, "vvRenderSlave::renderImage()");

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

  renderer->renderVolumeGL();

  glFlush();

  vvRect* screenRect = renderer->getProbedMask().getProjectedScreenRect();

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
}
