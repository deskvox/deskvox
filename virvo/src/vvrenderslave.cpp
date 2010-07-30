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

vvRenderSlave::vvRenderSlave()
  : _remoteRenderingBuffer(0), _remoteRenderingSocket(0)
{

}

vvRenderSlave::~vvRenderSlave()
{
  delete _remoteRenderingBuffer;
  delete _remoteRenderingSocket;
}

vvRenderSlave::ErrorType vvRenderSlave::initRemoteRenderingSocket(const int port, const vvSocket::SocketType st)
{
  _remoteRenderingSocket = new vvSocketIO(port, st);
  _remoteRenderingSocket->set_debuglevel(vvDebugMsg::getDebugLevel());

  const vvSocket::ErrorType err = _remoteRenderingSocket->init();
  _remoteRenderingSocket->no_nagle();

  if (err != vvSocket::VV_OK)
  {
    return VV_SOCKET_ERROR;
  }
  else
  {
    return VV_OK;
  }
}

vvRenderSlave::ErrorType vvRenderSlave::initRemoteRenderingData(vvVolDesc*& vd)
{
  bool loadVolumeFromFile;
  _remoteRenderingSocket->getBool(loadVolumeFromFile);

  if (loadVolumeFromFile)
  {
    char* fn = 0;
    _remoteRenderingSocket->getFileName(fn);
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
    switch (_remoteRenderingSocket->getVolume(vd))
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

vvRenderSlave::ErrorType vvRenderSlave::initRemoteRenderingBricks(std::vector<vvBrick*>& bricks)
{
  const vvSocket::ErrorType err = _remoteRenderingSocket->getBricks(bricks);
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

//----------------------------------------------------------------------------
/** Perform remote rendering, read back pixel data and send it over socket
    connections using a vvImage instance.
*/
void vvRenderSlave::remoteRenderingLoop(vvTexRend* renderer)
{
  vvMatrix pr;
  vvMatrix mv;
  while (1)
  {
    if ((_remoteRenderingSocket->getMatrix(&pr) == vvSocket::VV_OK)
       && (_remoteRenderingSocket->getMatrix(&mv) == vvSocket::VV_OK))
    {
      vvDebugMsg::msg(3, "vvView::renderRemotely()");

      if (_remoteRenderingBuffer == NULL)
      {
        _remoteRenderingBuffer = new vvOffscreenBuffer(1.0f, VV_BYTE);
        _remoteRenderingBuffer->initForRender();
      }

      _remoteRenderingBuffer->bindFramebuffer();
      _remoteRenderingBuffer->clearBuffer();

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

      const vvGLTools::Viewport viewport = vvGLTools::getViewport();
      uchar* pixels = new uchar[viewport[2] * viewport[3] * 4];
      glReadPixels(viewport[0], viewport[1], viewport[2], viewport[3], GL_RGBA, GL_UNSIGNED_BYTE, pixels);
      vvImage img(viewport[3], viewport[2], pixels);
      _remoteRenderingSocket->putImage(&img);
      delete[] pixels;
      _remoteRenderingBuffer->unbindFramebuffer();
    }
  }
}
