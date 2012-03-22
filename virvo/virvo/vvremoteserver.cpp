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
#include "vvrendercontext.h"
#include "vvrenderer.h"
#include "vvremoteserver.h"
#include "vvdebugmsg.h"
#include "vvsocketio.h"
#include "vvtcpsocket.h"
#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

using std::cerr;
using std::endl;

vvRemoteServer::vvRemoteServer(vvSocketIO *socket)
  : _socket(socket), _renderContext(NULL), _loadVolumeFromFile(false), _codetype(0)
{
  vvDebugMsg::msg(1, "vvRemoteServer::vvRemoteServer()");
  initSocket();
}

vvRemoteServer::~vvRemoteServer()
{
  vvDebugMsg::msg(1, "vvRemoteServer::~vvRemoteServer()");

  delete _socket;
  delete _renderContext;
}

bool vvRemoteServer::getLoadVolumeFromFile() const
{
  vvDebugMsg::msg(1, "vvRemoteServer::getLoadVolumeFromFile()");

  return _loadVolumeFromFile;
}

vvRemoteServer::ErrorType vvRemoteServer::initSocket()
{
  vvDebugMsg::msg(1, "vvRemoteServer::initSocket()");

  _socket->getSocket()->setParameter(vvSocket::VV_NO_NAGLE, true);

  return VV_OK;
}

vvRemoteServer::ErrorType vvRemoteServer::initData(vvVolDesc*& vd)
{
  vvDebugMsg::msg(1, "vvRemoteServer::initData()");

  _socket->getBool(_loadVolumeFromFile);

  if (_loadVolumeFromFile)
  {
    char* fn = 0;
    _socket->getFileName(fn);
    cerr << "Load volume from file: " << fn << endl;
    vd = new vvVolDesc(fn);
    delete[] fn;
    fn = NULL;

    vvFileIO fio;
    if (fio.loadVolumeData(vd) != vvFileIO::OK)
    {
      cerr << "Error loading volume file" << endl;
      return VV_FILEIO_ERROR;
    }
    else
    {
      vd->printInfoLine();
    }
    // Set default color scheme if no TF present:
    if (vd->tf.isEmpty())
    {
      vd->tf.setDefaultAlpha(0, 0.0, 1.0);
      vd->tf.setDefaultColors((vd->chan==1) ? 0 : 2, 0.0, 1.0);
    }
    _socket->putVolumeAttributes(vd);
    _socket->putTransferFunction(vd->tf);
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

vvRemoteServer::ErrorType vvRemoteServer::initRenderContext()
{
  delete _renderContext;
  vvRenderContext::ContextOptions co;
  co.type = vvRenderContext::VV_WINDOW;
  co.displayName = "0";
  _renderContext = new vvRenderContext(&co);
  if (_renderContext->makeCurrent())
  {
    return vvRemoteServer::VV_OK;
  }
  else
  {
    return vvRemoteServer::VV_RENDERCONTEXT_ERROR;
  }
}

bool vvRemoteServer::processEvents(vvRenderer* renderer)
{
  vvDebugMsg::msg(3, "vvRemoteServer::processEvents()");

  vvSocketIO::CommReason commReason;
  vvMatrix pr;
  vvMatrix mv;
  vvVector3 position;
  vvVector3 viewDir;
  vvVector3 objDir;
  int w;
  int h;
  int currentFrame;
  vvTransFunc tf;

  {
    vvGLTools::printGLError("begin vvRemoteServer::renderLoop()");

    vvSocket::ErrorType err = _socket->getCommReason(commReason);
    if (err == vvSocket::VV_OK)
    {
      switch (commReason)
      {
      case vvSocketIO::VV_EXIT:
        return false;
      case vvSocketIO::VV_MATRIX:
        if ((_socket->getMatrix(&pr) == vvSocket::VV_OK)
           && (_socket->getMatrix(&mv) == vvSocket::VV_OK))
        {
          renderImage(pr, mv, renderer);
        }
        break;
      case vvSocketIO::VV_CURRENT_FRAME:
        if ((_socket->getInt32(currentFrame)) == vvSocket::VV_OK)
        {
          renderer->setCurrentFrame(currentFrame);
        }
        break;
      case vvSocketIO::VV_OBJECT_DIRECTION:
        if ((_socket->getVector3(objDir)) == vvSocket::VV_OK)
        {
          renderer->setObjectDirection(&objDir);
        }
        break;
      case vvSocketIO::VV_VIEWING_DIRECTION:
        if ((_socket->getVector3(viewDir)) == vvSocket::VV_OK)
        {
          renderer->setViewingDirection(&viewDir);
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
          resize(w, h);
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
      case vvSocketIO::VV_PARAMETER_1:
        {
          int32_t param;
          float value = 0.f;
          if (_socket->getInt32(param) == vvSocket::VV_OK && _socket->getFloat(value) == vvSocket::VV_OK)
          {
            switch(param)
            {
            case vvRenderer::VV_CODEC:
              _codetype = (int)value;
              break;
            default:
              renderer->setParameter((vvRenderState::ParameterType)param, value);
              break;
            }
          }
        }
        break;
      case vvSocketIO::VV_PARAMETER_3:
        {
          int32_t param;
          vvVector3 value;
          if (_socket->getInt32(param) == vvSocket::VV_OK && _socket->getVector3(value) == vvSocket::VV_OK)
          {
            renderer->setParameterV3((vvRenderState::ParameterType)param, value);
          }
        }
        break;
      case vvSocketIO::VV_PARAMETER_4:
        {
          int32_t param;
          vvVector4 value;
          if (_socket->getInt32(param) == vvSocket::VV_OK && _socket->getVector4(value) == vvSocket::VV_OK)
          {
            renderer->setParameterV4((vvRenderState::ParameterType)param, value);
          }
        }
        break;
      default:
        vvDebugMsg::msg(0, "vvRemoteServer::mainLoop: comm reason not implemented: ", (int)commReason);
        return true;
      }
    }
    else if (err == vvSocket::VV_PEER_SHUTDOWN)
    {
      return false;
    }

    vvGLTools::printGLError("end vvRemoteServer::renderLoop()");
  }
  return true;
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
