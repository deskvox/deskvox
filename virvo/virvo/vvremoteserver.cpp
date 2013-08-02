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
#include "vvrenderer.h"
#include "vvremoteserver.h"
#include "vvdebugmsg.h"
#include "vvsocketio.h"
#include "vvtcpsocket.h"

#include "private/vvgltools.h"

using std::cerr;
using std::endl;

vvRemoteServer::vvRemoteServer(vvSocket *socket)
  : _codetype(0)
{
  _socketio = new vvSocketIO(socket);
  vvDebugMsg::msg(1, "vvRemoteServer::vvRemoteServer()");
  initSocket();
}

vvRemoteServer::~vvRemoteServer()
{
  vvDebugMsg::msg(1, "vvRemoteServer::~vvRemoteServer()");

  delete _socketio;
}

vvRemoteServer::ErrorType vvRemoteServer::initSocket()
{
  vvDebugMsg::msg(1, "vvRemoteServer::initSocket()");

  _socketio->getSocket()->setParameter(vvSocket::VV_NO_NAGLE, true);

  return VV_OK;
}

bool vvRemoteServer::processEvent(virvo::RemoteEvent event, vvRenderer* renderer)
{
  vvDebugMsg::msg(3, "vvRemoteServer::processEvents()");

  vvVector3 position;
  vvVector3 viewDir;
  vvVector3 objDir;
  int currentFrame;
  vvTransFunc tf;

  switch (event)
  {
  case virvo::CameraMatrix:
    {
      vvMatrix pr;
      vvMatrix mv;
      if ((_socketio->getMatrix(&pr) == vvSocket::VV_OK)
         && (_socketio->getMatrix(&mv) == vvSocket::VV_OK))
      {
        renderImage(pr, mv, renderer);
      }
    }
    break;
  case virvo::CurrentFrame:
    if ((_socketio->getInt32(currentFrame)) == vvSocket::VV_OK)
    {
      renderer->setCurrentFrame(currentFrame);
    }
    break;
  case virvo::ObjectDirection:
    if ((_socketio->getVector3(objDir)) == vvSocket::VV_OK)
    {
      renderer->setObjectDirection(objDir);
    }
    break;
  case virvo::ViewingDirection:
    if ((_socketio->getVector3(viewDir)) == vvSocket::VV_OK)
    {
      renderer->setViewingDirection(viewDir);
    }
    break;
  case virvo::Position:
    if ((_socketio->getVector3(position)) == vvSocket::VV_OK)
    {
      renderer->setPosition(position);
    }
    break;
  case virvo::TransFunc:
    for (std::vector<vvTFWidget*>::const_iterator it = tf._widgets.begin();
         it != tf._widgets.end(); ++it)
    {
      delete *it;
    }
    tf._widgets.clear();
    if ((_socketio->getTransferFunction(tf)) == vvSocket::VV_OK)
    {
      renderer->getVolDesc()->tf = tf;
      renderer->updateTransferFunction();
    }
    break;
  case virvo::ParameterBool:
    {
      int32_t param;
      bool value = false;
      if (_socketio->getInt32(param) == vvSocket::VV_OK && _socketio->getBool(value) == vvSocket::VV_OK)
      {
        renderer->setParameter((vvRenderState::ParameterType)param, value);
      }
    }
    break;
  case virvo::ParameterInt32:
    {
      int32_t param;
      int value = 0;
      if (_socketio->getInt32(param) == vvSocket::VV_OK && _socketio->getInt32(value) == vvSocket::VV_OK)
      {
        switch(param)
        {
        case vvRenderer::VV_CODEC:
          _codetype = value;
          break;
        default:
          renderer->setParameter((vvRenderState::ParameterType)param, value);
          break;
        }
      }
    }
    break;
  case virvo::ParameterInt64:
    {
      int32_t param;
      int64_t value = 0;
      if (_socketio->getInt32(param) == vvSocket::VV_OK && _socketio->getInt64(value) == vvSocket::VV_OK)
      {
        renderer->setParameter((vvRenderState::ParameterType)param, value);
      }
    }
    break;
  case virvo::ParameterUint32:
    {
      int32_t param;
      uint32_t value = 0;
      if (_socketio->getInt32(param) == vvSocket::VV_OK && _socketio->getUint32(value) == vvSocket::VV_OK)
      {
        renderer->setParameter((vvRenderState::ParameterType)param, value);
      }
    }
    break;
  case virvo::ParameterUint64:
    {
      int32_t param;
      uint64_t value = 0;
      if (_socketio->getInt32(param) == vvSocket::VV_OK && _socketio->getUint64(value) == vvSocket::VV_OK)
      {
        renderer->setParameter((vvRenderState::ParameterType)param, value);
      }
    }
    break;
  case virvo::ParameterFloat:
    {
      int32_t param;
      float value = 0.f;
      if (_socketio->getInt32(param) == vvSocket::VV_OK && _socketio->getFloat(value) == vvSocket::VV_OK)
      {
        renderer->setParameter((vvRenderState::ParameterType)param, value);
      }
    }
    break;
  case virvo::ParameterVec3:
    {
      int32_t param;
      vvVector3 value;
      if (_socketio->getInt32(param) == vvSocket::VV_OK && _socketio->getVector3(value) == vvSocket::VV_OK)
      {
        renderer->setParameter((vvRenderState::ParameterType)param, value);
      }
    }
    break;
  case virvo::ParameterVec4:
    {
      int32_t param;
      vvVector4 value;
      if (_socketio->getInt32(param) == vvSocket::VV_OK && _socketio->getVector4(value) == vvSocket::VV_OK)
      {
        renderer->setParameter((vvRenderState::ParameterType)param, value);
      }
    }
    break;
  case virvo::ParameterSize3:
    {
      int32_t param;
      vvBaseVector3<uint64_t> value;
      if (_socketio->getInt32(param) == vvSocket::VV_OK && _socketio->getUint64(value[0]) == vvSocket::VV_OK
       && _socketio->getUint64(value[1]) == vvSocket::VV_OK && _socketio->getUint64(value[2]) == vvSocket::VV_OK)
      {
        // little endian...
        assert((value[0] & 0x0000FFFF) == value[0] && (value[1] & 0x0000FFFF) == value[1] && (value[2] & 0x0000FFFF) == value[2]);
        vvsize3 svalue(static_cast<size_t>(value[0]), static_cast<size_t>(value[1]), static_cast<size_t>(value[2]));
        renderer->setParameter((vvRenderState::ParameterType)param, svalue);
      }
    }
    break;
  case virvo::ParameterColor:
    {
      int32_t param;
      vvColor value;
      if (_socketio->getInt32(param) == vvSocket::VV_OK && _socketio->getColor(value) == vvSocket::VV_OK)
      {
        renderer->setParameter((vvRenderState::ParameterType)param, value);
      }
    }
    break;
  case virvo::ParameterAABBI:
    {
      int32_t param;
      vvAABBi value = vvAABBi(vvVector3i(), vvVector3i());
      if (_socketio->getInt32(param) == vvSocket::VV_OK && _socketio->getAABBi(value) == vvSocket::VV_OK)
      {
        renderer->setParameter((vvRenderState::ParameterType)param, value);
      }
    }
    break;
  default:
    vvDebugMsg::msg(0, "vvRemoteServer::processEvent(): event not implemented: ", (int)event);
    return true;
  }

  return true;
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
