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

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "vvresourcemanager.h"
#include "vvsimpleserver.h"

#include <virvo/vvdebugmsg.h>
#include <virvo/vvrequestmanagement.h>
#include <virvo/vvremoteevents.h>
#include <virvo/vvremoteserver.h>
#include <virvo/vvsocketio.h>
#include <virvo/vvvoldesc.h>
#include <virvo/vvtcpsocket.h>

#include <iostream>
#include <pthread.h>

namespace
{
  struct ThreadArgs
  {
    vvTcpSocket *_sock;
  };
}

vvSimpleServer* vvSimpleServer::_instance = NULL;

vvSimpleServer::vvSimpleServer(bool useBonjour)
  : vvServer(useBonjour)
{
  _instance = this;
  if (_useBonjour)
  {
    registerToBonjour();
  }
}

vvSimpleServer::~vvSimpleServer()
{
  if(_useBonjour) unregisterFromBonjour();
}

bool vvSimpleServer::handleEvent(ThreadData *tData, virvo::RemoteEvent event, const vvSocketIO& io)
{
  switch (event)
  {
  case virvo::GpuInfo:
    {
      std::vector<vvGpu*> gpus = vvGpu::list();
      std::vector<vvGpu::vvGpuInfo> ginfos;
      for(std::vector<vvGpu*>::iterator gpu = gpus.begin(); gpu != gpus.end(); gpu++)
      {
        ginfos.push_back(vvGpu::getInfo(*gpu));
      }
      io.putGpuInfos(ginfos);
    }
    return true;
  case virvo::Disconnect:
    vvServer::handleEvent(tData, event, io);
    return false;
  default:
    return vvServer::handleEvent(tData, event, io);
  }
}

void vvSimpleServer::handleNextConnection(vvTcpSocket *sock)
{
  vvDebugMsg::msg(3, "vvSimpleServer::handleNextConnection()");

  ThreadArgs *args = new ThreadArgs;
  args->_sock = sock;

  pthread_t pthread;
  pthread_create(&pthread,  NULL, handleClientThread, args);
  pthread_detach(pthread);
}

void * vvSimpleServer::handleClientThread(void *param)
{
  vvDebugMsg::msg(3, "vvSimpleServer::handleClientThread()");

  ThreadData *tData = new ThreadData;
  ThreadArgs *args = reinterpret_cast<ThreadArgs*>(param);

  vvTcpSocket *sock = args->_sock;

  vvSocketIO io(sock);
  virvo::RemoteEvent event;
  while (io.getEvent(event) == vvSocket::VV_OK)
  {
    if (!_instance->handleEvent(tData, event, io))
    {
      break;
    }
  }
  _instance->handleEvent(tData, virvo::Disconnect, io);

  sock->disconnectFromHost();
  delete sock;
  delete args;
  delete tData;

  pthread_exit(NULL);
#ifdef _WIN32
  return NULL;
#endif
}

bool vvSimpleServer::registerToBonjour()
{
  vvDebugMsg::msg(3, "vvSimpleServer::registerToBonjour()");
  vvBonjourEntry entry = vvBonjourEntry("Virvo Server", "_vserver._tcp", "");
  return _registrar.registerService(entry, _port);
}

void vvSimpleServer::unregisterFromBonjour()
{
  vvDebugMsg::msg(3, "vvSimpleServer::unregisterFromBonjour()");
  _registrar.unregisterService();
}

bool vvSimpleServer::createRemoteServer(ThreadData* tData, vvTcpSocket* sock)
{
  vvDebugMsg::msg(3, "vvSimpleServer::createRemoteServer() Enter");

  tData->renderertype = tData->remoteServerType == vvRenderer::REMOTE_IMAGE
    ? "default"
    : "rayrendcuda";

  return vvServer::createRemoteServer(tData, sock);
}

//===================================================================
// End of File
//===================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
