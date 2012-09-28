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

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "vvresourcemanager.h"
#include "vvsimpleserver.h"

#include <vvcommon.h>

#include <virvo/vvdebugmsg.h>
#include <virvo/vvrequestmanagement.h>
#include <virvo/vvremoteserver.h>
#include <virvo/vvsocketio.h>
#include <virvo/vvvoldesc.h>
#include <virvo/vvtcpsocket.h>

#include <iostream>
#include <pthread.h>

vvSimpleServer::vvSimpleServer(bool useBonjour)
  : vvServer()
  , _useBonjour(useBonjour)
{
  if(_useBonjour)
  {
    registerToBonjour();
  }
}

vvSimpleServer::~vvSimpleServer()
{
  if(_useBonjour) unregisterFromBonjour();
}

void vvSimpleServer::handleNextConnection(vvTcpSocket *sock)
{
  vvDebugMsg::msg(3, "vvSimpleServer::handleNextConnection()");

  vvThreadArgs *args = new vvThreadArgs;
  args->_instance = this;
  args->_sock = sock;

  pthread_t pthread;
  pthread_create(&pthread,  NULL, handleClientThread, args);
  pthread_detach(pthread);
}

void * vvSimpleServer::handleClientThread(void *param)
{
  vvDebugMsg::msg(3, "vvSimpleServer::handleClientThread()");

  vvThreadArgs *args = reinterpret_cast<vvThreadArgs*>(param);

  vvTcpSocket *sock = args->_sock;

  vvSocketIO sockio(sock);

  bool goOn = true;
  int event;
  while(sockio.getInt32(event) == vvSocket::VV_OK && goOn)
  {
    switch(event)
    {
    case virvo::GpuInfo:
      {
        std::vector<vvGpu*> gpus = vvGpu::list();
        std::vector<vvGpu::vvGpuInfo> ginfos;
        for(std::vector<vvGpu*>::iterator gpu = gpus.begin(); gpu != gpus.end(); gpu++)
        {
          ginfos.push_back(vvGpu::getInfo(*gpu));
        }
        sockio.putGpuInfos(ginfos);
      }
      break;
    case virvo::Render:
      {
        vvRemoteServerRes res = args->_instance->createRemoteServer(sock);

        if(res.renderer && res.server && res.vd)
        {
          while(true)
          {
            if(!res.server->processEvents(res.renderer))
            {
              delete res.renderer;
              res.renderer = NULL;
              break;
            }
          }
        }

        if(res.server)
        {
          res.server->destroyRenderContext();
        }

        // Frames vector with bricks is deleted along with the renderer.
        // Don't free them here.
        // see setRenderer().

        delete res.server;
        delete res.vd;
      }
      // fall through...
    case virvo::Exit:
      goOn = false;
      break;
    default:
      vvDebugMsg::msg(2, "vvSimpleServer::handleClientThread() unknown event!");
      break;
    }
  }

  sock->disconnectFromHost();
  delete sock;
  delete args;

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

//===================================================================
// End of File
//===================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
