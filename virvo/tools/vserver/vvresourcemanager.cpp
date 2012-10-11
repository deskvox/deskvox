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

#include "vvresourcemanager.h"
#include "vvserver.h"
#include "vvsimpleserver.h"

#include <pthread.h>
#include <sstream>

#include <virvo/vvbonjour/vvbonjourentry.h>
#include <virvo/vvbonjour/vvbonjourbrowser.h>
#include <virvo/vvbonjour/vvbonjourresolver.h>
#include <virvo/vvdebugmsg.h>
#include <virvo/vvsocketio.h>
#include <virvo/vvremoteserver.h>
#include <virvo/vvsocketmap.h>
#include <virvo/vvtcpsocket.h>
#include <virvo/vvvirvo.h>

namespace
{
  struct ThreadArgs
  {
    vvResourceManager *instance;
    vvTcpSocket       *sock;
  };

  void * localServerLoop(void *param)
  {
    vvSimpleServer *sserver = reinterpret_cast<vvSimpleServer*>(param);
    sserver->run(0, NULL);

    pthread_exit(NULL);
  #ifdef _WIN32
    return NULL;
  #endif
  }

  void * processJob(void *param)
  {
    vvDebugMsg::msg(3, "vvResourceManager::processJob() Enter");
/*
    if (virvo::hasFeature("bonjour"))
    {
    vvResourceManager::vvJob *job = reinterpret_cast<vvResourceManager::vvJob*>(param);
    vvTcpSocket *clientsock = job->request->sock;

    bool ready = true;
    std::vector<vvTcpSocket*> sockets;
    std::stringstream sockstr;
    for(std::vector<vvResourceManager::vvResource*>::iterator res = job->resources.begin();res!=job->resources.end();res++)
    {
      vvTcpSocket *serversock = NULL;
      vvBonjourResolver resolver;

      serversock = new vvTcpSocket();
      if(vvSocket::VV_OK == serversock->connectToHost((*res)->hostname, (*res)->port))
      {
        vvSocketIO sockIO = vvSocketIO(serversock);
        //sockIO.putInt32(virvo::Render);
        bool vserverRdy;
        sockIO.getBool(vserverRdy);

        sockets.push_back(serversock);
        int s = vvSocketMap::add(serversock);
        if(sockstr.str() != "") sockstr << ",";
        sockstr << s;
      }
      else
      {
        vvDebugMsg::msg(0, "vvResourceManager::processJob() Could not connect to vserver");
        delete serversock;
        ready = false;
      }
    }

    if(ready) // all resources are connected and ready to use
    {
      vvRendererFactory::Options opt;
      opt["sockets"] = sockstr.str();

      if(job->request->nodes.size() > 1) // brick rendering case
      {
        std::stringstream brickNum;
        brickNum << job->request->nodes.size();
        opt["bricks"] = brickNum.str();
        opt["brickrenderer"] = "image";

        std::stringstream displaystr;
        displaystr << ":0,:0"; // TODO: fix this hardcoded setting!
        opt["displays"] = displaystr.str();

  //      createRemoteServer(clientsock, "parbrick", opt);
      }
      else // regular remote rendering case
      {
  //      createRemoteServer(clientsock, "forwarding", opt);
      }
    }
    else
    {
      //TODO: Clean up sockets or implement fallback behaviour!
    }

  #if 0
    if (false)//if(res.renderer && res.server && res.vd)
    {
      while(true)
      {
        if(!_server->processEvents(_renderer))
        {
          delete _renderer;
          _renderer = NULL;
          break;
        }
      }
    }
  #endif

    delete job;

    if(clientsock)
    {
      clientsock->disconnectFromHost();
    }
    delete clientsock;
    }
  */
    pthread_exit(NULL);
  #ifdef _WIN32
    return NULL;
  #endif

  }

  std::vector<vvGpu::vvGpuInfo> getResourceGpuInfos(vvTcpSocket *serversock)
  {
    if(NULL != serversock)
    {
      std::vector<vvGpu::vvGpuInfo> ginfos;
      vvSocketIO sockIO = vvSocketIO(serversock);
      virvo::RemoteEvent event;
      sockIO.getEvent(event);
#if 0
      if(virvo::WaitEvents == event)
      {
        sockIO.putEvent(virvo::GpuInfo);
        sockIO.getGpuInfos(ginfos);
      }
#endif
      sockIO.putInt32(virvo::Disconnect);
      delete serversock;
      return ginfos;
    }
    else
    {
      vvDebugMsg::msg(2, "vvResourceManager::registerResource() Could not connect to resolved vserver");
      return std::vector<vvGpu::vvGpuInfo>();
    }
  }
}

vvResourceManager::vvResourceManager()
  : vvServer(false)
{
  _simpleServer = NULL;

  pthread_mutex_init(&_requestsMutex, NULL);
  pthread_mutex_init(&_resourcesMutex, NULL);
}

vvResourceManager::~vvResourceManager()
{
  for(std::vector<vvResource*>::iterator res = _resources.begin(); res < _resources.end(); res++)
  {
    delete *res;
  }
  for(std::vector<vvRequest*>::iterator req = _requests.begin(); req < _requests.end(); req++)
  {
    delete *req;
  }

  delete _simpleServer;

  pthread_mutex_destroy(&_requestsMutex);
  pthread_mutex_destroy(&_resourcesMutex);
}

void vvResourceManager::addJob(vvTcpSocket *sock)
{
  vvSocketIO sockio(sock);

  vvRequest *req = new vvRequest;
  vvSocket::ErrorType err = sockio.getRequest(*req);
  if(err != vvSocket::VV_OK)
  {
    cerr << "incoming connection socket error" << endl;
    return;
  }

  if(vvDebugMsg::getDebugLevel() >= 3)
  {
    std::stringstream errmsg;
    errmsg << "vvResourceManager::addJob() Incoming request has niceness: " << req->niceness << " and number of nodes: " << req->nodes.size();
    vvDebugMsg::msg(0, errmsg.str().c_str());
  }

  req->sock = sock;

  pthread_mutex_lock(&_requestsMutex);
  _requests.push_back(req);
  pthread_mutex_unlock(&_requestsMutex);

  while(pairNextJob()) {}
}

void * vvResourceManager::handleClientThread(void *param)
{
  vvDebugMsg::msg(3, "vvResourceManager::addJob() Enter");

  ThreadArgs *args = reinterpret_cast<ThreadArgs*>(param);

  vvSocketIO sockio = vvSocketIO(args->sock);

#if 0
  if (sockio.putEvent(virvo::WaitEvents) != vvSocket::VV_OK)
  {
    vvDebugMsg::msg(0, "Socket error");
    return NULL;
  }
#endif
  std::cerr << "bla1" << std::endl;
  bool goOn = true;
  virvo::RemoteEvent event;
  while(goOn)
  {
    if(sockio.getEvent(event) != vvSocket::VV_OK)
    {
      vvDebugMsg::msg(2, "vvResourceManager::addJob() error getting next event");
      break;
    }

    switch(event)
    {
    case virvo::Statistics:
      {
            std::cerr << "bla2" << std::endl;
        float free = 0.0f;
        float total = 0.0f;
        for(std::vector<vvResource*>::iterator res = args->instance->_resources.begin(); res != args->instance->_resources.end(); res++)
        {
          for(std::vector<vvGpu::vvGpuInfo>::iterator ginfo = (*res)->ginfos.begin(); ginfo != (*res)->ginfos.end(); ginfo++)
          {
            free += (*ginfo).freeMem;
            total += (*ginfo).totalMem;
          }
        }
        sockio.putFloat(free/total);
        sockio.putInt32(args->instance->_resources.size());
      }
      break;
    case virvo::GpuInfo:
      // TODO: implement this case for ResourceManager too if reasonable
    default:
      //vvServer::handleEvent();
      break;
    }
  }

  delete args;

  pthread_exit(NULL);
#ifdef _WIN32
  return NULL;
#endif
}

bool vvResourceManager::pairNextJob()
{
  vvDebugMsg::msg(3, "vvResourceManager::initNextJob() Enter");

  pthread_mutex_lock(&_resourcesMutex);
  pthread_mutex_lock(&_requestsMutex);

  // sort for correct priority
  std::sort(_requests.begin(), _requests.end());
  if(_requests.size() > 0 && _resources.size() > 0)
  {
    std::vector<vvRequest*>::iterator req = _requests.begin();
    while((*req)->nodes.size() > 0 && (*req)->resources.size() == 0)
    {
      req++;
    }
    vvRequest* request = *req;
    if(getFreeResourceCount() >= request->nodes.size())
    {
      request->resources = getFreeResources(request->nodes.size());

      if(request->resources.size() == request->nodes.size())
      {
        pthread_t threadID;
        pthread_create(&threadID, NULL, ::processJob, request);
        pthread_detach(threadID);

        pthread_mutex_unlock(&_requestsMutex);
        pthread_mutex_unlock(&_resourcesMutex);

        // Update Gpu-Status of used Resrources
        for(std::vector<vvResource*>::iterator usedRes = request->resources.begin(); usedRes != request->resources.end(); usedRes++)
        {
          vvTcpSocket *sock = new vvTcpSocket();
          sock->connectToHost((*usedRes)->hostname, (*usedRes)->port);
          (*usedRes)->ginfos = getResourceGpuInfos(sock);
          delete sock;
        }
        return true;
      }
      else
      {
        vvDebugMsg::msg(0, "vvResourceManager::initNextJob() unexpected error: Job's resource count mismatch");
      }
    }
  }

  pthread_mutex_unlock(&_requestsMutex);
  pthread_mutex_unlock(&_resourcesMutex);

  return false;
}

void vvResourceManager::updateResources(void * param)
{
  if (virvo::hasFeature("bonjour"))
  {
  vvDebugMsg::msg(0, "vvResourceManager::updateResources() Enter");

  vvResourceManager *rm = reinterpret_cast<vvResourceManager*>(param);

  std::vector<vvBonjourEntry> entries = rm->_browser->getBonjourEntries();

  pthread_mutex_lock(&rm->_resourcesMutex);

  for(std::vector<vvResource*>::iterator resource = rm->_resources.begin(); resource != rm->_resources.end(); resource++)
  {
    // mark non-local vservers for updating
    if((*resource)->local == false)
    {
      (*resource)->upToDate = false;
    }
    else
    {
      (*resource)->upToDate = true;
    }
  }

  if(!entries.empty())
  {
    for(std::vector<vvBonjourEntry>::iterator entry = entries.begin(); entry != entries.end(); entry++)
    {
      bool inList = false;
      for(std::vector<vvResource*>::iterator resource = rm->_resources.begin(); resource != rm->_resources.end(); resource++)
      {
        if((*resource)->bonjourEntry == *entry)
        {
          (*resource)->upToDate = true;
          inList = true;
          break;
        }
      }

      if(!inList)
      {
        vvResource *res = new vvResource();
        vvBonjourResolver resolver;
        if(vvBonjour::VV_OK  == resolver.resolveBonjourEntry(*entry))
        {
          vvTcpSocket *serversock = resolver.getBonjourSocket();
          res->ginfos       = getResourceGpuInfos(serversock);
          res->bonjourEntry = *entry;
          res->hostname     = resolver._hostname;
          res->port         = resolver._port;

          std::cerr << "add resource into list with gpus: " << res->ginfos.size() << std::endl;
          std::cerr << "free mem: " << res->ginfos[0].freeMem << std::endl;

          rm->_resources.push_back(res);
        }
        else
        {
          vvDebugMsg::msg(2, "vvResourceManager::updateResources() Could not resolve bonjour service. Resource skipped.");
        }
      }
    }
  }

  for(std::vector<vvResource*>::iterator resource = rm->_resources.begin(); resource != rm->_resources.end(); resource++)
  {
    if(!(*resource)->upToDate)
    {
      delete *resource;
      rm->_resources.erase(resource);
      resource--; // prevent double-iteration, because 'for' and 'erase()' both iterate
    }
  }

  pthread_mutex_unlock(&rm->_resourcesMutex);

  while(rm->pairNextJob()) {}; // process waiting requests
  }
  else
  {
  vvDebugMsg::msg(0, "vvResourceManager::updateResources() resource live-updating not available");
  }
}

bool vvResourceManager::serverLoop()
{
  switch(_sm)
  {
  case vvServer::SERVER:
    _simpleServer = new vvSimpleServer(_useBonjour);
    break;
  case vvServer::RM_WITH_SERVER:
    {
      vvResource *serverRes = new vvResource;
      _simpleServer = new vvSimpleServer(false); // RM and vserver-Bonjour at the same time is not possible
      ushort serverResPort = vvServer::DEFAULT_PORT+1;
      _simpleServer->setPort(serverResPort);
      serverRes->hostname = "localhost";
      serverRes->port = serverResPort;

      std::vector<vvGpu*> gpus = vvGpu::list();
      for(std::vector<vvGpu*>::iterator gpu = gpus.begin(); gpu != gpus.end();gpu++)
      {
        vvGpu::vvGpuInfo ginfo = vvGpu::getInfo(*gpu);
        serverRes->ginfos.push_back(ginfo);
      }
      _resources.push_back(serverRes);

      pthread_t threadID;
      pthread_create(&threadID, NULL, ::localServerLoop, _simpleServer);
      pthread_detach(threadID);
    }
    break;
  case vvServer::RM:
  default:
    // nothing to do for other cases here.
    break;
  }

  if (virvo::hasFeature("bonjour"))
  {
  if(_sm != vvServer::SERVER)
  {
    _browser = new vvBonjourBrowser(updateResources, this);
    _browser->browseForServiceType("_vserver._tcp", "", -1.0); // browse in continous mode
    std::cerr << "browsing bonjour..." << std::endl;
  }
  }

  return vvServer::serverLoop();
}

void vvResourceManager::handleNextConnection(vvTcpSocket *sock)
{
  vvDebugMsg::msg(3, "vvResourceManager::handleNextConnection()");

  std::cerr << "server mode: " << _sm << std::endl;

  switch(_sm)
  {
  case vvServer::SERVER:
    _simpleServer->handleNextConnection(sock);
    break;
  case vvServer::RM:
  case vvServer::RM_WITH_SERVER:
      {
        ::ThreadArgs *args = new ::ThreadArgs;
        args->sock = sock;
        args->instance = this;
        pthread_t threadID;
        pthread_create(&threadID, NULL, handleClientThread, (void*)args);
        pthread_detach(threadID);
      }
    break;
  default:
    // unknown case
    break;
  }
}

uint vvResourceManager::getFreeResourceCount() const
{
  uint count = 0;

  for(std::vector<vvResource*>::const_iterator freeRes = _resources.begin();
      freeRes != _resources.end(); freeRes++)
  {
    int freeMemory = 0;
    for(std::vector<vvGpu::vvGpuInfo>::iterator ginfo = (*freeRes)->ginfos.begin();
        ginfo < (*freeRes)->ginfos.end(); ginfo++)
    {
      freeMemory += (*ginfo).freeMem;
    }

    if(freeMemory > 0)
    {
      count++;
    }
    else
    {
      if (virvo::hasFeature("bonjour"))
      {
      std::ostringstream msg;
      msg << "vvResourceManager::getFreeResourceCount() Resource on "
          << (*freeRes)->bonjourEntry.getServiceName()
          << "is out of memory";
      vvDebugMsg::msg(3, msg.str().c_str());
      }
    }
  }

  return count;
}

std::vector<vvResource*> vvResourceManager::getFreeResources(uint amount) const
{
  vvDebugMsg::msg(3, "vvResourceManager::getFreeResources() Enter");

  std::vector<vvResource*> freeResources;

  if(amount > _resources.size())
    return freeResources;

  std::vector<vvResource*>::const_iterator freeRes = _resources.begin();
  while(freeRes != _resources.end() && freeResources.size() < amount)
  {
    int freeMemory = 0;
    for(std::vector<vvGpu::vvGpuInfo>::iterator ginfo = (*freeRes)->ginfos.begin();
        ginfo < (*freeRes)->ginfos.end(); ginfo++)
    {
      freeMemory += (*ginfo).freeMem;
    }

    if(freeMemory > 0)
    {
      freeResources.push_back(*freeRes);
      freeRes++;
    }
    else
    {
      freeRes++;
      if (virvo::hasFeature("bonjour"))
      {
      std::ostringstream msg;
      msg << "vvResourceManager::getFreeResourceCount() Resource on "
          << (*freeRes)->bonjourEntry.getServiceName()
          << "is out of memory";
      vvDebugMsg::msg(3, msg.str().c_str());
      }
    }
  }

  if(freeResources.size() != amount)
  {
    vvDebugMsg::msg(3, "vvResourceManager::getFreeResource() not enough free resource found");
    freeResources.clear();
  }
  return freeResources;
}

//===================================================================
// End of File
//===================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
