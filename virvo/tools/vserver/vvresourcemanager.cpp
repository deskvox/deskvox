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
#include <virvo/vvibrserver.h>
#include <virvo/vvimageserver.h>
#include <virvo/vvsocketio.h>
#include <virvo/vvremoteserver.h>
#include <virvo/vvsocketmap.h>
#include <virvo/vvtcpsocket.h>
#include <virvo/vvvirvo.h>
#include <virvo/vvvoldesc.h>

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

  std::vector<vvGpu::vvGpuInfo> getResourceGpuInfos(vvTcpSocket *serversock)
  {
    if(NULL != serversock)
    {
      std::vector<vvGpu::vvGpuInfo> ginfos;
      vvSocketIO sockIO = vvSocketIO(serversock);
      sockIO.putInt32(virvo::Disconnect);
      return ginfos;
    }
    else
    {
      vvDebugMsg::msg(2, "vvResourceManager::getResourceGpuInfos() invalid socket");
      return std::vector<vvGpu::vvGpuInfo>();
    }
  }
}

vvResourceManager::vvResourceManager()
  : vvServer(false)
{
  _simpleServer = NULL;

  pthread_mutexattr_t reqAttr;
  pthread_mutexattr_init(&reqAttr);
  pthread_mutexattr_settype(&reqAttr, PTHREAD_MUTEX_RECURSIVE);
  pthread_mutex_init(&_requestsMutex,     &reqAttr);
  pthread_cond_init (&_requestsCondition, NULL);
  pthread_mutex_init(&_resourcesMutex,    NULL);
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
  pthread_cond_destroy (&_requestsCondition);
  pthread_mutex_destroy(&_resourcesMutex);
}

void vvResourceManager::handleNextConnection(vvTcpSocket *sock)
{
  vvDebugMsg::msg(3, "vvResourceManager::handleNextConnection()");

  vvDebugMsg::msg(3, "vvResourceManager::handleNextConnection() server mode: ", (_sm == vvServer::SERVER) ? "SimpleServer" : "ResourceManager");

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

bool vvResourceManager::handleEvent(ThreadData *tData, virvo::RemoteEvent event, const vvSocketIO& io)
{
  switch (event)
  {
    case virvo::Statistics:
      {
        /*
        float free = 0.0f;
        float total = 0.0f;
        for(std::vector<vvResource*>::iterator res = tData->instance->_resources.begin(); res != args->instance->_resources.end(); res++)
        {
          for(std::vector<vvGpu::vvGpuInfo>::iterator ginfo = (*res)->ginfos.begin(); ginfo != (*res)->ginfos.end(); ginfo++)
          {
            free += (*ginfo).freeMem;
            total += (*ginfo).totalMem;
          }
        }
        sockio.putFloat(free/total);
        sockio.putInt32(rm->_resources.size());*/
      }
      break;
    case virvo::GpuInfo:
      // TODO: implement this case for ResourceManager too if reasonable
      return false;
    case virvo::Disconnect:
      vvServer::handleEvent(tData, event, io);
      return false;
    default:
      return vvServer::handleEvent(tData, event, io);
  }
}

bool vvResourceManager::createRemoteServer(ThreadData* tData, vvTcpSocket* sock)
{
  vvDebugMsg::msg(3, "vvResourceManager::createRemoteServer() Enter");

  vvResourceManager *rm = tData->instance;

  // Ensure resources are allocated already
  if(tData->opt.empty())
  {
    pthread_mutex_lock(&rm->_requestsMutex);
    if(!tData->request)
    {
      // init with default values
      // TODO: make those values adjustable to fit estimated job
      tData->request = new vvRequest;
      tData->request->niceness = 0;
      tData->request->nodes.push_back(1);
      tData->request->type = vvRenderer::REMOTE_IMAGE;
    }

    // insert request if not already in list
    if(std::find(rm->_requests.begin(), rm->_requests.end(), tData->request) == rm->_requests.end())
    {
      rm->_requests.push_back(tData->request);
      rm->pairNextJobs();
    }

    if(tData->request->resources.size() == 0)
    {
      // wait until the request changed
      while(int i = pthread_cond_wait(&rm->_requestsCondition, &rm->_requestsMutex))
      {
        if(tData->request->resources.size() > 0)
        {
          // resources allocated
          break;
        }
      }
    }
    else //debug only
    {
      vvDebugMsg::msg(3, "vvResourceManager::createRemoteServer() no condition wait necessary");
    }
    std::vector<vvRequest*>::iterator rit = std::find(rm->_requests.begin(), rm->_requests.end(), tData->request);
    rm->_requests.erase(rit);

    pthread_mutex_unlock(&rm->_requestsMutex);

    std::stringstream sockstr;
    for(std::vector<vvResource*>::iterator res = tData->request->resources.begin();
        res != tData->request->resources.end(); res++)
    {
      std::string host = (*res)->hostname;
      ushort port = (*res)->port;
      vvTcpSocket *serversock = new vvTcpSocket();
      if(serversock->connectToHost(host, port) != vvSocket::VV_OK )
      {
        vvDebugMsg::msg(0, "vvResourceManager::createRemoteServer() fatal error: could not connect to vserver resource");
        delete serversock;
        // TODO: Delete sockets on socketmap too
        return false;
      }

      int s = vvSocketMap::add(serversock);
      if(sockstr.str() != "") sockstr << ",";
      sockstr << s;
    }
    tData->opt["sockets"] = sockstr.str();
  }

  switch(tData->remoteServerType)
  {
    case vvRenderer::REMOTE_IMAGE:
      tData->server = new vvImageServer(sock);
      break;
    case vvRenderer::REMOTE_IBR:
      tData->server = new vvIbrServer(sock);
      break;
    default:
      cerr << "Unknown remote rendering type " << tData->remoteServerType << std::endl;
      break;
  }

  vvGLTools::enableGLErrorBacktrace();

  if(tData->vd != NULL)
  {
    // Set default color scheme if no TF present:
    if(tData->vd->tf.isEmpty())
    {
      tData->vd->tf.setDefaultAlpha(0, 0.0, 1.0);
      tData->vd->tf.setDefaultColors((tData->vd->chan==1) ? 0 : 2, 0.0, 1.0);
    }

    vvRenderState rs;

    std::string renderertype;
    renderertype = (tData->remoteServerType == vvRenderer::REMOTE_IBR) ? "ibr" : "image";

    tData->renderer = vvRendererFactory::create(tData->vd,
      rs,
      renderertype.c_str(),
      tData->opt);

    tData->renderer->setParameter(vvRenderer::VV_USE_IBR, tData->remoteServerType == vvRenderer::REMOTE_IBR);
    return true;
  }
  else
  {
    vvDebugMsg::msg(0, "No volume loaded");
    return false;
  }
}

void * vvResourceManager::handleClientThread(void *param)
{
  vvDebugMsg::msg(3, "vvResourceManager::handleClientThread() Enter");

  ThreadArgs *args = reinterpret_cast<ThreadArgs*>(param);

  vvSocketIO sockio = vvSocketIO(args->sock);
  vvResourceManager *rm = args->instance;
  ThreadData *tData = new ThreadData();
  tData->instance = args->instance;

  virvo::RemoteEvent event;
  while(sockio.getEvent(event) == vvSocket::VV_OK)
  {
    if (!rm->handleEvent(tData, event, sockio))
    {
      break;
    }
  }
  delete args->sock;
  delete args;
  delete tData;

  pthread_exit(NULL);
#ifdef _WIN32
  return NULL;
#endif
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
          delete serversock;
          res->bonjourEntry = *entry;
          res->hostname     = resolver._hostname;
          res->port         = resolver._port;
          vvDebugMsg::msg(3, "vvResourceManager::updateResources() add resource into list with gpus: ", int(res->ginfos.size()));

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

  rm->pairNextJobs();
  }
  else
  {
  vvDebugMsg::msg(0, "vvResourceManager::updateResources() resource live-updating not available");
  }
}

void vvResourceManager::pairNextJobs()
{
  vvDebugMsg::msg(0, "vvResourceManager::pairNextJobs() Enter");

  pthread_mutex_lock(&_resourcesMutex);
  pthread_mutex_lock(&_requestsMutex);

  // sort for correct priority
  std::sort(_requests.begin(), _requests.end());
  if(_requests.size() > 0 && _resources.size() > 0)
  {
    // Update Gpu-Status of all Resources
    for(std::vector<vvResource*>::iterator res = _resources.begin(); res != _resources.end(); res++)
    {
      vvTcpSocket *sock = new vvTcpSocket();
      if(sock->connectToHost((*res)->hostname, (*res)->port) == vvSocket::VV_OK)
      {
        (*res)->ginfos = getResourceGpuInfos(sock);
      }
      else
      {
        vvDebugMsg::msg(0, "vvResourceManager::pairNextJobs() connecting to vserver failed ", (*res)->hostname.c_str());
      }
      delete sock;
    }

    for(std::vector<vvRequest*>::iterator req = _requests.begin(); req != _requests.end(); req++)
    {
      vvRequest* request = *req;
      if(getFreeResourceCount() >= request->nodes.size())
      {
        request->resources = getFreeResources(request->nodes.size());
        if(request->resources.size() != request->nodes.size())
        {
          vvDebugMsg::msg(0, "vvResourceManager::pairNextJobs() unexpected error: Job's resource count mismatch");
        }
      }
      else
      {
        break;
      }
    }
  }
  pthread_mutex_unlock(&_requestsMutex);
  pthread_mutex_unlock(&_resourcesMutex);

  if(pthread_cond_broadcast(&_requestsCondition) != 0)
  {
    vvDebugMsg::msg(0, "vvResourceManager::pairNextJobs() could not unblock request condition");
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
      vvDebugMsg::msg(3, "vvResourceManager::serverLoop() browsing bonjour");
    }
  }

  return vvServer::serverLoop();
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

    if(true) //QUICKFIX ONLY !!!  freeMemory > 0)
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

    if(true) // QUICKFIX ONLY !!!freeMemory > 0)
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
