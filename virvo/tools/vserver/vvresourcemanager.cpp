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

#include <vvcommon.h>
#ifdef HAVE_BONJOUR
#include <virvo/vvbonjour/vvbonjourentry.h>
#include <virvo/vvbonjour/vvbonjourbrowser.h>
#include <virvo/vvbonjour/vvbonjourresolver.h>
#endif
#include <virvo/vvdebugmsg.h>
#include <virvo/vvsocketio.h>
#include <virvo/vvremoteserver.h>
#include <virvo/vvsocketmap.h>
#include <virvo/vvtcpsocket.h>

vvResourceManager::vvResourceManager()
  : vvServer()
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

void vvResourceManager::addJob(vvTcpSocket* sock)
{
  vvDebugMsg::msg(3, "vvResourceManager::addJob() Enter");

  pthread_mutex_lock(&_requestsMutex);

  vvSocketIO sockio = vvSocketIO(sock);

  bool goOn = true;
  int event;
  while(goOn)
  {
    if(sockio.getInt32(event) != vvSocket::VV_OK)
    {
      vvDebugMsg::msg(2, "vvResourceManager::addJob() error getting next event");
      break;
    }

    switch(event)
    {
    case virvo::Render:
      {
        vvSocket::ErrorType err;
        sockio.putBool(false);

        vvRequest *req = new vvRequest;
        req->sock = sock;

        err = sockio.getRequest(*req);
        if(err != vvSocket::VV_OK)
        {
          cerr << "incoming connection socket error" << endl;
          goto abort;
        }

        if(vvDebugMsg::getDebugLevel() >= 3)
        {
          std::stringstream errmsg;
          errmsg << "Incoming request has niceness: " << req->niceness << " and number of nodes: " << req->nodes.size();
          vvDebugMsg::msg(0, errmsg.str().c_str());
        }

        _requests.push_back(req);
        goOn = false;
      }
      break;
    case virvo::Statistics:
      {
        float free = 0.0f;
        float total = 0.0f;
        for(std::vector<vvResource*>::iterator res = _resources.begin(); res != _resources.end(); res++)
        {
          for(std::vector<vvGpu::vvGpuInfo>::iterator ginfo = (*res)->ginfos.begin(); ginfo != (*res)->ginfos.end(); ginfo++)
          {
            free += (*ginfo).freeMem;
            total += (*ginfo).totalMem;
          }
        }
        sockio.putFloat(free/total);
        sockio.putInt32(_resources.size());
      }
      break;
    case virvo::GpuInfo:
      // TODO: implement this case for ResourceManager too if reasonable
    default:
      goto abort;
      break;
    }
  }

  abort:
  pthread_mutex_unlock(&_requestsMutex);
}

bool vvResourceManager::initNextJob()
{
  vvDebugMsg::msg(3, "vvResourceManager::initNextJob() Enter");

  pthread_mutex_lock(&_resourcesMutex);
  pthread_mutex_lock(&_requestsMutex);

  // sort for correct priority
  std::sort(_requests.begin(), _requests.end());
  vvJob *job = NULL;
  if(_requests.size() > 0 && _resources.size() > 0)
  {
    if(getFreeResourceCount() >= _requests.front()->nodes.size())
    {
      job = new vvJob;
      job->request = _requests.front();
      _requests.erase(_requests.begin());

      job->resources = getFreeResources(job->request->nodes.size());

      if(job->resources.size() != job->request->nodes.size())
      {
        vvDebugMsg::msg(1, "vvResourceManager::initNextJob() unexpected error: Job without enough Resources started");
        goto quitonerror;
      }

      // job ready to start?
      if(job->resources.size() == job->request->nodes.size())
      {
        pthread_t threadID;
        pthread_create(&threadID, NULL, processJob, job);
        pthread_detach(threadID);

        pthread_mutex_unlock(&_requestsMutex);
        pthread_mutex_unlock(&_resourcesMutex);

        // Update Gpu-Status of used Resrources
        for(std::vector<vvResource*>::iterator usedRes = job->resources.begin(); usedRes != job->resources.end(); usedRes++)
        {
#ifdef HAVE_BONJOUR
          (*usedRes)->ginfos = getResourceGpuInfos((*usedRes)->bonjourEntry);
#endif
        }
        return true;
      }
      else
      {
        vvDebugMsg::msg(1, "vvResourceManager::initNextJob() unexpected error: Job's resource count mismatch");
        goto quitonerror;
      }
    }
  }

  quitonerror:
  // put job back to queue
  if(job) _requests.push_back(job->request);
  delete job;

  pthread_mutex_unlock(&_requestsMutex);
  pthread_mutex_unlock(&_resourcesMutex);

  return false;
}

void vvResourceManager::updateResources(void * param)
{
#ifdef HAVE_BONJOUR
  vvDebugMsg::msg(0, "vvResourceManager::updateResources() Enter");

  vvResourceManager *rm = reinterpret_cast<vvResourceManager*>(param);

  std::vector<vvBonjourEntry> entries = rm->_browser->getBonjourEntries();

  pthread_mutex_lock(&rm->_resourcesMutex);

  for(std::vector<vvResource*>::iterator resource = rm->_resources.begin(); resource != rm->_resources.end(); resource++)
  {
    // mark non-local vservers for updating
    if((*resource)->server == NULL)
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
        res->ginfos = getResourceGpuInfos(*entry);
        std::cerr << "add resource into list with gpus: " << res->ginfos.size() << std::endl;
        std::cerr << "free mem: " << res->ginfos[0].freeMem << std::endl;
        res->bonjourEntry = *entry;

        rm->_resources.push_back(res);
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

  while(rm->initNextJob()) {}; // process all waiting jobs
#else
  vvDebugMsg::msg(0, "vvResourceManager::updateResources() resource live-updating not available");
  (void)param;
#endif
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
      serverRes->server = new vvSimpleServer(false); // RM and vserver-Bonjour at the same time is not possible
      serverRes->server->setPort(vvServer::DEFAULT_PORT+1);

      std::vector<vvGpu*> gpus = vvGpu::list();
      for(std::vector<vvGpu*>::iterator gpu = gpus.begin(); gpu != gpus.end();gpu++)
      {
        vvGpu::vvGpuInfo ginfo = vvGpu::getInfo(*gpu);
        serverRes->ginfos.push_back(ginfo);
      }
      _resources.push_back(serverRes);

      pthread_t threadID;
      pthread_create(&threadID, NULL, localServerLoop, serverRes->server);
      pthread_detach(threadID);
    }
    break;
  case vvServer::RM:
  default:
    // nothing to do for other cases here.
    break;
  }

#ifdef HAVE_BONJOUR
  if(_sm != vvServer::SERVER)
  {
    _browser = new vvBonjourBrowser(updateResources, this);
    _browser->browseForServiceType("_vserver._tcp", "", -1.0); // browse in continous mode
    std::cerr << "browsing bonjour..." << std::endl;
  }
#endif

  return vvServer::serverLoop();
}

void * vvResourceManager::localServerLoop(void *param)
{
  vvSimpleServer *sserver = reinterpret_cast<vvSimpleServer*>(param);
  sserver->run(0, NULL);

  pthread_exit(NULL);
#ifdef _WIN32
  return NULL;
#endif
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
      addJob(sock);
      while(initNextJob()) {}
    break;
  default:
    // unknown case
    break;
  }
}

void * vvResourceManager::processJob(void * param)
{
  vvDebugMsg::msg(3, "vvResourceManager::processJob() Enter");

#ifdef HAVE_BONJOUR
  vvJob *job = reinterpret_cast<vvJob*>(param);
  vvTcpSocket *clientsock = job->request->sock;

  bool ready = true;
  std::vector<vvTcpSocket*> sockets;
  std::stringstream sockstr;
  for(std::vector<vvResource*>::iterator res = job->resources.begin();res!=job->resources.end();res++)
  {
    vvTcpSocket *serversock = NULL;
    vvBonjourResolver resolver;

    // special case for local vserver: no bonjour resolving necessary
    if((*res)->server != NULL)
    {
      serversock = new vvTcpSocket();
      serversock->connectToHost("localhost", DEFAULT_PORT+1); // TODO: Fix this default-port behaviour
    }
    else if(vvBonjour::VV_OK  == resolver.resolveBonjourEntry((*res)->bonjourEntry))
    {
      serversock = resolver.getBonjourSocket();
      if(NULL == serversock)
      {
        vvDebugMsg::msg(2, "vvResourceManager::processJob() Could not connect to resolved vserver");
        ready = false;
      }
    }
    else
    {
      vvDebugMsg::msg(2, "vvResourceManager::processJob() Could not resolve bonjour service");
      ready = false;
    }

    if(NULL != serversock)
    {
      vvSocketIO sockIO = vvSocketIO(serversock);
      sockIO.putInt32(virvo::Render);
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
      ready = false;
    }
  }

  vvRemoteServerRes res;
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

      res = createRemoteServer(clientsock, "parbrick", opt);
    }
    else // regular remote rendering case
    {
      res = createRemoteServer(clientsock, "forwarding", opt);
    }
  }

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

  delete job;

  if(clientsock)
  {
    clientsock->disconnectFromHost();
  }
  delete clientsock;

#else
  (void)param;
#endif

  pthread_exit(NULL);
#ifdef _WIN32
  return NULL;
#endif
}

uint vvResourceManager::getFreeResourceCount()
{
  uint count = 0;

  for(std::vector<vvResource*>::iterator freeRes = _resources.begin();
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
#ifdef HAVE_BONJOUR
      std::ostringstream msg;
      msg << "vvResourceManager::getFreeResourceCount() Resource on "
          << (*freeRes)->bonjourEntry.getServiceName()
          << "is out of memory";
      vvDebugMsg::msg(3, msg.str().c_str());
#endif
    }
  }

  return count;
}

std::vector<vvResourceManager::vvResource*> vvResourceManager::getFreeResources(uint amount)
{
  vvDebugMsg::msg(3, "vvResourceManager::getFreeResources() Enter");

  std::vector<vvResourceManager::vvResource*> freeResources;

  if(amount > _resources.size())
    return freeResources;

  std::vector<vvResource*>::iterator freeRes = _resources.begin();
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
#ifdef HAVE_BONJOUR
      std::ostringstream msg;
      msg << "vvResourceManager::getFreeResourceCount() Resource on "
          << (*freeRes)->bonjourEntry.getServiceName()
          << "is out of memory";
      vvDebugMsg::msg(3, msg.str().c_str());
#endif
    }
  }

  if(freeResources.size() != amount)
  {
    vvDebugMsg::msg(3, "vvResourceManager::getFreeResource() not enough free resource found");
    freeResources.clear();
  }
  return freeResources;
}

#ifdef HAVE_BONJOUR
std::vector<vvGpu::vvGpuInfo> vvResourceManager::getResourceGpuInfos(const vvBonjourEntry entry)
{
  vvTcpSocket *serversock = NULL;

  vvBonjourResolver resolver;
  if(vvBonjour::VV_OK  == resolver.resolveBonjourEntry(entry))
  {
    serversock = resolver.getBonjourSocket();
    if(NULL != serversock)
    {
      vvSocketIO sockIO = vvSocketIO(serversock);
      sockIO.putInt32(virvo::GpuInfo);
      std::vector<vvGpu::vvGpuInfo> ginfos;
      sockIO.getGpuInfos(ginfos);
      sockIO.putInt32(virvo::Exit);
      delete serversock;
      return ginfos;
    }
    else
    {
      vvDebugMsg::msg(2, "vvResourceManager::registerResource() Could not connect to resolved vserver");
      return std::vector<vvGpu::vvGpuInfo>();
    }
  }
  else
  {
    vvDebugMsg::msg(2, "vvResourceManager::registerResource() Could not resolve bonjour service");
    return std::vector<vvGpu::vvGpuInfo>();
  }
}
#endif

//===================================================================
// End of File
//===================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
