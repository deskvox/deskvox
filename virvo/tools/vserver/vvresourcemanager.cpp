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

#include <pthread.h>

#ifdef HAVE_BONJOUR
#include <virvo/vvbonjour/vvbonjourentry.h>
#include <virvo/vvbonjour/vvbonjourbrowser.h>
#include <virvo/vvbonjour/vvbonjourresolver.h>
#endif
#include <virvo/vvdebugmsg.h>
#include <virvo/vvsocketmonitor.h>
#include <virvo/vvsocketio.h>
#include <virvo/vvtcpsocket.h>
#include <virvo/vvibrserver.h>
#include <virvo/vvimageserver.h>
#include <virvo/vvremoteserver.h>
#include <virvo/vvvoldesc.h>
#include <virvo/vvrendererfactory.h>
#include <virvo/vvsocketmap.h>

#include <sstream>

using namespace std;

vvResourceManager* vvResourceManager::inst = NULL;

vvResourceManager::vvResourceManager(vvServer *vserver)
{
  vvResourceManager::inst = this;

  if(vserver)
  {
    vvResource *serverRes = new vvResource;
    serverRes->server = vserver;

    // TODO: set appropiate values here
    vvGPU gpu;
    gpu.freeMem = gpu.totalMem = 1000;
    serverRes->GPUs.push_back(gpu);

    _resources.push_back(serverRes);
  }

  pthread_mutex_init(&_jobsMutex, NULL);
  pthread_mutex_init(&_resourcesMutex, NULL);

#ifdef HAVE_BONJOUR
  _browser = new vvBonjourBrowser(updateResources, this);
  _browser->browseForServiceType("_vserver._tcp", "", -1.0); // browse in continous mode
#endif
}

vvResourceManager::~vvResourceManager()
{
  pthread_mutex_destroy(&_jobsMutex);
  pthread_mutex_destroy(&_resourcesMutex);
}

void vvResourceManager::addJob(vvTcpSocket* sock)
{
  vvDebugMsg::msg(3, "vvResourceManager::addJob() Enter");

  pthread_mutex_lock(&_jobsMutex);

  vvRequest *req = new vvRequest;
  req->sock = sock;

  vvSocketIO sockio = vvSocketIO(sock);
  bool tellinfo;
  sockio.getBool(tellinfo); // need vvGpu info?
  if(tellinfo)
  {
    // TODO: implement this case for ResourceManager too if reasonable
    goto abort;
  }

  vvSocket::ErrorType err;
  sockio.putBool(false);
  err = sockio.getInt32(req->priority);
  if(err != vvSocket::VV_OK)
  {
    cerr << "incoming connection socket error" << endl;
    goto abort;
  }
  err = sockio.getInt32(req->requirements);
  if(err != vvSocket::VV_OK)
  {
    cerr << "incoming connection socket error" << endl;
    goto abort;
  }

  if(vvDebugMsg::getDebugLevel() >= 3)
  {
    std::stringstream errmsg;
    errmsg << "Incoming request has priority: " << req->priority << " and requirements: " << req->requirements;
    vvDebugMsg::msg(0, errmsg.str().c_str());
  }

  _requests.push_back(req);

  abort:
  pthread_mutex_unlock(&_jobsMutex);
}

bool vvResourceManager::initNextJob()
{
  vvDebugMsg::msg(3, "vvResourceManager::initNextJob() Enter");

  pthread_mutex_lock(&_resourcesMutex);
  pthread_mutex_lock(&_jobsMutex);

  // sort for correct priority
  std::sort(_requests.begin(), _requests.end());

  vvJob *job = NULL;

  if(_requests.size() > 0 && _resources.size() > 0)
  {
    if(getFreeResourceCount() >= _requests.front()->requirements)
    {
      job = new vvJob;
      job->request = _requests.front();
      _requests.erase(_requests.begin());

      for(int i=0; i<job->request->requirements; i++)
      {
        vvResource *freeRes = getFreeResource();
        if(freeRes != NULL)
        {
          freeRes->GPUs[0].freeMem = 0;
          job->resources.push_back(freeRes);
        }
        else
        {
          vvDebugMsg::msg(1, "vvResourceManager::initNextJob() unexpected error: Job without enough Resources started");
          goto quitonerror;
        }
      }

      // job ready to start?
      if(job->resources.size() == job->request->requirements)
      {
        pthread_t threadID;
        pthread_create(&threadID, NULL, processJob, job);
        pthread_detach(threadID);

        pthread_mutex_unlock(&_jobsMutex);
        pthread_mutex_unlock(&_resourcesMutex);

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

  pthread_mutex_unlock(&_jobsMutex);
  pthread_mutex_unlock(&_resourcesMutex);

  return false;
}

void vvResourceManager::updateResources(void * param)
{
#ifdef HAVE_BONJOUR
  vvDebugMsg::msg(3, "vvResourceManager::updateResources() Enter");

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
        if((*resource)->_bonjourEntry == *entry)
        {
          (*resource)->upToDate = true;
          inList = true;
          break;
        }
      }
      if(!inList)
      {
        vvResource *res = new vvResource();
        res->_bonjourEntry = *entry;
        // TODO: set appropiate values here
        vvGPU gpu;
        gpu.freeMem = gpu.totalMem = 1000;
        res->GPUs.push_back(gpu);
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

uint vvResourceManager::getFreeResourceCount()
{
  uint count = 0;

  for(std::vector<vvResource*>::iterator freeRes = _resources.begin();
      freeRes != _resources.end(); freeRes++)
  {
    if((*freeRes)->GPUs[0].freeMem > 0)
      count++;
  }

  return count;
}

vvResourceManager::vvResource* vvResourceManager::getFreeResource()
{
  std::vector<vvResource*>::iterator freeRes = _resources.begin();
  while(freeRes != _resources.end())
  {
    if((*freeRes)->GPUs[0].freeMem == 0)
      freeRes++;
    else
      return (*freeRes);
  }

  return NULL;
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
    if(vvBonjour::VV_OK  == resolver.resolveBonjourEntry((*res)->_bonjourEntry))
    {
      serversock = resolver.getBonjourSocket();
      if(NULL != serversock)
      {
        vvSocketIO sockIO = vvSocketIO(serversock);
        sockIO.putBool(false); // no vvGpu info needed
        bool vserverRdy;
        sockIO.getBool(vserverRdy);

        sockets.push_back(serversock);
        int s = vvSocketMap::add(serversock);
        if(sockstr.str() != "") sockstr << ",";
        sockstr << s;
      }
      else
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
  }

  vvServer::vvCreateRemoteServerRes res;
  if(ready) // all resources are connected and ready to use
  {
    vvRendererFactory::Options opt;
    opt["sockets"] = sockstr.str();

    std::cerr << "benötigte Knoten: " << job->request->requirements << std::endl;
    if(job->request->requirements > 1) // brick rendering case
    {
      std::stringstream brickNum;
      brickNum << job->request->requirements;
      opt["bricks"] = brickNum.str();
      opt["brickrenderer"] = "image";

      std::stringstream displaystr;
      displaystr << ":0,:0"; // TODO: fix this hardcoded setting!
      opt["displays"] = displaystr.str();

      res = vvServer::createRemoteServer(clientsock, "parbrick", opt);
    }
    else // regular remote rendering case
    {
      res = vvServer::createRemoteServer(clientsock, std::string("forwarding"), opt);
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

  pthread_mutex_lock(&vvResourceManager::inst->_resourcesMutex);

  for(std::vector<vvResource*>::iterator res = job->resources.begin();res!=job->resources.end();res++)
  {
    (*res)->GPUs[0].freeMem = 1000;
  }

  pthread_mutex_unlock(&vvResourceManager::inst->_resourcesMutex);
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

void vvResourceManager::exitLocalCallback()
{
  pthread_mutex_lock(&_resourcesMutex);

  std::vector<vvResource*>::iterator selfRes = _resources.begin();

  while(!(*selfRes)->server && selfRes != _resources.end()) selfRes++;

  (*selfRes)->GPUs[0].freeMem = 1000;

  pthread_mutex_unlock(&_resourcesMutex);

  while(initNextJob()) {}; // process all waiting jobs
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
