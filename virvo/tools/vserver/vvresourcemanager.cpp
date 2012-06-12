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

#include <algorithm>
#include <iostream>

#ifdef HAVE_BONJOUR
#include <virvo/vvbonjour/vvbonjourentry.h>
#include <virvo/vvbonjour/vvbonjourbrowser.h>
#include <virvo/vvbonjour/vvbonjourresolver.h>
#endif
#include <virvo/vvdebugmsg.h>
#include <virvo/vvsocketmonitor.h>
#include <virvo/vvsocketio.h>
#include <virvo/vvtcpsocket.h>
#include <virvo/vvtoolshed.h>

using namespace std;

vvResourceManager* vvResourceManager::inst = NULL;

vvResourceManager::vvResourceManager(vvServer *server)
{
  vvResourceManager::inst = this;

  if(server)
  {
    vvResource *serverRes = new vvResource;
    serverRes->_server = server;

    // TODO: set appropiate values here

    serverRes->numGPUsUp();
    //serverRes->numGPUsUp();

    _resources.push_back(serverRes);
  }

  pthread_mutex_init(&_jobsMutex, NULL);
  pthread_mutex_init(&_resourcesMutex, NULL);

  pthread_create(&_threadUR,  NULL, updateResources,      this);
}

vvResourceManager::~vvResourceManager()
{
  pthread_cancel(_threadUR);
  pthread_cancel(_threadCWQ);

  pthread_join(_threadUR,  NULL);
  pthread_join(_threadCWQ, NULL);

  pthread_mutex_destroy(&_jobsMutex);
  pthread_mutex_destroy(&_resourcesMutex);
}

void vvResourceManager::addJob(vvTcpSocket* sock)
{
  vvDebugMsg::msg(3, "vvResourceManager::addJob() Enter");

  pthread_mutex_lock(&_jobsMutex);

  vvRequest *req = new vvRequest;
  req->_sock = sock;

  vvSocketIO sockio = vvSocketIO(sock);
  vvSocket::ErrorType err;
  sockio.putBool(false);
  err = sockio.getInt32(req->_priority);
  if(err != vvSocket::VV_OK)
  {
    cerr << "incoming connection socket error" << endl;
    goto abort;
  }
  err = sockio.getInt32(req->_requirements);
  if(err != vvSocket::VV_OK)
  {
    cerr << "incoming connection socket error" << endl;
    goto abort;
  }

  cerr << "priority: " << req->_priority << ", requirements: " << req->_requirements << std::endl;

  _requests.push_back(req);

  abort:
  pthread_mutex_unlock(&_jobsMutex);
}

bool vvResourceManager::initNextJob()
{
  vvDebugMsg::msg(3, "vvResourceManager::initNextJob() Enter");

  bool ret = true;

  pthread_mutex_lock(&_resourcesMutex);
  pthread_mutex_lock(&_jobsMutex);

  if(_requests.empty() || _resources.empty())
  {
    ret = false;
  }
  else
  {
    // TODO: choose free resources smarter
    std::vector<vvResource*>::iterator freeRes = _resources.begin();
    while(freeRes != _resources.end())
    {
      if((*freeRes)->getGPUs() == 0)
        freeRes++;
      else
        break;
    }

    if(_resources.end() == freeRes)
    {
      ret = false; // no free resources yet
    }
    else
    {
      vvJob *job = new vvJob;

      // TODO: Add some smarter prioritoring here
      vvTcpSocket *nextRequest = _requests.front()->_sock;
      delete _requests.front();
      _requests.erase(_requests.begin());
      job->_requestSock = nextRequest;

      (*freeRes)->numGPUsDown();
      job->_resource = *freeRes;

      pthread_t threadID;
      pthread_create(&threadID, NULL, processJob, job);
      _threadsPJ.push_back(threadID);
    }
  }

  pthread_mutex_unlock(&_jobsMutex);
  pthread_mutex_unlock(&_resourcesMutex);

  return ret;
}

void * vvResourceManager::updateResources(void * param)
{
#ifdef HAVE_BONJOUR
  vvDebugMsg::msg(3, "vvResourceManager::updateResources() Enter");

  vvResourceManager *rm = reinterpret_cast<vvResourceManager*>(param);

  vvBonjourBrowser browser;
  browser.browseForServiceType("_vserver._tcp", "", -1.0); // browse in continous mode

  while(true)
  {
    std::vector<vvBonjourEntry> entries = browser.getBonjourEntries();

    pthread_mutex_lock(&rm->_resourcesMutex);

    for(std::vector<vvResource*>::iterator resource = rm->_resources.begin(); resource != rm->_resources.end(); resource++)
    {
      // mark non-local vservers for updating
      if((*resource)->_server == NULL)
      {
        (*resource)->_upToDate = false;
      }
      else
      {
        (*resource)->_upToDate = true;
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
            (*resource)->_upToDate = true;
            inList = true;
            break;
          }
        }
        if(!inList)
        {
          vvResource *res = new vvResource();
          res->_bonjourEntry = *entry;
          res->numGPUsUp();
          //res->numGPUsUp();
          rm->_resources.push_back(res);
        }
      }
    }

    for(std::vector<vvResource*>::iterator resource = rm->_resources.begin(); resource != rm->_resources.end(); resource++)
    {
      if(!(*resource)->_upToDate)
      {
        delete *resource;
        rm->_resources.erase(resource);
        resource--; // prevent double-iteration, because 'for' and 'erase()' both iterate
      }
    }

    pthread_mutex_unlock(&rm->_resourcesMutex);

    while(rm->initNextJob()) {}; // process all waiting jobs

    vvToolshed::sleep(1); // check from time to time only
  }
#else
  vvDebugMsg::msg(0, "vvResourceManager::updateResources() resource live-updating not available");
  (void)param;
#endif

  pthread_exit(NULL);
#ifdef _WIN32
  return NULL;
#endif
}

void * vvResourceManager::processJob(void * param)
{
  vvDebugMsg::msg(3, "vvResourceManager::processJob() Enter");

#ifdef HAVE_BONJOUR
  vvJob *job = reinterpret_cast<vvJob*>(param);
  vvTcpSocket *clientsock = job->_requestSock;

  // 1. case: local rendering
  if(job->_resource->_server)
  {
    vvServer::vvServerThreadArgs *args = new vvServer::vvServerThreadArgs;
    args->_sock = clientsock;
    args->_exitFunc = &vvResourceManager::exitCallback;

    pthread_t pthread;
    pthread_create(&pthread,  NULL, job->_resource->_server->handleClientThread, args);
    pthread_detach(pthread);

    delete job;

    pthread_exit(NULL);
  #ifdef _WIN32
    return NULL;
  #endif
  }

  // 2. case: socket forwarding
  vvTcpSocket *serversock = NULL;

  bool ready = false;
  vvBonjourResolver resolver;
  if(vvBonjour::VV_OK  == resolver.resolveBonjourEntry(job->_resource->_bonjourEntry))
  {
    serversock = resolver.getBonjourSocket();
    if(NULL == serversock)
    {
      ready = false;
      vvDebugMsg::msg(2, "vvResourceManager::processJob() Could not connect to resolved vserver");
    }
    ready = true;
  }
  else
  {
    ready = false;
    vvDebugMsg::msg(2, "vvResourceManager::processJob() Could not resolve bonjour service");
  }

  while(ready)
  {
    std::vector<vvSocket*> readFds;

    readFds.push_back(clientsock);
    readFds.push_back(serversock);

    vvSocket *readable;
    vvSocketMonitor sm;
    sm.setReadFds(readFds);
    vvSocketMonitor::ErrorType smErr = sm.wait(&readable);

    if(vvSocketMonitor::VV_OK == smErr)
    {
      ssize_t clientret = 0;
      ssize_t serverret = 0;

      if(readable == clientsock)
      {
        // data ready for client to server transfer
        int size = clientsock->isDataWaiting();

        if(0 == size)
        {
          vvDebugMsg::msg(3, "vvResourceManager::processJob() clientsocket closed");
          break;
        }

        uchar *buff = new uchar[size];
        clientsock->readData(buff, size, &clientret);
        if(size == clientret)
        {
          ssize_t ret;
          serversock->writeData(buff, size, &ret);
          if(ret != size)
          {
            delete[] buff;
            break;
          }
        }
        else
        {
          cerr << "vvResourceManager::processJob() clientsocket broken" << endl;
        }
        delete[] buff;
      }
      else if(readable == serversock)
      {
        // data ready for server to client transfer
        int size = serversock->isDataWaiting();

        if(0 == size)
        {
          vvDebugMsg::msg(3, "vvResourceManager::processJob() serversocket closed");
          break;
        }

        uchar *buff = new uchar[size];
        serversock->readData(buff, size, &serverret);
        if(size == serverret)
        {
          ssize_t ret;
          clientsock->writeData(buff, size, &ret);
          if(ret != size)
          {
            delete[] buff;
            break;
          }
        }
        else
        {
          cerr << "vvResourceManager::processJob() serversocket broken" << endl;
        }
        delete[] buff;
      }
      else if(readable == NULL)
      {
        cerr << "vvResourceManager::processJob() sm timeout reached" << endl;
        break;
      }
      else
      {
        cerr << "vvResourceManager::processJob() unexpected error" << endl;
        break;
      }

      if(serverret < 0 || clientret < 0)
      {
        vvDebugMsg::msg(2, "vvResourceManager::processJob(): client- or serversocket broken");
        break;
      }
    }
    else
    {
      vvDebugMsg::msg(2, "vvResourceManager::processJob(): socketmonitor returend error");
      break;
    }
  }

  // free resource again
  pthread_mutex_lock(&vvResourceManager::inst->_resourcesMutex);
  job->_resource->numGPUsUp();
  pthread_mutex_unlock(&vvResourceManager::inst->_resourcesMutex);
  delete job;

  if(clientsock)
  {
    clientsock->disconnectFromHost();
  }
  delete clientsock;

  if(serversock)
  {
    serversock->disconnectFromHost();
  }
  delete serversock;

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

  while(!(*selfRes)->_server && selfRes != _resources.end()) selfRes++;

  (*selfRes)->numGPUsUp();

  pthread_mutex_unlock(&_resourcesMutex);
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
