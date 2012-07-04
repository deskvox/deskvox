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

#ifndef _VV_RESOURCEMANAGER_H_
#define _VV_RESOURCEMANAGER_H_

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#include <vector>

#include <virvo/vvinttypes.h>
#include <virvo/vvbonjour/vvbonjourentry.h>

#include <pthread.h>

// forward declarations
class vvServer;
class vvTcpSocket;
class vvBonjourBrowser;

/**
 * Class for resouce manager
 *
 * vserver extension for managing additional resources on the network
 *
 * It is capable to detect vservers by its own using bonjour or uses
 * uses a provided list of adresses or ip-ranges to connect with them
 * directly
 *
 * @author Stavros Delisavas (stavros.delisavas@uni-koeln.de)
 */
class vvResourceManager
{
private:
  struct vvRequest
  {
    int         priority;
    int         type;
    int         requirements;
    vvTcpSocket *sock;

    bool operator<(vvRequest other)
    {
      // TODO: Fine adjust this
      return priority < other.priority;
    }
  };

  struct vvResource
  {
  public:
    vvResource()
    {
      upToDate = true;
      server = NULL;
    }

    bool   upToDate;
    ushort numGPUs;
    ushort numCPUs;
    uint   gpuMemSize;
    uint   cpuMemSize;
  #ifdef HAVE_BONJOUR
    vvBonjourEntry _bonjourEntry;
  #endif
    vvServer *server;
  };

  struct vvJob
  {
    vvRequest               *request;
    std::vector<vvResource*> resources;
  };

public:
  static vvResourceManager *inst;
  static void exitCallback(void*)
  {
    inst->exitLocalCallback();
  }


  /**
    Creates a resource manager connected with server
    \param server if set, resource manager also uses this server running locally
    */
  vvResourceManager(vvServer *server = NULL);
  ~vvResourceManager();

  void addJob(vvTcpSocket* sock);         ///< thread-savely adds new connection to the job list
  bool initNextJob();                     ///< thread-savely check for waiting jobs and free resources and pair if possible

  static void updateResources(void * param);
private:
  uint getFreeResourceCount();
  vvResource* getFreeResource();

#ifdef HAVE_BONJOUR
  vvBonjourBrowser *_browser;
#endif // HAVE_BONJOUR

  std::vector<vvRequest*>  _requests;
  std::vector<vvResource*> _resources;
  pthread_mutex_t _jobsMutex;
  pthread_mutex_t _resourcesMutex;

  static void * processJob(void *param);

  void exitLocalCallback();
};

#endif // _VV_RESOURCEMANAGER_H_

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
