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

#include <virvo/vvrequestmanagement.h>
#ifdef HAVE_BONJOUR
#include <virvo/vvbonjour/vvbonjourentry.h>
#endif // HAVE_BONJOUR
#include <pthread.h>

#include "vvserver.h"

// forward declarations
class vvSimpleServer;
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
class vvResourceManager : public vvServer
{  
private:
  struct vvResource
  {
  public:
    vvResource()
    {
      upToDate = true;
      server = NULL;
    }

    bool   upToDate;
    std::vector<vvGpu::vvGpuInfo> ginfos;
  #ifdef HAVE_BONJOUR
    vvBonjourEntry bonjourEntry;
  #endif
    vvServer *server;

    // vars for future use
    ushort numCPUs;
    uint   cpuMemSize;
  };

  struct vvJob
  {
    vvRequest               *request;
    std::vector<vvResource*> resources;
  };

public:
  /**
    Creates a resource manager connected with server
    \param server if set, resource manager also uses this server running locally
    */
  vvResourceManager();
  ~vvResourceManager();

  void addJob(vvTcpSocket* sock);         ///< thread-savely adds new connection to the job list
  bool initNextJob();                     ///< thread-savely check for waiting jobs and free resources and pair if possible

  static void updateResources(void * param);

private:
  bool serverLoop();
  static void * localServerLoop(void *param);
  void handleNextConnection(vvTcpSocket *sock);
  static void * processJob(void *param);
  uint getFreeResourceCount();
  vvResource* getFreeResource();

  vvSimpleServer *_simpleServer;
#ifdef HAVE_BONJOUR
  vvBonjourBrowser *_browser;
  static std::vector<vvGpu::vvGpuInfo> getResourceGpuInfos(const vvBonjourEntry entry);
#endif // HAVE_BONJOUR
  std::vector<vvRequest*>  _requests;
  std::vector<vvResource*> _resources;
  pthread_mutex_t _requestsMutex;
  pthread_mutex_t _resourcesMutex;
};

#endif // _VV_RESOURCEMANAGER_H_

//===================================================================
// End of File
//===================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
