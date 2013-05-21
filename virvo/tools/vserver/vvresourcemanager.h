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

#include <virvo/vvrequestmanagement.h>
#include <virvo/vvbonjour/vvbonjourentry.h>
#include <pthread.h>
#include <set>

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
public:
  vvResourceManager();
  ~vvResourceManager();

private:
  void handleNextConnection(vvTcpSocket *sock);
  bool handleEvent(ThreadData *tData, virvo::RemoteEvent event, const vvSocketIO& io);
  bool createRemoteServer(ThreadData* tData, vvTcpSocket* sock);
  bool allocateResources(ThreadData* tData);

  static void * handleClientThread(void *param);
  static void updateResources(void * param);

  void pairNextJobs();

  bool serverLoop();

  uint getFreeResourceCount() const;
  std::vector<vvResource*> getFreeResources(uint amount) const;

  vvSimpleServer *_simpleServer;
  vvBonjourBrowser *_browser;
  std::multiset<vvRequest*, vvRequest::Compare> _requests;
  std::vector<vvResource*> _resources;
  pthread_mutex_t _requestsMutex;
  pthread_cond_t  _requestsCondition;
  pthread_mutex_t _resourcesMutex;
};

#endif // _VV_RESOURCEMANAGER_H_

//===================================================================
// End of File
//===================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
