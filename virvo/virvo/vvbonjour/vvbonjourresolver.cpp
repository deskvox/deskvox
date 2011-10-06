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

#include "vvbonjourresolver.h"
#include "../vvdebugmsg.h"
#include "../vvsocket.h"

#include <iostream>
#include <vector>

using std::cerr;
using std::endl;

#ifdef HAVE_BONJOUR

vvBonjourResolver::vvBonjourResolver()
  : _dnsServiceRef(NULL), _socketMonitor(NULL)
{

}

vvBonjourResolver::~vvBonjourResolver()
{
  if (_dnsServiceRef != NULL)
  {
    DNSServiceRefDeallocate(_dnsServiceRef);
    _dnsServiceRef = NULL;
  }

  if (_socketMonitor != NULL)
  {
    _socketMonitor->clear();
  }
  delete _socketMonitor;
}

void vvBonjourResolver::resolveBonjourEntry(const vvBonjourEntry& entry)
{
  vvDebugMsg::msg(3, "vvBonjourResolver::resolveBonjourEntry()");
  if (_dnsServiceRef != NULL)
  {
    vvDebugMsg::msg(0, "vvBonjourResolver::resolveBonjourEntry(): resolve already in process");
    return;
  }

  DNSServiceErrorType error = DNSServiceResolve(&_dnsServiceRef, 0, 0,
                                                entry.getServiceName().c_str(),
                                                entry.getRegisteredType().c_str(),
                                                entry.getReplyDomain().c_str(),
                                                bonjourResolveReply, this);

  if (error != kDNSServiceErr_NoError)
  {
    vvDebugMsg::msg(0, "vvBonjourResolver::resolveBonjourEntry(): DNSServiceResolve failed with error code ", error);
    return;
  }

  const int sockfd = DNSServiceRefSockFD(_dnsServiceRef);

  if (sockfd == -1)
  {
    vvDebugMsg::msg(0, "vvBonjourResolver::resolveBonjourEntry(): DNSServiceRefSockFD returned -1");
    return;
  }

  vvSocket* socket = new vvSocket(sockfd);
  std::vector<vvSocket*> sockets;
  sockets.push_back(socket);

  _socketMonitor = new vvSocketMonitor(sockets);

  // No need to loop. Only one socket.
  _socketMonitor->wait();
  bonjourSocketReadyRead();
}

void vvBonjourResolver::bonjourSocketReadyRead()
{
  vvDebugMsg::msg(3, "vvBonjourResolver::bonjourSocketReadyRead()");

  DNSServiceErrorType error = DNSServiceProcessResult(_dnsServiceRef);
  if (error != kDNSServiceErr_NoError)
  {
    vvDebugMsg::msg(0, "vvBonjourResolver::bonjourSocketReadyRead(): DNSServiceProcessResult failed with error code ", error);
    return;
  }
}

void vvBonjourResolver::bonjourResolveReply(DNSServiceRef, DNSServiceFlags, uint32_t,
                                            DNSServiceErrorType errorCode, const char*,
                                            const char* hostTarget, uint16_t port, uint16_t,
                                            const uchar*, void* data)
{
  vvDebugMsg::msg(3, "vvBonjourResolver::bonjourResolveReply");
  vvBonjourResolver* instance = reinterpret_cast<vvBonjourResolver*>(data);

  if (errorCode != kDNSServiceErr_NoError)
  {
    vvDebugMsg::msg(0, "vvBonjourResolver::bonjourResolveReply: an error occurred in the callback function");
    return;
  }

  port = ntohs(port);
  cerr << port << endl;
  cerr << hostTarget << endl;
}

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
