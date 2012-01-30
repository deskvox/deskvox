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

#include "vvbonjourbrowser.h"
#include "../vvdebugmsg.h"

#include <algorithm>
#include <iostream>

using std::cerr;
using std::endl;

#ifdef HAVE_BONJOUR

vvBonjourBrowser::vvBonjourBrowser()
  : _dnsServiceRef(NULL), _socketMonitor(NULL),
    _threadRunning(false), _noServicesComing(false)
{

}

vvBonjourBrowser::~vvBonjourBrowser()
{
  if (_threadRunning)
  {
    pthread_join(_thread, NULL);
  }

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

void vvBonjourBrowser::browseForServiceType(const string& serviceType)
{
  vvDebugMsg::msg(3, "vvBonjourBrowser::browseForServiceType()");

  DNSServiceErrorType error = DNSServiceBrowse(&_dnsServiceRef, 0, 0, serviceType.c_str(), 0,
                                               bonjourBrowseReply, this);

  if (error != kDNSServiceErr_NoError)
  {
    vvDebugMsg::msg(0, "vvBonjourBrowser::browseForServiceType(): DNSServiceBrowse failed with error code ", error);
    return;
  }

  const int sockfd = DNSServiceRefSockFD(_dnsServiceRef);

  if (sockfd == -1)
  {
    vvDebugMsg::msg(0, "vvBonjourBrowser::browseForServiceType(): DNSServiceRefSockFD returned -1");
    return;
  }

  vvSocket* socket = new vvSocket(vvSocket::VV_TCP, sockfd);
  std::vector<vvSocket*> sockets;
  sockets.push_back(socket);

  _socketMonitor = new vvSocketMonitor();
  _socketMonitor->setReadFds(sockets);

  _threadRunning = true;
  pthread_create(&_thread, NULL, waitSocketReadyRead, this);
}

bool vvBonjourBrowser::expectingServices() const
{
  return !_noServicesComing;
}

std::vector<vvBonjourEntry> vvBonjourBrowser::getBonjourEntries() const
{
  return _bonjourEntries;
}

void vvBonjourBrowser::bonjourSocketReadyRead()
{
  vvDebugMsg::msg(3, "vvBonjourBrowser::bonjourSocketReadyRead()");

  DNSServiceErrorType error = DNSServiceProcessResult(_dnsServiceRef);
  if (error != kDNSServiceErr_NoError)
  {
    vvDebugMsg::msg(0, "vvBonjourRegistrar::bonjourSocketReadyRead(): DNSServiceProcessResult failed with error code ", error);
    return;
  }
}

void vvBonjourBrowser::bonjourBrowseReply(DNSServiceRef, DNSServiceFlags flags, unsigned int,
                                          DNSServiceErrorType errorCode,
                                          const char* serviceName, const char* registeredType,
                                          const char* replyDomain,
                                          void *data)
{
  vvDebugMsg::msg(3, "vvBonjourBrowser::bonjourBrowseReply()");
  vvBonjourBrowser* instance = reinterpret_cast<vvBonjourBrowser*>(data);

  if (errorCode != kDNSServiceErr_NoError)
  {
    vvDebugMsg::msg(0, "vvBonjourBrowser::bonjourBrowseReply: an error occurred in the callback function");
    return;
  }

  vvBonjourEntry entry = vvBonjourEntry(serviceName, registeredType, replyDomain);
  if (flags & kDNSServiceFlagsAdd)
  {
    // Add flag ==> add the entry.
    if (std::find(instance->_bonjourEntries.begin(), instance->_bonjourEntries.end(), entry)
        == instance->_bonjourEntries.end())
    {
      instance->_bonjourEntries.push_back(entry);
    }
    instance->_noServicesComing = false;
  }
  else
  {
    // No Add flag ==> remove all occurrences of the entry.
    std::remove(instance->_bonjourEntries.begin(), instance->_bonjourEntries.end(), entry);
  }

  if (!(flags & kDNSServiceFlagsMoreComing))
  {
    instance->_noServicesComing = true;
  }
}

void* vvBonjourBrowser::waitSocketReadyRead(void* threadargs)
{
  vvBonjourBrowser* instance = reinterpret_cast<vvBonjourBrowser*>(threadargs);

  if (instance != NULL)
  {
    while (1)
    {
      vvSocket *ready;
      instance->_socketMonitor->wait(&ready);
      instance->bonjourSocketReadyRead();
    }
  }
  return NULL; // no return value?!?
}

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
