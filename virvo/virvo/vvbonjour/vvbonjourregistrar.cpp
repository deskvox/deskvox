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

#include "vvbonjourregistrar.h"
#include "../vvdebugmsg.h"

#include <iostream>

using std::cerr;
using std::endl;

#ifdef HAVE_BONJOUR

vvBonjourRegistrar::vvBonjourRegistrar()
  : _dnsServiceRef(NULL), _socketMonitor(NULL)
{

}

vvBonjourRegistrar::~vvBonjourRegistrar()
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

void vvBonjourRegistrar::registerService(const vvBonjourEntry& entry, const int port)
{
  vvDebugMsg::msg(3, "vvBonjourRegistrar::registerService()");

  if (_dnsServiceRef != NULL)
  {
    vvDebugMsg::msg(0, "vvBonjourRegistrar::registerService(): already registered a service");
    return;
  }

  // Convert port to big endian.
  const ushort nwport = htons((ushort)port);

  DNSServiceErrorType error = DNSServiceRegister(&_dnsServiceRef, 0, 0, entry.getServiceName().c_str(),
                                                 entry.getRegisteredType().c_str(),
                                                 entry.getReplyDomain().empty() ? NULL : entry.getReplyDomain().c_str(),
                                                 0, nwport, 0, 0, bonjourRegisterService, this);

  if (error != kDNSServiceErr_NoError)
  {
    vvDebugMsg::msg(0, "vvBonjourRegistrar::registerService(): DNSServiceRegister failed with error code ", error);
    return;
  }

  vvDebugMsg::msg(3, "vvBonjourRegistrar::registerService(): DNSServiceRegister registered service: ",
                  (entry.getServiceName() + ", protocol: " + entry.getRegisteredType()).c_str());

  const int sockfd = DNSServiceRefSockFD(_dnsServiceRef);

  if (sockfd == -1)
  {
    vvDebugMsg::msg(0, "vvBonjourRegistrar::registerService(): DNSServiceRefSockFD returned -1");
    return;
  }

  vvSocket* socket = new vvSocket(vvSocket::VV_TCP, sockfd);
  std::vector<vvSocket*> sockets;
  sockets.push_back(socket);

  _socketMonitor = new vvSocketMonitor(sockets);

  // No need to loop. Only one socket.
  _socketMonitor->wait();
  bonjourSocketReadyRead();
}

vvBonjourEntry vvBonjourRegistrar::getRegisteredService() const
{
  return _registeredService;
}

void vvBonjourRegistrar::bonjourSocketReadyRead()
{
  vvDebugMsg::msg(3, "vvBonjourRegistrar::bonjourSocketReadyRead()");

  DNSServiceErrorType error = DNSServiceProcessResult(_dnsServiceRef);
  if (error != kDNSServiceErr_NoError)
  {
    vvDebugMsg::msg(0, "vvBonjourRegistrar::bonjourSocketReadyRead(): DNSServiceProcessResult failed with error code ", error);
    return;
  }
}

void vvBonjourRegistrar::bonjourRegisterService(DNSServiceRef, DNSServiceFlags, DNSServiceErrorType errorCode,
                                                const char* serviceName, const char* registeredType,
                                                const char* replyDomain, void* data)
{
  vvDebugMsg::msg(3, "vvBonjourRegistrar::bonjourRegisterService()");
  vvBonjourRegistrar* instance = reinterpret_cast<vvBonjourRegistrar*>(data);

  if (errorCode != kDNSServiceErr_NoError)
  {
    vvDebugMsg::msg(0, "vvBonjourRegistrar::bonjourRegisterService: an error occurred in the callback function");
    return;
  }

  instance->_registeredService = vvBonjourEntry(serviceName, registeredType, replyDomain);
}

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
