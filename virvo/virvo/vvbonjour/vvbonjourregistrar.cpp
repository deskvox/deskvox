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

#ifdef HAVE_BONJOUR

#include "../vvdebugmsg.h"
#include "../vvsocket.h"

#include <iostream>
#include <sstream>

vvBonjourRegistrar::vvBonjourRegistrar()
{
  _serviceRef = NULL;
  _eventLoop = NULL;
}

vvBonjourRegistrar::~vvBonjourRegistrar()
{
  if(_serviceRef)
    unregisterService();
}

DNSServiceErrorType vvBonjourRegistrar::registerService(const vvBonjourEntry& entry, const ushort port)
{
  vvDebugMsg::msg(3, "vvBonjourRegistrar::registerService() Enter");

  DNSServiceErrorType error;

  error = DNSServiceRegister(&_serviceRef,
                0,                // no flags
                0,                // all network interfaces
                entry.getServiceName().c_str(),
                entry.getRegisteredType().c_str(),
                entry.getReplyDomain().c_str(),
                NULL,             // use default host name
                htons(port),      // port number
                0,                // length of TXT record
                NULL,             // no TXT record
                RegisterCallBack, // call back function
                this);            // no context

  if (error == kDNSServiceErr_NoError)
  {
    _eventLoop = new vvBonjourEventLoop(_serviceRef);
    _eventLoop->run(true, -1.0);
    while(!_eventLoop->_noMoreFlags)
    {
      sleep(1);
      //waiting...
    }
  }
  else
  {
    vvDebugMsg::msg(2, "vvBonjourRegistrar::registerService(): DNSServiceResolve failed with error code ", error);
  }

  return error;
}

void vvBonjourRegistrar::unregisterService()
{
  if(!_serviceRef)
  {
    vvDebugMsg::msg(2, "vvBonjourRegistrar::unregisterService() no service registered");
    return;
  }

  if(_eventLoop)
  {
    _eventLoop->stop();
    delete _eventLoop;
    _eventLoop = NULL;
  }

  DNSServiceRefDeallocate(_serviceRef);
  _serviceRef = NULL;
}

void vvBonjourRegistrar::RegisterCallBack(DNSServiceRef service,
           DNSServiceFlags flags,
           DNSServiceErrorType errorCode,
           const char * name,
           const char * type,
           const char * domain,
           void * context)
{
  vvDebugMsg::msg(3, "vvBonjourRegistrar::RegisterCallBack() Enter");
  (void)service;
  (void)flags;

  vvBonjourRegistrar* instance = reinterpret_cast<vvBonjourRegistrar*>(context);

  if (errorCode != kDNSServiceErr_NoError)
  {
    instance->_registeredService = vvBonjourEntry(name, type, domain);
    vvDebugMsg::msg(3, "vvBonjourRegistrar::RegisterCallBack() error");
  }
  else
  {
    if(vvDebugMsg::getDebugLevel() >= 3)
    {
      std::ostringstream errmsg;
      errmsg << "vvBonjourRegistrar::RegisterCallBack() Entry registered: " << name << "." << type << domain;
      vvDebugMsg::msg(0, errmsg.str().c_str());
    }
  }

  if (!(flags & kDNSServiceFlagsMoreComing))
  {
    // registering done
    instance->_eventLoop->_noMoreFlags = true;
    instance->_eventLoop->stop();
  }
}

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
