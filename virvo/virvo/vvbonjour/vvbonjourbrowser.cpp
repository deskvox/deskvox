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

#ifdef HAVE_BONJOUR

#include "../vvdebugmsg.h"

#include <iostream>
#include <sstream>
#include <ostream>

vvBonjourBrowser::vvBonjourBrowser()
  : _eventLoop(NULL)
{}

vvBonjourBrowser::~vvBonjourBrowser()
{
  if(_eventLoop) delete _eventLoop;
}

DNSServiceErrorType vvBonjourBrowser::browseForServiceType(const std::string& serviceType, const std::string domain)
{
  DNSServiceErrorType error;
  DNSServiceRef  serviceRef = NULL;

  _bonjourEntries.clear();

  error = DNSServiceBrowse(&serviceRef,
              0,                    // no flags
              0,                    // all network interfaces
              serviceType.c_str(),  // service type
              domain.c_str(),       // default domains
              BrowseCallBack,       // call back function
              this);                // adress of pointer to eventloop
  if (error == kDNSServiceErr_NoError)
  {
    _eventLoop = new vvBonjourEventLoop(serviceRef);
    _eventLoop->run();
  }
  else
  {
    std::ostringstream errmsg;
    errmsg << "vvBonjourBrowser::browseForServiceType(): DNSServiceBrowse() returned with error no " << error;
    vvDebugMsg::msg(2, errmsg.str().c_str());
  }

  return error;
}

void vvBonjourBrowser::BrowseCallBack(DNSServiceRef, DNSServiceFlags flags, uint32_t interfaceIndex,
                                      DNSServiceErrorType errorCode,
                                      const char * name, const char * type, const char * domain,
                                      void * context)
{
  vvDebugMsg::msg(3, "vvBonjourBrowser::BrowseCallBack() Enter");

  vvBonjourBrowser *instance = reinterpret_cast<vvBonjourBrowser*>(context);

  if (errorCode != kDNSServiceErr_NoError)
    vvDebugMsg::msg(3, "vvBonjourBrowser::BrowseCallBack() Leave");
  else
  {
    if(vvDebugMsg::getDebugLevel() >= 3)
    {
      std::string addString  = (flags & kDNSServiceFlagsAdd) ? "ADD" : "REMOVE";
      std::string moreString = (flags & kDNSServiceFlagsMoreComing) ? "MORE" : "    ";

      std::ostringstream msg;
      msg << addString << " " << moreString << " " << interfaceIndex << " " << name << "." << type << domain;
      vvDebugMsg::msg(0, msg.str().c_str());
    }
    vvBonjourEntry entry = vvBonjourEntry(name, type, domain);
    instance->_bonjourEntries.push_back(entry);
  }

  if (!(flags & kDNSServiceFlagsMoreComing))
  {
    instance->_eventLoop->stop();
  }
}

std::vector<vvBonjourEntry> vvBonjourBrowser::getBonjourEntries() const
{
  return _bonjourEntries;
}

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
