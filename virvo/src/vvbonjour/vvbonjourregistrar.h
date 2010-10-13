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

#ifndef _VV_BONJOURREGISTRAR_H_
#define _VV_BONJOURREGISTRAR_H_

#include "vvbonjourentry.h"
#include "vvsocket.h"

#ifdef HAVE_BONJOUR

#include <dns_sd.h>

class vvBonjourRegistrar
{
public:
  vvBonjourRegistrar();
  ~vvBonjourRegistrar();

  void registerService(const vvBonjourEntry& entry, const int port);

  vvBonjourEntry getRegisteredService() const;
private:
  DNSServiceRef _dnsServiceRef;
  vvBonjourEntry _registeredService;

  /*!
   * \brief           Callback function passed to bonjour.
   */
  static void DNSSD_API bonjourRegisterService(DNSServiceRef, DNSServiceFlags, DNSServiceErrorType errorCode,
                                               const char* serviceName, const char* registeredType,
                                               const char* replyDomain, void* data);
};

#endif

#endif
