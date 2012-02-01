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

#ifndef _VV_BONJOURBROWSER_H_
#define _VV_BONJOURBROWSER_H_

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#ifdef HAVE_BONJOUR

#include "vvbonjourentry.h"
#include "vvbonjoureventloop.h"

#include <dns_sd.h>
#include <vector>

/*!
 * Browser class for bonjour services.
 * Found entries have to be resolved by vvBonjourResolver in order to create sockets.
 */
class VIRVOEXPORT vvBonjourBrowser
{
public:
  vvBonjourBrowser();
  ~vvBonjourBrowser();

  DNSServiceErrorType browseForServiceType(const std::string serviceType, const std::string domain = "");

  std::vector<vvBonjourEntry> getBonjourEntries() const;
  vvBonjourEventLoop* _eventLoop;
  std::vector<vvBonjourEntry> _bonjourEntries;

private:
  /*!
   * \brief Callback function passed to bonjour.
   */
  static void DNSSD_API BrowseCallBack(DNSServiceRef, DNSServiceFlags flags, uint32_t interfaceIndex,
                                           DNSServiceErrorType errorCode,
                                           const char *name, const char *type, const char *domain,
                                           void *context);
};

#endif

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
