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

#ifndef _VV_BONJOUR_H_
#define _VV_BONJOUR_H_

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#ifdef HAVE_BONJOUR

#include <vector>

#include "vvsocket.h"
#include "vvbonjourentry.h"

/** Wrapper Class for Bonjour
  This class automatically resolves all desired servicies for a given service type
  and returns a list of the demanded entries

  For programming-examples see vvbonjourtest.cpp (compiletarget: vvbonjourtest)

  @author Stavros Delisavas (stavros.delisavas@uni-koeln.de)
 */
class VIRVOEXPORT vvBonjour
{
public:
  vvBonjour();
  ~vvBonjour();

  /**
    Returns a vectorlist of ready to use connected sockets for all existing services.
    @param serviceType String of desired service type.
    @param domain      String of desired domain to search (optional). By default all local domains.
    @return Vectorlist of connected tcp sockets of type vvSocket*
   */
  std::vector<vvSocket*> getSocketsFor(std::string serviceType, std::string domain = "") const;

  /**
    Returns a vectorlist of BonjourEntries for all existing services.
    @param serviceType String of desired service type.
    @param domain      String of desired domain to search (optional). By default all local domains.
    @return Vectorlist of all entries of type vvBonjourEntry, which have to be resolved for further use.
   */
  std::vector<vvBonjourEntry> getEntriesFor(std::string serviceType, std::string domain = "") const;

  /**
    Returns a vectorlist of connectionstring for all existing services.
    @param serviceType String of desired service type.
    @param domain      String of desired domain to search (optional). By default all local domains.
    @return Vectorlist of strings formatted as "hostname:port".
   */
  std::vector<std::string> getConnectionStringsFor(std::string serviceType, std::string domain = "") const;
};

#endif

#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
