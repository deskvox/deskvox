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

#ifndef _VV_BONJOURENTRY_H_
#define _VV_BONJOURENTRY_H_

//#define HAVE_BONJOUR
#ifdef HAVE_BONJOUR

#include <string>

using std::string;

class vvBonjourEntry
{
public:
  vvBonjourEntry();
  vvBonjourEntry(const string& serviceName,
                 const string& registeredType,
                 const string& replyDomain);

  string getServiceName() const;
  string getRegisteredType() const;
  string getReplyDomain() const;

  bool operator==(const vvBonjourEntry& rhs) const;
private:
  string _serviceName;
  string _registeredType;
  string _replyDomain;
};

#endif

#endif
