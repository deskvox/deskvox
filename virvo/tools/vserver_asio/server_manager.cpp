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

#include "server_manager.h"
#include "resource_manager.h"
#include "simple_server.h"

vvServerManager::vvServerManager(unsigned short port, bool useBonjour)
    : BaseType(port)
    , serverMode_(SERVER)
    , useBonjour_(useBonjour)
{
}

vvServerManager::~vvServerManager()
{
}

bool vvServerManager::on_accept(virvo::ServerManager::ConnectionPointer conn, boost::system::error_code const& /*e*/)
{
    std::cout << "vvServerManager: new connection accepted" << std::endl;

    boost::shared_ptr<vvServer> server;

    // Wrap a new vvSimpleServer or vvResourceManager in conn
    if (serverMode_ == SERVER)
        server = boost::make_shared<vvSimpleServer>();
    else
        server = boost::make_shared<vvResourceManager>();

    // Tell the connection about the server
    return conn->accept(server);
}
