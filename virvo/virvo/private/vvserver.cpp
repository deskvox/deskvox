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


#include "vvserver.h"

#include <boost/asio/placeholders.hpp>
#include <boost/bind.hpp>

#include <iostream>


using virvo::Server;
using virvo::MessageHandler;
using virvo::MessagePointer;

using boost::asio::ip::tcp;


Server::Server(boost::asio::io_service& io_service, unsigned short port, MessageHandler handler)
    : connection_ptr_()
    , acceptor_(io_service, tcp::endpoint(tcp::v4(), port))
    , handler_(handler)
{
    assert( static_cast<bool>(handler) );

    connection_ptr_ = makeConnection(io_service);

    // Start an accept operation for a new connection.
    acceptor_.async_accept(
            connection_ptr_->socket(),
            boost::bind(&Server::handle_accept, this, boost::asio::placeholders::error)
            );
}


void Server::write(MessagePointer message)
{
    connection_ptr_->write(message);
}


void Server::handle_accept(boost::system::error_code const& e)
{
    if (!e)
    {
        read_next(); // Start reading from the socket
    }
    else
    {
        // An error occurred.
        std::cerr << "Server::handle_accept: " << e.message() << std::endl;
    }
}


void Server::handle_read(MessagePointer message)
{
    // Handle the message
    handler_(message);

    // Read the next message
    read_next();
}


void Server::read_next()
{
    connection_ptr_->read( boost::bind(&Server::handle_read, this, _1) );
}
