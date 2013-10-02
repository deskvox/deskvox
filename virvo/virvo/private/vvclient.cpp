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


#include "vvclient.h"

#include <boost/asio/connect.hpp>
#include <boost/asio/placeholders.hpp>
#include <boost/bind.hpp>

#include <iostream>


using virvo::Client;
using virvo::MessageHandler;
using virvo::MessagePointer;

using boost::asio::ip::tcp;


template<class T>
inline std::string to_string(T const& x)
{
    std::ostringstream stream;
    stream << x;
    return stream.str();
}


Client::Client(boost::asio::io_service& io_service, std::string const& host, unsigned short port)
    : connection_(io_service)
{
    // Resolve the host name into an IP address.
    tcp::resolver resolver(io_service);
    tcp::resolver::query query(host, to_string(port));
    tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);

    // Start an asynchronous connect operation.
    boost::asio::async_connect(
            connection_.socket(),
            endpoint_iterator,
            boost::bind(&Client::handle_connect, this, boost::asio::placeholders::error)
            );
}


void Client::write(MessagePointer message, MessageHandler handler)
{
    if (handler)
        handlers_.insert(Handlers::value_type(message->id(), handler));

    connection_.write(message);
}


void Client::handle_connect(boost::system::error_code const& e)
{
    if (!e)
    {
        // Successfully established connection.
        // Start reading the messages.
        read_next();
    }
    else
    {
        // An error occurred.
        std::cerr << "Client::handle_connect: " << e.message() << std::endl;
    }
}


void Client::handle_read(MessagePointer message)
{
    // Find the completion handler
    Handlers::iterator I = handlers_.find(message->id());

    if (I != handlers_.end())
    {
        // Invoke the completion handler
        I->second(message);
        // Remove the handler from the list
        handlers_.erase(I);
    }

    // Read the next message
    read_next();
}


void Client::read_next()
{
    // Read the next message
    connection_.read( boost::bind(&Client::handle_read, this, _1) );
}
