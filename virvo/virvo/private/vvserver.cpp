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

#include <boost/asio/buffer.hpp>
#include <boost/asio/placeholders.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>

#include <boost/bind.hpp>
#include <boost/ref.hpp>

#include <iostream>


using virvo::Server;
using virvo::MessagePointer;

using boost::asio::ip::tcp;


//--------------------------------------------------------------------------------------------------
// Server::Connection
//--------------------------------------------------------------------------------------------------


Server::Connection::Connection(boost::asio::io_service& io_service)
    : socket_(io_service)
{
}


//--------------------------------------------------------------------------------------------------
// Server
//--------------------------------------------------------------------------------------------------


Server::Server(boost::asio::io_service& io_service, unsigned short port, Handler handler)
    : io_service_(io_service)
    , acceptor_(io_service, tcp::endpoint(tcp::v4(), port))
    , connections_()
    , handler_(handler)
{
    assert( static_cast<bool>(handler_) && "invalid message handler" );

    // Start an accept operation for a new connection.
    do_accept();
}


void Server::write(MessagePointer message, ConnectionPointer conn)
{
    // Start the write operation
    io_service_.post(boost::bind(&Server::do_write, this, message, conn));
}


void Server::broadcast(MessagePointer message)
{
    // Start the write operation
    io_service_.post(boost::bind(&Server::do_broadcast, this, message));
}


void Server::do_accept()
{
    ConnectionPointer conn = boost::make_shared<Connection>(boost::ref(io_service_));

    // Start an accept operation for a new connection.
    acceptor_.async_accept(
            conn->socket_,
            boost::bind(&Server::handle_accept, this, boost::asio::placeholders::error, conn)
            );
}


void Server::handle_accept(boost::system::error_code const& e, ConnectionPointer conn)
{
    if (!e)
    {
        std::cerr << "Server::handle_accept: connection established" << std::endl;

        // Connection established
        connections_.insert(conn);

        // Start reading from the socket
        do_read(conn);

        // Start a new accept operation
        do_accept();
    }
    else
    {
        // An error occurred.
        std::cerr << "Server::handle_accept: " << e.message() << std::endl;

        // No need to remove the connection...
    }
}


void Server::do_read(ConnectionPointer conn)
{
    MessagePointer message = makeMessage();

    // Issue a read operation to read exactly the number of bytes in a header.
    boost::asio::async_read(
            conn->socket_,
            boost::asio::buffer(static_cast<void*>(&message->header_), sizeof(message->header_)),
            boost::bind(&Server::handle_read_header, this, boost::asio::placeholders::error, message, conn)
            );
}


void Server::handle_read_header(boost::system::error_code const& e, MessagePointer message, ConnectionPointer conn)
{
    if (!e)
    {
        //
        // TODO:
        // Need to deserialize the message-header!
        //

        // Allocate memory for the message data
        message->data_.resize(message->header_.size_);

        // Start an asynchronous call to receive the data.
        boost::asio::async_read(
                conn->socket_,
                boost::asio::buffer(&message->data_[0], message->data_.size()),
                boost::bind(&Server::handle_read_data, this, boost::asio::placeholders::error, message, conn)
                );
    }
    else
    {
        // An error occurred.
        std::cerr << "Server::handle_read_header: " << e.message() << std::endl;

        // Remove the connection from the list of active connections
        connections_.erase(conn);
    }
}


void Server::handle_read_data(boost::system::error_code const& e, MessagePointer message, ConnectionPointer conn)
{
    if (!e)
    {
        // Invoke the message handler
        handler_(message, conn);

        // Read the next message
        do_read(conn);
    }
    else
    {
        // An error occurred.
        std::cerr << "Server::handle_read_data: " << e.message() << std::endl;

        // Remove the connection from the list of active connections
        connections_.erase(conn);
    }
}


void Server::do_write(MessagePointer message, ConnectionPointer conn)
{
    //
    // TODO:
    // Need to serialize the message-header!
    //

    assert( message->header_.size_ == message->data_.size() );

    // Send the header and the data in a single write operation.
    std::vector<boost::asio::const_buffer> buffers;

    buffers.push_back(boost::asio::buffer(static_cast<void const*>(&message->header_), sizeof(message->header_)));
    buffers.push_back(boost::asio::buffer(message->data_));

    // Start the write operation.
    boost::asio::async_write(
            conn->socket_,
            buffers,
            boost::bind(&Server::handle_write, this, boost::asio::placeholders::error, message, conn)
            );
}


void Server::do_broadcast(MessagePointer message)
{
    Connections::iterator I = connections_.begin();
    Connections::iterator E = connections_.end();

    for ( ; I != E; ++I)
    {
        do_write(message, *I);
    }
}


void Server::handle_write(boost::system::error_code const& e, MessagePointer /*message*/, ConnectionPointer conn)
{
    if (!e)
    {
        // Message successfully sent to server.
    }
    else
    {
        // An error occurred.
        std::cerr << "Server::handle_write: " << e.message() << std::endl;

        // Remove the connection from the list of active connections
        connections_.erase(conn);
    }
}
