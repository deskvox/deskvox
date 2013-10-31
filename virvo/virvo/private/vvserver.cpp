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

#ifndef NDEBUG
#include <iostream>
#endif


using virvo::MessagePointer;
using virvo::Server;
using virvo::ServerManager;

using boost::asio::ip::tcp;


//--------------------------------------------------------------------------------------------------
// ServerManager::Connection
//--------------------------------------------------------------------------------------------------


ServerManager::Connection::Connection(boost::asio::io_service& io_service, ServerManager* server)
    : socket_(io_service)
    , manager_(server)
    , server_()
{
}


ServerManager::Connection::~Connection()
{
    boost::system::error_code e;

    socket_.shutdown(boost::asio::ip::tcp::socket::shutdown_both, e);
    socket_.close(e);
}


namespace
{
    struct NoDelete {
        void operator ()(Server*) {}
    };
}

bool ServerManager::Connection::accept(Server* server)
{
    server_ = boost::shared_ptr<Server>(server, NoDelete());
    return true;
}


bool ServerManager::Connection::accept(boost::shared_ptr<Server> server)
{
    server_ = server;
    return true;
}


void ServerManager::Connection::write(MessagePointer message)
{
    manager_->write(message, shared_from_this());
}


//--------------------------------------------------------------------------------------------------
// ServerManager
//--------------------------------------------------------------------------------------------------


ServerManager::ServerManager(unsigned short port)
    : io_service_()
    , acceptor_(io_service_, tcp::endpoint(tcp::v6(), port))
    , connections_()
{
    // Start an accept operation for a new connection.
    do_accept();
}


ServerManager::~ServerManager()
{
}


void ServerManager::run()
{
#ifndef NDEBUG
    try
    {
        io_service_.run();
    }
    catch (std::exception& e)
    {
        std::cout << "ServerManager::run: EXCEPTION caught: " << e.what() << std::endl;
        throw;
    }
#else
    io_service_.run();
#endif
}


void ServerManager::stop()
{
    io_service_.stop();
}


void ServerManager::write(MessagePointer message, ConnectionPointer conn)
{
    // Start the write operation
    io_service_.post(boost::bind(&ServerManager::do_write, this, message, conn));
}


void ServerManager::broadcast(MessagePointer message)
{
    // Start the write operation
    io_service_.post(boost::bind(&ServerManager::do_broadcast, this, message));
}


bool ServerManager::on_accept(ConnectionPointer /*conn*/, boost::system::error_code const& /*e*/)
{
    return false;
}


void ServerManager::do_accept()
{
#ifndef NDEBUG
    std::cout << "ServerManager::do_accept..." << std::endl;
#endif

#if 1
    ConnectionPointer conn(new Connection(boost::ref(io_service_), this));
#else
    ConnectionPointer conn = boost::make_shared<Connection>(boost::ref(io_service_), this);
#endif

    // Start an accept operation for a new connection.
    acceptor_.async_accept(
            conn->socket_,
            boost::bind(&ServerManager::handle_accept, this, boost::asio::placeholders::error, conn)
            );
}


void ServerManager::handle_accept(boost::system::error_code const& e, ConnectionPointer conn)
{
    bool ok = on_accept(conn, e);

    if (!e)
    {
        if (ok)
        {
            // Connection established
            connections_.insert(conn);
            // Start reading from the socket
            do_read(conn);
        }

        // Start a new accept operation
        do_accept();
    }
    else
    {
#ifndef NDEBUG
        std::cerr << "ServerManager::handle_accept: " << e.message() << std::endl;
#endif

        // No need to remove the connection...
    }
}


void ServerManager::do_read(ConnectionPointer conn)
{
    MessagePointer message = makeMessage();

    // Issue a read operation to read exactly the number of bytes in a header.
    boost::asio::async_read(
            conn->socket_,
            boost::asio::buffer(static_cast<void*>(&message->header_), sizeof(message->header_)),
            boost::bind(&ServerManager::handle_read_header, this, boost::asio::placeholders::error, message, conn)
            );
}


void ServerManager::handle_read_header(boost::system::error_code const& e, MessagePointer message, ConnectionPointer conn)
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
                boost::bind(&ServerManager::handle_read_data, this, boost::asio::placeholders::error, message, conn)
                );
    }
    else
    {
        // Call the handler
        if (conn->server_)
            conn->server_->on_error(conn, e);

        // Remove the connection from the list of active connections
        connections_.erase(conn);
    }
}


void ServerManager::handle_read_data(boost::system::error_code const& e, MessagePointer message, ConnectionPointer conn)
{
    if (!e)
    {
        // Call the handler
        if (conn->server_)
            conn->server_->on_read(conn, message);

        // Read the next message
        do_read(conn);
    }
    else
    {
        // Call the handler
        if (conn->server_)
            conn->server_->on_error(conn, e);

        // Remove the connection from the list of active connections
        connections_.erase(conn);
    }
}


void ServerManager::do_write(MessagePointer message, ConnectionPointer conn)
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
            boost::bind(&ServerManager::handle_write, this, boost::asio::placeholders::error, message, conn)
            );
}


void ServerManager::do_broadcast(MessagePointer message)
{
    Connections::iterator I = connections_.begin();
    Connections::iterator E = connections_.end();

    for ( ; I != E; ++I)
    {
        do_write(message, *I);
    }
}


void ServerManager::handle_write(boost::system::error_code const& e, MessagePointer message, ConnectionPointer conn)
{
    if (!e)
    {
        // Message successfully sent to server.
        // Call the handler
        if (conn->server_)
            conn->server_->on_write(conn, message);
    }
    else
    {
        // Call the handler
        if (conn->server_)
            conn->server_->on_error(conn, e);

        // Remove the connection from the list of active connections
        connections_.erase(conn);
    }
}


//--------------------------------------------------------------------------------------------------
// Server
//--------------------------------------------------------------------------------------------------


void Server::on_error(ServerManager::ConnectionPointer /*conn*/, boost::system::error_code const& e)
{
#ifndef NDEBUG
    std::cout << "Server::on_error: " << e.message() << std::endl;
#else
    static_cast<void>(e);
#endif
}
