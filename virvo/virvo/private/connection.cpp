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

#include "connection.h"

#include <boost/asio/buffer.hpp>
#include <boost/asio/connect.hpp>
#include <boost/asio/placeholders.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>
#include <boost/bind.hpp>
#include <boost/thread.hpp>

#ifndef NDEBUG
#include <iostream>
#endif
#include <sstream>

using virvo::Connection;
using virvo::MessagePointer;

using boost::asio::ip::tcp;

//--------------------------------------------------------------------------------------------------
// Misc.
//--------------------------------------------------------------------------------------------------

template<class T>
inline std::string to_string(T const& x)
{
    std::ostringstream stream;
    stream << x;
    return stream.str();
}

//--------------------------------------------------------------------------------------------------
// Connection::Handler
//--------------------------------------------------------------------------------------------------

Connection::Handler::~Handler()
{
}

bool Connection::Handler::on_connect(Connection* /*conn*/)
{
    return true;
}

bool Connection::Handler::on_read(Connection* /*conn*/, MessagePointer /*message*/)
{
    return true;
}

bool Connection::Handler::on_write(Connection* /*conn*/, MessagePointer /*message*/)
{
    return true;
}

bool Connection::Handler::on_error(Connection* /*conn*/, boost::system::error_code const& e)
{
#ifndef NDEBUG
    std::cout << "Connection::on_error: " << e.message() << std::endl;
#else
    static_cast<void>(e);
#endif

    return true;
}

//--------------------------------------------------------------------------------------------------
// Connection
//--------------------------------------------------------------------------------------------------

Connection::Connection(Handler* handler)
    : handler_(handler)
    , io_service_()
    , socket_(io_service_)
    , strand_(io_service_)
{
}

Connection::~Connection()
{
    stop();
}

void Connection::set_handler(Handler* handler)
{
    handler_ = handler;
}

void Connection::run()
{
#ifndef NDEBUG
    try
    {
        io_service_.run();
    }
    catch (std::exception& e)
    {
        std::cout << "Connection::run: EXCEPTION caught: " << e.what() << std::endl;
        throw;
    }
#else
    io_service_.run();
#endif
}

void Connection::run_in_thread()
{
    boost::thread runner = boost::thread(&Connection::run, this);

    runner.detach();
}

void Connection::stop()
{
    io_service_.stop();
    io_service_.reset();
}

void Connection::connect(std::string const& host, int port)
{
    do_connect(host, port);
}

void Connection::disconnect()
{
    boost::system::error_code e;

    socket_.shutdown(tcp::socket::shutdown_both, e);
    socket_.close(e);
}

void Connection::write(MessagePointer message)
{
    strand_.post(boost::bind(&Connection::do_write, this, message));
}

//--------------------------------------------------------------------------------------------------
// Connection (implementation)
//--------------------------------------------------------------------------------------------------

void Connection::do_connect(std::string const& host, int port)
{
#ifndef NDEBUG
    std::cout << "Connection::do_connect... \"" << host << "\", port " << port << std::endl;
#endif

    // Resolve the host name into an IP address.
    tcp::resolver resolver(io_service_);
    tcp::resolver::query query(host, to_string(port));
    tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);

    // Start an asynchronous connect operation.
    boost::asio::async_connect(
            socket_,
            endpoint_iterator,
            boost::bind(&Connection::handle_connect, this, boost::asio::placeholders::error)
            );
}

void Connection::handle_connect(boost::system::error_code const& e)
{
    assert(handler_);

    if (!e)
    {
        if (handler_->on_connect(this))
        {
            // Successfully established connection.
            // Start reading the messages.
            do_read();
        }
    }
    else
    {
        handler_->on_error(this, e);
    }
}

void Connection::do_read()
{
    MessagePointer message = makeMessage();

    // Issue a read operation to read exactly the number of bytes in a header.
    boost::asio::async_read(
            socket_,
            boost::asio::buffer(&message->header_, sizeof(message->header_)),
            boost::bind(&Connection::handle_read_header, this, boost::asio::placeholders::error, message)
            );
}

void Connection::handle_read_header(boost::system::error_code const& e, MessagePointer message)
{
    assert(handler_);

    if (!e)
    {
        //
        // TODO:
        // Need to deserialize the message-header!
        //

        // Allocate memory for the message data
        message->data_.resize(message->header_.size_);

        assert( message->header_.size_ != 0 );
        assert( message->header_.size_ == message->data_.size() );

        // Start an asynchronous call to receive the data.
        boost::asio::async_read(
                socket_,
                boost::asio::buffer(&message->data_[0], message->data_.size()),
                boost::bind(&Connection::handle_read_data, this, boost::asio::placeholders::error, message)
                );
    }
    else
    {
        handler_->on_error(this, e);
    }
}

void Connection::handle_read_data(boost::system::error_code const& e, MessagePointer message)
{
    assert(handler_);

    if (!e)
    {
        // Call the handler
        handler_->on_read(this, message);

        // Read the next message
        do_read();
    }
    else
    {
        handler_->on_error(this, e);
    }
}

void Connection::do_write(MessagePointer message)
{
    write_queue_.push_back(message);

    if (write_queue_.size() == 1)
    {
        do_write();
    }
}

void Connection::do_write()
{
    // Get the next message
    MessagePointer message = write_queue_.front();

    //
    // TODO:
    // Need to serialize the message-header!
    //

    // Send the header and the data in a single write operation.
    std::vector<boost::asio::const_buffer> buffers;

    buffers.push_back(boost::asio::const_buffer(&message->header_, sizeof(message->header_)));
    buffers.push_back(boost::asio::const_buffer(&message->data_[0], message->data_.size()));

    // Start the write operation.
    boost::asio::async_write(
            socket_,
            buffers,
            strand_.wrap(boost::bind(&Connection::handle_write, this, boost::asio::placeholders::error, message))
            );
}

void Connection::handle_write(boost::system::error_code const& e, MessagePointer message)
{
    assert(handler_);

    write_queue_.pop_front();

    if (!e)
    {
        // Message successfully sent to server.
        // Call the handler
        handler_->on_write(this, message);

        if (!write_queue_.empty())
        {
            do_write();
        }
    }
    else
    {
        handler_->on_error(this, e);
    }
}
