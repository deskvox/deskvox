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

#include <boost/asio/buffer.hpp>
#if BOOST_VERSION >= 104700
#include <boost/asio/connect.hpp>
#endif
#include <boost/asio/placeholders.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>

#include <boost/bind.hpp>

#ifndef NDEBUG
#include <iostream>
#endif


using virvo::Client;
using virvo::MessagePointer;

using boost::asio::ip::tcp;


template<class T>
inline std::string to_string(T const& x)
{
    std::ostringstream stream;
    stream << x;
    return stream.str();
}


Client::Client()
    : io_service_()
    , socket_(io_service_)
{
}


Client::~Client()
{
    boost::system::error_code e;

    socket_.shutdown(boost::asio::ip::tcp::socket::shutdown_both, e);
    socket_.close(e);
}


void Client::run()
{
#ifndef NDEBUG
    try
    {
        io_service_.run();
    }
    catch (std::exception& e)
    {
        std::cout << "Client::run: EXCEPTION caught: " << e.what() << std::endl;
        throw;
    }
#else
    io_service_.run();
#endif
}


void Client::stop()
{
    io_service_.stop();
}


void Client::connect(std::string const& host, unsigned short port)
{
    // Start the connect operation
    do_connect(host, port);
}


void Client::write(MessagePointer message)
{
    // Start the write operation
    io_service_.post(boost::bind(&Client::do_write, this, message));
}


void Client::write(MessagePointer message, Callback handler)
{
    // Register the callback
    callbacks_.insert(Callbacks::value_type(message->id(), handler));

    // Start the write operation
    write(message);
}


bool Client::on_connect(boost::system::error_code const& /*e*/)
{
    return true;
}


void Client::on_error(boost::system::error_code const& e)
{
#ifndef NDEBUG
    std::cout << "Client::on_error: " << e.message() << std::endl;
#else
    static_cast<void>(e);
#endif
}


void Client::do_connect(std::string const& host, unsigned short port)
{
#if BOOST_VERSION >= 104700

    // Resolve the host name into an IP address.
    tcp::resolver resolver(io_service_);
    tcp::resolver::query query(host, to_string(port));
    tcp::resolver::iterator endpoint_iterator = resolver.resolve(query);

    // Start an asynchronous connect operation.
    boost::asio::async_connect(
            socket_,
            endpoint_iterator,
            boost::bind(&Client::handle_connect, this, boost::asio::placeholders::error)
            );

#else

    tcp::endpoint endpoint(boost::asio::ip::address::from_string(host), port);

    // Start an asynchronous connect operation.
    socket_.async_connect(
            endpoint,
            boost::bind(&Client::handle_connect, this, boost::asio::placeholders::error)
            );

#endif
}


void Client::handle_connect(boost::system::error_code const& e)
{
    bool ok = on_connect(e);

    if (!e)
    {
        if (ok)
        {
            // Successfully established connection.
            // Start reading the messages.
            do_read();
        }
    }
    else
    {
#ifndef NDEBUG
        std::cerr << "Client::handle_connect: " << e.message() << std::endl;
#endif
    }
}


void Client::do_read()
{
    MessagePointer message = makeMessage();

    // Issue a read operation to read exactly the number of bytes in a header.
    boost::asio::async_read(
            socket_,
            boost::asio::buffer(static_cast<void*>(&message->header_), sizeof(message->header_)),
            boost::bind(&Client::handle_read_header, this, boost::asio::placeholders::error, message)
            );
}


void Client::handle_read_header(boost::system::error_code const& e, MessagePointer message)
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
                socket_,
                boost::asio::buffer(&message->data_[0], message->data_.size()),
                boost::bind(&Client::handle_read_data, this, boost::asio::placeholders::error, message)
                );
    }
    else
    {
        // Call the handler
        on_error(e);

        // Remove the callback -- if any
        remove_callback(message);
    }
}


void Client::handle_read_data(boost::system::error_code const& e, MessagePointer message)
{
    Callbacks::iterator I = callbacks_.find(message->id());

    if (!e)
    {
        // Call the handler
        if (I == callbacks_.end())
        {
            on_read(message);
        }
        else
        {
            I->second(message);
        }

        // Read the next message
        do_read();
    }
    else
    {
        // Call the handler
        on_error(e);
    }

    // Remove the callback -- if any
    remove_callback(I);
}


void Client::do_write(MessagePointer message)
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
            socket_,
            buffers,
            boost::bind(&Client::handle_write, this, boost::asio::placeholders::error, message)
            );
}


void Client::handle_write(boost::system::error_code const& e, MessagePointer message)
{
    if (!e)
    {
        // Message successfully sent to server.
        // Call the handler
        on_write(message);
    }
    else
    {
        // Call the handler
        on_error(e);

        // Remove the callback from the list -- if any
        remove_callback(message);
    }
}


void Client::remove_callback(MessagePointer message)
{
    remove_callback(callbacks_.find(message->id()));
}


void Client::remove_callback(Callbacks::iterator I)
{
    if (I != callbacks_.end())
    {
        callbacks_.erase(I);
    }
}
