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

#include <iostream>


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


Client::Client(boost::asio::io_service& io_service, std::string const& host, unsigned short port, Handler unexpected_handler)
    : io_service_(io_service)
    , socket_(io_service)
    , handlers_()
    , unexpected_handler_(unexpected_handler)
{
    // Start a new connect operation
    do_connect(host, port);
}


void Client::write(MessagePointer message, Handler handler)
{
    // Start the write operation
    io_service_.post(boost::bind(&Client::do_write, this, message, handler));
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
    if (!e)
    {
        // Successfully established connection.
        // Start reading the messages.
        do_read();
    }
    else
    {
        // An error occurred.
        std::cerr << "Client::handle_connect: " << e.message() << std::endl;
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
        // An error occurred.
        std::cerr << "Client::handle_read_header: " << e.message() << std::endl;
    }
}


void Client::handle_read_data(boost::system::error_code const& e, MessagePointer message)
{
    if (!e)
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
        else
        {
            if (unexpected_handler_)
                unexpected_handler_(message);
        }

        // Read the next message
        do_read();
    }
    else
    {
        // An error occurred.
        std::cerr << "Client::handle_read_data: " << e.message() << std::endl;
    }
}


void Client::do_write(MessagePointer message, Handler handler)
{
    // Add the handler to the list
    if (handler)
        handlers_.insert(Handlers::value_type(message->id(), handler));

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


void Client::handle_write(boost::system::error_code const& e, MessagePointer /*message*/)
{
    if (!e)
    {
        // Message successfully sent to server.
    }
    else
    {
        // An error occurred.
        std::cerr << "Client::handle_write: " << e.message() << std::endl;
    }
}
