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


#include "vvconnection.h"

#include <boost/asio/buffer.hpp>
#include <boost/asio/placeholders.hpp>
#include <boost/asio/read.hpp>
#include <boost/asio/write.hpp>
#include <boost/bind.hpp>

#include <cassert>
#include <iostream>
#include <vector>


using virvo::Connection;
using virvo::MessageHandler;
using virvo::MessagePointer;


Connection::Connection(boost::asio::io_service& io_service)
    : io_service_(io_service)
    , socket_(io_service)
{
}


void Connection::read(MessageHandler handler)
{
    io_service_.post(boost::bind(&Connection::do_read, this, handler));
}


void Connection::write(MessagePointer message, MessageHandler handler)
{
    io_service_.post(boost::bind(&Connection::do_write, this, message, handler));
}


void Connection::do_read(MessageHandler handler)
{
    MessagePointer message = makeMessage();

    // Issue a read operation to read exactly the number of bytes in a header.
    boost::asio::async_read(
            socket_,
            boost::asio::buffer(static_cast<void*>(&message->header_), sizeof(message->header_)),
            boost::bind(&Connection::handle_read_header, this, boost::asio::placeholders::error, message, handler)
            );
}


void Connection::handle_read_header(boost::system::error_code const& e, MessagePointer message, MessageHandler handler)
{
    if (!e)
    {
        // TODO:
        // Need to deserialize the message-header!

        // Allocate memory for the message data
        message->data_.resize(message->header_.size_);

        // Start an asynchronous call to receive the data.
        boost::asio::async_read(
                socket_,
                boost::asio::buffer(&message->data_[0], message->data_.size()),
                boost::bind(&Connection::handle_read_data, this, boost::asio::placeholders::error, message, handler)
                );
    }
    else
    {
        // An error occurred.
        std::cerr << "Connection::handle_read_header: " << e.message() << std::endl;
    }
}


void Connection::handle_read_data(boost::system::error_code const& e, MessagePointer message, MessageHandler handler)
{
    if (!e)
    {
        // Invoke the completion handler -- if any
        if (handler)
            handler(message);
    }
    else
    {
        // An error occurred.
        std::cerr << "Connection::handle_read_data: " << e.message() << std::endl;
    }
}


void Connection::do_write(MessagePointer message, MessageHandler handler)
{
    // TODO:
    // Need to serialize the message-header!

    assert( message->header_.size_ == message->data_.size() );

    // Send the header and the data in a single write operation
    std::vector<boost::asio::const_buffer> buffers;

    buffers.push_back(boost::asio::buffer(static_cast<void const*>(&message->header_), sizeof(message->header_)));
    buffers.push_back(boost::asio::buffer(message->data_));

    boost::asio::async_write(
            socket_,
            buffers,
            boost::bind(&Connection::handle_write, this, boost::asio::placeholders::error, message, handler)
            );
}


void Connection::handle_write(boost::system::error_code const& e, MessagePointer message, MessageHandler handler)
{
    if (!e)
    {
        // Invoke the completion handler -- if any
        if (handler)
            handler(message);
    }
    else
    {
        // An error occurred.
        std::cerr << "Connection::handle_write: " << e.message() << std::endl;
    }
}
