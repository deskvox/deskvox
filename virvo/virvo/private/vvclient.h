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


#ifndef VV_PRIVATE_CLIENT_H
#define VV_PRIVATE_CLIENT_H


// Boost.ASIO needs _WIN32_WINNT
#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0501 // Require Windows XP or later
#endif
#endif


#include "vvexport.h"
#include "vvmessage.h"

#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/tcp.hpp>

#include <map>


namespace virvo
{


    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------


    class Client
        : public boost::enable_shared_from_this<Client>
    {
    public:
        typedef boost::function<void(MessagePointer)> Handler;

        // Constructor starts the asynchronous connect operation.
        VVAPI Client(boost::asio::io_service& io_service, std::string const& host, unsigned short port, Handler unexpected_handler);

        // Sends a message to the server.
        // If the server replies, the given handler is executed (may be null).
        VVAPI void write(MessagePointer message, Handler handler = Handler());

    private:
        // Starts a new connect operation.
        void do_connect(std::string const& host, unsigned short port);

        // Handle completion of a connect operation.
        void handle_connect(boost::system::error_code const& e);

        // Starts a new read operation.
        void do_read();

        // Called when a message header is read.
        void handle_read_header(boost::system::error_code const& e, MessagePointer message);

        // Called when a complete message is read.
        void handle_read_data(boost::system::error_code const& e, MessagePointer message);

        // Starts a new write operation.
        void do_write(MessagePointer message, Handler handler);

        // Called when a complete message is written.
        void handle_write(boost::system::error_code const& e, MessagePointer message);

    private:
        typedef std::map<boost::uuids::uuid, Handler> Handlers;

        // The IO service
        boost::asio::io_service& io_service_;
        // The underlying socket.
        boost::asio::ip::tcp::socket socket_;
        // List of callbacks
        Handlers handlers_;
        // The handler to process "unexpected" messages
        Handler unexpected_handler_;
    };


} // namespace virvo


#endif // !VV_PRIVATE_CLIENT_H
