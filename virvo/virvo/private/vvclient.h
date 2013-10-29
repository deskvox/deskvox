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
    {
    public:
        // Constructor.
        VVAPI Client();

        // Destructor.
        VVAPI virtual ~Client();

        // Runs the message loop
        VVAPI void run();

        // Stops the message loop
        VVAPI void stop();

        // Starts a new connect operation.
        VVAPI void connect(std::string const& host, unsigned short port);

        // Sends a message to the server.
        // If the server replies, the given handler is executed (may be null).
        VVAPI void write(MessagePointer message);

        // Called when a new connection has been established.
        // Return true to accept the connection, false to discard the connection.
        VVAPI virtual bool on_connect(boost::system::error_code const& e);

        // Called when an error occurred during a read or a write operation.
        VVAPI virtual void on_error(boost::system::error_code const& e);

        // Called when a new message has successfully been read from the server.
        virtual void on_read(MessagePointer message) = 0;

        // Called when a message has successfully been written to the server.
        virtual void on_write(MessagePointer message) = 0;

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
        void do_write(MessagePointer message);

        // Called when a complete message is written.
        void handle_write(boost::system::error_code const& e, MessagePointer message);

    private:
        // The IO service
        boost::asio::io_service io_service_;
        // The underlying socket.
        boost::asio::ip::tcp::socket socket_;
    };


} // namespace virvo


#endif // !VV_PRIVATE_CLIENT_H
