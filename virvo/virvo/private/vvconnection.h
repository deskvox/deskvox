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


#ifndef VV_PRIVATE_CONNECTION_H
#define VV_PRIVATE_CONNECTION_H


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
#include <boost/ref.hpp>


namespace virvo
{


    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------


    class Connection
        : public boost::enable_shared_from_this<Connection>
    {
    public:
        // Constructor.
        VVAPI explicit Connection(boost::asio::io_service& io_service);

        // Get the underlying socket.
        // Used for making a connection or for accepting an incoming connection.
        boost::asio::ip::tcp::socket& socket()
        {
            return socket_;
        }

        // Asynchronously read a message from the socket.
        VVAPI void read(MessageHandler handler = MessageHandler());

        // Asynchronously write a message to the socket.
        VVAPI void write(MessagePointer message, MessageHandler handler = MessageHandler());

    private:
        void do_read(MessageHandler handler);

        void handle_read_header(boost::system::error_code const& e, MessagePointer message, MessageHandler handler);
        void handle_read_data(boost::system::error_code const& e, MessagePointer message, MessageHandler handler);

        void do_write(MessagePointer message, MessageHandler handler);

        void handle_write(boost::system::error_code const& e, MessagePointer message, MessageHandler handler);

    private:
        // The IO service
        boost::asio::io_service& io_service_;
        // The underlying socket.
        boost::asio::ip::tcp::socket socket_;
    };


    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------


    typedef boost::shared_ptr<Connection> ConnectionPointer;

    inline ConnectionPointer makeConnection(boost::asio::io_service& io_service)
    {
        return boost::make_shared<Connection>(boost::ref(io_service));
    }


} // namespace virvo


#endif // !VV_PRIVATE_CONNECTION_H
