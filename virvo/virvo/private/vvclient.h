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


#include "vvconnection.h"

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
        // Constructor starts the asynchronous connect operation.
        VVAPI explicit Client(boost::asio::io_service& io_service, std::string const& host, unsigned short port);

        // Sends a message to the server.
        // If the server replies, the given handler is executed (might be null!)
        VVAPI void write(MessagePointer message, MessageHandler handler = MessageHandler());

    private:
        // Handle completion of a connect operation.
        void handle_connect(boost::system::error_code const& e);

        // Handle completion of a read operation.
        void handle_read(MessagePointer message);

        // Read the next message
        void read_next();

    private:
        typedef std::map<unsigned/*ID*/, MessageHandler> Handlers;

        // The connection to the server
        Connection connection_;
        // List of callbacks
        Handlers handlers_;
    };


} // namespace virvo


#endif // !VV_PRIVATE_CLIENT_H
