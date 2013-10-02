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


#ifndef VV_PRIVATE_SERVER_H
#define VV_PRIVATE_SERVER_H


#include "vvconnection.h"


namespace virvo
{


    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------


    class Server
        : public boost::enable_shared_from_this<Server>
    {
    public:
        VVAPI explicit Server(boost::asio::io_service& io_service, unsigned short port, MessageHandler handler);

        // Sends a message to the client
        VVAPI void write(MessagePointer message);

    private:
        // Handle completion of a accept operation.
        void handle_accept(boost::system::error_code const& e);

        // Handle completion of a read operation.
        void handle_read(MessagePointer message);

        // Read the next message
        void read_next();

    private:
        // The connection to the client.
        // TODO: Handle multiple clients...
        ConnectionPointer connection_ptr_;
        // The acceptor object used to accept incoming socket connections.
        boost::asio::ip::tcp::acceptor acceptor_;
        // The message handler
        MessageHandler handler_;
    };


} // namespace virvo


#endif // !VV_PRIVATE_SERVER_H
