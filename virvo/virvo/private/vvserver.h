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


// Boost.ASIO needs _WIN32_WINNT
#ifdef _WIN32
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0501 // Require Windows XP or later
#endif
#endif


#include "vvmessage.h"

#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/tcp.hpp>

#include <set>


namespace virvo
{


    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------


    class Server;
    class ServerManager;


    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------


    class ServerManager
    {
    public:
        class Connection : public boost::enable_shared_from_this<Connection>
        {
            friend class ServerManager;

            explicit Connection(boost::asio::io_service& io_service, ServerManager* manager);

            // The underlying socket
            boost::asio::ip::tcp::socket socket_;
            // The server managing this connection
            ServerManager* manager_;
            // The message handler
            boost::shared_ptr<Server> server_;

        public:
            VVAPI ~Connection();

            // Called in ServerManager::on_accept to accept the connection.
            VVAPI bool accept(Server* server);

            // Called in ServerManager::on_accept to accept the connection.
            VVAPI bool accept(boost::shared_ptr<Server> server);

            // Sends a message to the client
            VVAPI void write(MessagePointer message);
        };

        typedef boost::shared_ptr<Connection> ConnectionPointer;

    public:
        // Constructor starts a new accept operation.
        VVAPI ServerManager(unsigned short port);

        // Destructor.
        VVAPI virtual ~ServerManager();

        // Runs the message loop
        VVAPI void run();

        // Stops the message loop
        VVAPI void stop();

        // Sends a message to the given client
        VVAPI void write(MessagePointer message, ConnectionPointer conn);

        // Sends a message to all currently connected clients
        VVAPI void broadcast(MessagePointer message);

        // Called when a new connection is accepted.
        // Return false to discard the new connection, return conn->accept(...)
        // to finally accept the new connection.
        VVAPI virtual bool on_accept(ConnectionPointer conn, boost::system::error_code const& e) = 0;

    private:
        // Start an accept operation
        void do_accept();

        // Handle completion of a accept operation.
        void handle_accept(boost::system::error_code const& e, ConnectionPointer conn);

        // Read the next message from the given client.
        void do_read(ConnectionPointer conn);

        // Called when a message header is read.
        void handle_read_header(boost::system::error_code const& e, MessagePointer message, ConnectionPointer conn);

        // Called when a complete message is read.
        void handle_read_data(boost::system::error_code const& e, MessagePointer message, ConnectionPointer conn);

        // Starts a new write operation.
        void do_write(MessagePointer message, ConnectionPointer conn);

        // Starts a new write operation an all currently active connections.
        void do_broadcast(MessagePointer message);

        // Called when a complete message is written.
        void handle_write(boost::system::error_code const& e, MessagePointer message, ConnectionPointer conn);

    private:
        typedef std::set<ConnectionPointer> Connections;

        // The IO service
        boost::asio::io_service io_service_;
        // The acceptor object used to accept incoming socket connections.
        boost::asio::ip::tcp::acceptor acceptor_;
        // The connection to the client.
        Connections connections_;
    };


    //----------------------------------------------------------------------------------------------
    //
    //----------------------------------------------------------------------------------------------


    class Server
    {
    public:
        virtual ~Server()
        {
        }

        // Called when an error occurred during a read or a write operation.
        VVAPI virtual void on_error(ServerManager::ConnectionPointer conn, boost::system::error_code const& e);

        // Called when a new message has successfully been read from the server.
        virtual void on_read(ServerManager::ConnectionPointer conn, MessagePointer message) = 0;

        // Called when a message has successfully been written to the server.
        virtual void on_write(ServerManager::ConnectionPointer conn, MessagePointer message) = 0;
    };


} // namespace virvo


#endif // !VV_PRIVATE_SERVER_H
