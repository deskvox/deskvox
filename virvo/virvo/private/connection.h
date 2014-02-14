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

#include "vvmessage.h"

#include <boost/asio/io_service.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/strand.hpp>
#include <boost/smart_ptr/enable_shared_from_this.hpp>

#include <list>

namespace virvo
{

class Connection : public boost::enable_shared_from_this<Connection>
{
public:
    class Handler
    {
    public:
        VVAPI virtual ~Handler();

        // Called when a new connection has been established.
        // Return true to accept the connection, false to discard the connection.
        VVAPI virtual bool on_connect(Connection* conn);

        // Called when a new message has successfully been read from the server.
        VVAPI virtual bool on_read(Connection* conn, MessagePointer message);

        // Called when a message has successfully been written to the server.
        VVAPI virtual bool on_write(Connection* conn, MessagePointer message);

        // Called when an error occurred during a read or a write operation.
        VVAPI virtual bool on_error(Connection* conn, boost::system::error_code const& e);
    };

public:
    // Constructor.
    VVAPI Connection(Handler* handler = 0);

    // Destructor.
    VVAPI ~Connection();

    // Returns the underlying socket
    boost::asio::ip::tcp::socket& socket() {
        return socket_;
    }

    // Returns the underlying socket
    boost::asio::ip::tcp::socket const& socket() const {
        return socket_;
    }

    // Sets a new message handler.
    // The connection must not be 'running'
    VVAPI void set_handler(Handler* handler);

    // Starts the message loop
    VVAPI void run();

    // Starts a new thread which in turn starts the message loop
    VVAPI void run_in_thread();

    // Stops the message loop
    VVAPI void stop();

    // Starts a new connect operation.
    VVAPI void connect(std::string const& host, int port);

    // Disconnect from server
    VVAPI void disconnect();

    // Sends a message to the other connection.
    VVAPI void write(MessagePointer message);

private:
    // Starts a new connect operation.
    void do_connect(std::string const& host, int port);

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

    // Writes the next message
    void do_write();

    // Called when a complete message is written.
    void handle_write(boost::system::error_code const& e, MessagePointer message);

private:
    // Handles messages
    Handler* handler_;
    // The IO service
    boost::asio::io_service io_service_;
    // The underlying socket.
    boost::asio::ip::tcp::socket socket_;
    // To protect the list of messages...
    boost::asio::strand strand_;
    // List of messages to be written
    std::list<MessagePointer> write_queue_;
};

inline boost::shared_ptr<Connection> makeConnection(Connection::Handler* handler = 0)
{
    return boost::make_shared<Connection>(handler);
}

} // namespace virvo

#endif // !VV_PRIVATE_CONNECTION_H
