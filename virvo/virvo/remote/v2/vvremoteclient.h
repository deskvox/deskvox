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

#ifndef VV_REMOTE_CLIENT_H
#define VV_REMOTE_CLIENT_H

#include "private/connection.h"
#include "vvcompiler.h"
#include "vvrenderer.h"

class vvRemoteClient
    : public vvRenderer
    , public virvo::Connection::Handler
{
public:
    typedef virvo::MessagePointer MessagePtr;

public:
    // Constructor
    VVAPI vvRemoteClient(vvVolDesc *vd, vvRenderState renderState, const std::string &filename);

    // Constructor
    VVAPI vvRemoteClient(vvVolDesc *vd, vvRenderState renderState, const std::string &filename,
        boost::shared_ptr<virvo::Connection> conn);

    // Destructor
    VVAPI virtual ~vvRemoteClient();

    // Returns the connection
    boost::shared_ptr<virvo::Connection>& conn() {
        return conn_;
    }

    // Returns the connection
    boost::shared_ptr<virvo::Connection> const& conn() const {
        return conn_;
    }

    // Returns the current projection matrix
    vvMatrix const& proj() const { return proj_; }

    // Returns the current model-view matrix
    vvMatrix const& view() const { return view_; }

    // vvRenderer API ----------------------------------------------------------

    VVAPI virtual bool beginFrame(unsigned clearMask) VV_OVERRIDE;
    VVAPI virtual bool endFrame() VV_OVERRIDE;
    VVAPI virtual bool present() const VV_OVERRIDE;
    VVAPI virtual bool render() = 0;
    VVAPI virtual void renderVolumeGL() VV_OVERRIDE;
    VVAPI virtual bool resize(int w, int h) VV_OVERRIDE;
    VVAPI virtual void setCurrentFrame(size_t index) VV_OVERRIDE;
    VVAPI virtual void setObjectDirection(const vvVector3& od) VV_OVERRIDE;
    VVAPI virtual void setParameter(ParameterType param, const vvParam& value) VV_OVERRIDE;
    VVAPI virtual void setPosition(const vvVector3& p) VV_OVERRIDE;
    VVAPI virtual void setViewingDirection(const vvVector3& vd) VV_OVERRIDE;
    VVAPI virtual void updateTransferFunction() VV_OVERRIDE;

    // virvo::Connection::Handler API ------------------------------------------

    // Called when a new connection has been established.
    // Return true to accept the connection, false to discard the connection.
    VVAPI virtual bool on_connect(virvo::Connection* conn) VV_OVERRIDE;

    // Called when a new message has successfully been read from the server.
    VVAPI virtual bool on_read(virvo::Connection* conn, virvo::MessagePointer message) VV_OVERRIDE;

    // Called when a message has successfully been written to the server.
    VVAPI virtual bool on_write(virvo::Connection* conn, virvo::MessagePointer message) VV_OVERRIDE;

    // Called when an error occurred during a read or a write operation.
    VVAPI virtual bool on_error(virvo::Connection* conn, boost::system::error_code const& e) VV_OVERRIDE;

protected:
    // Sets the handler, connects to the server and starts processing messages
    VVAPI void run(virvo::Connection::Handler* handler, std::string const& host, int port);

protected:
    // The connection to the server
    boost::shared_ptr<virvo::Connection> conn_;
    // Current projection matrix
    vvMatrix proj_;
    // Current modelview matrix
    vvMatrix view_;
};

#endif
