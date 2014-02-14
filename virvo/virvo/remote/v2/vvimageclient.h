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

#ifndef VV_IMAGE_CLIENT_H
#define VV_IMAGE_CLIENT_H

#include "vvremoteclient.h"

class vvRenderer;
class vvVolDesc;

class vvImageClient : public vvRemoteClient
{
public:
    typedef vvRemoteClient::MessagePtr MessagePtr;

public:
    VVAPI vvImageClient(vvVolDesc *vd, vvRenderState renderState,
        std::string const& host, int port, std::string const& filename = "");

    VVAPI vvImageClient(vvVolDesc *vd, vvRenderState renderState,
        boost::shared_ptr<virvo::Connection> conn, std::string const& filename = "");

    VVAPI virtual ~vvImageClient();

    // vvRemoteClient API ------------------------------------------------------

    VVAPI virtual bool render() VV_OVERRIDE;

    VVAPI virtual bool on_connect(virvo::Connection* conn) VV_OVERRIDE;

    VVAPI virtual bool on_read(virvo::Connection* conn, virvo::MessagePointer message) VV_OVERRIDE;

private:
    void init();

    void processImage(virvo::MessagePointer message);

private:
    struct Impl;
    boost::shared_ptr<Impl> impl_;
};

#endif
