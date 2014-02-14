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

#include "vvremoteclient.h"

#include "private/vvgltools.h"
#include "private/vvmessages.h"
#include "vvvoldesc.h"

using virvo::makeMessage;
using virvo::Message;

vvRemoteClient::vvRemoteClient(vvVolDesc *vd, vvRenderState renderState, const std::string& /*filename*/)
    : vvRenderer(vd, renderState)
    , conn_(virvo::makeConnection())
{
}

vvRemoteClient::vvRemoteClient(vvVolDesc *vd, vvRenderState renderState, const std::string& /*filename*/,
        boost::shared_ptr<virvo::Connection> conn)
    : vvRenderer(vd, renderState)
    , conn_(conn)
{
}

vvRemoteClient::~vvRemoteClient()
{
}

bool vvRemoteClient::beginFrame(unsigned /*clearMask*/)
{
    return true;
}

bool vvRemoteClient::endFrame()
{
    return true;
}

void vvRemoteClient::renderVolumeGL()
{
    vvGLTools::getProjectionMatrix(&proj_);
    vvGLTools::getModelviewMatrix(&view_);

    render();
}

bool vvRemoteClient::resize(int w, int h)
{
    virvo::RenderTarget* rt = getRenderTarget();

    if (rt->width() == w && rt->height() == h)
        return true;

    conn_->write(makeMessage(Message::WindowResize, virvo::messages::WindowResize(w, h)));

    return vvRenderer::resize(w, h);
}

bool vvRemoteClient::present() const
{
    return true;
}

void vvRemoteClient::setCurrentFrame(size_t index)
{
    conn_->write(makeMessage(Message::CurrentFrame, index));

    vvRenderer::setCurrentFrame(index);
}

void vvRemoteClient::setObjectDirection(const vvVector3& od)
{
    conn_->write(makeMessage(Message::ObjectDirection, od));

    vvRenderer::setObjectDirection(od);
}

void vvRemoteClient::setViewingDirection(const vvVector3& vd)
{
    conn_->write(makeMessage(Message::ViewingDirection, vd));

    vvRenderer::setViewingDirection(vd);
}

void vvRemoteClient::setPosition(const vvVector3& p)
{
    conn_->write(makeMessage(Message::Position, p));

    vvRenderer::setPosition(p);
}

void vvRemoteClient::updateTransferFunction()
{
#if 1
    conn_->write(makeMessage(Message::TransFuncChanged, true));
#else
    conn_->write(makeMessage(Message::TransFunc, vd->tf));
#endif

    vvRenderer::updateTransferFunction();
}

void vvRemoteClient::setParameter(ParameterType name, const vvParam& value)
{
    vvRenderer::setParameter(name, value);

    conn_->write(makeMessage(Message::Parameter, virvo::messages::Param(name, value)));
}

bool vvRemoteClient::on_connect(virvo::Connection* /*conn*/)
{
    return true;
}

bool vvRemoteClient::on_read(virvo::Connection* conn, virvo::MessagePointer message)
{
    switch (message->type())
    {
    case virvo::Message::TransFuncChanged:
        conn->write(makeMessage(Message::TransFunc, vd->tf));
        break;
    }

    return true;
}

bool vvRemoteClient::on_write(virvo::Connection* /*conn*/, virvo::MessagePointer /*message*/)
{
    return true;
}

bool vvRemoteClient::on_error(virvo::Connection* /*conn*/, boost::system::error_code const& /*e*/)
{
    return true;
}

void vvRemoteClient::run(virvo::Connection::Handler* handler, std::string const& host, int port)
{
    try
    {
        // Set the new handler
        conn_->set_handler(handler);

        // Connect to server
        conn_->connect(host, port);

        // Start processing messages
        conn_->run_in_thread();
    }
    catch (std::exception& /*e*/)
    {
    }
}
