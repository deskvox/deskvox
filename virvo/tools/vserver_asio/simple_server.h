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

#ifndef VSERVER_SIMPLE_SERVER_H
#define VSERVER_SIMPLE_SERVER_H

#include "server.h"
#include "message_queue.h"

#include <virvo/vvrenderer.h>

#include <boost/thread/thread.hpp>

#include <memory>

class vvRemoteServer;
class vvRenderContext;
class vvVolDesc;

class vvSimpleServer : public vvServer
{
    typedef vvServer BaseType;

public:
    using BaseType::MessagePointer;
    using BaseType::ConnectionPointer;

public:
    // Constructor.
    vvSimpleServer();

    // Destructor.
    virtual ~vvSimpleServer();

    // Starts processing messages
    void start();

    // Stops processing messages
    void stop();

    // Called when a new message has successfully been read from the server.
    virtual void on_read(ConnectionPointer conn, MessagePointer message);

    // Called when a message has successfully been written to the server.
    virtual void on_write(ConnectionPointer conn, MessagePointer message);

private:
    bool createRenderContext(int w = -1/*use default*/, int h = -1/*use default*/);

    bool createRemoteServer(vvRenderer::RendererType type);

    void processSingleMessage(MessagePointer message);
    void processMessages();

    void processNull(MessagePointer message);
    void processCameraMatrix(MessagePointer message);
    void processCurrentFrame(MessagePointer message);
    void processDisconnect(MessagePointer message);
    void processGpuInfo(MessagePointer message);
    void processObjectDirection(MessagePointer message);
    void processParameter(MessagePointer message);
    void processPosition(MessagePointer message);
    void processRemoteServerType(MessagePointer message);
    void processServerInfo(MessagePointer message);
    void processStatistics(MessagePointer message);
    void processTransFunc(MessagePointer message);
    void processTransFuncChanged(MessagePointer message);
    void processViewingDirection(MessagePointer message);
    void processVolume(MessagePointer message);
    void processVolumeFile(MessagePointer message);
    void processWindowResize(MessagePointer message);

    void handleNewVolume();

private:
    // The message queue
    vvMessageQueue queue_;
    // The thread to process to the message queue
    boost::thread worker_;
    // Whether to stop processing messages
    bool cancel_;
    // The current volume
    std::auto_ptr<vvVolDesc> volume_;
    // The current render context
    std::auto_ptr<vvRenderContext> renderContext_;
    // The current remote server (IBR or Image)
    std::auto_ptr<vvRemoteServer> server_;
    // The current renderer
    std::auto_ptr<vvRenderer> renderer_;
    // The current renderer type
    vvRenderer::RendererType rendererType_;
};

#endif
