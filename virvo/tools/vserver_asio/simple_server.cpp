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

#include "simple_server.h"

#include <virvo/gl/util.h>
#include <virvo/private/vvgltools.h>
#include <virvo/private/vvmessage.h>
#include <virvo/private/vvmessages.h>
#include <virvo/private/vvtimer.h>
#include <virvo/vvfileio.h>
#include <virvo/vvibrserver.h>
#include <virvo/vvimageserver.h>
#include <virvo/vvparam.h>
#include <virvo/vvremoteserver.h>
#include <virvo/vvrendercontext.h>
#include <virvo/vvrendererfactory.h>
#include <virvo/vvtfwidget.h>
#include <virvo/vvtransfunc.h>
#include <virvo/vvvoldesc.h>

#include <cassert>
#include <iostream>

#if 1
BOOST_CLASS_EXPORT_IMPLEMENT(vvTFBell)
BOOST_CLASS_EXPORT_IMPLEMENT(vvTFColor)
BOOST_CLASS_EXPORT_IMPLEMENT(vvTFCustom)
BOOST_CLASS_EXPORT_IMPLEMENT(vvTFPyramid)
BOOST_CLASS_EXPORT_IMPLEMENT(vvTFSkip)
#endif

static const int DEFAULT_WINDOW_SIZE = 512;

vvSimpleServer::vvSimpleServer()
    : cancel_(true)
    , rendererType_(vvRenderer::INVALID)
{
    std::cout << "vvSimpleServer::running..." << std::endl;
    start();
}

vvSimpleServer::~vvSimpleServer()
{
    stop();
    std::cout << "vvSimpleServer::stopped." << std::endl;
}

void vvSimpleServer::start()
{
    assert(cancel_ == true && "server already running");

    // Clear the message queue
    queue_.clear();

    // Reset state
    cancel_ = false;

    // Create a new thread
    worker_ = boost::thread(&vvSimpleServer::processMessages, this);
}

void vvSimpleServer::stop()
{
    if (cancel_)
        return;

#ifndef NDEBUG
    std::cout << "SimpleServer::stop: waiting for thread to finish...\n";
#endif

    // Tell the thread to cancel
    cancel_ = true;
    // Wake up the thread in case it's sleeping
    queue_.push_back_always(virvo::makeMessage());

    // Wait for the thread to finish
    worker_.join();

#ifndef NDEBUG
    std::cout << "SimpleServer::stop: complete.\n";
#endif
}

void vvSimpleServer::on_read(ConnectionPointer conn, MessagePointer message)
{
#ifndef NDEBUG
    std::cout << "vvSimpleServer::on_read: " << (int)message->type() << std::endl;
#endif

    switch (message->type())
    {
    case virvo::Message::CameraMatrix:
    case virvo::Message::TransFunc:
    case virvo::Message::TransFuncChanged:
    case virvo::Message::WindowResize:
        queue_.push_back(message);
        break;
    default:
        queue_.push_back_always(message);
        break;
    }
}

void vvSimpleServer::on_write(ConnectionPointer /*conn*/, MessagePointer /*message*/)
{
}

bool vvSimpleServer::createRenderContext(int w, int h)
{
    assert(renderContext_.get() == 0);

#ifndef NDEBUG
    std::cout << "vvSimpleServer::createRenderContext(" << w << ", " << h << ")" << std::endl;
#endif

    // Destroy the old context first - if any.
    renderContext_.reset();

    if (w <= 0) w = DEFAULT_WINDOW_SIZE;
    if (h <= 0) h = DEFAULT_WINDOW_SIZE;

#ifndef NDEBUG
    std::cout << "vvSimpleServer::createRenderContext(" << w << ", " << h << ")" << std::endl;
#endif

    vvContextOptions options;

    options.type        = vvContextOptions::VV_PBUFFER;
    options.width       = w;
    options.height      = h;
    options.displayName = "";

    // Create the new context
    renderContext_.reset(new vvRenderContext(options));

    return renderContext_->makeCurrent();
}

bool vvSimpleServer::createRemoteServer(vvRenderer::RendererType type)
{
#ifndef NDEBUG
    std::cout << "vvSimpleServer::createRemoteServer(" << (int)type << ")" << std::endl;
#endif

    assert(volume_.get());

    // Create a render context if not already done.
    if (renderContext_.get() == 0 && !createRenderContext())
        return false;

    // Create a new remote server
    switch (type)
    {
    case vvRenderer::REMOTE_IBR:
        std::cout << "vvSimpleServer::createRemoteServer: Creating vvIbrServer" << std::endl;
        server_.reset(new vvIbrServer);
        break;
    case vvRenderer::REMOTE_IMAGE:
        std::cout << "vvSimpleServer::createRemoteServer: Creating vvImageServer" << std::endl;
        server_.reset(new vvImageServer);
        break;
    default:
        assert(0 && "unhandled renderer type");
        return false;
    }

    // Install a debug callback to catch GL errors
    virvo::gl::enableDebugCallback();

    vvRenderState state;

    if (volume_->tf.isEmpty()) // Set default color scheme!
    {
        float min = volume_->real[0];
        float max = volume_->real[1];

        volume_->tf.setDefaultAlpha(0, min, max);
        volume_->tf.setDefaultColors(volume_->chan == 1 ? 0 : 2, min, max);
    }

    // Create a new renderer
    renderer_.reset(vvRendererFactory::create(volume_.get(),
                                              state,
                                              type == vvRenderer::REMOTE_IMAGE ? "default" : "rayrendcuda",
                                              vvRendererFactory::Options()));

    if (renderer_.get() == 0)
    {
#ifndef NDEBUG
        std::cout << "vvSimpleServer::createRemoteServer: could not create renderer." << std::endl;
#endif
        return false;
    }

    // Enable IBR
    renderer_->setParameter(vvRenderer::VV_USE_IBR, type == vvRenderer::REMOTE_IBR);
    renderer_->updateTransferFunction();

    return true;
}

void vvSimpleServer::processSingleMessage(MessagePointer message)
{
#ifndef NDEBUG
#define XXX(X)                                                                      \
    case virvo::Message::X: {                                                       \
        std::cout << "vvSimpleServer::processSingleMessage: " << #X << std::endl;   \
        process##X(message);                                                        \
        break;                                                                      \
    }
#else
#define XXX(X)                                                                      \
    case virvo::Message::X: {                                                       \
        process##X(message);                                                        \
        break;                                                                      \
    }
#endif

    switch (message->type())
    {
    XXX( CameraMatrix )
    XXX( CurrentFrame )
    XXX( Disconnect )
    XXX( GpuInfo )
    XXX( ObjectDirection )
    XXX( Parameter )
    XXX( Position )
    XXX( RemoteServerType )
    XXX( ServerInfo )
    XXX( Statistics )
    XXX( TransFunc )
    XXX( TransFuncChanged )
    XXX( ViewingDirection )
    XXX( Volume )
    XXX( VolumeFile )
    XXX( WindowResize )
    default:
        break;
    }

#undef XXX
}

void vvSimpleServer::processMessages()
{
    std::cout << "vvSimpleServer::processMessages..." << std::endl;

    while (!cancel_)
    {
        // Fetch the next from the message queue.
        // If there is no such message, wait for one.
        MessagePointer message;

        queue_.pop_front_blocking(message);

        // Process the message
        processSingleMessage(message);
    }
}

void vvSimpleServer::processNull(MessagePointer /*message*/)
{
}

void vvSimpleServer::processCameraMatrix(MessagePointer message)
{
#ifndef NDEBUG
    std::cout << "Rendering frame..." << std::endl;

    virvo::Timer timer;
#endif

    assert(renderer_.get());

    // Extract the matrices from the message
    virvo::messages::CameraMatrix p = message->deserialize<virvo::messages::CameraMatrix>();

    // Render a new image
    server_->renderImage(conn(), message, p.proj, p.view, renderer_.get());

#ifndef NDEBUG
    std::cout << "Frame complete: " << timer.elapsedSeconds() << " sec" << std::endl;
#endif
}

void vvSimpleServer::processCurrentFrame(MessagePointer message)
{
    assert(renderer_.get());

    // Extract the frame number from the message
    size_t frame = message->deserialize<size_t>();

    // Set the current frame
    renderer_->setCurrentFrame(frame);
}

void vvSimpleServer::processDisconnect(MessagePointer /*message*/)
{
}

void vvSimpleServer::processGpuInfo(MessagePointer /*message*/)
{
}

void vvSimpleServer::processObjectDirection(MessagePointer message)
{
    assert(renderer_.get());

    // Extract the 3D-vector from the message
    // and set the new direction
    renderer_->setObjectDirection(message->deserialize<vvVector3>());
}

void vvSimpleServer::processParameter(MessagePointer message)
{
    assert(renderer_.get());

    // Extract the parameters from the message
    virvo::messages::Param p = message->deserialize<virvo::messages::Param>();

    // Set the new renderer parameter
    renderer_->setParameter(p.name, p.value);
}

void vvSimpleServer::processPosition(MessagePointer message)
{
    assert(renderer_.get());

    // Extract the 3D-vector from the message
    // and set the new direction
    renderer_->setPosition(message->deserialize<vvVector3>());
}

void vvSimpleServer::processRemoteServerType(MessagePointer message)
{
    assert(volume_.get());

    // Get the requested renderer type from the message
    vvRenderer::RendererType type = message->deserialize<vvRenderer::RendererType>();

    // Create the renderer!
    createRemoteServer(type);
}

void vvSimpleServer::processServerInfo(MessagePointer /*message*/)
{
}

void vvSimpleServer::processStatistics(MessagePointer /*message*/)
{
}

void vvSimpleServer::processTransFunc(MessagePointer message)
{
    assert(renderer_.get());

    // Extract the transfer function
    message->deserialize(renderer_->getVolDesc()->tf);
    // Update the transfer function
    renderer_->updateTransferFunction();
}

void vvSimpleServer::processTransFuncChanged(MessagePointer message)
{
    conn()->write(message);
}

void vvSimpleServer::processViewingDirection(MessagePointer message)
{
    assert(renderer_.get());

    // Extract the 3D-vector from the message
    // and set the new direction
    renderer_->setViewingDirection(message->deserialize<vvVector3>());
}

void vvSimpleServer::processVolume(MessagePointer message)
{
    // Create a new volume
    volume_.reset(new vvVolDesc("hello"));

    // Extract the volume from the message
    message->deserialize(*volume_);

    volume_->printInfoLine();

    // Update the volume
    handleNewVolume();
}

void vvSimpleServer::processVolumeFile(MessagePointer message)
{
    // Extract the filename from the message
    std::string filename = message->deserialize<std::string>();

    // Create a new volume description
    volume_.reset(new vvVolDesc(filename.c_str()));

    // Load the volume
    vvFileIO fileIO;

    if (fileIO.loadVolumeData(volume_.get()) == vvFileIO::OK)
    {
        volume_->printInfoLine();
    }
    else
    {
        volume_->printInfoLine(); // volume_.reset();
    }

    // Update the volume
    handleNewVolume();
}

void vvSimpleServer::processWindowResize(MessagePointer message)
{
    assert(renderer_.get());

    // Extract the window size from the message
    virvo::messages::WindowResize p;

    message->deserialize(p);

    // Set the new window size
    renderer_->resize(p.w, p.h);
}

void vvSimpleServer::handleNewVolume()
{
    if (renderer_.get())
    {
        // If a renderer already exists, just update the volume
        renderer_->setVolDesc(volume_.get());
        renderer_->updateTransferFunction();
    }
}
