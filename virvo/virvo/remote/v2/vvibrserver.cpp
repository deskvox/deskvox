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

#include "vvibrserver.h"

#include "private/vvgltools.h"
#include "private/vvibrimage.h"
#include "vvibr.h"
#include "vvrenderer.h"
#include "vvvoldesc.h"

#include <cassert>

vvIbrServer::vvIbrServer()
    : vvRemoteServer()
{
}

vvIbrServer::~vvIbrServer()
{
}

void vvIbrServer::renderImage(ConnectionPtr conn, MessagePtr message,
    vvMatrix const& pr, vvMatrix const& mv, vvRenderer* renderer)
{
    assert(renderer->getRenderTarget()->width() > 0);
    assert(renderer->getRenderTarget()->height() > 0);

    // Update matrices
    vvGLTools::setProjectionMatrix(pr);
    vvGLTools::setModelviewMatrix(mv);

    // Render volume:
    renderer->beginFrame(virvo::CLEAR_COLOR | virvo::CLEAR_DEPTH);
    renderer->renderVolumeGL();
    renderer->endFrame();

    // Compute depth range
    vvAABB aabb = vvAABB(vvVector3(), vvVector3());

    renderer->getVolDesc()->getBoundingBox(aabb);

    float drMin = 0.0f;
    float drMax = 0.0f;

    vvIbr::calcDepthRange(pr, mv, aabb, drMin, drMax);

    renderer->setParameter(vvRenderer::VV_IBR_DEPTH_RANGE, vvVector2(drMin, drMax));

    virvo::RenderTarget* rt = renderer->getRenderTarget();

    int w = rt->width();
    int h = rt->height();

    // Create a new IBR image
    virvo::IbrImage image(w, h, rt->colorFormat(), rt->depthFormat());

    image.setDepthMin(drMin);
    image.setDepthMax(drMax);
    image.setViewMatrix(mv);
    image.setProjMatrix(pr);
    image.setViewport(virvo::Viewport(0, 0, w, h));

    // Fetch rendered image
    if (!rt->downloadColorBuffer(image.colorBuffer().data().ptr(), image.colorBuffer().size()))
    {
        std::cout << "vvIbrServer: download color buffer failed" << std::endl;
        return;
    }
    if (!rt->downloadDepthBuffer(image.depthBuffer().data().ptr(), image.depthBuffer().size()))
    {
        std::cout << "vvIbrServer: download depth buffer failed" << std::endl;
        return;
    }

    // Compress the image
    if (!image.compress(/*virvo::Compress_JPEG*/))
    {
        std::cout << "vvIbrServer::renderImage: failed to compress the image.\n";
        return;
    }

    // Serialize the image
    MessagePtr next = virvo::makeMessage(virvo::Message::IbrImage, image);

    // Send the image
    conn->write(next);
}
