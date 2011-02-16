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

#include "vvglew.h"
#include "vvbsptree.h"
#include "vvbsptreevisitors.h"
#include "vvgltools.h"

vvThreadVisitor::vvThreadVisitor()
  : vvVisitor()
{
  _offscreenBuffers = NULL;
  _numOffscreenBuffers = 0;
}

vvThreadVisitor::~vvThreadVisitor()
{
  clearOffscreenBuffers();
}

//----------------------------------------------------------------------------
/** Thread visitor visit method. Supplies logic to render results of worker
    thread.
  @param obj  node to render
*/
void vvThreadVisitor::visit(vvVisitable* obj) const
{
  // This is rather specific: the visitor knows the thread id
  // of the bsp tree node, looks up the appropriate thread
  // and renders its data.

  vvHalfSpace* hs = dynamic_cast<vvHalfSpace*>(obj);
  // Make sure not to recalculate the screen rect, since the
  // modelview and perspective transformations currently applied
  // won't match the one's used for rendering.
  const vvRect* screenRect = hs->getProjectedScreenRect(0, 0, false);

  glActiveTextureARB(GL_TEXTURE0_ARB);
  glEnable(GL_TEXTURE_2D);
  _offscreenBuffers[hs->getId()]->bindTexture();
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16,
               screenRect->width, screenRect->height,
               0, GL_RGBA, GL_FLOAT, _pixels[hs->getId()]);

  // Fix the tex coords to a range of 0.0 to 1.0 and adjust
  // the size of the viewport aligned quad.

  // Transform screen rect from viewport coordinate system to
  // coordinate system ranging from -1 to 1 in x and y direction.
  const float x1 = (static_cast<float>(screenRect->x)
                    / static_cast<float>(_offscreenBuffers[hs->getId()]->getBufferWidth()))
                   * 2.0f - 1.0f;
  const float x2 = (static_cast<float>(screenRect->x + screenRect->width)
                    / static_cast<float>(_offscreenBuffers[hs->getId()]->getBufferWidth()))
                   * 2.0f - 1.0f;
  const float y1 = (static_cast<float>(screenRect->y)
                    / static_cast<float>(_offscreenBuffers[hs->getId()]->getBufferHeight()))
                   * 2.0f - 1.0f;
  const float y2 = (static_cast<float>(screenRect->y + screenRect->height)
                    / static_cast<float>(_offscreenBuffers[hs->getId()]->getBufferHeight()))
                   * 2.0f - 1.0f;

  vvGLTools::drawViewAlignedQuad(x1, y1, x2, y2);
}

void vvThreadVisitor::setOffscreenBuffers(vvOffscreenBuffer** offscreenBuffers,
                                          const int numOffscreenBuffers)
{
  clearOffscreenBuffers();
  _offscreenBuffers = offscreenBuffers;
  _numOffscreenBuffers = numOffscreenBuffers;
}

void vvThreadVisitor::setPixels(GLfloat**& pixels)
{
  _pixels = pixels;
}

void vvThreadVisitor::setWidth(const int width)
{
  _width = width;
}

void vvThreadVisitor::setHeight(const int height)
{
  _height = height;
}

void vvThreadVisitor::clearOffscreenBuffers()
{
  for (int i = 0; i < _numOffscreenBuffers; ++i)
  {
    delete _offscreenBuffers[i];
  }
  delete[] _offscreenBuffers;
}

vvSlaveVisitor::vvSlaveVisitor()
  : vvVisitor()
{
  _textureIds = NULL;
}

vvSlaveVisitor::~vvSlaveVisitor()
{
  if (_images != NULL)
  {
    clearImages();
  }
  delete[] _textureIds;
}

void vvSlaveVisitor::visit(vvVisitable* obj) const
{
  // The relation between halfspace and socket is based
  // upon the visitors knowledge which node it is currently
  // processing.
  vvHalfSpace* hs = dynamic_cast<vvHalfSpace*>(obj);

  const int s = hs->getId();

  vvImage* img = _images->at(s);

  glActiveTextureARB(GL_TEXTURE0_ARB);
  glEnable(GL_TEXTURE_2D);
  glBindTexture(GL_TEXTURE_2D, _textureIds[s]);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

  const vvRect* screenRect = hs->getProjectedScreenRect();
  const vvGLTools::Viewport viewport = vvGLTools::getViewport();

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, screenRect->width, screenRect->height,
               0, GL_RGBA, GL_UNSIGNED_BYTE, img->getCodedImage());

  // See documentation for vvThreadVisitor::visit().
  const float x1 = (static_cast<float>(screenRect->x)
                    / static_cast<float>(viewport[2]))
                   * 2.0f - 1.0f;
  const float x2 = (static_cast<float>(screenRect->x + screenRect->width)
                    / static_cast<float>(viewport[2]))
                   * 2.0f - 1.0f;
  const float y1 = (static_cast<float>(screenRect->y)
                    / static_cast<float>(viewport[3]))
                   * 2.0f - 1.0f;
  const float y2 = (static_cast<float>(screenRect->y + screenRect->height)
                    / static_cast<float>(viewport[3]))
                   * 2.0f - 1.0f;
  vvGLTools::drawViewAlignedQuad(x1, y1, x2, y2);
}

void vvSlaveVisitor::generateTextureIds(const int numImages)
{
  _textureIds = new GLuint[numImages];
  for (int i=0; i<numImages; ++i)
  {
    glGenTextures(1, &_textureIds[i]);
  }
}

void vvSlaveVisitor::setImages(std::vector<vvImage*>* images)
{
  _images = images;
}

void vvSlaveVisitor::clearImages()
{
  for (std::vector<vvImage*>::const_iterator it = _images->begin(); it != _images->end();
       ++it)
  {
    delete (*it);
  }
}
