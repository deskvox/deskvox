// Virvo - Virtual Reality Volume Rendering
// Contact: Stefan Zellmann, zellmans@uni-koeln.de
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

// Glew:

// No circular dependencies between gl.h and glew.h
#ifndef GLEW_INCLUDED
#include <GL/glew.h>
#define GLEW_INCLUDED
#endif

#include "vvgltools.h"
#include "vvoffscreenbuffer.h"

vvOffscreenBuffer::vvOffscreenBuffer(const float scale = 1.0f, const BufferPrecision precision = VV_BYTE)
  : vvRenderTarget()
{
#if !defined(__APPLE__)
  glewInit();
#endif
  _type = VV_OFFSCREEN_BUFFER;
  vvGLTools::Viewport v = vvGLTools::getViewport();
  _viewportWidth = v.values[2];
  _viewportHeight = v.values[3];
  _scale = scale;
  _precision = precision;
  glGenTextures(1, &_textureId);
  _updatePosted = true;
  resize(_viewportWidth, _viewportHeight);
}

vvOffscreenBuffer::~vvOffscreenBuffer()
{

}

void vvOffscreenBuffer::initForRender()
{
  vvGLTools::Viewport v = vvGLTools::getViewport();
  resize(v.values[2], v.values[3]);

  glPushAttrib(GL_VIEWPORT_BIT);

  // If width and height haven't changed, resize will return immediatly.
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _frameBufferObject);
  glViewport(0, 0, _bufferWidth, _bufferHeight);
}

void vvOffscreenBuffer::writeBack(const int w, const int h)
{
  GLint viewport[4];
  if ((w == -1) || (h == -1))
  {
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    glPopAttrib();

    glGetIntegerv(GL_VIEWPORT, viewport);
  }
  else
  {
    viewport[2] = w;
    viewport[3] = h;
  }

  glBindFramebufferEXT(GL_READ_FRAMEBUFFER_EXT, _frameBufferObject);
  glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER_EXT, 0);
  glBlitFramebufferEXT(0, 0, _bufferWidth, _bufferHeight,
                       0, 0, viewport[2], viewport[3],
                       GL_COLOR_BUFFER_BIT, GL_LINEAR);
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

void vvOffscreenBuffer::resize(const int w, const int h)
{
  if ((_viewportWidth == w) && (_viewportHeight == h) && (!_updatePosted))
  {
    return;
  }

  _viewportWidth = w;
  _viewportHeight = h;

  doScale();

  GLuint tex;
  glGenFramebuffersEXT(1, &_frameBufferObject);
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _frameBufferObject);
  glGenRenderbuffersEXT(1, &_depthBuffer);
  glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, _depthBuffer);
  glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT32, _bufferWidth, _bufferHeight);
  glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,
                               GL_RENDERBUFFER_EXT, _depthBuffer);
  glGenTextures(1, &tex);
  glBindTexture(GL_TEXTURE_2D, tex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  switch (_precision)
  {
  case VV_BYTE:
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _bufferWidth, _bufferHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    break;
  case VV_SHORT:
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, _bufferWidth, _bufferHeight, 0, GL_RGBA, GL_SHORT, NULL);
    break;
  case VV_FLOAT:
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, _bufferWidth, _bufferHeight, 0, GL_RGBA, GL_FLOAT, NULL);
    break;
  }

  glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D,
                            tex, 0);

  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

  _updatePosted = false;
}

void vvOffscreenBuffer::clearBuffer()
{
  glClear(GL_COLOR_BUFFER_BIT);
}

void vvOffscreenBuffer::bindFramebuffer() const
{
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _frameBufferObject);
}

void vvOffscreenBuffer::unbindFramebuffer() const
{
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

void vvOffscreenBuffer::bindTexture() const
{
  glBindTexture(GL_TEXTURE_2D, _textureId);
}

void vvOffscreenBuffer::setScale(const float scale)
{
  _scale = scale;
  update();
}

void vvOffscreenBuffer::setPrecision(const BufferPrecision& precision)
{
  _precision = precision;
  update();
}

int vvOffscreenBuffer::getBufferWidth() const
{
  return _bufferWidth;
}

int vvOffscreenBuffer::getBufferHeight() const
{
  return _bufferHeight;
}

float vvOffscreenBuffer::getScale() const
{
  return _scale;
}

BufferPrecision vvOffscreenBuffer::getPrecision() const
{
  return _precision;
}

void vvOffscreenBuffer::doScale()
{
  _bufferWidth = (int)((float)_viewportWidth * _scale);
  _bufferHeight = (int)((float)_viewportHeight * _scale);
}

void vvOffscreenBuffer::update()
{
  // After the next render step, the resize() method won't return although size didn't change.
  _updatePosted = true;
}
