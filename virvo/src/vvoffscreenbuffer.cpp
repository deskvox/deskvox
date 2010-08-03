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
  glewInit();
  _type = VV_OFFSCREEN_BUFFER;
  const vvGLTools::Viewport v = vvGLTools::getViewport();
  _viewportWidth = v.values[2];
  _viewportHeight = v.values[3];
  _scale = scale;
  _preserveDepthBuffer = false;
  _precision = precision;
  glGenTextures(1, &_textureId);
  _updatePosted = true;
  _pixels = NULL;
  _depthPixels = NULL;
  resize(_viewportWidth, _viewportHeight);
}

vvOffscreenBuffer::~vvOffscreenBuffer()
{
  delete[] _pixels;
  delete[] _depthPixels;
}

void vvOffscreenBuffer::initForRender()
{
  if (_preserveDepthBuffer)
  {
    storeDepthBuffer();
  }

  const vvGLTools::Viewport v = vvGLTools::getViewport();
  resize(v.values[2], v.values[3]);

  glPushAttrib(GL_VIEWPORT_BIT);

  // If width and height haven't changed, resize will return immediatly.
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _frameBufferObject);
  glViewport(0, 0, _bufferWidth, _bufferHeight);
}

void vvOffscreenBuffer::writeBack(const int w, const int h)
{
  if (_preserveDepthBuffer)
  {
    storeColorBuffer();
  }

  vvGLTools::Viewport viewport;
  if ((w == -1) || (h == -1))
  {
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

    glPopAttrib();

    glGetIntegerv(GL_VIEWPORT, viewport.values);
  }
  else
  {
    viewport.values[2] = w;
    viewport.values[3] = h;
  }

  if (_preserveDepthBuffer)
  {
    renderToViewAlignedQuad();
  }
  else
  {
    glBindFramebufferEXT(GL_READ_FRAMEBUFFER_EXT, _frameBufferObject);
    glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER_EXT, 0);
    glBlitFramebufferEXT(0, 0, _bufferWidth, _bufferHeight,
                         0, 0, viewport[2], viewport[3],
                         GL_COLOR_BUFFER_BIT, GL_LINEAR);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
  }
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
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT);

  if (_preserveDepthBuffer)
  {
    writeBackDepthBuffer();
  }
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

void vvOffscreenBuffer::setPreserveDepthBuffer(const bool preserveDepthBuffer)
{
  _preserveDepthBuffer = preserveDepthBuffer;
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

bool vvOffscreenBuffer::getPreserveFramebuffer() const
{
  return _preserveDepthBuffer;
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

void vvOffscreenBuffer::storeColorBuffer()
{
  delete[] _pixels;
  _pixels = new unsigned char[_bufferWidth * _bufferHeight * 4];
  glReadPixels(0, 0, _bufferWidth, _bufferHeight, GL_RGBA, GL_UNSIGNED_BYTE, _pixels);
}

void vvOffscreenBuffer::storeDepthBuffer()
{
  glFinish();

  delete[] _depthPixels;
  _depthPixels = new float[_bufferWidth * _bufferHeight];
  glReadPixels(0, 0, _bufferWidth, _bufferHeight, GL_DEPTH_COMPONENT, GL_FLOAT, _depthPixels);
}

void vvOffscreenBuffer::renderToViewAlignedQuad() const
{
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();

  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  glPushAttrib(GL_COLOR_BUFFER_BIT | GL_CURRENT_BIT | GL_DEPTH_BUFFER_BIT
               | GL_ENABLE_BIT | GL_TEXTURE_BIT | GL_TRANSFORM_BIT);

  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_COLOR_MATERIAL);
  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

  glEnable(GL_TEXTURE_2D);
  bindTexture();
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _bufferWidth, _bufferHeight,
               0, GL_RGBA, GL_UNSIGNED_BYTE, _pixels);
  vvGLTools::drawViewAlignedQuad();

  glPopAttrib();

  glPopMatrix();
  glMatrixMode(GL_PROJECTION);
  glPopMatrix();
}

void vvOffscreenBuffer::writeBackDepthBuffer() const
{
  glPushAttrib(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  glPushClientAttrib(GL_CLIENT_PIXEL_STORE_BIT);

  glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
  glDepthMask(GL_TRUE);
  glDepthFunc(GL_ALWAYS);
  glEnable(GL_DEPTH_TEST);
  glWindowPos2i(0, 0);
  glDrawPixels(_bufferWidth, _bufferHeight, GL_DEPTH_COMPONENT, GL_FLOAT, _depthPixels);

  glPopClientAttrib();
  glPopAttrib();
}
