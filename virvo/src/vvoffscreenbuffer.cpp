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
  const vvGLTools::Viewport v = vvGLTools::getViewport();
  init(v[2], v[3], scale, precision);
}

vvOffscreenBuffer::vvOffscreenBuffer(const int w, const int h, const float scale, const BufferPrecision precision)
  : vvRenderTarget()
{
  init(w, h, scale, precision);
}

vvOffscreenBuffer::~vvOffscreenBuffer()
{
  delete[] _pixels;
  delete _scaledDepthBuffer;
  delete[] _depthPixelsF;
  delete[] _depthPixelsNV;
  freeGLResources();
}

void vvOffscreenBuffer::initForRender()
{
  const vvGLTools::Viewport v = vvGLTools::getViewport();

  if (_preserveDepthBuffer)
  {
    storeDepthBuffer();
  }

  resize(v.values[2], v.values[3]);

  glPushAttrib(GL_VIEWPORT_BIT);

  // If width and height haven't changed, resize will return immediatly.
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _frameBufferObject);
  glViewport(0, 0, _bufferWidth, _bufferHeight);
  glBindTexture(GL_TEXTURE_2D, 0);
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

  if (!_initialized)
  {
    initFbo();
  }
  else
  {
    genColorAndDepthTextures();
  }

  _updatePosted = false;
}

void vvOffscreenBuffer::clearBuffer()
{
  glClearColor(0.0, 0.0, 0.0, 0.0);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

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

void vvOffscreenBuffer::setUseNVDepthStencil(const bool useNVDepthStencil)
{
  _useNVDepthStencil = useNVDepthStencil;
}

void vvOffscreenBuffer::setPrecision(const BufferPrecision& precision)
{
  _precision = precision;
  update();
}

void vvOffscreenBuffer::setInterpolation(const bool interpolation)
{
  _interpolation = interpolation;
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

bool vvOffscreenBuffer::getUseNVDepthStencil() const
{
  return _useNVDepthStencil;
}

BufferPrecision vvOffscreenBuffer::getPrecision() const
{
  return _precision;
}

bool vvOffscreenBuffer::getInterpolation() const
{
  return _interpolation;
}

void vvOffscreenBuffer::init(const int w, const int h,
                             const float scale, const BufferPrecision precision)
{
  glewInit();
  _type = VV_OFFSCREEN_BUFFER;
  _viewportWidth = w;
  _viewportHeight = h;
  _scale = scale;
  _preserveDepthBuffer = false;
  _useNVDepthStencil = false;
  _precision = precision;
  _interpolation = true;
  glGenTextures(1, &_textureId);
  glGenTextures(1, &_depthTextureId);
  _initialized = false;
  _updatePosted = true;
  _pixels = NULL;
  _scaledDepthBuffer = NULL;
  _depthPixelsF = NULL;
  _depthPixelsNV = NULL;
  resize(_viewportWidth, _viewportHeight);
}

void vvOffscreenBuffer::initFbo()
{
  freeGLResources();

  glGenFramebuffersEXT(1, &_frameBufferObject);
  genColorAndDepthTextures();

  _initialized = true;
}

void vvOffscreenBuffer::genColorAndDepthTextures()
{
  glDeleteRenderbuffersEXT(1, &_depthBuffer);
  glDeleteTextures(1, &_colorBuffer);

  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _frameBufferObject);
  glGenRenderbuffersEXT(1, &_depthBuffer);
  glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, _depthBuffer);
  glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT32, _bufferWidth, _bufferHeight);
  glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,
                               GL_RENDERBUFFER_EXT, _depthBuffer);

  glGenTextures(1, &_colorBuffer);
  glBindTexture(GL_TEXTURE_2D, _colorBuffer);
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
                            _colorBuffer, 0);

  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
}

void vvOffscreenBuffer::freeGLResources() const
{
  glDeleteTextures(1, &_colorBuffer);
  glDeleteRenderbuffersEXT(1, &_depthBuffer);
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
  glDeleteFramebuffersEXT(1, &_frameBufferObject);
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

  delete _scaledDepthBuffer;
  _scaledDepthBuffer = new vvOffscreenBuffer(_bufferWidth, _bufferHeight, 1.0f, _precision);

  glBindFramebufferEXT(GL_READ_FRAMEBUFFER_EXT, 0);
  glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER_EXT, _scaledDepthBuffer->_frameBufferObject);
  glBlitFramebufferEXT(0, 0, _viewportWidth, _viewportHeight,
                       0, 0, _scaledDepthBuffer->_bufferWidth, _scaledDepthBuffer->_bufferHeight,
                       GL_DEPTH_BUFFER_BIT, GL_NEAREST);
  _scaledDepthBuffer->bindFramebuffer();

  delete[] _depthPixelsNV;
  delete[] _depthPixelsF;

  if (_useNVDepthStencil)
  {
    _depthPixelsNV = new GLuint[_bufferWidth * _bufferHeight];
    glReadPixels(0, 0, _bufferWidth, _bufferHeight, GL_DEPTH_STENCIL_NV, GL_UNSIGNED_INT_24_8_NV, _depthPixelsNV);
  }
  else
  {
    _depthPixelsF = new float[_bufferWidth * _bufferHeight];
    glReadPixels(0, 0, _bufferWidth, _bufferHeight, GL_DEPTH_COMPONENT, GL_FLOAT, _depthPixelsF);
  }
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
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, (_interpolation) ? GL_LINEAR : GL_NEAREST);
  glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, (_interpolation) ? GL_LINEAR : GL_NEAREST);
  glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _bufferWidth, _bufferHeight,
               0, GL_RGBA, GL_UNSIGNED_BYTE, _pixels);
  vvGLTools::drawViewAlignedQuad();
  glDeleteTextures(1, &_textureId);

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

  if (_useNVDepthStencil)
  {
    glDrawPixels(_bufferWidth, _bufferHeight, GL_DEPTH_STENCIL_NV, GL_UNSIGNED_INT_24_8_NV, _depthPixelsNV);
  }
  else
  {
    glDrawPixels(_bufferWidth, _bufferHeight, GL_DEPTH_COMPONENT, GL_FLOAT, _depthPixelsF);
  }

  glPopClientAttrib();
  glPopAttrib();
}
