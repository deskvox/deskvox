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

#include <GL/glew.h>

#include "vvdebugmsg.h"
#include "vvgltools.h"
#include "vvoffscreenbuffer.h"

struct vvOffscreenBuffer::GLData
{
  GLuint fbo;
  GLuint colorBuffer;
  GLuint depthBuffer;
  GLuint textureId;

  /*!
   * \brief         This pointer is used if the depth buffer is stored as GL_FLOAT.
   */
  GLfloat* depthPixelsF;
  /*!
   * \brief         This pointer is used with the GL_DEPTH_STENCIL_NV for storing the depth buffer.
   */
  GLuint* depthPixelsNV;
};

vvOffscreenBuffer::vvOffscreenBuffer(float scale, virvo::BufferPrecision precision)
  : _gldata(new GLData)
{
  virvo::Viewport v = vvGLTools::getViewport();
  init(v[2], v[3], scale, precision);
}

vvOffscreenBuffer::vvOffscreenBuffer(int w, int h, float scale, virvo::BufferPrecision precision)
  : _gldata(new GLData)
{
  init(w, h, scale, precision);
}

vvOffscreenBuffer::~vvOffscreenBuffer()
{
  delete[] _pixels;
  delete _scaledDepthBuffer;
  delete[] _gldata->depthPixelsF;
  delete[] _gldata->depthPixelsNV;

  glDeleteTextures(1, &_gldata->colorBuffer);
  if (_preserveDepthBuffer)
  {
    glDeleteRenderbuffersEXT(1, &_gldata->depthBuffer);
  }
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
  glDeleteFramebuffersEXT(1, &_gldata->fbo);
  delete _gldata;
}

void vvOffscreenBuffer::bind()
{
  const virvo::Viewport v = vvGLTools::getViewport();

  if (_preserveDepthBuffer)
  {
    storeDepthBuffer(getScaled(v[2]), getScaled(v[3]));
  }

  resize(v[2], v[3]);

  glPushAttrib(GL_VIEWPORT_BIT);

  // If width and height haven't changed, resize will return immediatly.
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _gldata->fbo);
  glViewport(0, 0, _bufferWidth, _bufferHeight);
  glBindTexture(GL_TEXTURE_2D, 0);
}

void vvOffscreenBuffer::unbind()
{
  vvGLTools::printGLError("enter vvOffscreenBuffer::writeBack()");

  if (_preserveDepthBuffer)
  {
    storeColorBuffer();
  }

  virvo::Viewport viewport;
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

  glPopAttrib();

  glGetIntegerv(GL_VIEWPORT, viewport.values);

  if (_preserveDepthBuffer)
  {
    renderToViewAlignedQuad();
  }
  else
  {
    glBindFramebufferEXT(GL_READ_FRAMEBUFFER_EXT, _gldata->fbo);
    glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER_EXT, 0);
    glBlitFramebufferEXT(0, 0, _bufferWidth, _bufferHeight,
                         0, 0, viewport[2], viewport[3],
                         GL_COLOR_BUFFER_BIT, GL_LINEAR);
    glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
  }

  vvGLTools::printGLError("leave vvOffscreenBuffer::writeBack()");
}

void vvOffscreenBuffer::resize(int w, int h)
{
  if (_viewportWidth == w && _viewportHeight == h && !_updatePosted)
  {
    return;
  }

  _viewportWidth = w;
  _viewportHeight = h;

  _bufferWidth = getScaled(_viewportWidth);
  _bufferHeight = getScaled(_viewportHeight);

  glDeleteTextures(1, &_gldata->colorBuffer);

  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _gldata->fbo);
  if (_preserveDepthBuffer)
  {
    glDeleteRenderbuffersEXT(1, &_gldata->depthBuffer);
    glGenRenderbuffersEXT(1, &_gldata->depthBuffer);
    glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, _gldata->depthBuffer);
    glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_DEPTH_COMPONENT32, _bufferWidth, _bufferHeight);
    glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_ATTACHMENT_EXT,
                                 GL_RENDERBUFFER_EXT, _gldata->depthBuffer);
  }

  glGenTextures(1, &_gldata->colorBuffer);
  glBindTexture(GL_TEXTURE_2D, _gldata->colorBuffer);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  switch (_precision)
  {
  case virvo::Byte:
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _bufferWidth, _bufferHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
    break;
  case virvo::Short:
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F_ARB, _bufferWidth, _bufferHeight, 0, GL_RGBA, GL_SHORT, NULL);
    break;
  case virvo::Float:
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA32F_ARB, _bufferWidth, _bufferHeight, 0, GL_RGBA, GL_FLOAT, NULL);
    break;
  }
  glViewport(0, 0, _bufferWidth, _bufferHeight);

  glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D,
                            _gldata->colorBuffer, 0);

  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);
  _updatePosted = false;

  vvGLTools::printGLError("leave vvOffscreenBuffer::resize()");
}

void vvOffscreenBuffer::clear()
{
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  if (_preserveDepthBuffer)
  {
    writeBackDepthBuffer();
  }
}

void vvOffscreenBuffer::bindTexture() const
{
  glBindTexture(GL_TEXTURE_2D, _gldata->textureId);
}

void vvOffscreenBuffer::setScale(float scale)
{
  _scale = scale;
  update();
}

void vvOffscreenBuffer::setPreserveDepthBuffer(bool preserveDepthBuffer)
{
  _preserveDepthBuffer = preserveDepthBuffer;
}

void vvOffscreenBuffer::setUseNVDepthStencil(bool useNVDepthStencil)
{
  _useNVDepthStencil = useNVDepthStencil;
}

void vvOffscreenBuffer::setPrecision(virvo::BufferPrecision precision)
{
  _precision = precision;
  update();
}

void vvOffscreenBuffer::setInterpolation(bool interpolation)
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

virvo::BufferPrecision vvOffscreenBuffer::getPrecision() const
{
  return _precision;
}

bool vvOffscreenBuffer::getInterpolation() const
{
  return _interpolation;
}

void vvOffscreenBuffer::init(int w, int h, float scale, virvo::BufferPrecision precision)
{
  glewInit();
  _viewportWidth = w;
  _viewportHeight = h;
  _scale = scale;
  _preserveDepthBuffer = false;
  _useNVDepthStencil = false;
  _precision = precision;
  _interpolation = true;
  glGenTextures(1, &_gldata->textureId);
  glGenFramebuffersEXT(1, &_gldata->fbo);
  _updatePosted = true;
  _pixels = NULL;
  _scaledDepthBuffer = NULL;
  _gldata->depthPixelsF = NULL;
  _gldata->depthPixelsNV = NULL;
  resize(_viewportWidth, _viewportHeight);
}

int vvOffscreenBuffer::getScaled(int v) const
{
  return (int)((float)v * _scale);
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

void vvOffscreenBuffer::storeDepthBuffer(const int scaledWidth, const int scaledHeight)
{
  glFinish();

  delete _scaledDepthBuffer;
  _scaledDepthBuffer = new vvOffscreenBuffer(scaledWidth, scaledHeight, 1.0f, _precision);

  glBindFramebufferEXT(GL_READ_FRAMEBUFFER_EXT, 0);
  glBindFramebufferEXT(GL_DRAW_FRAMEBUFFER_EXT, _scaledDepthBuffer->_gldata->fbo);
  glBlitFramebufferEXT(0, 0, _viewportWidth, _viewportHeight,
                       0, 0, _scaledDepthBuffer->_bufferWidth, _scaledDepthBuffer->_bufferHeight,
                       GL_DEPTH_BUFFER_BIT, GL_NEAREST);
  _scaledDepthBuffer->bindFramebuffer();

  delete[] _gldata->depthPixelsNV;
  delete[] _gldata->depthPixelsF;

  if (_useNVDepthStencil)
  {
    _gldata->depthPixelsNV = new GLuint[_bufferWidth * _bufferHeight];
    glReadPixels(0, 0, _bufferWidth, _bufferHeight, GL_DEPTH_STENCIL_NV, GL_UNSIGNED_INT_24_8_NV, _gldata->depthPixelsNV);
  }
  else
  {
    _gldata->depthPixelsF = new float[_bufferWidth * _bufferHeight];
    glReadPixels(0, 0, _bufferWidth, _bufferHeight, GL_DEPTH_COMPONENT, GL_FLOAT, _gldata->depthPixelsF);
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
  vvGLTools::drawQuad();
  glDeleteTextures(1, &_gldata->textureId);

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

  if (_useNVDepthStencil)
  {
    glDrawPixels(_bufferWidth, _bufferHeight, GL_DEPTH_STENCIL_NV, GL_UNSIGNED_INT_24_8_NV, _gldata->depthPixelsNV);
  }
  else
  {
    glDrawPixels(_bufferWidth, _bufferHeight, GL_DEPTH_COMPONENT, GL_FLOAT, _gldata->depthPixelsF);
  }

  glPopClientAttrib();
  glPopAttrib();
}

void vvOffscreenBuffer::bindFramebuffer() const
{
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, _gldata->fbo);

  if (glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT) != GL_FRAMEBUFFER_COMPLETE_EXT)
  {
    vvDebugMsg::msg(0, "vvOffscreenBuffer::bindFramebuffer(): Error binding fbo");
  }
  vvGLTools::printGLError("leave vvOffscreenBuffer::bindFramebuffer()");
}

void vvOffscreenBuffer::unbindFramebuffer() const
{
  glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);

  vvGLTools::printGLError("leave vvOffscreenBuffer::unbindFramebuffer()");
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
