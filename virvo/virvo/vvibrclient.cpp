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

#include <limits>

#include "vvglew.h"
#include "vvibr.h"
#include "vvibrclient.h"
#include "vvgltools.h"
#include "vvtexrend.h"
#include "float.h"
#include "vvrayrend.h"
#include "vvshaderfactory.h"
#include "vvshaderprogram.h"
#include "vvtoolshed.h"
#include "vvsocketio.h"
#include "vvdebugmsg.h"
#include "vvvoldesc.h"
#include "vvibrimage.h"

using std::cerr;
using std::endl;

vvIbrClient::vvIbrClient(vvVolDesc *vd, vvRenderState renderState,
                         const char* slaveName, int slavePort,
                         const char* slaveFileName)
  : vvRemoteClient(vd, renderState, vvRenderer::REMOTE_IBR, slaveName, slavePort, slaveFileName)
{
  vvDebugMsg::msg(1, "vvIbrClient::vvIbrClient()");

  rendererType = REMOTE_IBR;

  _thread = NULL;

  glewInit();
  glGenBuffers(1, &_pointVBO);
  glGenBuffers(4, _indexBO);

  glGenTextures(1, &_rgbaTex);
  glGenTextures(1, &_depthTex);

  _haveFrame = false; // no rendered frame available
  _newFrame = true; // request a new frame
  _image = new vvIbrImage;

  _shaderFactory = new vvShaderFactory();
  _shader = _shaderFactory->createProgram("ibr", "", "");
  if(!_shader)
    vvDebugMsg::msg(0, "vvIbrClient::vvIbrClient: could not find ibr shader");

  pthread_mutex_init(&_imageMutex, NULL);
  pthread_mutex_init(&_signalMutex, NULL);
  pthread_cond_init(&_imageCond, NULL);

  createThreads();
}

vvIbrClient::~vvIbrClient()
{
  vvDebugMsg::msg(1, "vvIbrClient::~vvIbrClient()");

  destroyThreads();
  pthread_mutex_destroy(&_imageMutex);
  pthread_mutex_destroy(&_signalMutex);
  pthread_cond_destroy(&_imageCond);
  glDeleteBuffers(1, &_pointVBO);
  glDeleteBuffers(4, _indexBO);
  glDeleteTextures(1, &_rgbaTex);
  glDeleteTextures(1, &_depthTex);
  delete _shaderFactory;
  delete _shader;
  delete _image;
}

vvRemoteClient::ErrorType vvIbrClient::render()
{
  vvDebugMsg::msg(1, "vvIbrClient::render()");

  pthread_mutex_lock(&_signalMutex);
  bool haveFrame = _haveFrame;
  bool newFrame = _newFrame;
  pthread_mutex_unlock(&_signalMutex);

  // Draw boundary lines
  if (_boundaries || !haveFrame)
  {
    const vvVector3 size(vd->getSize()); // volume size [world coordinates]
    drawBoundingBox(&size, &vd->pos, &_boundColor);
  }

  if (_shader == NULL)
  {
    return vvRemoteClient::VV_SHADER_ERROR;
  }

  vvGLTools::getModelviewMatrix(&_currentMv);
  vvGLTools::getProjectionMatrix(&_currentPr);
  vvMatrix currentMatrix = _currentMv * _currentPr;

  if(newFrame && haveFrame)
  {
    pthread_mutex_lock(&_imageMutex);
    initIbrFrame();
    pthread_mutex_unlock(&_imageMutex);
  }

  float drMin = 0.0f;
  float drMax = 0.0f;
  vvIbr::calcDepthRange(_currentPr, _currentMv,
                        vd->getBoundingBox(),
                        drMin, drMax);
  const vvGLTools::Viewport vp = vvGLTools::getViewport();
  vvMatrix currentImgMatrix = vvIbr::calcImgMatrix(_currentPr, _currentMv, vp, drMin, drMax);
  const bool matrixChanged = (!currentImgMatrix.equal(&_imgMatrix));

  if (newFrame) // no frame pending
  {
    _changes = matrixChanged;

    if(_changes)
    {
      pthread_mutex_lock(&_signalMutex);
      vvRemoteClient::ErrorType err = requestIbrFrame();
      pthread_cond_signal(&_imageCond);
      _newFrame = false;
      pthread_mutex_unlock(&_signalMutex);
      _changes = false;
      if(err != vvRemoteClient::VV_OK)
        std::cerr << "vvibrClient::requestIbrFrame() - error() " << err << std::endl;
    }
  }

  if(!haveFrame)
  {
    // no frame was yet received
    return VV_OK;
  }

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // prepare VBOs
  glBindBuffer(GL_ARRAY_BUFFER, _pointVBO);
  glVertexPointer(3, GL_FLOAT, 0, NULL);
  glEnableClientState(GL_VERTEX_ARRAY);

  // Index Buffer Object for points
  const Corner c = getNearestCorner();
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indexBO[c]);

  _shader->enable();

  _shader->setParameter1f("vpWidth", _viewportWidth);
  _shader->setParameter1f("vpHeight", _viewportHeight);
  _shader->setParameter1f("imageWidth", _imgVp[2]);
  _shader->setParameter1f("imageHeight", _imgVp[3]);
  _shader->setParameterTex2D("rgbaTex", _rgbaTex);
  _shader->setParameterTex2D("depthTex", _depthTex);

  vvMatrix reprojectionMatrix;
  if (!matrixChanged)
  {
    reprojectionMatrix = vvIbr::calcDepthScaleMatrix(drMin, drMax) * vvIbr::calcViewportMatrix(vp);
  }
  else
  {
    reprojectionMatrix = _imgMatrix * currentMatrix;
  }
  reprojectionMatrix.transpose();
  float reprojectionMatrixGL[16];
  reprojectionMatrix.get(reprojectionMatrixGL);
  _shader->setParameterMatrix4f("reprojectionMatrix" , reprojectionMatrixGL);

  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
  glDrawElements(GL_POINTS, _imgVp[2]*_imgVp[3], GL_UNSIGNED_INT, NULL);

  _shader->disable();

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  return VV_OK;
}

vvRemoteClient::ErrorType vvIbrClient::requestIbrFrame()
{
  vvDebugMsg::msg(1, "vvIbrClient::requestIbrFrame()");

  if(_socket->putCommReason(vvSocketIO::VV_MATRIX) != vvSocket::VV_OK)
      return vvRemoteClient::VV_SOCKET_ERROR;

  if(_socket->putMatrix(&_currentPr) != vvSocket::VV_OK)
      return vvRemoteClient::VV_SOCKET_ERROR;

  if(_socket->putMatrix(&_currentMv) != vvSocket::VV_OK)
      return vvRemoteClient::VV_SOCKET_ERROR;

  return vvRemoteClient::VV_OK;
}

void vvIbrClient::initIbrFrame()
{
  vvDebugMsg::msg(1, "vvIbrClient::initIbrFrame()");

  const int h = _image->getHeight();
  const int w = _image->getWidth();
  _imgMatrix = _image->getReprojectionMatrix();
  _imgPr = _image->getProjectionMatrix();
  _imgMv = _image->getModelViewMatrix();
  _image->getDepthRange(&_imgDepthRange[0], &_imgDepthRange[1]);

  // get pixel and depth-data
  glBindTexture(GL_TEXTURE_2D, _rgbaTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, _image->getImagePtr());

  glBindTexture(GL_TEXTURE_2D, _depthTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  uchar* depth = _image->getPixelDepth();
  switch(_image->getDepthPrecision())
  {
    case 8:
      glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, w, h, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, depth);
      break;
    case 16:
      glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE16, w, h, 0, GL_LUMINANCE, GL_UNSIGNED_SHORT, depth);
      break;
    case 32:
      glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE32F_ARB, w, h, 0, GL_LUMINANCE, GL_FLOAT, depth);
      break;
  }

  if(_imgVp[2] == w && _imgVp[3] == h)
      return;

  _imgVp = _image->getViewport();

  std::vector<GLfloat> points(w*h*3);

  for(int y = 0; y<h; y++)
  {
    for(int x = 0; x<w; x++)
    {
      points[y*w*3+x*3]   = x;
      points[y*w*3+x*3+1] = y;
      points[y*w*3+x*3+2] = 0.f;
    }
  }

  // VBO for points
  glBindBuffer(GL_ARRAY_BUFFER, _pointVBO);
  glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(GLfloat), &points[0], GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  initIndexArrays();
}

void vvIbrClient::exit()
{
  vvDebugMsg::msg(1, "vvIbrClient::exit()");

  _socket->putCommReason(vvSocketIO::VV_EXIT);
  delete _socket;
}

void vvIbrClient::initIndexArrays()
{
  vvDebugMsg::msg(3, "vvIbrClient::initIndexArray()");

  const int width = _imgVp[2];
  const int height = _imgVp[3];

  for (int i = 0; i < 4; ++i)
  {
    _indexArray[i].clear();
    _indexArray[i].reserve(width*height);
  }

  // Top-left to bottom-right.
  for (int y = 0; y < height; ++y)
  {
    for (int x = 0; x < width; ++x)
    {
      _indexArray[VV_TOP_LEFT].push_back(y * width + x);
    }
  }

  // Top-right to bottom-left.
  for (int y = 0; y < height; ++y)
  {
    for (int x = width - 1; x >= 0; --x)
    {
      _indexArray[VV_TOP_RIGHT].push_back(y * width + x);
    }
  }

  // Bottom-right to top-left.
  for (int y = height - 1; y >= 0; --y)
  {
    for (int x = width - 1; x >= 0; --x)
    {
      _indexArray[VV_BOTTOM_RIGHT].push_back(y * width + x);
    }
  }

  // Bottom-left to top-right.
  for (int y = height - 1; y >= 0; --y)
  {
    for (int x = 0; x < width; ++x)
    {
      _indexArray[VV_BOTTOM_LEFT].push_back(y * width + x);
    }
  }

  for(int i=0; i<4; ++i)
  {
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indexBO[i]);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, _indexArray[i].size() * sizeof(GLuint), &(_indexArray[i])[0], GL_STATIC_DRAW);
  }
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
}

vvIbrClient::Corner vvIbrClient::getNearestCorner() const
{
  vvDebugMsg::msg(3, "vvIbrClient::getNearestCorner()");

  vvVector4 normal = vvVector4(0.0f, 0.0f, 1.0f, 1.0f);

  // Cancel out old matrix from normal.
  vvMatrix oldMatrix = _imgMatrix;
  // The operations below cancel each other out.
  // Left the code this way for higher legibility.
  // Vectors are transformed.
  oldMatrix.invert();
  oldMatrix.transpose();
  oldMatrix.invert();
  normal.multiply(&oldMatrix);

  vvMatrix newMatrix = _currentMv * _currentPr;
  newMatrix.transpose();
  newMatrix.invert();
  normal.multiply(&newMatrix);

  const float x = normal[0]/normal[3], y = normal[1]/normal[3];
  if(y < 0.f)
  {
    if(x < 0.f)
      return VV_TOP_LEFT;
    else
      return VV_TOP_RIGHT;
  }
  else
  {
    if(x < 0.f)
      return VV_BOTTOM_LEFT;
    else
      return VV_BOTTOM_RIGHT;
  }
}

void vvIbrClient::createThreads()
{
  vvDebugMsg::msg(1, "vvIbrClient::createThreads()");

  _thread = new pthread_t;
  pthread_create(_thread, NULL, getImageFromSocket, this);
}

void vvIbrClient::destroyThreads()
{
  vvDebugMsg::msg(1, "vvIbrClient::destroyThreads()");

  exit();
  pthread_cancel(*_thread);
  pthread_join(*_thread, NULL);
  delete _thread;
  _thread = NULL;
}

void* vvIbrClient::getImageFromSocket(void* threadargs)
{
  vvDebugMsg::msg(1, "vvIbrClient::getImageFromSocket()");

  std::cerr << "Image thread start" << std::endl;

  vvIbrClient *ibr = static_cast<vvIbrClient*>(threadargs);
  vvIbrImage* img = ibr->_image;

  while (1)
  {
    pthread_mutex_lock( &ibr->_imageMutex );
    pthread_cond_wait(&ibr->_imageCond, &ibr->_imageMutex);
    vvSocketIO::ErrorType err = ibr->_socket->getIbrImage(img);
    img->decode();
    if(err != vvSocketIO::VV_OK)
    {
      std::cerr << "vvIbrClient::getImageFromSocket: socket-error (" << err << ") - exiting..." << std::endl;
      pthread_mutex_unlock( &ibr->_imageMutex );
      break;
    }
    pthread_mutex_unlock( &ibr->_imageMutex );
    //vvToolshed::sleep(1000);

    pthread_mutex_lock( &ibr->_signalMutex );
    ibr->_newFrame = true;
    ibr->_haveFrame = true;
    pthread_mutex_unlock( &ibr->_signalMutex );
  }
  pthread_exit(NULL);
#ifdef _WIN32
  return NULL;
#endif
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
