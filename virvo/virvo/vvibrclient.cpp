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
#include "vvibrclient.h"
#include "vvgltools.h"
#include "vvtexrend.h"
#include "float.h"
#include "vvrayrend.h"
#include "vvshaderfactory.h"
#include "vvshaderprogram.h"
#include "vvtoolshed.h"

using std::cerr;
using std::endl;

vvIbrClient::vvIbrClient(vvVolDesc *vd, vvRenderState renderState,
                         const char* slaveName, int slavePort,
                         const char* slaveFileName)
  : vvRemoteClient(vd, renderState, slaveName, slavePort, slaveFileName)
{
  vvDebugMsg::msg(1, "vvIbrClient::vvIbrClient()");

  rendererType = REMOTE_IBR;

  _thread = NULL;
  _threadData = NULL;

  glewInit();
  glGenBuffers(1, &_pointVBO);
  glGenBuffers(4, _indexBO);

  glGenTextures(1, &_rgbaTex);
  glGenTextures(1, &_depthTex);

  _haveFrame = false; // no rendered frame available
  _newFrame = true; // request a new frame

  _shaderFactory = new vvShaderFactory();
  _shader = _shaderFactory->createProgram("ibr", "", "");

  pthread_mutex_init(&_slaveMutex, NULL);

  createImages();
  createThreads();
}

vvIbrClient::~vvIbrClient()
{
  vvDebugMsg::msg(1, "vvIbrClient::~vvIbrClient()");

  destroyThreads();
  glDeleteBuffers(1, &_pointVBO);
  glDeleteBuffers(4, _indexBO);
  glDeleteTextures(1, &_rgbaTex);
  glDeleteTextures(1, &_depthTex);
  delete _shaderFactory;
  delete _shader;
}

vvRemoteClient::ErrorType vvIbrClient::render()
{
  vvDebugMsg::msg(1, "vvIbrClient::render()");

  pthread_mutex_lock(&_slaveMutex);
  bool haveFrame = _haveFrame;
  bool newFrame = _newFrame;
  pthread_mutex_unlock(&_slaveMutex);

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
    pthread_mutex_lock(&_slaveMutex);
    initIbrFrame();
    pthread_mutex_unlock(&_slaveMutex);
  }

  if (newFrame) // no frame pending
  {
    // request new ibr frame if anything changed
    if (!currentMatrix.equal(&_imgMatrix))
    {
      _changes = true;
    }

    if(_changes)
    {
      pthread_mutex_lock(&_slaveMutex);
      vvRemoteClient::ErrorType err = requestIbrFrame();
      _newFrame = false;
      pthread_mutex_unlock(&_slaveMutex);
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

  _shader->setParameter1f("width", _width);
  _shader->setParameter1f("height", _height);
  _shader->setParameterTex2D("rgbaTex", _rgbaTex);
  _shader->setParameterTex2D("depthTex", _depthTex);

  vvMatrix reprojectionMatrix = _imgMatrix * currentMatrix;
  reprojectionMatrix.transpose();
  float reprojectionMatrixGL[16];
  reprojectionMatrix.get(reprojectionMatrixGL);
  _shader->setParameterMatrix4f("reprojectionMatrix" , reprojectionMatrixGL);

  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
  glDrawElements(GL_POINTS, _width*_height, GL_UNSIGNED_INT, NULL);

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

  _ibrImg = dynamic_cast<vvIbrImage*>(_threadData->images->at(0));
  if(!_ibrImg)
    return;

  const int h = _ibrImg->getHeight();
  const int w = _ibrImg->getWidth();
  _imgMatrix = _ibrImg->getReprojectionMatrix();

  // get pixel and depth-data
  glBindTexture(GL_TEXTURE_2D, _rgbaTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  uchar* dataRGBA = _ibrImg->getImagePtr();
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, _ibrImg->getImagePtr());

  glBindTexture(GL_TEXTURE_2D, _depthTex);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  uchar* depth = _ibrImg->getPixelDepth();
  switch(_ibrImg->getDepthPrecision())
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

  if(_width == w && _height == h)
      return;

  _width = w;
  _height = h;

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

  const int width = _width;
  const int height = _height;

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
  vvMatrix oldMatrix = _ibrImg->getReprojectionMatrix();
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

  if ((normal[0] < normal[1]) && (normal[0] < 0.0f))
  {
    return VV_TOP_LEFT;
  }
  else if ((normal[0] >= normal[1]) && (normal[0] >= 0.0f))
  {
    return VV_TOP_RIGHT;
  }
  else if ((normal[1] < normal[0]) && (normal[1] < 0.0f))
  {
    return VV_BOTTOM_RIGHT;
  }
  else if ((normal[1] >= normal[0]) && (normal[1] >= 0.0f))
  {
    return VV_BOTTOM_LEFT;
  }
  else
  {
    // Arbitrary default.
    return VV_TOP_LEFT;
  }
}

void vvIbrClient::createThreads()
{
  vvDebugMsg::msg(1, "vvIbrClient::createThreads()");

  _threadData = new ThreadArgs;
  _thread = new pthread_t;
  _threadData->renderMaster = this;
  _threadData->images = &_images;
  pthread_create(_thread, NULL, getImageFromSocket, _threadData);
}

void vvIbrClient::destroyThreads()
{
  vvDebugMsg::msg(1, "vvIbrClient::destroyThreads()");

  exit();
  pthread_cancel(*_thread);
  pthread_join(*_thread, NULL);
  delete _thread;
  delete _threadData;
  _thread = NULL;
  _threadData = NULL;
}

void* vvIbrClient::getImageFromSocket(void* threadargs)
{
  vvDebugMsg::msg(1, "vvIbrClient::getImageFromSocket()");

  std::cerr << "Image thread start" << std::endl;

  ThreadArgs* data = static_cast<ThreadArgs*>(threadargs);

  while (1)
  {
    vvIbrImage* img = static_cast<vvIbrImage *>(data->images->at(1));
    vvSocketIO::ErrorType err = data->renderMaster->_socket->getIbrImage(img);
    img->decode();
    if(err != vvSocketIO::VV_OK)
    {
      std::cerr << "vvIbrClient::getImageFromSocket: socket-error (" << err << ") - exiting..." << std::endl;
      break;
    }
#if 0
#ifdef _WIN32
    Sleep(1000);
#else
    usleep(1000000);
#endif
#endif
    // switch pointers securely
    pthread_mutex_lock( &data->renderMaster->_slaveMutex );
    std::swap(data->images->at(0), data->images->at(1));
    data->renderMaster->_newFrame = true;
    data->renderMaster->_haveFrame = true;
    pthread_mutex_unlock( &data->renderMaster->_slaveMutex );
  }
  pthread_exit(NULL);
#ifdef _WIN32
  return NULL;
#endif
}


void vvIbrClient::createImages()
{
  vvDebugMsg::msg(3, "vvIbrClient::createImages()");
  for(int i=0; i<2; ++i)
    _images.push_back(new vvIbrImage);
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
