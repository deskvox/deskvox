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
                         const char* slaveFileName,
                         const vvImage2_5d::DepthPrecision dp,
                         vvImage2_5d::IbrDepthScale ds)
  : vvRemoteClient(vd, renderState, slaveName, slavePort, slaveFileName),
    _depthPrecision(dp), _depthScale(ds)
{
  vvDebugMsg::msg(1, "vvIbrClient::vvIbrClient()");

  rendererType = REMOTE_IBR;

  _thread = NULL;
  _threadData = NULL;

  glewInit();
  glGenBuffers(1, &_pointVBO);
  glGenBuffers(1, &_indexBO);
  glGenBuffers(1, &_colorVBO);

  glEnable(GL_TEXTURE_2D);
  glGenTextures(3, _ibrTex);

  _isaRect[0] = new vvRect();
  _isaRect[1] = new vvRect();

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
  glDeleteBuffers(1, &_indexBO);
  glDeleteBuffers(1, &_colorVBO);
  glDeleteTextures(3, _ibrTex);
  delete _isaRect[0];
  delete _isaRect[1];
  delete _shaderFactory;
  delete _shader;
}

vvRemoteClient::ErrorType vvIbrClient::render()
{
  // Draw boundary lines
  if (_boundaries)
  {
    const vvVector3 size(vd->getSize()); // volume size [world coordinates]
    drawBoundingBox(&size, &vd->pos, &_boundColor);
  }

  if (_shader == NULL)
  {
    return vvRemoteClient::VV_SHADER_ERROR;
  }

  pthread_mutex_lock(&_slaveMutex);
  bool haveFrame = _haveFrame;
  bool newFrame = newFrame;
  bool remoteRunning = !_newFrame;
  pthread_mutex_unlock(&_slaveMutex);

  vvGLTools::getModelviewMatrix(&_currentMv);
  vvGLTools::getProjectionMatrix(&_currentPr);

  if (!remoteRunning)
  {
    // request new ibr frame if anything changed
    if (!_currentPr.equal(&_imagePr) || !_currentMv.equal(&_imageMv))
    {
      _changes = true;
    }

    if(_changes)
    {
      pthread_mutex_lock(&_slaveMutex);
      vvRemoteClient::ErrorType err = requestIbrFrame();
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

  if(_newFrame)
  {
    pthread_mutex_lock(&_slaveMutex);
    _imageMv.copy(&_requestedMv);
    _imagePr.copy(&_requestedPr);
    initIbrFrame();
    _newFrame = false;
    pthread_mutex_unlock(&_slaveMutex);
  }

  // TODO: don't do this each time... .
  initIndexArrays();

  // Index Buffer Object for points
  const Corner c = getNearestCorner();
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indexBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, _indexArray[c].size() * sizeof(GLuint), &(_indexArray[c])[0], GL_STATIC_DRAW);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  // prepare VBOs
  glBindBuffer(GL_ARRAY_BUFFER, _pointVBO);
  glVertexPointer(3, GL_FLOAT, 0, NULL);
  glEnableClientState(GL_VERTEX_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, _colorVBO);
  glColorPointer(4, GL_UNSIGNED_BYTE, 0, NULL);
  glEnableClientState(GL_COLOR_ARRAY);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, _indexBO);

  _shader->enable();

  vvMatrix mv;
  mv.copy(&_imageMv);
  vvMatrix pr;
  pr.copy(&_imagePr);

  mv.multiplyPost(&pr);
  mv.invert();
  mv.transpose();
  float modelprojectinv[16];
  mv.get(modelprojectinv);
  _shader->setParameterMatrix4f("ModelProjectInv" , modelprojectinv);

  vvGLTools::Viewport tempVP;
  tempVP[0] = _vp[0][0];
  tempVP[1] = _vp[0][1];
  tempVP[2] = _vp[0][2];
  tempVP[3] = _vp[0][3];
  _shader->setParameter4f("vp", tempVP[0], tempVP[1], tempVP[2], tempVP[3]);

  glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
  glDrawElements(GL_POINTS, 0, _width*_height*3, GL_UNSIGNED_INT, NULL);

  _shader->disable();

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glFlush();

  return VV_OK;
}

vvRemoteClient::ErrorType vvIbrClient::requestIbrFrame()
{
  // switch to current screen-rect and viewport
  _isaRect[0]->x = _isaRect[1]->x;
  _isaRect[0]->y = _isaRect[1]->y;
  _isaRect[0]->height = _isaRect[1]->height;
  _isaRect[0]->width  = _isaRect[1]->width;
  _vp[0][0] = _vp[1][0];
  _vp[0][1] = _vp[1][1];
  _vp[0][2] = _vp[1][2];
  _vp[0][3] = _vp[1][3];

  // save screenrect for later frames
  _isaRect[1]->x = getVolDesc()->getBoundingBox().getProjectedScreenRect()->x;
  _isaRect[1]->y = getVolDesc()->getBoundingBox().getProjectedScreenRect()->y;
  _isaRect[1]->height = getVolDesc()->getBoundingBox().getProjectedScreenRect()->height;
  _isaRect[1]->width = getVolDesc()->getBoundingBox().getProjectedScreenRect()->width;

  // save Viewport for later frames
  _vp[1] = vvGLTools::getViewport();

  // remember MV and PR matrix
  _requestedPr.copy(&_currentPr);
  _requestedMv.copy(&_currentMv);

  _ibrPlanes[0] = _ibrPlanes[2];
  _ibrPlanes[1] = _ibrPlanes[3];
  if(_depthScale == vvImage2_5d::VV_SCALED_DEPTH)
  {
    // calculate bounding sphere
    vvAABB bbox(getVolDesc()->getBoundingBox().min(), getVolDesc()->getBoundingBox().max());
    vvVector4 center4(bbox.getCenter()[0], bbox.getCenter()[1], bbox.getCenter()[2], 1.0f);
    vvVector4 min4(bbox.min()[0], bbox.min()[1], bbox.min()[2], 1.0f);
    vvVector4 max4(bbox.max()[0], bbox.max()[1], bbox.max()[2], 1.0f);

    const vvMatrix &pr = _requestedPr;
    const vvMatrix &mv = _requestedMv;

    center4.multiply(&mv);
    min4.multiply(&mv);
    max4.multiply(&mv);

    vvVector3 center(center4[0], center4[1], center4[2]);
    vvVector3 min(min4.e[0], min4.e[1], min4.e[2]);
    vvVector3 max(max4.e[0], max4.e[1], max4.e[2]);

    float radius = (max-min).length() * 0.5f;

    // Depth buffer of ibrPlanes
    vvVector3 scal(center);
    scal.normalize();
    scal.scale(radius);
    min = center - scal;
    max = center + scal;

    min4 = vvVector4(&min, 1.f);
    max4 = vvVector4(&max, 1.f);
    min4.multiply(&pr);
    max4.multiply(&pr);
    min4.perspectiveDivide();
    max4.perspectiveDivide();

    _ibrPlanes[2] = (min4[2]+1.f)/2.f;
    _ibrPlanes[3] = (max4[2]+1.f)/2.f;
  }

  float matrixGL[16];
  vvMatrix pr;
  glGetFloatv(GL_PROJECTION_MATRIX, matrixGL);
  pr.set(matrixGL);

  vvMatrix mv;
  glGetFloatv(GL_MODELVIEW_MATRIX, matrixGL);
  mv.set(matrixGL);

  if(_socket->putCommReason(vvSocketIO::VV_MATRIX) != vvSocket::VV_OK)
      return vvRemoteClient::VV_SOCKET_ERROR;

  if(_socket->putMatrix(&pr) != vvSocket::VV_OK)
      return vvRemoteClient::VV_SOCKET_ERROR;

  if(_socket->putMatrix(&mv) != vvSocket::VV_OK)
      return vvRemoteClient::VV_SOCKET_ERROR;

  return vvRemoteClient::VV_OK;
}

void vvIbrClient::initIbrFrame()
{
  vvImage2_5d* ibrImg = dynamic_cast<vvImage2_5d*>(_threadData->images->at(0));
  if(!ibrImg)
    return;

  int h = ibrImg->getHeight();
  int w = ibrImg->getWidth();
  _width = w;
  _height = h;

  // get pixel and depth-data
  uchar* dataRGBA = ibrImg->getCodedImage();
  std::vector<float> depth(w*h);
  switch(_depthPrecision)
  {
  case vvImage2_5d::VV_UCHAR:
    {
      uchar* d_uchar = ibrImg->getpixeldepthUchar();
      for(int y = 0; y<h; y++)
        for(int x = 0; x<w; x++)
          depth[y*w+x] = float(d_uchar[y*w+x]) / float(UCHAR_MAX);
    }
    break;
  case vvImage2_5d::VV_USHORT:
    {
      ushort* d_ushort = ibrImg->getpixeldepthUshort();
      for(int y = 0; y<h; y++)
        for(int x = 0; x<w; x++)
          depth[y*w+x] = float(d_ushort[y*w+x]) / float(USHRT_MAX);
    }
    break;
  case vvImage2_5d::VV_UINT:
    {
      uint* d_uint = ibrImg->getpixeldepthUint();
      for(int y = 0; y<h; y++)
        for(int x = 0; x<w; x++)
          depth[y*w+x] = float(d_uint[y*w+x]) / float(UINT_MAX);
    }
    break;
  }

  std::vector<GLubyte> colors(w*h*4);
  std::vector<GLfloat> points(w*h*3);

  for(int y = 0; y<h; y++)
  {
    for(int x = 0; x<w; x++)
    {
      int colorIndex = (y*w+x)*4;

      // save color
      colors[colorIndex]   = dataRGBA[colorIndex];
      colors[colorIndex+1] = dataRGBA[colorIndex+1];
      colors[colorIndex+2] = dataRGBA[colorIndex+2];

      if(depth[y*w+x] <= 0.0f || depth[y*w+x] >= 1.0f)
        colors[colorIndex+3] = 0;
      else
        colors[colorIndex+3] = static_cast<GLubyte>(dataRGBA[colorIndex+3]);

      points[y*w*3+x*3]   = x;
      points[y*w*3+x*3+1] = y;

      if(_depthScale == vvImage2_5d::VV_FULL_DEPTH)
      {
        points[y*w*3+x*3+2] = depth[y*w+x];
      }
      else if(_depthScale == vvImage2_5d::VV_SCALED_DEPTH)
      {
        if(depth[y*w+x] == 0.0f || depth[y*w+x] == 1.0f)
        {
          depth[y*w+x] = 1.0f;
        }
        else
        {
          depth[y*w+x] = depth[y*w+x]*(_ibrPlanes[1] - _ibrPlanes[0]) +_ibrPlanes[0];
        }
        points[y*w*3+x*3+2] = depth[y*w+x];
      }
    }
  }

  // VBO for points
  glBindBuffer(GL_ARRAY_BUFFER, _pointVBO);
  glBufferData(GL_ARRAY_BUFFER, points.size() * sizeof(GLfloat), &points[0], GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // VBO for colors
  glBindBuffer(GL_ARRAY_BUFFER, _colorVBO);
  glBufferData(GL_ARRAY_BUFFER, colors.size() * sizeof(GLubyte), &colors[0], GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void vvIbrClient::exit()
{
  _socket->putCommReason(vvSocketIO::VV_EXIT);
  delete _socket;
}

void vvIbrClient::initIndexArrays()
{
  vvImage2_5d* ibrImg = dynamic_cast<vvImage2_5d*>(_threadData->images->at(0));
  if(!ibrImg)
    return;

  const int width = ibrImg->getWidth();
  const int height = ibrImg->getHeight();

  for (int i = 0; i < 4; ++i)
  {
    _indexArray[i].clear();
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
}

vvIbrClient::Corner vvIbrClient::getNearestCorner() const
{
  vvImage2_5d* ibrImg = dynamic_cast<vvImage2_5d*>(_threadData[0].images->at(0));
  if(!ibrImg)
    return VV_NONE;

  vvVector4 screenNormal = vvVector4(0.0f, 0.0f, 1.0f, 1.0f);

  vvMatrix inv = _currentMv;
  inv.invert();
  const vvMatrix m = _oldMv - inv;
  vvVector4 normal = screenNormal;
  normal.multiply(&m);

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
  return VV_NONE;
}

void vvIbrClient::createThreads()
{
  _threadData = new ThreadArgs;
  _thread = new pthread_t;
  _threadData->renderMaster = this;
  _threadData->images = &_images;
  pthread_create(_thread, NULL, getImageFromSocket, _threadData);
}

void vvIbrClient::destroyThreads()
{
  pthread_cancel(*_thread);
  pthread_join(*_thread, NULL);
  delete _thread;
  delete _threadData;
  _thread = NULL;
  _threadData = NULL;
}

void vvIbrClient::setDepthPrecision(vvImage2_5d::DepthPrecision dp)
{
  _depthPrecision = dp;
}

void* vvIbrClient::getImageFromSocket(void* threadargs)
{
  std::cerr << "Image thread start" << std::endl;

  ThreadArgs* data = static_cast<ThreadArgs*>(threadargs);

  while (1)
  {
    vvImage2_5d* img = static_cast<vvImage2_5d *>(data->images->at(1));
    img->setDepthPrecision(data->renderMaster->_depthPrecision);

    vvSocketIO::ErrorType err = data->renderMaster->_socket->getImage2_5d(img);
    if(err != vvSocketIO::VV_OK)
    {
      std::cerr << "vvIbrClient::getImageFromSocket: socket-error (" << err << ") - exiting..." << std::endl;
      break;
    }
#if 0
#ifdef _WIN32
   Sleep(1000);
#else
    sleep(1);
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
    _images.push_back(new vvImage2_5d);
}
