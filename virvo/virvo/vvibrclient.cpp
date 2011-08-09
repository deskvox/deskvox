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
#include "vvtoolshed.h"

using std::cerr;
using std::endl;

vvIbrClient::vvIbrClient(vvRenderState renderState,
                         std::vector<const char*>& slaveNames, std::vector<int>& slavePorts,
                         std::vector<const char*>& slaveFileNames,
                         const char* fileName,
                         const vvImage2_5d::DepthPrecision dp,
                         vvImage2_5d::IbrDepthScale ds)
  : vvRemoteClient(renderState, slaveNames, slavePorts, slaveFileNames, fileName),
    _depthPrecision(dp), _depthScale(ds)
{
  _threads = NULL;
  _threadData = NULL;

  glewInit();
  glGenBuffers(1, &_pointVBO);
  glGenBuffers(1, &_colorVBO);

  glEnable(GL_TEXTURE_2D);
  glGenTextures(3, _ibrTex);

  _slaveCnt = 0;
  _slaveRdy = 0;
  _isaRect[0] = new vvRect();
  _isaRect[1] = new vvRect();

  _firstFrame = false;

  pthread_mutex_init(&_slaveMutex, NULL);
}

vvIbrClient::~vvIbrClient()
{
  destroyThreads();
  glDeleteBuffers(1, &_pointVBO);
  glDeleteBuffers(1, &_colorVBO);
  glDeleteTextures(3, _ibrTex);
  delete _isaRect[0];
  delete _isaRect[1];
}

vvRemoteClient::ErrorType vvIbrClient::setRenderer(vvRenderer* renderer)
{
  vvTexRend* texRend = dynamic_cast<vvTexRend*>(renderer);
  if (texRend == NULL)
  {
    cerr << "vvRenderMaster::setRenderer(): Renderer is no texture based renderer" << endl;
    return vvRemoteClient::VV_WRONG_RENDERER;
  }
  _renderer = texRend;
  return VV_OK;
}

vvRemoteClient::ErrorType vvIbrClient::render()
{
  if (dynamic_cast<vvTexRend*>(_renderer) == NULL)
  {
    cerr << "Renderer is no texture based renderer" << endl;
    return vvRemoteClient::VV_WRONG_RENDERER;
  }

  // check _slaveCnt securely
  pthread_mutex_lock( &_slaveMutex );
  bool slavesFinished = (_slaveCnt == 0);

  if(slavesFinished)
  {
    // don't request new ibr frame if nothing changed
    _changes = false;
    GLdouble tempMM[16];
    GLdouble tempPM[16];
    glGetDoublev(GL_MODELVIEW_MATRIX,&tempMM[0]);
    glGetDoublev(GL_PROJECTION_MATRIX,&tempPM[0]);
    for(int i=0;i<16;i++)
    {
      if(tempMM[i] != _modelMatrix[i] || tempPM[i] != _projMatrix[i] || _changes)
      {
        _changes = true;
        break;
      }
    }

    if(_changes)
    {
      vvRemoteClient::ErrorType err = requestIbrFrame();
      if(err != vvRemoteClient::VV_OK)
        std::cerr << "vvibrClient::requestIbrFrame() - error() " << err << std::endl;
    }

    if(_newFrame)
    {
      initIbrFrame();
      _newFrame = false;
    }
  }

  if(_firstFrame == false)
  {
    // no frame ever rendered yet
    pthread_mutex_unlock( &_slaveMutex );
    return VV_OK;
  }

  if(_threadData[0].images->at(0) == NULL)
  {
    return vvRemoteClient::VV_BAD_IMAGE;
  }

  vvImage2_5d* isaImg = dynamic_cast<vvImage2_5d*>(_threadData[0].images->at(0));
  if(!isaImg)
  {
    return vvRemoteClient::VV_BAD_IMAGE;
  }

  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  //glBlendEquation(GL_MAX);

  // prepare VBOs
  glBindBuffer(GL_ARRAY_BUFFER, _pointVBO);
  glVertexPointer(3, GL_FLOAT, 0, NULL);
  glEnableClientState(GL_VERTEX_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, _colorVBO);
  glColorPointer(4, GL_UNSIGNED_BYTE, 0, NULL);
  glEnableClientState(GL_COLOR_ARRAY);

  glDrawArrays(GL_POINTS, 0, isaImg->getWidth()*isaImg->getHeight()*3);

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, 0);

  pthread_mutex_unlock( &_slaveMutex );

  glFlush();

  return VV_OK;
}

vvRemoteClient::ErrorType vvIbrClient::requestIbrFrame()
{
  vvTexRend* renderer = dynamic_cast<vvTexRend*>(_renderer);

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
  _isaRect[1]->x = renderer->getVolDesc()->getBoundingBox().getProjectedScreenRect()->x;
  _isaRect[1]->y = renderer->getVolDesc()->getBoundingBox().getProjectedScreenRect()->y;
  _isaRect[1]->height = renderer->getVolDesc()->getBoundingBox().getProjectedScreenRect()->height;
  _isaRect[1]->width = renderer->getVolDesc()->getBoundingBox().getProjectedScreenRect()->width;

  // save Viewport for later frames
  _vp[1] = vvGLTools::getViewport();

  // remember MV and PR matrix
  for(int i=0; i<16;i++)
  {
    _modelMatrix[i] = _modelMatrix[i+16];
    _projMatrix[i]  = _projMatrix[i+16];
  }
  glGetDoublev(GL_MODELVIEW_MATRIX,&_modelMatrix[16]);
  glGetDoublev(GL_PROJECTION_MATRIX,&_projMatrix[16]);

  _ibrPlanes[0] = _ibrPlanes[2];
  _ibrPlanes[1] = _ibrPlanes[3];
  if(_depthScale == vvImage2_5d::VV_SCALED_DEPTH)
  {
    // calculate bounding sphere
    vvAABB bbox(renderer->getVolDesc()->getBoundingBox().min(), renderer->getVolDesc()->getBoundingBox().max());
    vvVector4 center4(bbox.getCenter()[0], bbox.getCenter()[1], bbox.getCenter()[2], 1.0f);
    vvVector4 min4(bbox.min()[0], bbox.min()[1], bbox.min()[2], 1.0f);
    vvVector4 max4(bbox.max()[0], bbox.max()[1], bbox.max()[2], 1.0f);

    float matrixGL[16];
    vvMatrix pr;
    glGetFloatv(GL_PROJECTION_MATRIX, matrixGL);
    pr.set(matrixGL);

    vvMatrix mv;
    glGetFloatv(GL_MODELVIEW_MATRIX, matrixGL);
    mv.set(matrixGL);

    mv.transpose();
    pr.transpose();

    center4.multiply(&mv);
    min4.multiply(&mv);
    max4.multiply(&mv);

    vvVector3 center(center4[0], center4[1], center4[2]);
    vvVector3 min(min4.e[0], min4.e[1], min4.e[2]);
    vvVector3 max(max4.e[0], max4.e[1], max4.e[2]);

    float radius = (max-min).length() * 0.5;

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

  for (int s=0; s<int(_sockets.size()); ++s)
  {
    if(_sockets[s]->putCommReason(vvSocketIO::VV_MATRIX) != vvSocket::VV_OK)
      return vvRemoteClient::VV_SOCKET_ERROR;

    if(_sockets[s]->putMatrix(&pr) != vvSocket::VV_OK)
      return vvRemoteClient::VV_SOCKET_ERROR;

    if(_sockets[s]->putMatrix(&mv) != vvSocket::VV_OK)
      return vvRemoteClient::VV_SOCKET_ERROR;
  }
  _slaveCnt = _sockets.size();

  return vvRemoteClient::VV_OK;
}

void vvIbrClient::initIbrFrame()
{
  vvImage2_5d* isaImg = dynamic_cast<vvImage2_5d*>(_threadData[0].images->at(0));
  if(!isaImg)
    return;

  int h = isaImg->getHeight();
  int w = isaImg->getWidth();

  // get pixel and depth-data
  uchar* dataRGBA = isaImg->getCodedImage();
  std::vector<float> depth(w*h);
  switch(_depthPrecision)
  {
  case vvImage2_5d::VV_UCHAR:
    {
      uchar* d_uchar = isaImg->getpixeldepthUchar();
      for(int y = 0; y<h; y++)
        for(int x = 0; x<w; x++)
          depth[y*w+x] = float(d_uchar[y*w+x]) / float(UCHAR_MAX);
    }
    break;
  case vvImage2_5d::VV_USHORT:
    {
      ushort* d_ushort = isaImg->getpixeldepthUshort();
      for(int y = 0; y<h; y++)
        for(int x = 0; x<w; x++)
          depth[y*w+x] = float(d_ushort[y*w+x]) / float(USHRT_MAX);
    }
    break;
  case vvImage2_5d::VV_UINT:
    {
      uint* d_uint = isaImg->getpixeldepthUint();
      for(int y = 0; y<h; y++)
        for(int x = 0; x<w; x++)
          depth[y*w+x] = float(d_uint[y*w+x]) / float(UINT_MAX);
    }
    break;
  }

//  vvToolshed::pixels2Ppm(&depth[0], w, h, "tiefenwerte.ppm", vvToolshed::VV_LUMINANCE);
  std::vector<uchar> colors(w*h*4);
  std::vector<float> points(w*h*3);
  double winPoint[3];

  // barycenter
  float bc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for(int y = 0; y<h; y++)
  {
    for(int x = 0; x<w; x++)
    {
      int colorIndex = (y*w+x)*4;

      //calc barycenter
      bc[0] += float(dataRGBA[colorIndex+3])*x;
      bc[1] += float(dataRGBA[colorIndex+3])*y;

      bc[3] += float(dataRGBA[colorIndex+3]);

      // save color
      colors[colorIndex]   = dataRGBA[colorIndex];
      colors[colorIndex+1] = dataRGBA[colorIndex+1];
      colors[colorIndex+2] = dataRGBA[colorIndex+2];

      if(depth[y*w+x] <= 0.0f || depth[y*w+x] >= 1.0f)
        colors[colorIndex+3] = 0;
      else
        colors[colorIndex+3] = dataRGBA[colorIndex+3];

      // save point-vertex
      if(_depthScale == vvImage2_5d::VV_FULL_DEPTH)
      {
        gluUnProject(_isaRect[0]->x+x, _isaRect[0]->y+y, double(depth[y*w+x]),
                     _modelMatrix, _projMatrix, _vp[0].values,
                     &winPoint[0],&winPoint[1],&winPoint[2]);
        bc[2] += float(dataRGBA[colorIndex+3])*depth[y*w+x];
      }
      else if(_depthScale == vvImage2_5d::VV_SCALED_DEPTH)
      {
        bc[2] += float(dataRGBA[colorIndex+3])*(depth[y*w+x]*(_ibrPlanes[1] - _ibrPlanes[0]) +_ibrPlanes[0]);
        // push away clipped pixels
        if(depth[y*w+x] == 0.0f || depth[y*w+x] == 1.0f)
        {
          depth[y*w+x] = 1.0f;
        }
        else
        {
          depth[y*w+x] = depth[y*w+x]*(_ibrPlanes[1] - _ibrPlanes[0]) +_ibrPlanes[0];
        }

        gluUnProject(x, y, double(depth[y*w+x]),
                     _modelMatrix, _projMatrix, _vp[0].values,
                     &winPoint[0],&winPoint[1],&winPoint[2]);
      }
      points[y*w*3+x*3]   = winPoint[0];
      points[y*w*3+x*3+1] = winPoint[1];
      points[y*w*3+x*3+2] = winPoint[2];
    }
  }

  //bc
  bc[0] /= bc[3];
  bc[1] /= bc[3];
  bc[2] /= bc[3];

  double winBc[3];

  gluUnProject(bc[0], bc[1], bc[2],
               _modelMatrix, _projMatrix, _vp[0].values,
               &winBc[0],&winBc[1],&winBc[2]);

  std::cerr << "barycenter: " << bc[0] <<" "<< bc[1] <<" ("<< bc[2] <<")"<< std::endl;

  glPointSize(5.);
  glBegin(GL_POINTS);
  glColor3f(1.0,0.,0.);
  glVertex3f(winBc[0], winBc[1], winBc[2]);
  glEnd();
  glPointSize(1.);

  // VBO for points
  glBindBuffer(GL_ARRAY_BUFFER, _pointVBO);
  glBufferData(GL_ARRAY_BUFFER, w*h*3*sizeof(float), &points[0], GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // VBO for colors
  glBindBuffer(GL_ARRAY_BUFFER, _colorVBO);
  glBufferData(GL_ARRAY_BUFFER, w*h*4*sizeof(uchar), &colors[0], GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void vvIbrClient::exit()
{
  for (size_t s=0; s<_sockets.size(); ++s)
  {
    _sockets[s]->putCommReason(vvSocketIO::VV_EXIT);
    delete _sockets[s];
  }
}

void vvIbrClient::createThreads()
{
  _threadData = new ThreadArgs[_sockets.size()];
  _threads = new pthread_t[_sockets.size()];
  for (size_t s=0; s<_sockets.size(); ++s)
  {
    _threadData[s].threadId = s;
    _threadData[s].renderMaster = this;
    _threadData[s].images = _images;
    pthread_create(&_threads[s], NULL, getImageFromSocket, (void*)&_threadData[s]);
  }

  if(_sockets.size()>1)
  {
    // current implementaion only
    std::cerr << "Immagespace-approximation works with one slave only." << std::endl;
  }
}

void vvIbrClient::destroyThreads()
{
  for (size_t s=0; s<_sockets.size(); ++s)
  {
    pthread_join(_threads[s], NULL);
  }
  delete[] _threads;
  delete[] _threadData;
  _threads = NULL;
  _threadData = NULL;
}

void vvIbrClient::setDepthPrecision(vvImage2_5d::DepthPrecision dp)
{
  _depthPrecision = dp;
}

void vvIbrClient::setParameter(vvRenderer::ParameterType param, float newValue)
{
  vvRemoteClient::setParameter(param, newValue);
  _changes = true;
}

void vvIbrClient::updateTransferFunction(vvTransFunc& tf)
{
  vvRemoteClient::updateTransferFunction(tf);
  _changes = true;
}

void* vvIbrClient::getImageFromSocket(void* threadargs)
{
  ThreadArgs* data = reinterpret_cast<ThreadArgs*>(threadargs);

  while (1)
  {
    vvImage2_5d* img = new vvImage2_5d();
    img->setDepthPrecision(data->renderMaster->_depthPrecision);

    vvSocketIO::ErrorType err = data->renderMaster->_sockets.at(data->threadId)->getImage2_5d(img);
    if(err != vvSocketIO::VV_OK)
    {
      std::cerr << "vvIbrClient::getImageFromSocket: socket-error (" << err << ") - exiting..." << std::endl;
      break;
    }
    //sleep(3);
    // switch pointers securely
    pthread_mutex_lock( &data->renderMaster->_slaveMutex );
    delete data->images->at(data->threadId);
    data->images->at(data->threadId) = img;
    data->renderMaster->_slaveCnt--;
    data->renderMaster->_newFrame = true;
    if(!data->renderMaster->_firstFrame) data->renderMaster->_firstFrame = true;
    pthread_mutex_unlock( &data->renderMaster->_slaveMutex );
  }
  pthread_exit(NULL);
#ifdef _WIN32
  return NULL;
#endif
}
