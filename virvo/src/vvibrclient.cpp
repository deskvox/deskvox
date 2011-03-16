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

vvIbrClient::vvIbrClient(vvRenderState renderState,
                         std::vector<const char*>& slaveNames, std::vector<int>& slavePorts,
                         std::vector<const char*>& slaveFileNames,
                         const char* fileName,
                         vvImage2_5d::DepthPrecision dp)
  : vvRemoteClient(renderState, slaveNames, slavePorts, slaveFileNames, fileName),
    _depthPrecision(dp)
{
  _threads = NULL;
  _threadData = NULL;

  glewInit();
  glGenBuffers(1, &_pointVBO);
  glGenBuffers(1, &_colorVBO);

  _slaveCnt = 0;
  _slaveRdy = 0;
  _gapStart = 1;
  _isaRect[0] = new vvRect();
  _isaRect[1] = new vvRect();

  pthread_mutex_init(&_slaveMutex, NULL);
}

vvIbrClient::~vvIbrClient()
{
  destroyThreads();
  glDeleteBuffers(1, &_pointVBO);
  glDeleteBuffers(1, &_colorVBO);
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
  vvTexRend* renderer = dynamic_cast<vvTexRend*>(_renderer);
  if (renderer == NULL)
  {
    cerr << "Renderer is no texture based renderer" << endl;
    return vvRemoteClient::VV_WRONG_RENDERER;
  }

  // check _slaveCnt securely
  pthread_mutex_lock( &_slaveMutex );
  bool slavesFinished = (_slaveCnt == 0);
  pthread_mutex_unlock( &_slaveMutex );

  if(slavesFinished)
  {
    // switch to current screen-rect and viewport
    _isaRect[0]->x = _isaRect[1]->x;
    _isaRect[0]->y = _isaRect[1]->y;
    _isaRect[0]->height = _isaRect[1]->height;
    _isaRect[0]->width  = _isaRect[1]->width;
    _vp[0] = _vp[1];

    // save screenrect for later frames
    _isaRect[1]->x = renderer->getVolDesc()->getBoundingBox().getProjectedScreenRect()->x;
    _isaRect[1]->y = renderer->getVolDesc()->getBoundingBox().getProjectedScreenRect()->y;
    _isaRect[1]->height = renderer->getVolDesc()->getBoundingBox().getProjectedScreenRect()->height;
    _isaRect[1]->width = renderer->getVolDesc()->getBoundingBox().getProjectedScreenRect()->width;

    // save Viewport for later frames
    _vp[1] = vvGLTools::getViewport();

    // save position
    _objPos[0] = _objPos[3];
    _objPos[1] = _objPos[4];
    _objPos[2] = _objPos[5];
    _objPos[3] = renderer->getVolDesc()->pos[0];
    _objPos[4] = renderer->getVolDesc()->pos[1];
    _objPos[5] = renderer->getVolDesc()->pos[2];

    // remember MV and PR matrix
    for(int i=0; i<16;i++)
    {
      _modelMatrix[i] = _modelMatrix[i+16];
      _projMatrix[i]  = _projMatrix[i+16];
    }
    glGetDoublev(GL_MODELVIEW_MATRIX,&_modelMatrix[16]);
    glGetDoublev(GL_PROJECTION_MATRIX,&_projMatrix[16]);

    float matrixGL[16];
    vvMatrix pr;
    glGetFloatv(GL_PROJECTION_MATRIX, matrixGL);
    pr.set(matrixGL);

    vvMatrix mv;
    glGetFloatv(GL_MODELVIEW_MATRIX, matrixGL);
    mv.set(matrixGL);

    pthread_mutex_lock(&_slaveMutex);

    for (int s=0; s<_sockets.size(); ++s)
    {
      _sockets[s]->putCommReason(vvSocketIO::VV_MATRIX);
      _sockets[s]->putMatrix(&pr);
      _sockets[s]->putMatrix(&mv);
    }

    _slaveCnt = _sockets.size();  // reset count-down
    initIbrFrame();               // initialize gap-frame

    pthread_mutex_unlock(&_slaveMutex);
  }

  pthread_mutex_lock( &_slaveMutex );
  if(_threadData[0].images->at(0) == NULL)
  {
    pthread_mutex_unlock( &_slaveMutex );
    return vvRemoteClient::VV_BAD_IMAGE;
  }

  vvImage2_5d* isaImg = dynamic_cast<vvImage2_5d*>(_threadData[0].images->at(0));
  if(!isaImg) return vvRemoteClient::VV_BAD_IMAGE;

//  glBlendFunc(GL_ONE, GL_ONE_MINUS_DST_ALPHA);
//  glEnable(GL_BLEND);

  // prepare VBOs
  glBindBuffer(GL_ARRAY_BUFFER, _pointVBO);
  glVertexPointer(3, GL_FLOAT, 0, NULL);
  glEnableClientState(GL_VERTEX_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, _colorVBO);
  glColorPointer(4, GL_UNSIGNED_BYTE, 0, NULL);
  glEnableClientState(GL_COLOR_ARRAY);

  glPointSize(1.5); // Test

  glDrawArrays(GL_POINTS, 0, isaImg->getWidth()*isaImg->getHeight()*3);

  glDisableClientState(GL_VERTEX_ARRAY);
  glDisableClientState(GL_COLOR_ARRAY);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
  pthread_mutex_unlock( &_slaveMutex );

  glFlush();

  return VV_OK;
}

void vvIbrClient::initIbrFrame()
{
  vvImage2_5d* isaImg = dynamic_cast<vvImage2_5d*>(_threadData[0].images->at(0));
  if(!isaImg)
  {
    return;
  }

  int h = isaImg->getHeight();
  int w = isaImg->getWidth();

  // project center of Volume to window coordinates
  GLdouble winCenter[3];
  gluProject(	_objPos[0], _objPos[1], _objPos[2], _modelMatrix, _projMatrix, _vp[0].values, &winCenter[0], &winCenter[1], &winCenter[2]);

  // get pixel and depth-data
  uchar* dataRGBA = isaImg->getCodedImage();
  float depth[w*h];
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

  uchar colors[w*h*4];
  float points[w*h*3];
  double winPoint[3];

  for(int y = 0; y<h; y++)
  {
    for(int x = 0; x<w; x++)
    {
      int colorIndex = (y*w+x)*4;
      // save color
      colors[colorIndex]   = dataRGBA[colorIndex];
      colors[colorIndex+1] = dataRGBA[colorIndex+1];
      colors[colorIndex+2] = dataRGBA[colorIndex+2];
      if(depth[y*w+x] == 0 && depth[y*w+x] == 65535)
        colors[colorIndex+3] = 0;
      else
        colors[colorIndex+3] = dataRGBA[colorIndex+3];

      // save point-vertex
      gluUnProject(_isaRect[0]->x+x, _isaRect[0]->y+y, double(depth[y*w+x]),
                   _modelMatrix, _projMatrix, _vp[0].values,
                   &winPoint[0],&winPoint[1],&winPoint[2]);
      points[y*w*3+x*3]   = winPoint[0];
      points[y*w*3+x*3+1] = winPoint[1];
      points[y*w*3+x*3+2] = winPoint[2];
    }
  }

  // VBO for points
  glBindBuffer(GL_ARRAY_BUFFER, _pointVBO);
  glBufferData(GL_ARRAY_BUFFER, w*h*3*sizeof(float), points, GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // VBO for colors
  glBindBuffer(GL_ARRAY_BUFFER, _colorVBO);
  glBufferData(GL_ARRAY_BUFFER, w*h*4*sizeof(uchar), colors, GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void vvIbrClient::exit()
{
  for (int s=0; s<_sockets.size(); ++s)
  {
    _sockets[s]->putCommReason(vvSocketIO::VV_EXIT);
    delete _sockets[s];
  }
}

void vvIbrClient::createThreads()
{
  _threadData = new ThreadArgs[_sockets.size()];
  _threads = new pthread_t[_sockets.size()];
  for (int s=0; s<_sockets.size(); ++s)
  {
    _threadData[s].threadId = s;
    _threadData[s].renderMaster = this;
    _threadData[s].images = _images;
    pthread_create(&_threads[s], NULL, getImageFromSocket, (void*)&_threadData[s]);
  }

  if(_sockets.size()>1)
  {
    std::cerr << "Immagespace-approximation works with one slave only." << std::endl;
  }
}

void vvIbrClient::destroyThreads()
{
  for (int s=0; s<_sockets.size(); ++s)
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

    // switch pointers securely
    pthread_mutex_lock( &data->renderMaster->_slaveMutex );
    delete data->images->at(data->threadId);
    data->images->at(data->threadId) = img;
    data->renderMaster->_slaveCnt--;
    pthread_mutex_unlock( &data->renderMaster->_slaveMutex );
  }

  pthread_exit(NULL);
}
