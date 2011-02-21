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
#include "vvisaclient.h"
#include "vvbsptreevisitors.h"
#include "vvgltools.h"
#include "vvtexrend.h"
#include "float.h"

vvIsaClient::vvIsaClient(std::vector<const char*>& slaveNames, std::vector<int>& slavePorts,
                               std::vector<const char*>& slaveFileNames,
                               const char* fileName)
  : vvRemoteClient(slaveNames, slavePorts, slaveFileNames, fileName)
{
  _threads = NULL;
  _threadData = NULL;
  _visitor = new vvSlaveVisitor();

  _imagespaceApprox = false;

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

vvIsaClient::~vvIsaClient()
{
  destroyThreads();
  delete _visitor;
  glDeleteBuffers(1, &_pointVBO);
  glDeleteBuffers(1, &_colorVBO);
  delete _isaRect[0];
  delete _isaRect[1];
}

vvRemoteClient::ErrorType vvIsaClient::setRenderer(vvRenderer* renderer)
{
  vvTexRend* texRend = dynamic_cast<vvTexRend*>(renderer);
  if (texRend == NULL)
  {
    cerr << "vvIsaClient::setRenderer(): Renderer is no texture based renderer" << endl;
    return vvRemoteClient::VV_BAD_RENDERER_ERROR;
  }

  // This will build up the bsp tree of the master node.
  texRend->prepareDistributedRendering(_slaveNames.size());

  // Store a pointer to the bsp tree and set its visitor.
  _bspTree = texRend->getBspTree();
  _bspTree->setVisitor(_visitor);

  _renderer = texRend;

  // Distribute the bricks from the bsp tree
  std::vector<BrickList>** bricks = texRend->getBrickListsToDistribute();
  for (int s=0; s<_sockets.size(); ++s)
  {
    for (int f=0; f<texRend->getVolDesc()->frames; ++f)
    {
      switch (_sockets[s]->putBricks(bricks[s]->at(f)))
      {
      case vvSocket::VV_OK:
        cerr << "Brick outlines transferred successfully" << endl;
        break;
      default:
        cerr << "Unable to transfer brick outlines" << endl;
        return VV_SOCKET_ERROR;
      }
    }
  }
  return VV_OK;
}

vvRemoteClient::ErrorType vvIsaClient::render()
{
  vvTexRend* renderer = dynamic_cast<vvTexRend*>(_renderer);

  if (renderer == NULL)
  {
    cerr << "Renderer is no texture based renderer" << endl;
    return vvRemoteClient::VV_BAD_RENDERER_ERROR;
  }

  if(_slaveRdy == 0)
  {
    renderer->calcProjectedScreenRects();

    if(_imagespaceApprox)
    {
      // save screenrect for later frames
      _isaRect[1]->x = renderer->getVolDesc()->getBoundingBox().getProjectedScreenRect()->x;
      _isaRect[1]->y = renderer->getVolDesc()->getBoundingBox().getProjectedScreenRect()->y;
      _isaRect[1]->height = renderer->getVolDesc()->getBoundingBox().getProjectedScreenRect()->height;
      _isaRect[1]->width = renderer->getVolDesc()->getBoundingBox().getProjectedScreenRect()->width;
    }

    float matrixGL[16];

    vvMatrix pr;
    glGetFloatv(GL_PROJECTION_MATRIX, matrixGL);
    pr.set(matrixGL);

    vvMatrix mv;
    glGetFloatv(GL_MODELVIEW_MATRIX, matrixGL);
    mv.set(matrixGL);

    // Retrieve the eye position for bsp-tree traversal
    renderer->getEyePosition(&_eye);
    vvMatrix invMV(&mv);
    invMV.invert();
    // This is a gl matrix ==> transpose.
    invMV.transpose();
    _eye.multiply(&invMV);

    for (int s=0; s<_sockets.size(); ++s)
    {
      _sockets[s]->putCommReason(vvSocketIO::VV_MATRIX);
      _sockets[s]->putMatrix(&pr);
      _sockets[s]->putMatrix(&mv);
    }

    pthread_mutex_lock(&_slaveMutex);
    _slaveCnt = _sockets.size();
    pthread_mutex_unlock(&_slaveMutex);

    _slaveRdy = 1;
  }

  // check _slaveCnt securely
  pthread_mutex_lock( &_slaveMutex );
  bool slavesFinished = (_slaveCnt == 0);
  pthread_mutex_unlock( &_slaveMutex );

  if(slavesFinished)
  {
    _slaveRdy = 0;

    if(_imagespaceApprox)
    {
      _gapStart = 1;

      // switch to current screen-rect
      _isaRect[0]->x = _isaRect[1]->x;
      _isaRect[0]->y = _isaRect[1]->y;
      _isaRect[0]->height = _isaRect[1]->height;
      _isaRect[0]->width = _isaRect[1]->width;
    }

    glDrawBuffer(GL_BACK);
    glClearColor(_bgColor[0], _bgColor[1], _bgColor[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // Orthographic projection.
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();vvGLTools::printGLError("push pr");
    glLoadIdentity();

    // Fix the proxy quad for the frame buffer texture.
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();vvGLTools::printGLError("push mv");
    glLoadIdentity();

    // Setup compositing.
    GLboolean lighting;
    glGetBooleanv(GL_LIGHTING, &lighting);
    glDisable(GL_CULL_FACE);
    glDisable(GL_LIGHTING);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_COLOR_MATERIAL);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

    _bspTree->traverse(_eye);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();vvGLTools::printGLError("pop pr");
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();vvGLTools::printGLError("pop mv");

    glFlush();

    glEnable(GL_CULL_FACE);
    if (lighting)
    {
      glEnable(GL_LIGHTING);
    }
    else
    {
      glDisable(GL_LIGHTING);
    }
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_BLEND);

  }
  else if(_imagespaceApprox)
  {
    pthread_mutex_lock( &_slaveMutex );
    if(_threadData[0].images->at(0) == NULL)
    {
      pthread_mutex_unlock( &_slaveMutex );
      return vvRemoteClient::VV_MUTEX_ERROR;
    }

    vvImage2_5d* isaImg = dynamic_cast<vvImage2_5d*>(_threadData[0].images->at(0));

    pthread_mutex_unlock( &_slaveMutex );
    if(!isaImg) return vvRemoteClient::VV_BAD_IMAGE;

    // only calculate points-positions on very first "gap"-frame
    if(_gapStart)
    {
      initIsaFrame();
      _gapStart=false;
    }

    // prepare VBOs
    glBindBuffer(GL_ARRAY_BUFFER, _pointVBO);
    glVertexPointer(3, GL_FLOAT, 0, NULL);
    glEnableClientState(GL_VERTEX_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, _colorVBO);
    glColorPointer(4, GL_UNSIGNED_BYTE, 0, NULL);
    glEnableClientState(GL_COLOR_ARRAY);

    //glClearColor(1,0,0,1); // red for debug
    glClearColor(_bgColor[0], _bgColor[1], _bgColor[2], 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    vvGLTools::Viewport vp = vvGLTools::getViewport();

    glDrawArrays(GL_POINTS, 0, isaImg->getWidth()*isaImg->getHeight()*3);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }
}

void vvIsaClient::initIsaFrame()
{
  vvTexRend* renderer = dynamic_cast<vvTexRend*>(_renderer);
  if (renderer == NULL)
  {
    cerr << "Renderer is no texture based renderer" << endl;
    return;
  }

  vvImage2_5d* isaImg = dynamic_cast<vvImage2_5d*>(_threadData[0].images->at(0));
  if(!isaImg) std::cerr << "error - illegal image pointer" << std::cerr;

  vvGLTools::Viewport vp = vvGLTools::getViewport();

  int h = isaImg->getHeight();
  int w = isaImg->getWidth();

  // Find out Pixel-gaps for drawing
  GLdouble modelMatrix[16];
  glGetDoublev(GL_MODELVIEW_MATRIX,modelMatrix);
  GLdouble projMatrix[16];
  glGetDoublev(GL_PROJECTION_MATRIX,projMatrix);

  // prject center of Volume to window coordinates
  float objPos[3] = {renderer->getVolDesc()->pos[0], renderer->getVolDesc()->pos[1], renderer->getVolDesc()->pos[2]};

  GLdouble winCenter[3];
  gluProject(	objPos[0], objPos[1], objPos[2], modelMatrix, projMatrix, vp.values, &winCenter[0], &winCenter[1], &winCenter[2]);

  // [Note: since the rendered images are mirror-inverted, all following operations will be done upside-down.]

  // Get top-left edge of screenrect
  float winX = _isaRect[0]->x;
  float winY = _isaRect[0]->y;
  const float winZ = winCenter[2];

  double topLeft[3];
  gluUnProject(winX, winY, winZ, modelMatrix,projMatrix, vp.values, &topLeft[0],&topLeft[1],&topLeft[2]);

  // get top-right edge
  winX += w;
  double toRight[3];
  gluUnProject(winX, winY, winZ, modelMatrix,projMatrix, vp.values, &toRight[0],&toRight[1],&toRight[2]);

  winX = _isaRect[0]->x;
  winY += h;

  // get left-down-edge
  double toDown[3];
  gluUnProject(winX,winY,winZ, modelMatrix,projMatrix, vp.values, &toDown[0],&toDown[1],&toDown[2]);

  // write double-vars into our Vector-class for further operations
  vvVector3 right = vvVector3(toRight[0], toRight[1], toRight[2]);
  vvVector3 down = vvVector3(toDown[0], toDown[1], toDown[2]);
  vvVector3 tLeft = vvVector3(topLeft[0], topLeft[1], topLeft[2]);

  right = right - tLeft;
  down = down - tLeft;

  vvVector3 deep = vvVector3(right);
  deep.cross(&down);
  deep.normalize();

  right.scale(float(1./w));
  down.scale(float(1./h));
  deep.scale(right.length());

  // get pixel and depth-data
  uchar* dataRGBA = isaImg->getCodedImage();
  float* depth = isaImg->getpixeldepth();

  uchar* colors = new uchar[w*h*4];
  float* points = new float[w*h*3];

  for(int y = 0; y<h; y++)
  {
    vvVector3 currPos = vvVector3(tLeft);
    currPos += y*down;

    for(int x = 0; x<w; x++)
    {
      int colorIndex = (y*w+x)*4;

      // save color
      colors[y*w*4+x*4]   = dataRGBA[colorIndex];
      colors[y*w*4+x*4+1] = dataRGBA[colorIndex+1];
      colors[y*w*4+x*4+2] = dataRGBA[colorIndex+2];
      colors[y*w*4+x*4+3] = dataRGBA[colorIndex+3];

      // save point-vertex
      float deeper = depth[y*w+x];
      points[y*w*3+x*3]   = currPos[0] + deeper*deep[0];
      points[y*w*3+x*3+1] = currPos[1] + deeper*deep[1];
      points[y*w*3+x*3+2] = currPos[2] + deeper*deep[2];

      currPos += right;
    }
  }

//  +++ DEBUG +++
//  vvToolshed::pixels2Ppm(isaImg->getCodedImage(), isaImg->getWidth(), isaImg->getHeight(), "isaImage.ppm");
//  _isaRect[0]->print();
//  std::cerr << isaImg->getWidth() << " " << isaImg->getHeight() << std::endl;

  // VBO for points
  glBindBuffer(GL_ARRAY_BUFFER, _pointVBO);
  glBufferData(GL_ARRAY_BUFFER, w*h*3*sizeof(float), points, GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  // VBO for colors
  glBindBuffer(GL_ARRAY_BUFFER, _colorVBO);
  glBufferData(GL_ARRAY_BUFFER, w*h*4*sizeof(uchar), colors, GL_STATIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void vvIsaClient::exit()
{
  for (int s=0; s<_sockets.size(); ++s)
  {
    _sockets[s]->putCommReason(vvSocketIO::VV_EXIT);
    delete _sockets[s];
  }
}

void vvIsaClient::resize(const int w, const int h)
{
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_RESIZE) == vvSocket::VV_OK)
    {
      _sockets[s]->putWinDims(w, h);
    }
  }
}

void vvIsaClient::setCurrentFrame(const int index)
{
  vvDebugMsg::msg(3, "vvIsaClient::setCurrentFrame()");
  vvRemoteClient::setCurrentFrame(index);
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_CURRENT_FRAME) == vvSocket::VV_OK)
    {
      _sockets[s]->putInt32(index);
    }
  }
}

void vvIsaClient::setMipMode(const int mipMode)
{
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_MIPMODE) == vvSocket::VV_OK)
    {
      _sockets[s]->putInt32(mipMode);
    }
  }
}

void vvIsaClient::setObjectDirection(const vvVector3* od)
{
  vvDebugMsg::msg(3, "vvIsaClient::setObjectDirection()");
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_OBJECT_DIRECTION) == vvSocket::VV_OK)
    {
      _sockets[s]->putVector3(*od);
    }
  }
}

void vvIsaClient::setViewingDirection(const vvVector3* vd)
{
  vvDebugMsg::msg(3, "vvIsaClient::setViewingDirection()");
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_VIEWING_DIRECTION) == vvSocket::VV_OK)
    {
      _sockets[s]->putVector3(*vd);
    }
  }
}

void vvIsaClient::setPosition(const vvVector3* p)
{
  vvDebugMsg::msg(3, "vvIsaClient::setPosition()");
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_POSITION) == vvSocket::VV_OK)
    {
      _sockets[s]->putVector3(*p);
    }
  }
}

void vvIsaClient::setROIEnable(const bool roiEnabled)
{
  vvDebugMsg::msg(1, "vvIsaClient::setROIEnable()");
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_TOGGLE_ROI) == vvSocket::VV_OK)
    {
      _sockets[s]->putBool(roiEnabled);
    }
  }
}

void vvIsaClient::setProbePosition(const vvVector3* pos)
{
  vvDebugMsg::msg(1, "vvIsaClient::setProbePosition()");
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_ROI_POSITION) == vvSocket::VV_OK)
    {
      _sockets[s]->putVector3(*pos);
    }
  }
}

void vvIsaClient::setProbeSize(const vvVector3* newSize)
{
  vvDebugMsg::msg(1, "vvIsaClient::setProbeSize()");
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_ROI_SIZE) == vvSocket::VV_OK)
    {
      _sockets[s]->putVector3(*newSize);
    }
  }
}

void vvIsaClient::toggleBoundingBox()
{
  vvDebugMsg::msg(3, "vvIsaClient::toggleBoundingBox()");
  for (int s=0; s<_sockets.size(); ++s)
  {
    _sockets[s]->putCommReason(vvSocketIO::VV_TOGGLE_BOUNDINGBOX);
  }
}

void vvIsaClient::updateTransferFunction(vvTransFunc& tf)
{
  vvDebugMsg::msg(1, "vvIsaClient::updateTransferFunction()");
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_TRANSFER_FUNCTION) == vvSocket::VV_OK)
    {
      _sockets[s]->putTransferFunction(tf);
    }
  }
}

void vvIsaClient::setParameter(const vvRenderer::ParameterType param, const float newValue, const char*)
{
  vvDebugMsg::msg(3, "vvIsaClient::setParameter()");
  switch (param)
  {
  case vvRenderer::VV_QUALITY:
    adjustQuality(newValue);
    break;
  case vvRenderer::VV_SLICEINT:
    setInterpolation((newValue != 0.0f));
    break;
  default:
    vvRemoteClient::setParameter(param, newValue);
    break;
  }
}

void vvIsaClient::adjustQuality(const float quality)
{
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_QUALITY) == vvSocket::VV_OK)
    {
      _sockets[s]->putFloat(quality);
    }
  }
}

void vvIsaClient::setInterpolation(const bool interpolation)
{
  vvDebugMsg::msg(3, "vvIsaClient::setInterpolation()");
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_INTERPOLATION) == vvSocket::VV_OK)
    {
      _sockets[s]->putBool(interpolation);
    }
  }
}

void vvIsaClient::setISA(const bool isa)
{
  if(_sockets.size() > 1)
  {
    std::cerr << "immagespace-approximation is available for one slave only!" << std::endl;
    return;
  }
  else
  {
    if (_sockets[0]->putCommReason(vvSocketIO::VV_IMMAGESPACE_APPROX) == vvSocket::VV_OK)
    {
      _imagespaceApprox = isa;
      _sockets[0]->putBool(isa);
    }
  }
}

void vvIsaClient::createThreads()
{
  _visitor->generateTextureIds(_sockets.size());
  _threadData = new ThreadArgs[_sockets.size()];
  _threads = new pthread_t[_sockets.size()];
  pthread_barrier_init(&_barrier, NULL, _sockets.size() + 1);
  for (int s=0; s<_sockets.size(); ++s)
  {
    _threadData[s].threadId = s;
    _threadData[s].renderMaster = this;
    _threadData[s].images = _images;
    pthread_create(&_threads[s], NULL, getImageFromSocket, (void*)&_threadData[s]);
  }

  _visitor->setImages(_images);

  if(_sockets.size()>1)
  {
    std::cerr << "Immagespace-approximation deactivated (more than 1 renderslaves active)" << std::endl;
    _imagespaceApprox = false;
  }
}

void vvIsaClient::destroyThreads()
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

void* vvIsaClient::getImageFromSocket(void* threadargs)
{
  ThreadArgs* data = reinterpret_cast<ThreadArgs*>(threadargs);

  vvSocketIO::CommReason commReason;
  vvSocket::ErrorType err;

  while (1)
  {
    err = data->renderMaster->_sockets.at(data->threadId)->getCommReason(commReason);

    if (err == vvSocket::VV_OK)
    {
      switch (commReason)
      {
      case vvSocketIO::VV_IMAGE:
        {
          std::cerr<<"get regular image"<<std::endl;

          vvImage* img = new vvImage();
          data->renderMaster->_sockets.at(data->threadId)->getImage(img);
          delete data->images->at(data->threadId);
          data->images->at(data->threadId) = img;
        }
        break;

      case vvSocketIO::VV_IMAGE2_5D:
        {
          vvImage2_5d* img = new vvImage2_5d();

          vvSocketIO::ErrorType err = data->renderMaster->_sockets.at(data->threadId)->getImage2_5d(img);
          if(err != vvSocketIO::VV_OK)
            std::cerr << "socket error" <<std::endl;

          // switch pointers securely
          pthread_mutex_lock( &data->renderMaster->_slaveMutex );
          delete data->images->at(data->threadId);
          data->images->at(data->threadId) = img;
          pthread_mutex_unlock( &data->renderMaster->_slaveMutex );
        }
        break;
      default:
        std::cerr<<"getImageFromSocket(): wrong or missing CommReason no. "<<commReason<<std::endl;
        break;
      }
    }
    else
    {
      std::cerr<<"SocketIO error no."<<err<<std::endl;
      break;
    }

    pthread_mutex_lock( &data->renderMaster->_slaveMutex );
    data->renderMaster->_slaveCnt--;
    pthread_mutex_unlock( &data->renderMaster->_slaveMutex );
  }

  pthread_exit(NULL);
}
