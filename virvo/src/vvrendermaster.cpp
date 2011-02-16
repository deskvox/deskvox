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

#include "vvbsptreevisitors.h"
#include "vvgltools.h"
#include "vvrendermaster.h"
#include "vvtexrend.h"

vvRenderMaster::vvRenderMaster(std::vector<const char*>& slaveNames, std::vector<int>& slavePorts,
                               std::vector<const char*>& slaveFileNames,
                               const char* fileName)
  : vvRemoteClient(fileName),
  _slaveNames(slaveNames), _slavePorts(slavePorts),
  _slaveFileNames(slaveFileNames)
{
  _threads = NULL;
  _threadData = NULL;
  _visitor = new vvSlaveVisitor();
}

vvRenderMaster::~vvRenderMaster()
{
  destroyThreads();
  // The visitor will delete the sockets either.
  delete _visitor;
}

vvRemoteClient::ErrorType vvRenderMaster::initSockets(const int defaultPort, vvSocket::SocketType st,
                                                      const bool redistributeVolData,
                                                      vvVolDesc*& vd)
{
  const bool loadVolumeFromFile = !redistributeVolData;
  for (int s=0; s<_slaveNames.size(); ++s)
  {
    if (_slavePorts[s] == -1)
    {
      _sockets.push_back(new vvSocketIO(defaultPort, _slaveNames[s], st));
    }
    else
    {
      _sockets.push_back(new vvSocketIO(_slavePorts[s], _slaveNames[s], st));
    }
    _sockets[s]->set_debuglevel(vvDebugMsg::getDebugLevel());

    if (_sockets[s]->init() == vvSocket::VV_OK)
    {
      _sockets[s]->no_nagle();
      _sockets[s]->putBool(loadVolumeFromFile);

      if (loadVolumeFromFile)
      {
        const bool allFileNamesAreEqual = (_slaveFileNames.size() == 0);
        if (allFileNamesAreEqual)
        {
          _sockets[s]->putFileName(_fileName);
        }
        else
        {
          if (_slaveFileNames.size() > s)
          {
            _sockets[s]->putFileName(_slaveFileNames[s]);
          }
          else
          {
            // Not enough file names specified, try this one.
            _sockets[s]->putFileName(_fileName);
          }
        }
      }
      else
      {
        switch (_sockets[s]->putVolume(vd))
        {
        case vvSocket::VV_OK:
          cerr << "Volume transferred successfully" << endl;
          break;
        case vvSocket::VV_ALLOC_ERROR:
          cerr << "Not enough memory" << endl;
          return VV_SOCKET_ERROR;
        default:
          cerr << "Cannot write volume to socket" << endl;
          return VV_SOCKET_ERROR;
        }
      }
    }
    else
    {
      cerr << "No connection to remote rendering server established at: " << _slaveNames[0] << endl;
      cerr << "Falling back to local rendering" << endl;
      return VV_SOCKET_ERROR;
    }
  }
  _visitor->generateTextureIds(_sockets.size());
  createThreads();
  return VV_OK;
}

vvRemoteClient::ErrorType vvRenderMaster::setRenderer(vvRenderer* renderer)
{
  vvTexRend* texRend = dynamic_cast<vvTexRend*>(renderer);
  if (texRend == NULL)
  {
    cerr << "vvRenderMaster::setRenderer(): Renderer is no texture based renderer" << endl;
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

vvRemoteClient::ErrorType vvRenderMaster::render()
{
  vvTexRend* renderer = dynamic_cast<vvTexRend*>(_renderer);

  if (renderer == NULL)
  {
    cerr << "Renderer is no texture based renderer" << endl;
    return vvRemoteClient::VV_BAD_RENDERER_ERROR;
  }
  float matrixGL[16];

  vvMatrix pr;
  glGetFloatv(GL_PROJECTION_MATRIX, matrixGL);
  pr.set(matrixGL);

  vvMatrix mv;
  glGetFloatv(GL_MODELVIEW_MATRIX, matrixGL);
  mv.set(matrixGL);

  for (int s=0; s<_sockets.size(); ++s)
  {
    _sockets[s]->putCommReason(vvSocketIO::VV_MATRIX);
    _sockets[s]->putMatrix(&pr);
    _sockets[s]->putMatrix(&mv);
  }

  renderer->calcProjectedScreenRects();

  pthread_barrier_wait(&_barrier);

  glDrawBuffer(GL_BACK);
  glClearColor(_bgColor[0], _bgColor[1], _bgColor[2], 1.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  // Retrieve the eye position for bsp-tree traversal
  vvVector3 eye;
  _renderer->getEyePosition(&eye);
  vvMatrix invMV(&mv);
  invMV.invert();
  // This is a gl matrix ==> transpose.
  invMV.transpose();
  eye.multiply(&invMV);

  // Orthographic projection.
  glMatrixMode(GL_PROJECTION);
  glPushMatrix();
  glLoadIdentity();

  // Fix the proxy quad for the frame buffer texture.
  glMatrixMode(GL_MODELVIEW);
  glPushMatrix();
  glLoadIdentity();

  // Setup compositing.
  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);
  glDisable(GL_DEPTH_TEST);
  glEnable(GL_COLOR_MATERIAL);
  glEnable(GL_BLEND);
  glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);

  _bspTree->traverse(eye);

  glMatrixMode(GL_PROJECTION);
  glPopMatrix();

  glMatrixMode(GL_MODELVIEW);
  glPopMatrix();

  _visitor->clearImages();

  return VV_OK;
}

void vvRenderMaster::exit()
{
  for (int s=0; s<_sockets.size(); ++s)
  {
    _sockets[s]->putCommReason(vvSocketIO::VV_EXIT);
    delete _sockets[s];
  }
}

void vvRenderMaster::resize(const int w, const int h)
{
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_RESIZE) == vvSocket::VV_OK)
    {
      _sockets[s]->putWinDims(w, h);
    }
  }
}

void vvRenderMaster::setCurrentFrame(const int index)
{
  vvDebugMsg::msg(3, "vvRenderMaster::setCurrentFrame()");
  vvRemoteClient::setCurrentFrame(index);
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_CURRENT_FRAME) == vvSocket::VV_OK)
    {
      _sockets[s]->putInt32(index);
    }
  }
}

void vvRenderMaster::setMipMode(const int mipMode)
{
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_MIPMODE) == vvSocket::VV_OK)
    {
      _sockets[s]->putInt32(mipMode);
    }
  }
}

void vvRenderMaster::setObjectDirection(const vvVector3* od)
{
  vvDebugMsg::msg(3, "vvRenderMaster::setObjectDirection()");
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_OBJECT_DIRECTION) == vvSocket::VV_OK)
    {
      _sockets[s]->putVector3(*od);
    }
  }
}

void vvRenderMaster::setViewingDirection(const vvVector3* vd)
{
  vvDebugMsg::msg(3, "vvRenderMaster::setViewingDirection()");
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_VIEWING_DIRECTION) == vvSocket::VV_OK)
    {
      _sockets[s]->putVector3(*vd);
    }
  }
}

void vvRenderMaster::setPosition(const vvVector3* p)
{
  vvDebugMsg::msg(3, "vvRenderMaster::setPosition()");
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_POSITION) == vvSocket::VV_OK)
    {
      _sockets[s]->putVector3(*p);
    }
  }
}

void vvRenderMaster::setROIEnable(const bool roiEnabled)
{
  vvDebugMsg::msg(1, "vvRenderMaster::setROIEnable()");
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_TOGGLE_ROI) == vvSocket::VV_OK)
    {
      _sockets[s]->putBool(roiEnabled);
    }
  }
}

void vvRenderMaster::setProbePosition(const vvVector3* pos)
{
  vvDebugMsg::msg(1, "vvRenderMaster::setProbePosition()");
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_ROI_POSITION) == vvSocket::VV_OK)
    {
      _sockets[s]->putVector3(*pos);
    }
  }
}

void vvRenderMaster::setProbeSize(const vvVector3* newSize)
{
  vvDebugMsg::msg(1, "vvRenderMaster::setProbeSize()");
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_ROI_SIZE) == vvSocket::VV_OK)
    {
      _sockets[s]->putVector3(*newSize);
    }
  }
}

void vvRenderMaster::toggleBoundingBox()
{
  vvDebugMsg::msg(3, "vvRenderMaster::toggleBoundingBox()");
  for (int s=0; s<_sockets.size(); ++s)
  {
    _sockets[s]->putCommReason(vvSocketIO::VV_TOGGLE_BOUNDINGBOX);
  }
}

void vvRenderMaster::updateTransferFunction(vvTransFunc& tf)
{
  vvDebugMsg::msg(1, "vvRenderMaster::updateTransferFunction()");
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_TRANSFER_FUNCTION) == vvSocket::VV_OK)
    {
      _sockets[s]->putTransferFunction(tf);
    }
  }
}

void vvRenderMaster::setParameter(const vvRenderer::ParameterType param, const float newValue, const char*)
{
  vvDebugMsg::msg(3, "vvRenderMaster::setParameter()");
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

void vvRenderMaster::adjustQuality(const float quality)
{
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_QUALITY) == vvSocket::VV_OK)
    {
      _sockets[s]->putFloat(quality);
    }
  }
}

void vvRenderMaster::setInterpolation(const bool interpolation)
{
  vvDebugMsg::msg(3, "vvRenderMaster::setInterpolation()");
  for (int s=0; s<_sockets.size(); ++s)
  {
    if (_sockets[s]->putCommReason(vvSocketIO::VV_INTERPOLATION) == vvSocket::VV_OK)
    {
      _sockets[s]->putBool(interpolation);
    }
  }
}

void vvRenderMaster::createThreads()
{
  _threadData = new ThreadArgs[_sockets.size()];
  _threads = new pthread_t[_sockets.size()];
  pthread_barrier_init(&_barrier, NULL, _sockets.size() + 1);
  std::vector<vvImage*>* images = new std::vector<vvImage*>(_sockets.size());
  for (int s=0; s<_sockets.size(); ++s)
  {
    _threadData[s].threadId = s;
    _threadData[s].renderMaster = this;
    _threadData[s].images = images;
    pthread_create(&_threads[s], NULL, getImageFromSocket, (void*)&_threadData[s]);
  }
  _visitor->setImages(images);
}

void vvRenderMaster::destroyThreads()
{
  pthread_barrier_destroy(&_barrier);
  for (int s=0; s<_sockets.size(); ++s)
  {
    pthread_join(_threads[s], NULL);
  }
  delete[] _threads;
  delete[] _threadData;
  _threads = NULL;
  _threadData = NULL;
}

void* vvRenderMaster::getImageFromSocket(void* threadargs)
{
  ThreadArgs* data = reinterpret_cast<ThreadArgs*>(threadargs);

  while (1)
  {
    vvImage* img = new vvImage();
    data->renderMaster->_sockets.at(data->threadId)->getImage(img);
    data->images->at(data->threadId) = img;

    pthread_barrier_wait(&data->renderMaster->_barrier);
  }
  pthread_exit(NULL);
}
