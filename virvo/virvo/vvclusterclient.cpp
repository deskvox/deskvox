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
#include "vvclusterclient.h"
#include "vvtexrend.h"

using std::cerr;
using std::endl;

vvClusterClient::vvClusterClient(vvRenderState renderState,
                                 std::vector<const char*>& slaveNames, std::vector<int>& slavePorts,
                                 std::vector<const char*>& slaveFileNames,
                                 const char* fileName)
  : vvRemoteClient(renderState, slaveNames, slavePorts, slaveFileNames, fileName)
{
  vvDebugMsg::msg(1, "vvClusterClient::vvClusterClient()");

  _threads = NULL;
  _threadData = NULL;
  _visitor = new vvSlaveVisitor();
}

vvClusterClient::~vvClusterClient()
{
  vvDebugMsg::msg(1, "vvClusterClient::~vvClusterClient()");

  destroyThreads();
  delete _visitor;
}

vvRemoteClient::ErrorType vvClusterClient::setRenderer(vvRenderer* renderer)
{
  vvDebugMsg::msg(1, "vvClusterClient::setRenderer()");

  vvTexRend* texRend = dynamic_cast<vvTexRend*>(renderer);
  if (texRend == NULL)
  {
    cerr << "vvRenderMaster::setRenderer(): Renderer is no texture based renderer" << endl;
    return vvRemoteClient::VV_WRONG_RENDERER;
  }

  // This will build up the bsp tree of the master node.
  texRend->prepareDistributedRendering(_slaveNames.size());

  // Store a pointer to the bsp tree and set its visitor.
  _bspTree = texRend->getBspTree();
  _bspTree->setVisitor(_visitor);

  _renderer = texRend;

  // Distribute the bricks from the bsp tree
  std::vector<BrickList>** bricks = texRend->getBrickListsToDistribute();
  for (size_t s=0; s<_sockets.size(); ++s)
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

vvRemoteClient::ErrorType vvClusterClient::render()
{
  vvDebugMsg::msg(3, "vvClusterClient::render()");

  vvTexRend* renderer = dynamic_cast<vvTexRend*>(_renderer);

  if (renderer == NULL)
  {
    cerr << "Renderer is no texture based renderer" << endl;
    return vvRemoteClient::VV_WRONG_RENDERER;
  }
  float matrixGL[16];

  vvMatrix pr;
  glGetFloatv(GL_PROJECTION_MATRIX, matrixGL);
  pr.set(matrixGL);

  vvMatrix mv;
  glGetFloatv(GL_MODELVIEW_MATRIX, matrixGL);
  mv.set(matrixGL);

  for (size_t s=0; s<_sockets.size(); ++s)
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

  clearImages();

  return VV_OK;
}

void vvClusterClient::exit()
{
  vvDebugMsg::msg(1, "vvClusterClient::exit()");

  for (size_t s=0; s<_sockets.size(); ++s)
  {
    _sockets[s]->putCommReason(vvSocketIO::VV_EXIT);
    delete _sockets[s];
  }
}

void vvClusterClient::createThreads()
{
  vvDebugMsg::msg(1, "vvClusterClient::createThreads()");

  _visitor->generateTextureIds(_sockets.size());
  _threadData = new ThreadArgs[_sockets.size()];
  _threads = new pthread_t[_sockets.size()];
  pthread_barrier_init(&_barrier, NULL, _sockets.size() + 1);
  for (size_t s=0; s<_sockets.size(); ++s)
  {
    _threadData[s].threadId = s;
    _threadData[s].clusterClient = this;
    _threadData[s].images = _images;
    pthread_create(&_threads[s], NULL, getImageFromSocket, (void*)&_threadData[s]);
  }
  _visitor->setImages(_images);
}

void vvClusterClient::destroyThreads()
{
  vvDebugMsg::msg(1, "vvClusterClient::destroyThreads()");

  pthread_barrier_destroy(&_barrier);
  for (size_t s=0; s<_sockets.size(); ++s)
  {
    pthread_join(_threads[s], NULL);
  }
  delete[] _threads;
  delete[] _threadData;
  _threads = NULL;
  _threadData = NULL;
}

void* vvClusterClient::getImageFromSocket(void* threadargs)
{
  vvDebugMsg::msg(1, "vvClusterClient::getImageFromSocket()");

  ThreadArgs* data = reinterpret_cast<ThreadArgs*>(threadargs);

  while (1)
  {
    vvImage* img = new vvImage();
    data->clusterClient->_sockets.at(data->threadId)->getImage(img);
    data->images->at(data->threadId) = img;

    pthread_barrier_wait(&data->clusterClient->_barrier);
  }
  pthread_exit(NULL);
#ifdef _WIN32
  return NULL;
#endif
}
