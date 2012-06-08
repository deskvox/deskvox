// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 2010 University of Cologne
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

#include "vvbsptree.h"
#include "vvbsptreevisitors.h"
#include "vvdebugmsg.h"
#include "vvgltools.h"
#include "vvparbrickrend.h"
#include "vvpthread.h"
#include "vvrendercontext.h"
#include "vvvoldesc.h"

#include <queue>

struct vvParBrickRend::Thread
{
  Thread()
    : parbrickrend(NULL)
    , renderer(NULL)
    , aabb(vvVector3(), vvVector3())
  {
  }

  vvContextOptions contextOptions;
  vvParBrickRend* parbrickrend;
  vvRenderer* renderer;
  vvSortLastVisitor::Texture texture;

  int id;
  pthread_t threadHandle;
  pthread_barrier_t* barrier;
  pthread_mutex_t* mutex;

  vvAABB aabb;

  vvMatrix mv;
  vvMatrix pr;

  enum Event
  {
    VV_NEW_PARAM = 0,
    VV_RENDER,
    VV_TRANS_FUNC,
    VV_EXIT
  };

  struct Param
  {
    vvRenderState::ParameterType type;
    vvParam newValue;
  };

  std::queue<Event> events;
  std::queue<Param> newParams;
};

vvParBrickRend::vvParBrickRend(vvVolDesc* vd, vvRenderState rs,
                               const std::vector<std::string>& displays,
                               const std::string& type, const vvRendererFactory::Options& options)
  : vvBrickRend(vd, rs, displays.size() + 1, type, options)
{
  vvDebugMsg::msg(1, "vvParBrickRend::vvParBrickRend()");

  glewInit();

  // TODO: for now, the main thread is used for rendering in any case
  const int numBricks = displays.size() + 1;

  // if the current render context has no alpha channel, we cannot use it
  int ac;
  glGetIntegerv(GL_ALPHA_BITS, &ac);
  if (ac == 0)
  {
    vvDebugMsg::msg(0, "vvParBrickRend::vvParBrickRend(): main OpenGL context has no alpha channel");
  }

  // start out with empty rendering regions
  vvAABBi emptyBox = vvAABBi(vvVector3i(), vvVector3i());
  setParameter(vvRenderer::VV_VISIBLE_REGION, emptyBox);
  setParameter(vvRenderer::VV_PADDING_REGION, emptyBox);

  // main thread
  _thread = new Thread;
  _thread->id = 0;
  _thread->renderer = vvRendererFactory::create(vd, *this, _type.c_str(), _options);
  setVisibleRegion(_thread->renderer, _bspTree->getLeafs().at(0)->getAabb());
  _thread->barrier = new pthread_barrier_t;
  _thread->mutex = new pthread_mutex_t;
  _thread->aabb = vvAABB(vd->objectCoords(_bspTree->getLeafs()[0]->getAabb().getMin()),
                         vd->objectCoords(_bspTree->getLeafs()[0]->getAabb().getMax()));

  int ret = pthread_barrier_init(_thread->barrier, NULL, numBricks);
  if (ret != 0)
  {
    vvDebugMsg::msg(0, "vvParBrickRend::vvParBrickRend(): Cannot create barrier");
    return;
  }
  ret = pthread_mutex_init(_thread->mutex, NULL);
  if (ret != 0)
  {
    vvDebugMsg::msg(0, "vvParBrickRend::vvParBrickRend(): Cannot create mutex");
  }

  const vvGLTools::Viewport vp = vvGLTools::getViewport();

  _thread->texture.pixels = new std::vector<float>(vp[2] * vp[3] * 4);
  _thread->texture.rect = new vvRecti;

  // workers
  for (int i = 1; i < numBricks; ++i)
  {
    Thread* thread = new Thread;
    thread->id = i;
    thread->contextOptions.displayName = displays.at(i - 1);
    thread->parbrickrend = this;
    thread->texture.pixels = new std::vector<float>(vp[2] * vp[3] * 4);
    thread->texture.rect = new vvRecti;

    thread->barrier = _thread->barrier;
    thread->mutex = _thread->mutex;

    thread->aabb = vvAABB(vd->objectCoords(_bspTree->getLeafs()[i]->getAabb().getMin()),
                          vd->objectCoords(_bspTree->getLeafs()[i]->getAabb().getMax()));

    vvGLTools::getModelviewMatrix(&thread->mv);
    vvGLTools::getProjectionMatrix(&thread->pr);

    pthread_create(&thread->threadHandle, NULL, renderFunc, thread);
    _threads.push_back(thread);
  }

  _textures.push_back(_thread->texture);
  for (std::vector<Thread*>::const_iterator it = _threads.begin();
       it != _threads.end(); ++it)
  {
    _textures.push_back((*it)->texture);
  }

  _sortLastVisitor = new vvSortLastVisitor;
  _sortLastVisitor->setTextures(_textures);

  pthread_barrier_wait(_thread->barrier);
}

vvParBrickRend::~vvParBrickRend()
{
  vvDebugMsg::msg(1, "vvParBrickRend::~vvParBrickRend()");

  for (std::vector<Thread*>::const_iterator it = _threads.begin();
       it != _threads.end(); ++it)
  {
    (*it)->events.push(Thread::VV_EXIT);
    if (pthread_join((*it)->threadHandle, NULL) != 0)
    {
      vvDebugMsg::msg(0, "vvParBrickRend::~vvParBrickRend(): Error joining thread");
    }
    delete (*it)->texture.pixels;
    delete (*it)->texture.rect;
    delete *it;
  }

  pthread_barrier_destroy(_thread->barrier);
  pthread_mutex_destroy(_thread->mutex);

  delete _thread->barrier;
  delete _thread->mutex;
  delete _thread->texture.pixels;
  delete _thread->texture.rect;
  delete _thread;

  delete _sortLastVisitor;
}

void vvParBrickRend::renderVolumeGL()
{
  vvDebugMsg::msg(4, "vvParBrickRend::renderVolumeGL()");

  if (!_showBricks)
  {
    vvMatrix mv;
    vvGLTools::getModelviewMatrix(&mv);

    vvMatrix pr;
    vvGLTools::getProjectionMatrix(&pr);

    _thread->mv = mv;
    _thread->pr = pr;
    for (std::vector<Thread*>::iterator it = _threads.begin();
         it != _threads.end(); ++it)
    {
      (*it)->mv = mv;
      (*it)->pr = pr;
      (*it)->events.push(Thread::VV_RENDER);
    }

   
    // TODO: if main thread renders, store color and depth buffer right here
    // TODO: configure whether main thread renders or not
    render(_thread);

    vvMatrix invMV;
    invMV.copy(&mv);
    invMV.invert();

    // find eye position:
    vvVector3 eye;
    getEyePosition(&eye);
    eye.multiply(&invMV);

    // bsp tree maintains boxes in voxel coordinates
    vvVector3i veye = vd->voxelCoords(eye);

    // TODO: if we want to use this context for rendering,
    // store the framebuffer before rendering and restore
    // it here. buffer clearing is only a quick solution
    glClear(GL_COLOR_BUFFER_BIT);

    glDisable(GL_CULL_FACE);
    glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glBlendEquation(GL_FUNC_ADD);

    _bspTree->setVisitor(_sortLastVisitor);
    _bspTree->traverse(veye);
  }
  else
  {
    _bspTree->setVisitor(NULL);
  }

  vvBrickRend::renderVolumeGL();
}

void vvParBrickRend::setParameter(ParameterType param, const vvParam& newValue)
{
  vvDebugMsg::msg(3, "vvParBrickRend::setParameter()");

  const Thread::Param p = { param, newValue };

  if (_thread != NULL && _thread->renderer != NULL)
  {
    _thread->renderer->setParameter(param, newValue);
  }

  for (std::vector<Thread*>::iterator it = _threads.begin();
       it != _threads.end(); ++it)
  {
    pthread_mutex_lock((*it)->mutex);
    (*it)->events.push(Thread::VV_NEW_PARAM);
    (*it)->newParams.push(p);
    pthread_mutex_unlock((*it)->mutex);
  }

  vvBrickRend::setParameter(param, newValue);
}

void vvParBrickRend::updateTransferFunction()
{
  vvDebugMsg::msg(3, "vvParBrickRend::updateTransferFunction()");

  if (_thread != NULL && _thread->renderer != NULL)
  {
    _thread->renderer->updateTransferFunction();
  }

  for (std::vector<Thread*>::iterator it = _threads.begin();
       it != _threads.end(); ++it)
  {
    pthread_mutex_lock((*it)->mutex);
    (*it)->events.push(Thread::VV_TRANS_FUNC);
    pthread_mutex_unlock((*it)->mutex);
  }
}

void* vvParBrickRend::renderFunc(void* args)
{
  vvDebugMsg::msg(3, "vvParBrickRend::renderFunc()");

  Thread* thread = static_cast<Thread*>(args);

  vvRenderContext* ctx = new vvRenderContext(thread->contextOptions);

  if (!ctx->makeCurrent())
  {
    vvDebugMsg::msg(0, "vvParBrickRend::renderFunc(): cannot make render context current in worker thread");
    pthread_exit(NULL);
    return NULL;
  }

  thread->renderer = vvRendererFactory::create(thread->parbrickrend->vd, *(thread->parbrickrend),
                                               thread->parbrickrend->_type.c_str(), thread->parbrickrend->_options);
  setVisibleRegion(thread->renderer, thread->parbrickrend->_bspTree->getLeafs().at(thread->id)->getAabb());

  pthread_barrier_wait(thread->barrier);

  while (true)
  {
    vvDebugMsg::msg(3, "vvParBrickRend::renderFunc() - render loop");
    if (!thread->events.empty())
    {
      Thread::Event e = thread->events.front();

      switch (e)
      {
      case Thread::VV_EXIT:
        goto cleanup;
      case Thread::VV_NEW_PARAM:
      {
        if (!thread->newParams.empty())
        {
          Thread::Param p = thread->newParams.front();
          thread->renderer->setParameter(p.type, p.newValue);
          thread->newParams.pop();
        }
        break;
      }
      case Thread::VV_RENDER:
        render(thread);
        break;
      case Thread::VV_TRANS_FUNC:
        thread->renderer->updateTransferFunction();
        break;
      }
      thread->events.pop();
    }
  }

cleanup:
  delete ctx;
  pthread_exit(NULL);
  return NULL;
}

void vvParBrickRend::render(Thread* thread)
{
  pthread_barrier_wait(thread->barrier);

  vvGLTools::setModelviewMatrix(&thread->mv);
  vvGLTools::setProjectionMatrix(&thread->pr);

  glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  thread->renderer->renderVolumeGL();
  const vvGLTools::Viewport vp = vvGLTools::getViewport();
  vvRecti vprect = { vp[0], vp[1], vp[2], vp[3] };

  vvRecti bounds = vvGLTools::getBoundingRect(thread->aabb);
  bounds.intersect(vprect);

  thread->texture.rect->x = bounds.x;
  thread->texture.rect->y = bounds.y;
  thread->texture.rect->width = bounds.width;
  thread->texture.rect->height = bounds.height;

  glReadPixels(thread->texture.rect->x, thread->texture.rect->y,
               thread->texture.rect->width, thread->texture.rect->height,
               GL_RGBA, GL_FLOAT, &(*thread->texture.pixels)[0]);
  pthread_barrier_wait(thread->barrier);
}
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
