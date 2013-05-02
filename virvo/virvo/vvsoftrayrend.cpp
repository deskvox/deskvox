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

#include "vvaabb.h"
#include "vvdebugmsg.h"
#include "vvpthread.h"
#include "vvsoftrayrend.h"
#include "vvtoolshed.h"
#include "vvvoldesc.h"

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#ifdef HAVE_GLEW
#include <GL/glew.h>
#endif

#ifdef HAVE_OPENGL
#include "private/vvgltools.h"
#endif

#include <queue>

struct Ray
{
  vvVector3 o;
  vvVector3 d;
};

bool intersectBox(const Ray& ray, const vvAABB& aabb,
                  float* tnear, float* tfar)
{
  using std::min;
  using std::max;

  // compute intersection of ray with all six bbox planes
  vvVector3 invR(1.0f / ray.d[0], 1.0f / ray.d[1], 1.0f / ray.d[2]);
  float t1 = (aabb.getMin()[0] - ray.o[0]) * invR[0];
  float t2 = (aabb.getMax()[0] - ray.o[0]) * invR[0];
  float tmin = min(t1, t2);
  float tmax = max(t1, t2);

  t1 = (aabb.getMin()[1] - ray.o[1]) * invR[1];
  t2 = (aabb.getMax()[1] - ray.o[1]) * invR[1];
  tmin = max(min(t1, t2), tmin);
  tmax = min(max(t1, t2), tmax);

  t1 = (aabb.getMin()[2] - ray.o[2]) * invR[2];
  t2 = (aabb.getMax()[2] - ray.o[2]) * invR[2];
  tmin = max(min(t1, t2), tmin);
  tmax = min(max(t1, t2), tmax);

  *tnear = tmin;
  *tfar = tmax;

  return ((tmax >= tmin) && (tmax >= 0.0f));
}

struct vvSoftRayRend::Thread
{
  size_t id;
  pthread_t threadHandle;

  vvSoftRayRend* renderer;
  vvMatrix invViewMatrix;
  std::vector<float>* colors;

  pthread_barrier_t* barrier;
  pthread_mutex_t* mutex;
  std::vector<Tile>* tiles;

  enum Event
  {
    VV_RENDER = 0,
    VV_EXIT
  };

  std::queue<Event> events;
};

vvSoftRayRend::vvSoftRayRend(vvVolDesc* vd, vvRenderState renderState)
  : vvRenderer(vd, renderState)
  , _width(512)
  , _height(512)
  , _firstThread(NULL)
  , _rgbaTF(NULL)
{
  vvDebugMsg::msg(1, "vvSoftRayRend::vvSoftRayRend()");

#ifdef HAVE_GLEW
  glewInit();
#endif

  updateTransferFunction();

  size_t numThreads = static_cast<size_t>(vvToolshed::getNumProcessors());
  pthread_barrier_t* barrier = numThreads > 0 ? new pthread_barrier_t : NULL;
  pthread_mutex_t* mutex = numThreads > 0 ? new pthread_mutex_t : NULL;

  pthread_barrier_init(barrier, NULL, numThreads + 1);
  pthread_mutex_init(mutex, NULL);

  for (size_t i = 0; i < numThreads; ++i)
  {
    Thread* thread = new Thread;
    thread->id = i;

    thread->renderer = this;

    thread->barrier = barrier;
    thread->mutex = mutex;

    pthread_create(&thread->threadHandle, NULL, renderFunc, thread);
    _threads.push_back(thread);

    if (i == 0)
    {
      _firstThread = thread;
    }
  }
}

vvSoftRayRend::~vvSoftRayRend()
{
  vvDebugMsg::msg(1, "vvSoftRayRend::~vvSoftRayRend()");

  for (std::vector<Thread*>::const_iterator it = _threads.begin();
       it != _threads.end(); ++it)
  {
    pthread_mutex_lock((*it)->mutex);
    (*it)->events.push(Thread::VV_EXIT);
    pthread_mutex_unlock((*it)->mutex);

    if (pthread_join((*it)->threadHandle, NULL) != 0)
    {
      vvDebugMsg::msg(0, "vvSoftRayRend::~vvSoftRayRend(): Error joining thread");
    }
  }

  for (std::vector<Thread*>::const_iterator it = _threads.begin();
       it != _threads.end(); ++it)
  {
    if (it == _threads.begin())
    {
      pthread_barrier_destroy((*it)->barrier);
      pthread_mutex_destroy((*it)->mutex);
      delete (*it)->barrier;
      delete (*it)->mutex;
    }

    delete *it;
  }
}

void vvSoftRayRend::renderVolumeGL()
{
  vvDebugMsg::msg(3, "vvSoftRayRend::renderVolumeGL()");

  vvMatrix mv;
  vvMatrix pr;

#ifdef HAVE_OPENGL
  vvGLTools::getModelviewMatrix(&mv);
  vvGLTools::getProjectionMatrix(&pr);
  const virvo::Viewport viewport = vvGLTools::getViewport();
  _width = viewport[2];
  _height = viewport[3];
#endif

  vvMatrix invViewMatrix = mv;
  invViewMatrix.multiplyLeft(pr);
  invViewMatrix.invert();

  std::vector<Tile> tiles = makeTiles(_width, _height);

  std::vector<float> colors;
  colors.resize(_width * _height * 4);

  for (std::vector<Thread*>::const_iterator it = _threads.begin();
       it != _threads.end(); ++it)
  {
    (*it)->invViewMatrix = invViewMatrix;
    (*it)->colors = &colors;
    (*it)->tiles = &tiles;
    (*it)->events.push(Thread::VV_RENDER);
  }
  pthread_barrier_wait(_firstThread->barrier);

  // threads render

  pthread_barrier_wait(_firstThread->barrier);
#ifdef HAVE_OPENGL
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glWindowPos2i(0, 0);
  glDrawPixels(_width, _height, GL_RGBA, GL_FLOAT, &colors[0]);
#endif
}

void vvSoftRayRend::updateTransferFunction()
{
  vvDebugMsg::msg(3, "vvSoftRayRend::updateTransferFunction()");

  if (_firstThread != NULL && _firstThread->mutex != NULL)
  {
    pthread_mutex_lock(_firstThread->mutex);
  }

  size_t lutEntries = getLUTSize();
  delete[] _rgbaTF;
  _rgbaTF = new float[4 * lutEntries];

  vd->computeTFTexture(lutEntries, 1, 1, _rgbaTF);

  if (_firstThread != NULL && _firstThread->mutex != NULL)
  {
    pthread_mutex_unlock(_firstThread->mutex);
  }
}

size_t vvSoftRayRend::getLUTSize() const
{
  vvDebugMsg::msg(3, "vvSoftRayRend::getLUTSize()");
  return (vd->getBPV()==2) ? 4096 : 256;
}

void vvSoftRayRend::setParameter(ParameterType param, const vvParam& newValue)
{
  vvDebugMsg::msg(3, "vvSoftRayRend::setParameter()");
  vvRenderer::setParameter(param, newValue);
}

vvParam vvSoftRayRend::getParameter(ParameterType param) const
{
  vvDebugMsg::msg(3, "vvSoftRayRend::getParameter()");
  return vvRenderer::getParameter(param);
}

std::vector<vvSoftRayRend::Tile> vvSoftRayRend::makeTiles(const int w, const int h)
{
  vvDebugMsg::msg(3, "vvSoftRayRend::makeTiles()");

  const int tilew = 16;
  const int tileh = 16;

  int numtilesx = virvo::toolshed::iDivUp(w, tilew);
  int numtilesy = virvo::toolshed::iDivUp(h, tileh);

  std::vector<Tile> result;
  for (int y = 0; y < numtilesy; ++y)
  {
    for (int x = 0; x < numtilesx; ++x)
    {
      Tile t;
      t.left = tilew * x;
      t.bottom = tileh * y;
      t.right = t.left + tilew;
      if (t.right > w)
      {
        t.right = w;
      }
      t.top = t.bottom + tileh;
      if (t.top > h)
      {
        t.top = h;
      }
      result.push_back(t);
    }
  }
  return result;
}

void vvSoftRayRend::renderTile(const vvSoftRayRend::Tile& tile, const vvMatrix& invViewMatrix, std::vector<float>* colors)
{
  vvDebugMsg::msg(3, "vvSoftRayRend::renderTile()");

  const float opacityThreshold = 0.95f;

  vvsize3 minVox = _visibleRegion.getMin();
  vvsize3 maxVox = _visibleRegion.getMax();
  for (size_t i = 0; i < 3; ++i)
  {
    minVox[i] = std::max(minVox[i], size_t(0));
    maxVox[i] = std::min(maxVox[i], vd->vox[i]);
  }
  const vvVector3 minCorner = vd->objectCoords(minVox);
  const vvVector3 maxCorner = vd->objectCoords(maxVox);
  const vvAABB aabb = vvAABB(minCorner, maxCorner);

  vvVector3 size2 = vd->getSize() * 0.5f;
  const float diagonalVoxels = sqrtf(float(vd->vox[0] * vd->vox[0] +
                                           vd->vox[1] * vd->vox[1] +
                                           vd->vox[2] * vd->vox[2]));
  size_t numSlices = std::max(size_t(1), static_cast<size_t>(_quality * diagonalVoxels));

  uint8_t* raw = vd->getRaw(vd->getCurrentFrame());

  for (int y = tile.bottom; y < tile.top; ++y)
  {
    for (int x = tile.left; x < tile.right; ++x)
    {
      const float u = (x / static_cast<float>(_width)) * 2.0f - 1.0f;
      const float v = (y / static_cast<float>(_height)) * 2.0f - 1.0f;

      vvVector4 o(u, v, -1.0f, 1.0f);
      o.multiply(invViewMatrix);
      vvVector4 d(u, v, 1.0f, 1.0f);
      d.multiply(invViewMatrix);

      Ray ray;
      ray.o = vvVector3(o[0] / o[3], o[1] / o[3], o[2] / o[3]);
      ray.d = vvVector3(d[0] / d[3], d[1] / d[3], d[2] / d[3]);
      ray.d = ray.d - ray.o;
      ray.d.normalize();

      float tbnear = 0.0f;
      float tbfar = 0.0f;

      const bool hit = intersectBox(ray, aabb, &tbnear, &tbfar);
      if (hit)
      {
        float dist = diagonalVoxels / float(numSlices);
        float t = tbnear;
        vvVector3 pos = ray.o + ray.d * tbnear;
        const vvVector3 step = ray.d * dist;
        vvVector4 dst(0.0f);

        while (true)
        {
          vvVector3 texcoord = vvVector3((pos[0] - vd->pos[0] + size2[0]) / (size2[0] * 2.0f),
                     (-pos[1] - vd->pos[1] + size2[1]) / (size2[1] * 2.0f),
                     (-pos[2] - vd->pos[2] + size2[2]) / (size2[2] * 2.0f));
          // calc voxel coordinates
          vvsize3 texcoordi = vvsize3(size_t(texcoord[0] * float(vd->vox[0] - 1)),
                                      size_t(texcoord[1] * float(vd->vox[1] - 1)),
                                      size_t(texcoord[2] * float(vd->vox[2] - 1)));
          size_t idx = texcoordi[2] * vd->vox[0] * vd->vox[1] + texcoordi[1] * vd->vox[0] + texcoordi[0];
          float sample = float(raw[idx]) / 256.0f;
          vvVector4 src(_rgbaTF[size_t(sample * 4 * getLUTSize())],
                        _rgbaTF[size_t(sample * 4 * getLUTSize()) + 1],
                        _rgbaTF[size_t(sample * 4 * getLUTSize()) + 2],
                        _rgbaTF[size_t(sample * 4 * getLUTSize()) + 3]);

          if (_opacityCorrection)
          {
            src[3] = 1 - powf(1 - src[3], dist);
          }

          // pre-multiply alpha
          src[0] *= src[3];
          src[1] *= src[3];
          src[2] *= src[3];

          dst = dst + src * (1.0f - dst[3]);

          if (_earlyRayTermination && (dst[3] > opacityThreshold))
          {
            break;
          }

          t += dist;
          if (t > tbfar)
          {
            break;
          }
          pos += step;
        }

        for (size_t c = 0; c < 4; ++c)
        {
          (*colors)[y * _width * 4 + x * 4 + c] = dst[c];
        }
      }
    }
  }
}

void* vvSoftRayRend::renderFunc(void* args)
{
  vvDebugMsg::msg(3, "vvSoftRayRend::renderFunc()");

  Thread* thread = static_cast<Thread*>(args);

  while (true)
  {
    pthread_mutex_lock(thread->mutex);
    bool haveEvent = !thread->events.empty();
    pthread_mutex_unlock(thread->mutex);

    if (haveEvent)
    {
      pthread_mutex_lock(thread->mutex);
      Thread::Event e = thread->events.front();
      pthread_mutex_unlock(thread->mutex);

      switch (e)
      {
      case Thread::VV_EXIT:
        goto cleanup;
      case Thread::VV_RENDER:
        render(thread);
        break;
      }

      pthread_mutex_lock(thread->mutex);
      thread->events.pop();
      pthread_mutex_unlock(thread->mutex);
    }
  }

cleanup:
  pthread_exit(NULL);
  return NULL;
}

void vvSoftRayRend::render(vvSoftRayRend::Thread* thread)
{
  vvDebugMsg::msg(3, "vvSoftRayRend::render()");

  pthread_barrier_wait(thread->barrier);
  while (true)
  {
    pthread_mutex_lock(thread->mutex);
    if (thread->tiles->empty())
    {
      pthread_mutex_unlock(thread->mutex);
      break;
    }
    Tile tile = thread->tiles->back();
    thread->tiles->pop_back();
    pthread_mutex_unlock(thread->mutex);
    thread->renderer->renderTile(tile, thread->invViewMatrix, thread->colors);
  }
  pthread_barrier_wait(thread->barrier);
}

