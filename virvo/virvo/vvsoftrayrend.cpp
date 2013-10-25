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
#include "vvforceinline.h"
#include "vvpthread.h"
#include "vvrect.h"
#include "vvsoftrayrend.h"
#include "vvtoolshed.h"
#include "vvvoldesc.h"

#include "mem/allocator.h"
#include "private/vvlog.h"


#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#ifdef HAVE_GLEW
#include <GL/glew.h>
#endif

#ifdef HAVE_OPENGL
#include "private/vvgltools.h"
#endif

#include <cstdlib>
#include <cstring>
#include <queue>

#include <boost/math/special_functions/round.hpp>
using namespace boost::math;

typedef std::vector<float, virvo::mem::aligned_allocator<float, CACHE_LINE> > vecf;

#include <boost/detail/endian.hpp>

#ifdef BOOST_LITTLE_ENDIAN
static const size_t high_byte_offset = 1;
#else
static const size_t high_byte_offset = 0;
#endif

#if VV_USE_SSE

#include "sse/sse.h"

typedef virvo::sse::Veci dim_t;
#if 0//__LP64__
struct index_t
{
  virvo::sse::Veci lo;
  virvo::sse::Veci hi;
};
#else
typedef virvo::sse::Veci index_t;
#endif

#define PACK_SIZE_X 2
#define PACK_SIZE_Y 2

using virvo::sse::clamp;
using virvo::sse::min;
using virvo::sse::max;
namespace fast = virvo::sse::fast;
typedef virvo::sse::Veci Vecs;
typedef virvo::sse::Vec3i Vec3s;
typedef virvo::sse::Vec4i Vec4s;
using virvo::sse::AABB;
using virvo::sse::Vec;
using virvo::sse::Vec3;
using virvo::sse::Vec4;
using virvo::sse::Matrix;

#else

typedef size_t dim_t;
typedef size_t index_t;
#define PACK_SIZE_X 1
#define PACK_SIZE_Y 1

#define any(x) (x)
#define all(x) (x)
using virvo::toolshed::clamp;
typedef size_t Vecs;
namespace fast
{
inline virvo::Vec3 normalize(virvo::Vec3 const& v)
{
  return virvo::vecmath::normalize(v);
}
}
typedef vvssize3 Vec3s;
using virvo::AABB;
typedef float Vec;
using virvo::Vec3;
using virvo::Vec4;
using virvo::Matrix;

inline Vec sub(Vec const& u, Vec const& v, Vec const& mask)
{
  (void)mask;
  return u - v;
}

inline Vec4 mul(Vec4 const& v, Vec const& s, Vec const& mask)
{
  (void)mask;
  return v * s;
}

#endif

template <class T, class U>
VV_FORCE_INLINE T vec_cast(U u)
{
#if VV_USE_SSE
  return virvo::sse::sse_cast<T>(u);
#else
  return static_cast<T>(u);
#endif
}

VV_FORCE_INLINE Vec volume(const uint8_t* raw, index_t idx, int bpc)
{
#if VV_USE_SSE
#if 0//__LP64__

#else
  CACHE_ALIGN int indices[4];
  index_t ridx = idx*bpc+high_byte_offset;
  virvo::sse::store(ridx, &indices[0]);
  CACHE_ALIGN float vals[4];
  for (size_t i = 0; i < 4; ++i)
  {
    vals[i] = raw[indices[i]];
  }
  return Vec(&vals[0]);
#endif
#else
  return raw[idx*bpc+high_byte_offset];
#endif
}

VV_FORCE_INLINE Vec4 rgba(vecf* tf, Vecs idx)
{
#if VV_USE_SSE
  CACHE_ALIGN int indices[4];
  store(idx, &indices[0]);
  Vec4 colors;
  for (size_t i = 0; i < 4; ++i)
  {
    colors[i] = &(*tf)[0] + indices[i];
  }
  colors = transpose(colors);
  return colors;
#else
  return Vec4(&(*tf)[0] + idx);
#endif
}

VV_FORCE_INLINE size_t getLUTSize(vvVolDesc* vd)
{
  return (vd->getBPV()==2) ? 4096 : 256;
}

VV_FORCE_INLINE Vec pixelx(int x)
{
#if VV_USE_SSE
  return Vec(x, x + 1, x, x + 1);
#else
  return x;
#endif
}

VV_FORCE_INLINE Vec pixely(int y)
{
#if VV_USE_SSE
  return Vec(y, y, y + 1, y + 1);
#else
  return y;
#endif
}

struct Ray
{
  Ray(Vec3 const& ori, Vec3 const& dir)
    : o(ori)
    , d(dir)
  {
  }

  Vec3 o;
  Vec3 d;
};

Vec intersectBox(const Ray& ray, const AABB& aabb,
                 Vec* tnear, Vec* tfar)
{
  using std::min;
  using std::max;

  // compute intersection of ray with all six bbox planes
  Vec3 invR(1.0f / ray.d[0], 1.0f / ray.d[1], 1.0f / ray.d[2]);
  Vec t1 = (aabb.getMin()[0] - ray.o[0]) * invR[0];
  Vec t2 = (aabb.getMax()[0] - ray.o[0]) * invR[0];
  Vec tmin = min(t1, t2);
  Vec tmax = max(t1, t2);

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


namespace virvo
{

struct Tile
{
  int left;
  int bottom;
  int right;
  int top;
};

}


std::vector<virvo::Tile> makeTiles(vvRecti const& rect, virvo::Viewport const& vp)
{
  const int tilew = 16;
  const int tileh = 16;

  int w = rect[2];
  int h = rect[3];

  int numtilesx = virvo::toolshed::iDivUp(w, tilew);
  int numtilesy = virvo::toolshed::iDivUp(h, tileh);

  std::vector<virvo::Tile> result;
  for (int y = 0; y < numtilesy; ++y)
  {
    for (int x = 0; x < numtilesx; ++x)
    {
      virvo::Tile t;
      t.left = rect[0] + tilew * x;
      t.bottom = rect[1] + tileh * y;
      t.right = t.left + tilew;
      if (t.right > vp[2])
      {
        t.right = w;
      }
      t.top = t.bottom + tileh;
      if (t.top > vp[3])
      {
        t.top = h;
      }
      result.push_back(t);
    }
  }
  return result;
}


struct Thread
{
  size_t id;
  pthread_t threadHandle;

  vvSoftRayRend* renderer;
  Matrix invViewMatrix;
  float* colors;

  pthread_barrier_t* barrier;
  pthread_mutex_t* mutex;
  std::vector<virvo::Tile>* tiles;
  vecf* rgbaTF;

  enum Event
  {
    VV_RENDER = 0,
    VV_EXIT
  };

  std::queue<Event> events;
};

struct vvSoftRayRend::Impl
{
  Impl() : firstThread(NULL) {}

  std::vector< Thread* > threads;
  Thread* firstThread;
  vecf rgbaTF;
};

vvSoftRayRend::vvSoftRayRend(vvVolDesc* vd, vvRenderState renderState)
  : vvRenderer(vd, renderState)
  , impl_(new Impl)
{
  vvDebugMsg::msg(1, "vvSoftRayRend::vvSoftRayRend()");

#ifdef HAVE_GLEW
  glewInit();
#endif

  setRenderTarget(virvo::HostBufferRT::create(virvo::PF_RGBA32F, virvo::PF_LUMINANCE8));

  updateTransferFunction();

  size_t numThreads = static_cast<size_t>(vvToolshed::getNumProcessors());
  char* envNumThreads = getenv("VV_NUM_THREADS");
  if (envNumThreads != NULL)
  {
    numThreads = atoi(envNumThreads);
    VV_LOG(0) << "VV_NUM_THREADS: " << envNumThreads;
  }

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

    thread->rgbaTF = &impl_->rgbaTF;

    pthread_create(&thread->threadHandle, NULL, renderFunc, thread);
    impl_->threads.push_back(thread);

    if (i == 0)
    {
      impl_->firstThread = thread;
    }
  }
}

vvSoftRayRend::~vvSoftRayRend()
{
  vvDebugMsg::msg(1, "vvSoftRayRend::~vvSoftRayRend()");

  for (std::vector<Thread*>::const_iterator it = impl_->threads.begin();
       it != impl_->threads.end(); ++it)
  {
    pthread_mutex_lock((*it)->mutex);
    (*it)->events.push(Thread::VV_EXIT);
    pthread_mutex_unlock((*it)->mutex);

    if (pthread_join((*it)->threadHandle, NULL) != 0)
    {
      vvDebugMsg::msg(0, "vvSoftRayRend::~vvSoftRayRend(): Error joining thread");
    }
  }

  for (std::vector<Thread*>::const_iterator it = impl_->threads.begin();
       it != impl_->threads.end(); ++it)
  {
    if (it == impl_->threads.begin())
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

  Matrix mv;
  Matrix pr;
  virvo::Viewport vp;

  mv = virvo::gltools::getModelViewMatrix();
  pr = virvo::gltools::getProjectionMatrix();
  vp = vvGLTools::getViewport();

  virvo::RenderTarget* rt = getRenderTarget();

  int w = rt->width();
  int h = rt->height();

  Matrix invViewMatrix = mv;
  invViewMatrix = pr * invViewMatrix;
  invViewMatrix.invert();

  vvAABB aabb = vvAABB(virvo::Vec3(), virvo::Vec3());
  vd->getBoundingBox(aabb);
  vvRecti r = vvGLTools::getBoundingRect(aabb);
  std::vector<virvo::Tile> tiles = makeTiles(r, vp);

  float* colorBuffer = reinterpret_cast<float*>(rt->deviceColor());

  for (std::vector<Thread*>::const_iterator it = impl_->threads.begin();
       it != impl_->threads.end(); ++it)
  {
    (*it)->invViewMatrix = invViewMatrix;
    (*it)->colors = colorBuffer;
    (*it)->tiles = &tiles;
    (*it)->events.push(Thread::VV_RENDER);
  }
  pthread_barrier_wait(impl_->firstThread->barrier);

  // threads render

  pthread_barrier_wait(impl_->firstThread->barrier);
}

void vvSoftRayRend::updateTransferFunction()
{
  vvDebugMsg::msg(3, "vvSoftRayRend::updateTransferFunction()");

  if (impl_->firstThread != NULL && impl_->firstThread->mutex != NULL)
  {
    pthread_mutex_lock(impl_->firstThread->mutex);
  }

  size_t lutEntries = getLUTSize(vd);
  impl_->rgbaTF.resize(4 * lutEntries);

  vd->computeTFTexture(lutEntries, 1, 1, &impl_->rgbaTF[0]);

  if (impl_->firstThread != NULL && impl_->firstThread->mutex != NULL)
  {
    pthread_mutex_unlock(impl_->firstThread->mutex);
  }
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

void renderTile(const virvo::Tile& tile, const Thread* thread)
{
  vvDebugMsg::msg(3, "vvSoftRayRend::renderTile()");

  static const Vec opacityThreshold = 0.95f;

  vvVolDesc* vd            = thread->renderer->getVolDesc();
  virvo::AABBss const& vr  = thread->renderer->getParameter(vvRenderer::VV_VISIBLE_REGION);
  float quality            = thread->renderer->getParameter(vvRenderer::VV_QUALITY);
  bool interpolation       = thread->renderer->getParameter(vvRenderer::VV_SLICEINT);
  bool opacityCorrection   = thread->renderer->getParameter(vvRenderer::VV_OPCORR);
  bool earlyRayTermination = thread->renderer->getParameter(vvRenderer::VV_TERMINATEEARLY);

  virvo::ssize3 minVox = vr.getMin();
  virvo::ssize3 maxVox = vr.getMax();
  for (size_t i = 0; i < 3; ++i)
  {
    minVox[i] = std::max(minVox[i], ssize_t(0));
    maxVox[i] = std::min(maxVox[i], vd->vox[i]);
  }
  const virvo::Vec3 minCorner = vd->objectCoords(minVox);
  const virvo::Vec3 maxCorner = vd->objectCoords(maxVox);
  const AABB aabb(minCorner, maxCorner);

  Vec3 size = vd->getSize();
  Vec3 invsize = 1.0f / size;
  Vec3 size2 = vd->getSize() * 0.5f;
  const float diagonalVoxels = sqrtf(float(vd->vox[0] * vd->vox[0] +
                                           vd->vox[1] * vd->vox[1] +
                                           vd->vox[2] * vd->vox[2]));
  size_t numSlices = std::max(size_t(1), static_cast<size_t>(quality * diagonalVoxels));

  const uint8_t* raw = vd->getRaw(vd->getCurrentFrame());
  const size_t bpc = vd->bpc;

  virvo::RenderTarget const* rt = thread->renderer->getRenderTarget();

  for (int y = tile.bottom; y < tile.top; y += PACK_SIZE_Y)
  {
    for (int x = tile.left; x < tile.right; x += PACK_SIZE_X)
    {
      const Vec u = (pixelx(x) / static_cast<float>(rt->width() - 1)) * 2.0f - 1.0f;
      const Vec v = (pixely(y) / static_cast<float>(rt->height() - 1)) * 2.0f - 1.0f;

      Vec4 o(u, v, -1.0f, 1.0f);
      o = thread->invViewMatrix * o;
      Vec4 d(u, v, 1.0f, 1.0f);
      d = thread->invViewMatrix * d;

      Ray ray(Vec3(o[0] / o[3], o[1] / o[3], o[2] / o[3]),
              Vec3(d[0] / d[3], d[1] / d[3], d[2] / d[3]));
      ray.d = ray.d - ray.o;
      ray.d = fast::normalize(ray.d);

      Vec tbnear = 0.0f;
      Vec tbfar = 0.0f;

      Vec active = intersectBox(ray, aabb, &tbnear, &tbfar);
      if (any(active))
      {
        Vec dist = diagonalVoxels / Vec(numSlices);
        Vec t = tbnear;
        Vec3 pos = ray.o + ray.d * tbnear;
        const Vec3 step = ray.d * dist;
        Vec4 dst(0.0f);

        // construct simd vectors only once
        Vec3s vox = vd->vox;

        while (any(active))
        {
          Vec3 texcoord((pos[0] - vd->pos[0] + size2[0]) * invsize[0],
                        (-pos[1] - vd->pos[1] + size2[1]) * invsize[1],
                        (-pos[2] - vd->pos[2] + size2[2]) * invsize[2]);
          texcoord[0] = clamp(texcoord[0], Vec(0.0f), Vec(1.0f));
          texcoord[1] = clamp(texcoord[1], Vec(0.0f), Vec(1.0f));
          texcoord[2] = clamp(texcoord[2], Vec(0.0f), Vec(1.0f));

          Vec sample = 0.0f;
          if (interpolation)
          {
            Vec3 texcoordf(texcoord[0] * static_cast<float>(vd->vox[0] - 1),
                           texcoord[1] * static_cast<float>(vd->vox[1] - 1),
                           texcoord[2] * static_cast<float>(vd->vox[2] - 1));

            // store truncated texcoord to avoid lots of _mm_cvtps_epi32 calls below
            Vec3s tci(vec_cast<Vecs>(texcoordf[0]), vec_cast<Vecs>(texcoordf[1]), vec_cast<Vecs>(texcoordf[2]));

            Vec samples[8];

            Vec3s tc = tci + Vec3s(0, 0, 0);
            tc[0] = clamp<dim_t>(tc[0], 0, vox[0] - 1);
            tc[1] = clamp<dim_t>(tc[1], 0, vox[1] - 1);
            tc[2] = clamp<dim_t>(tc[2], 0, vox[2] - 1);
            index_t idx = tc[2] * vox[0] * vox[1] + tc[1] * vox[0] + tc[0];
            samples[0] = volume(raw, idx, bpc);

            tc = tci + Vec3s(1, 0, 0);
            tc[0] = clamp<dim_t>(tc[0], 0, vox[0] - 1);
            tc[1] = clamp<dim_t>(tc[1], 0, vox[1] - 1);
            tc[2] = clamp<dim_t>(tc[2], 0, vox[2] - 1);
            idx = tc[2] * vox[0] * vox[1] + tc[1] * vox[0] + tc[0];
            samples[1] = volume(raw, idx, bpc);

            tc = tci + Vec3s(1, 1, 0);
            tc[0] = clamp<dim_t>(tc[0], 0, vox[0] - 1);
            tc[1] = clamp<dim_t>(tc[1], 0, vox[1] - 1);
            tc[2] = clamp<dim_t>(tc[2], 0, vox[2] - 1);
            idx = tc[2] * vox[0] * vox[1] + tc[1] * vox[0] + tc[0];
            samples[2] = volume(raw, idx, bpc);

            tc = tci + Vec3s(0, 1, 0);
            tc[0] = clamp<dim_t>(tc[0], 0, vox[0] - 1);
            tc[1] = clamp<dim_t>(tc[1], 0, vox[1] - 1);
            tc[2] = clamp<dim_t>(tc[2], 0, vox[2] - 1);
            idx = tc[2] * vox[0] * vox[1] + tc[1] * vox[0] + tc[0];
            samples[3] = volume(raw, idx, bpc);

            tc = tci + Vec3s(1, 0, 1);
            tc[0] = clamp<dim_t>(tc[0], 0, vox[0] - 1);
            tc[1] = clamp<dim_t>(tc[1], 0, vox[1] - 1);
            tc[2] = clamp<dim_t>(tc[2], 0, vox[2] - 1);
            idx = tc[2] * vox[0] * vox[1] + tc[1] * vox[0] + tc[0];
            samples[4] = volume(raw, idx, bpc);

            tc = tci + Vec3s(0, 0, 1);
            tc[0] = clamp<dim_t>(tc[0], 0, vox[0] - 1);
            tc[1] = clamp<dim_t>(tc[1], 0, vox[1] - 1);
            tc[2] = clamp<dim_t>(tc[2], 0, vox[2] - 1);
            idx = tc[2] * vox[0] * vox[1] + tc[1] * vox[0] + tc[0];
            samples[5] = volume(raw, idx, bpc);

            tc = tci + Vec3s(0, 1, 1);
            tc[0] = clamp<dim_t>(tc[0], 0, vox[0] - 1);
            tc[1] = clamp<dim_t>(tc[1], 0, vox[1] - 1);
            tc[2] = clamp<dim_t>(tc[2], 0, vox[2] - 1);
            idx = tc[2] * vox[0] * vox[1] + tc[1] * vox[0] + tc[0];
            samples[6] = volume(raw, idx, bpc);

            tc = tci + Vec3s(1, 1, 1);
            tc[0] = clamp<dim_t>(tc[0], 0, vox[0] - 1);
            tc[1] = clamp<dim_t>(tc[1], 0, vox[1] - 1);
            tc[2] = clamp<dim_t>(tc[2], 0, vox[2] - 1);
            idx = tc[2] * vox[0] * vox[1] + tc[1] * vox[0] + tc[0];
            samples[7] = volume(raw, idx, bpc);


            Vec3 tmp(vec_cast<Vec>(tci[0]), vec_cast<Vec>(tci[1]), vec_cast<Vec>(tci[2]));
            Vec3 uvw = texcoordf - tmp;

            // lerp
            Vec p1 = (1 - uvw[0]) * samples[0] + uvw[0] * samples[1];
            Vec p2 = (1 - uvw[0]) * samples[3] + uvw[0] * samples[2];
            Vec p12 = (1 - uvw[1]) * p1 + uvw[1] * p2;

            Vec p3 = (1 - uvw[0]) * samples[5] + uvw[0] * samples[4];
            Vec p4 = (1 - uvw[0]) * samples[6] + uvw[0] * samples[7];
            Vec p34 = (1 - uvw[1]) * p3 + uvw[1] * p4;

            sample = (1 - uvw[2]) * p12 + uvw[2] * p34;
          }
          else
          {
            // calc voxel coordinates using Manhattan distance
            Vec3s texcoordi(vec_cast<Vecs>(round(texcoord[0] * static_cast<float>(vd->vox[0] - 1))),
                            vec_cast<Vecs>(round(texcoord[1] * static_cast<float>(vd->vox[1] - 1))),
                            vec_cast<Vecs>(round(texcoord[2] * static_cast<float>(vd->vox[2] - 1))));
  
            // clamp to edge
            texcoordi[0] = clamp<dim_t>(texcoordi[0], 0, vox[0] - 1);
            texcoordi[1] = clamp<dim_t>(texcoordi[1], 0, vox[1] - 1);
            texcoordi[2] = clamp<dim_t>(texcoordi[2], 0, vox[2] - 1);

            index_t idx = texcoordi[2] * vox[0] * vox[1] + texcoordi[1] * vox[0] + texcoordi[0];
            sample = volume(raw, idx, bpc);
          }

          sample /= 255.0f;
          Vec4 src = rgba(&(*thread->rgbaTF), vec_cast<Vecs>(sample * static_cast<float>(getLUTSize(vd))) * 4);

          if (opacityCorrection)
          {
            src[3] = 1 - powf(1 - src[3], dist);
          }

          // pre-multiply alpha
          src[0] *= src[3];
          src[1] *= src[3];
          src[2] *= src[3];

          dst = dst + mul(src, sub(1.0f, dst[3], active), active);

          if (earlyRayTermination)
          {
            active = active && dst[3] <= opacityThreshold;
          }

          t += dist;
          active = active && (t < tbfar);
          pos += step;
        }

#if VV_USE_SSE
        // transform to AoS for framebuffer write
        dst = transpose(dst);
        store(dst.x, &(thread->colors)[y * rt->width() * 4 + x * 4]);
        if (x + 1 < tile.right)
        {
          store(dst.y, &(thread->colors)[y * rt->width() * 4 + (x + 1) * 4]);
        }
        if (y + 1 < tile.top)
        {
          store(dst.z, &(thread->colors)[(y + 1) * rt->width() * 4 + x * 4]);
        }
        if (x + 1 < tile.right && y + 1 < tile.top)
        {
          store(dst.w, &(thread->colors)[(y + 1) * rt->width() * 4 + (x + 1) * 4]);
        }
#else
        memcpy(&(thread->colors)[y * rt->width() * 4 + x * 4], &dst[0], 4 * sizeof(float));
#endif
      }
    }
  }
}

void render(Thread* thread)
{
  pthread_barrier_wait(thread->barrier);
  while (true)
  {
    pthread_mutex_lock(thread->mutex);
    if (thread->tiles->empty())
    {
      pthread_mutex_unlock(thread->mutex);
      break;
    }
    virvo::Tile tile = thread->tiles->back();
    thread->tiles->pop_back();
    pthread_mutex_unlock(thread->mutex);
    renderTile(tile, thread);
  }
  pthread_barrier_wait(thread->barrier);
}

void* vvSoftRayRend::renderFunc(void* args)
{
  vvDebugMsg::msg(3, "vvSoftRayRend::renderFunc()");

  Thread* thread = static_cast<Thread*>(args);

#ifdef __linux__
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  CPU_SET(thread->id, &cpuset);
  int s = pthread_setaffinity_np(thread->threadHandle, sizeof(cpu_set_t), &cpuset);
  if (s != 0)
  {
    VV_LOG(0) << "Error setting thread affinity: " << strerror(s);
  }
#endif

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

vvRenderer* createRayRend(vvVolDesc* vd, vvRenderState const& rs)
{
  return new vvSoftRayRend(vd, rs);
}

