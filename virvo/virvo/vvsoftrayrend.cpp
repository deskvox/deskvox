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

#include "vvdebugmsg.h"
#include "vvmacros.h"
#include "vvpthread.h"
#include "vvsoftrayrend.h"
#include "vvvoldesc.h"

#include "math/math.h"
#include "mem/allocator.h"
#include "private/vvlog.h"
#include "private/project.h"
#include "texture/texture.h"


#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#ifdef HAVE_OPENGL
#include "gl/util.h"
#endif

#include <algorithm>
#include <cstdlib>
#include <cstring>

#if VV_CXX_MSVC
#include <intrin.h>
#endif

typedef std::vector<float, virvo::mem::aligned_allocator<float, 16> > vecf;

namespace virvo
{

static const int tile_width  = 16;
static const int tile_height = 16;

}

namespace simd = virvo::simd;

#define DIV_UP(a, b) ((a + b - 1) / b)


/* TODO: cross-platform atomics */
#if VV_CXX_CLANG || VV_CXX_GCC || VV_CXX_INTEL
#define atom_fetch_and_add(a, b)     __sync_fetch_and_add(a, b)
#define atom_lock_test_and_set(a, b) __sync_lock_test_and_set(a, b)
#elif VV_CXX_MSVC
#define atom_fetch_and_add(a, b)     _InterlockedExchangeAdd(a, b)
#define atom_lock_test_and_set(a, b) _InterlockedExchange(a, b)
#else
#define atom_fetch_and_add(a, b)
#define atom_lock_test_and_set(a, b)
#endif


#if VV_USE_SSE

#define PACK_SIZE_X 2
#define PACK_SIZE_Y 2

typedef simd::int4 int_type;
typedef simd::float4 float_type;

#else

#define PACK_SIZE_X 1
#define PACK_SIZE_Y 1

using virvo::any;
using virvo::all;
using std::min;
using std::max;
typedef size_t int_type;
typedef float float_type;

#endif

using simd::sub;
using simd::mul;

typedef virvo::basic_aabb< float_type > AABB;
typedef virvo::basic_ray< float_type > Ray;
typedef virvo::vector< 3, int_type > Vec3s;
typedef virvo::vector< 4, int_type > Vec4s;
typedef virvo::vector< 3, float_type > Vec3;
typedef virvo::vector< 4, float_type > Vec4;
typedef virvo::matrix< 4, 4, float_type > Mat4;


VV_FORCE_INLINE size_t getLUTSize(vvVolDesc* vd)
{
  return (vd->getBPV()==2) ? 4096 : 256;
}

VV_FORCE_INLINE float_type pixelx(float x)
{
#if VV_USE_SSE
  return float_type(x, x + 1.0f, x, x + 1.0f);
#else
  return x;
#endif
}

VV_FORCE_INLINE float_type pixely(float y)
{
#if VV_USE_SSE
  return float_type(y, y, y + 1.0f, y + 1.0f);
#else
  return y;
#endif
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


struct Thread
{
  size_t id;
  pthread_t threadHandle;

  uint8_t** raw;
  float** colors;
  vecf* rgbaTF;

  float** invViewMatrix;

  struct RenderParams
  {

    // render target
    int                     width;
    int                     height;

    virvo::Tile             rect;

    // visible volume region
    float                   mincorner[3];
    float                   maxcorner[3];

    // vd attributes
    int                     vox[3];
    float                   volsize[3];
    float                   volpos[3];
    size_t                  bpc;

    // tf
    size_t                  lutsize;

    // renderer params
    float                   quality;
    virvo::tex_filter_mode  filter_mode;
    bool                    opacity_correction;
    bool                    early_ray_termination;
    int                     mip_mode;

  };

  struct SyncParams
  {
    SyncParams()
#ifdef __APPLE__
      : image_ready(virvo::NamedSemaphore("vv_softrayrend_image_ready"))
#else
      : image_ready(0)
#endif
      , exit_render_loop(false)
    {
    }

    long tile_idx_counter;
    long tile_fin_counter;
    long tile_idx_max;
    virvo::SyncedCondition start_render;
#ifdef __APPLE__
    virvo::NamedSemaphore image_ready;
#else
    virvo::Semaphore image_ready;
#endif
    bool exit_render_loop;
  };

  RenderParams* render_params;
  SyncParams*   sync_params;
};


void  wake_render_threads(Thread::RenderParams rparams, Thread::SyncParams* sparams);
template < typename T >
void  render(virvo::texture< T, virvo::ElementType, 3 > const& volume, Thread* thread);
void* renderFunc(void* args);


struct vvSoftRayRend::Impl
{
  Impl()
    : raw(0)
    , colors(0)
    , inv_view_matrix(0)
  {
  }

  ~Impl()
  {
    delete[] inv_view_matrix;
  }

  std::vector< Thread* > threads;
  uint8_t* raw;
  float* colors;
  vecf rgbaTF;

  float* inv_view_matrix;

  Thread::RenderParams render_params;
  Thread::SyncParams   sync_params;
};

vvSoftRayRend::vvSoftRayRend(vvVolDesc* vd, vvRenderState renderState)
  : vvRenderer(vd, renderState)
  , impl_(new Impl)
{

  rendererType = RAYREND;

#if VV_USE_SSE
  // TODO: find a better place to hide this
  _MM_SET_ROUNDING_MODE(_MM_ROUND_DOWN);
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


  for (size_t i = 0; i < numThreads; ++i)
  {
    Thread* thread        = new Thread;
    thread->id            = i;

    thread->raw           = &impl_->raw;
    thread->colors        = &impl_->colors;
    thread->rgbaTF        = &impl_->rgbaTF;
    thread->invViewMatrix = &impl_->inv_view_matrix;

    thread->render_params = &impl_->render_params;
    thread->sync_params   = &impl_->sync_params;


    pthread_create(&thread->threadHandle, NULL, renderFunc, thread);
    impl_->threads.push_back(thread);
  }
}

vvSoftRayRend::~vvSoftRayRend()
{
  vvDebugMsg::msg(1, "vvSoftRayRend::~vvSoftRayRend()");

  impl_->sync_params.exit_render_loop = true;
  impl_->sync_params.start_render.broadcast();

  for (std::vector<Thread*>::const_iterator it = impl_->threads.begin();
       it != impl_->threads.end(); ++it)
  {
    if (pthread_join((*it)->threadHandle, NULL) != 0)
    {
      vvDebugMsg::msg(0, "vvSoftRayRend::~vvSoftRayRend(): Error joining thread");
    }
  }

  for (std::vector<Thread*>::const_iterator it = impl_->threads.begin();
       it != impl_->threads.end(); ++it)
  {
    delete *it;
  }
}

void vvSoftRayRend::renderVolumeGL()
{
  vvDebugMsg::msg(3, "vvSoftRayRend::renderVolumeGL()");

  virvo::mat4 mv;
  virvo::mat4 pr;

#ifdef HAVE_OPENGL
  mv = virvo::gl::getModelviewMatrix();
  pr = virvo::gl::getProjectionMatrix();
#endif

  virvo::mat4 inv_view_matrix = inverse( pr * mv );

  if (impl_->inv_view_matrix == 0)
  {
    impl_->inv_view_matrix = new float[16];
  }
  float* tmp = inv_view_matrix.data();
  std::copy( tmp, tmp + 16, impl_->inv_view_matrix );

  virvo::RenderTarget* rt = getRenderTarget();

  virvo::recti vp;
  vp[0] = 0;
  vp[1] = 0;
  vp[2] = rt->width();
  vp[3] = rt->height();

  virvo::aabb bbox = vd->getBoundingBox();
  virvo::recti r = virvo::bounds(bbox, mv, pr, vp);

  impl_->raw    = vd->getRaw(vd->getCurrentFrame());
  impl_->colors = reinterpret_cast<float*>(rt->deviceColor());

  virvo::Tile rect;
  rect.left   = r[0];
  rect.right  = r[0] + r[2];
  rect.bottom = r[1];
  rect.top    = r[1] + r[3];

  virvo::basic_aabb< ssize_t > const& vr     = getParameter(vvRenderer::VV_VISIBLE_REGION);
  virvo::vector< 3, ssize_t > minvox         = vr.min;
  virvo::vector< 3, ssize_t > maxvox         = vr.max;
  for (size_t i = 0; i < 3; ++i)
  {
    minvox[i] = std::max(minvox[i], ssize_t(0));
    maxvox[i] = std::min(maxvox[i], vd->vox[i]);
  }

  virvo::vec3 mincorner     = vd->objectCoords(minvox);
  virvo::vec3 maxcorner     = vd->objectCoords(maxvox);

  impl_->render_params.width                 = rt->width();
  impl_->render_params.height                = rt->height();
  impl_->render_params.rect                  = rect;
  impl_->render_params.mincorner[0]          = mincorner[0];
  impl_->render_params.mincorner[1]          = mincorner[1];
  impl_->render_params.mincorner[2]          = mincorner[2];
  impl_->render_params.maxcorner[0]          = maxcorner[0];
  impl_->render_params.maxcorner[1]          = maxcorner[1];
  impl_->render_params.maxcorner[2]          = maxcorner[2];
  impl_->render_params.vox[0]                = vd->vox[0];
  impl_->render_params.vox[1]                = vd->vox[1];
  impl_->render_params.vox[2]                = vd->vox[2];
  impl_->render_params.volsize[0]            = vd->getSize()[0];
  impl_->render_params.volsize[1]            = vd->getSize()[1];
  impl_->render_params.volsize[2]            = vd->getSize()[2];
  impl_->render_params.volpos[0]             = vd->pos[0];
  impl_->render_params.volpos[1]             = vd->pos[1];
  impl_->render_params.volpos[2]             = vd->pos[2];
  impl_->render_params.bpc                   = vd->bpc;
  impl_->render_params.lutsize               = getLUTSize(vd);
  impl_->render_params.quality               = getParameter(vvRenderer::VV_QUALITY);
  impl_->render_params.filter_mode           = static_cast< virvo::tex_filter_mode >(getParameter(vvRenderer::VV_SLICEINT).asInt());
  impl_->render_params.opacity_correction    = getParameter(vvRenderer::VV_OPCORR);
  impl_->render_params.early_ray_termination = getParameter(vvRenderer::VV_TERMINATEEARLY);
  impl_->render_params.mip_mode              = getParameter(vvRenderer::VV_MIP_MODE);

  wake_render_threads(impl_->render_params, &impl_->sync_params);
}

void vvSoftRayRend::updateTransferFunction()
{

  size_t lutEntries = getLUTSize(vd);
  impl_->rgbaTF.resize(4 * lutEntries);

  vd->computeTFTexture(lutEntries, 1, 1, &impl_->rgbaTF[0]);

}


bool vvSoftRayRend::checkParameter(ParameterType param, vvParam const& value) const
{
  switch (param)
  {
  case VV_SLICEINT:

    {
      virvo::tex_filter_mode mode = static_cast< virvo::tex_filter_mode >(value.asInt());

      if (mode == virvo::Nearest || mode == virvo::Linear
       || mode == virvo::BSpline || mode == virvo::BSplineInterpol
       || mode == virvo::CardinalSpline)
      {
        return true;
      }
    }

    return false;;

  default:

    return vvRenderer::checkParameter(param, value);

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

template < typename VoxelT >
void renderTile
(
    const virvo::Tile& tile,
    virvo::texture< VoxelT, virvo::ElementType, 3 > const& volume,
    const Thread* thread)
{
  static const float_type opacityThreshold = 0.95f;

  Mat4 inv_view_matrix(*thread->invViewMatrix);
  int w                         = thread->render_params->width;
  int h                         = thread->render_params->height;

  Vec3s vox(thread->render_params->vox[0], thread->render_params->vox[1], thread->render_params->vox[2]);
  Vec3 fvox(thread->render_params->vox[0], thread->render_params->vox[1], thread->render_params->vox[2]);

  Vec3 size(thread->render_params->volsize[0], thread->render_params->volsize[1], thread->render_params->volsize[2]);
  Vec3 invsize                  = float_type(1.0f) / size;
  Vec3 size2                    = size * float_type(0.5f);
  Vec3 volpos(thread->render_params->volpos[0], thread->render_params->volpos[1], thread->render_params->volpos[2]);

  size_t const lutsize          = thread->render_params->lutsize;

  float quality                 = thread->render_params->quality;

  bool opacityCorrection        = thread->render_params->opacity_correction;
  bool earlyRayTermination      = thread->render_params->early_ray_termination;
  int mipMode                   = thread->render_params->mip_mode;

  AABB const aabb
  (
    Vec3(thread->render_params->mincorner[0], thread->render_params->mincorner[1], thread->render_params->mincorner[2]),
    Vec3(thread->render_params->maxcorner[0], thread->render_params->maxcorner[1], thread->render_params->maxcorner[2])
  );

  const float diagonalVoxels = sqrtf(float(thread->render_params->vox[0] * thread->render_params->vox[0] +
                                           thread->render_params->vox[1] * thread->render_params->vox[1] +
                                           thread->render_params->vox[2] * thread->render_params->vox[2]));


  typedef virvo::vector< 4, float > float4;
  virvo::texture< float4, virvo::ElementType, 1 > tf(lutsize);
  tf.data = reinterpret_cast< float4* >( &(*thread->rgbaTF)[0] );
  tf.set_address_mode(virvo::Clamp);
  tf.set_filter_mode( virvo::Linear );

  size_t numSlices = std::max(size_t(1), static_cast<size_t>(quality * diagonalVoxels));

  for (int y = tile.bottom; y < tile.top; y += PACK_SIZE_Y)
  {
    for (int x = tile.left; x < tile.right; x += PACK_SIZE_X)
    {
      //
      // 0           t           1      continuous
      // |---+---|---+---|---+---|
      // 0       1=x     2       3=w    discrete
      //
      // t = 1/w (x + 1/2)   where   0 <= t <= 1
      // u = 2 t - 1         where  -1 <= u <= 1
      //

      const float_type u = 2.0f * (pixelx(x) + 0.5f) / float_type(w) - 1.0f;
      const float_type v = 2.0f * (pixely(y) + 0.5f) / float_type(h) - 1.0f;

      Vec4 o(u, v, -1.0f, 1.0f);
      o = inv_view_matrix * o;
      Vec4 d(u, v, 1.0f, 1.0f);
      d = inv_view_matrix * d;

      Ray ray(o.xyz() / o.w, d.xyz() / d.w);
      ray.dir = normalize( ray.dir - ray.ori );

      virvo::hit_record< Ray, AABB > hr = intersect(ray, aabb);
      float_type active = hr.hit;
      if (any(active))
      {

        float_type dist = diagonalVoxels / float_type(numSlices);
        float_type t = hr.tnear;
        Vec4 dst(0.0f);

        while (any(active))
        {
          Vec3 pos = ray.ori + ray.dir * t;
          Vec3 texcoord((pos[0] - volpos[0] + size2[0]) * invsize[0],
                       (-pos[1] - volpos[1] + size2[1]) * invsize[1],
                       (-pos[2] - volpos[2] + size2[2]) * invsize[2]);

          float_type sample = virvo::tex3D(volume, texcoord);
          sample /= float_type( std::numeric_limits< VoxelT >::max() );

          Vec4 src = tex1D(tf, sample);

          if (mipMode == 1)
          {
            dst[0] = max(src[0], dst[0]);
            dst[1] = max(src[1], dst[1]);
            dst[2] = max(src[2], dst[2]);
            dst[3] = max(src[3], dst[3]);
          }
          else if (mipMode == 2)
          {
            dst[0] = min(src[0], dst[0]);
            dst[1] = min(src[1], dst[1]);
            dst[2] = min(src[2], dst[2]);
            dst[3] = min(src[3], dst[3]);
          }

          if (opacityCorrection)
          {
            src[3] = 1 - pow(1 - src[3], dist);
          }

          if (mipMode == 0)
          {
            // pre-multiply alpha
            src[0] *= src[3];
            src[1] *= src[3];
            src[2] *= src[3];
          }

          if (mipMode == 0)
          {
            dst = dst + mul(src, sub(1.0f, dst[3], active), active);
          }

          if (earlyRayTermination)
          {
            active = active && dst[3] <= opacityThreshold;
          }

          t += dist;
          active = active && (t < hr.tfar);
        }

#if VV_USE_SSE
        // transform to AoS for framebuffer write
        dst = transpose(dst);
        store(&(*thread->colors)[y * w * 4 + x * 4], dst.x);
        if (x + 1 < tile.right)
        {
          store(&(*thread->colors)[y * w * 4 + (x + 1) * 4], dst.y);
        }
        if (y + 1 < tile.top)
        {
          store(&(*thread->colors)[(y + 1) * w * 4 + x * 4], dst.z);
        }
        if (x + 1 < tile.right && y + 1 < tile.top)
        {
          store(&(*thread->colors)[(y + 1) * w * 4 + (x + 1) * 4], dst.w);
        }
#else
        memcpy(&(*thread->colors)[y * w * 4 + x * 4], &dst[0], 4 * sizeof(float));
#endif
      }
    }
  }
}


void wake_render_threads(Thread::RenderParams rparams, Thread::SyncParams* sparams)
{
  int w = rparams.rect.right - rparams.rect.left;
  int h = rparams.rect.top   - rparams.rect.bottom;

  int tilew = virvo::tile_width;
  int tileh = virvo::tile_height;

  int numtilesx = DIV_UP(w, tilew);
  int numtilesy = DIV_UP(h, tileh);

  atom_lock_test_and_set(&sparams->tile_idx_counter, 0);
  atom_lock_test_and_set(&sparams->tile_fin_counter, 0);
  atom_lock_test_and_set(&sparams->tile_idx_max, numtilesx * numtilesy);

  sparams->start_render.broadcast();

  sparams->image_ready.wait();
}


template < typename T >
void render
(
    virvo::texture< T, virvo::ElementType, 3 > const& volume,
    Thread* thread
)
{
  while (true)
  {
    long tile_idx = atom_fetch_and_add(&thread->sync_params->tile_idx_counter, 1);

    if (tile_idx >= thread->sync_params->tile_idx_max)
    {
      break;
    }

    int w = thread->render_params->rect.right - thread->render_params->rect.left;

    int tilew = virvo::tile_width;
    int tileh = virvo::tile_height;

    int numtilesx = DIV_UP(w, tilew);

    virvo::Tile t;
    t.left   = thread->render_params->rect.left   + (tile_idx % numtilesx) * tilew;
    t.bottom = thread->render_params->rect.bottom + (tile_idx / numtilesx) * tileh;
    t.right  = std::min(t.left   + tilew, thread->render_params->rect.right);
    t.top    = std::min(t.bottom + tileh, thread->render_params->rect.top);

    renderTile(t, volume, thread);

    long num_tiles_fin = atom_fetch_and_add(&thread->sync_params->tile_fin_counter, 1);

    if (num_tiles_fin >= thread->sync_params->tile_idx_max - 1)
    {
      assert(num_tiles_fin == thread->sync_params->tile_idx_max - 1);

      // the last tile has just been rendered.
      // wake up the main thread waiting for the image.
      thread->sync_params->image_ready.signal();
      break;
    }
  }
}

void* renderFunc(void* args)
{

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

  virvo::texture< uint8_t,  virvo::ElementType, 3 > volume8;
  virvo::texture< uint16_t, virvo::ElementType, 3 > volume16;

  while (true)
  {

    thread->sync_params->start_render.wait();
    if (thread->sync_params->exit_render_loop)
    {
      break;
    }

    // TODO: volume at class scope, not one copy per thread.
    // This probably necessitates to make the whole renderer a template. Arghh.

    // TODO: currently, each set_filter_mode(BSplineInterpol) will invoke
    // the b-spline prefilter.

    if (thread->render_params->bpc == 1)
    {
        volume8.resize
        (
            thread->render_params->vox[0],
            thread->render_params->vox[1],
            thread->render_params->vox[2]
        );
        volume8.data = reinterpret_cast< uint8_t* >(*thread->raw);
        volume8.set_address_mode(virvo::Clamp);
        volume8.set_filter_mode( thread->render_params->filter_mode );
        render(volume8, thread);
    }
    else if (thread->render_params->bpc == 2)
    {
        volume16.resize
        (
            thread->render_params->vox[0],
            thread->render_params->vox[1],
            thread->render_params->vox[2]
        );
        volume16.data = reinterpret_cast< uint16_t* >(*thread->raw);
        volume16.set_address_mode(virvo::Clamp);
        volume16.set_filter_mode( thread->render_params->filter_mode );
        render(volume16, thread);
    }

  }

  pthread_exit(NULL);
  return NULL;
}

vvRenderer* createRayRend(vvVolDesc* vd, vvRenderState const& rs)
{
  return new vvSoftRayRend(vd, rs);
}

