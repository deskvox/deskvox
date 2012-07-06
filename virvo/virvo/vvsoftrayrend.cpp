#include "vvaabb.h"
#include "vvdebugmsg.h"
#include "vvsoftrayrend.h"
#include "vvvoldesc.h"

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#if HAVE_GL
#include "vvgltools.h"
#endif

struct Ray
{
  vvVector3 o;
  vvVector3 d;
};

bool intersectBox(const Ray& ray, const vvAABB& aabb,
                  float* tnear, float* tfar)
{
  // compute intersection of ray with all six bbox planes
  vvVector3 invR(1.0f / ray.d[0], 1.0f / ray.d[1], 1.0f / ray.d[2]);
  float t1 = (aabb.getMin()[0] - ray.o[0]) * invR[0];
  float t2 = (aabb.getMax()[0] - ray.o[0]) * invR[0];
  float tmin = fminf(t1, t2);
  float tmax = fmaxf(t1, t2);

  t1 = (aabb.getMin()[1] - ray.o[1]) * invR[1];
  t2 = (aabb.getMax()[1] - ray.o[1]) * invR[1];
  tmin = fmaxf(fminf(t1, t2), tmin);
  tmax = fminf(fmaxf(t1, t2), tmax);

  t1 = (aabb.getMin()[2] - ray.o[2]) * invR[2];
  t2 = (aabb.getMax()[2] - ray.o[2]) * invR[2];
  tmin = fmaxf(fminf(t1, t2), tmin);
  tmax = fminf(fmaxf(t1, t2), tmax);

  *tnear = tmin;
  *tfar = tmax;

  return ((tmax >= tmin) && (tmax >= 0.0f));
}

vvSoftRayRend::vvSoftRayRend(vvVolDesc* vd, vvRenderState renderState)
  : vvRenderer(vd, renderState)
  , _opacityCorrection(false)
{
  updateTransferFunction();
}

vvSoftRayRend::~vvSoftRayRend()
{

}

void vvSoftRayRend::renderVolumeGL()
{
  vvMatrix mv;
  vvMatrix pr;

#if HAVE_GL
  vvGLTools::getModelviewMatrix(&mv);
  vvGLTools::getProjectionMatrix(&pr);
#endif

  // hardcoded
  const int W = 512;
  const int H = 512;

  vvMatrix invMvpr = mv;
  invMvpr.multiplyLeft(pr);
  invMvpr.invert();

  vvAABB aabb = vvAABB(vvVector3(), vvVector3());
  vd->getBoundingBox(aabb);

  vvVector3 size2 = vd->getSize() * 0.5f;
  const float diagonalVoxels = sqrtf(float(vd->vox[0] * vd->vox[0] +
                                           vd->vox[1] * vd->vox[1] +
                                           vd->vox[2] * vd->vox[2]));
  int numSlices = std::max(1, static_cast<int>(_quality * diagonalVoxels));

  std::vector<float> colors;
  colors.resize(W * H * 4);
  uchar* raw = vd->getRaw(0);
  for (int y = 0; y < H; ++y)
  {
    for (int x = 0; x < W; ++x)
    {
      const float u = (x / static_cast<float>(W)) * 2.0f - 1.0f;
      const float v = (y / static_cast<float>(H)) * 2.0f - 1.0f;

      vvVector4 o(u, v, -1.0f, 1.0f);
      o.multiply(invMvpr);
      vvVector4 d(u, v, 1.0f, 1.0f);
      d.multiply(invMvpr);

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
          vvVector3i texcoordi = vvVector3i(int(texcoord[0] * float(vd->vox[0] - 1)),
                                            int(texcoord[1] * float(vd->vox[1] - 1)),
                                            int(texcoord[2] * float(vd->vox[2] - 1)));
          int idx = texcoordi[2] * vd->vox[0] * vd->vox[1] + texcoordi[1] * vd->vox[0] + texcoordi[0];
          float sample = float(raw[idx]) / 256.0f;
          vvVector4 src(_rgbaTF[int(sample * 4 * getLUTSize())],
                        _rgbaTF[int(sample * 4 * getLUTSize()) + 1],
                        _rgbaTF[int(sample * 4 * getLUTSize()) + 2],
                        _rgbaTF[int(sample * 4 * getLUTSize()) + 3]);

          if (_opacityCorrection)
          {
            src[3] = 1 - powf(1 - src[3], dist);
          }

          // pre-multiply alpha
          src[0] *= src[3];
          src[1] *= src[3];
          src[2] *= src[3];

          dst = dst + src * (1.0f - dst[3]);

          t += dist;
          if (t > tbfar)
          {
            break;
          }
          pos += step;
        }

        for (int c = 0; c < 4; ++c)
        {
          colors[y * W * 4 + x * 4 + c] = dst[c];
        }
      }
    }
  }

#if HAVE_GL
  glWindowPos2i(0, 0);
  glDrawPixels(W, H, GL_RGBA, GL_FLOAT, &colors[0]);
#endif
}

void vvSoftRayRend::updateTransferFunction()
{
  int lutEntries = getLUTSize();
  delete[] _rgbaTF;
  _rgbaTF = new float[4 * lutEntries];

  vd->computeTFTexture(lutEntries, 1, 1, _rgbaTF);
}

int vvSoftRayRend::getLUTSize() const
{
   vvDebugMsg::msg(3, "vvSoftRayRend::getLUTSize()");
   return (vd->getBPV()==2) ? 4096 : 256;
}

