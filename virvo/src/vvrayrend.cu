#include "vvcudautils.h"
#include "vvdebugmsg.h"
#include "vvglew.h"
#include "vvgltools.h"
#include "vvrayrend.h"

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_gl_interop.h>
#include <iostream>

using std::cerr;
using std::endl;

texture<uchar, 3, cudaReadModeNormalizedFloat> volTexture;
texture<float4, 1, cudaReadModeElementType> tfTexture;

GLuint pbo = 0;     // OpenGL pixel buffer object
GLuint gltex = 0;     // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource; // CUDA Graphics Resource (to transfer PBO)

cudaArray* d_volumeArray = 0;

void initPbo()
{
  vvGLTools::Viewport vp = vvGLTools::getViewport();
  const int width = vp[2];
  const int height = vp[3];

  const int bitsPerPixel = 4;
  const int bufferSize = sizeof(GLubyte) * width * height * bitsPerPixel;
  GLubyte* pboSrc = (GLubyte*)malloc(bufferSize);
  glGenBuffers(1, &pbo);
  glBindBuffer(GL_ARRAY_BUFFER, pbo);
  glBufferData(GL_ARRAY_BUFFER, bufferSize, pboSrc, GL_DYNAMIC_DRAW);
  free(pboSrc);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  cudaGLRegisterBufferObject(pbo);

  glGenTextures(1, &gltex);
  glBindTexture(GL_TEXTURE_2D, gltex);

  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA,
               GL_UNSIGNED_BYTE, NULL);
}

void renderQuad(const int width, const int height)
{
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_LIGHTING);
    glEnable(GL_TEXTURE_2D);
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);

    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glLoadIdentity();
    glOrtho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

    glMatrixMode( GL_MODELVIEW);
    glLoadIdentity();

    glViewport(0, 0, width, height);

    glClear(GL_COLOR_BUFFER_BIT);
    glBegin(GL_QUADS);
        glTexCoord2f(0.0, 0.0); glVertex3f(-1.0, -1.0, 0.5);
        glTexCoord2f(1.0, 0.0); glVertex3f(1.0, -1.0, 0.5);
        glTexCoord2f(1.0, 1.0); glVertex3f(1.0, 1.0, 0.5);
        glTexCoord2f(0.0, 1.0); glVertex3f(-1.0, 1.0, 0.5);
    glEnd();

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();

    glDisable(GL_TEXTURE_2D);
}

int iDivUp(int a, int b)
{
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

typedef struct
{
  float m[4][4];
} matrix4x4;

__constant__ matrix4x4 c_invViewMatrix;

struct Ray
{
  float3 o;
  float3 d;
};

__device__ float volume(const float x, const float y, const float z)
{
  return tex3D(volTexture, x, y, z);
}

__device__ float volume(const float3& pos)
{
  return tex3D(volTexture, pos.x, pos.y, pos.z);
}

__device__ bool intersectBox(const Ray& r, const float3& boxmin, const float3& boxmax,
                             float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f, 1.0f, 1.0f) / r.d;
    float t1 = (boxmin.x - r.o.x) * invR.x;
    float t2 = (boxmax.x - r.o.x) * invR.x;
    float tmin = fminf(t1, t2);
    float tmax = fmaxf(t1, t2);

    t1 = (boxmin.y - r.o.y) * invR.y;
    t2 = (boxmax.y - r.o.y) * invR.y;
    tmin = fmaxf(fminf(t1, t2), tmin);
    tmax = fminf(fmaxf(t1, t2), tmax);

    t1 = (boxmin.z - r.o.z) * invR.z;
    t2 = (boxmax.z - r.o.z) * invR.z;
    tmin = fmaxf(fminf(t1, t2), tmin);
    tmax = fminf(fmaxf(t1, t2), tmax);

    *tnear = tmin;
    *tfar = tmax;

    return ((tmax >= tmin) && (tmax >= 0.0f));
}

__device__ float4 mul(const matrix4x4& M, const float4& v)
{
    float4 result;
    result.x = M.m[0][0] * v.x + M.m[0][1] * v.y + M.m[0][2] * v.z + M.m[0][3] * v.w;
    result.y = M.m[1][0] * v.x + M.m[1][1] * v.y + M.m[1][2] * v.z + M.m[1][3] * v.w;
    result.z = M.m[2][0] * v.x + M.m[2][1] * v.y + M.m[2][2] * v.z + M.m[2][3] * v.w;
    result.w = M.m[3][0] * v.x + M.m[3][1] * v.y + M.m[3][2] * v.z + M.m[3][3] * v.w;

//    result.x = M.m[0][0] * v.x + M.m[1][0] * v.y + M.m[2][0] * v.z + M.m[3][0] * v.w;
//    result.y = M.m[0][1] * v.x + M.m[1][1] * v.y + M.m[2][1] * v.z + M.m[3][1] * v.w;
//    result.z = M.m[0][2] * v.x + M.m[1][2] * v.y + M.m[2][2] * v.z + M.m[3][2] * v.w;
//    result.w = M.m[0][3] * v.x + M.m[1][3] * v.y + M.m[2][3] * v.z + M.m[3][3] * v.w;

    return result;
}

__device__ float3 perspectiveDivide(const float4& v)
{
    const float wInv = 1.0f / v.w;
    return make_float3(v.x * wInv, v.y * wInv, v.z * wInv);
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    clamp(rgba.x);
    clamp(rgba.y);
    clamp(rgba.z);
    clamp(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

__device__ uint rgbaFloatToInt(float3 rgb)
{
  float4 rgba = make_float4(rgb.x, rgb.y, rgb.z, 1.0f);
  return rgbaFloatToInt(rgba);
}

__device__ float4 phong(const float4& classification, const float3& pos,
                        const float3& L, const float3& H,
                        const float3& Ka, const float3& Kd, const float3& Ks,
                        const float shininess)
{
  float3 N;
  float3 sample1;
  float3 sample2;
  const float DELTA = 0.01f;
  sample1.x = volume(pos - make_float3(DELTA, 0.0f, 0.0f));
  sample2.x = volume(pos + make_float3(DELTA, 0.0f, 0.0f));
  sample1.y = volume(pos - make_float3(0.0f, DELTA, 0.0f));
  sample2.y = volume(pos + make_float3(0.0f, DELTA, 0.0f));
  sample1.z = volume(pos - make_float3(0.0f, 0.0f, DELTA));
  sample2.z = volume(pos + make_float3(0.0f, 0.0f, DELTA));

  N = normalize(sample2 - sample1);
  const float diffuse = fabsf(dot(L, N));
  const float specular = powf(dot(H, N), shininess);

  const float3 c = make_float3(classification);
  float3 tmp = Ka * c + Kd * diffuse * c;
  if (specular > 0.0f)
  {
    tmp += Ks * specular * c;
  }
  return make_float4(tmp.x, tmp.y, tmp.z, classification.w);
}

template<bool frontToBack, bool lighting>
__global__ void render(uint *d_output, const uint width, const uint height, const float dist,
                       const float3 volSizeHalf, const float3 L, const float3 H)
{
  const int maxSteps = INT_MAX;
  const float tstep = dist;
  const float opacityThreshold = 0.95f;
  const float3 boxMin = make_float3(-volSizeHalf.x, -volSizeHalf.y, -volSizeHalf.z);
  const float3 boxMax = make_float3(volSizeHalf.x, volSizeHalf.y, volSizeHalf.z);

  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  const uint y = blockIdx.y * blockDim.y + threadIdx.y;

  if ((x >= width) || (y >= height))
  {
    return;
  }

  const float u = (x / static_cast<float>(width)) * 2.0f - 1.0f;
  const float v = (y / static_cast<float>(height)) * 2.0f - 1.0f;

  /*
   * Rays like if the view were orthographic with origins over each pixel.
   * These are multiplied with the inverse modelview projection matrix.
   * First of all, the rays will be transformed to fit to the frustum.
   * Then the rays will be oriented so that they can hit the volume.
   */
  const float4 o = mul(c_invViewMatrix, make_float4(u, v, -1.0f, 1.0f));
  const float4 d = mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 1.0f, 1.0f));

  Ray eyeRay;
  eyeRay.o = perspectiveDivide(o);
  eyeRay.d = perspectiveDivide(d);
  eyeRay.d = normalize(eyeRay.d);

  float tnear, tfar;
  int hit = intersectBox(eyeRay, boxMin, boxMax, &tnear, &tfar);
  if (!hit)
  {
    d_output[y * width + x] = 0;
    return;
  }

  if (tnear < 0.0f)
  {
    tnear = 0.0f;
  }

  float4 dst = make_float4(0.0f);
  float t = tnear;
  float3 pos = eyeRay.o + eyeRay.d * tnear;
  const float3 step = eyeRay.d * tstep;

  for (int i=0; i<maxSteps; ++i)
  {
    const float3 texCoord = make_float3((pos.x + volSizeHalf.x) / (volSizeHalf.x * 2),
                                        (pos.y + volSizeHalf.y) / (volSizeHalf.y * 2),
                                        (pos.z + volSizeHalf.z) / (volSizeHalf.z * 2));
    const float sample = volume(texCoord);

    // Post-classification transfer-function lookup.
    float4 src = tex1D(tfTexture, sample);

    // Local illumination.
    if (lighting && (src.w > 0.1))
    {
      const float3 Ka = make_float3(0.0f, 0.0f, 0.0f);
      const float3 Kd = make_float3(0.8f, 0.8f, 0.8f);
      const float3 Ks = make_float3(0.8f, 0.8f, 0.8f);
      const float shininess = 1000.0f;
      src = phong(src, texCoord, L, H, Ka, Kd, Ks, shininess);
    }

    // "under" operator for back-to-front blending
    //dst = lerp(dst, src, src.w);

    // pre-multiply alpha
    src.x *= src.w;
    src.y *= src.w;
    src.z *= src.w;

    if (frontToBack)
    {
      dst = dst + src * (1.0f - dst.w);
    }

    // Early ray termination.
    if (dst.w > opacityThreshold)
    {
      break;
    }

    t += tstep;
    if (t > tfar)
    {
      break;
    }

    pos += step;
  }
  d_output[y * width + x] = rgbaFloatToInt(dst);
}

vvRayRend::vvRayRend(vvVolDesc* vd, vvRenderState renderState)
  : vvRenderer(vd, renderState)
{
  glewInit();
  cudaGLSetGLDevice(0);
  initPbo();

  cudaExtent volumeSize = make_cudaExtent(vd->vox[0], vd->vox[1], vd->vox[2]);

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar>();
  cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize);

  cudaMemcpy3DParms copyParams = { 0 };
  copyParams.srcPtr = make_cudaPitchedPtr(vd->getRaw(0), volumeSize.width*sizeof(uchar), volumeSize.width, volumeSize.height);
  copyParams.dstArray = d_volumeArray;
  copyParams.extent = volumeSize;
  copyParams.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&copyParams);

  interpolation = true;
  volTexture.normalized = true;
  if (interpolation)
  {
    volTexture.filterMode = cudaFilterModeLinear;
  }
  else
  {
    volTexture.filterMode = cudaFilterModePoint;
  }
  volTexture.addressMode[0] = cudaAddressModeClamp;
  volTexture.addressMode[1] = cudaAddressModeClamp;

  // bind array to 3D texture
  cudaBindTextureToArray(volTexture, d_volumeArray, channelDesc);

  d_transferFuncArray = 0;
  updateTransferFunction();
}

vvRayRend::~vvRayRend()
{
  cudaFreeArray(d_transferFuncArray);
}

int vvRayRend::getLUTSize() const
{
   vvDebugMsg::msg(2, "vvSoftVR::getLUTSize()");
   return (vd->getBPV()==2) ? 4096 : 256;
}

void vvRayRend::updateTransferFunction()
{
  int lutEntries = getLUTSize();
  float* rgba = new float[4 * lutEntries];

  vd->computeTFTexture(lutEntries, 1, 1, rgba);

  cudaChannelFormatDesc channelDesc2 = cudaCreateChannelDesc<float4>();

  cudaFreeArray(d_transferFuncArray);
  cudaMallocArray(&d_transferFuncArray, &channelDesc2, lutEntries, 1);
  cudaMemcpyToArray(d_transferFuncArray, 0, 0, rgba, lutEntries * 4 * sizeof(float), cudaMemcpyHostToDevice);

  tfTexture.filterMode = cudaFilterModeLinear;
  tfTexture.normalized = true;    // access with normalized texture coordinates
  tfTexture.addressMode[0] = cudaAddressModeClamp;   // wrap texture coordinates

  cudaBindTextureToArray(tfTexture, d_transferFuncArray, channelDesc2);

  delete[] rgba;
}

void vvRayRend::renderVolumeGL()
{
  vvDebugMsg::msg(1, "vvRayRend::renderVolumeGL()");

  const vvGLTools::Viewport vp = vvGLTools::getViewport();
  const int width = vp[2];
  const int height = vp[3];

  uint *d_output = 0;
  // map PBO to get CUDA device pointer
  cudaGLMapBufferObject((void**)&d_output, pbo);

  dim3 blockSize(16, 16);
  dim3 gridSize = dim3(iDivUp(width, blockSize.x), iDivUp(height, blockSize.y));

  const vvVector3 size(vd->getSize());
  const vvVector3 size2 = size * 0.5f;

  // Assume: no probe.
  vvVector3 probeSizeObj;
  probeSizeObj.copy(&size);
  vvVector3 probeMin;
  probeMin = -size2;
  vvVector3 probeMax;
  probeMax = size2;

  vvVector3 clippedProbeSizeObj;
  clippedProbeSizeObj.copy(&probeSizeObj);
  for (int i=0; i<3; ++i)
  {
    if (clippedProbeSizeObj[i] < vd->getSize()[i])
    {
      clippedProbeSizeObj[i] = vd->getSize()[i];
    }
  }

  const float diagonal = sqrtf(clippedProbeSizeObj[0] * clippedProbeSizeObj[0] +
                               clippedProbeSizeObj[1] * clippedProbeSizeObj[1] +
                               clippedProbeSizeObj[2] * clippedProbeSizeObj[2]);

  const float diagonalVoxels = sqrtf(float(vd->vox[0] * vd->vox[0] +
                                           vd->vox[1] * vd->vox[1] +
                                           vd->vox[2] * vd->vox[2]));
  int numSlices = max(1, static_cast<int>(_renderState._quality * diagonalVoxels));
  //std::cerr << diagonalVoxels << " " << numSlices << std::endl;

  // Inverse modelview-projection matrix.
  vvMatrix mvp, pr;
  getModelviewMatrix(&mvp);

  // Not related.
  vvMatrix invMV;
  invMV.copy(&mvp);
  invMV.invert();
  // Not related.

  getProjectionMatrix(&pr);
  mvp.multiplyPost(&pr);
  mvp.invert();

  float* viewM = new float[16];
  mvp.get(viewM);
  cudaMemcpyToSymbol(c_invViewMatrix, viewM, sizeof(float4) * 4);
  delete[] viewM;

  float3 volSize;

  if ((vd->vox[0] >= vd->vox[1]) && (vd->vox[0] >= vd->vox[2]))
  {
    const float inv = 1.0f / vd->vox[0];
    volSize = make_float3(1.0f, vd->vox[1] * inv, vd->vox[2] * inv);
  }
  else if ((vd->vox[1] >= vd->vox[0]) && (vd->vox[1] >= vd->vox[2]))
  {
    const float inv = 1.0f / vd->vox[1];
    volSize = make_float3(vd->vox[0] * inv, 1.0f, vd->vox[2] * inv);
  }
  else
  {
    const float inv = 1.0f / vd->vox[2];
    volSize = make_float3(vd->vox[0] * inv, vd->vox[1] * inv, 1.0f);
  }

  bool isOrtho = pr.isProjOrtho();

  vvVector3 eye;
  getEyePosition(&eye);
  eye.multiply(&invMV);

  vvVector3 origin;

  vvVector3 normal;
  getObjNormal(normal, origin, eye, invMV, isOrtho);

  const float3 N = make_float3(normal[0], normal[1], normal[2]);

  const float3 L(-N);

  // Viewing direction.
  const float3 V(-N);

  // Half way vector.
  const float3 H = normalize(L + V);
  render<true, true><<<gridSize, blockSize>>>(d_output, width, height, 2.0f / (float)numSlices, volSize * 0.5f, L, H);
  cudaGLUnmapBufferObject(pbo);

  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
  glBindTexture(GL_TEXTURE_2D, gltex);
  glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glBindBuffer(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

  renderQuad(width, height);
}
