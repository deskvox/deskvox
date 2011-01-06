//
// This software contains source code provided by NVIDIA Corporation.
//

#ifndef _VV_RAYREND_H_
#define _VV_RAYREND_H_

#include "vvexport.h"
#include "vvopengl.h"
#include "vvrenderer.h"
#include "vvtransfunc.h"
#include "vvvoldesc.h"

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#if defined(HAVE_CUDA) && defined(NV_PROPRIETARY_CODE)

/** Ray casting renderer. Based on the volume
  rendering implementation from the NVIDIA CUDA SDK,
  as of November 25th, 2010 could be downloaded from
  the following location:
  http://developer.download.nvidia.com/compute/cuda/sdk/website/samples.html
 */
class VIRVOEXPORT vvRayRend : public vvRenderer
{
public:
  vvRayRend(vvVolDesc* vd, vvRenderState renderState);
  ~vvRayRend();

  int getLUTSize() const;
  void updateTransferFunction();
  void resize(int width, int height);
  void renderVolumeGL();
  void setParameter(const ParameterType, const float, char* = NULL);
private:
  cudaArray* d_volumeArray;
  cudaArray* d_transferFuncArray;
  cudaArray* d_randArray;

  GLuint _pbo;                      ///< gl pbo object
  GLuint _gltex;                    ///< texture associated with \see _pbo

  bool _earlyRayTermination;        ///< Terminate ray marching when enough alpha was gathered
  bool _illumination;               ///< Use local illumination
  bool _interpolation;              ///< interpolation mode: true=linear interpolation (default), false=nearest neighbor
  bool _opacityCorrection;          ///< true = opacity correction on

  void initPbo(int width, int height);
  void initRandTexture();
  void initVolumeTexture();
  void renderQuad(int width, int height) const;
};

#endif // HAVE_CUDA

#endif
