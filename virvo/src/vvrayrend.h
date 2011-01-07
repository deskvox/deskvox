//
// This software contains source code provided by NVIDIA Corporation.
//

#ifndef _VV_RAYREND_H_
#define _VV_RAYREND_H_

#include "vvexport.h"
#include "vvopengl.h"
#include "vvsoftvr.h"
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
class VIRVOEXPORT vvRayRend : public vvSoftVR
{
public:
  vvRayRend(vvVolDesc* vd, vvRenderState renderState);
  ~vvRayRend();

  int getLUTSize() const;
  void updateTransferFunction();
  void compositeVolume(int = -1, int = -1);
  void setParameter(const ParameterType, const float, char* = NULL);

  bool getEarlyRayTermination() const;
  bool getIllumination() const;
  bool getInterpolation() const;
  bool getOpacityCorrection() const;
private:
  cudaArray* d_volumeArray;
  cudaArray* d_transferFuncArray;
  cudaArray* d_randArray;

  bool _earlyRayTermination;        ///< Terminate ray marching when enough alpha was gathered
  bool _illumination;               ///< Use local illumination
  bool _interpolation;              ///< interpolation mode: true=linear interpolation (default), false=nearest neighbor
  bool _opacityCorrection;          ///< true = opacity correction on

  void initPbo(int width, int height);
  void initRandTexture();
  void initVolumeTexture();
  void factorViewMatrix();
};

#endif // HAVE_CUDA

#endif
