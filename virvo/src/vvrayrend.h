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

#include "vvcuda.h"

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
  void setParameter(ParameterType param, float newValue);
  void setNumSpaceSkippingCells(const int numSpaceSkippingCells[3]);

  bool getEarlyRayTermination() const;
  bool getIllumination() const;
  bool getInterpolation() const;
  bool getOpacityCorrection() const;
  bool getSpaceSkipping() const;
private:
  cudaChannelFormatDesc _channelDesc;
  cudaArray** d_volumeArrays;
  cudaArray* d_transferFuncArray;
  cudaArray* d_randArray;
  cudaArray* d_spaceSkippingArray;

  bool* h_spaceSkippingArray;
  int* h_cellMinValues;
  int* h_cellMaxValues;
  int h_numCells[3];

  float* _rgbaTF;

  bool _earlyRayTermination;        ///< Terminate ray marching when enough alpha was gathered
  bool _illumination;               ///< Use local illumination
  bool _interpolation;              ///< interpolation mode: true=linear interpolation (default), false=nearest neighbor
  bool _opacityCorrection;          ///< true = opacity correction on
  bool _spaceSkipping;              ///< true = skip over homogeneous regions
  bool _volumeCopyToGpuOk;          ///< must be true for memCopy to be run

  void initRandTexture();
  void initSpaceSkippingTexture();
  void initVolumeTexture();
  void factorViewMatrix();
  void findAxisRepresentations();

  void calcSpaceSkippingGrid();
  void computeSpaceSkippingTexture();
};

#endif // HAVE_CUDA

#endif
