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

#ifdef HAVE_CUDA

class VIRVOEXPORT vvRayRend : public vvRenderer
{
public:
  vvRayRend(vvVolDesc* vd, vvRenderState renderState);
  ~vvRayRend();

  int getLUTSize() const;
  void updateTransferFunction();
  void resize(int width, int height);
  void renderVolumeGL();
private:
  bool _interpolation;              ///< interpolation mode: true=linear interpolation (default), false=nearest neighbor
  cudaArray* d_transferFuncArray;
  cudaArray* d_randArray;

  GLuint _pbo;                      ///< gl pbo object
  GLuint _gltex;                    ///< texture associated with \see _pbo

  void initPbo(int width, int height);
  void initRandTexture();
  void initVolumeTexture();
  void renderQuad(int width, int height) const;
};

#endif // HAVE_CUDA

#endif
