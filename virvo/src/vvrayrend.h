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
  void  renderVolumeGL();
private:
  bool interpolation;                           ///< interpolation mode: true=linear interpolation (default), false=nearest neighbor
  cudaArray* d_transferFuncArray;
};

#endif // HAVE_CUDA

#endif
