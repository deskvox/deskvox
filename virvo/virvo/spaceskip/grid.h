// Grid accelerator from Ospray
//
// Adapted from:
// https://github.com/ospray/ospray/blob/master/ospray/volume/structured/GridAccelerator.ih
// (accessed: 3/19/2019)
//
// Original license follows
//

// ======================================================================== //
// Copyright 2009-2019 Intel Corporation                                    //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#ifndef VV_SPACESKIP_GRID_H
#define VV_SPACESKIP_GRID_H

#undef MATH_NAMESPACE

#include <visionaray/math/forward.h>
#include <visionaray/math/vector.h>
#include <visionaray/texture/texture.h>

#undef MATH_NAMESPACE

class vvVolDesc;

namespace virvo
{

//! \brief A spatial acceleration structure over a BlockBrickedVolume, used
//!  for opacity and variance based space skipping.
//!
class GridAccelerator
{
public:
  typedef visionaray::texture_ref<visionaray::vec4, 1> TransfuncTex;

public:

  //! The range of volumetric values within a grid cell.
  visionaray::vec2f* cellRange;

  //! Grid size in cells per dimension.
  visionaray::vec3i gridDimensions;

  float* maxOpacities;

  GridAccelerator();
 ~GridAccelerator();

  void updateVolume(vvVolDesc const& vd, int channel = 0);

  void updateTransfunc(TransfuncTex transfunc);
};

} // virvo

#endif // VV_SPACESKIP_GRID_H
//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
