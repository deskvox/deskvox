// Grid accelerator from Ospray
//
// Adapted from:
// https://github.com/ospray/ospray/blob/master/ospray/volume/structured/GridAccelerator.ispc
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

#include <vvvoldesc.h>

#include "grid.h"

// NOTE: Ospray uses a 2-d structure (bricks + cells) while we
// only have one hierarchical level (cells)

//! Bit count used to represent the grid cell width.
#define CELL_WIDTH_BITCOUNT (4)

//! Grid cell width in volumetric elements.
#define CELL_WIDTH (1 << CELL_WIDTH_BITCOUNT)

namespace virvo
{

GridAccelerator::GridAccelerator()
  : cellRange(nullptr)
  , maxOpacities(nullptr)
{
}

GridAccelerator::~GridAccelerator()
{
  delete[] cellRange;
}

void GridAccelerator::updateVolume(vvVolDesc const& vd, int channel)
{
  delete[] cellRange;

  virvo::vec3i gd = ((vec3i)vd.vox+CELL_WIDTH-1) / CELL_WIDTH;

  gridDimensions = visionaray::vec3i(gd.data());

  size_t cellCount = gridDimensions.x * gridDimensions.y * gridDimensions.z;

  cellRange = (cellCount > 0) ?
              new visionaray::vec2[cellCount] :
              NULL;

  if (!cellRange)
    return;

  // cf. GridAccelerator_encodeVolumeBrick
  for (int z = 0; z < gd.z; ++z)
  {
    for (int y = 0; y < gd.y; ++y)
    {
      for (int x = 0; x < gd.x; ++x)
      {
        vec3i cellIndex(x, y, z);
        size_t cellIndexH = z * gd.x * gd.y + y * gd.x + x;

        cellRange[cellIndexH] = visionaray::vec2(99999.0f, -99999.0f);

        // GridAccelerator_encodeBrickCell
        for (int k = 0; k < CELL_WIDTH; ++k)
        {
          for (int j = 0; j < CELL_WIDTH; ++j)
          {
            for (int i = 0; i < CELL_WIDTH; ++i)
            {
              vec3i voxelIndex = cellIndex * CELL_WIDTH + vec3i(i, j, k);

              if (voxelIndex.x < vd.vox[0] && voxelIndex.y < vd.vox[1] && voxelIndex.z < vd.vox[2])
              {
                float value = vd.getChannelValue(vd.getCurrentFrame(),
                        voxelIndex.x,
                        voxelIndex.y,
                        voxelIndex.z,
                        channel);

                cellRange[cellIndexH].x = std::min(cellRange[cellIndexH].x, value);
                cellRange[cellIndexH].y = std::max(cellRange[cellIndexH].y, value);
              }
            }
          }
        }
      }
    }
  }
}

void GridAccelerator::updateTransfunc(GridAccelerator::TransfuncTex transfunc)
{
  // Brute-force precalculate table with opacity ranges
  // This could be accelerated with one of Ingo Wald's techniques
  // from Ray Tracing Gems (but we won't..)

  visionaray::vec4 const* arr = transfunc.data();
  int size = transfunc.size();

#ifdef RANGE_TREE
  delete[] maxOpacities;
  maxOpacities = new float[size-1];

  for (int i = 0; i < size/2; ++i)
  {
    maxOpacities[i] = std::max(arr[2 * i].w, arr[2*i+1].w);
  }

  size_t levelBegin = 0;
  size_t num_in = size/2;
  size_t num_nodes = size/2;
  while (num_in >= 2)
  {
    #pragma omp parallel for
    for( size_t i = 0; i < num_in/2; i++)
    {
        maxOpacities[num_nodes + i] = std::max(maxOpacities[levelBegin + 2 * i], maxOpacities[levelBegin + 2 * i+1]);
    }
    levelBegin += num_in;
    num_in /= 2;
    num_nodes += num_in;
  }
#else
  delete[] maxOpacities;
  maxOpacities = new float[size*size];

  // Well, size is just 256 - so what..
  for (int i = 0; i < size; ++i)
  {
    for (int j = 0; j < size; ++j)
    {
      maxOpacities[i * size + j] = 0.f;
      for (int k = std::min(i,j); k <= std::max(i,j); ++k)
      {
        maxOpacities[i * size + j] = std::max(maxOpacities[i * size + j], arr[k].w);
      }
    }
  }
#endif
}

} // virvo
