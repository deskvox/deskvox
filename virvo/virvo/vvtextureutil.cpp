// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
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

#include <bitset>
#include <cstring> // memcpy

#include "vvtextureutil.h"
#include "vvvoldesc.h"

namespace virvo
{

  //--- Helpers ---------------------------------------------------------------

  // Determine native texel format of voldesc
  PixelFormat nativeFormat(const vvVolDesc* vd)
  {
    if (vd->bpc == 1 && vd->chan == 1)
    {
      return PF_R8;
    }
    else if (vd->bpc == 1 && vd->chan == 2)
    {
      return PF_RG8;
    }
    else if (vd->bpc == 1 && vd->chan == 3)
    {
      return PF_RGB8;
    }
    else if (vd->bpc == 1 && vd->chan == 4)
    {
      return PF_RGBA8;
    }
    else if (vd->bpc == 2 && vd->chan == 1)
    {
      return PF_R16UI;
    }
    else if (vd->bpc == 2 && vd->chan == 2)
    {
      return PF_RG16UI;
    }
    else if (vd->bpc == 2 && vd->chan == 3)
    {
      return PF_RGB16UI;
    }
    else if (vd->bpc == 2 && vd->chan == 4)
    {
      return PF_RGBA16UI;
    }
    else if (vd->bpc == 4 && vd->chan == 1)
    {
      return PF_R32F;
    }
    else if (vd->bpc == 4 && vd->chan == 2)
    {
      return PF_RG32F;
    }
    else if (vd->bpc == 4 && vd->chan == 3)
    {
      return PF_RGB32F;
    }
    else if (vd->bpc == 4 && vd->chan == 4)
    {
      return PF_RGBA32F;
    }
    else
    {
      // Unsupported!
      return PixelFormat(-1);
    }
  }


  //--- Interface -------------------------------------------------------------

  size_t TextureUtil::computeTextureSize(vec3i first, vec3i last, PixelFormat tf)
  {
    PixelFormatInfo info = mapPixelFormat(tf);

    // Employ some sanity checks
    // TODO...

    vec3i size = last - first;
    return size.x * size.y * size.z * info.size;
  }

  TextureUtil::ErrorType TextureUtil::createTexture(uint8_t* dst,
      const vvVolDesc* vd,
      PixelFormat tf,
      TextureUtil::Channels chans,
      int frame)
  {
    return createTexture(dst,
        vd,
        vec3i(0),
        vec3i(vd->vox),
        tf,
        chans,
        frame);
  }

  TextureUtil::ErrorType TextureUtil::createTexture(uint8_t* dst,
      const vvVolDesc* vd,
      vec3i first,
      vec3i last,
      PixelFormat tf,
      TextureUtil::Channels chans,
      int frame)
  {
    PixelFormatInfo info = mapPixelFormat(tf);

    //--- Sanity checks ---------------

    // More than 4 channels: user needs to explicitly state
    // which channels (s)he's interested in
    if (vd->chan > 4 && chans == All)
      return NumChannelsMismatch;

    // TODO: requires C++11 std::bitset(ull) ctor!
    if (chans != All && info.components != std::bitset<64>(chans).count())
      return NumChannelsMismatch;


    //--- Make texture ----------------

    // Maybe we can just memcpy the whole frame
    if (nativeFormat(vd) == tf && first == vec3i(0) && last == vec3i(vd->vox))
    {
      size_t size = computeTextureSize(first, last, tf);
      memcpy(dst, vd->getRaw(frame), size);
      return Ok;
    }

    // Maybe we can just memcpy consecutive slices
    if (nativeFormat(vd) == tf && first.xy() == vec2i(0) && last.xy() == vec2i(vd->vox.xy()))
    {
      const uint8_t* raw = vd->getRaw(frame);
      for (int z = first.z; z < last.z; ++z)
      {
        memcpy(dst + z * vd->getSliceBytes(),
            raw + z * vd->getSliceBytes(),
            vd->getSliceBytes());
      }
      return Ok;
    }

    // Maybe the conversion operation is trivial and we can
    // at least copy sections of the volume data
    if (nativeFormat(vd) == tf)
    {
      const uint8_t* raw = vd->getRaw(frame);
      for (int z = first.z; z < last.z; ++z)
      {
        for (int y = first.y; y < last.y; ++y)
        {
          for (int x = first.x; x < last.x; ++x)
          {
            memcpy(dst, raw, vd->getBPV());
            raw += vd->getBPV();
            dst += vd->getBPV();
          }
        }
      }
      return Ok;
    }

    // No use, have to iterate over all voxels
    // TODO: support N-byte
    if (info.size / info.components == 1/*byte*/)
    {
      const uint8_t* raw = vd->getRaw(frame);
      for (int z = first.z; z < last.z; ++z)
      {
        for (int y = first.y; y < last.y; ++y)
        {
          for (int x = first.x; x < last.x; ++x)
          {
            for (int c = 0; c < vd->chan; ++c)
            {
              if ((chans >> c) & 1)
              {
                *dst++ = static_cast<uint8_t>(vd->rescaleVoxel(raw, 1/*byte*/, c));
              }

              raw += vd->bpc;
            }
          }
        }
      }

      return Ok;
    }

    return Unknown;
  }

  TextureUtil::ErrorType TextureUtil::createTexture(uint8_t* dst,
      const vvVolDesc* vd,
      vec3i first,
      vec3i last,
      const uint8_t* rgba,
      int bpcDst,
      int frame)
  {
    // This is only supported if volume has <= 4 channels
    if (vd->chan > 4)
      return Unknown;

    // Single channel: rescale voxel to 8-bit, use as index into RGBA lut
    if (vd->chan == 1)
    {
      const uint8_t* raw = vd->getRaw(frame);
      for (int z = first.z; z < last.z; ++z)
      {
        for (int y = first.y; y < last.y; ++y)
        {
          for (int x = first.x; x < last.x; ++x)
          {
            int index = vd->rescaleVoxel(raw, 1/*byte*/, 0) * bpcDst;
            dst[0] = rgba[index * 4];
            dst[1] = rgba[index * 4 + 1];
            dst[2] = rgba[index * 4 + 2];
            dst[3] = rgba[index * 4 + 3];
            raw += vd->bpc;
            dst += bpcDst * 4;
          }
        }
      }

      return Ok;
    }

    // Two or three channels: RG(B) values come from 3-D texture,
    // calculate alpha as mean of sum of RG(B) conversion table results
    if (vd->chan == 2 || vd->chan == 3)
    {
      // TODO: only implemented for RGBA8 lut!
      if (bpcDst != 1)
        return Unknown;

      const uint8_t* raw = vd->getRaw(frame);
      for (int z = first.z; z != last.z; ++z)
      {
        for (int y = first.y; y != last.y; ++y)
        {
          for (int x = first.x; x != last.x; ++x)
          {
            int alpha = 0;
            for (int c = 0; c < vd->chan; ++c)
            {
              uint8_t index = vd->rescaleVoxel(raw, 1/*byte*/, c);
              alpha += static_cast<int>(rgba[index * 4 + c]);
              dst[c] = index;
              raw += vd->bpc;
            }

            dst[3] = static_cast<uint8_t>(alpha / vd->chan);
            dst += 4;
          }
        }
      }

      return Ok;
    }

    // Four channels: just skip the RGBA lut.
    // TODO: this is legacy behavior, but is it actually desired??
    if (vd->chan == 4)
    {
      return createTexture(dst,
          vd,
          first,
          last,
          nativeFormat(vd),
          RGBA,
          frame);
    }

    return Unknown;
  }
}

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
