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

#include "vvrendererfactory.h"

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#include "vvdebugmsg.h"
#include "vvvoldesc.h"
#include "vvtexrend.h"
#include "vvcudasw.h"
#include "vvsoftsw.h"
#include "vvrayrend.h"
#include "vvibrclient.h"
#include "vvimageclient.h"
#ifdef HAVE_VOLPACK
#include "vvrendervp.h"
#endif

#include <map>
#include <string>
#include <cstring>
#include <algorithm>

namespace {

typedef std::map<std::string, vvRenderer::RendererType> RendererTypeMap;
typedef std::map<std::string, vvTexRend::GeometryType> GeometryTypeMap;
typedef std::map<std::string, vvTexRend::VoxelType> VoxelTypeMap;
typedef std::map<std::string, std::string> RendererAliasMap;

RendererAliasMap rendererAliasMap;
RendererTypeMap rendererTypeMap;
GeometryTypeMap geometryTypeMap;
VoxelTypeMap voxelTypeMap;

void init()
{
  if(!rendererTypeMap.empty())
    return;

  vvDebugMsg::msg(3, "vvRendererFactory::init()");

  // used in vview
  rendererAliasMap["0"] = "default";
  rendererAliasMap["1"] = "slices";
  rendererAliasMap["2"] = "cubic2d";
  rendererAliasMap["3"] = "planar";
  rendererAliasMap["4"] = "spherical";
  rendererAliasMap["5"] = "bricks";
  rendererAliasMap["6"] = "soft";
  rendererAliasMap["7"] = "cudasw";
  rendererAliasMap["8"] = "rayrend";
  rendererAliasMap["9"] = "volpack";
  // used in COVER and Inventor renderer
  rendererAliasMap["tex2d"] = "slices";
  rendererAliasMap["slices2d"] = "slices";
  rendererAliasMap["preint"] = "planar";
  rendererAliasMap["fragprog"] = "planar";
  rendererAliasMap["tex"] = "planar";
  rendererAliasMap["tex3d"] = "planar";
  rendererAliasMap["brick"] = "bricks";

  voxelTypeMap["default"] = vvTexRend::VV_BEST;
  voxelTypeMap["rgba"] = vvTexRend::VV_RGBA;
  voxelTypeMap["arb"] = vvTexRend::VV_FRG_PRG;
  voxelTypeMap["paltex"] = vvTexRend::VV_PAL_TEX;
  voxelTypeMap["sgilut"] = vvTexRend::VV_SGI_LUT;
  voxelTypeMap["shader"] = vvTexRend::VV_PIX_SHD;
  voxelTypeMap["regcomb"] = vvTexRend::VV_TEX_SHD;

  // TexRend
  rendererTypeMap["default"] = vvRenderer::TEXREND;
  geometryTypeMap["default"] = vvTexRend::VV_AUTO;

  rendererTypeMap["slices"] = vvRenderer::TEXREND;
  geometryTypeMap["slices"] = vvTexRend::VV_SLICES;

  rendererTypeMap["cubic2d"] = vvRenderer::TEXREND;
  geometryTypeMap["cubic2d"] = vvTexRend::VV_CUBIC2D;

  rendererTypeMap["planar"] = vvRenderer::TEXREND;
  geometryTypeMap["planar"] = vvTexRend::VV_VIEWPORT;

  rendererTypeMap["spherical"] = vvRenderer::TEXREND;
  geometryTypeMap["spherical"] = vvTexRend::VV_SPHERICAL;

  rendererTypeMap["bricks"] = vvRenderer::TEXREND;
  geometryTypeMap["bricks"] = vvTexRend::VV_BRICKS;

  // other renderers
  rendererTypeMap["generic"] = vvRenderer::GENERIC;
  rendererTypeMap["soft"] = vvRenderer::SOFTSW;
  rendererTypeMap["cudasw"] = vvRenderer::CUDASW;
  rendererTypeMap["rayrend"] = vvRenderer::RAYREND;
  rendererTypeMap["volpack"] = vvRenderer::VOLPACK;
  rendererTypeMap["image"] = vvRenderer::REMOTE_IMAGE;
  rendererTypeMap["ibr"] = vvRenderer::REMOTE_IBR;
}

} // namespace

/**
 * \param vd volume description
 * \param rs renderer state
 * \param t renderer type or vvTexRend's geometry type
 * \param o options for renderer or vvTexRend's voxel type
 */
vvRenderer *vvRendererFactory::create(vvVolDesc *vd, const vvRenderState &rs, const char *t, const char *o)
{
  vvDebugMsg::msg(3, "vvRendererFactory::create: type=", t);
  vvDebugMsg::msg(3, "vvRendererFactory::create: options=", o);

  init();

  if(!t || !strcmp(t, "default"))
    t = getenv("VV_RENDERER");

  if(!t)
    t = "default";

  std::string type(t);
  std::transform(type.begin(), type.end(), type.begin(), ::tolower);

  if(!o)
    o = "default";

  std::string options(o);
  std::transform(options.begin(), options.end(), options.begin(), ::tolower);

  RendererAliasMap::iterator ait = rendererAliasMap.find(type);
  if(ait != rendererAliasMap.end())
    type = ait->second.c_str();
  
  RendererTypeMap::iterator it = rendererTypeMap.find(type);
  if(it == rendererTypeMap.end())
  {
    type = "default";
    it = rendererTypeMap.find(type);
  }
  assert(it != rendererTypeMap.end());

  switch(it->second)
  {
  case vvRenderer::GENERIC:
    return new vvRenderer(vd, rs);
  case vvRenderer::REMOTE_IMAGE:
    return new vvImageClient(vd, rs);
  case vvRenderer::REMOTE_IBR:
    return new vvIbrClient(vd, rs);
  case vvRenderer::SOFTSW:
    return new vvSoftShearWarp(vd, rs);
#ifdef HAVE_VOLPACK
  case vvRenderer::VOLPACK:
    return new vvVolPack(vd, rs);
#endif
#ifdef HAVE_CUDA
  case vvRenderer::RAYREND:
    return new vvRayRend(vd, rs);
  case vvRenderer::CUDASW:
    return new vvCudaShearWarp(vd, rs);
#endif
  case vvRenderer::TEXREND:
  default:
    {
      vvTexRend::VoxelType vox= vd->getBPV()<3 ? vvTexRend::VV_BEST : vvTexRend::VV_RGBA;

      VoxelTypeMap::iterator vit = voxelTypeMap.find(options);
      if(vit != voxelTypeMap.end())
        vox = vit->second;
      vvTexRend::GeometryType geo = vvTexRend::VV_AUTO;
      GeometryTypeMap::iterator git = geometryTypeMap.find(type);
      if(git != geometryTypeMap.end())
        geo = git->second;
      return new vvTexRend(vd, rs, geo, vox);
    }
    break;
  }
}

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0

