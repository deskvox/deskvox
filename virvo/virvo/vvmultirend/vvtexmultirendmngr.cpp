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

// No circular dependencies between gl.h and glew.h
#ifndef GLEW_INCLUDED
#include <GL/glew.h>
#define GLEW_INCLUDED
#endif

#include <iostream>
#include <iomanip>

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#if defined(__linux) || defined(LINUX)
#define GL_GLEXT_PROTOTYPES 1
#define GL_GLEXT_LEGACY 1
#include <string.h>
#endif

#include "../vvopengl.h"

#include "../vvdynlib.h"
#if !defined(_WIN32) && !defined(__APPLE__)
#include <dlfcn.h>
#include <GL/glx.h>
#endif

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "../vvvecmath.h"
#include "../vvdebugmsg.h"
#include "../vvtoolshed.h"
#include "../vvgltools.h"
#include "../vvshaderfactory.h"
#include "../vvsphere.h"
#include "../vvstopwatch.h"
#include "../vvprintgl.h"
#include "vvtexmultirendmngr.h"




using namespace std;

vvTexMultiRendMngr::vvTexMultiRendMngr()
{
  _numVolume = 0;
  _currentVolume = 0;

  _lastRenderTime = 0.0f;
  _lastComputeTime = 0.0f;
  _lastPlaneSortingTime = 0.0f;


  if (vvDebugMsg::isActive(1))
  {
#ifdef _WIN32
    cerr << "_WIN32 is defined" << endl;
#elif WIN32
    cerr << "_WIN32 is not defined, but should be if running under Windows" << endl;
#endif

#ifdef HAVE_CG
    cerr << "HAVE_CG is defined" << endl;
#else
    cerr << "Tip: define HAVE_CG for pixel shader support" << endl;
#endif

    cerr << "Compiler knows OpenGL versions: ";
#ifdef GL_VERSION_1_1
    cerr << "1.1";
#endif
#ifdef GL_VERSION_1_2
    cerr << ", 1.2";
#endif
#ifdef GL_VERSION_1_3
    cerr << ", 1.3";
#endif
#ifdef GL_VERSION_1_4
    cerr << ", 1.4";
#endif
#ifdef GL_VERSION_1_5
    cerr << ", 1.5";
#endif
#ifdef GL_VERSION_2_0
    cerr << ", 2.0";
#endif
    cerr << endl;
  }

  extMinMax = vvGLTools::isGLextensionSupported("GL_EXT_blend_minmax");
  extBlendEquation = vvGLTools::isGLextensionSupported("GL_EXT_blend_equation");
  if (extBlendEquation) glBlendEquationVV = (glBlendEquationEXT_type*)vvDynLib::glSym("glBlendEquationEXT");
  else glBlendEquationVV = (glBlendEquationEXT_type*)vvDynLib::glSym("glBlendEquation");

  _shaderFactory = new vvShaderFactory();

  // TODO: needs to be parameterized
  glslShader.push_back(_shaderFactory->createProgram("", "", "glsl_1chan.frag"));
  glslShader.push_back(_shaderFactory->createProgram("", "", "glsl_2chan.frag"));
  glslShader.push_back(_shaderFactory->createProgram("", "", "glsl_3chan.frag"));
  glslShader.push_back(_shaderFactory->createProgram("", "", "glsl_multichan.frag"));
}

vvTexMultiRendMngr::~vvTexMultiRendMngr()
{
  delete _shaderFactory;
  for(size_t i=0; i<_rendererList.size();i++)
    delete _rendererList[i];
}

void vvTexMultiRendMngr::init()
{
  // we assume voxelType == VV_GLSL
}

void vvTexMultiRendMngr::addVolume(vvVolDesc* vd)
{
  _numVolume++;

  vvRenderState renderState;
  vvTexMultiRend* aRenderer = new vvTexMultiRend(vd, renderState, vvTexMultiRend::VV_VIEWPORT, vvTexMultiRend::VV_GLSL);
  aRenderer->setParameter(vvRenderer::VV_SLICEORIENT, vvTexMultiRend::VV_VIEWPLANE);

  _rendererList.push_back(aRenderer);
}


void vvTexMultiRendMngr::renderMultipleVolume()
{
  vvTexMultiRend* aRenderer;
  vvVolDesc* vd;
  int totalSlices = 0;						// total number of slices to be drawn
  vvMatrix pm;								// OpenGL projection matrix
  static vvStopwatch sw, swC, swS;	// stop watch for performance measurements

  sw.start();
  swC.start();
  _lastPlaneSortingTime = 0.0f;
  //_lastGLdrawTime = 0.0f;
  //TexRendData *texRendArray = new TexRendData[_numVolume];


  // Get projection matrix:
  vvGLTools::getProjectionMatrix(&pm);

  // Compute normal vector of view planes
  vvVector3 planeNormal(0.0f, 0.0f, -1.0f);                 // (0|0|1) is normal on projection plane
  vvMatrix invPM(&pm);
  invPM.invert();
  planeNormal.multiply(&invPM);

  for(int v = 0; v < _numVolume; v++)
  {
	//TexRendData &tr = texRendArray[v];

	_rendererList[v]->preRendering();

	totalSlices += _rendererList[v]->getNumSlices();

#if 0
	vvMatrix invMV;                                 // inverse of model-view matrix
	vvVector3 eye;                                  // user's eye position [object space]
	vvVector3 probePosObj;                          // probe midpoint [object space]
	vvVector3 probeSizeObj;                         // probe size [object space]
	vvVector3 probeTexels;                          // number of texels in each probe dimension
	vvVector3 texSize;                              // size of 3D texture [object space]
	vvVector3 pos;                                  // volume location
	vvVector3 size(vd->getSize());                  // volume size [world coordinates]
	
	// Get OpenGL modelview matrix:
	aRenderer->getModelviewMatrix(&tr.mv);
	tr.mv.translate(aRenderer->getTranslation());
	tr.mv.multiplyPre(aRenderer->getRotation());
	tr.mv.makeGL(tr.glMV);

	// drawing bounding boxes
	if(aRenderer->_renderState._boundaries)
	{
	  glMatrixMode(GL_MODELVIEW);
	  glPushMatrix();
	  glLoadMatrixf(tr.glMV);
	  aRenderer->drawBoundingBox(&size, &vd->pos, aRenderer->_renderState._boundColor);
	  glPopMatrix();
	}

	// Determine texture object extensions:
	for (int i=0; i<3; ++i)
	{
	  aRenderer->setTexMin(i, 0.5f / (float)aRenderer->getTexels(i));
	  aRenderer->setTexMax(i, (float)vd->vox[i] / (float)aRenderer->getTexels(i) - aRenderer->getTexMin(i));
	}

	// Determine texture object dimensions and half object size as a shortcut:
	for (int i=0; i<3; ++i)
	{
	  texSize.e[i] = size.e[i] * (float)aRenderer->getTexels(i) / (float)vd->vox[i];
	  tr.size2.e[i]   = 0.5f * size.e[i];
	}
	pos.copy(&vd->pos);

	// Calculate inverted modelview matrix:
	invMV.copy(&tr.mv);
	invMV.invert();

	// Find eye position:
	aRenderer->getEyePosition(&eye);
	eye.multiply(&invMV);

	probeSizeObj.copy(&size);
	tr.probeMin.set(-tr.size2[0], -tr.size2[1], -tr.size2[2]);
	tr.probeMax.copy(&tr.size2);
	probePosObj.zero();

	// Initialize texture counters
	if (aRenderer->_renderState._roiSize[0])
	{
		probeTexels.zero();
		for (int i=0; i<3; ++i)
		{
			probeTexels[i] = aRenderer->getTexels(i) * probeSizeObj[i] / texSize.e[i];
		}
	}
	else                                            // probe mode off
	{
		probeTexels.set((float)vd->vox[0], (float)vd->vox[1], (float)vd->vox[2]);
	}
	
	//if (_sliceOrientation != VV_VIEWPLANE)
		//cerr << "sliceOrientation != VV_VIEWPLANE\n";

	// Compute distance vector between textures:
	tr.normal.copy(planeNormal);
	tr.normal.multiply(&invMV);
	tr.normal.normalize();

	// compute number of slices to draw
	float depth = fabs(tr.normal[0]*probeSizeObj[0]) + fabs(tr.normal[1]*probeSizeObj[1]) + fabs(tr.normal[2]*probeSizeObj[2]);

	int minDistanceInd = 0;
	if(probeSizeObj[1]/probeTexels[1] < probeSizeObj[minDistanceInd]/probeTexels[minDistanceInd])
		minDistanceInd=1;
	if(probeSizeObj[2]/probeTexels[2] < probeSizeObj[minDistanceInd]/probeTexels[minDistanceInd])
		minDistanceInd=2;
	float voxelDistance = probeSizeObj[minDistanceInd]/probeTexels[minDistanceInd];

	float sliceDistance = voxelDistance / aRenderer->getQuality();
	/*
	if(_renderState._roiSize[0] && _renderState._quality < 2.0)
	{
		// draw at least twice as many slices as there are samples in the probe depth.
		sliceDistance = voxelDistance / 2.0f;
	}
	*/
	tr.numSlices = 2*(int)ceilf(depth/sliceDistance*.5f);

	/*set num slices to constant*/
	//aRenderer->numSlices = (int) ( min(min(vd->texels[0], vd->texels[1]),vd->texels[2]) * vd->quality );
	//sliceDistance = depth/aRenderer->numSlices; 

	if (tr.numSlices < 1)                              // make sure that at least one slice is drawn
		tr.numSlices = 1;

	// Use alpha correction in indexed mode: adapt alpha values to number of textures:
	if (aRenderer->instantClassification())
	{
		float thickness = sliceDistance/voxelDistance;

		// just tolerate slice distance differences imposed on us
		// by trying to keep the number of slices constant
		if(aRenderer->getLUTDistance()/thickness < 0.88 || thickness/aRenderer->getLUTDistance() < 0.88)
		{
			aRenderer->updateLUT(thickness);
		}
	}

	tr.delta.copy(&tr.normal);
	tr.delta.scale(sliceDistance);

	// Compute farthest point to draw texture at:
	tr.farthest.copy(&tr.delta);
	tr.farthest.scale((float)(tr.numSlices - 1) / -2.0f);
	tr.farthest.add(&probePosObj);
	totalSlices += tr.numSlices;
	aRenderer->setNumSlices(tr.numSlices);
	tr.farWS.copy(tr.farthest);
	tr.farWS.multiply(&tr.mv);
#endif

  }
  // drawing bounding boxes
  // set tr data (numSlices, farWS)
  // update LUT according to thickness

  setGLenvironment();

  glMatrixMode(GL_MODELVIEW);

  int oldVol = -1;
  for(int drawn = 0; drawn < totalSlices; drawn++)
  {
	swS.start();
	int vol = 0;
	while(_rendererList[vol]->getNumSlices() <= 0) vol++;

	// TODO: check if it works
	for(int i = vol+1; i < _numVolume; i++)
	{
	  if(_rendererList[i]->getNumSlices() <= 0) continue;

	  vvVector3 diff = _rendererList[i]->tr.farWS - _rendererList[vol]->tr.farWS;
	  if(planeNormal.dot(&diff) < 0.0f)
		vol = i;
	}

	//vvDebugMsg::msg(1, "volume number: ", vol);

	_lastPlaneSortingTime += swS.getTime();


	aRenderer = _rendererList[vol];
	vd = aRenderer->getVolDesc();

	if(oldVol != vol)
	{
    int n = (static_cast<size_t>(vd->chan) < glslShader.size()) ? vd->chan - 1 : glslShader.size() - 1;
    aRenderer->enableLUTMode(glslShader[n]);
	  oldVol = vol;
	}

	aRenderer->decreaseNumSlices();

	aRenderer->renderMultipleVolume();

#if 0
	// can't happen??
	//TexRendData &tr = texRendArray[vol];
	if(aRenderer->decreaseNumSlices() < 0)
	  vvDebugMsg::msg(1, "numSlices exceeded");

	aRenderer->renderVolumeGL();
#endif

	//_lastGLdrawTime += swGL.getTime();
  }

  for(size_t i=0;i<glslShader.size();i++)
    glslShader[i]->disable();

  //_rendererList[0]->unsetGLenvironment();
  unsetGLenvironment();

  _lastComputeTime = swC.getTime();
  glFinish();
  _lastRenderTime = sw.getTime();
}


//----------------------------------------------------------------------------
/// Set GL environment for texture rendering.
void vvTexMultiRendMngr::setGLenvironment()
{
  vvDebugMsg::msg(3, "vvTexMultiRendMngr::setGLenvironment()");

  // Save current GL state:
  glGetBooleanv(GL_CULL_FACE, &glsCulling);
  glGetBooleanv(GL_BLEND, &glsBlend);
  glGetBooleanv(GL_COLOR_MATERIAL, &glsColorMaterial);
  glGetIntegerv(GL_BLEND_SRC, &glsBlendSrc);
  glGetIntegerv(GL_BLEND_DST, &glsBlendDst);
  glGetBooleanv(GL_LIGHTING, &glsLighting);
  glGetBooleanv(GL_DEPTH_TEST, &glsDepthTest);
  glGetIntegerv(GL_MATRIX_MODE, &glsMatrixMode);
  glGetIntegerv(GL_DEPTH_FUNC, &glsDepthFunc);

  if (extMinMax) glGetIntegerv(GL_BLEND_EQUATION_EXT, &glsBlendEquation);
  glGetBooleanv(GL_DEPTH_WRITEMASK, &glsDepthMask);

#if 0
  switch (voxelType)
  {
    case VV_SGI_LUT:
      glGetBooleanv(GL_TEXTURE_COLOR_TABLE_SGI, &glsTexColTable);
      break;
    case VV_PAL_TEX:
      glGetBooleanv(GL_SHARED_TEXTURE_PALETTE_EXT, &glsSharedTexPal);
      break;
    default: break;
  }
#endif

  // Set new GL state:
  glDisable(GL_CULL_FACE);
  glDisable(GL_LIGHTING);
  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LESS);                           // default depth function
  glEnable(GL_COLOR_MATERIAL);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
  glMatrixMode(GL_TEXTURE);
  glLoadIdentity();
  glMatrixMode(GL_MODELVIEW);
  glDepthMask(GL_FALSE);

#if 0
  switch (voxelType)
  {
    case VV_SGI_LUT:
      glDisable(GL_TEXTURE_COLOR_TABLE_SGI);
      break;
    case VV_PAL_TEX:
      glDisable(GL_SHARED_TEXTURE_PALETTE_EXT);
      break;
    default: break;
  }
#endif

  if (glBlendEquationVV)
  {
    switch ((int)_rendererList[0]->getParameter(vvRenderState::VV_MIP_MODE))
    {
                                                  // alpha compositing
      case 0: glBlendEquationVV(GL_FUNC_ADD); break;
      case 1: glBlendEquationVV(GL_MAX); break;   // maximum intensity projection
      case 2: glBlendEquationVV(GL_MIN); break;   // minimum intensity projection

      default:
        glBlendEquationVV((int)_rendererList[0]->getParameter(vvRenderState::VV_MIP_MODE));
		  break;
    }
  }

  vvDebugMsg::msg(3, "vvTexMultiRendMngr::setGLenvironment() done");
}

//----------------------------------------------------------------------------
/// Unset GL environment for texture rendering.
void vvTexMultiRendMngr::unsetGLenvironment()
{
  vvDebugMsg::msg(3, "vvTexMultiRendMngr::unsetGLenvironment()");

  if (glsCulling==(GLboolean)true) glEnable(GL_CULL_FACE);
  else glDisable(GL_CULL_FACE);

  if (glsBlend==(GLboolean)true) glEnable(GL_BLEND);
  else glDisable(GL_BLEND);

  if (glsColorMaterial==(GLboolean)true) glEnable(GL_COLOR_MATERIAL);
  else glDisable(GL_COLOR_MATERIAL);

  if (glsDepthTest==(GLboolean)true) glEnable(GL_DEPTH_TEST);
  else glDisable(GL_DEPTH_TEST);

  if (glsLighting==(GLboolean)true) glEnable(GL_LIGHTING);
  else glDisable(GL_LIGHTING);

  glDepthMask(glsDepthMask);
  glDepthFunc(glsDepthFunc);
  glBlendFunc(glsBlendSrc, glsBlendDst);
  if (glBlendEquationVV) glBlendEquationVV(glsBlendEquation);
  glMatrixMode(glsMatrixMode);
  vvDebugMsg::msg(3, "vvTexMultiRendMngr::unsetGLenvironment() done");
}



// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
