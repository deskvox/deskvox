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

#ifndef _VVTEXMULTIRENDMNGR_H_
#define _VVTEXMULTIRENDMNGR_H_

#include <vector>

#include "../vvexport.h"
#include "../vvdebugmsg.h"
#include "../vvvoldesc.h"
#include "../vvrenderer.h"
#include "../vvglsl.h"
#include "vvtexmultirend.h"

//============================================================================
// Class Definitions
//============================================================================

/** Volume rendering manager for rendering multiple volume datasets
  This manager is designed specifically for texture-based multi-channel rendering
  The original algorithm is designed by Chih Liang
  @author Chih
  @author Han S Kim
  @author Juergen Schulze
  @see vvRenderer
  @see vvTexMultiRend
*/
class VIRVOEXPORT vvTexMultiRendMngr
{
  public:
	vvTexMultiRendMngr();
	~vvTexMultiRendMngr();

	void init();
	void addVolume(vvVolDesc* vd);
	void renderMultipleVolume();
	vvTexMultiRend* getRenderer(int volNum) { return _rendererList[volNum]; }
	bool isEmpty() { return _rendererList.empty(); }
	int getNumVolume() { return _numVolume; }
	float getLastRenderTime() { return _lastRenderTime; }
	float getLastComputeTime() { return _lastComputeTime; }
	float getLastGLDrawTime() { return _lastGLdrawTime; }
	float getLastPlaneSortingTime() { return _lastPlaneSortingTime; }
	void setGLenvironment();
	void unsetGLenvironment();

  protected:
	vector<vvTexMultiRend*> _rendererList;
	int _numVolume;
	int _currentVolume;

    vvGLSL* glslShader;
	GLuint* shaderProgram;

    float _lastRenderTime;                   ///< time it took to render the previous frame (seconds)
	float _lastComputeTime;
	float _lastPlaneSortingTime;
	float _lastGLdrawTime;

	bool isShaderLoaded;

    bool extMinMax;                               ///< true = maximum/minimum intensity projections supported
    bool extBlendEquation;                        ///< true = support for blend equation extension
    typedef void (glBlendEquationEXT_type)(GLenum);
    glBlendEquationEXT_type* glBlendEquationVV;

    // GL state variables:
    GLboolean glsCulling;                         ///< stores GL_CULL_FACE
    GLboolean glsBlend;                           ///< stores GL_BLEND
    GLboolean glsColorMaterial;                   ///< stores GL_COLOR_MATERIAL
    GLint glsBlendSrc;                            ///< stores glBlendFunc(source,...)
    GLint glsBlendDst;                            ///< stores glBlendFunc(...,destination)
    GLboolean glsLighting;                        ///< stores GL_LIGHTING
    GLboolean glsDepthTest;                       ///< stores GL_DEPTH_TEST
    GLint glsMatrixMode;                          ///< stores GL_MATRIX_MODE
    GLint glsDepthFunc;                           ///< stores glDepthFunc
    GLint glsBlendEquation;                       ///< stores GL_BLEND_EQUATION_EXT
    GLboolean glsDepthMask;                       ///< stores glDepthMask
    GLboolean glsTexColTable;                     ///< stores GL_TEXTURE_COLOR_TABLE_SGI
    GLboolean glsSharedTexPal;                    ///< stores GL_SHARED_TEXTURE_PALETTE_EXT
};

#endif

//============================================================================
// End of File
//============================================================================

