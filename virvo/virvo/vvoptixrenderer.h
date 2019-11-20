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

#ifndef VV_OPTIXRENDERER_H
#define VV_OPTIXRENDERER_H 1

#include <memory>

#include "vvrenderer.h"

class vvOptixRenderer : public vvRenderer
{
public:
    VVAPI vvOptixRenderer(vvVolDesc* vd, vvRenderState renderState);
    VVAPI ~vvOptixRenderer();

    VVAPI virtual void renderVolumeGL() VV_OVERRIDE;
    VVAPI virtual void updateTransferFunction() VV_OVERRIDE;
    VVAPI virtual void updateVolumeData() VV_OVERRIDE;
    VVAPI virtual void  setCurrentFrame(size_t frame) VV_OVERRIDE;
    VVAPI bool checkParameter(ParameterType param, vvParam const& value) const VV_OVERRIDE;
    VVAPI virtual void setParameter(ParameterType param, const vvParam& newValue) VV_OVERRIDE;
    /*VVAPI virtual vvParam getParameter(ParameterType param) const VV_OVERRIDE;*/
    VVAPI virtual bool instantClassification() const VV_OVERRIDE;

private:

    VV_NOT_COPYABLE(vvOptixRenderer)

    struct Impl;
    std::unique_ptr<Impl> impl_;
};


#endif // VV_OPTIXRENDERER_H

