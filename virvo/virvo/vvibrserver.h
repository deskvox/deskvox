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

#ifndef _VV_IBRSERVER_H_
#define _VV_IBRSERVER_H_

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#ifdef HAVE_CUDA

#include "vvexport.h"
#include "vvoffscreenbuffer.h"
#include "vvrayrend.h"
#include "vvsocketio.h"
#include "vvremoteserver.h"

class VIRVOEXPORT vvIbrServer : public vvRemoteServer
{
public:
  vvIbrServer(const vvImage2_5d::DepthPrecision dp = vvImage2_5d::VV_USHORT,
              const vvImage2_5d::IbrDepthScale ds = vvImage2_5d::VV_FULL_DEPTH,
              const vvRayRend::IbrMode mode = vvRayRend::VV_MAX_GRADIENT);
  ~vvIbrServer();

  void setDepthPrecision(const vvImage2_5d::DepthPrecision dp);

private:
  vvImage2_5d::DepthPrecision _depthPrecision;  ///< precision of depth buffer for image based rendering
  vvImage2_5d::IbrDepthScale    _depthScale;
  vvRayRend::IbrMode          _ibrMode;

  void renderImage(vvMatrix& pr, vvMatrix& mv, vvRenderer* renderer);
  void resize(int w, int h);
};

#endif

#endif
