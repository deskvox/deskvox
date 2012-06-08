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

#ifndef _VVPARBRICKREND_H_
#define _VVPARBRICKREND_H_

#include "vvbrickrend.h"
#include "vvbsptreevisitors.h"

#include <vector>

class vvRenderContext;
class vvVolDesc;

class VIRVOEXPORT vvParBrickRend : public vvBrickRend
{
public:
  vvParBrickRend(vvVolDesc* vd, vvRenderState rs,
                 const std::vector<std::string>& displays,
                 const std::string& type, const vvRendererFactory::Options& options);
  ~vvParBrickRend();

  void renderVolumeGL();

  void setParameter(ParameterType param, const vvParam& newValue);
  void updateTransferFunction();
private:
  struct Thread;

  vvSortLastVisitor* _sortLastVisitor;

  static void* renderFunc(void* args);
  static void render(Thread* thread);

  Thread* _thread;                                   ///< main thread
  std::vector<Thread*> _threads;                     ///< worker threads
  std::vector<vvSortLastVisitor::Texture> _textures;
};

#endif // _VVPARBRICKREND_H_
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
