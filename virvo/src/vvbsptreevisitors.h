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

#ifndef _VV_BSPTREEVISITORS_H_
#define _VV_BSPTREEVISITORS_H_

#include "vvoffscreenbuffer.h"
#include "vvopengl.h"
#include "vvvisitor.h"
#include "vvimage.h"
#include "vvsocketio.h"

#include <vector>

/** The content of each thread is rendered via the visitor pattern.
  The rendered results of each thread are managed using a bsp tree
  structure. Using the visitor pattern was a design decision so
  that the bsp tree doesn't need knowledge about the (rather specific)
  rendering code its half spaces utilize to display their results.
  This logic is supplied by the visitor which needs to be initialized
  once and passed to the bsp tree after initialization. Thus the bsp
  tree may be utilized in context not that specific as this one.
  @author Stefan Zellmann
  @see vvVisitor
 */
class VIRVOEXPORT vvThreadVisitor : public vvVisitor
{
public:
  vvThreadVisitor();
  ~vvThreadVisitor();
  void visit(vvVisitable* obj) const;

  void setOffscreenBuffers(vvOffscreenBuffer** offscreenBuffers,
                           const int numOffscreenBuffers);
  void setPixels(GLfloat**& pixels);
  void setWidth(const int width);
  void setHeight(const int height);
private:
  vvOffscreenBuffer** _offscreenBuffers;
  int _numOffscreenBuffers;
  GLfloat** _pixels;
  int _width;
  int _height;

  void clearOffscreenBuffers();
};

class VIRVOEXPORT vvSlaveVisitor : public vvVisitor
{
public:
  vvSlaveVisitor();
  ~vvSlaveVisitor();
  void visit(vvVisitable* obj) const;

  void setSockets(std::vector<vvSocketIO*>& sockets);

  void setProjectionMatrix(const vvMatrix& pr);
  void setModelviewMatrix(const vvMatrix& mv);
private:
  std::vector<vvSocketIO*> _sockets;

  vvMatrix _pr;
  vvMatrix _mv;

  GLuint* _textureIds;
};

#endif
