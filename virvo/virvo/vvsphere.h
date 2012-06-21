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

#ifndef _VVSPHERE_H_
#define _VVSPHERE_H_

#include <stdlib.h>
#include "vvexport.h"
#include "vvvecmath.h"

/** Vertex definition for vvSphere.
    @author Daniel Weiskopf
    @see vvSphere
*/
class VIRVOEXPORT vvVertex
{
  public:
    vvVertex() : x(0.f), y(0.f), z(0.f), th(0.f), ph(0.f) {}

    vvVertex(const vvVertex&);
    const vvVertex & operator=(const vvVertex &);
    void  scale(float);

    float x;
    float y;
    float z;
    float th;
    float ph;
};

/** Texture coordinate for vvSphere.
  @author Daniel Weiskopf
  @see vvSphere
*/
class vvTexCoord
{
  public:
    vvTexCoord() { t[0] = t[1] = t[2] = 0.f; }
    float t[3];
};

/** Triangle definition for vvSphere.
  @author Daniel Weiskopf
  @see vvSphere
*/
class VIRVOEXPORT vvTriangle
{
  public:
    vvTriangle();
    vvTriangle(const vvTriangle&);
    const vvTriangle& operator=(const vvTriangle&);

    int v1;
    int v2;
    int v3;
    int visibility;
};

/** Generates texture coordinates for random spheres.
  @author Daniel Weiskopf
*/
class VIRVOEXPORT  vvSphere
{
  public:
    vvSphere();                                   // Default constructor
    void render();
    void renderWireframe(int type=0);
    void initDodecaeder();
    void subdivide();
    void performCulling();
    void calculateTexCoords();
    void setRadius(float);
    void setVolumeDim(const vvVector3& dim);
    void setViewMatrix(const vvMatrix& viewMatrix);
    void setTextureOffset(float*);

  private:
    static int dodecaederConnectivity[60];
    vvVertex*   mVertices;
    vvVertex*   mVerticesWorld;
    int         mNumVertices;
    vvTriangle* mTriangles;
    int         mNumTriangles;
    vvTexCoord* mTexCoords;
    double      mRadius;
    vvVector3    mDimCube;
    vvMatrix    mModelView;
    float       texOffset[3];

    bool     isVisibleVertex(int vert);
    vvVertex midpoint(const vvVertex& a, const vvVertex& b) const;
    void     copyVerticesWorld();
};
#endif
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
