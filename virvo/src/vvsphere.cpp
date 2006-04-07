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

#include "vvsphere.h"

#ifdef _WIN32
#include <windows.h>
#endif
#include <stdlib.h>
#include <math.h>
#include "vvopengl.h"

#ifdef VV_DEBUG_MEMORY
#include <crtdbg.h>
#define new new(_NORMAL_BLOCK,__FILE__, __LINE__)
#endif

#include "vvvecmath.h"
#include "vvdebugmsg.h"
#include "vvtoolshed.h"

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

/// Constructor
vvVertex::vvVertex(const vvVertex& vert)
: x(vert.x),
y(vert.y),
z(vert.z),
th(vert.th),
ph(vert.ph)
{
}

/// Overload '=' operator
const vvVertex& vvVertex::operator=(const vvVertex & vert)
{
  x = vert.x;
  y = vert.y;
  z = vert.z;
  th = vert.th;
  ph = vert.ph;

  return *this;
}

void vvVertex::scale(float s)
{
  x *= s;
  y *= s;
  z *= s;
}

/// Copy constructor
vvTriangle::vvTriangle(const vvTriangle& t) : v1(t.v1),v2(t.v2),v3(t.v3),
visibility(t.visibility)
{
}

const vvTriangle& vvTriangle::operator=(const vvTriangle& t)
{
  v1 = t.v1;
  v2 = t.v2;
  v3 = t.v3;
  visibility = t.visibility;

  return *this;
}

/// Copy constructor
vvTriangle::vvTriangle() : v1(0),v2(0),v3(0),visibility(0)
{
}

//---------------------------------------------------------------------

/// Default constructor
vvSphere::vvSphere()
: mVertices(NULL),
mVerticesWorld(NULL),
mNumVertices(0),
mTriangles(NULL),
mNumTriangles(0),
mTexCoords(NULL),
mRadius(1.0)
{
  mModelView.identity();
  initDodecaeder();
}

void vvSphere::setRadius(float r)
{
  mRadius = r;
  copyVerticesWorld();
}

void vvSphere::setVolumeDim(vvVector3* dim)
{
  mDimCube.copy(dim);
}

void vvSphere::setTextureOffset(float* offset)
{
  texOffset[0] = offset[0];
  texOffset[1] = offset[1];
  texOffset[2] = offset[2];
}

void vvSphere::setViewMatrix(vvMatrix* matrix)
{
  mModelView.copy(matrix);
  mModelView.invert();
}

bool vvSphere::isVisibleVertex(int vert)
{

  float min = -0.2f;
  float max = 1.2f;

  vvTexCoord* tex = &mTexCoords[vert];
  return ((tex->t[0] > min) && (tex->t[0] < max)
    && (tex->t[1] > min) && (tex->t[1] < max)
    && (tex->t[2] > min) && (tex->t[2] < max) );
}

void vvSphere::performCulling()
{
  for (int i = 0; i < mNumTriangles; i++)
  {
    mTriangles[i].visibility = -1;
    if (isVisibleVertex(mTriangles[i].v1) ||
      isVisibleVertex(mTriangles[i].v2) ||
      isVisibleVertex(mTriangles[i].v3))
    {
      mTriangles[i].visibility = 1;
    }
  }
}

void vvSphere::calculateTexCoords()
{
  vvMatrix texMatrix;

  texMatrix.identity();
  texMatrix.translate(0.5,0.5,0.5);
  texMatrix.scale(1.0f/mDimCube.e[0], 1.0f/mDimCube.e[1], 1.0f/mDimCube.e[2]);
  texMatrix.translate(texOffset[0], texOffset[1], texOffset[2]);
  texMatrix.multiplyPre(&mModelView);

  for (int i = 0; i < mNumVertices; i++)
  {
    vvVector3 tmp(mVerticesWorld[i].x, mVerticesWorld[i].y, mVerticesWorld[i].z);
    tmp.multiply(&texMatrix);
    mTexCoords[i].t[0] = tmp.e[0];
    mTexCoords[i].t[1] = tmp.e[1];
    mTexCoords[i].t[2] = tmp.e[2];
  }
}

void vvSphere::render()
{
  glBegin(GL_TRIANGLES);
  for (int i = 0; i < mNumTriangles; i++)
  {
    if (mTriangles[i].visibility == 1)
    {
      vvVertex*   v1 = &mVerticesWorld[mTriangles[i].v1];
      vvVertex*   v2 = &mVerticesWorld[mTriangles[i].v2];
      vvVertex*   v3 = &mVerticesWorld[mTriangles[i].v3];
      vvTexCoord* t1 = &mTexCoords[mTriangles[i].v1];
      vvTexCoord* t2 = &mTexCoords[mTriangles[i].v2];
      vvTexCoord* t3 = &mTexCoords[mTriangles[i].v3];

      glTexCoord3f(t1->t[0], t1->t[1], t1->t[2]);
      glNormal3f(v1->x, v1->y, v1->z);
      glVertex3f(v1->x, v1->y, v1->z);
      glTexCoord3f(t2->t[0], t2->t[1], t2->t[2]);
      glNormal3f(v2->x, v2->y, v2->z);
      glVertex3f(v2->x, v2->y, v2->z);
      glTexCoord3f(t3->t[0], t3->t[1], t3->t[2]);
      glNormal3f(v3->x, v3->y, v3->z);
      glVertex3f(v3->x, v3->y, v3->z);
    }
  }
  glEnd();
}

void vvSphere::renderWireframe(int type)
{
  glBegin(GL_LINES);
  for (int i = 0; i < mNumTriangles; i++)
  {
    if (mTriangles[i].visibility == 1 || type == 1)
    {
      vvVertex* v1 = &mVerticesWorld[mTriangles[i].v1];
      vvVertex* v2 = &mVerticesWorld[mTriangles[i].v2];
      vvVertex* v3 = &mVerticesWorld[mTriangles[i].v3];
      glVertex3f(v1->x, v1->y, v1->z);
      glVertex3f(v2->x, v2->y, v2->z);
      glVertex3f(v2->x, v2->y, v2->z);
      glVertex3f(v3->x, v3->y, v3->z);
      glVertex3f(v3->x, v3->y, v3->z);
      glVertex3f(v1->x, v1->y, v1->z);
    }
  }
  glEnd();
}

/// Return the midpoint on the line between two points.
vvVertex vvSphere::midpoint(const vvVertex& a, const vvVertex& b) const
{
  vvVertex r;

  r.x = (a.x + b.x) * 0.5f;
  r.y = (a.y + b.y) * 0.5f;
  r.z = (a.z + b.z) * 0.5f;

  float hypot = (float)sqrt(r.x*r.x+r.y*r.y);
  r.th = (float)atan2(hypot,r.z);
  if(r.z == 0.)
    r.th = (float)M_PI / 2.0f;

  r.ph = (float)atan2(r.y,r.x);

  r.x = (float)(sin(r.th) * cos(r.ph));
  r.y = (float)(sin(r.th) * sin(r.ph));
  r.z = (float)cos(r.th);

  return r;
}

/// Refine polygon mesh.
void vvSphere::subdivide()
{
  int numTrianglesNew = mNumTriangles *4;
  int numVerticesNew = mNumVertices + mNumTriangles*3;
  vvTriangle* trianglesNew = new vvTriangle[numTrianglesNew];
  vvVertex* verticesNew = new vvVertex[numVerticesNew];
  vvVertex* verticesWorldNew = new vvVertex[numVerticesNew];

  if (mTexCoords)
  {
    delete[] mTexCoords;
  }
  mTexCoords = new vvTexCoord[numVerticesNew];

  for (int j = 0; j < mNumVertices; j++)
  {
    verticesNew[j] = mVertices[j];
  }

  int vertIndex = mNumVertices;
  int triaIndex = 0;
  for (int i = 0; i < mNumTriangles; i++)
  {
    int v1 = mTriangles[i].v1;
    int v2 = mTriangles[i].v2;
    int v3 = mTriangles[i].v3;

    verticesNew[vertIndex++] = midpoint(mVertices[v1],
      mVertices[v2]);
    verticesNew[vertIndex++] = midpoint(mVertices[v2],
      mVertices[v3]);
    verticesNew[vertIndex++] = midpoint(mVertices[v3],
      mVertices[v1]);

    vvTriangle triangle1;
    triangle1.v1 = mTriangles[i].v1;
    triangle1.v2 = vertIndex-3;
    triangle1.v3 = vertIndex-1;
    trianglesNew[triaIndex++] = triangle1;

    vvTriangle triangle2;
    triangle2.v1 = vertIndex-3;
    triangle2.v2 = mTriangles[i].v2;
    triangle2.v3 = vertIndex-2;
    trianglesNew[triaIndex++] = triangle2;

    vvTriangle triangle3;
    triangle3.v1 = vertIndex-2;
    triangle3.v2 = mTriangles[i].v3;
    triangle3.v3 = vertIndex-1;
    trianglesNew[triaIndex++] = triangle3;

    vvTriangle triangle4;
    triangle4.v1 = vertIndex-3;
    triangle4.v2 = vertIndex-2;
    triangle4.v3 = vertIndex-1;
    trianglesNew[triaIndex++] = triangle4;
  }

  delete[] mVertices;
  delete[] mTriangles;
  delete [] mVerticesWorld;

  mNumTriangles = numTrianglesNew;
  mNumVertices = numVerticesNew;
  mVertices = verticesNew;
  mVerticesWorld = verticesWorldNew;
  mTriangles = trianglesNew;

  copyVerticesWorld();
}

void vvSphere::copyVerticesWorld()
{

  for (int j = 0; j < mNumVertices; j++)
  {
    mVerticesWorld[j] = mVertices[j];
    mVerticesWorld[j].scale((float)mRadius);
  }
}

/// Storage for dodecaeder connectivity.
int vvSphere::dodecaederConnectivity[60] =
{
  0, 1, 5,
  0, 1, 2,
  0, 2, 3,
  0, 3, 4,
  0, 4, 5,
  1, 6, 10,
  1, 2, 6,
  2, 6, 7,
  2, 3, 7,
  3, 7, 8,
  3, 4, 8,
  4, 8, 9,
  4, 5, 9,
  5, 9, 10,
  1, 5, 10,
  6, 10,11,
  6, 7, 11,
  7, 8, 11,
  8, 9, 11,
  9, 10,11
};

void vvSphere::initDodecaeder()
{
  float th = 0.0;
  float ph = 0.0;
  int   i;

  mNumVertices = 12;
  mNumTriangles = 20;

  if (mVertices != NULL)
    delete[] mVertices;
  if (mVerticesWorld != NULL)
    delete[] mVerticesWorld;
  if (mTriangles != NULL)
    delete[] mTriangles;
  if (mTexCoords != NULL)
    delete[] mTexCoords;

  mVertices  = new vvVertex[mNumVertices];
  mVerticesWorld  = new vvVertex[mNumVertices];
  mTriangles = new vvTriangle[mNumTriangles];
  mTexCoords = new vvTexCoord[mNumVertices];

  mVertices[0].x   = 0.;
  mVertices[0].y   = 0.;
  mVertices[11].x  = 0.;
  mVertices[11].y  = 0.;
  mVertices[0].z   = 1.;
  mVertices[11].z  = -1.;
  mVertices[0].th  = 0.;
  mVertices[0].ph  = 0.;
  mVertices[11].ph = 0.;
  mVertices[11].th = (float)M_PI;

  th = (float)M_PI / 3.0f;
  ph = 0.;
  for ( i = 1; i <= 5; i++)
  {
    mVertices[i].x  = (float)(sin(th) * cos(ph));
    mVertices[i].y  = (float)(sin(th) * sin(ph));
    mVertices[i].z  = (float)cos(th);
    mVertices[i].th = th;
    mVertices[i].ph = ph;
    ph += 2.0f * (float)M_PI / 5.0f;
  }

  th = 2.0f * (float)M_PI / 3.0f;
  ph = (float)M_PI / 5.0f;
  for ( i = 6; i <= 10; i++)
  {
    mVertices[i].x  = (float)(sin(th) * cos(ph));
    mVertices[i].y  = (float)(sin(th) * sin(ph));
    mVertices[i].z  = (float)cos(th);
    mVertices[i].th = th;
    mVertices[i].ph = ph;
    ph += 2.0f * (float) M_PI / 5.0f;
  }

  for (i = 0; i < mNumTriangles; i++)
  {
    mTriangles[i].v1 = dodecaederConnectivity[3*i + 0];
    mTriangles[i].v2 = dodecaederConnectivity[3*i + 1];
    mTriangles[i].v3 = dodecaederConnectivity[3*i + 2];
  }

  copyVerticesWorld();
}
