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

#include "vvglew.h"

#include "vvbrick.h"
#include "vvtexrend.h"

#include <algorithm>
#include <math.h>

void vvBrick::render(vvTexRend* const renderer, const vvVector3& normal,
                     const vvVector3& farthest, const vvVector3& delta,
                     const vvVector3& probeMin, const vvVector3& probeMax,
                     GLuint*& texNames, vvShaderManager* const isectShader, const bool setupEdges,
                     const vvVector3& eye, const bool isOrtho) const
{
  std::vector<vvBrick*> bricks;

  const vvVector3 voxSize(renderer->vd->getSize()[0] / (renderer->vd->vox[0] - 1),
                          renderer->vd->getSize()[1] / (renderer->vd->vox[1] - 1),
                          renderer->vd->getSize()[2] / (renderer->vd->vox[2] - 1));

  const vvVector3 halfBrick(float(renderer->texels[0]-renderer->_renderState._brickTexelOverlap) * 0.25f * voxSize[0],
                            float(renderer->texels[1]-renderer->_renderState._brickTexelOverlap) * 0.25f * voxSize[1],
                            float(renderer->texels[2]-renderer->_renderState._brickTexelOverlap) * 0.25f * voxSize[2]);
#if 0
  const float xHalf = (max[0] + min[0]) * 0.5f;
  const float x14   = pos[0] - halfBrick[0];
  const float x34   = pos[0] + halfBrick[0];

  const float yHalf = (max[1] + min[1]) * 0.5f;
  const float y14   = pos[1] - halfBrick[1];
  const float y34   = pos[1] + halfBrick[1];

  const float zHalf = (max[2] + min[2]) * 0.5f;
  const float z14   = pos[2] - halfBrick[2];
  const float z34   = pos[2] + halfBrick[2];

  const float texRangeX = (texRange[0] - texMin[0]) * 0.5f;
  const float texRangeY = (texRange[1] - texMin[1]) * 0.5f;
  const float texRangeZ = (texRange[2] - texMin[2]) * 0.5f;

  vvBrick brick0 = vvBrick(this);
  brick0.min = vvVector3(min[0], min[1], min[0]);//brick0.min.print("min");
  brick0.max = vvVector3(xHalf, yHalf, zHalf);//brick0.max.print("max");
  brick0.pos = vvVector3(x14, y14, z14);//brick0.pos.print("pos");
  brick0.texMin = vvVector3(texMin[0], texMin[1], texMin[2]);
  brick0.texRange = vvVector3(texRangeX, texRangeY, texRangeZ);
  bricks.push_back(&brick0);

  vvBrick brick1 = vvBrick(this);
  brick1.min = vvVector3(xHalf, min[1], min[2]);
  brick1.max = vvVector3(max[0], yHalf, zHalf);
  brick1.pos = vvVector3(x34, y14, z14);
  brick1.texMin = vvVector3(0.5f, texMin[1], texMin[2]);
  brick1.texRange = vvVector3(texRangeX, texRangeY, texRangeZ);
  bricks.push_back(&brick1);

  vvBrick brick2 = vvBrick(this);
  brick2.min = vvVector3(min[0], yHalf, min[2]);
  brick2.max = vvVector3(xHalf, max[1], zHalf);
  brick2.pos = vvVector3(x14, y34, z14);
  brick2.texMin = vvVector3(texMin[0], 0.5f, texMin[2]);
  brick2.texRange = vvVector3(texRangeX, texRangeY, texRangeZ);
  bricks.push_back(&brick2);

  vvBrick brick3 = vvBrick(this);
  brick3.min = vvVector3(xHalf, yHalf, min[2]);
  brick3.max = vvVector3(max[0], max[1], zHalf);
  brick3.pos = vvVector3(x34, y34, z14);
  brick3.texMin = vvVector3(0.5f, 0.5f, texMin[2]);
  brick3.texRange = vvVector3(texRangeX, texRangeY, texRangeZ);
  bricks.push_back(&brick3);

  vvBrick brick4 = vvBrick(this);
  brick4.min = vvVector3(min[0], min[1], zHalf);
  brick4.max = vvVector3(xHalf, yHalf, max[2]);
  brick4.pos = vvVector3(x14, y14, z34);
  brick4.texMin = vvVector3(texMin[0], texMin[1], 0.5f);
  brick4.texRange = vvVector3(texRangeX, texRangeY, texRangeZ);
  bricks.push_back(&brick4);

  vvBrick brick5 = vvBrick(this);
  brick5.min = vvVector3(xHalf, min[1], zHalf);
  brick5.max = vvVector3(max[0], yHalf, max[2]);
  brick5.pos = vvVector3(x34, y14, z34);
  brick5.texMin = vvVector3(0.5f, texMin[1], 0.5f);
  brick5.texRange = vvVector3(texRangeX, texRangeY, texRangeZ);
  bricks.push_back(&brick5);

  vvBrick brick6 = vvBrick(this);
  brick6.min = vvVector3(min[0], yHalf, zHalf);
  brick6.max = vvVector3(xHalf, max[1], max[2]);
  brick6.pos = vvVector3(x14, y34, z34);
  brick6.texMin = vvVector3(texMin[0], 0.5f,  0.5f);
  brick6.texRange = vvVector3(texRangeX, texRangeY, texRangeZ);
  bricks.push_back(&brick6);

  vvBrick brick7 = vvBrick(this);
  brick7.min = vvVector3(xHalf, yHalf, zHalf);
  brick7.max = vvVector3(max[0], max[1], max[2]);
  brick7.pos = vvVector3(x34, y34, z34);
  brick7.texMin = vvVector3(0.5f, 0.5f, 0.5f);
  brick7.texRange = vvVector3(texRangeX, texRangeY, texRangeZ);
  bricks.push_back(&brick7);

  renderer->sortBrickList(bricks, eye, normal, isOrtho);
#else
  vvBrick brick = vvBrick(this);
  bricks.push_back(&brick);
#endif

  glBindTexture(GL_TEXTURE_3D_EXT, texNames[index]);
  for (std::vector<vvBrick*>::const_iterator it = bricks.begin(); it != bricks.end(); ++it)
  {
    (*it)->renderGL(renderer, normal, farthest, delta, probeMin, probeMax,
                    texNames, isectShader, setupEdges);
  }
}

void vvBrick::renderGL(vvTexRend* const renderer, const vvVector3& normal,
                       const vvVector3& farthest, const vvVector3& delta,
                       const vvVector3& probeMin, const vvVector3& probeMax,
                       GLuint*& texNames, vvShaderManager* const isectShader, const bool setupEdges) const
{
  const vvVector3 dist = max - min;

  // Clip probe object to brick extents.
  vvVector3 minClipped;
  vvVector3 maxClipped;
  for (int i = 0; i < 3; ++i)
  {
    if (min[i] < probeMin[i])
    {
      minClipped[i] = probeMin[i];
    }
    else
    {
      minClipped[i] = min[i];
    }

    if (max[i] > probeMax[i])
    {
      maxClipped[i] = probeMax[i];
    }
    else
    {
      maxClipped[i] = max[i];
    }
  }

  const vvVector3 (&verts)[8] = vvAABB(minClipped, maxClipped).getVertices();

  float minDot;
  float maxDot;
  const ushort idx = getFrontIndex(verts, farthest, normal, minDot, maxDot);

  const float deltaInv = 1.0f / delta.length();

  const int startSlices = static_cast<const int>(ceilf(minDot * deltaInv));
  const int endSlices = static_cast<const int>(floorf(maxDot * deltaInv));

  if (renderer->_proxyGeometryOnGpu)
  {
#ifdef ISECT_CG
    isectShader->setArrayParameter3f(0, ISECT_SHADER_VERTICES, 0, verts[0].e[0], verts[0].e[1], verts[0].e[2]);
    if (setupEdges)
    {
      for (int i = 1; i < 8; ++i)
      {
        isectShader->setArrayParameter3f(0, ISECT_SHADER_VERTICES, i,
                                         verts[i].e[0]-verts[0].e[0],
                                         verts[i].e[1]-verts[0].e[1],
                                         verts[i].e[2]-verts[0].e[2]);
      }
    }
#else
    float edges[8 * 3];
    for (int i = 0; i < 8; ++i)
    {
      edges[i * 3] = verts[i].e[0];
      edges[i * 3 + 1] = verts[i].e[1];
      edges[i * 3 + 2] =  verts[i].e[2];
    }
    isectShader->setArray3f(0, ISECT_SHADER_VERTICES, edges, 8 * 3);
#endif

    // Pass planeStart along with brickMin and spare one setParameter call.
    isectShader->setParameter4f(0, ISECT_SHADER_BRICKMIN, min[0], min[1], min[2], -farthest.length());
    // Pass front index along with brickDimInv and spare one setParameter call.
    isectShader->setParameter4f(0, ISECT_SHADER_BRICKDIMINV, 1.0f/dist[0], 1.0f/dist[1], 1.0f/dist[2], idx);
    // Mind that textures overlap a little bit for correct interpolation at the borders.
    // Thus add that little difference.
    isectShader->setParameter3f(0, ISECT_SHADER_TEXRANGE, texRange[0], texRange[1], texRange[2]);
    isectShader->setParameter3f(0, ISECT_SHADER_TEXMIN, texMin[0], texMin[1], texMin[2]);

    const int primCount = (endSlices - startSlices) + 1;

#ifndef ISECT_GLSL_INST
#ifdef ISECT_GLSL_GEO
    glVertexPointer(2, GL_INT, 0, &renderer->_vertArray[startSlices*2]);
    glMultiDrawElements(GL_POINTS, &renderer->_elemCounts[0], GL_UNSIGNED_INT, (const GLvoid**)&renderer->_vertIndices[0], primCount);
#else
    glVertexPointer(2, GL_INT, 0, &renderer->_vertArray[startSlices*12]);
    glMultiDrawElements(GL_POLYGON, &renderer->_elemCounts[0], GL_UNSIGNED_INT, (const GLvoid**)&renderer->_vertIndices[0], primCount);
#endif

#else
    GLushort indexArray[6] = { 0, 1, 2, 3, 4, 5 };

    isectShader->setParameter1i(0, ISECT_SHADER_FIRSTPLANE, startSlices);
    glDrawElementsInstanced(GL_POLYGON, 6, GL_UNSIGNED_SHORT, indexArray, primCount);
#endif
  }
  else // render proxy geometry on gpu? else then:
  {
    vvVector3 startPoint = farthest + delta * static_cast<float>(startSlices);

    for (int i = startSlices; i <= endSlices; ++i)
    {
      vvVector3 isect[6];
      const int isectCnt = isect->isectPlaneCuboid(&normal, &startPoint, &minClipped, &maxClipped);
      startPoint.add(&delta);

      if (isectCnt < 3) continue;                 // at least 3 intersections needed for drawing

      // Check volume section mode:
      if (renderer->minSlice != -1 && i < renderer->minSlice) continue;
      if (renderer->maxSlice != -1 && i > renderer->maxSlice) continue;

      // Put the intersecting 3 to 6 vertices in cyclic order to draw adjacent
      // and non-overlapping triangles:
      isect->cyclicSort(isectCnt, &normal);

      // Generate vertices in texture coordinates:
      vvVector3 texcoord[6];
      for (int j = 0; j < isectCnt; ++j)
      {
        for (int k = 0; k < 3; ++k)
        {
          texcoord[j][k] = (isect[j][k] - min.e[k]) / dist[k];
          texcoord[j][k] = texcoord[j][k] * texRange[k] + texMin[k];
        }
      }

      glBegin(GL_TRIANGLE_FAN);
      glColor4f(1.0, 1.0, 1.0, 1.0);
      glNormal3f(normal[0], normal[1], normal[2]);
      for (int j = 0; j < isectCnt; ++j)
      {
        // The following lines are the bottleneck of this method:
        glTexCoord3f(texcoord[j][0], texcoord[j][1], texcoord[j][2]);
        glVertex3f(isect[j][0], isect[j][1], isect[j][2]);
      }
      glEnd();
    }
  }
}

void vvBrick::renderOutlines(const vvVector3& probeMin, const vvVector3& probeMax) const
{
  vvVector3 minClipped;
  vvVector3 maxClipped;
  for (int i = 0; i < 3; i++)
  {
    if (min.e[i] < probeMin.e[i])
    {
      minClipped.e[i] = probeMin.e[i];
    }
    else
    {
      minClipped.e[i] = min.e[i];
    }

    if (max.e[i] > probeMax.e[i])
    {
      maxClipped.e[i] = probeMax.e[i];
    }
    else
    {
      maxClipped.e[i] = max.e[i];
    }
  }
  glBegin(GL_LINES);
    glColor3f(1.0f, 1.0f, 1.0f);
    glVertex3f(minClipped[0], minClipped[1], minClipped[2]);
    glVertex3f(maxClipped[0], minClipped[1], minClipped[2]);

    glVertex3f(minClipped[0], minClipped[1], minClipped[2]);
    glVertex3f(minClipped[0], maxClipped[1], minClipped[2]);

    glVertex3f(minClipped[0], minClipped[1], minClipped[2]);
    glVertex3f(minClipped[0], minClipped[1], maxClipped[2]);

    glVertex3f(maxClipped[0], maxClipped[1], maxClipped[2]);
    glVertex3f(minClipped[0], maxClipped[1], maxClipped[2]);

    glVertex3f(maxClipped[0], maxClipped[1], maxClipped[2]);
    glVertex3f(maxClipped[0], minClipped[1], maxClipped[2]);

    glVertex3f(maxClipped[0], maxClipped[1], maxClipped[2]);
    glVertex3f(maxClipped[0], maxClipped[1], minClipped[2]);

    glVertex3f(maxClipped[0], minClipped[1], minClipped[2]);
    glVertex3f(maxClipped[0], maxClipped[1], minClipped[2]);

    glVertex3f(maxClipped[0], minClipped[1], minClipped[2]);
    glVertex3f(maxClipped[0], minClipped[1], maxClipped[2]);

    glVertex3f(minClipped[0], maxClipped[1], minClipped[2]);
    glVertex3f(maxClipped[0], maxClipped[1], minClipped[2]);

    glVertex3f(minClipped[0], maxClipped[1], minClipped[2]);
    glVertex3f(minClipped[0], maxClipped[1], maxClipped[2]);

    glVertex3f(minClipped[0], minClipped[1], maxClipped[2]);
    glVertex3f(maxClipped[0], minClipped[1], maxClipped[2]);

    glVertex3f(minClipped[0], minClipped[1], maxClipped[2]);
    glVertex3f(minClipped[0], maxClipped[1], maxClipped[2]);
  glEnd();
}

bool vvBrick::upload3DTexture(const GLuint& texName, const uchar* texData,
                              const GLenum texFormat, const GLint internalTexFormat,
                              const bool interpolation) const
{
  glBindTexture(GL_TEXTURE_3D_EXT, texName);

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
  glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MAG_FILTER, (interpolation) ? GL_LINEAR : GL_NEAREST);
  glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_MIN_FILTER, (interpolation) ? GL_LINEAR : GL_NEAREST);
  glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_R_EXT, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_S, GL_CLAMP);
  glTexParameteri(GL_TEXTURE_3D_EXT, GL_TEXTURE_WRAP_T, GL_CLAMP);

  glTexImage3D(GL_PROXY_TEXTURE_3D_EXT, 0, internalTexFormat,
    texels[0], texels[1], texels[2], 0, texFormat, GL_UNSIGNED_BYTE, NULL);

  GLint glWidth;
  glGetTexLevelParameteriv(GL_PROXY_TEXTURE_3D_EXT, 0, GL_TEXTURE_WIDTH, &glWidth);
  if (glWidth != 0)
  {
    glTexImage3D(GL_TEXTURE_3D_EXT, 0, internalTexFormat, texels[0], texels[1], texels[2], 0,
      texFormat, GL_UNSIGNED_BYTE, texData);
    return true;
  }
  else
  {
    return false;
  }
}

/** Get front index of the brick based upon the current modelview matrix.
  @param vertices  NOTE THE FOLLOWING: albeit vertices are constituents of
                   bricks, these are passed along with the function call in
                   order not to recalculate them. This is due to the fact that
                   AABBs actually don't store pointers to their eight verts,
                   but only to two corners.
  @param point     The point to calc the distance from. The distance of interest
                   is the one from point to the first and last vertex along
                   normal
  @param normal    The normal to calc the distance along.
  @param minDot    The minimum dot product of vector point-vertex and normal,
                   passed along for later calculations.
  @param maxDot    The maximum dot product of vector point-vertex and normal,
                   passed along for later calculations.
*/
ushort vvBrick::getFrontIndex(const vvVector3* vertices,
                              const vvVector3& point,
                              const vvVector3& normal,
                              float& minDot,
                              float& maxDot) const
{

  // Get vertices with max and min distance to point along normal.
  maxDot = -VV_FLT_MAX;
  minDot = VV_FLT_MAX;
  ushort frontIndex;

  for (int i=0; i<8; i++)
  {
    const vvVector3 v = vertices[i] - point;
    const float dot = v.dot(&normal);
    if (dot > maxDot)
    {
      maxDot = dot;
      frontIndex = i;
    }

    if (dot < minDot)
    {
      minDot = dot;
    }
  }
  return frontIndex;
}

void vvBrick::print() const
{
  cerr << "Brick:\t" << index << endl;
  pos.print("pos:");
  min.print("min:");
  max.print("max:");
  texRange.print("texRange:");
  texMin.print("texMin:");
  cerr << "minValue:\t" << minValue << endl;
  cerr << "maxValue:\t" << maxValue << endl;
  cerr << "visible:\t" << visible << endl;
  cerr << "atBorder:\t" << atBorder << endl;
  cerr << "insideProbe:\t" << insideProbe << endl;
  cerr << "startOffset:\t" << startOffset[0] << " " << startOffset[1] << " " << startOffset[2] << endl;
  cerr << "texels:\t" << texels[0] << " " << texels[1] << " " << texels[2] << endl;
  cerr << "dist:\t" << dist << endl;
}

void vvBrick::sortByCenter(std::vector<vvBrick*>& bricks, const vvVector3& axis)
{
  const vvVector3 axisGetter(0, 1, 2);
  const int a = static_cast<const int>(axis.dot(&axisGetter));

  for(std::vector<vvBrick*>::iterator it = bricks.begin(); it != bricks.end(); ++it)
  {
    (*it)->dist = (*it)->getAABB().getCenter()[a];
  }
  std::sort(bricks.begin(), bricks.end(), vvBrick::Compare());
}
