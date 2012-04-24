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
#include "vvgltools.h"
#include "vvtexrend.h"
#include "vvshaderprogram.h"

#include <algorithm>
#include <math.h>

using std::cerr;
using std::endl;

void vvBrick::render(vvTexRend* const renderer, const vvVector3& normal,
                     const vvVector3& farthest, const vvVector3& delta,
                     const vvVector3& probeMin, const vvVector3& probeMax,
                     GLuint*& texNames, vvShaderProgram* shader) const
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

  if(shader && renderer->voxelType == vvTexRend::VV_PIX_SHD )
  {
    shader->setParameterTex3D("pix3dtex", texNames[index]);
  }
  else
  {
    glBindTexture(GL_TEXTURE_3D_EXT, texNames[index]);
  }
  if (renderer->_isectType != vvTexRend::CPU)
  {
    float edges[8 * 3];
    if (renderer->_isectType == vvTexRend::REFERENCE_IMPL)
    {
      for (int i = 0; i < 8; ++i)
      {
        edges[i * 3] = verts[i].e[0];
        edges[i * 3 + 1] = verts[i].e[1];
        edges[i * 3 + 2] = verts[i].e[2];
      }
    }
    else
    {
      const int sequence[64] = { 0, 1, 2, 3, 4, 5, 6, 7,
                                 1, 2, 3, 0, 7, 4, 5, 6,
                                 2, 7, 6, 3, 4, 1, 0, 5,
                                 3, 6, 5, 0, 7, 2, 1, 4,
                                 4, 5, 6, 7, 0, 1, 2, 3,
                                 5, 0, 3, 6, 1, 4, 7, 2,
                                 6, 7, 4, 5, 2, 3, 0, 1,
                                 7, 6, 3, 2, 5, 4, 1, 0 };

      for (int i = 0; i < 8; ++i)
      {
        edges[i * 3] = verts[sequence[idx * 8 + i]][0];
        edges[i * 3 + 1] = verts[sequence[idx * 8 + i]][1];
        edges[i * 3 + 2] = verts[sequence[idx * 8 + i]][2];
      }
    }
    shader->setParameterArray3f("vertices", edges, 8);
    // Pass planeStart along with brickMin and spare one setParameter call.
    shader->setParameter4f("brickMin", min[0], min[1], min[2], -farthest.length());
    if (renderer->_isectType == vvTexRend::REFERENCE_IMPL)
    {
      // Pass front index along with brickDimInv and spare one setParameter call.
      shader->setParameter4f("brickDimInv", 1.0f/dist[0], 1.0f/dist[1], 1.0f/dist[2], idx);
    }
    else
    {
      shader->setParameter3f("brickDimInv", 1.0f/dist[0], 1.0f/dist[1], 1.0f/dist[2]);
    }
    // Mind that textures overlap a little bit for correct interpolation at the borders.
    // Thus add that little difference.
    shader->setParameter3f("texRange", texRange[0], texRange[1], texRange[2]);
    shader->setParameter3f("texMin", texMin[0], texMin[1], texMin[2]);
    const int primCount = (endSlices - startSlices) + 1;

    int vertArrayIdx = startSlices * 12; // vert shader
    if (renderer->_isectType == vvTexRend::GEOM_SHADER_ONLY)
    {
      vertArrayIdx = startSlices * 2;
    }
    else if (renderer->_isectType == vvTexRend::VERT_GEOM_COMBINED)
    {
      vertArrayIdx = startSlices * 6;
    }
    glVertexPointer(2, GL_INT, 0, &renderer->_vertArray[vertArrayIdx]);

    GLenum inputType = GL_POLYGON; // vert shader
    if (renderer->_isectType == vvTexRend::GEOM_SHADER_ONLY)
    {
      inputType = GL_POINTS;
    }
    else if (renderer->_isectType == vvTexRend::VERT_GEOM_COMBINED)
    {
      inputType = GL_TRIANGLES;
    }
    glMultiDrawElements(inputType, &renderer->_elemCounts[0], GL_UNSIGNED_INT, (const GLvoid**)&renderer->_vertIndices[0], primCount);
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
          texcoord[j][k] = (isect[j][k] - min[k]) / dist[k];
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
  ushort frontIndex = 0;

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
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
