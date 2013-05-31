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

#ifndef _VVGLTOOLS_H_
#define _VVGLTOOLS_H_

#include "vvexport.h"
#include "vvvecmath.h"
#include "vvaabb.h"
#include "vvopengl.h"

#include <iostream>

//============================================================================
// Class Definitions
//============================================================================

/** Collection of OpenGL raleted tools.
    Consists of static helper functions which are project independent.
    @author Juergen Schulze-Doebold
*/
class VIRVOEXPORT vvGLTools
{
  public:
    enum DisplayStyle                             /// string display style for extensions display
    {
      CONSECUTIVE = 0,                            ///< using entire line length
      ONE_BY_ONE  = 1                             ///< one extension per line
    };
    struct GLInfo
    {
      const char* vendor;
      const char* renderer;
      const char* version;
    };

    static bool enableGLErrorBacktrace(bool printBacktrace = true, bool abortOnError = false);
    static void printGLError(const char*);
    static GLInfo getGLInfo();
    static bool isGLVersionSupported(int major, int minor, int release);
    static bool isGLextensionSupported(const char*);
    static void displayOpenGLextensions(const DisplayStyle);
    static void checkOpenGLextensions();
    static void drawQuad(float x1 = -1.0f, float y1 = -1.0f, float x2 =  1.0f, float y2 =  1.0f);
    static virvo::Viewport getViewport();
    static vvVector4 queryClearColor();
    static void getModelviewMatrix(vvMatrix*);
    static void getProjectionMatrix(vvMatrix*);
    static void setModelviewMatrix(const vvMatrix& mv);
    static void setProjectionMatrix(const vvMatrix& pr);
    static void getClippingPlanes(vvPlane& znear, vvPlane& zfar);
    static vvVector3 project(const vvVector3& obj);
    static vvVector3 unProject(const vvVector3& win);

    /*! calc bounding rect of box in screen space coordinates
     */ 
    template<typename T>
    static vvRecti getBoundingRect(const vvBaseAABB<T>& aabb)
    {
      GLdouble modelview[16];
      glGetDoublev(GL_MODELVIEW_MATRIX, modelview);

      GLdouble projection[16];
      glGetDoublev(GL_PROJECTION_MATRIX, projection);

      const virvo::Viewport viewport = getViewport();

      T minX = std::numeric_limits<T>::max();
      T minY = std::numeric_limits<T>::max();
      T maxX = -std::numeric_limits<T>::max();
      T maxY = -std::numeric_limits<T>::max();
      
      const typename vvBaseAABB<T>::vvBoxCorners& vertices = aabb.getVertices();

      for (int i = 0; i < 8; ++i)
      {
        GLdouble vertex[3];
        gluProject(vertices[i][0], vertices[i][1], vertices[i][2],
                   modelview, projection, viewport.values,
                   &vertex[0], &vertex[1], &vertex[2]);

        if (vertex[0] < minX)
        {
          minX = static_cast<T>(vertex[0]);
        }
        if (vertex[0] > maxX)
        {
          maxX = static_cast<T>(vertex[0]);
        }

        if (vertex[1] < minY)
        {
          minY = static_cast<T>(vertex[1]);
        }
        if (vertex[1] > maxY)
        {
          maxY = static_cast<T>(vertex[1]);
        }
      }

      vvRecti result;
      result[0] = std::max(0, static_cast<int>(floorf(minX)));
      result[1] = std::max(0, static_cast<int>(floorf(minY)));
      result[2] = std::min(static_cast<int>(ceilf(fabsf(maxX - minX))), viewport[2] - result[0]);
      result[3] = std::min(static_cast<int>(ceilf(fabsf(maxY - minY))), viewport[3] - result[1]);

      return result;
    }
    
    template<typename T>
    static void render(const vvBaseAABB<T>& aabb)
    {
      const typename vvBaseAABB<T>::vvBoxCorners& vertices = aabb.getVertices();

      glBegin(GL_LINES);
        glColor3f(1.0f, 1.0f, 1.0f);

        // front
        glVertex3f(vertices[0][0], vertices[0][1], vertices[0][2]);
        glVertex3f(vertices[1][0], vertices[1][1], vertices[1][2]);

        glVertex3f(vertices[1][0], vertices[1][1], vertices[1][2]);
        glVertex3f(vertices[2][0], vertices[2][1], vertices[2][2]);

        glVertex3f(vertices[2][0], vertices[2][1], vertices[2][2]);
        glVertex3f(vertices[3][0], vertices[3][1], vertices[3][2]);

        glVertex3f(vertices[3][0], vertices[3][1], vertices[3][2]);
        glVertex3f(vertices[0][0], vertices[0][1], vertices[0][2]);

        // back
        glVertex3f(vertices[4][0], vertices[4][1], vertices[4][2]);
        glVertex3f(vertices[5][0], vertices[5][1], vertices[5][2]);

        glVertex3f(vertices[5][0], vertices[5][1], vertices[5][2]);
        glVertex3f(vertices[6][0], vertices[6][1], vertices[6][2]);

        glVertex3f(vertices[6][0], vertices[6][1], vertices[6][2]);
        glVertex3f(vertices[7][0], vertices[7][1], vertices[7][2]);

        glVertex3f(vertices[7][0], vertices[7][1], vertices[7][2]);
        glVertex3f(vertices[4][0], vertices[4][1], vertices[4][2]);

        // left
        glVertex3f(vertices[5][0], vertices[5][1], vertices[5][2]);
        glVertex3f(vertices[0][0], vertices[0][1], vertices[0][2]);

        glVertex3f(vertices[3][0], vertices[3][1], vertices[3][2]);
        glVertex3f(vertices[6][0], vertices[6][1], vertices[6][2]);

        // right
        glVertex3f(vertices[1][0], vertices[1][1], vertices[1][2]);
        glVertex3f(vertices[4][0], vertices[4][1], vertices[4][2]);

        glVertex3f(vertices[7][0], vertices[7][1], vertices[7][2]);
        glVertex3f(vertices[2][0], vertices[2][1], vertices[2][2]);
      glEnd();
    }
};

namespace virvo
{
namespace gltools
{
std::string lastError(const std::string& file, int line);
VVAPI Matrix getModelViewMatrix();
VVAPI Matrix getProjectionMatrix();
}
}

inline bool operator==(const virvo::Viewport& vp1, const virvo::Viewport& vp2)
{
  return (vp1[0] == vp2[0] && vp1[1] == vp2[1] && vp1[2] == vp2[2] && vp1[3] == vp2[3]);
}

inline bool operator!=(const virvo::Viewport& vp1, const virvo::Viewport& vp2)
{
  return (vp1[0] != vp2[0] || vp1[1] != vp2[1] || vp1[2] != vp2[2] || vp1[3] != vp2[3]);
}

inline std::ostream& operator<<(std::ostream& out, const virvo::Viewport& vp)
{
  out << "Left: " << vp[0] << " "
      << "Top: " << vp[1] << " "
      << "Width: " << vp[2] << " "
      << "Height: " << vp[3];
  return out;
}

#define VV_GLERROR virvo::gltools::lastError(__FILE__, __LINE__)

#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
