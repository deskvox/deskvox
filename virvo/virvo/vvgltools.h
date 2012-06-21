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

#if !defined(VV_LIBRARY_BUILD) && !defined(VV_APPLICATION_BUILD)
#error "vvgltools.h is meant for internal use only"
#endif

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
    struct Viewport                               /// for convenience
    {
      Viewport() { for(int i=0; i<4; ++i) values[i]=0; }
      Viewport(int x, int y, int w, int h) { values[0]=x; values[1]=y; values[2]=w; values[3]=h; }
      int values[4];

      inline int &operator[](const unsigned int i)
      {
        return values[i];
      }

      inline int operator[](const unsigned int i) const
      {
        return values[i];
      }

      inline void print() const
      {
          std::cerr << "Left: " << values[0] << " "
             << "Top: " << values[1] << " "
             << "Width: " << values[2] << " "
             << "Height: " << values[3] << std::endl;
      }
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
    static void drawViewAlignedQuad(const float x1 = -1.0f, const float y1 = -1.0f,
                                    const float x2 =  1.0f, const float y2 =  1.0f);
    static Viewport getViewport();
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

      const Viewport viewport = getViewport();

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
      result.x = std::max(0, static_cast<int>(floorf(minX)));
      result.y = std::max(0, static_cast<int>(floorf(minY)));
      result.width = std::min(static_cast<int>(ceilf(fabsf(maxX - minX))), viewport[2] - result.x);
      result.height = std::min(static_cast<int>(ceilf(fabsf(maxY - minY))), viewport[3] - result.y);

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
#endif

//============================================================================
// End of File
//============================================================================
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
