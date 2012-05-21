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

#ifndef VV_AABB_H
#define VV_AABB_H

#include "vvvecmath.h"
#ifdef min
#undef min
#undef max
#endif

template <typename T>
class vvBaseRect
{
public:
  T x;
  T y;
  T width;
  T height;
};

typedef vvBaseRect<int> vvRecti;
typedef vvBaseRect<float> vvRectf;
typedef vvBaseRect<double> vvRectd;
typedef vvRecti vvRect;

template <typename T>
class vvBaseAABB;

typedef vvBaseAABB<int> vvAABBi;
typedef vvBaseAABB<float> vvAABBf;
typedef vvBaseAABB<double> vvAABBd;
typedef vvAABBf vvAABB;

/*!
 * \brief           Axis aligned bounding box (AABB).
 *
 *                  These can simply be specified by two opposite
 *                  corner points. This implementations stores
 *                  the precalculated values of the eight corner
 *                  vertices and the center vertex.
 */
template <typename T>
class vvBaseAABB
{
public:
  typedef vvBaseVector3<T> vvBoxCorners[8];
 
  vvBaseAABB(const vvBaseVector3<T>& min, const vvBaseVector3<T>& max);

  const vvBaseVector3<T>& min() const;
  const vvBaseVector3<T>& max() const;

  /*!
   * \brief         Calc the width of the aabb.
   *
   *                Width is calculated from the corners rather than
   *                stored by the aabb data type. Mind this when using
   *                this method in time critical situations.
   * \return        The calculated width.
   */
  T calcWidth() const;
  /*!
   * \brief         Calc the height of the aabb.
   *
   *                Height is calculated from the corners rather than
   *                stored by the aabb data type. Mind this when using
   *                this method in time critical situations.
   * \return        The calculated height.
   */
  T calcHeight() const;
  /*!
   * \brief         Calc the depth of the aabb.
   *
   *                Depth is calculated from the corners rather than
   *                stored by the aabb data type. Mind this when using
   *                this method in time critical situations.
   * \return        The calculated depth.
   */
  T calcDepth() const;

  /*!
   * \brief         Get the box vertices.
   *
   *                Returns the precalculated box corner vertices.
   */
  const vvBoxCorners& getVertices() const;

  /*!
   * \brief         Get the center point.
   *
   *                Returns the stored center.
   */
  const vvBaseVector3<T>& getCenter() const;

  /*!
   * \brief         Get a rectangle of the projected screen section.
   *
   *                Calcs the rectangle defined to fully enclose the
   *                the projected area do to the current camera transformations.
   * \return        The rectangle of the projected screen section.
   */
  vvRecti getProjectedScreenRect() const;

  /*!
   * \brief         Shrink the box to the intersection with another one.
   *
   *                Get the box resulting from intersecting this box with
   *                the one specified.
   * \param         rhs The box to intersect with.
   */
  void intersect(const vvBaseAABB<T>& rhs);

  /*!
   * \brief         Render the bounding box.
   *
   *                Render the outlines of the bounding box using opengl
   *                commands.
   */
  void render() const;
private:
  vvBaseVector3<T> _min;
  vvBaseVector3<T> _max;
  vvBaseVector3<T> _vertices[8];
  vvBaseVector3<T> _center;

  /*!
   * \brief         Calc the 8 corner vertices.
   *
   *                Calc the 8 corner vertices given the two vectors
   *                with maximum extend.
   */
  void calcVertices();
};

inline std::ostream& operator<<(std::ostream& out, const vvRect& r)
{
  out << "x: " << r.x << ", y: " << r.y << ", width: " << r.width << ", height: " << r.height;
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const vvAABB& aabb)
{
  out << aabb.min() << "\n" << aabb.max();
  return out;
}

#include "vvaabb.impl.h"

#endif // VV_AABB_H
// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
