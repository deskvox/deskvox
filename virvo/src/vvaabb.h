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

typedef vvVector3 vvBoxCorners[8];

class vvRect
{
public:
  int x;
  int y;
  int width;
  int height;

  void print() const;
};

/*!
 * \brief           Axis aligned bounding box (AABB).
 *
 *                  These can simply be specified by two opposite
 *                  corner points. This implementations stores
 *                  the precalculated values of the eight corner
 *                  vertices and the center vertex.
 */
class vvAABB
{
public:
  vvAABB(const vvVector3& min, const vvVector3& max);

  vvVector3 min() const;
  vvVector3 max() const;

  /*!
   * \brief         Calc the width of the aabb.
   *
   *                Width is calculated from the corners rather than
   *                stored by the aabb data type. Mind this when using
   *                this method in time critical situations.
   * \return        The calculated width.
   */
  float calcWidth() const;
  /*!
   * \brief         Calc the height of the aabb.
   *
   *                Height is calculated from the corners rather than
   *                stored by the aabb data type. Mind this when using
   *                this method in time critical situations.
   * \return        The calculated height.
   */
  float calcHeight() const;
  /*!
   * \brief         Calc the depth of the aabb.
   *
   *                Depth is calculated from the corners rather than
   *                stored by the aabb data type. Mind this when using
   *                this method in time critical situations.
   * \return        The calculated depth.
   */
  float calcDepth() const;

  /*!
   * \brief         Calc the minimum extent along the specified axis.
   *
   *                If you desire the x-value of the left side of the
   *                box, pass vvVector3(1, 0, 0) as axis. Make sure that
   *                the vector component depicting the desired component
   *                equals 1.
   * \param         axis A normalized vector representing the coord axis.
   */
  float calcMinExtent(const vvVector3& axis) const;
  /*!
   * \brief         Calc the maximum extent along the specified axis.
   *
   *                If you desire the x-value of the right side of the
   *                box, pass vvVector3(1, 0, 0) as axis. Make sure that
   *                the vector component depicting the desired component
   *                equals 1.
   * \param         axis A normalized vector representing the coord axis.
   */
  float calcMaxExtent(const vvVector3& axis) const;

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
  vvVector3 getCenter() const;

  /*!
   * \brief         Get a rectangle of the projected screen section.
   *
   *                Calcs the rectangle defined to fully enclose the
   *                the projected area do to the current camera transformations.
   * \return        The rectangle of the projected screen section.
   */
  vvRect* getProjectedScreenRect() const;

  /*!
   * \brief         Shring the box to the intersection with another one.
   *
   *                Get the box resulting from intersecting this box with
   *                the one specified.
   * \param         rhs The box to intersect with.
   */
  void intersect(vvAABB* rhs);

  /*!
   * \brief         Render the bounding box.
   *
   *                Render the outlines of the bounding box using opengl
   *                commands.
   */
  void render() const;
  /*!
   * \brief         Print the box extents to stdout.
   *
   *                Print the bottom/left/back corner and the
   *                top/right/front corner in that order.
   */
  void print() const;
private:
  vvVector3 _min;
  vvVector3 _max;
  vvVector3 _vertices[8];
  vvVector3 _center;

  /*!
   * \brief         Calc the 8 corner vertices.
   *
   *                Calc the 8 corner vertices given the two vectors
   *                with maximum extend.
   */
  void calcVertices();
};

#endif // VV_AABB_H
