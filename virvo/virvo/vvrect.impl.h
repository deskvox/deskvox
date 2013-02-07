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

#include "vvvecmath.h"

template <typename T>
vvBaseRect<T>::vvBaseRect()
{
  for (int i = 0; i < 4; ++i)
  {
    values[i] = 0;
  }
}

template <typename T>
vvBaseRect<T>::vvBaseRect(T x, T y, T width, T height)
{
  values[0] = x;
  values[1] = y;
  values[2] = width;
  values[3] = height;
}

template <typename T>
bool vvBaseRect<T>::contains(const vvBaseRect<T>& rhs)
{
  T x = values[0];
  T y = values[1];
  T width = values[2];
  T height = values[3];

  const vvBaseVector2<T> pt1 = vvBaseVector2<T>(x, y);
  const vvBaseVector2<T> pt2 = vvBaseVector2<T>(x + width, y + height);

  const vvBaseVector2<T> rpt1 = vvBaseVector2<T>(rhs[0], rhs[1]);
  const vvBaseVector2<T> rpt2 = vvBaseVector2<T>(rhs[0] + rhs[2], rhs[1] + rhs[3]);

  return (pt1[0] >= rpt1[0] && pt2[0] <= rpt2[0] && pt1[1] >= rpt1[1] && pt2[1] <= rpt2[1]);
}

template <typename T>
bool vvBaseRect<T>::overlaps(const vvBaseRect<T>& rhs)
{
  T x = values[0];
  T y = values[1];
  T width = values[2];
  T height = values[3];

  const vvBaseVector2<T> pt1 = vvBaseVector2<T>(x, y);
  const vvBaseVector2<T> pt2 = vvBaseVector2<T>(x + width, y + height);

  const vvBaseVector2<T> rpt1 = vvBaseVector2<T>(rhs[0], rhs[1]);
  const vvBaseVector2<T> rpt2 = vvBaseVector2<T>(rhs[0] + rhs[2], rhs[1] + rhs[3]);

  return !(pt1[0] > rpt2[0] || pt2[0] < rpt1[0] || pt1[1] > rpt2[1] || pt2[1] < rpt1[1]);
}

template <typename T>
void vvBaseRect<T>::intersect(const vvBaseRect<T>& rhs)
{
  T x = values[0];
  T y = values[1];
  T width = values[2];
  T height = values[3];

  if (overlaps(rhs))
  {
    const vvBaseVector2<T> pt1 = vvBaseVector2<T>(x, y);
    const vvBaseVector2<T> pt2 = vvBaseVector2<T>(x + width, y + height);

    const vvBaseVector2<T> rpt1 = vvBaseVector2<T>(rhs[0], rhs[1]);
    const vvBaseVector2<T> rpt2 = vvBaseVector2<T>(rhs[0] + rhs[2], rhs[1] + rhs[3]);

    x = std::max(pt1[0], rpt1[0]);
    y = std::max(pt1[1], rpt1[1]);

    const int x2 = std::min(pt2[0], rpt2[0]);
    const int y2 = std::min(pt2[1], rpt2[1]);

    width = x2 - x;
    height = y2 - y;
  }
  else
  {
    x = 0;
    y = 0;
    width = 0;
    height = 0;
  }
}

