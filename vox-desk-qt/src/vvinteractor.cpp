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

#include "vvinteractor.h"

vvInteractor::vvInteractor()
  : _enabled(true)
  , _hasFocus(false)
  , _visible(true)
  , _pos(vvVector3())
{

}

vvInteractor::~vvInteractor()
{

}

void vvInteractor::setEnabled(bool enabled)
{
  _enabled = enabled;
}

void vvInteractor::setFocus()
{
  _hasFocus = true;
}

void vvInteractor::clearFocus()
{
  _hasFocus = false;
}

void vvInteractor::setVisible(bool visible)
{
  _visible = visible;
}

void vvInteractor::setPos(const vvVector3& pos)
{
  _pos = pos;
}

bool vvInteractor::enabled() const
{
  return _enabled;
}

bool vvInteractor::hasFocus() const
{
  return _hasFocus;
}

bool vvInteractor::visible() const
{
  return _visible;
}

vvVector3 vvInteractor::pos() const
{
  return _pos;
}

