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

#ifndef VVPARAM_H_INCLUDED
#define VVPARAM_H_INCLUDED


#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#include "vvinttypes.h"
#include "vvvecmath.h"
#include "vvcolor.h"
#include "vvaabb.h"

#include <cassert>

#include <boost/any.hpp>


class vvParam
{
public:
  enum Type {
    VV_EMPTY,
    VV_BOOL,
    VV_CHAR,
    VV_UCHAR,
    VV_SHORT,
    VV_USHORT,
    VV_INT,
    VV_UINT,
    VV_LONG,
    VV_ULONG,
    VV_LLONG,
    VV_ULLONG,
    VV_FLOAT,
    VV_VEC2,
    VV_VEC2I,
    VV_VEC3,
    VV_VEC3I,
    VV_VEC4,
    VV_SIZE3,
    VV_COLOR,
    VV_AABB,
    VV_AABBI,
    VV_AABBS
  };

private:
  // The type of this parameter
  Type type;
  // The value of this parameter
  boost::any value;

public:
  vvParam() : type(VV_EMPTY)
  {
  }

  vvParam(const bool& val)
    : type(VV_BOOL)
    , value(val)
  {
  }

  vvParam(const char& val)
    : type(VV_CHAR)
    , value(val)
  {
  }

  vvParam(const unsigned char& val)
    : type(VV_UCHAR)
    , value(val)
  {
  }

  vvParam(const short& val)
    : type(VV_SHORT)
    , value(val)
  {
  }

  vvParam(const unsigned short& val)
    : type(VV_USHORT)
    , value(val)
  {
  }

  vvParam(const int& val)
    : type(VV_INT)
    , value(val)
  {
  }

  vvParam(const unsigned& val)
    : type(VV_UINT)
    , value(val)
  {
  }

  vvParam(const long& val)
    : type(VV_LONG)
    , value(val)
  {
  }

  vvParam(const unsigned long& val)
    : type(VV_ULONG)
    , value(val)
  {
  }

#if VV_HAVE_LLONG
  vvParam(const long long& val)
    : type(VV_LLONG)
    , value(val)
  {
  }
#endif

#if VV_HAVE_ULLONG
  vvParam(const unsigned long long& val)
    : type(VV_ULLONG)
    , value(val)
  {
  }
#endif

  vvParam(const float& val)
    : type(VV_FLOAT)
    , value(val)
  {
  }

  vvParam(const vvVector2& val)
    : type(VV_VEC2)
    , value(val)
  {
  }

  vvParam(const vvVector2i& val)
    : type(VV_VEC2I)
    , value(val)
  {
  }

  vvParam(const vvVector3& val)
    : type(VV_VEC3)
    , value(val)
  {
  }

  vvParam(const vvVector3i& val)
    : type(VV_VEC3I)
    , value(val)
  {
  }

  vvParam(const vvVector4& val)
    : type(VV_VEC4)
    , value(val)
  {
  }

  vvParam(const vvsize3& val)
    : type(VV_SIZE3)
    , value(val)
  {
  }

  vvParam(const vvColor& val)
    : type(VV_COLOR)
    , value(val)
  {
  }
  
  vvParam(const vvAABB& val)
    : type(VV_AABB)
    , value(val)
  {
  }

  vvParam(const vvAABBi& val)
    : type(VV_AABBI)
    , value(val)
  {
  }

  vvParam(const vvAABBs& val)
    : type(VV_AABBS)
    , value(val)
  {
  }

  bool asBool() const
  {
    return boost::any_cast<bool>(value);
  }

  char asChar() const
  {
    return boost::any_cast<char>(value);
  }

  unsigned char asUchar() const
  {
    return boost::any_cast<unsigned char>(value);
  }

  short asShort() const
  {
    return boost::any_cast<short>(value);
  }

  unsigned short asUshort() const
  {
    return boost::any_cast<unsigned short>(value);
  }

  int asInt() const
  {
    return boost::any_cast<int>(value);
  }

  unsigned int asUint() const
  {
    return boost::any_cast<unsigned int>(value);
  }

  long asLong() const
  {
    return boost::any_cast<long>(value);
  }

  unsigned long asUlong() const
  {
    return boost::any_cast<unsigned long>(value);
  }

#if VV_HAVE_LLONG
  long long asLlong() const
  {
    return boost::any_cast<long long>(value);
  }
#endif

#if VV_HAVE_ULLONG
  unsigned long long asUllong() const
  {
    return boost::any_cast<unsigned long long>(value);
  }
#endif

  float asFloat() const
  {
    return boost::any_cast<float>(value);
  }

  vvVector2 asVec2() const
  {
    return boost::any_cast<vvVector2>(value);
  }

  vvVector2i asVec2i() const
  {
    return boost::any_cast<vvVector2i>(value);
  }

  vvVector3 asVec3() const
  {
    return boost::any_cast<vvVector3>(value);
  }

  vvVector3i asVec3i() const
  {
    return boost::any_cast<vvVector3i>(value);
  }

  vvVector4 asVec4() const
  {
    return boost::any_cast<vvVector4>(value);
  }

  vvsize3 asSize3() const
  {
    return boost::any_cast<vvsize3>(value);
  }

  vvColor asColor() const
  {
    return boost::any_cast<vvColor>(value);
  }
  
  vvAABB asAABB() const
  {
    return boost::any_cast<vvAABB>(value);
  }

  vvAABBi asAABBi() const
  {
    return boost::any_cast<vvAABBi>(value);
  }

  vvAABBs asAABBs() const
  {
    return boost::any_cast<vvAABBs>(value);
  }

  operator bool() const { return asBool(); }
  operator char() const { return asChar(); }
  operator unsigned char() const { return asUchar(); }
  operator short() const { return asShort(); }
  operator unsigned short() const { return asUshort(); }
  operator int() const { return asInt(); }
  operator unsigned int() const { return asUint(); }
  operator long() const { return asLong(); }
  operator unsigned long() const { return asUlong(); }
#if VV_HAVE_LLONG
  operator long long() const { return asLlong(); }
#endif
#if VV_HAVE_ULLONG
  operator unsigned long long() const { return asUllong(); }
#endif
  operator float() const { return asFloat(); }
  operator vvVector2() const { return asVec2(); }
  operator vvVector2i() const { return asVec2i(); }
  operator vvVector3() const { return asVec3(); }
  operator vvVector3i() const { return asVec3i(); }
  operator vvVector4() const { return asVec4(); }
  operator vvsize3() const { return asSize3(); }
  operator vvColor() const { return asColor(); }
  operator vvAABB() const { return asAABB(); }
  operator vvAABBi() const { return asAABBi(); }
  operator vvAABBs() const { return asAABBs(); }

  // Returns the type of this parameter
  Type getType() const { return type; }

  // Returns whether this parameter is of type t
  bool isa(Type t) const { return type == t; }
};


#endif

