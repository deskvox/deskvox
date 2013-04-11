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


#include "vvinttypes.h"
#include "vvvecmath.h"
#include "vvcolor.h"
#include "vvaabb.h"

#include <cassert>


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
#ifdef _WIN32
    VV_LLONG,
    VV_ULLONG,
#endif
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
  union Value {
    bool B;
    char C;
    unsigned char UC;
    short S;
    unsigned short US;
    int I;
    unsigned UI;
    long L;
    unsigned long UL;
#ifdef _WIN32
    long long int LL;
    unsigned long long int ULL;
#endif
    float F;
    const vvVector2* Vec2;
    const vvVector2i* Vec2I;
    const vvVector3* Vec3;
    const vvVector3i* Vec3I;
    const vvVector4* Vec4;
    const vvsize3* Size3;
    const vvColor* Color;
    const vvAABB* AABB;
    const vvAABBi* AABBI;
    const vvAABBs* AABBS;
  };

  // The type of this parameter
  Type type;
  // The value of this parameter
  Value value;

public:
  vvParam() : type(VV_EMPTY)
  {
  }

  vvParam(const bool& val) : type(VV_BOOL)
  {
    value.B = val;
  }

  vvParam(const char& val) : type(VV_CHAR)
  {
    value.C = val;
  }

  vvParam(const unsigned char& val) : type(VV_UCHAR)
  {
    value.UC = val;
  }

  vvParam(const short& val) : type(VV_SHORT)
  {
    value.S = val;
  }

  vvParam(const unsigned short& val) : type(VV_USHORT)
  {
    value.US = val;
  }

  vvParam(const int& val) : type(VV_INT)
  {
    value.I = val;
  }

  vvParam(const unsigned& val) : type(VV_UINT)
  {
    value.UI = val;
  }

  vvParam(const long& val) : type(VV_LONG)
  {
    value.L = val;
  }

  vvParam(const unsigned long& val) : type(VV_ULONG)
  {
    value.UL = val;
  }

#ifdef _WIN32
  vvParam(const long long& val) : type(VV_LLONG)
  {
    value.LL = val;
  }

  vvParam(const unsigned long long& val) : type(VV_ULLONG)
  {
    value.ULL = val;
  }
#endif

  vvParam(const float& val) : type(VV_FLOAT)
  {
    value.F = val;
  }

  vvParam(const vvVector2& val) : type(VV_VEC2)
  {
    value.Vec2 = &val;
  }

  vvParam(const vvVector2i& val) : type(VV_VEC2I)
  {
    value.Vec2I = &val;
  }

  vvParam(const vvVector3& val) : type(VV_VEC3)
  {
    value.Vec3 = &val;
  }

  vvParam(const vvVector3i& val) : type(VV_VEC3I)
  {
    value.Vec3I = &val;
  }

  vvParam(const vvVector4& val) : type(VV_VEC4)
  {
    value.Vec4 = &val;
  }

  vvParam(const vvsize3& val) : type(VV_SIZE3)
  {
    value.Size3 = &val;
  }

  vvParam(const vvColor& val) : type(VV_COLOR)
  {
    value.Color = &val;
  }
  
  vvParam(const vvAABB& val) : type(VV_AABB)
  {
    value.AABB = &val;
  }

  vvParam(const vvAABBi& val) : type(VV_AABBI)
  {
    value.AABBI = &val;
  }

  vvParam(const vvAABBs& val) : type(VV_AABBS)
  {
    value.AABBS = &val;
  }

  bool asBool() const
  {
    assert( type == VV_BOOL );
    return value.B;
  }

  char asChar() const
  {
    assert( type == VV_CHAR );
    return value.C;
  }

  unsigned char asUchar() const
  {
    assert( type == VV_UCHAR );
    return value.UC;
  }

  short asShort() const
  {
    assert( type == VV_SHORT );
    return value.S;
  }

  unsigned short asUshort() const
  {
    assert( type == VV_USHORT );
    return value.US;
  }

  int asInt() const
  {
    assert( type == VV_INT );
    return value.I;
  }

  unsigned int asUint() const
  {
    assert( type == VV_UINT );
    return value.UI;
  }

  long asLong() const
  {
    assert( type == VV_LONG );
    return value.L;
  }

  unsigned long asUlong() const
  {
    assert( type == VV_ULONG );
    return value.UL;
  }

#ifdef _WIN32
  long long asLlong() const
  {
    assert( type == VV_LLONG );
    return value.LL;
  }

  unsigned long long asUllong() const
  {
    assert( type == VV_ULLONG );
    return value.ULL;
  }
#endif

  float asFloat() const
  {
    assert( type == VV_FLOAT );
    return value.F;
  }

  const vvVector2& asVec2() const
  {
    assert( type == VV_VEC2 );
    return *value.Vec2;
  }

  const vvVector2i& asVec2i() const
  {
    assert( type == VV_VEC2I);
    return *value.Vec2I;
  }

  const vvVector3& asVec3() const
  {
    assert( type == VV_VEC3 );
    return *value.Vec3;
  }

  const vvVector3i& asVec3i() const
  {
    assert( type == VV_VEC3I );
    return *value.Vec3I;
  }

  const vvVector4& asVec4() const
  {
    assert( type == VV_VEC4 );
    return *value.Vec4;
  }

  const vvsize3& asSize3() const
  {
    assert( type == VV_SIZE3 );
    return *value.Size3;
  }

  const vvColor& asColor() const
  {
    assert( type == VV_COLOR );
    return *value.Color;
  }
  
  const vvAABB& asAABB() const
  {
    assert( type == VV_AABB );
    return *value.AABB;
  }

  const vvAABBi& asAABBi() const
  {
    assert( type == VV_AABBI );
    return *value.AABBI;
  }

  const vvAABBs& asAABBs() const
  {
    assert( type == VV_AABBS );
    return *value.AABBS;
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
#ifdef _WIN32
  operator long long() const { return asLlong(); }
  operator unsigned long long() const { return asUllong(); }
#endif
  operator float() const { return asFloat(); }
  operator const vvVector2&() const { return asVec2(); }
  operator const vvVector2i&() const { return asVec2i(); }
  operator const vvVector3&() const { return asVec3(); }
  operator const vvVector3i&() const { return asVec3i(); }
  operator const vvVector4&() const { return asVec4(); }
  operator const vvsize3&() const { return asSize3(); }
  operator const vvColor&() const { return asColor(); }
  operator const vvAABB&() const { return asAABB(); }
  operator const vvAABBi&() const { return asAABBi(); }
  operator const vvAABBs&() const { return asAABBs(); }

  // Returns the type of this parameter
  Type getType() const { return type; }

  // Returns whether this parameter is of type t
  bool isa(Type t) const { return type == t; }
};


#endif

