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
#include <stdexcept>

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
    VV_VEC2F,
    VV_VEC2I,
    VV_VEC3F,
    VV_VEC3D,
    VV_VEC3S,
    VV_VEC3US,
    VV_VEC3I,
    VV_VEC3UI,
    VV_VEC3L,
    VV_VEC3UL,
    VV_VEC3LL,
    VV_VEC3ULL,
    VV_VEC4F,
    VV_COLOR,
    VV_AABBF,
    VV_AABBD,
    VV_AABBI,
    VV_AABBUI,
    VV_AABBL,
    VV_AABBUL,
    VV_AABBLL,
    VV_AABBULL
  };

private:
  // The type of this parameter
  Type type;
  // The value of this parameter
  boost::any value;

public:
  template<class A>
  void save(A& a, unsigned /*version*/)
  {
    a & static_cast<unsigned>(type);
    switch (type)
    {
    case VV_EMPTY:    /* DO NOTHING */                                  return;
    case VV_BOOL:     a & boost::any_cast< bool               >(value); return;
    case VV_CHAR:     a & boost::any_cast< char               >(value); return;
    case VV_UCHAR:    a & boost::any_cast< unsigned char      >(value); return;
    case VV_SHORT:    a & boost::any_cast< short              >(value); return;
    case VV_USHORT:   a & boost::any_cast< unsigned short     >(value); return;
    case VV_INT:      a & boost::any_cast< int                >(value); return;
    case VV_UINT:     a & boost::any_cast< unsigned           >(value); return;
    case VV_LONG:     a & boost::any_cast< long               >(value); return;
    case VV_ULONG:    a & boost::any_cast< unsigned long      >(value); return;
#if VV_HAVE_LLONG
    case VV_LLONG:    a & boost::any_cast< long long          >(value); return;
#else
    case VV_LLONG:    break;
#endif
#if VV_HAVE_ULLONG
    case VV_ULLONG:   a & boost::any_cast< unsigned long long >(value); return;
#else
    case VV_ULLONG:   break;
#endif
    case VV_FLOAT:    a & boost::any_cast< float              >(value); return;
    case VV_VEC2F:    a & boost::any_cast< virvo::Vec2f       >(value); return;
    case VV_VEC2I:    a & boost::any_cast< virvo::Vec2i       >(value); return;
    case VV_VEC3F:    a & boost::any_cast< virvo::Vec3f       >(value); return;
    case VV_VEC3D:    a & boost::any_cast< virvo::Vec3d       >(value); return;
    case VV_VEC3S:    a & boost::any_cast< virvo::Vec3s       >(value); return;
    case VV_VEC3US:   a & boost::any_cast< virvo::Vec3us      >(value); return;
    case VV_VEC3I:    a & boost::any_cast< virvo::Vec3i       >(value); return;
    case VV_VEC3UI:   a & boost::any_cast< virvo::Vec3ui      >(value); return;
    case VV_VEC3L:    a & boost::any_cast< virvo::Vec3l       >(value); return;
    case VV_VEC3UL:   a & boost::any_cast< virvo::Vec3ul      >(value); return;
    case VV_VEC3LL:   a & boost::any_cast< virvo::Vec3ll      >(value); return;
    case VV_VEC3ULL:  a & boost::any_cast< virvo::Vec3ull     >(value); return;
    case VV_VEC4F:    a & boost::any_cast< virvo::Vec4f       >(value); return;
    case VV_COLOR:    a & boost::any_cast< vvColor            >(value); return;
    case VV_AABBF:    a & boost::any_cast< virvo::AABBf       >(value); return;
    case VV_AABBD:    a & boost::any_cast< virvo::AABBd       >(value); return;
    case VV_AABBI:    a & boost::any_cast< virvo::AABBi       >(value); return;
    case VV_AABBUI:   a & boost::any_cast< virvo::AABBui      >(value); return;
    case VV_AABBL:    a & boost::any_cast< virvo::AABBl       >(value); return;
    case VV_AABBUL:   a & boost::any_cast< virvo::AABBul      >(value); return;
    case VV_AABBLL:   a & boost::any_cast< virvo::AABBll      >(value); return;
    case VV_AABBULL:  a & boost::any_cast< virvo::AABBull     >(value); return;
    //
    // NOTE:
    //
    // No default case here: Let the compiler emit a warning if a type is
    // missing in this list!!!
    //
    }

    throw std::runtime_error("unable to serialize parameter");
  }

private:
  template<class T, class A> static T load_value(A& a) {
    T x; a & x; return x;
  }

public:
  template<class A>
  void load(A& a, unsigned /*version*/)
  {
    a & static_cast<unsigned>(type);
    switch (type)
    {
    case VV_EMPTY:    value = boost::any();                        return;
    case VV_BOOL:     value = load_value< bool               >(a); return;
    case VV_CHAR:     value = load_value< char               >(a); return;
    case VV_UCHAR:    value = load_value< unsigned char      >(a); return;
    case VV_SHORT:    value = load_value< short              >(a); return;
    case VV_USHORT:   value = load_value< unsigned short     >(a); return;
    case VV_INT:      value = load_value< int                >(a); return;
    case VV_UINT:     value = load_value< unsigned           >(a); return;
    case VV_LONG:     value = load_value< long               >(a); return;
    case VV_ULONG:    value = load_value< unsigned long      >(a); return;
#if VV_HAVE_LLONG
    case VV_LLONG:    value = load_value< long long          >(a); return;
#else
    case VV_LLONG:    break;
#endif
#if VV_HAVE_ULLONG
    case VV_ULLONG:   value = load_value< unsigned long long >(a); return;
#else
    case VV_ULLONG:   break;
#endif
    case VV_FLOAT:    value = load_value< float              >(a); return;
    case VV_VEC2F:    value = load_value< virvo::Vec2f       >(a); return;
    case VV_VEC2I:    value = load_value< virvo::Vec2i       >(a); return;
    case VV_VEC3F:    value = load_value< virvo::Vec3f       >(a); return;
    case VV_VEC3D:    value = load_value< virvo::Vec3d       >(a); return;
    case VV_VEC3S:    value = load_value< virvo::Vec3s       >(a); return;
    case VV_VEC3US:   value = load_value< virvo::Vec3us      >(a); return;
    case VV_VEC3I:    value = load_value< virvo::Vec3i       >(a); return;
    case VV_VEC3UI:   value = load_value< virvo::Vec3ui      >(a); return;
    case VV_VEC3L:    value = load_value< virvo::Vec3l       >(a); return;
    case VV_VEC3UL:   value = load_value< virvo::Vec3ul      >(a); return;
    case VV_VEC3LL:   value = load_value< virvo::Vec3ll      >(a); return;
    case VV_VEC3ULL:  value = load_value< virvo::Vec3ull     >(a); return;
    case VV_VEC4F:    value = load_value< virvo::Vec4f       >(a); return;
    case VV_COLOR:    value = load_value< vvColor            >(a); return;
    case VV_AABBF:    value = load_value< virvo::AABBf       >(a); return;
    case VV_AABBD:    value = load_value< virvo::AABBd       >(a); return;
    case VV_AABBI:    value = load_value< virvo::AABBi       >(a); return;
    case VV_AABBUI:   value = load_value< virvo::AABBui      >(a); return;
    case VV_AABBL:    value = load_value< virvo::AABBl       >(a); return;
    case VV_AABBUL:   value = load_value< virvo::AABBul      >(a); return;
    case VV_AABBLL:   value = load_value< virvo::AABBll      >(a); return;
    case VV_AABBULL:  value = load_value< virvo::AABBull     >(a); return;
    //
    // NOTE:
    //
    // No default case here: Let the compiler emit a warning if a type is
    // missing in this list!!!
    //
    }

    throw std::runtime_error("unable to deserialize parameter");
  }

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

  vvParam(const virvo::Vec2f& val)
    : type(VV_VEC2F)
    , value(val)
  {
  }

  vvParam(const virvo::Vec2i& val)
    : type(VV_VEC2I)
    , value(val)
  {
  }

  vvParam(const virvo::Vec3f& val)
    : type(VV_VEC3F)
    , value(val)
  {
  }

  vvParam(const virvo::Vec3d& val)
    : type(VV_VEC3D)
    , value(val)
  {
  }

  vvParam(const virvo::Vec3s& val)
    : type(VV_VEC3S)
    , value(val)
  {
  }

  vvParam(const virvo::Vec3us& val)
    : type(VV_VEC3US)
    , value(val)
  {
  }

  vvParam(const virvo::Vec3i& val)
    : type(VV_VEC3I)
    , value(val)
  {
  }

  vvParam(const virvo::Vec3ui& val)
    : type(VV_VEC3UI)
    , value(val)
  {
  }

  vvParam(const virvo::Vec3l& val)
    : type(VV_VEC3L)
    , value(val)
  {
  }

  vvParam(const virvo::Vec3ul& val)
    : type(VV_VEC3UL)
    , value(val)
  {
  }

  vvParam(const virvo::Vec3ll& val)
    : type(VV_VEC3LL)
    , value(val)
  {
  }

  vvParam(const virvo::Vec3ull& val)
    : type(VV_VEC3ULL)
    , value(val)
  {
  }

  vvParam(const virvo::Vec4f& val)
    : type(VV_VEC4F)
    , value(val)
  {
  }

  vvParam(const vvColor& val)
    : type(VV_COLOR)
    , value(val)
  {
  }

  vvParam(const virvo::AABBf& val)
    : type(VV_AABBF)
    , value(val)
  {
  }

  vvParam(const virvo::AABBd& val)
    : type(VV_AABBD)
    , value(val)
  {
  }

  vvParam(const virvo::AABBi& val)
    : type(VV_AABBI)
    , value(val)
  {
  }

  vvParam(const virvo::AABBui& val)
    : type(VV_AABBUI)
    , value(val)
  {
  }

  vvParam(const virvo::AABBl& val)
    : type(VV_AABBL)
    , value(val)
  {
  }

  vvParam(const virvo::AABBul& val)
    : type(VV_AABBUL)
    , value(val)
  {
  }

  vvParam(const virvo::AABBll& val)
    : type(VV_AABBLL)
    , value(val)
  {
  }

  vvParam(const virvo::AABBull& val)
    : type(VV_AABBULL)
    , value(val)
  {
  }

  bool asBool() const {
    return boost::any_cast<bool>(value);
  }

  char asChar() const {
    return boost::any_cast<char>(value);
  }

  unsigned char asUchar() const {
    return boost::any_cast<unsigned char>(value);
  }

  short asShort() const {
    return boost::any_cast<short>(value);
  }

  unsigned short asUshort() const {
    return boost::any_cast<unsigned short>(value);
  }

  int asInt() const {
    return boost::any_cast<int>(value);
  }

  unsigned int asUint() const {
    return boost::any_cast<unsigned int>(value);
  }

  long asLong() const {
    return boost::any_cast<long>(value);
  }

  unsigned long asUlong() const {
    return boost::any_cast<unsigned long>(value);
  }

#if VV_HAVE_LLONG
  long long asLlong() const {
    return boost::any_cast<long long>(value);
  }
#endif

#if VV_HAVE_ULLONG
  unsigned long long asUllong() const {
    return boost::any_cast<unsigned long long>(value);
  }
#endif

  float asFloat() const {
    return boost::any_cast<float>(value);
  }

  virvo::Vec2f asVec2f() const {
    return boost::any_cast<virvo::Vec2f>(value);
  }

  virvo::Vec2i asVec2i() const {
    return boost::any_cast<virvo::Vec2i>(value);
  }

  virvo::Vec3f asVec3f() const {
    return boost::any_cast<virvo::Vec3f>(value);
  }

  virvo::Vec3d asVec3d() const {
    return boost::any_cast<virvo::Vec3d>(value);
  }

  virvo::Vec3s asVec3s() const {
    return boost::any_cast<virvo::Vec3s>(value);
  }

  virvo::Vec3us asVec3us() const {
    return boost::any_cast<virvo::Vec3us>(value);
  }

  virvo::Vec3i asVec3i() const {
    return boost::any_cast<virvo::Vec3i>(value);
  }

  virvo::Vec3ui asVec3ui() const {
    return boost::any_cast<virvo::Vec3ui>(value);
  }

  virvo::Vec3l asVec3l() const {
    return boost::any_cast<virvo::Vec3l>(value);
  }

  virvo::Vec3ul asVec3ul() const {
    return boost::any_cast<virvo::Vec3ul>(value);
  }

  virvo::Vec3ll asVec3ll() const {
    return boost::any_cast<virvo::Vec3ll>(value);
  }

  virvo::Vec3ull asVec3ull() const {
    return boost::any_cast<virvo::Vec3ull>(value);
  }

  virvo::Vec4f asVec4f() const {
    return boost::any_cast<virvo::Vec4f>(value);
  }

  vvColor asColor() const {
    return boost::any_cast<vvColor>(value);
  }

  virvo::AABBf asAABBf() const {
    return boost::any_cast<virvo::AABBf>(value);
  }

  virvo::AABBd asAABBd() const {
    return boost::any_cast<virvo::AABBd>(value);
  }

  virvo::AABBi asAABBi() const {
    return boost::any_cast<virvo::AABBi>(value);
  }

  virvo::AABBui asAABBui() const {
    return boost::any_cast<virvo::AABBui>(value);
  }

  virvo::AABBl asAABBl() const {
    return boost::any_cast<virvo::AABBl>(value);
  }

  virvo::AABBul asAABBul() const {
    return boost::any_cast<virvo::AABBul>(value);
  }

  virvo::AABBll asAABBll() const {
    return boost::any_cast<virvo::AABBll>(value);
  }

  virvo::AABBull asAABBull() const {
    return boost::any_cast<virvo::AABBull>(value);
  }

  operator bool() const {
    return asBool();
  }

  operator char() const {
    return asChar();
  }

  operator unsigned char() const {
    return asUchar();
  }

  operator short() const {
    return asShort();
  }

  operator unsigned short() const {
    return asUshort();
  }

  operator int() const {
    return asInt();
  }

  operator unsigned int() const {
    return asUint();
  }

  operator long() const {
    return asLong();
  }

  operator unsigned long() const {
    return asUlong();
  }

#if VV_HAVE_LLONG
  operator long long() const {
    return asLlong();
  }
#endif

#if VV_HAVE_ULLONG
  operator unsigned long long() const {
    return asUllong();
  }
#endif

  operator float() const {
    return asFloat();
  }

  operator virvo::Vec2f() const {
    return asVec2f();
  }

  operator virvo::Vec2i() const {
    return asVec2i();
  }

  operator virvo::Vec3f() const {
    return asVec3f();
  }

  operator virvo::Vec3d() const {
    return asVec3d();
  }

  operator virvo::Vec3s() const {
    return asVec3s();
  }

  operator virvo::Vec3us() const {
    return asVec3us();
  }

  operator virvo::Vec3i() const {
    return asVec3i();
  }

  operator virvo::Vec3ui() const {
    return asVec3ui();
  }

  operator virvo::Vec3l() const {
    return asVec3l();
  }

  operator virvo::Vec3ul() const {
    return asVec3ul();
  }

  operator virvo::Vec3ll() const {
    return asVec3ll();
  }

  operator virvo::Vec3ull() const {
    return asVec3ull();
  }

  operator virvo::Vec4f() const {
    return asVec4f();
  }

  operator vvColor() const {
    return asColor();
  }

  operator virvo::AABBf() const {
    return asAABBf();
  }

  operator virvo::AABBd() const {
    return asAABBd();
  }

  operator virvo::AABBi() const {
    return asAABBi();
  }

  operator virvo::AABBui() const {
    return asAABBui();
  }

  operator virvo::AABBl() const {
    return asAABBl();
  }

  operator virvo::AABBul() const {
    return asAABBul();
  }

  operator virvo::AABBll() const {
    return asAABBll();
  }

  operator virvo::AABBull() const {
    return asAABBull();
  }

  // Returns the type of this parameter
  Type getType() const {
    return type;
  }

  // Returns whether this parameter is of type t
  bool isa(Type t) const {
    return type == t;
  }
};


#endif
