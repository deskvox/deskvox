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

#ifndef VV_PARAM_H
#define VV_PARAM_H

#ifdef HAVE_CONFIG_H
#include "vvconfig.h"
#endif

#include "math/math.h"

#include "vvinttypes.h"
#include "vvcolor.h"
#include "vvaabb.h"

#include <cassert>
#include <stdexcept>

#include <boost/any.hpp>
#include <boost/serialization/split_member.hpp>

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

private:
  template<class T, class A>
  static void save_value(A& a, T const& x)
  {
    a & x;
  }

  template<class T, class A>
  void save_as(A& a) const
  {
    T x = boost::any_cast<T>(value);
    a & x;
  }

  template<class T, class A>
  static T load_value(A& a)
  {
    T x; a & x; return x;
  }

  template<class T, class A>
  void load_as(A& a)
  {
    T x; a & x; value = x;
  }

public:
  //--- serialization ------------------------------------------------------------------------------

  BOOST_SERIALIZATION_SPLIT_MEMBER()

  template<class A>
  void save(A& a, unsigned /*version*/) const
  {
    save_value(a, static_cast<unsigned>(type));

    switch (type)
    {
    case VV_EMPTY:    /* DO NOTHING */                    return;
    case VV_BOOL:     save_as< bool                 >(a); return;
    case VV_CHAR:     save_as< char                 >(a); return;
    case VV_UCHAR:    save_as< unsigned char        >(a); return;
    case VV_SHORT:    save_as< short                >(a); return;
    case VV_USHORT:   save_as< unsigned short       >(a); return;
    case VV_INT:      save_as< int                  >(a); return;
    case VV_UINT:     save_as< unsigned             >(a); return;
    case VV_LONG:     save_as< long                 >(a); return;
    case VV_ULONG:    save_as< unsigned long        >(a); return;
#if VV_HAVE_LLONG
    case VV_LLONG:    save_as< long long            >(a); return;
#else
    case VV_LLONG:    break;
#endif
#if VV_HAVE_ULLONG
    case VV_ULLONG:   save_as< unsigned long long   >(a); return;
#else
    case VV_ULLONG:   break;
#endif
    case VV_FLOAT:    save_as< float                >(a); return;
    case VV_VEC2F:    save_as< virvo::vec2f         >(a); return;
    case VV_VEC2I:    save_as< virvo::vec2i         >(a); return;
    case VV_VEC3F:    save_as< virvo::vec3f         >(a); return;
    case VV_VEC3D:    save_as< virvo::vec3d         >(a); return;
    case VV_VEC3S:    save_as< virvo::Vec3s         >(a); return;
    case VV_VEC3US:   save_as< virvo::Vec3us        >(a); return;
    case VV_VEC3I:    save_as< virvo::vec3i         >(a); return;
    case VV_VEC3UI:   save_as< virvo::vec3ui        >(a); return;
    case VV_VEC3L:    save_as< virvo::Vec3l         >(a); return;
    case VV_VEC3UL:   save_as< virvo::Vec3ul        >(a); return;
    case VV_VEC3LL:   save_as< virvo::Vec3ll        >(a); return;
    case VV_VEC3ULL:  save_as< virvo::Vec3ull       >(a); return;
    case VV_VEC4F:    save_as< virvo::vec4f         >(a); return;
    case VV_COLOR:    save_as< vvColor              >(a); return;
    case VV_AABBF:    save_as< virvo::AABBf         >(a); return;
    case VV_AABBD:    save_as< virvo::AABBd         >(a); return;
    case VV_AABBI:    save_as< virvo::AABBi         >(a); return;
    case VV_AABBUI:   save_as< virvo::AABBui        >(a); return;
    case VV_AABBL:    save_as< virvo::AABBl         >(a); return;
    case VV_AABBUL:   save_as< virvo::AABBul        >(a); return;
    case VV_AABBLL:   save_as< virvo::AABBll        >(a); return;
    case VV_AABBULL:  save_as< virvo::AABBull       >(a); return;
    //
    // NOTE:
    //
    // No default case here: Let the compiler emit a warning if a type is
    // missing in this list!!!
    //
    }

    throw std::runtime_error("unable to serialize parameter");
  }

  template<class A>
  void load(A& a, unsigned /*version*/)
  {
    type = static_cast<Type>(load_value<unsigned>(a));

    switch (type)
    {
    case VV_EMPTY:    value = boost::any();               return;
    case VV_BOOL:     load_as< bool                 >(a); return;
    case VV_CHAR:     load_as< char                 >(a); return;
    case VV_UCHAR:    load_as< unsigned char        >(a); return;
    case VV_SHORT:    load_as< short                >(a); return;
    case VV_USHORT:   load_as< unsigned short       >(a); return;
    case VV_INT:      load_as< int                  >(a); return;
    case VV_UINT:     load_as< unsigned             >(a); return;
    case VV_LONG:     load_as< long                 >(a); return;
    case VV_ULONG:    load_as< unsigned long        >(a); return;
#if VV_HAVE_LLONG
    case VV_LLONG:    load_as< long long            >(a); return;
#else
    case VV_LLONG:    break;
#endif
#if VV_HAVE_ULLONG
    case VV_ULLONG:   load_as< unsigned long long   >(a); return;
#else
    case VV_ULLONG:   break;
#endif
    case VV_FLOAT:    load_as< float                >(a); return;
    case VV_VEC2F:    load_as< virvo::vec2f         >(a); return;
    case VV_VEC2I:    load_as< virvo::vec2i         >(a); return;
    case VV_VEC3F:    load_as< virvo::vec3f         >(a); return;
    case VV_VEC3D:    load_as< virvo::vec3d         >(a); return;
    case VV_VEC3S:    load_as< virvo::Vec3s         >(a); return;
    case VV_VEC3US:   load_as< virvo::Vec3us        >(a); return;
    case VV_VEC3I:    load_as< virvo::vec3i         >(a); return;
    case VV_VEC3UI:   load_as< virvo::vec3ui        >(a); return;
    case VV_VEC3L:    load_as< virvo::Vec3l         >(a); return;
    case VV_VEC3UL:   load_as< virvo::Vec3ul        >(a); return;
    case VV_VEC3LL:   load_as< virvo::Vec3ll        >(a); return;
    case VV_VEC3ULL:  load_as< virvo::Vec3ull       >(a); return;
    case VV_VEC4F:    load_as< virvo::vec4f         >(a); return;
    case VV_COLOR:    load_as< vvColor              >(a); return;
    case VV_AABBF:    load_as< virvo::AABBf         >(a); return;
    case VV_AABBD:    load_as< virvo::AABBd         >(a); return;
    case VV_AABBI:    load_as< virvo::AABBi         >(a); return;
    case VV_AABBUI:   load_as< virvo::AABBui        >(a); return;
    case VV_AABBL:    load_as< virvo::AABBl         >(a); return;
    case VV_AABBUL:   load_as< virvo::AABBul        >(a); return;
    case VV_AABBLL:   load_as< virvo::AABBll        >(a); return;
    case VV_AABBULL:  load_as< virvo::AABBull       >(a); return;
    //
    // NOTE:
    //
    // No default case here: Let the compiler emit a warning if a type is
    // missing in this list!!!
    //
    }

    throw std::runtime_error("unable to deserialize parameter");
  }

  //------------------------------------------------------------------------------------------------

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

  vvParam(const virvo::vec2f& val)
    : type(VV_VEC2F)
    , value(val)
  {
  }

  vvParam(const virvo::vec2i& val)
    : type(VV_VEC2I)
    , value(val)
  {
  }

  vvParam(virvo::vec3f const& val)
    : type(VV_VEC3F)
    , value(val)
  {
  }

  vvParam(virvo::vec3d const& val)
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

  vvParam(const virvo::vec3i& val)
    : type(VV_VEC3I)
    , value(val)
  {
  }

  vvParam(const virvo::vec3ui& val)
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

  vvParam(const virvo::vec4f& val)
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

  virvo::vec2f asVec2f() const {
    return boost::any_cast<virvo::vec2f>(value);
  }

  virvo::vec2i asVec2i() const {
    return boost::any_cast<virvo::vec2i>(value);
  }

  virvo::vec3f asVec3f() const {
    return boost::any_cast<virvo::vec3f>(value);
  }

  virvo::vec3d asVec3d() const {
    return boost::any_cast<virvo::vec3d>(value);
  }

  virvo::Vec3s asVec3s() const {
    return boost::any_cast<virvo::Vec3s>(value);
  }

  virvo::Vec3us asVec3us() const {
    return boost::any_cast<virvo::Vec3us>(value);
  }

  virvo::vec3i asVec3i() const {
    return boost::any_cast<virvo::vec3i>(value);
  }

  virvo::vec3ui asVec3ui() const {
    return boost::any_cast<virvo::vec3ui>(value);
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

  virvo::vec4f asVec4f() const {
    return boost::any_cast<virvo::vec4f>(value);
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

  operator virvo::vec2f() const {
    return asVec2f();
  }

  operator virvo::vec2i() const {
    return asVec2i();
  }

  operator virvo::vec3f() const {
    return asVec3f();
  }

  operator virvo::vec3d() const {
    return asVec3d();
  }

  operator virvo::Vec3s() const {
    return asVec3s();
  }

  operator virvo::Vec3us() const {
    return asVec3us();
  }

  operator virvo::vec3i() const {
    return asVec3i();
  }

  operator virvo::vec3ui() const {
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

  operator virvo::vec4f() const {
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
