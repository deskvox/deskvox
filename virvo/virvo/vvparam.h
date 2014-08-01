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
    case VV_EMPTY:    /* DO NOTHING */                                        return;
    case VV_BOOL:     save_as< bool                                     >(a); return;
    case VV_CHAR:     save_as< char                                     >(a); return;
    case VV_UCHAR:    save_as< unsigned char                            >(a); return;
    case VV_SHORT:    save_as< short                                    >(a); return;
    case VV_USHORT:   save_as< unsigned short                           >(a); return;
    case VV_INT:      save_as< int                                      >(a); return;
    case VV_UINT:     save_as< unsigned                                 >(a); return;
    case VV_LONG:     save_as< long                                     >(a); return;
    case VV_ULONG:    save_as< unsigned long                            >(a); return;
#if VV_HAVE_LLONG
    case VV_LLONG:    save_as< long long                                >(a); return;
#else
    case VV_LLONG:    break;
#endif
#if VV_HAVE_ULLONG
    case VV_ULLONG:   save_as< unsigned long long                       >(a); return;
#else
    case VV_ULLONG:   break;
#endif
    case VV_FLOAT:    save_as< float                                    >(a); return;
    case VV_VEC2F:    save_as< virvo::vector< 2, float >                >(a); return;
    case VV_VEC2I:    save_as< virvo::vector< 2, int >                  >(a); return;
    case VV_VEC3F:    save_as< virvo::vector< 3, float >                >(a); return;
    case VV_VEC3D:    save_as< virvo::vector< 3, double >               >(a); return;
    case VV_VEC3S:    save_as< virvo::vector< 3, short >                >(a); return;
    case VV_VEC3US:   save_as< virvo::vector< 3, unsigned short >       >(a); return;
    case VV_VEC3I:    save_as< virvo::vector< 3, int >                  >(a); return;
    case VV_VEC3UI:   save_as< virvo::vector< 3, unsigned int >         >(a); return;
    case VV_VEC3L:    save_as< virvo::vector< 3, long >                 >(a); return;
    case VV_VEC3UL:   save_as< virvo::vector< 3, unsigned long >        >(a); return;
    case VV_VEC3LL:   save_as< virvo::vector< 3, long long >            >(a); return;
    case VV_VEC3ULL:  save_as< virvo::vector< 3, unsigned long long >   >(a); return;
    case VV_VEC4F:    save_as< virvo::vector< 4, float >                >(a); return;
    case VV_COLOR:    save_as< vvColor                                  >(a); return;
    case VV_AABBF:    save_as< virvo::base_aabb< float >                >(a); return;
    case VV_AABBD:    save_as< virvo::base_aabb< double >               >(a); return;
    case VV_AABBI:    save_as< virvo::base_aabb< int >                  >(a); return;
    case VV_AABBUI:   save_as< virvo::base_aabb< unsigned int >         >(a); return;
    case VV_AABBL:    save_as< virvo::base_aabb< long >                 >(a); return;
    case VV_AABBUL:   save_as< virvo::base_aabb< unsigned long >        >(a); return;
    case VV_AABBLL:   save_as< virvo::base_aabb< long long >            >(a); return;
    case VV_AABBULL:  save_as< virvo::base_aabb< unsigned long long >   >(a); return;
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
    case VV_EMPTY:    value = boost::any();                                   return;
    case VV_BOOL:     load_as< bool                                     >(a); return;
    case VV_CHAR:     load_as< char                                     >(a); return;
    case VV_UCHAR:    load_as< unsigned char                            >(a); return;
    case VV_SHORT:    load_as< short                                    >(a); return;
    case VV_USHORT:   load_as< unsigned short                           >(a); return;
    case VV_INT:      load_as< int                                      >(a); return;
    case VV_UINT:     load_as< unsigned                                 >(a); return;
    case VV_LONG:     load_as< long                                     >(a); return;
    case VV_ULONG:    load_as< unsigned long                            >(a); return;
#if VV_HAVE_LLONG
    case VV_LLONG:    load_as< long long                                >(a); return;
#else
    case VV_LLONG:    break;
#endif
#if VV_HAVE_ULLONG
    case VV_ULLONG:   load_as< unsigned long long                       >(a); return;
#else
    case VV_ULLONG:   break;
#endif
    case VV_FLOAT:    load_as< float                                    >(a); return;
    case VV_VEC2F:    load_as< virvo::vector< 2, float >                >(a); return;
    case VV_VEC2I:    load_as< virvo::vector< 2, int >                  >(a); return;
    case VV_VEC3F:    load_as< virvo::vector< 3, float >                >(a); return;
    case VV_VEC3D:    load_as< virvo::vector< 3, double >               >(a); return;
    case VV_VEC3S:    load_as< virvo::vector< 3, short >                >(a); return;
    case VV_VEC3US:   load_as< virvo::vector< 3, unsigned short >       >(a); return;
    case VV_VEC3I:    load_as< virvo::vector< 3, int >                  >(a); return;
    case VV_VEC3UI:   load_as< virvo::vector< 3, unsigned int >         >(a); return;
    case VV_VEC3L:    load_as< virvo::vector< 3, long  >                >(a); return;
    case VV_VEC3UL:   load_as< virvo::vector< 3, unsigned long >        >(a); return;
    case VV_VEC3LL:   load_as< virvo::vector< 3, long long >            >(a); return;
    case VV_VEC3ULL:  load_as< virvo::vector< 3, unsigned long long >   >(a); return;
    case VV_VEC4F:    load_as< virvo::vector< 3, float >                >(a); return;
    case VV_COLOR:    load_as< vvColor                                  >(a); return;
    case VV_AABBF:    load_as< virvo::base_aabb< float >                >(a); return;
    case VV_AABBD:    load_as< virvo::base_aabb< double >               >(a); return;
    case VV_AABBI:    load_as< virvo::base_aabb< int >                  >(a); return;
    case VV_AABBUI:   load_as< virvo::base_aabb< unsigned int >         >(a); return;
    case VV_AABBL:    load_as< virvo::base_aabb< long >                 >(a); return;
    case VV_AABBUL:   load_as< virvo::base_aabb< unsigned long >        >(a); return;
    case VV_AABBLL:   load_as< virvo::base_aabb< long long >            >(a); return;
    case VV_AABBULL:  load_as< virvo::base_aabb< unsigned long long >   >(a); return;
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

  vvParam(virvo::vector< 2, float > const& val)
    : type(VV_VEC2F)
    , value(val)
  {
  }

  vvParam(virvo::vector< 2, int > const& val)
    : type(VV_VEC2I)
    , value(val)
  {
  }

  vvParam(virvo::vector< 3, float > const& val)
    : type(VV_VEC3F)
    , value(val)
  {
  }

  vvParam(virvo::vector< 3, double > const& val)
    : type(VV_VEC3D)
    , value(val)
  {
  }

  vvParam(virvo::vector< 3, short > const& val)
    : type(VV_VEC3S)
    , value(val)
  {
  }

  vvParam(virvo::vector< 3, unsigned short > const& val)
    : type(VV_VEC3US)
    , value(val)
  {
  }

  vvParam(virvo::vector< 3, int > const& val)
    : type(VV_VEC3I)
    , value(val)
  {
  }

  vvParam(virvo::vector< 3, unsigned int > const& val)
    : type(VV_VEC3UI)
    , value(val)
  {
  }

  vvParam(virvo::vector< 3, long > const& val)
    : type(VV_VEC3L)
    , value(val)
  {
  }

  vvParam(virvo::vector< 3, unsigned long > const& val)
    : type(VV_VEC3UL)
    , value(val)
  {
  }

  vvParam(virvo::vector< 3, long long > const& val)
    : type(VV_VEC3LL)
    , value(val)
  {
  }

  vvParam(virvo::vector< 3, unsigned long long > const& val)
    : type(VV_VEC3ULL)
    , value(val)
  {
  }

  vvParam(virvo::vector< 4, float > const& val)
    : type(VV_VEC4F)
    , value(val)
  {
  }

  vvParam(const vvColor& val)
    : type(VV_COLOR)
    , value(val)
  {
  }

  vvParam(virvo::base_aabb< float > const& val)
    : type(VV_AABBF)
    , value(val)
  {
  }

  vvParam(virvo::base_aabb< double > const& val)
    : type(VV_AABBD)
    , value(val)
  {
  }

  vvParam(virvo::base_aabb< int > const& val)
    : type(VV_AABBI)
    , value(val)
  {
  }

  vvParam(virvo::base_aabb< unsigned int > const& val)
    : type(VV_AABBUI)
    , value(val)
  {
  }

  vvParam(virvo::base_aabb< long > const& val)
    : type(VV_AABBL)
    , value(val)
  {
  }

  vvParam(virvo::base_aabb< unsigned long > const& val)
    : type(VV_AABBUL)
    , value(val)
  {
  }

  vvParam(virvo::base_aabb< long long > const& val)
    : type(VV_AABBLL)
    , value(val)
  {
  }

  vvParam(virvo::base_aabb< unsigned long long > const& val)
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

  virvo::vector< 2, float > asVec2f() const {
    return boost::any_cast< virvo::vector< 2, float > >(value);
  }

  virvo::vector< 2, int > asVec2i() const {
    return boost::any_cast< virvo::vector< 2, int > >(value);
  }

  virvo::vector< 3, float > asVec3f() const {
    return boost::any_cast< virvo::vector< 3, float > >(value);
  }

  virvo::vector< 3, double > asVec3d() const {
    return boost::any_cast< virvo::vector< 3, double > >(value);
  }

  virvo::vector< 3, short > asVec3s() const {
    return boost::any_cast< virvo::vector< 3, short > >(value);
  }

  virvo::vector< 3, unsigned short > asVec3us() const {
    return boost::any_cast< virvo::vector< 3, unsigned short > >(value);
  }

  virvo::vector< 3, int > asVec3i() const {
    return boost::any_cast< virvo::vector< 3, int > >(value);
  }

  virvo::vector< 3, unsigned int > asVec3ui() const {
    return boost::any_cast< virvo::vector< 3, unsigned int > >(value);
  }

  virvo::vector< 3, long > asVec3l() const {
    return boost::any_cast< virvo::vector< 3, long > >(value);
  }

  virvo::vector< 3, unsigned long > asVec3ul() const {
    return boost::any_cast< virvo::vector< 3, unsigned long > >(value);
  }

  virvo::vector< 3, long long > asVec3ll() const {
    return boost::any_cast< virvo::vector< 3, long long > >(value);
  }

  virvo::vector< 3, unsigned long long > asVec3ull() const {
    return boost::any_cast< virvo::vector< 3, unsigned long long > >(value);
  }

  virvo::vector< 4, float > asVec4f() const {
    return boost::any_cast< virvo::vector< 4, float > >(value);
  }

  vvColor asColor() const {
    return boost::any_cast<vvColor>(value);
  }

  virvo::base_aabb< float > asAABBf() const {
    return boost::any_cast< virvo::base_aabb< float > >(value);
  }

  virvo::base_aabb< double > asAABBd() const {
    return boost::any_cast< virvo::base_aabb< double > >(value);
  }

  virvo::base_aabb< int > asAABBi() const {
    return boost::any_cast< virvo::base_aabb< int > >(value);
  }

  virvo::base_aabb< unsigned int > asAABBui() const {
    return boost::any_cast< virvo::base_aabb< unsigned int > >(value);
  }

  virvo::base_aabb< long > asAABBl() const {
    return boost::any_cast< virvo::base_aabb< long > >(value);
  }

  virvo::base_aabb< unsigned long > asAABBul() const {
    return boost::any_cast< virvo::base_aabb< unsigned long > >(value);
  }

  virvo::base_aabb< long long > asAABBll() const {
    return boost::any_cast< virvo::base_aabb< long long > >(value);
  }

  virvo::base_aabb< unsigned long long >  asAABBull() const {
    return boost::any_cast< virvo::base_aabb< unsigned long long > >(value);
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

  operator virvo::vector< 2, float >() const {
    return asVec2f();
  }

  operator virvo::vector< 2, int >() const {
    return asVec2i();
  }

  operator virvo::vector< 3, float >() const {
    return asVec3f();
  }

  operator virvo::vector< 3, double >() const {
    return asVec3d();
  }

  operator virvo::vector< 3, short >() const {
    return asVec3s();
  }

  operator virvo::vector< 3, unsigned short >() const {
    return asVec3us();
  }

  operator virvo::vector< 3, int >() const {
    return asVec3i();
  }

  operator virvo::vector< 3, unsigned int >() const {
    return asVec3ui();
  }

  operator virvo::vector< 3, long >() const {
    return asVec3l();
  }

  operator virvo::vector< 3, unsigned long >() const {
    return asVec3ul();
  }

  operator virvo::vector< 3, long long >() const {
    return asVec3ll();
  }

  operator virvo::vector< 3, unsigned long long >() const {
    return asVec3ull();
  }

  operator virvo::vector< 4, float >() const {
    return asVec4f();
  }

  operator vvColor() const {
    return asColor();
  }

  operator virvo::base_aabb< float >() const {
    return asAABBf();
  }

  operator virvo::base_aabb< double >() const {
    return asAABBd();
  }

  operator virvo::base_aabb< int >() const {
    return asAABBi();
  }

  operator virvo::base_aabb< unsigned int >() const {
    return asAABBui();
  }

  operator virvo::base_aabb< long >() const {
    return asAABBl();
  }

  operator virvo::base_aabb< unsigned long >() const {
    return asAABBul();
  }

  operator virvo::base_aabb< long long >() const {
    return asAABBll();
  }

  operator virvo::base_aabb< unsigned long long >() const {
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
