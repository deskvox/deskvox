// vvparam.h


#ifndef VVPARAM_H_INCLUDED
#define VVPARAM_H_INCLUDED


#include "vvvecmath.h"
#include "vvcolor.h"

#include <cassert>


class vvParam
{
public:
  enum Type {
    VV_EMPTY,
    VV_BOOL,
    VV_INT,
    VV_FLOAT,
    VV_VEC3,
    VV_VEC3I,
    VV_VEC4,
    VV_COLOR
  };

private:
  union Value {
    bool B;
    int I;
    float F;
    const vvVector3* Vec3;
    const vvVector3i* Vec3I;
    const vvVector4* Vec4;
    const vvColor* Color;
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

  vvParam(const int& val) : type(VV_INT)
  {
    value.I = val;
  }

  vvParam(const float& val) : type(VV_FLOAT)
  {
    value.F = val;
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

  vvParam(const vvColor& val) : type(VV_COLOR)
  {
    value.Color = &val;
  }

  bool asBool() const
  {
    assert( type == VV_BOOL );
    return value.B;
  }

  int asInt() const
  {
    assert( type == VV_INT );
    return value.I;
  }

  float asFloat() const
  {
    assert( type == VV_FLOAT );
    return value.F;
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

  const vvColor& asColor() const
  {
    assert( type == VV_COLOR );
    return *value.Color;
  }

  operator bool() const { return asBool(); }
  operator int() const { return asInt(); }
  operator float() const { return asFloat(); }
  operator const vvVector3&() const { return asVec3(); }
  operator const vvVector3i&() const { return asVec3i(); }
  operator const vvVector4&() const { return asVec4(); }
  operator const vvColor&() const { return asColor(); }

  // Returns the type of this parameter
  Type getType() const { return type; }

  // Returns whether this parameter is of type t
  bool isa(Type t) const { return type == t; }
};


#endif

