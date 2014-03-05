#pragma once

#include "vec3.h"

namespace virvo
{
namespace simd
{

class CACHE_ALIGN AABB
{
public:
  inline AABB(Vec3 const& min, Vec3 const& max)
    : m_min(min)
    , m_max(max)
  {
  }

  inline Vec3 getMin() const
  {
    return m_min;
  }

  inline Vec3 getMax() const
  {
    return m_max;
  }
private:
  Vec3 m_min;
  Vec3 m_max;
};

} // simd
} // virvo

