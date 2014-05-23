#pragma once

#include "vec3.h"

namespace virvo
{
namespace simd
{

template < typename T >
class CACHE_ALIGN base_aabb
{
public:
  inline base_aabb(base_vec3< T > const& min, base_vec3< T > const& max)
    : m_min(min)
    , m_max(max)
  {
  }

  inline base_vec3< T > getMin() const
  {
    return m_min;
  }

  inline base_vec3< T > getMax() const
  {
    return m_max;
  }
private:
  base_vec3< T > m_min;
  base_vec3< T > m_max;
};

} // simd
} // virvo

