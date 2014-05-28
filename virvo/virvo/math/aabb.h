#pragma once

#include "vector.h"

namespace virvo
{


namespace math
{


template < typename T >
class base_aabb
{
public:

  inline base_aabb(vector< 3, T > const& min, vector< 3, T > const& max)
    : m_min(min)
    , m_max(max)
  {
  }

  inline vector< 3, T > getMin() const
  {
    return m_min;
  }

  inline vector< 3, T > getMax() const
  {
    return m_max;
  }

private:

  vector< 3, T > m_min;
  vector< 3, T > m_max;

};

} // math


} // virvo



