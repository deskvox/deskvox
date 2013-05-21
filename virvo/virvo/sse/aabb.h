#pragma once

#include "vec3.h"

namespace virvo
{
namespace sse
{

class CACHE_ALIGN AABB
{
public:
  inline AABB(sse::Vec3 const& min, sse::Vec3 const& max)
    : m_min(min)
    , m_max(max)
  {
  }

  inline sse::Vec3 getMin() const
  {
    return m_min;
  }

  inline sse::Vec3 getMax() const
  {
    return m_max;
  }
private:
  sse::Vec3 m_min;
  sse::Vec3 m_max;
};

}
}

