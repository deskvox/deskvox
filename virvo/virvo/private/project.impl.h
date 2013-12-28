#pragma once


#include <limits>


namespace virvo
{


void project(Vec3* win, Vec3 const& obj, Matrix const& modelview, Matrix const& projection, Viewport const& viewport)
{

  Vec4 u(obj, 1.0);
  Vec4 v = projection * modelview * u;


  v[0] /= v[3];
  v[1] /= v[3];
  v[2] /= v[3];


  (*win)[0] = viewport[0] + viewport[2] * (v[0] + 1) / 2;
  (*win)[1] = viewport[1] + viewport[3] * (v[1] + 1) / 2;
  (*win)[2] = (v[2] + 1) / 2;

}


void unproject(Vec3* obj, Vec3 const& win, Matrix const& modelview, Matrix const& projection, Viewport const& viewport)
{

  Vec4 u
  (
    2.0 * (win[0] - viewport[0]) / viewport[2] - 1.0,
    2.0 * (win[1] - viewport[1]) / viewport[3] - 1.0,
    2.0 * win[2] - 1.0,
    1.0
  );

  Matrix invpm = projection * modelview;
  invpm.invert();

  Vec4 v = invpm * u;

  (*obj)[0] = v[0] / v[3];
  (*obj)[1] = v[1] / v[3];
  (*obj)[2] = v[2] / v[3];

}


template < typename T >
Recti bounds(vvBaseAABB< T > const& aabb, Matrix const& modelview, Matrix const& projection, Viewport const& viewport)
{

  T minx =  std::numeric_limits< T >::max();
  T miny =  std::numeric_limits< T >::max();
  T maxx = -std::numeric_limits< T >::max();
  T maxy = -std::numeric_limits< T >::max();


  const typename vvBaseAABB< T >::vvBoxCorners& vertices = aabb.getVertices();

  for (size_t i = 0; i < 8; ++i)
  {

    Vec3 win;
    project(&win, vertices[i], modelview, projection, viewport);

    minx = std::min(win[0], minx);
    miny = std::min(win[1], miny);
    maxx = std::max(win[0], maxx);
    maxy = std::max(win[1], maxy);

  }


  Recti result;

  result[0] = std::max(0, static_cast<int>(floorf(minx)));
  result[1] = std::max(0, static_cast<int>(floorf(miny)));
  result[2] = std::min(static_cast<int>(ceilf(fabsf(maxx - minx))), viewport[2] - result[0]);
  result[3] = std::min(static_cast<int>(ceilf(fabsf(maxy - miny))), viewport[3] - result[1]);

  return result;

}


} // virvo


