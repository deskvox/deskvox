#pragma once


#include <limits>


namespace virvo
{


template < typename T >
inline void project(math::vector< 3, T >* win, math::vector< 3, T > const& obj, math::matrix< 4, 4, T > const& modelview,
    math::matrix< 4, 4, T > const& projection, math::recti const& viewport)
{

    using namespace math;

    vec4 u(obj, T(1.0));
    vec4 tmp = projection * modelview * u;

    vec3 v = tmp.xyz() / tmp.w;

    (*win)[0] = viewport[0] + viewport[2] * (v[0] + 1) / 2;
    (*win)[1] = viewport[1] + viewport[3] * (v[1] + 1) / 2;
    (*win)[2] = (v[2] + 1) / 2;

}


template < typename T >
inline void unproject(math::vector< 3, T >* obj, math::vector< 3, T > const& win, math::matrix< 4, 4, T > const& modelview,
    math::matrix< 4, 4, T > const& projection, math::recti const& viewport)
{

    using namespace math;

    vec4 u
    (
        T(2.0 * (win[0] - viewport[0]) / viewport[2] - 1.0),
        T(2.0 * (win[1] - viewport[1]) / viewport[3] - 1.0),
        T(2.0 * win[2] - 1.0),
        T(1.0)
    );

    matrix< 4, 4, T > invpm = inverse( projection * modelview );

    vec4 v = invpm * u;
    (*obj) = v.xyz() / v.w;

}


template < typename T >
math::recti bounds(vvBaseAABB< T > const& aabb, math::matrix< 4, 4, T > const& modelview,
    math::matrix< 4, 4, T > const& projection, math::recti const& viewport)
{

    using namespace math;

    T minx =  std::numeric_limits< T >::max();
    T miny =  std::numeric_limits< T >::max();
    T maxx = -std::numeric_limits< T >::max();
    T maxy = -std::numeric_limits< T >::max();


    const typename vvBaseAABB< T >::vvBoxCorners& vertices = aabb.getVertices();

    for (size_t i = 0; i < 8; ++i)
    {

        vec3f win;
        project(&win, vec3( vertices[i] ), modelview, projection, viewport);

        minx = std::min(win[0], minx);
        miny = std::min(win[1], miny);
        maxx = std::max(win[0], maxx);
        maxy = std::max(win[1], maxy);

    }


    recti result
    (
        static_cast<int>(floorf(minx)),
        static_cast<int>(floorf(miny)),
        static_cast<int>(ceilf(fabsf(maxx - minx))),
        static_cast<int>(ceilf(fabsf(maxy - miny)))
    );

    return intersect( result, viewport );

}


} // virvo


