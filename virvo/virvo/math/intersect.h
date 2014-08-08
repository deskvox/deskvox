#ifndef VV_MATH_INTERSECT_H
#define VV_MATH_INTERSECT_H

#include "vector.h"

namespace virvo
{

template < typename T1, typename T2 >
struct hit_record;

template < typename T >
struct hit_record< basic_ray< T >, basic_plane< 3, T > >
{

    typedef T value_type;

    hit_record() : hit(false) {}

    bool                    hit;
    value_type              t;
    vector< 3, value_type > pos;

};

template < typename T >
inline hit_record< basic_ray< T >, basic_plane< 3, T > > intersect
(
    basic_ray< T > const& ray, basic_plane< 3, T > const& p
)
{

    hit_record< basic_ray< T >, basic_plane< 3, T > > result;
    T s = dot(p.normal, ray.dir);

    if (s == T(0.0))
    {
        result.hit = false;
    }
    else
    {
        result.hit = true;
        result.t   = ( p.offset - dot(p.normal, ray.ori) ) / s;
        result.pos = ray.ori + result.t * ray.dir;
    }
    return result;

}

} // virvo

#endif // VV_MATH_INTERSECT_H


