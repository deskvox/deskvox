#ifndef VV_MATH_INTERSECT_H
#define VV_MATH_INTERSECT_H

#include "simd/mask.h"
#include "simd/vec.h"

#include "aabb.h"
#include "plane.h"
#include "vector.h"

namespace virvo
{

template < typename T1, typename T2 >
struct hit_record;


//-------------------------------------------------------------------------------------------------
// ray / plane
//

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


//-------------------------------------------------------------------------------------------------
// ray / aabb
//

template < typename T >
struct hit_record< basic_ray< T >, basic_aabb< T > >
{

    typedef T value_type;

    hit_record() : hit(false) {}

    bool            hit;
    value_type      tnear;
    value_type      tfar;

};

template < >
struct hit_record< basic_ray< simd::float4 >, basic_aabb< simd::float4 > >
{

    simd::mask4     hit;
    simd::float4    tnear;
    simd::float4    tfar;

};

template < typename T >
inline hit_record< basic_ray< T >, basic_aabb< T > > intersect
(
    basic_ray< T > const& ray, basic_aabb< T > const& aabb
)
{

    hit_record< basic_ray< T >, basic_aabb< T > > result;

    vector< 3, T > invr( T(1.0) / ray.dir.x, T(1.0) / ray.dir.y, T(1.0) / ray.dir.z );
    vector< 3, T > t1 = (aabb.min - ray.ori) * invr;
    vector< 3, T > t2 = (aabb.max - ray.ori) * invr;

    result.tnear = max( min(t1.z, t2.z), max( min(t1.y, t2.y), min(t1.x, t2.x) ) );
    result.tfar  = min( max(t1.z, t2.z), min( max(t1.y, t2.y), max(t1.x, t2.x) ) );
    result.hit   = result.tfar >= result.tnear && result.tfar >= T(0.0);

    return result;

}


} // virvo

#endif // VV_MATH_INTERSECT_H


