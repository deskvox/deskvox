#include "../axis.h"

#include <utility>


namespace virvo
{


//--------------------------------------------------------------------------------------------------
// aabb members
//

template < typename T >
inline base_aabb< T >::base_aabb()
{
}

template < typename T >
inline base_aabb< T >::base_aabb
(
    typename base_aabb< T >::vec_type const& min,
    typename base_aabb< T >::vec_type const& max
)
    : min(min)
    , max(max)
{
}

template < typename T >
inline typename base_aabb< T >::vec_type base_aabb< T >::center() const
{
    return (max + min) * value_type(0.5);
}

template < typename T >
inline typename base_aabb< T >::vec_type base_aabb< T >::size() const
{
    return max - min;
}

template < typename T >
inline bool base_aabb< T >::contains(typename base_aabb< T >::vec_type const& v) const
{
    return v.x >= min.x && v.x <= max.x
        && v.y >= min.y && v.y <= max.y
        && v.z >= min.z && v.z <= max.z;
}


//--------------------------------------------------------------------------------------------------
// Geometric functions
//

template < typename T >
base_aabb< T > combine(base_aabb< T > const& a, base_aabb< T > const& b)
{
    return base_aabb< T >( min(a.min, b.min), max(a.max, b.max) );
}

template < typename T >
base_aabb< T > intersect(base_aabb< T > const& a, base_aabb< T > const& b)
{
    return base_aabb< T >( max(a.min, b.min), min(a.max, b.max) );
}

template < typename T >
std::pair< base_aabb< T >, base_aabb< T > > split(base_aabb< T > const& box, cartesian_axis< 3 > axis, T splitpos)
{

    vector< 3, T > min1 = box.min;
    vector< 3, T > min2 = box.min;
    vector< 3, T > max1 = box.max;
    vector< 3, T > max2 = box.max;

    max1[axis] = splitpos;
    min2[axis] = splitpos;

    base_aabb< T > box1(min1, max1);
    base_aabb< T > box2(min2, max2);
    return std::make_pair(box1, box2);

}

template < typename T >
typename base_aabb< T >::vertex_list compute_vertices(base_aabb< T > const& box)
{

    vector< 3, T > min = box.min;
    vector< 3, T > max = box.max;

    typename base_aabb< T >::vertex_list result =
    {{
        vector< 3, T >(max.x, max.y, max.z),
        vector< 3, T >(min.x, max.y, max.z),
        vector< 3, T >(min.x, min.y, max.z),
        vector< 3, T >(max.x, min.y, max.z),
        vector< 3, T >(min.x, max.y, min.z),
        vector< 3, T >(max.x, max.y, min.z),
        vector< 3, T >(max.x, min.y, min.z),
        vector< 3, T >(min.x, min.y, min.z)
    }};

    return result;

}

} // virvo


