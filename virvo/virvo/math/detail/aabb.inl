namespace virvo
{


namespace math
{

template < typename T >
base_aabb< T > combine(base_aabb< T > const& a, base_aabb< T > const& b)
{
    return base_aabb< T >( min(a.min, b.min), max(a.max, b.max) );
}

template < typename T >
base_aabb< T > intersect(base_aabb< T > const& a, base_aabb< T > const& b)
{
    return base_aabb< T >( max(a.min, b, min), min(a.max, b.max) );
}

template < typename T >
typename base_aabb< T >::vertex_list compute_vertices(base_aabb< T > const& box)
{

    vector< 3, T > min = box.min;
    vector< 3, T > max = box.max;

    typename base_aabb< T >::vertex_list result =
    {
        vector< 3, T >(max.x, max.y, max.z),
        vector< 3, T >(min.x, max.y, max.z),
        vector< 3, T >(min.x, min.y, max.z),
        vector< 3, T >(max.x, min.y, max.z),
        vector< 3, T >(min.x, max.y, min.z),
        vector< 3, T >(max.x, max.y, min.z),
        vector< 3, T >(max.x, min.y, min.z),
        vector< 3, T >(min.x, min.y, min.z)
    };

    return result;

}

} // math

} // virvo


