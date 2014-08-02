#ifndef VV_MATH_PLANE_H
#define VV_MATH_PLANE_H

#include "forward.h"

#include <cstddef>


namespace virvo
{

template < typename T >
class hyper_plane< 3, T >
{
public:

    typedef T value_type;
    typedef vector< 3, T > vec_type;

    vec_type normal;
    vec_type offset;

    hyper_plane();

    // NOTE: n must be normalied!
    hyper_plane(vec_type const& n, value_type o);
    hyper_plane(vec_type const& n, vec_type const& p);

};

} // virvo


#include "detail/plane3.inl"


#endif // VV_MATH_PLANE_H


