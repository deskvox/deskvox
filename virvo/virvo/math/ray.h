#ifndef VV_MATH_RAY_H
#define VV_MATH_RAY_H

#include <virvo/vvmacros.h>

namespace virvo
{

template < typename T >
class basic_ray
{
public:

    vector< 3, T > ori;
    vector< 3, T > dir;

    VV_FORCE_INLINE basic_ray() {}
    VV_FORCE_INLINE basic_ray(vector< 3, T > const& o, vector< 3, T > const& d) : ori(o), dir(d) {}

};

} // virvo

#endif // VV_MATH_RAY_H


