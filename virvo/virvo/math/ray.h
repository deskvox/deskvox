#ifndef VV_MATH_RAY_H
#define VV_MATH_RAY_H

#include <virvo/vvmacros.h>

namespace MATH_NAMESPACE
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

} // MATH_NAMESPACE

#endif // VV_MATH_RAY_H


