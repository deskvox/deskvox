#ifndef VV_DETAIL_MATH_H
#define VV_DETAIL_MATH_H


#include <virvo/vvmacros.h>

#include <cmath>


namespace virvo
{

namespace math
{

//--------------------------------------------------------------------------------------------------
// Import required math functions from the standard library.
// Enable ADL!
//

using std::sqrt;

template < typename T >
VV_FORCE_INLINE T min(T const& x, T const& y)
{
    return x < y ? x : y;
}

template < typename T >
VV_FORCE_INLINE T max(T const& x, T const& y)
{
    return x < y ? y : x;
}


template < typename T >
VV_FORCE_INLINE T clamp(T const& x, T const& a, T const& b)
{
    return max( a, min(x, b) );
}


template < typename T >
VV_FORCE_INLINE T rsqrt(T const& x)
{
    return T(1.0) / sqrt(x);
}


} // math


} // virvo


#endif // VV_DETAIL_MATH_H


