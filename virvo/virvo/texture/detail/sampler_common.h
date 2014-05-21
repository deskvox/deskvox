#ifndef VV_SAMPLER_COMMON_H
#define VV_SAMPLER_COMMON_H


#include "vvforceinline.h"

#include "simd/simd.h"


namespace virvo
{


template < typename FloatT >
VV_FORCE_INLINE FloatT lerp(FloatT a, FloatT b, FloatT x)
{
    return a + x * (b - a);
}


namespace detail
{


template < typename U, typename T >
struct StaticCaster
{
    VV_FORCE_INLINE T operator()(U const& u) { return static_cast< T >(u); }
};


template < typename U, typename T >
struct SimdCaster
{
    VV_FORCE_INLINE T operator()(U const& u) { return virvo::simd::simd_cast< T >(u); }
};



} // detail


} // virvo


#endif // VV_SAMPLER_COMMON_H


