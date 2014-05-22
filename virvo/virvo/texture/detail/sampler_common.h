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
struct Caster
{
    VV_FORCE_INLINE T operator()(U const& u) { return static_cast< T >(u); }
};


template < >
struct Caster< simd::Vec, simd::Veci >
{
    VV_FORCE_INLINE simd::Veci operator()(simd::Vec const& u) { return virvo::simd::simd_cast< simd::Veci >(u); }
};


template < >
struct Caster< simd::Veci, simd::Vec >
{
    VV_FORCE_INLINE simd::Vec operator()(simd::Veci const& u) { return virvo::simd::simd_cast< simd::Vec >(u); }
};



// weight functions for Mitchell - Netravalli B-Spline
template < typename FloatT >
VV_FORCE_INLINE FloatT w0( FloatT a ) { return FloatT( (1.0 / 6.0) * (-(a * a * a) + 3.0 * a * a - 3.0 * a + 1.0) ); }

template < typename FloatT >
VV_FORCE_INLINE FloatT w1( FloatT a ) { return FloatT( (1.0 / 6.0) * (3.0 * a * a * a - 6.0 * a * a + 4.0) ); }

template < typename FloatT >
VV_FORCE_INLINE FloatT w2( FloatT a ) { return FloatT( (1.0 / 6.0) * (-3.0 * a * a * a + 3.0 * a * a + 3.0 * a + 1.0) ); }

template < typename FloatT >
VV_FORCE_INLINE FloatT w3( FloatT a ) { return FloatT( (1.0 / 6.0) * (a * a * a) ); }


// helper functions for cubic interpolation
template < typename FloatT >
VV_FORCE_INLINE FloatT g0( FloatT x ) { return w0(x) + w1(x); }

template < typename FloatT >
VV_FORCE_INLINE FloatT g1( FloatT x ) { return w2(x) + w3(x); }

template < typename FloatT >
VV_FORCE_INLINE FloatT h0( FloatT x ) { return ((floor( x ) - FloatT(1.0) + w1(x)) / (w0(x) + w1(x))) + x; }

template < typename FloatT >
VV_FORCE_INLINE FloatT h1( FloatT x ) { return ((floor( x ) + FloatT(1.0) + w3(x)) / (w2(x) + w3(x))) - x; }


} // detail


} // virvo


#endif // VV_SAMPLER_COMMON_H


