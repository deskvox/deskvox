#ifndef VV_SAMPLER_COMMON_H
#define VV_SAMPLER_COMMON_H


#include "vvforceinline.h"

#include "math/math.h"


namespace virvo
{


template < typename T, typename FloatT >
VV_FORCE_INLINE T lerp(T a, T b, FloatT x)
{
    return a + x * (b - a);
}


namespace detail
{


template < typename T >
VV_FORCE_INLINE T point(T const* tex, ssize_t idx)
{
    return tex[idx];
}


template < typename T >
VV_FORCE_INLINE math::sse_vec point(T const* tex, math::sse_vec idx)
{

    math::sse_veci iidx( idx );
    CACHE_ALIGN int indices[4];
    store(idx, &indices[0]);
    return math::sse_vec
    (
        tex[indices[0]],
        tex[indices[1]],
        tex[indices[2]],
        tex[indices[3]]
    );

}


VV_FORCE_INLINE math::vector< 4, math::sse_vec > point(math::vector< 4, float > const* tex, math::sse_vec idx)
{

    // Special case: colors are AoS. Those can be obtained
    // without a context switch to GP registers by transposing
    // to SoA after memory lookup.

    math::sse_veci iidx( idx * 4 );
    CACHE_ALIGN int indices[4];
    store(iidx, &indices[0]);

    float const* tmp = reinterpret_cast< float const* >(tex);

    math::vector< 4, math::sse_vec > colors
    (
        &tmp[0] + indices[0],
        &tmp[0] + indices[1],
        &tmp[0] + indices[2],
        &tmp[0] + indices[3]
    );

    colors = transpose(colors);
    return colors;

}


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


