#ifndef VV_TEXTURE_SAMPLER3D_H
#define VV_TEXTURE_SAMPLER3D_H


#include "sampler_common.h"
#include "texture_common.h"

#include "vvtoolshed.h"

#include "math/math.h"

#include <boost/detail/endian.hpp>


namespace virvo
{


namespace detail
{


#ifdef BOOST_LITTLE_ENDIAN
static const size_t high_byte_offset = 1;
#else
static const size_t high_byte_offset = 0;
#endif


template < typename IntT, typename Int3T >
VV_FORCE_INLINE IntT index(IntT x, IntT y, IntT z, Int3T texsize)
{
    return z * texsize[0] * texsize[1] + y * texsize[0] + x;
}


template < int bpc, typename VoxelT >
VV_FORCE_INLINE float point(VoxelT const* tex, ssize_t idx)
{

    return tex[idx * bpc + high_byte_offset * (bpc - 1)];

}


template < int bpc, typename VoxelT >
VV_FORCE_INLINE math::sse_vec point(VoxelT const* tex, math::sse_veci idx)
{

    CACHE_ALIGN int indices[4];
    math::sse_veci ridx = idx * bpc + high_byte_offset * (bpc - 1);
    math::store(ridx, &indices[0]);
    CACHE_ALIGN float vals[4];
    for (size_t i = 0; i < 4; ++i)
    {
        vals[i] = tex[indices[i]];
    }
    return math::sse_vec(&vals[0]);

}


template
<
    int bpc,
    typename Float3T,
    typename Int3T,
    typename Float2IntFunc,
    typename Int2FloatFunc,
    typename VoxelT
>
#ifndef _MSC_VER
VV_FORCE_INLINE
#endif
typename Float3T::value_type nearest(VoxelT const* tex, Float3T coord, Int3T texsize,
    Float2IntFunc ftoi, Int2FloatFunc itof)
{

#if 1

    using toolshed::clamp;

    typedef typename Int3T::value_type int_type;

    Float3T texsizef(itof(texsize[0]), itof(texsize[1]), itof(texsize[2]));

    Int3T texcoordi(ftoi(coord[0] * texsizef[0]), ftoi(coord[1] * texsizef[1]),
        ftoi(coord[2] * texsizef[2]));

    texcoordi[0] = clamp(texcoordi[0], int_type(0), texsize[0] - 1);
    texcoordi[1] = clamp(texcoordi[1], int_type(0), texsize[1] - 1);
    texcoordi[2] = clamp(texcoordi[2], int_type(0), texsize[2] - 1);

    int_type idx = index(texcoordi[0], texcoordi[1], texcoordi[2], texsize);
    return point< bpc >(tex, idx);

#else

    // TODO: should be done similar to the code below..

    Float3T texsizef(itof(texsize[0]), itof(texsize[1]), itof(texsize[2]));

    Float3T texcoordf(coord[0] * texsizef[0] - FloatT(0.5),
                      coorm[1] * texsizef[1] - FloatT(0.5),
                      coord[2] * texsizef[2] - FloatT(0.5));

    texcoordf[0] = clamp( texcoordf[0], FloatT(0.0), texsizef[0] - 1 );
    texcoordf[1] = clamp( texcoordf[1], FloatT(0.0), texsizef[1] - 1 );
    texcoordf[2] = clamp( texcoordf[2], FloatT(0.0), texsizef[2] - 1 );

    Float3T lof( floor(texcoordf[0]), floor(texcoordf[1]), floor(texcoordf[2]) );
    Float3T hif( ceil(texcoordf[0]),  ceil(texcoordf[1]),  ceil(texcoordf[2]) );
    Int3T   lo( ftoi(lof[0]), ftoi(lof[1]), ftoi(lof[2]) );
    Int3T   hi( ftoi(hif[0]), ftoi(hif[1]), ftoi(hif[2]) );

    Float3T uvw = texcoordf - uvw;

    IntT idx = index( uvw[0] < FloatT(0.5) ? lo[0] : hi[0],
                      uvw[1] < FloatT(0.5) ? lo[1] : hi[1],
                      uvw[2] < FloatT(0.5) ? lo[2] : hi[2],
                      texsize);

    return point< bpc >(tex, idx);

#endif

}


template
<
    int bpc,
    typename Float3T,
    typename Int3T,
    typename Float2IntFunc,
    typename Int2FloatFunc,
    typename VoxelT
>
#ifndef _MSC_VER
VV_FORCE_INLINE
#endif
typename Float3T::value_type linear(VoxelT const* tex, Float3T coord, Int3T texsize,
    Float2IntFunc ftoi, Int2FloatFunc itof)
{

    using toolshed::clamp;

    typedef typename Float3T::value_type float_type;
    typedef typename Int3T::value_type int_type;

    Float3T texsizef(itof(texsize[0]), itof(texsize[1]), itof(texsize[2]));

    Float3T texcoordf(coord[0] * texsizef[0] - float_type(0.5),
                      coord[1] * texsizef[1] - float_type(0.5),
                      coord[2] * texsizef[2] - float_type(0.5));

    texcoordf[0] = clamp( texcoordf[0], float_type(0.0), texsizef[0] - 1 );
    texcoordf[1] = clamp( texcoordf[1], float_type(0.0), texsizef[1] - 1 );
    texcoordf[2] = clamp( texcoordf[2], float_type(0.0), texsizef[2] - 1 );

    Float3T lof( floor(texcoordf[0]), floor(texcoordf[1]), floor(texcoordf[2]) );
    Float3T hif( ceil(texcoordf[0]),  ceil(texcoordf[1]),  ceil(texcoordf[2]) );
    Int3T   lo( ftoi(lof[0]), ftoi(lof[1]), ftoi(lof[2]) );
    Int3T   hi( ftoi(hif[0]), ftoi(hif[1]), ftoi(hif[2]) );


    float_type samples[8] =
    {
        point< bpc >(tex, index( lo[0], lo[1], lo[2], texsize )),
        point< bpc >(tex, index( hi[0], lo[1], lo[2], texsize )),
        point< bpc >(tex, index( lo[0], hi[1], lo[2], texsize )),
        point< bpc >(tex, index( hi[0], hi[1], lo[2], texsize )),
        point< bpc >(tex, index( lo[0], lo[1], hi[2], texsize )),
        point< bpc >(tex, index( hi[0], lo[1], hi[2], texsize )),
        point< bpc >(tex, index( lo[0], hi[1], hi[2], texsize )),
        point< bpc >(tex, index( hi[0], hi[1], hi[2], texsize ))
    };


    Float3T uvw = texcoordf - lof;

    float_type p1  = lerp(samples[0], samples[1], uvw[0]);
    float_type p2  = lerp(samples[2], samples[3], uvw[0]);
    float_type p3  = lerp(samples[4], samples[5], uvw[0]);
    float_type p4  = lerp(samples[6], samples[7], uvw[0]);

    float_type p12 = lerp(p1, p2, uvw[1]);
    float_type p34 = lerp(p3, p4, uvw[1]);

    return lerp(p12, p34, uvw[2]);

}


template
<
    int bpc,
    typename Float3T,
    typename Int3T,
    typename Float2IntFunc,
    typename Int2FloatFunc,
    typename VoxelT
>
#ifndef _MSC_VER
VV_FORCE_INLINE
#endif
typename Float3T::value_type cubic(VoxelT const* tex, Float3T coord, Int3T texsize,
    Float2IntFunc ftoi, Int2FloatFunc itof)
{

    typedef typename Float3T::value_type float_type;
    typedef typename Int3T::value_type int_type;

    Float3T texsizef(itof(texsize[0]), itof(texsize[1]), itof(texsize[2]));

    float_type x = (coord[0] * texsizef[0]) - float_type(0.5);
    float_type floorx = floor( x );
    float_type fracx  = x - floor( x );

    float_type y = (coord[1] * texsizef[1]) - float_type(0.5);
    float_type floory = floor( y );
    float_type fracy  = y - floor( y );

    float_type z = (coord[2] * texsizef[2]) - float_type(0.5);
    float_type floorz = floor( z );
    float_type fracz  = z - floor( z );


    float_type tmp000 = ( w1(fracx) ) / ( w0(fracx) + w1(fracx) ) ;
    float_type h_000 = ( floorx - float_type(0.5) + tmp000 ) / texsizef[0];

    float_type tmp100 = ( w3(fracx) ) / ( w2(fracx) + w3(fracx) ) ;
    float_type h_100 = ( floorx + float_type(1.5) + tmp100 ) / texsizef[0];

    float_type tmp010 = ( w1(fracy) ) / ( w0(fracy) + w1(fracy) ) ;
    float_type h_010 = ( floory - float_type(0.5) + tmp010 ) / texsizef[1] ;

    float_type tmp110 = ( w3(fracy) ) / ( w2(fracy) + w3(fracy) ) ;
    float_type h_110 = ( floory + float_type(1.5) + tmp110 ) / texsizef[1];

    float_type tmp001 = ( w1(fracz) ) / ( w0(fracz) + w1(fracz) ) ;
    float_type h_001 = ( floorz - float_type(0.5) + tmp001 ) / texsizef[2];

    float_type tmp101 = ( w3(fracz) ) / ( w2(fracz) + w3(fracz) ) ;
    float_type h_101 = ( floorz + float_type(1.5) + tmp101 ) / texsizef[2];


    float_type f_000 = linear< bpc >( tex, Float3T(h_000, h_010, h_001), texsize, ftoi, itof );
    float_type f_100 = linear< bpc >( tex, Float3T(h_100, h_010, h_001), texsize, ftoi, itof );
    float_type f_010 = linear< bpc >( tex, Float3T(h_000, h_110, h_001), texsize, ftoi, itof );
    float_type f_110 = linear< bpc >( tex, Float3T(h_100, h_110, h_001), texsize, ftoi, itof );

    float_type f_001 = linear< bpc >( tex, Float3T(h_000, h_010, h_101), texsize, ftoi, itof );
    float_type f_101 = linear< bpc >( tex, Float3T(h_100, h_010, h_101), texsize, ftoi, itof );
    float_type f_011 = linear< bpc >( tex, Float3T(h_000, h_110 ,h_101), texsize, ftoi, itof );
    float_type f_111 = linear< bpc >( tex, Float3T(h_100, h_110, h_101), texsize, ftoi, itof );

    float_type f_00  = g0(fracx) * f_000 + g1(fracx) * f_100;
    float_type f_10  = g0(fracx) * f_010 + g1(fracx) * f_110;
    float_type f_01  = g0(fracx) * f_001 + g1(fracx) * f_101;
    float_type f_11  = g0(fracx) * f_011 + g1(fracx) * f_111;

    float_type f_0   = g0(fracy) * f_00 + g1(fracy) * f_10;
    float_type f_1   = g0(fracy) * f_01 + g1(fracy) * f_11;

    return g0(fracz) * f_0 + g1(fracz) * f_1;

}


template
<
    int bpc,
    typename Float3T,
    typename Int3T,
    typename VoxelT
>
VV_FORCE_INLINE typename Float3T::value_type tex3D(VoxelT const* tex, Float3T coord, Int3T texsize,
    virvo::tex_filter_mode filter_mode = virvo::Nearest)
{

    typedef typename Float3T::value_type float_type;
    typedef typename Int3T::value_type int_type;

    Caster< float_type, int_type > ftoi;
    Caster< int_type, float_type > itof;

    switch (filter_mode)
    {

    default:
        // fall-through
    case virvo::Nearest:
        return nearest< bpc >( tex, coord, texsize, ftoi, itof );

    case virvo::Linear:
        return linear< bpc >( tex, coord, texsize, ftoi, itof );

    case virvo::BSpline:
        // fall-through
    case virvo::BSplineInterpol:
        return cubic< bpc >( tex, coord, texsize, ftoi, itof );

    }

}


} // detail


} // virvo


#endif // VV_TEXTURE_SAMPLER3D_H


