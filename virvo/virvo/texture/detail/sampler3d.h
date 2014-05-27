#ifndef VV_TEXTURE_SAMPLER3D_H
#define VV_TEXTURE_SAMPLER3D_H


#include "sampler_common.h"
#include "texture_common.h"

#include "math/math.h"


namespace virvo
{


namespace detail
{


template < typename T >
VV_FORCE_INLINE T index(T x, T y, T z, math::base_vec3< T > texsize)
{
    return z * texsize[0] * texsize[1] + y * texsize[0] + x;
}


template
<
    typename ReturnT,
    typename FloatT,
    typename VoxelT
>
#ifndef _MSC_VER
VV_FORCE_INLINE
#endif
ReturnT nearest(VoxelT const* tex, math::base_vec3< FloatT > coord, math::base_vec3< FloatT > texsize)
{

#if 1

    using toolshed::clamp;

    typedef FloatT float_type;
    typedef math::base_vec3< float_type > float3_type;

    float3_type lo
    (
        floor(coord.x * texsize.x),
        floor(coord.y * texsize.y),
        floor(coord.z * texsize.z)
    );

    lo[0] = clamp(lo[0], float_type(0.0f), texsize[0] - 1);
    lo[1] = clamp(lo[1], float_type(0.0f), texsize[1] - 1);
    lo[2] = clamp(lo[2], float_type(0.0f), texsize[2] - 1);

    float_type idx = index(lo[0], lo[1], lo[2], texsize);
    return point(tex, idx);

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

    return point(tex, idx);

#endif

}


template
<
    typename ReturnT,
    typename FloatT,
    typename VoxelT
>
#ifndef _MSC_VER
VV_FORCE_INLINE
#endif
ReturnT linear(VoxelT const* tex, math::base_vec3< FloatT > coord, math::base_vec3< FloatT > texsize)
{

    using toolshed::clamp;

    typedef FloatT float_type;
    typedef math::base_vec3< float_type > float3_type;
    typedef ReturnT return_type;

    float3_type texcoordf( coord * texsize - float_type(0.5) );

    texcoordf[0] = clamp( texcoordf[0], float_type(0.0), texsize[0] - 1 );
    texcoordf[1] = clamp( texcoordf[1], float_type(0.0), texsize[1] - 1 );
    texcoordf[2] = clamp( texcoordf[2], float_type(0.0), texsize[2] - 1 );

    float3_type lo( floor(texcoordf[0]), floor(texcoordf[1]), floor(texcoordf[2]) );
    float3_type hi( ceil(texcoordf[0]),  ceil(texcoordf[1]),  ceil(texcoordf[2]) );


    // Implicit cast from return type to float type.
    // TODO: what if return type is e.g. a float4?
    float_type samples[8] =
    {
        point(tex, index( lo[0], lo[1], lo[2], texsize )),
        point(tex, index( hi[0], lo[1], lo[2], texsize )),
        point(tex, index( lo[0], hi[1], lo[2], texsize )),
        point(tex, index( hi[0], hi[1], lo[2], texsize )),
        point(tex, index( lo[0], lo[1], hi[2], texsize )),
        point(tex, index( hi[0], lo[1], hi[2], texsize )),
        point(tex, index( lo[0], hi[1], hi[2], texsize )),
        point(tex, index( hi[0], hi[1], hi[2], texsize ))
    };


    float3_type uvw = texcoordf - lo;

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
    typename ReturnT,
    typename FloatT,
    typename VoxelT
>
#ifndef _MSC_VER
VV_FORCE_INLINE
#endif
ReturnT cubic(VoxelT const* tex, math::base_vec3< FloatT > coord, math::base_vec3< FloatT > texsize)
{

    typedef FloatT float_type;
    typedef math::base_vec3< float_type > float3_type;

    float_type x = coord[0] * texsize[0] - float_type(0.5);
    float_type floorx = floor( x );
    float_type fracx  = x - floor( x );

    float_type y = coord[1] * texsize[1] - float_type(0.5);
    float_type floory = floor( y );
    float_type fracy  = y - floor( y );

    float_type z = coord[2] * texsize[2] - float_type(0.5);
    float_type floorz = floor( z );
    float_type fracz  = z - floor( z );


    float_type tmp000 = ( w1(fracx) ) / ( w0(fracx) + w1(fracx) );
    float_type h_000  = ( floorx - float_type(0.5) + tmp000 ) / texsize[0];

    float_type tmp100 = ( w3(fracx) ) / ( w2(fracx) + w3(fracx) );
    float_type h_100  = ( floorx + float_type(1.5) + tmp100 ) / texsize[0];

    float_type tmp010 = ( w1(fracy) ) / ( w0(fracy) + w1(fracy) );
    float_type h_010  = ( floory - float_type(0.5) + tmp010 ) / texsize[1] ;

    float_type tmp110 = ( w3(fracy) ) / ( w2(fracy) + w3(fracy) );
    float_type h_110  = ( floory + float_type(1.5) + tmp110 ) / texsize[1];

    float_type tmp001 = ( w1(fracz) ) / ( w0(fracz) + w1(fracz) );
    float_type h_001  = ( floorz - float_type(0.5) + tmp001 ) / texsize[2];

    float_type tmp101 = ( w3(fracz) ) / ( w2(fracz) + w3(fracz) );
    float_type h_101  = ( floorz + float_type(1.5) + tmp101 ) / texsize[2];


    // Implicit cast from return type to float type.
    // TODO: what if return type is e.g. a float4?
    float_type f_000 = linear< float_type >( tex, float3_type(h_000, h_010, h_001), texsize );
    float_type f_100 = linear< float_type >( tex, float3_type(h_100, h_010, h_001), texsize );
    float_type f_010 = linear< float_type >( tex, float3_type(h_000, h_110, h_001), texsize );
    float_type f_110 = linear< float_type >( tex, float3_type(h_100, h_110, h_001), texsize );

    float_type f_001 = linear< float_type >( tex, float3_type(h_000, h_010, h_101), texsize );
    float_type f_101 = linear< float_type >( tex, float3_type(h_100, h_010, h_101), texsize );
    float_type f_011 = linear< float_type >( tex, float3_type(h_000, h_110 ,h_101), texsize );
    float_type f_111 = linear< float_type >( tex, float3_type(h_100, h_110, h_101), texsize );

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
    typename ReturnT,
    typename FloatT,
    typename VoxelT
>
VV_FORCE_INLINE ReturnT tex3D(VoxelT const* tex, math::base_vec3< FloatT > coord, math::base_vec3< FloatT > texsize,
    virvo::tex_filter_mode filter_mode = virvo::Nearest)
{

    switch (filter_mode)
    {

    default:
        // fall-through
    case virvo::Nearest:
        return nearest< ReturnT >( tex, coord, texsize );

    case virvo::Linear:
        return linear< ReturnT >( tex, coord, texsize );

    case virvo::BSpline:
        // fall-through
    case virvo::BSplineInterpol:
        return cubic< ReturnT >( tex, coord, texsize );

    }

}


} // detail


} // virvo


#endif // VV_TEXTURE_SAMPLER3D_H


