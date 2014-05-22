#ifndef VV_TEXTURE_SAMPLER3D_H
#define VV_TEXTURE_SAMPLER3D_H


#include "sampler_common.h"

#include "vvtoolshed.h"

#include "simd/simd.h"

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


template
<
    typename FloatT,
    typename IntT,
    int bpc,
    typename VoxelT
>
VV_FORCE_INLINE FloatT point(VoxelT const* tex, IntT idx)
{
#if VV_USE_SSE

    CACHE_ALIGN int indices[4];
    IntT ridx = idx * bpc + high_byte_offset * (bpc - 1);
    virvo::simd::store(ridx, &indices[0]);
    CACHE_ALIGN float vals[4];
    for (size_t i = 0; i < 4; ++i)
    {
        vals[i] = tex[indices[i]];
    }
    return FloatT(&vals[0]);

#else

    return tex[idx * bpc + high_byte_offset * (bpc - 1)];

#endif
}


template
<
    int bpc,
    typename FloatT,
    typename IntT,
    typename Float3T,
    typename Int3T,
    typename Float2IntFunc,
    typename Int2FloatFunc,
    typename VoxelT
>
VV_FORCE_INLINE FloatT nearest(VoxelT const* tex, Float3T coord, Int3T texsize,
    Float2IntFunc ftoi, Int2FloatFunc itof)
{

#if 1

    Float3T texsizef(itof(texsize[0]), itof(texsize[1]), itof(texsize[2]));

    Int3T texcoordi(ftoi(coord[0] * texsizef[0]), ftoi(coord[1] * texsizef[1]),
        ftoi(coord[2] * texsizef[2]));

    texcoordi[0] = clamp(texcoordi[0], IntT(0), texsize[0] - 1);
    texcoordi[1] = clamp(texcoordi[1], IntT(0), texsize[1] - 1);
    texcoordi[2] = clamp(texcoordi[2], IntT(0), texsize[2] - 1);

    IntT idx = index(texcoordi[0], texcoordi[1], texcoordi[2], texsize);
    return point< FloatT, IntT, bpc >(tex, idx);

#else

    // TODO: should be done similar to the code below..

    Float3T texsizef(itof(texsize[0]), itof(texsize[1]), itof(texsize[2]));

    Float3T texcoordf(coord[0] * texsizef[0] - FloatT(0.5),
                      coorm[1] * texsizef[1] - FloatT(0.5),
eglichen
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

    return point< FloatT, IntT, bpc >(tex, idx);

#endif

}


template
<
    int bpc,
    typename FloatT,
    typename IntT,
    typename Float3T,
    typename Int3T,
    typename Float2IntFunc,
    typename Int2FloatFunc,
    typename VoxelT
>
VV_FORCE_INLINE FloatT linear(VoxelT const* tex, Float3T coord, Int3T texsize,
    Float2IntFunc ftoi, Int2FloatFunc itof)
{

    Float3T texsizef(itof(texsize[0]), itof(texsize[1]), itof(texsize[2]));

    Float3T texcoordf(coord[0] * texsizef[0] - FloatT(0.5),
                      coord[1] * texsizef[1] - FloatT(0.5),
                      coord[2] * texsizef[2] - FloatT(0.5));

    texcoordf[0] = clamp( texcoordf[0], FloatT(0.0), texsizef[0] - 1 );
    texcoordf[1] = clamp( texcoordf[1], FloatT(0.0), texsizef[1] - 1 );
    texcoordf[2] = clamp( texcoordf[2], FloatT(0.0), texsizef[2] - 1 );

    Float3T lof( floor(texcoordf[0]), floor(texcoordf[1]), floor(texcoordf[2]) );
    Float3T hif( ceil(texcoordf[0]),  ceil(texcoordf[1]),  ceil(texcoordf[2]) );
    Int3T   lo( ftoi(lof[0]), ftoi(lof[1]), ftoi(lof[2]) );
    Int3T   hi( ftoi(hif[0]), ftoi(hif[1]), ftoi(hif[2]) );


    FloatT samples[8] =
    {
        point< FloatT, IntT, bpc >(tex, index( lo[0], lo[1], lo[2], texsize )),
        point< FloatT, IntT, bpc >(tex, index( hi[0], lo[1], lo[2], texsize )),
        point< FloatT, IntT, bpc >(tex, index( lo[0], hi[1], lo[2], texsize )),
        point< FloatT, IntT, bpc >(tex, index( hi[0], hi[1], lo[2], texsize )),
        point< FloatT, IntT, bpc >(tex, index( lo[0], lo[1], hi[2], texsize )),
        point< FloatT, IntT, bpc >(tex, index( hi[0], lo[1], hi[2], texsize )),
        point< FloatT, IntT, bpc >(tex, index( lo[0], hi[1], hi[2], texsize )),
        point< FloatT, IntT, bpc >(tex, index( hi[0], hi[1], hi[2], texsize ))
    };


    Float3T uvw = texcoordf - lof;

    FloatT p1  = lerp(samples[0], samples[1], uvw[0]);
    FloatT p2  = lerp(samples[2], samples[3], uvw[0]);
    FloatT p3  = lerp(samples[4], samples[5], uvw[0]);
    FloatT p4  = lerp(samples[6], samples[7], uvw[0]);

    FloatT p12 = lerp(p1, p2, uvw[1]);
    FloatT p34 = lerp(p3, p4, uvw[1]);

    return lerp(p12, p34, uvw[2]);

}


} // detail


/* nearest neighbor reconstruction (default)
 */
template < int bpc, typename VoxelT >
VV_FORCE_INLINE float nearest(VoxelT const* tex, virvo::Vec3 coord, virvo::ssize3 texsize)
{

    using namespace virvo::detail;


    StaticCaster< float, ssize_t > ftoi;
    StaticCaster< ssize_t, float > itof;

    return virvo::detail::nearest
    <
        bpc,
        float,
        ssize_t
    >
    ( tex, coord, texsize, ftoi, itof );
}


/* linear filtering (default)
 */
template < int bpc, typename VoxelT >
VV_FORCE_INLINE float linear(VoxelT const* tex, virvo::Vec3 coord, virvo::ssize3 texsize)
{

    using namespace virvo::detail;


    StaticCaster< float, ssize_t > ftoi;
    StaticCaster< ssize_t, float > itof;

    return virvo::detail::linear
    <
        bpc,
        float,
        ssize_t
    >
    ( tex, coord, texsize, ftoi, itof );

}


namespace simd
{


/* nearest neighbor reconstruction (default)
 */
template < int bpc, typename VoxelT >
VV_FORCE_INLINE virvo::simd::Vec nearest(VoxelT const* tex, virvo::simd::Vec3 coord, virvo::simd::Vec3i texsize)
{

    using namespace virvo::detail;
    using virvo::simd::Vec;
    using virvo::simd::Veci;


    SimdCaster< Vec, Veci > ftoi;
    SimdCaster< Veci, Vec > itof;

    return virvo::detail::nearest
    <
        bpc,
        virvo::simd::Vec,
        virvo::simd::Veci
    >
    ( tex, coord, texsize, ftoi, itof );

}


/* linear filtering (simd)
 */
template < int bpc, typename VoxelT >
VV_FORCE_INLINE virvo::simd::Vec linear(VoxelT const* tex, virvo::simd::Vec3 coord, virvo::simd::Vec3i texsize)
{

    using namespace virvo::detail;
    using virvo::simd::Vec;
    using virvo::simd::Veci;


    SimdCaster< Vec, Veci > ftoi;
    SimdCaster< Veci, Vec > itof;

    return virvo::detail::linear
    <
        bpc,
        virvo::simd::Vec,
        virvo::simd::Veci
    >
    ( tex, coord, texsize, ftoi, itof );

}


} // simd


} // virvo


#endif // VV_TEXTURE_SAMPLER3D_H


