#ifndef VV_TEXTURE_TEX3D_H
#define VV_TEXTURE_TEX3D_H


#include "detail/sampler3d.h"
#include "detail/texture3d.h"



namespace virvo
{


template < typename VoxelT, int bpc >
VV_FORCE_INLINE float tex3D(VoxelT const* tex, virvo::Vec3 coord, virvo::ssize3 texsize,
    virvo::tex_filter_mode filter_mode = virvo::Nearest)
{

    switch (filter_mode)
    {

    case virvo::Nearest:
        return virvo::nearest< VoxelT, bpc >(tex, coord, texsize);

    case virvo::Linear:
        return virvo::linear< VoxelT, bpc >(tex, coord, texsize);

    case virvo::BSpline:
        return 0.0;

    case virvo::BSplineInterpol:
        return 0.0;

    default:
        return virvo::nearest< VoxelT, bpc >(tex, coord, texsize);

    }

}


namespace simd
{


template < typename VoxelT, int bpc >
VV_FORCE_INLINE virvo::simd::Vec tex3D(VoxelT const* tex, virvo::simd::Vec3 coord, virvo::simd::Vec3i texsize,
    virvo::tex_filter_mode filter_mode = virvo::Nearest)
{

    switch (filter_mode)
    {

    case virvo::Nearest:
        return virvo::simd::nearest< VoxelT, bpc >(tex, coord, texsize);

    case virvo::Linear:
        return virvo::simd::linear< VoxelT, bpc >(tex, coord, texsize);

    case virvo::BSpline:
        return 0.0;

    case virvo::BSplineInterpol:
        return 0.0;

    default:
        return virvo::simd::nearest< VoxelT, bpc >(tex, coord, texsize);

    }

}



} // simd


} // virvo


#endif // VV_TEXTURE_TEX3D_H


