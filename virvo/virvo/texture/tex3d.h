#ifndef VV_TEXTURE_TEX3D_H
#define VV_TEXTURE_TEX3D_H


#include "detail/sampler3d.h"
#include "detail/texture3d.h"



namespace virvo
{


template < int bpc, typename VoxelT >
VV_FORCE_INLINE float tex3D(VoxelT const* tex, virvo::Vec3 coord, virvo::ssize3 texsize,
    virvo::tex_filter_mode filter_mode = virvo::Nearest)
{

    switch (filter_mode)
    {

    case virvo::Nearest:
        return virvo::nearest< bpc >(tex, coord, texsize);

    case virvo::Linear:
        return virvo::linear< bpc >(tex, coord, texsize);

    case virvo::BSpline:
        return 0.0;

    case virvo::BSplineInterpol:
        return 0.0;

    default:
        return virvo::nearest< bpc >(tex, coord, texsize);

    }

}


template < int bpc, typename VoxelT >
VV_FORCE_INLINE float tex3D(texture< VoxelT, 3 > const& tex, virvo::Vec3 coord)
{

    virvo::ssize3 size( tex.width(), tex.height(), tex.depth() );
    return tex3D< bpc, VoxelT >( tex.data, coord, size, tex.get_filter_mode() );

}


namespace simd
{


template < int bpc, typename VoxelT >
VV_FORCE_INLINE virvo::simd::Vec tex3D(VoxelT const* tex, virvo::simd::Vec3 coord, virvo::simd::Vec3i texsize,
    virvo::tex_filter_mode filter_mode = virvo::Nearest)
{

    switch (filter_mode)
    {

    case virvo::Nearest:
        return virvo::simd::nearest< bpc >(tex, coord, texsize);

    case virvo::Linear:
        return virvo::simd::linear< bpc >(tex, coord, texsize);

    case virvo::BSpline:
        return 0.0;

    case virvo::BSplineInterpol:
        return 0.0;

    default:
        return virvo::simd::nearest< bpc >(tex, coord, texsize);

    }

}


template < int bpc, typename VoxelT >
VV_FORCE_INLINE virvo::simd::Vec tex3D(texture< VoxelT, 3 > const& tex, virvo::simd::Vec3 coord)
{

    virvo::simd::Vec3i size( tex.width(), tex.height(), tex.depth() );
    return tex3D< bpc, VoxelT >( tex.data, coord, size, tex.get_filter_mode() );

}




} // simd


} // virvo


#endif // VV_TEXTURE_TEX3D_H


