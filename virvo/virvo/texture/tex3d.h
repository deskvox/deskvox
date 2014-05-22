#ifndef VV_TEXTURE_TEX3D_H
#define VV_TEXTURE_TEX3D_H


#include "detail/sampler3d.h"
#include "detail/texture3d.h"



namespace virvo
{


template < int bpc, typename VoxelT >
VV_FORCE_INLINE float tex3D(texture< VoxelT, 3 > const& tex, virvo::Vec3 coord)
{

    virvo::ssize3 size( tex.width(), tex.height(), tex.depth() );
    return detail::tex3D< bpc, VoxelT >( tex.data, coord, size, tex.get_filter_mode() );

}


namespace simd
{


template < int bpc, typename VoxelT >
VV_FORCE_INLINE virvo::simd::Vec tex3D(texture< VoxelT, 3 > const& tex, virvo::simd::Vec3 coord)
{

    virvo::simd::Vec3i size( tex.width(), tex.height(), tex.depth() );
    return detail::simd::tex3D< bpc, VoxelT >( tex.data, coord, size, tex.get_filter_mode() );

}




} // simd


} // virvo


#endif // VV_TEXTURE_TEX3D_H


